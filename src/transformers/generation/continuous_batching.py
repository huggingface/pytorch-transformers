# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import statistics
from typing import TYPE_CHECKING, Dict, List, Optional, Union, DefaultDict
import torch
from collections import defaultdict
if False:
    from flash_attn import flash_attn_varlen_func

from ..utils import (
    is_accelerate_available,
    logging,
)
from .configuration_utils import (
    GenerationConfig,
    PretrainedConfig,
)
from .logits_process import (
    LogitsProcessorList,
)
from .stopping_criteria import (
    StoppingCriteriaList,
)
from .utils import (
    is_torchdynamo_compiling,
    logging,
)


if TYPE_CHECKING:
    from .streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    pass


from .. import GenerationMixin


def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_seqlens_q=None,
    cumulative_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    block_table: Optional[torch.Tensor] = None,
    cache=None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Args:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full k
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full v
        cumulative_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cumulative_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        block_table [optional]: (num_blocks, max_num_blocks_per_seq), dtype torch.int32. This array should be used to index into
            the cache. Here, you already pass concatenated K and V. k[block_table] gives the cache the positions in cache that we need to fill?
            If we use with_kv_cache as it supports paged attention, it means it supports writing in a paged cache. But it does not support computing with
            ragged input. 
            Whiile flash_attn_varlen_func, supports ragged inputs, but it does not write into the kv_cache.
            Paged <==> fragmented cache, helpful for very long sequences.
            continuous <==> ragged inputs -> no padding
    """
    k, v = cache.update(k, v, module.layer_idx, cumulative_seqlens_q, cumulative_seqlens_k)

    attn_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cumulative_seqlens_q,
        cumulative_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        block_table=block_table,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        rotary_interleaved=True,
        **kwargs,
    )

    return attn_output


def compute_optimal_blocks(
    device: torch.device,
    generation_config: PretrainedConfig,
    inputs: List[List[int]],
    dtype: torch.dtype = torch.bfloat16,
    safety_margin: float = 0.9  # Safety margin for memory usage
):
    head_dim = generation_config.head_dim
    num_kv_heads = generation_config.num_key_value_heads

    # block size needs to be a multiple of 256
    # Get device memory properties
    if device.type == "cuda":
        device_properties = torch.cuda.get_device_properties(device)
        total_memory = device_properties.total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        available_memory = total_memory - max(allocated_memory, reserved_memory)
    else:
        return 32, 256
        raise ValueError("This function currently supports CUDA devices only.")

    tokens_to_generate = generation_config.max_new_tokens
    median_prefill_length = statistics.median(len(elem) for elem in inputs) # we don't need too many blocks

    # Apply safety margin
    available_memory *= safety_margin
    # Memory per tensor element
    dtype_size = torch.tensor([], dtype=dtype).element_size()

    # Estimate memory usage per block
    input_memory = median_prefill_length * head_dim * num_kv_heads * dtype_size
    output_memory = (median_prefill_length+tokens_to_generate) * head_dim * num_kv_heads * dtype_size
    per_block_memory = input_memory + output_memory

    # Compute the optimal number of blocks and block size
    max_blocks = available_memory // per_block_memory
    block_size = available_memory // (per_block_memory * head_dim * num_kv_heads * dtype_size)
    next_block_size = pow(2, math.ceil(math.log(block_size)/math.log(2))) # Round to next power of 2

    return int(max_blocks),  int(next_block_size)

class PagedAttentionCache:
    def __init__(
        self,
        config: PretrainedConfig,
        input_shapes:List[int],
        generation_config=None,
        device=None,
        dtype: torch.dtype = torch.float16,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        num_blocks = getattr(generation_config, "num_blocks", None)
        block_size = getattr(generation_config,"block_size", None)
        if num_blocks is None or block_size is None:
            # We determine the best size,
            num_blocks, block_size = compute_optimal_blocks(device, config, input_shapes)

        cache_shape = (  self.num_key_value_heads, num_blocks, block_size, self.head_dim)

        self.dtype = dtype

        self.key_cache: torch.Tensor = []
        self.value_cache: torch.Tensor = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            self.key_cache.append(torch.zeros(cache_shape, dtype=self.dtype, device=layer_device))
            self.value_cache.append(torch.zeros(cache_shape, dtype=self.dtype, device=layer_device))

        self.num_blocks = num_blocks
        self.block_size = block_size
        # a single sequence should take at most  config.max_num_blocks_per_seq
        self.block_table = defaultdict(list)
        self.free_blocks = list(range(num_blocks))
        # max_num_blocks_per_seq = 3, b = 2
        # ex:   0: [0, -1, -1]
        #       1: [1,2, -1]
        #

    def update(self, key, value, layer_idx, cumulative_seqlens_k, cache_index,fill_index, **kwargs):
        cache_view = (self.num_key_value_heads, self.num_blocks * self.block_size, self.head_dim)
        self.key_cache[layer_idx].view(cache_view)[:,fill_index, :] = key[0]
        self.value_cache[layer_idx].view(cache_view)[:,fill_index, :] = value[0]
        return self.key_cache[layer_idx].view(cache_view)[:,cache_index,:][None,...], self.value_cache[layer_idx].view(cache_view)[:,cache_index,:][None,...]

    def get_free_blocks(self):
        """"
        Returns the index of the memory blocks that are free to write on.
        Probably many ways to do this, for now I am just basing this of the block table
        which stores which blocks are used for each sentence.
        """
        return self.free_blocks

class ContinuousBatch:
    eos_token_id = None
    input_tokens = None
    finished_sequences = []
    max_seqlens_q = 0
    max_seqlens_k = 0
    generated_ids = defaultdict(list)
    prompts = defaultdict(list)

    def __init__(self, input_tokens, generation_config):
        self.eos_token_id = torch.tensor(generation_config.eos_token_id, dtype = torch.int32)
        self.input_tokens = sorted(input_tokens, key = lambda x: len(x)) # for now we sort by longest / smallest
        self.max_new_tokens = generation_config.max_new_tokens or 20
        self.cumulative_seqlens_q = None
        self.cumulative_seqlens_k = None
        self.cache_index = []
        self.next_ids = torch.tensor([], dtype=torch.long)
        self.batch_size = generation_config.batch_size

    def prepare_next_inputs(self, cache: PagedAttentionCache):
        # 1. we need to check for free blocs
        # 2. we update the ragged input, adding at the end the new sequences
        new_ids = []
        position_ids = []
        new_cache_index = []
        fill_index = []
        free_block_index = cache.get_free_blocks()
        max_seqlens_q = 0
        max_seqlens_k = 0
        next_full_cache_position = []
        for k in range(len(self.cache_index)):
            if (self.cache_index[k][-1] % cache.block_size + 1) // cache.block_size > 0:
                if len(free_block_index)==0:
                    print("We can't process seq, because there is not space left. Will wait for a seq to finish")
                    continue # TODO for now only
                # we need a new block for a current sequence
                new_block = free_block_index.pop()
                self.cache_index[k] += [new_block * cache.block_size]
            else:
                self.cache_index[k] += [self.cache_index[k][-1] +1]
            self.cumulative_seqlens_q[k] = k
            self.cumulative_seqlens_k[k] += k + 1# we added tokens before and after!
            position_ids += torch.tensor([[self.position_ids[k][-1]+1]])
            next_full_cache_position += self.cache_index[k]
            fill_index += [self.cache_index[k][-1]]

        i = len(self.next_ids)
        # how to efficiently select the next block? -> we probably just take the next longest sequence for now!
        while len(self.cache_index)  <= self.batch_size and len(free_block_index) > 0:
            next_sequence = self.input_tokens.pop(10)
            sample_length = len(next_sequence)
            if len(free_block_index) < (sample_length // cache.block_size) + 1:
                # we have to make sure there are enough free blocks
                self.input_tokens.insert(0, next_sequence)
                print("not enough memory to process this one, skippi")
                continue
            new_ids += next_sequence

            blocks_to_use = free_block_index[:(sample_length // cache.block_size) + 1]
            cache.block_table[i] = blocks_to_use
            free_block_index = free_block_index[(sample_length // cache.block_size) + 1:]
            if self.cumulative_seqlens_q is None or self.cumulative_seqlens_q .shape[0] ==0:
                self.cumulative_seqlens_q = torch.tensor([sample_length-1]).long()
                self.cumulative_seqlens_k = torch.tensor([sample_length-1]).long()
            else:
                self.cumulative_seqlens_q = torch.cat((self.cumulative_seqlens_q, torch.tensor([self.cumulative_seqlens_q[-1]+sample_length]))).long()
                self.cumulative_seqlens_k = torch.cat((self.cumulative_seqlens_k, torch.tensor([self.cumulative_seqlens_k[-1]+sample_length]))).long()

            position_ids += [list(range(sample_length))]
            if sample_length < cache.block_size:
                current_cache_index = list(range(blocks_to_use[0] * cache.block_size, (blocks_to_use[0] * cache.block_size) + sample_length))
            else:
                current_cache_index = []
                # If a sequence if big, it will be written on more than 1 block. Thus we need to write all the indexes
                # TIPS the blocks can be completely separeted apart
                for k in blocks_to_use[:-1]:
                    current_cache_index += list(range(k* cache.block_size, (k+1)* cache.block_size))
                current_cache_index += list(range(blocks_to_use[-1] * cache.block_size, (blocks_to_use[-1] * cache.block_size) + (sample_length % cache.block_size)))
            self.cache_index += [current_cache_index]
            assert len(current_cache_index) == sample_length
            fill_index += current_cache_index
            next_full_cache_position += current_cache_index
            max_seqlens_q = max(max_seqlens_q, sample_length)
            max_seqlens_k = max(max_seqlens_k, sample_length)

            self.prompts[i] += next_sequence
            i = i+1


        cache.free_blocks = free_block_index
        self.max_seqlens_q = max_seqlens_q
        self.max_seqlens_k = max_seqlens_k
        self.position_ids = position_ids



        position_ids = torch.cat([torch.tensor(k) for k in position_ids])[ None,:]
        new_ids = torch.cat((self.next_ids, torch.tensor(new_ids))).long().reshape(1, -1) # new sequence placed at the end

        return new_ids, position_ids, torch.tensor(next_full_cache_position), torch.tensor(fill_index)

    def update(self, generated_ids, cache:PagedAttentionCache):
        evict_mask = generated_ids == self.eos_token_id
        for i, k in enumerate(self.next_ids):
            self.generated_ids[i] += [k.detach().item()] # add the token to the full sequence
            evict_mask[i] |= len(self.generated_ids[i]) > self.max_new_tokens

        keep_mask = ~evict_mask
        self.next_ids = generated_ids[keep_mask].clone()

        self.cumulative_seqlens_q = self.cumulative_seqlens_q[keep_mask]
        self.cumulative_seqlens_k  = self.cumulative_seqlens_k[keep_mask]

        if evict_mask.sum(-1) > 0:
            evict_mask = torch.where(evict_mask)[0]        # delete the cache positions for these tokens
            evict_mask = evict_mask.tolist()
            self.cache_index = [self.cache_index[k] for k in range(len(self.cache_index)) if k not in evict_mask]
            for k in evict_mask:
                cache.free_blocks += cache.block_table[k]
                del cache.block_table[k]
                self.finished_sequences.append( self.prompts[k] + self.generated_ids[k])
                del self.generated_ids[k]
                del self.prompts[k]


    def __len__(self):
        return len(self.input_tokens)

class ContinuousMixin:

    @torch.no_grad()
    def fast_generate(
        self,
        input_tokens: list = None,
        logits_processor: Optional[LogitsProcessorList] = None, # TODO
        stopping_criteria: Optional[StoppingCriteriaList] = None, # TODO
        synced_gpus: Optional[bool] = None, # TODO
        streamer: Optional["BaseStreamer"] = None, # TODO
        **kwargs,
    ):
        r"""
            Fast generate that leverages continuous batching.

            1. You prepare the inputs
            2. You run forward pass
            3. You check for eos:
                a. Evict requests that are done
                b. Add new requests
            4. Pack finished sequences to return them
            5. Go to step 2.

            NO ATTENTION MASKS
            NO ASSISTANT MODEL

        """
        self.generation_config.batch_size = 8
        self.generation_config.max_num_blocks_per_seq = 16
        self.generation_config.eos_token_id = 1

        input_tokens = [k["input_ids"] for k in input_tokens]
        paged_attention_cache = PagedAttentionCache(self.config, input_tokens, self.generation_config, self.device) # init with throughput desiered, batch, etc
        current_batch = ContinuousBatch(input_tokens, self.generation_config)

        while len(current_batch) > 0:
            continous_batch, position_ids, cache_index, fill_index = current_batch.prepare_next_inputs(paged_attention_cache)
            kwargs = {
                "cumulative_seqlens_q":current_batch.cumulative_seqlens_q,
                "cumulative_seqlens_k":current_batch.cumulative_seqlens_k,
                "max_seqlens_q":current_batch.max_seqlens_q,
                "max_seqlens_k":current_batch.max_seqlens_k,
                "cache_index": cache_index,
                "fill_index": fill_index,
                "cache":paged_attention_cache,
            }
            out = self.model.forward(
                continous_batch, position_ids=position_ids,  **kwargs
            ).last_hidden_state
            logits = self.lm_head(out[:,current_batch.cumulative_seqlens_q, :])
            # we don't sample for now :)
            logits = torch.softmax(logits, dim=-1)
            generated_ids = torch.argmax(logits, dim=-1)
            current_batch.update(generated_ids[0], paged_attention_cache)
            # yield list(current_batch.generated_ids.values())
            if len(current_batch.finished_sequences) >  self.generation_config.batch_size:
                yield current_batch.finished_sequences
                current_batch.finished_sequences = []

        return current_batch.all_generated_sequences
