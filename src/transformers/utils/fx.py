# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import builtins
import collections
import contextlib
import functools
import inspect
import math
import operator
import os
import random
import sys
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch.fx import Graph, GraphModule, Node, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx._symbolic_trace import is_fx_tracing
from torch.fx.proxy import ParameterProxy

from .. import logging
from ..cache_utils import Cache, DynamicCache, SinkCache, StaticCache
from ..modeling_utils import PretrainedConfig, PreTrainedModel
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    TORCH_FX_REQUIRED_VERSION,
    get_torch_version,
    is_peft_available,
    is_torch_fx_available,
)


if is_peft_available():
    from peft import PeftModel


logger = logging.get_logger(__name__)
_IS_IN_DEBUG_MODE = os.environ.get("FX_DEBUG_MODE", "").upper() in ENV_VARS_TRUE_VALUES


def _generate_supported_model_class_names(
    model_name: Type[PretrainedConfig],
    supported_tasks: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    task_mapping = {
        "default": MODEL_MAPPING_NAMES,
        "pretraining": MODEL_FOR_PRETRAINING_MAPPING_NAMES,
        "next-sentence-prediction": MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
        "masked-lm": MODEL_FOR_MASKED_LM_MAPPING_NAMES,
        "causal-lm": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        "seq2seq-lm": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        "speech-seq2seq": MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
        "multiple-choice": MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
        "document-question-answering": MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
        "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
        "sequence-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
        "masked-image-modeling": MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
        "image-classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        "zero-shot-image-classification": MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        "ctc": MODEL_FOR_CTC_MAPPING_NAMES,
        "audio-classification": MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
        "semantic-segmentation": MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
        "backbone": MODEL_FOR_BACKBONE_MAPPING_NAMES,
        "image-feature-extraction": MODEL_FOR_IMAGE_MAPPING_NAMES,
    }

    if supported_tasks is None:
        supported_tasks = task_mapping.keys()
    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]

    model_class_names = []
    for task in supported_tasks:
        class_name = task_mapping[task].get(model_name, None)
        if class_name:
            model_class_names.append(class_name)

    return model_class_names


_REGULAR_SUPPORTED_MODEL_NAMES_AND_TASKS = [
    "altclip",
    "albert",
    "bart",
    "bert",
    "blenderbot",
    "blenderbot-small",
    "bloom",
    "clip",
    "convnext",
    "deberta",
    "deberta-v2",
    "dinov2",
    "distilbert",
    "donut-swin",
    "electra",
    "gpt2",
    "gpt_neo",
    "gptj",
    "hiera",
    "hubert",
    "ijepa",
    "layoutlm",
    "llama",
    "cohere",
    "lxmert",
    "m2m_100",
    "marian",
    "mbart",
    "megatron-bert",
    "mistral",
    "mixtral",
    "mobilebert",
    "mt5",
    "nezha",
    "opt",
    "pegasus",
    "plbart",
    "qwen2",
    "qwen2_moe",
    "resnet",
    "roberta",
    "segformer",
    "speech_to_text",
    "speech_to_text_2",
    "swin",
    "t5",
    "trocr",
    "vit",
    "xglm",
    "wav2vec2",
    #    "xlnet",
]

_FX_SUPPORTED_MODELS_WITH_KV_CACHE = ["llama", "opt"]

_REGULAR_SUPPORTED_MODELS = []
for item in _REGULAR_SUPPORTED_MODEL_NAMES_AND_TASKS:
    if isinstance(item, dict):
        _REGULAR_SUPPORTED_MODELS.extend(_generate_supported_model_class_names(**item))
    else:
        _REGULAR_SUPPORTED_MODELS.extend(_generate_supported_model_class_names(item))

_SPECIAL_SUPPORTED_MODELS = [
    "CLIPTextModel",
    "CLIPTextModelWithProjection",
    "CLIPVisionModel",
    "CLIPVisionModelWithProjection",
    "AltCLIPTextModel",
    "AltCLIPVisionModel",
    "GitVisionModel",
    "GPT2DoubleHeadsModel",
    "Speech2Text2Decoder",
    "TrOCRDecoder",
    "PeftModelForCausalLM",
    "PeftModelForSeq2SeqLM",
    # TODO: add support for them as it should be quite easy to do so (small blocking issues).
    # XLNetForQuestionAnswering,
]
_SUPPORTED_MODELS = tuple(sorted(set(_REGULAR_SUPPORTED_MODELS + _SPECIAL_SUPPORTED_MODELS)))

_CURRENT_TRACER = None


def torch_nn_embedding(self, input):
    return torch.empty(*input.shape, self.weight.shape[-1], device="meta", dtype=self.weight.dtype)


def torch_nn_functional_embedding(
    input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False
):
    return torch.empty(*input.shape, weight.shape[-1], device="meta", dtype=weight.dtype)


def torch_nn_layernorm(self, input):
    return input


def torch_nn_groupnorm(self, input):
    return input


def torch_nn_linear(self, input):
    return torch.empty(input.shape[:-1] + (self.out_features,), device="meta")


def torch_relu(x):
    return x


def torch_nn_relu(self, x):
    return x


def torch_nn_functional_relu(x, inplace=False):
    if not inplace:
        raise ValueError("Don't support in-place functional.relu for MetaTensor analysis")
    return x


def torch_where(condition, x, y):
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")


def torch_abs(input, *, out=None):
    if out is not None:
        raise ValueError("Don't support in-place abs for MetaTensor analysis")
    return input


def torch_arange(*args, **kwargs):
    n = len(args)
    step = 1
    if n == 1:
        start = 0
        end = args[0]
    elif n == 2:
        start, end = args
    else:
        start, end, step = args
    if isinstance(start, float):
        start = int(start)
    if isinstance(end, float):
        start = int(end)
    if isinstance(step, float):
        step = int(step)
    step = kwargs.get("step", step)
    dtype = kwargs.get("dtype")
    return torch.empty((end - start) // step, dtype=dtype, device="meta")


def torch_full(*args, **kwargs):
    args = list(args)
    # We set the fill value to 1 as its value is not important as long as it's not a tensor on the `meta` device.
    if len(args) > 1:
        args[1] = 1
    else:
        kwargs["fill_value"] = 1
    kwargs_without_device = dict(kwargs)
    kwargs_without_device.pop("device", None)
    return torch.full(*args, **kwargs_without_device, device="meta")


def torch_cat(tensors, dim=None, axis=None, *, out=None):
    if dim is None and axis is None:
        dim = 0
    if dim is None and axis is not None:
        dim = axis
    if dim < 0:
        dim = tensors[0].dim() + dim
    shapes = [t.shape for t in tensors]
    shape = list(shapes[0])
    concatenated_dim = sum(shape[dim] for shape in shapes)
    final_shape = shape[:dim] + [concatenated_dim] + shape[dim + 1 :]
    return torch.empty(final_shape, device="meta")


def torch_stack(tensors, dim=None, axis=None, *, out=None):
    if dim is None and axis is None:
        dim = 0
    if dim is None and axis is not None:
        dim = axis
    if dim < 0:
        dim = tensors[0].dim() + 1 + dim
    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    return torch.empty(shape, device="meta")


def torch_add(input, other, *, alpha=1, out=None):
    if not isinstance(input, torch.Tensor):
        return torch.empty_like(other, device="meta")
    if not isinstance(other, torch.Tensor):
        return torch.empty_like(input, device="meta")
    max_length = max(input.dim(), other.dim())
    input_shape = list(input.shape) + [1] * (max_length - input.dim())
    other_shape = list(other.shape) + [1] * (max_length - other.dim())
    shape = []
    for i in range(max_length):
        shape.append(max(input_shape[i], other_shape[i]))
    return torch.empty(shape, device="meta")


def torch_mul(input, other, *, out=None):
    return torch_add(input, other, out=out)


def torch_tensor_mul(self, other):
    return torch_mul(self, other)


def torch_matmul(input, other, *, out=None):
    d1 = input.dim()
    d2 = other.dim()
    shape = None
    if d1 == 1 and d2 == 1:
        shape = None
    elif d1 == 2 and d2 == 2:
        shape = (input.size(0), other.size(1))
    elif d1 == 1 and d2 == 2:
        shape = (other.size(1),)
    elif d1 == 2 and d1 == 1:
        shape = (input.size(0),)
    else:
        max_length = max(input.dim(), other.dim())
        shape1 = list(input.shape)
        shape2 = list(other.shape)
        if d1 == 1:
            shape1 = [1] + shape1
        if d2 == 1:
            shape2.append(1)
        shape1 = [-1] * (max_length - d1) + list(input.shape)
        shape2 = [-1] * (max_length - d2) + list(other.shape)
        shape = []
        for i in range(max_length):
            shape.append(max(shape1[i], shape2[i]))
        shape[-2] = shape1[-2]
        shape[-1] = shape2[-1]
        if d1 == 1:
            shape.pop(-2)
        if d2 == 1:
            shape.pop(-1)
    if shape is None:
        return torch.tensor(0.0, device="meta")
    return torch.empty(*shape, device="meta")


def torch_bmm(input, mat2, *, out=None):
    if out is not None:
        raise ValueError("Don't support in-place bmm for MetaTensor analysis")
    batch_size, n, m = input.shape
    _, _, p = mat2.shape
    return torch.empty(batch_size, n, p, device="meta")


def torch_baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    if out is not None:
        raise ValueError("Don't support in-place baddbmm for MetaTensor analysis")
    return torch_bmm(batch1, batch2)


def torch_tensor_baddbmm(self, batch1, batch2, *, beta=1, alpha=1, out=None):
    return torch_baddbmm(self, batch1, batch2, beta=beta, alpha=alpha, out=out)


def torch_einsum(equation, *operands):
    # TODO: infer shape without performing the computation, this might be quite hard.
    concrete_operands = (torch.empty_like(operand, device="cpu") for operand in operands)
    return torch.einsum(equation, *concrete_operands).to("meta")


def torch_tensor_repeat(self, *sizes):
    shape = list(self.shape)
    for i, x in enumerate(sizes):
        shape[i] *= x
    return torch.empty(shape, device="meta")


def torch_repeat_interleave(*args, dim=None, output_size=None):
    num_args = len(args)
    if num_args == 1:
        shape = [output_size if output_size is not None else args[0].sum()]
    else:
        shape = list(args[0].shape)
        if dim is None:
            if num_args > 2:
                dim = args[2]
            else:
                shape = [sum(shape)]
                dim = 0
        repeats = args[1]
        if isinstance(repeats, int) or torch.numel(repeats) == 1:
            shape[dim] *= int(repeats)
        else:
            shape[dim] = output_size if output_size is not None else repeats.sum()
    return torch.empty(*shape, device="meta")


def torch_index_select(input, dim, index, *, out=None):
    shape = list(input.shape)
    shape[dim] = len(index)
    return torch.empty(*shape, device="meta")


def torch_tensor_index_select(self, dim, index):
    return torch_index_select(self, dim, index)


def torch_gather(input, dim, index, *, sparse_grad=False, out=None):
    shape = list(input.shape)
    shape[dim] = index.shape[dim]
    return torch.empty(*shape, device="meta")


def torch_tensor_gather(self, dim, index):
    return torch_gather(self, dim, index)


def torch_roll(input, shifts, dims=None):
    return input


def torch_flip(input, dims):
    return input


def torch_tensor_flip(self, dims):
    return self


def torch_nn_conv1d(self, input):
    l_in = input.shape[-1]
    shape = None
    padding = self.padding
    if padding == "valid":
        padding = (0, 0)
    if padding == "same":
        shape = list(input.shape)
    if shape is None:
        shape = list(input.shape)
        l_out = math.floor(
            (l_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        shape[-1] = l_out
    shape[-2] = self.out_channels
    return torch.empty(shape, device="meta")


def torch_nn_conv2d(self, input):
    h_in, w_in = input.shape[-2:]
    shape = None
    padding = self.padding
    if padding == "valid":
        padding = (0, 0)
    if padding == "same":
        shape = list(input.shape)
    if shape is None:
        shape = list(input.shape)
        h_out = math.floor(
            (h_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        w_out = math.floor(
            (w_in + 2 * padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        )
        shape[-2:] = [h_out, w_out]
    shape[-3] = self.out_channels
    return torch.empty(shape, device="meta")


def torch_squeeze(input, dim=None):
    shape = list(input.shape)
    if dim is not None:
        if dim < 0:
            dim = input.dim() + dim
        if shape[dim] == 1:
            shape.pop(dim)
    else:
        new_shape = []
        for dim_value in shape:
            if dim_value == 1:
                continue
            new_shape.append(dim_value)
        shape = new_shape
    return torch.empty(shape, device="meta")


def torch_tensor_squeeze(self, dim=None):
    return torch_squeeze(self, dim)


def torch_unsqueeze(input, dim):
    shape = list(input.shape)
    if dim < 0:
        dim = input.dim() + 1 + dim
    shape.insert(dim, 1)
    return torch.empty(shape, device="meta")


def torch_tensor_unsqueeze(self, dim):
    return torch_unsqueeze(self, dim)


def torch_unique_consecutive(input, **kwargs):
    output = torch.unique_consecutive(torch.zeros_like(input, device="cpu"), **kwargs)
    if isinstance(output, torch.Tensor):
        return output.to("meta")
    else:
        return tuple(map(output, lambda x: x.to("meta")))


def torch_nn_functional_one_hot(tensor, num_classes=-1):
    if num_classes < 0:
        raise ValueError("Don't support automatic num_classes inference for MetaTensor analysis")
    shape = list(tensor.shape) + [num_classes]
    return torch.empty(shape, device="meta")


def torch_nn_functional_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    target_length = query.shape[-2]
    head_dim = value.shape[-1]
    return torch.empty((*query.shape[:-2], target_length, head_dim), device="meta")


def torch_nn_mseloss(self, input, target):
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    return torch.empty(shape, device="meta")


def torch_nn_crossentropyloss(self, input, target):
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    return torch.empty(shape, device="meta")


def torch_nn_bcewithlogitsloss(self, input, target):
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    return torch.empty(shape, device="meta")


def operator_getitem(a, b):
    def to_concrete(t):
        if isinstance(t, torch.Tensor):
            concrete = torch.ones_like(t, device="cpu")
            if concrete.dtype in [torch.float16, torch.float32, torch.float64, torch.int32]:
                concrete = concrete.to(torch.int64)
            return concrete
        return t

    if isinstance(a, torch.Tensor):
        # TODO: infer shape without performing the computation.
        if isinstance(b, tuple):
            b = tuple(map(to_concrete, b))
        else:
            b = to_concrete(b)
        return operator.getitem(torch.empty_like(a, device="cpu"), b).to("meta")
    return operator.getitem(a, b)


_MANUAL_META_OVERRIDES: Dict[Callable, Callable] = {
    torch.nn.Embedding: torch_nn_embedding,
    torch.nn.functional.embedding: torch_nn_functional_embedding,
    torch.nn.LayerNorm: torch_nn_layernorm,
    torch.nn.GroupNorm: torch_nn_groupnorm,
    torch.nn.Linear: torch_nn_linear,
    torch.relu: torch_relu,
    torch.nn.functional.relu: torch_nn_functional_relu,
    torch.nn.ReLU: torch_nn_relu,
    torch.where: torch_where,
    torch.abs: torch_abs,
    torch.arange: torch_arange,
    torch.full: torch_full,
    torch.cat: torch_cat,
    torch.stack: torch_stack,
    torch.add: torch_add,
    torch.mul: torch_mul,
    torch.Tensor.mul: torch_tensor_mul,
    torch.matmul: torch_matmul,
    torch.bmm: torch_bmm,
    torch.baddbmm: torch_baddbmm,
    torch.Tensor.baddbmm: torch_tensor_baddbmm,
    torch.einsum: torch_einsum,
    torch.Tensor.repeat: torch_tensor_repeat,
    torch.repeat_interleave: torch_repeat_interleave,
    torch.roll: torch_roll,
    torch.flip: torch_flip,
    torch.Tensor.flip: torch_tensor_flip,
    torch.index_select: torch_index_select,
    torch.Tensor.index_select: torch_tensor_index_select,
    torch.gather: torch_gather,
    torch.Tensor.gather: torch_tensor_gather,
    torch.nn.Conv1d: torch_nn_conv1d,
    torch.nn.Conv2d: torch_nn_conv2d,
    torch.squeeze: torch_squeeze,
    torch.Tensor.squeeze: torch_tensor_squeeze,
    torch.unsqueeze: torch_unsqueeze,
    torch.Tensor.unsqueeze: torch_tensor_unsqueeze,
    torch.unique_consecutive: torch_unique_consecutive,
    torch.nn.functional.one_hot: torch_nn_functional_one_hot,
    torch.nn.MSELoss: torch_nn_mseloss,
    torch.nn.CrossEntropyLoss: torch_nn_crossentropyloss,
    torch.nn.BCEWithLogitsLoss: torch_nn_bcewithlogitsloss,
    operator.getitem: operator_getitem,
}

_MANUAL_META_OVERRIDES[torch.nn.functional.scaled_dot_product_attention] = (
    torch_nn_functional_scaled_dot_product_attention
)


class HFProxy(Proxy):
    """
    Proxy that uses metadata to handle data-dependent control-flow.
    """

    def install_metadata(self, metadata):
        self._metadata = metadata

    @property
    def shape(self):
        return self.tracer.create_proxy("call_method", "size", (self,), {})

    @property
    def device(self):
        # Hack so we can track when devices are used. During meta-tensor propagation,
        # replace these values with a constant 'meta'
        return MetaDeviceAttribute(self, "device")

    def __len__(self):
        if hasattr(self, "_metadata") and self._metadata is not None:
            return len(self._metadata)
        return super().__len__()

    def __bool__(self):
        if hasattr(self, "_metadata") and self._metadata is not None:
            return self._metadata
        return super().__bool__()

    def __getattr__(self, k):
        if k == "_metadata":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return HFAttribute(self, k)

    def __setitem__(self, indices, values):
        return self.tracer.create_proxy("call_function", operator.setitem, (self, indices, values), {})

    def __contains__(self, key):
        if hasattr(self, "_metadata") and self._metadata is not None:
            return key in self._metadata
        return super().__contains__(key)


class HFAttribute(HFProxy):
    def __init__(self, root, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

        if hasattr(self.root, "_metadata"):
            self.install_metadata(getattr(self.root._metadata, attr))

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy("call_function", builtins.getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)


class MetaDeviceAttribute(HFAttribute):
    pass


class HFCacheProxy(HFProxy):
    """
    Proxy that represents an instance of `transformers.cache_utils.Cache`.
    """

    def install_orig_cache_cls(self, orig_cache_cls: Type[Cache]):
        self._orig_cache_cls = orig_cache_cls

    @property
    def __class__(self):
        if not hasattr(self, "_orig_cache_cls"):
            raise RuntimeError("The original Cache class must be installed to the HFCacheProxy.")
        return self.tracer._CLASSES_TO_PATCH[self._orig_cache_cls]


def create_wrapper(
    function: Callable,
    op_type: Union[Literal["call_function"], Literal["call_method"], Literal["get_attr"]],
    proxy_factory_fn: Optional[Callable[[Node], Proxy]] = None,
) -> Callable:
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if not is_fx_tracing():
            return function(*args, **kwargs)

        found_proxies = []

        def check_proxy(a):
            if isinstance(a, Proxy):
                found_proxies.append(a)

        torch.fx.node.map_aggregate(args, check_proxy)
        torch.fx.node.map_aggregate(kwargs, check_proxy)

        if len(found_proxies) > 0:
            tracer = found_proxies[0].tracer
            if op_type == "call_function":
                target = function
            elif op_type == "call_method":
                target = function.__name__
            elif op_type == "get_attr":
                target = function.__name__
            else:
                raise ValueError(f"op_type {op_type} not supported.")
            return tracer.create_proxy(op_type, target, args, kwargs, proxy_factory_fn=proxy_factory_fn)
        else:
            return function(*args, **kwargs)

    return wrapper


class HFProxyableClassMeta(type):
    """
    Metaclass that creates a class with its main methods wrapped to be proxyable.
    """

    def __new__(
        cls,
        name: str,
        bases: Tuple[Type, ...],
        attrs: Dict[str, Any],
        proxy_factory_fn: Optional[Callable[[Node], Proxy]] = None,
    ):
        cls = super().__new__(cls, name, bases, attrs)
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name, None)
            if attr is None:
                continue
            if attr_name == "__init__":
                op_type = "call_function"
            elif attr_name.startswith("__"):
                op_type = None
            elif inspect.ismethod(attr):
                op_type = "call_function"
            elif inspect.isfunction(attr):
                op_type = "call_method"
            else:
                op_type = None
            if op_type is not None:
                setattr(cls, attr_name, create_wrapper(attr, op_type, proxy_factory_fn=proxy_factory_fn))
        return cls


def gen_constructor_wrapper(target: Callable) -> Tuple[Callable, Callable]:
    """
    Wraps `target` to be proxyable. Used for tensor creators like `torch.ones`, `torch.arange` and so on.
    """
    wrapper = create_wrapper(target, "call_function")
    return wrapper, target


def _proxies_to_metas(v):
    """Returns the underlying metadata for HFProxies, and behaves like the identity for the others."""
    if isinstance(v, MetaDeviceAttribute):
        return "meta"
    if isinstance(v, torch.fx.Proxy):
        if not (isinstance(v, HFProxy) and hasattr(v, "_metadata")):
            raise RuntimeError(f"No metadata was found for {v}")
        return v._metadata
    return v


def create_cache_proxy_factory_fn(orig_cache_cls: Type[Cache]) -> Callable[[Node], HFCacheProxy]:
    def cache_proxy_factory_fn(n: Node) -> HFCacheProxy:
        global _CURRENT_TRACER
        if not isinstance(_CURRENT_TRACER, HFTracer):
            raise RuntimeError("Cannot create HFCacheProxy because there is no HFTracer currently tracing.")
        cache_proxy = HFCacheProxy(n, _CURRENT_TRACER)
        cache_proxy.install_orig_cache_cls(orig_cache_cls)
        return cache_proxy

    return cache_proxy_factory_fn


# # Proxyable equivalent of the cache classes defined in `transformers.cache_utils`.
# ProxyableCache = HFProxyableClassMeta(
#     "ProxyableCache", (Cache,), {}, proxy_factory_fn=create_cache_proxy_factory_fn(Cache)
# )
# ProxyableDynamicCache = HFProxyableClassMeta(
#     "ProxyableDynamicCache",
#     (DynamicCache,),
#     {},
#     proxy_factory_fn=create_cache_proxy_factory_fn(DynamicCache),
# )
# ProxyableSinkCache = HFProxyableClassMeta(
#     "ProxyableSinkCache",
#     (SinkCache,),
#     {},
#     proxy_factory_fn=create_cache_proxy_factory_fn(SinkCache),
# )
# ProxyableStaticCache = HFProxyableClassMeta(
#     "ProxyableStaticCache",
#     (StaticCache,),
#     {},
#     proxy_factory_fn=create_cache_proxy_factory_fn(StaticCache),
# )


def _generate_random_int(low: int = 10, high: int = 20, forbidden_values: Optional[List[int]] = None):
    if forbidden_values is None:
        forbidden_values = []
    value = random.randint(low, high)
    while value in forbidden_values:
        value = random.randint(low, high)
    return value


class HFTracer(Tracer):
    """
    Tracer that is able to symbolically trace models from the library. To do that, it uses the HFProxy instead of the
    regular PyTorch torch.fx.Proxy.
    """

    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes: bool = True
    allow_insert_stateless_mods: bool = True
    _TORCH_METHODS_TO_PATCH = [
        "arange",
        "zeros",
        "ones",
        "full",
        "full_like",
        "eye",
        "empty",
        "tensor",
        "clamp",
        "finfo",
        "tril",
    ]
    _CLASSES_TO_PATCH = {
        # Cache: ProxyableCache,
        # DynamicCache: ProxyableDynamicCache,
        # SinkCache: ProxyableSinkCache,
        # StaticCache: ProxyableStaticCache,
    }

    supported_archs = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)

    def __init__(self, autowrap_modules=(math,), autowrap_functions=()):
        super().__init__(autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions)

        if not is_torch_fx_available():
            raise ImportError(
                f"Found an incompatible version of torch. Found version {get_torch_version()}, but only version "
                f"{TORCH_FX_REQUIRED_VERSION} is supported."
            )

    def _generate_dummy_input(
        self, model: "PreTrainedModel", input_name: str, shape: List[int], input_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Generates dummy input for model inference recording."""
        # Retrieving the model class, either from the "class_for_deserialization" attribute if the model was restored
        # from pickle, or from the "__class__" attribute in the general case.
        model_class_name = getattr(model, "class_for_deserialization", model.__class__).__name__
        device = model.device
        inputs_dict = {}

        # when tracing a model with KV cache, we simply need to unsure that the KV cache length is larger than one to
        # rightfully pass certain controlflows (Example: https://github.com/huggingface/transformers/blob/5c8d941d66734811d2ef6f57f15b44f7fb7a98c4/src/transformers/modeling_attn_mask_utils.py#L162).
        # After tracing, the model can then still be used with arbitrary lengths different than the one used during tracing.
        kv_cache_length = 5

        if input_name in ["labels", "start_positions", "end_positions"]:
            batch_size = shape[0]
            if model_class_name in [
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES),
                *get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
                *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES),
            ]:
                inputs_dict["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class_name in [
                *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
                *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
                "XLNetForQuestionAnswering",
            ]:
                inputs_dict["start_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
                inputs_dict["end_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class_name in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
                if not hasattr(model.config, "problem_type") or model.config.problem_type is None:
                    raise ValueError(
                        "Could not retrieve the problem type for the sequence classification task, please set "
                        'model.config.problem_type to one of the following values: "regression", '
                        '"single_label_classification", or "multi_label_classification".'
                    )

                if model.config.problem_type == "regression":
                    labels_shape = (batch_size, model.config.num_labels)
                    labels_dtype = torch.float32
                elif model.config.problem_type == "single_label_classification":
                    labels_shape = (batch_size,)
                    labels_dtype = torch.long
                elif model.config.problem_type == "multi_label_classification":
                    labels_shape = (batch_size, model.config.num_labels)
                    labels_dtype = torch.float32
                else:
                    raise ValueError(
                        'Expected model.config.problem_type to be either: "regression", "single_label_classification"'
                        f', or "multi_label_classification", but "{model.config.problem_type}" was provided.'
                    )
                inputs_dict["labels"] = torch.zeros(*labels_shape, dtype=labels_dtype, device=device)

            elif model_class_name in [
                *get_values(MODEL_FOR_PRETRAINING_MAPPING_NAMES),
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES),
                "GPT2DoubleHeadsModel",
                "PeftModelForCausalLM",
                "PeftModelForSeq2SeqLM",
            ]:
                inputs_dict["labels"] = torch.zeros(shape, dtype=torch.long, device=device)
            elif model_class_name in [*get_values(MODEL_FOR_CTC_MAPPING_NAMES)]:
                inputs_dict["labels"] = torch.zeros(shape, dtype=torch.float32, device=device)
            else:
                raise NotImplementedError(
                    f"Generating the dummy input named {input_name} for {model_class_name} is not supported yet."
                )
        elif "pixel_values" in input_name:
            batch_size = shape[0]
            image_size = getattr(model.config, "image_size", None)
            if image_size is None:
                if hasattr(model.config, "vision_config"):
                    image_size = model.config.vision_config.image_size
                elif hasattr(model.config, "encoder"):
                    image_size = model.config.encoder.image_size
                else:
                    image_size = (_generate_random_int(), _generate_random_int())

            # If no num_channels is in the config, use some arbitrary value.
            num_channels = getattr(model.config, "num_channels", 3)
            if not isinstance(image_size, collections.abc.Iterable):
                image_size = (image_size, image_size)
            height, width = image_size
            inputs_dict[input_name] = torch.zeros(
                batch_size, num_channels, height, width, dtype=torch.float32, device=device
            )
        elif "bbox" in input_name:
            inputs_dict[input_name] = torch.zeros(*shape, 4, dtype=torch.float, device=device)
        elif "input_features" in input_name:
            inputs_dict[input_name] = torch.zeros(
                *shape, model.config.input_feat_per_channel, dtype=torch.float, device=device
            )
        elif "inputs_embeds" in input_name:
            batch_size = shape[0]

            if (
                getattr(model.config, "embedding_size", None) is not None
                and model.config.model_type != "megatron-bert"
            ):
                embedding_size = model.config.embedding_size
            else:
                embedding_size = model.config.hidden_size

            if len(shape) == 3:
                # (batch_size, num_choices, sequence_length, embedding_size)
                embedding_shape = (batch_size, shape[1], shape[2], embedding_size)
            else:
                # (batch_size, sequence_length, embedding_size)
                embedding_shape = (batch_size, shape[1], embedding_size)

            inputs_dict[input_name] = torch.zeros(embedding_shape, dtype=torch.float, device=device)
        elif "visual_feats" in input_name:
            inputs_dict[input_name] = torch.zeros(
                shape
                + [
                    model.config.visual_feat_dim,
                ],
                dtype=torch.float,
                device=device,
            )
        elif "visual_pos" in input_name:
            inputs_dict[input_name] = torch.zeros(
                shape
                + [
                    model.config.visual_pos_dim,
                ],
                dtype=torch.float,
                device=device,
            )
        elif "inputs" in input_name:
            inputs_dict[input_name] = torch.zeros(*shape, dtype=torch.float, device=device)
        elif "input_values" in input_name:
            batch_size, _ = shape
            # Generating big sequence length for audio inputs.
            seq_length = _generate_random_int(low=10000, high=20000)
            inputs_dict[input_name] = torch.zeros(batch_size, seq_length, dtype=torch.float, device=device)
        elif "mask" in input_name:
            if "past_key_values" in input_names:
                mask_shape = [shape[0], shape[1] + kv_cache_length]
            else:
                mask_shape = shape

            inputs_dict[input_name] = torch.zeros(mask_shape, dtype=torch.long, device=device)
        elif "ids" in input_name:
            inputs_dict[input_name] = torch.zeros(shape, dtype=torch.long, device=device)
        elif "past_key_values" in input_name:
            if model.config.model_type not in _FX_SUPPORTED_MODELS_WITH_KV_CACHE:
                raise NotImplementedError(
                    f"Symbolic trace with past_key_values input is not supported yet for the model {model.config.model_type}. Please open an issue or a PR in Transformers repository if you would like to see the support added."
                )
            num_heads = model.config.num_attention_heads
            head_dim = model.config.hidden_size // model.config.num_attention_heads

            cache_shape = (shape[0], num_heads, kv_cache_length, head_dim)
            pkv = tuple(
                (
                    torch.rand(cache_shape, dtype=torch.float, device=device),
                    torch.rand(cache_shape, dtype=torch.float, device=device),
                )
                for i in range(model.config.num_hidden_layers)
            )
            inputs_dict[input_name] = pkv
        else:
            shape_with_hidden_size = shape + [model.config.hidden_size]
            inputs_dict[input_name] = torch.zeros(shape_with_hidden_size, dtype=torch.float, device=device)

        return inputs_dict

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

        if kind == "placeholder" and target in self.meta_args:
            rv.install_metadata(self.meta_args[target])
            return rv

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas = torch.fx.node.map_aggregate(args, _proxies_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, _proxies_to_metas)

            should_install_metadata = True

            self._disable_module_getattr = True
            self._disable_call_module = True

            if kind == "call_function":
                meta_target = _MANUAL_META_OVERRIDES.get(target, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
                if isinstance(meta_out, torch.Tensor):
                    meta_out = meta_out.to(device="meta")
            elif kind == "call_method":
                method = getattr(args_metas[0].__class__, target)
                meta_target = _MANUAL_META_OVERRIDES.get(method, method)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_module":
                if not hasattr(self, "orig_forward"):
                    raise AttributeError(f"{self} does not have an attribute called orig_forward")
                mod = self.root.get_submodule(target)
                mod_type = type(mod)
                if mod_type in _MANUAL_META_OVERRIDES:
                    meta_out = _MANUAL_META_OVERRIDES[mod_type](mod, *args_metas, **kwargs_metas)
                else:
                    meta_out = self.orig_forward(*args_metas, **kwargs_metas)
            elif kind == "get_attr":
                attr_itr = self.root
                atoms = target.split(".")
                for atom in atoms:
                    attr_itr = getattr(attr_itr, atom)
                if isinstance(attr_itr, torch.Tensor):
                    meta_out = attr_itr.to(device="meta")
                else:
                    meta_out = attr_itr
            else:
                should_install_metadata = False

            if should_install_metadata:
                if not isinstance(rv, Proxy):
                    raise ValueError("Don't support composite output yet")
                rv.install_metadata(meta_out)

        except Exception as e:
            if _IS_IN_DEBUG_MODE:
                warnings.warn(f"Could not compute metadata for {kind} target {target}: {e}")

        self._disable_module_getattr = False
        self._disable_call_module = False

        return rv

    # Replaced by .getattr from PyTorch 1.13
    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:

            def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
                for n, p in collection_to_search:
                    if attr_val is p:
                        if n not in parameter_proxy_cache:
                            kwargs = {}
                            if "proxy_factory_fn" in inspect.signature(self.create_proxy).parameters:
                                kwargs["proxy_factory_fn"] = (
                                    None
                                    if not self.param_shapes_constant
                                    else lambda node: ParameterProxy(self, node, n, attr_val)
                                )
                            val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                            parameter_proxy_cache[n] = val_proxy
                        return parameter_proxy_cache[n]
                return None

            if isinstance(attr_val, torch.nn.Parameter):
                maybe_parameter_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_parameters(), parameter_proxy_cache
                )
                if maybe_parameter_proxy is not None:
                    return maybe_parameter_proxy

            if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
                maybe_buffer_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_buffers(), parameter_proxy_cache
                )
                if maybe_buffer_proxy is not None:
                    return maybe_buffer_proxy

            return attr_val

    # Needed for PyTorch 1.13+
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        return self._module_getattr(attr, attr_val, parameter_proxy_cache)

    def call_module(self, m, forward, args, kwargs):
        if getattr(self, "_disable_call_module", False):
            return forward(*args, **kwargs)
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    def proxy(self, node):
        return HFProxy(node, self)

    @contextlib.contextmanager
    def patch_for_tracing(self, root: Union[torch.nn.Module, Callable[..., Any]]):
        # Patching torch functions
        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        # Patching classes
        patched = []
        module_of_model = inspect.getmodule(root)
        for name, mod in sys.modules.items():
            if module_of_model is not None and mod is not module_of_model:
                continue
            if not name.startswith("transformers"):
                continue
            for orig_cls, patched_cls in self._CLASSES_TO_PATCH.items():
                for attr_name, attr in mod.__dict__.items():
                    if attr is orig_cls:
                        patched.append((mod, attr_name, orig_cls))
                        setattr(mod, attr_name, patched_cls)

        yield

        # Restoring patched functions and classes.
        for name, (_, orig) in self.patched_torch_methods.items():
            setattr(torch, name, orig)
        self.patched_torch_methods = {}
        self.orig_fns = set()

        for mod, attr_name, orig_cls in patched:
            setattr(mod, attr_name, orig_cls)

    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
        dummy_inputs: Optional[Dict[str, Any]] = None,
        complete_concrete_args_with_inputs_not_in_dummy_inputs: bool = True,
    ) -> Graph:
        """
        Traces `root` and returns the corresponding FX `torch.fx.Graph` representation. `root` can either be a
        `torch.nn.Module` instance or a Python callable. Note that after this call, `self.root` may be different from
        the `root` passed in here. For example, when a free function is passed to `trace()`, we will create a
        `torch.nn.Module` instance to use as the root and add embedded constants to.

        Args:
            root (`torch.nn.Module` or  `Callable`):
                Either a `torch.nn.Module`` or a function to be traced through. If root is not a
                [`~transformers.PreTrainedModel`], then `dummy_inputs` must be passed, otherwise tracing will fail.
            concrete_args (`Dict[str, Any], *optional*):
                Concrete arguments that should not be treated as Proxies
            dummy_inputs (`Dict[str, Any]`, *optional*):
                The dummy inputs needed to handle data-dependent control-flow if `root` is not a
                [`~transformers.PreTrainedModel`]. It can also be used when `root` is a
                [`~transformers.PreTrainedModel`] to specify custom dummy inputs for a subset or all the model inputs.
            complete_concrete_args_with_inputs_not_in_dummy_inputs (`bool`, *optional*, defaults to `True`):
                If `True`, and `dummy_inputs` is specified, every argument that `root` can take that is not in
                `dummy_inputs` and not in `concrete_args` will be added to `concrete_args`, otherwise does nothing.

        Returns:
            `torch.fx.Graph`:
                A FX `torch.fx.Graph` representing the semantics of the passed-in `root`.

        """
        sig = inspect.signature(root.forward if isinstance(root, torch.nn.Module) else root)

        if concrete_args is None:
            concrete_args = {}

        if dummy_inputs is not None and complete_concrete_args_with_inputs_not_in_dummy_inputs:
            for param in sig.parameters.values():
                if param.name in dummy_inputs:
                    continue
                if param.default is inspect.Parameter.empty:
                    raise ValueError(f"You need to specify a default value for the parameter {param.name}.")
            concrete_args.update(
                {
                    p.name: p.default
                    for p in sig.parameters.values()
                    if (p.name not in dummy_inputs and p.name not in concrete_args)
                }
            )

        input_names = sig.parameters.keys() - concrete_args.keys()

        # Creating a random input shape to generate dummy inputs.
        batch_size = _generate_random_int()
        sequence_length = _generate_random_int()
        shape = [batch_size, sequence_length]

        if root.__class__.__name__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES):
            num_choices = _generate_random_int(low=2, high=5)
            shape.insert(1, num_choices)

        inputs = dict(dummy_inputs) if dummy_inputs is not None else {}
        for input_name in input_names:
            if input_name in inputs:
                continue
            # We enforce that root must either be a PreTrainedModel or deserialized from a serialized traced model to
            # be able to use HFTracer._generate_dummy_input.
            if isinstance(root, self.supported_archs) or type(root).__qualname__.startswith(
                ("_deserialize_graph_module", "_CodeOnlyModule")
            ):
                inputs.update(self._generate_dummy_input(root, input_name, shape, input_names=input_names))
            else:
                raise RuntimeError(
                    f"Could not generate input named {input_name} for because root is not a"
                    " transformers.PreTrainedModel."
                )

        def to_meta(value):
            if isinstance(value, torch.Tensor):
                return value.to("meta")
            return value

        concrete_metas = pytree.tree_map(to_meta, inputs)

        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD and param.name not in input_names:
                concrete_metas[f"**{param.name}"] = {}
        self.meta_args = concrete_metas

        global _CURRENT_TRACER
        _CURRENT_TRACER = self
        with self.patch_for_tracing(root):
            try:
                self.graph = super().trace(root, concrete_args=concrete_args)
            finally:
                _CURRENT_TRACER = None

        # This is necessary because concrete args are added as input to the traced module since
        # https://github.com/pytorch/pytorch/pull/55888.
        for node in self.graph.nodes:
            if node.op == "placeholder":
                # Removing default values for inputs as the forward pass will fail with them.
                if node.target in input_names:
                    node.args = ()
                    # Without this, torch.jit.script fails because the inputs type is Optional[torch.Tensor].
                    # It cannot infer on the attributes and methods the input should have, and fails.
                    node.type = torch.Tensor
                # It is a concrete arg so it is not used and should be removed.
                else:
                    to_visit = [node]
                    to_delete = collections.OrderedDict()
                    while to_visit:
                        n = to_visit.pop(0)
                        to_delete[n] = None
                        to_visit += list(n.users.keys())

                    for user in reversed(to_delete.keys()):
                        self.graph.erase_node(user)

            # TODO: solves GraphModule creation.
            # Without this, return type annotation "Tuple" is causing code execution failure.
            if node.op == "output":
                node.type = None

        return self.graph

    def _stateless_mod_instanciation_depends_on_proxies(self, mod: nn.Module) -> bool:
        """
        Whether the module was instantiated with Proxies. If that is the case, such module cannot be a leaf module
        because its attributes are input-dependent.
        """
        return any(isinstance(attr, Proxy) for attr in mod.__dict__.values())

    def _insert_module_as_submodule(self, mod: nn.Module) -> str:
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        # If one of the module attributes is a Proxy, it means that its instantiation is input-dependent.
        # It is not possible to insert such modules, those should be traced through.
        if self._stateless_mod_instanciation_depends_on_proxies(mod):
            return ""
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        already_inserted = False
        while hasattr(self.root, path):
            if getattr(self.root, path) is mod:
                already_inserted = True
                break
            path = f"{mod_name}_{idx}"
            idx += 1

        # No need to add multiple instances of the same module.
        if not already_inserted:
            self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: nn.Module) -> str:
        """
        Helper method to find the qualified name of `mod` in the Module hierarchy of `root`. For example, if `root` has
        a submodule named `foo`, which has a submodule named `bar`, passing `bar` into this function will return the
        string "foo.bar".

        Args:
            mod (str): The `Module` to retrieve the qualified name for.
        """
        try:
            return super().path_of_module(mod)
        except NameError as e:
            if self.allow_insert_stateless_mods and len(list(mod.parameters())) == 0 and len(list(mod.buffers())) == 0:
                path = self._insert_module_as_submodule(mod)
                return path
            raise e

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (not self._stateless_mod_instanciation_depends_on_proxies(m)) and super().is_leaf_module(
            m, module_qualified_name
        )

    @compatibility(is_backward_compatible=True)
    def keys(self, obj: "Proxy") -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an iterator if ** is supposed to work in
        your custom tracer.
        """
        attribute = HFAttribute(obj, "keys")()
        if obj.node.target.startswith("**"):
            return attribute._metadata
        return attribute


def get_concrete_args(model: nn.Module, input_names: List[str]):
    sig = inspect.signature(model.forward)

    if not (set(input_names) <= set(sig.parameters.keys())):
        formatted_input_names = input_names[0] if len(input_names) == 1 else ", ".join(input_names)
        formatted_allowed_input_names = ", ".join(sig.parameters.keys())
        raise ValueError(
            f"The model does not have input(s) named: {formatted_input_names}, expected a subset of the following:"
            f" {formatted_allowed_input_names}"
        )

    return {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}


def is_model_supported(model: "PreTrainedModel"):
    return model.__class__.__name__ in _SUPPORTED_MODELS


def check_if_model_is_supported(model: "PreTrainedModel"):
    if not is_model_supported(model):
        supported_model_names = ", ".join(_SUPPORTED_MODELS)
        raise NotImplementedError(
            f"Model {model.__class__.__name__} is not supported yet, supported models: {supported_model_names}"
        )


def symbolic_trace(
    model: "PreTrainedModel",
    input_names: Optional[List[str]] = None,
    disable_check: bool = False,
    tracer_cls: Type[HFTracer] = HFTracer,
) -> GraphModule:
    """
    Performs symbolic tracing on the model.

    Args:
        model ([`PretrainedModel`]):
            The model to trace.
        input_names (`List[str]`, *optional*):
            The names of the inputs of the traced model. If unset, model.dummy_inputs.keys() are used instead.
        disable_check (`bool`, *optional*, defaults to `False`):
            If `True`, no check is done before trying to trace the model, this is mostly usesul for debugging purposes.
        tracer_cls (`Type[HFTracer]`, *optional*, defaults to `HFTracer`):
            The tracer class to use for instantiating the tracer. If unset, `HFTracer` is used instead.

    Returns:
        `torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example:

        ```python
        from transformers.utils.fx import symbolic_trace

        traced_model = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        ```
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    input_names = list(input_names)
    concrete_args = get_concrete_args(model, input_names)

    if not disable_check:
        check_if_model_is_supported(model)

    if "past_key_values" in input_names and not getattr(model.config, "use_cache", False):
        logger.warning(
            "`past_key_values` were specified as input names, but model.config.use_cache = False, this might lead to "
            "unexpected behavior."
        )
    if "past_key_values" not in input_names and getattr(model.config, "use_cache", False):
        logger.warning(
            "`past_key_values` were not specified as input names, but model.config.use_cache = True. Setting "
            "model.config.use_cache = False."
        )
        model.config.use_cache = False

    # Tracing.
    tracer = tracer_cls()
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    traced.config = model.config
    # The model class must be stored as an attribute to allow model deserialization, which uses trace, and thus
    # _generate_dummy_input, where the model class is needed.
    traced.class_for_deserialization = model.__class__
    traced.device = model.device

    return traced
