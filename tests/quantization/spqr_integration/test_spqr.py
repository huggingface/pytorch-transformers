# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import gc
import tempfile
import unittest

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, SpQRConfig, StaticCache
from transformers.testing_utils import (
    require_accelerate,
    require_spqr,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_accelerate_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import init_empty_weights


@require_torch_gpu
class SpQRConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = SpQRConfig()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {
            "beta1": 16,
            "beta2": 16,
            "bits": 3,
            "modules_to_not_convert": ["lm_head.weight"],
            "shapes": {"model.layers.0.self_attn.q_proj.dense_weights.shape": 16},
        }
        quantization_config = SpQRConfig.from_dict(dict)

        self.assertEqual(dict["beta1"], quantization_config.beta1)
        self.assertEqual(dict["beta2"], quantization_config.beta2)
        self.assertEqual(dict["bits"], quantization_config.bits)
        self.assertEqual(dict["modules_to_not_convert"], quantization_config.modules_to_not_convert)
        self.assertEqual(dict["shapes"], quantization_config.shapes)


@slow
@require_torch_gpu
@require_spqr
@require_accelerate
class SpQRTest(unittest.TestCase):
    model_name = "elvircrn/Llama-2-7b-SPQR-3Bit-16x16-red_pajama-hf"

    input_text = "Hello my name is"
    max_new_tokens = 32

    EXPECTED_OUTPUT = (
        "Hello my name is Jesse. (I'm also known as Jesse) I'm a 25 year old male from United States. I'm looking for"
    )
    EXPECTED_OUTPUT_COMPILE = "Hello my name is Jake and I am a 20 year old student at the University of North Texas. (Go Mean Green!) I am a huge fan of the Dallas"

    device_map = "cuda"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            device_map=cls.device_map,
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """
        from spqr_quant import QuantizedLinear

        from transformers.integrations import replace_with_spqr_linear

        model_id = "meta-llama/Llama-2-7b-hf"
        config = AutoConfig.from_pretrained(model_id)
        quantization_config = AutoConfig.from_pretrained(self.model_name, return_dict=False).quantization_config
        quantization_config = SpQRConfig.from_dict(quantization_config)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, config=config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1

        model, _ = replace_with_spqr_linear(
            model,
            quantization_config=quantization_config,
            modules_to_not_convert=quantization_config.modules_to_not_convert,
        )

        nb_spqr_linear = 0
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                nb_spqr_linear += 1

        self.assertEqual(nb_linears - 1, nb_spqr_linear)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_raise_if_non_quantized(self):
        model_id = "meta-llama/Llama-2-7b-hf"
        quantization_config = SpQRConfig()

        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    @unittest.skip
    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_quantized_model_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        quantized_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")

        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)

        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_quantized_model_compile(self):
        """
        Simple test that checks if the quantized model is working properly
        """

        # Sample tokens greedily
        def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
            logits = model(
                cur_token,
                position_ids=input_pos,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True,
            )[0]
            new_token = torch.argmax(logits[:, [-1]], dim=-1).to(torch.int)

            return new_token

        # Tokenize the test input
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)["input_ids"]
        seq_length = input_ids.shape[1]

        # Setup static KV cache for generation
        past_key_values = StaticCache(
            config=self.quantized_model.config,
            batch_size=1,
            max_cache_len=seq_length + self.max_new_tokens + 1,
            device=torch_device,
            dtype=self.quantized_model.config._pre_quantization_dtype,
        )

        # Allocate token ids to be generated and copy prefix ids
        cache_position = torch.arange(seq_length, device=torch_device)
        generated_ids = torch.zeros(1, seq_length + self.max_new_tokens, dtype=torch.int, device=torch_device)
        generated_ids[:, cache_position] = input_ids.to(torch_device).to(torch.int)

        # Do a forward pass to fill the prefix cache and compile the kernels if necessary
        logits = self.quantized_model(
            input_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        )[0]
        next_token = torch.argmax(logits[:, [-1]], dim=-1).to(torch.int)
        generated_ids[:, [seq_length]] = next_token

        with torch.no_grad():
            # Compile the CUDA graph
            decode_one_tokens = torch.compile(decode_one_tokens, mode="default", backend="inductor", fullgraph=True)

            # Generate tokens one by one
            cache_position = torch.tensor([seq_length + 1], device=torch_device)
            for _ in range(1, self.max_new_tokens):
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    next_token = decode_one_tokens(
                        self.quantized_model, next_token.clone(), None, cache_position, past_key_values
                    )
                    generated_ids.index_copy_(1, cache_position, next_token)
                cache_position += 1

        # Check generated text
        self.assertEqual(
            self.tokenizer.decode(generated_ids[0], skip_special_tokens=True), self.EXPECTED_OUTPUT_COMPILE
        )
