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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, EetqConfig, OPTForCausalLM
from transformers.testing_utils import (
    require_accelerate,
    require_eetq,
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
class EetqConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = EetqConfig()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {"modules_to_not_convert": ["lm_head.weight"], "quant_method": "eetq", "weights": "int8"}
        quantization_config = EetqConfig.from_dict(dict)

        self.assertEqual(dict["modules_to_not_convert"], quantization_config.modules_to_not_convert)
        self.assertEqual(dict["quant_method"], quantization_config.quant_method)
        self.assertEqual(dict["weights"], quantization_config.weights)


@slow
@require_torch_gpu
@require_eetq
@require_accelerate
class EetqTest(unittest.TestCase):
    model_name = "facebook/opt-350m"

    input_text = "What are we having for dinner?"
    max_new_tokens = 9

    EXPECTED_OUTPUT = "What are we having for dinner?\nI'm having a steak and a salad"

    device_map = "cuda"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        quantization_config = EetqConfig(weights="int8")
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map=cls.device_map, quantization_config=quantization_config
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """
        from eetq import EetqLinear

        from transformers.integrations import replace_with_eetq_linear

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        quantization_config = EetqConfig(weights="int8")

        with init_empty_weights():
            model = OPTForCausalLM(config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1

        model = replace_with_eetq_linear(model, quantization_config=quantization_config)
        nb_eetq_linear = 0
        for module in model.modules():
            if isinstance(module, EetqLinear):
                nb_eetq_linear += 1

        self.assertEqual(nb_linears - 1, nb_eetq_linear)

        # Try with `modules_to_not_convert`
        with init_empty_weights():
            model = OPTForCausalLM(config)
        quantization_config = EetqConfig(modules_to_not_convert=["fc1"])
        model = replace_with_eetq_linear(model, quantization_config=quantization_config)
        nb_eetq_linear = 0
        for module in model.modules():
            if isinstance(module, EetqLinear):
                nb_eetq_linear += 1
        # 25 corresponds to the lm_head along with 24 fc1 layers.
        self.assertEqual(nb_linears - 25, nb_eetq_linear)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

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
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUS
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)
        quantization_config = EetqConfig()
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=quantization_config
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)
