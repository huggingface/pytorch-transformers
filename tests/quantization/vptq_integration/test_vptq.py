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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, VptqConfig
from transformers.testing_utils import (
    require_accelerate,
    require_torch_gpu,
    require_torch_multi_gpu,
    require_vptq,
    slow,
    torch_device,
)
from transformers.utils import is_accelerate_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import init_empty_weights


class VptqConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = VptqConfig()
        vptq_orig_config = quantization_config.to_dict()

        self.assertEqual(vptq_orig_config["quant_method"], quantization_config.quant_method)


@slow
@require_torch_gpu
@require_vptq
@require_accelerate
class VptqTest(unittest.TestCase):
    model_name = "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft"

    input_text = "Hello my name is"
    max_new_tokens = 32

    EXPECTED_OUTPUT = "Hello my name is Sarah and I am a 25 year old woman from the United States. I am a college graduate and I am currently working as a marketing specialist for a small"

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

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_raise_if_non_quantized(self):
        model_id = "facebook/opt-125m"
        quantization_config = VptqConfig()

        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_quantized_model_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        quantized_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")

        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)

        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """
        from vptq import VQuantLinear

        from transformers.integrations import replace_with_vptq_linear

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        modules_to_not_convert = ["lm_head"]
        names = [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "fc1",
            "fc2",
        ]
        value = {
            "enable_norm": True,
            "enable_perm": True,
            "group_num": 1,
            "group_size": 128,
            "indices_as_float": False,
            "num_centroids": [-1, 128],
            "num_res_centroids": [-1, 128],
            "outlier_size": 0,
            "vector_lens": [-1, 12],
        }
        shared_layer_config = {}
        for name in names:
            shared_layer_config[name] = value
        for i in range(24):
            modules_to_not_convert.append("model.decoder.layers.{layer_idx}.fc1".format(layer_idx=i))
        layer_configs = {}
        layer_configs["model.decoder.project_out"] = value
        layer_configs["model.decoder.project_in"] = value
        quantization_config = VptqConfig(config_for_layers=layer_configs, shared_layer_config=shared_layer_config)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1

        model, _ = replace_with_vptq_linear(model, quantization_config=quantization_config)
        nb_vptq_linear = 0
        for module in model.modules():
            if isinstance(module, VQuantLinear):
                nb_vptq_linear += 1

        self.assertEqual(nb_linears - 1, nb_vptq_linear)

        # Try with `linear_weights_not_to_quantize`
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        quantization_config = VptqConfig(config_for_layers=layer_configs, shared_layer_config=shared_layer_config)
        model, _ = replace_with_vptq_linear(
            model, quantization_config=quantization_config, modules_to_not_convert=modules_to_not_convert
        )
        nb_vptq_linear = 0
        for module in model.modules():
            if isinstance(module, VQuantLinear):
                nb_vptq_linear += 1
        # 25 comes from 24 decoder.layers.{layer_idx}.fc1
        # and the last lm_head
        self.assertEqual(nb_linears - 25, nb_vptq_linear)
