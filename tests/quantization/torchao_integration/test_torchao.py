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

from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from transformers.testing_utils import (
    require_torch_gpu,
    require_torch_multi_gpu,
    require_torchao,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchao_available


if is_torch_available():
    import torch

if is_torchao_available():
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.dtypes.affine_quantized_tensor import TensorCoreTiledLayoutType


def check_torchao_quantized(test_module, qlayer, batch_size=1, context_size=1024):
    weight = qlayer.weight
    test_module.assertTrue(isinstance(weight, AffineQuantizedTensor))
    test_module.assertEqual(weight.quant_min, 0)
    test_module.assertEqual(weight.quant_max, 15)
    test_module.assertTrue(isinstance(weight.layout_type, TensorCoreTiledLayoutType))


def check_forward(test_module, model, batch_size=1, context_size=1024):
    # Test forward pass
    with torch.no_grad():
        out = model(torch.zeros([batch_size, context_size], device=model.device, dtype=torch.int32)).logits
    test_module.assertEqual(out.shape[0], batch_size)
    test_module.assertEqual(out.shape[1], context_size)


@require_torch_gpu
@require_torchao
class TorchAoConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = TorchAoConfig("int4_weight_only")
        torchao_orig_config = quantization_config.to_dict()

        for key in torchao_orig_config:
            self.assertEqual(getattr(quantization_config, key), torchao_orig_config[key])

    def test_post_init_check(self):
        """
        Test kwargs validations in TorchAoConfig
        """
        _ = TorchAoConfig("int4_weight_only")
        with self.assertRaisesRegex(ValueError, "is not supported yet"):
            _ = TorchAoConfig("fp6")

        with self.assertRaisesRegex(ValueError, "Unexpected keyword arg"):
            _ = TorchAoConfig("int4_weight_only", group_size1=32)

    def test_repr(self):
        """
        Check that there is no error in the repr
        """
        quantization_config = TorchAoConfig("int4_weight_only", modules_to_not_convert=["conv"], group_size=8)
        repr(quantization_config)


@require_torch_gpu
@require_torchao
class TorchAoTest(unittest.TestCase):
    input_text = "What are we having for dinner?"
    max_new_tokens = 10

    EXPECTED_OUTPUT = "What are we having for dinner?\n- 1. What is the temperature outside"

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_int4wo_quant(self):
        """
        Simple LLM model testing int4 weight only quantization
        """
        quant_config = TorchAoConfig("int4_weight_only", group_size=32)

        # Note: we quantize the bfloat16 model on the fly to int4
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        check_torchao_quantized(self, quantized_model.model.layers[0].self_attn.v_proj)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_int4wo_quant_bfloat16_conversion(self):
        """
        Testing the dtype of model will be modified to be bfloat16 for int4 weight only quantization
        """
        quant_config = TorchAoConfig("int4_weight_only", group_size=32)

        # Note: we quantize the bfloat16 model on the fly to int4
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=None,
            device_map=torch_device,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        check_torchao_quantized(self, quantized_model.model.layers[0].self_attn.v_proj)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_int4wo_quant_multi_gpu(self):
        """
        Simple test that checks if the quantized model int4 wieght only is working properly with multiple GPUs
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUS
        """

        quant_config = TorchAoConfig("int4_weight_only", group_size=32)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_int4wo_offload(self):
        """
        Simple test that checks if the quantized model int4 wieght only is working properly with cpu/disk offload
        """

        device_map_offload = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 0,
            "model.layers.2": 0,
            "model.layers.3": 0,
            "model.layers.4": 0,
            "model.layers.5": 0,
            "model.layers.6": 0,
            "model.layers.7": 0,
            "model.layers.8": 0,
            "model.layers.9": 0,
            "model.layers.10": 0,
            "model.layers.11": 0,
            "model.layers.12": 0,
            "model.layers.13": 0,
            "model.layers.14": 0,
            "model.layers.15": 0,
            "model.layers.16": 0,
            "model.layers.17": 0,
            "model.layers.18": 0,
            "model.layers.19": "cpu",
            "model.layers.20": "cpu",
            "model.layers.21": "disk",
            "model.norm": 0,
            "model.rotary_emb": 0,
            "lm_head": 0,
        }

        quant_config = TorchAoConfig("int4_weight_only", group_size=32)

        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map_offload,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = "What are we having for dinner?\n- 2. What is the temperature outside"

        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_int8_dynamic_activation_int8_weight_quant(self):
        """
        Simple LLM model testing int8_dynamic_activation_int8_weight
        """
        quant_config = TorchAoConfig("int8_dynamic_activation_int8_weight")

        # Note: we quantize the bfloat16 model on the fly to int4
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=torch_device,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(torch_device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)


@require_torch_gpu
@require_torchao
class TorchAoSerializationTest(unittest.TestCase):
    input_text = "What are we having for dinner?"
    max_new_tokens = 10
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n- 1. What is the temperature outside"
    # TODO: investigate why we don't have the same output as the original model for this test
    SERIALIZED_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    quant_scheme, quant_scheme_kwargs = "int4_weight_only", {"group_size": 32}
    device = "cuda:0"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        cls.quant_config = TorchAoConfig(cls.quant_scheme, **cls.quant_scheme_kwargs)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            torch_dtype=torch.bfloat16,
            device_map=cls.device,
            quantization_config=cls.quant_config,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_original_model_expected_output(self):
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device)
        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)

        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.ORIGINAL_EXPECTED_OUTPUT)

    def check_serialization_expected_output(self, device, expected_output):
        """
        Test if we can serialize and load/infer the model again on the same device
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname, safe_serialization=False)
            loaded_quantized_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, device_map=self.device
            )
            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device)

            output = loaded_quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), expected_output)

    def test_serialization_expected_output(self):
        self.check_serialization_expected_output(self.device, self.SERIALIZED_EXPECTED_OUTPUT)


class TorchAoSerializationW8A8Test(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_dynamic_activation_int8_weight", {}
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
    device = "cuda:0"


class TorchAoSerializationW8Test(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_weight_only", {}
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
    device = "cuda:0"


class TorchAoSerializationW8A8CPUTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_dynamic_activation_int8_weight", {}
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
    device = "cpu"

    def test_serialization_expected_output_cuda(self):
        """
        Test if we can serialize on device (cpu) and load/infer the model on cuda
        """
        new_device = "cuda:0"
        self.check_serialization_expected_output(new_device, self.SERIALIZED_EXPECTED_OUTPUT)


class TorchAoSerializationW8CPUTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_weight_only", {}
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
    device = "cpu"

    def test_serialization_expected_output_cuda(self):
        """
        Test if we can serialize on device (cpu) and load/infer the model on cuda
        """
        new_device = "cuda:0"
        self.check_serialization_expected_output(new_device, self.SERIALIZED_EXPECTED_OUTPUT)


if __name__ == "__main__":
    unittest.main()
