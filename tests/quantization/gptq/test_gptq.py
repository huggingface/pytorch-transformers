# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import tempfile
import unittest

import pytest

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from transformers.testing_utils import (
    is_torch_available,
    require_accelerate,
    require_gptq,
    require_optimum,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)
from transformers.utils import is_auto_gptq_available, is_gptqmodel_available, is_ipex_available


if is_torch_available():
    import torch


class GPTQConfigTest(unittest.TestCase):
    def test_bits(self):
        with self.assertRaises(ValueError):
            GPTQConfig(bits="")
            GPTQConfig(bits=1)
        GPTQConfig(bits=2)
        GPTQConfig(bits=4)

    def test_dataset(self):
        with self.assertRaises(ValueError):
            GPTQConfig(bits=2, dataset="auto_gpt")
        GPTQConfig(bits=2, dataset="c4")

    def test_damp_percent(self):
        with self.assertRaises(ValueError):
            GPTQConfig(bits=2, damp_percent=10)
            GPTQConfig(bits=2, damp_percent=-1)
            GPTQConfig(bits=2, damp_percent="0")
        GPTQConfig(bits=2, damp_percent=0.01)

    def test_to_dict(self):
        quantization_config = GPTQConfig(bits=2)
        quantization_config.to_dict()

    def test_from_dict(self):
        dict = {"bits": 2}
        quantization_config = GPTQConfig.from_dict(dict)
        self.assertEqual(dict["bits"], quantization_config.bits)

    @require_optimum
    def test_optimum_config(self):
        from optimum.gptq import GPTQQuantizer

        config = GPTQConfig(bits=2)
        optimum_config = GPTQQuantizer.from_dict(config.to_dict_optimum())
        self.assertEqual(optimum_config.bits, config.bits)
        new_config = GPTQConfig.from_dict_optimum(optimum_config.to_dict())
        self.assertEqual(optimum_config.bits, new_config.bits)


@slow
@require_optimum
@require_gptq
class GPTQTest(unittest.TestCase):
    model_name = "bigscience/bloom-560m"

    input_text = "Hello my name is"

    EXPECTED_OUTPUTS = set()
    # flaky test: gptqmodel and auto-gptq are not output equivalent nor is string compare deterministic even between transformer/torch versions
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a professional photographer. I")
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a professional photographer and I")
    EXPECTED_OUTPUTS.add("Hello my name is John, I am a student in the University of")
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a very good looking man.")
    EXPECTED_OUTPUTS.add("Hello my name is Alyson, I am a student in the")
    EXPECTED_OUTPUTS.add("Hello my name is Alyson and I am a very sweet,")
    EXPECTED_OUTPUTS.add("Hello my name is Aiden, I am a student at the University")
    EXPECTED_OUTPUTS.add("Hello my name is Nate and I am a member of the N")
    EXPECTED_OUTPUTS.add("Hello my name is Nellie and I am a student at the")

    # this seems a little small considering that we are doing 4bit quant but we have a small model and ww don't quantize the embeddings
    EXPECTED_RELATIVE_DIFFERENCE = 1.664253062

    bits = 4
    sym = True
    group_size = 128
    desc_act = False
    use_exllama = False

    dataset = [
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    ]

    device_map = "cpu" if is_gptqmodel_available() else None

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.model_fp16 = AutoModelForCausalLM.from_pretrained(
            cls.model_name, torch_dtype=torch.float16, device_map=cls.device_map
        )
        cls.mem_fp16 = cls.model_fp16.get_memory_footprint()

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)
        cls.config = AutoConfig.from_pretrained(cls.model_name)

        cls.quantization_config = GPTQConfig(
            bits=cls.bits,
            dataset=cls.dataset,
            tokenizer=cls.tokenizer,
            group_size=cls.group_size,
            desc_act=cls.desc_act,
            sym=cls.sym,
            use_exllama=cls.use_exllama,
        )

        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            torch_dtype=torch.float16,
            device_map=cls.device_map,
            quantization_config=cls.quantization_config,
        )

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model
        """

        mem_quantized = self.quantized_model.get_memory_footprint()

        self.assertAlmostEqual(self.mem_fp16 / mem_quantized, self.EXPECTED_RELATIVE_DIFFERENCE, places=4)

    def test_device_and_dtype_assignment(self):
        r"""
        Test whether trying to cast (or assigning a device to) a model after quantization will throw an error.
        Checks also if other models are casted correctly.
        """
        # This should work
        if self.device_map in (None, "cpu"):
            _ = self.quantized_model.to(0)

        with self.assertRaises(ValueError):
            # Tries with a `dtype``
            self.quantized_model.to(torch.float16)

    def test_original_dtype(self):
        r"""
        A simple test to check if the model succesfully stores the original dtype
        """
        self.assertTrue(hasattr(self.quantized_model.config, "_pre_quantization_dtype"))
        self.assertFalse(hasattr(self.model_fp16.config, "_pre_quantization_dtype"))
        self.assertTrue(self.quantized_model.config._pre_quantization_dtype == torch.float16)

    def test_quantized_layers_class(self):
        """
        Simple test to check if the model conversion has been done correctly by checking on
        the class type of the linear layers of the converted models
        """
        if is_gptqmodel_available():
            from gptqmodel.utils.importer import hf_select_quant_linear

            if hasattr(self.config, "quantization_config"):
                checkpoint_format = self.config.quantization_config.get("checkpoint_format")
                meta = self.config.quantization_config.get("meta")
            else:
                checkpoint_format = "gptq"
                meta = None
            QuantLinear = hf_select_quant_linear(
                bits=self.bits,
                group_size=self.group_size,
                desc_act=self.desc_act,
                sym=self.sym,
                device_map=self.device_map,
                checkpoint_format=checkpoint_format,
                meta=meta,
                backend=self.quantization_config.backend,
            )
        elif is_auto_gptq_available():
            from auto_gptq.utils.import_utils import dynamically_import_QuantLinear as hf_select_quant_linear

            QuantLinear = hf_select_quant_linear(
                use_triton=False,
                desc_act=self.desc_act,
                group_size=self.group_size,
                bits=self.bits,
                disable_exllama=not self.use_exllama,
                disable_exllamav2=True,
            )
        self.assertTrue(self.quantized_model.transformer.h[0].mlp.dense_4h_to_h.__class__ == QuantLinear)

    def check_inference_correctness(self, model):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(model.device), max_new_tokens=10)

        # Get the generation
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def check_quantized_layers_type(self, model, value):
        self.assertTrue(model.transformer.h[0].mlp.dense_4h_to_h.QUANT_TYPE == value)

    def test_generate_quality(self):
        """
        Simple test to check the quality of the model by comparing the generated tokens with the expected tokens
        """
        if self.device_map is None:
            self.check_inference_correctness(self.quantized_model.to(0))
        else:
            if self.device_map == "cpu" and self.quantized_model.device.type != "cpu":
                self.quantized_model.to("cpu")
            self.check_inference_correctness(self.quantized_model)

    def test_serialization(self):
        """
        Test the serialization of the model and the loading of the quantized weights works
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            if is_auto_gptq_available() and not is_gptqmodel_available():
                quant_type = "cuda-old" if not self.use_exllama else "exllama"
                if not self.use_exllama:
                    quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(
                        tmpdirname, quantization_config=GPTQConfig(use_exllama=False, bits=4)
                    )
                    if self.device_map != "cpu":
                        quantized_model_from_saved = quantized_model_from_saved.to(0)
                else:
                    quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(
                        tmpdirname, device_map=self.device_map
                    )
            else:
                if self.device_map == "cpu":
                    quant_type = "ipex" if is_ipex_available() else "torch"
                else:
                    quant_type = "exllama"
                quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(
                    tmpdirname, device_map=self.device_map
                )

            self.check_quantized_layers_type(quantized_model_from_saved, quant_type)
            self.check_inference_correctness(quantized_model_from_saved)

    @require_accelerate
    def test_serialization_big_model_inference(self):
        """
        Test the serialization of the model and the loading of the quantized weights with big model inference
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            device_map = self.device_map or "auto"
            quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=device_map)
            self.check_inference_correctness(quantized_model_from_saved)


@require_torch_gpu
class GPTQTestCUDA(GPTQTest):
    device_map = {"": 0}

    def test_change_loading_attributes(self):
        """
        Test the serialization of the model and the loading of the quantized weights works with another config file
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            if is_auto_gptq_available() and not is_gptqmodel_available() and not self.use_exllama:
                self.check_quantized_layers_type(self.quantized_model, "cuda-old")
                # we need to put it directly to the gpu. Otherwise, we won't be able to initialize the exllama kernel
                quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(
                    tmpdirname, quantization_config=GPTQConfig(use_exllama=True, bits=4), device_map=self.device_map
                )
                self.assertEqual(quantized_model_from_saved.config.quantization_config.bits, self.bits)
                self.check_quantized_layers_type(quantized_model_from_saved, "exllama")
                self.check_inference_correctness(quantized_model_from_saved)


@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMap(GPTQTestCUDA):
    device_map = "auto"


@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMapExllama(GPTQTestCUDA):
    device_map = "auto"
    use_exllama = True


@slow
@require_optimum
@require_gptq
@require_torch_gpu
@require_accelerate
class GPTQTestActOrderExllama(unittest.TestCase):
    """
    Test GPTQ model with exllama kernel and desc_act=True (also known as act-order).
    More information on those arguments here:
    https://huggingface.co/docs/transformers/main_classes/quantization#transformers.GPTQConfig
    """

    EXPECTED_OUTPUTS = set()
    # flaky test: gptqmodel and auto-gptq are not output equivalent nor is string compare deterministic even between transformer/torch versions
    EXPECTED_OUTPUTS.add("Hello, how are you ? I'm doing good, thanks for asking.")
    # 4bit + act_order + 128g
    model_name = "hf-internal-testing/TinyLlama-1.1B-Chat-v0.3-GPTQ"
    input_text = "Hello, how are you ?"

    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.quantization_config = GPTQConfig(bits=4, max_input_length=4028)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            torch_dtype=torch.float16,
            device_map={"": 0},
            quantization_config=cls.quantization_config,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)

    def check_inference_correctness(self, model):
        """
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """

        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        # Get the generation
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_quantized_layers_type(self):
        self.assertTrue(self.quantized_model.model.layers[0].self_attn.k_proj.QUANT_TYPE == "exllama")

    def test_generate_quality(self):
        """
        Simple test to check the quality of the model by comparing the generated tokens with the expected tokens
        """
        self.check_inference_correctness(self.quantized_model)

    def test_max_input_length(self):
        """
        Test if the max_input_length works. It modifies the maximum input length that of the model that runs with exllama backend.
        """

        prompt = "I am in Paris and" * 1000
        inp = self.tokenizer(prompt, return_tensors="pt").to(0)
        self.assertTrue(inp["input_ids"].shape[1] > 4028)
        with self.assertRaises(RuntimeError) as cm:
            self.quantized_model.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)
            self.assertTrue("temp_state buffer is too small" in str(cm.exception))

        prompt = "I am in Paris and"
        inp = self.tokenizer(prompt, return_tensors="pt").to(0)
        self.assertTrue(inp["input_ids"].shape[1] < 4028)
        self.quantized_model.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)


@slow
@require_optimum
@require_gptq
@require_torch_gpu
@require_accelerate
class GPTQTestExllamaV2(unittest.TestCase):
    """
    Test GPTQ model with exllamav2 kernel and desc_act=True (also known as act-order).
    More information on those arguments here:
    https://huggingface.co/docs/transformers/main_classes/quantization#transformers.GPTQConfig
    """

    EXPECTED_OUTPUTS = set()
    # flaky test: gptqmodel and auto-gptq are not output equivalent nor is string compare deterministic even between transformer/torch versions
    EXPECTED_OUTPUTS.add("Hello, how are you ? I'm doing good, thanks for asking.")
    # 4bit + act_order + 128g
    model_name = "hf-internal-testing/TinyLlama-1.1B-Chat-v0.3-GPTQ"
    input_text = "Hello, how are you ?"

    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.quantization_config = GPTQConfig(bits=4, exllama_config={"version": 2})
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            torch_dtype=torch.float16,
            device_map={"": 0},
            quantization_config=cls.quantization_config,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)

    def test_quantized_layers_type(self):
        self.assertEqual(
            self.quantized_model.model.layers[0].self_attn.k_proj.QUANT_TYPE,
            "exllama" if is_gptqmodel_available() else "exllamav2",
        )

    def check_inference_correctness(self, model):
        """
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """

        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        # Get the generation
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality(self):
        """
        Simple test to check the quality of the model by comapring the the generated tokens with the expected tokens
        """
        self.check_inference_correctness(self.quantized_model)


# fail when run all together
@pytest.mark.skip
@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMapCPUOffload(GPTQTest):
    device_map = {
        "transformer.word_embeddings": 0,
        "transformer.word_embeddings_layernorm": 0,
        "lm_head": 0,
        "transformer.h.0": 0,
        "transformer.h.1": 0,
        "transformer.h.2": 0,
        "transformer.h.3": 0,
        "transformer.h.4": 0,
        "transformer.h.5": 0,
        "transformer.h.6": 0,
        "transformer.h.7": 0,
        "transformer.h.8": 0,
        "transformer.h.9": 0,
        "transformer.h.10": 1,
        "transformer.h.11": 1,
        "transformer.h.12": 1,
        "transformer.h.13": 1,
        "transformer.h.14": 1,
        "transformer.h.15": 1,
        "transformer.h.16": 1,
        "transformer.h.17": 0,
        "transformer.h.18": "cpu",
        "transformer.h.19": "cpu",
        "transformer.h.20": "cpu",
        "transformer.h.21": "cpu",
        "transformer.h.22": "cpu",
        "transformer.h.23": 1,
        "transformer.ln_f": 0,
    }
