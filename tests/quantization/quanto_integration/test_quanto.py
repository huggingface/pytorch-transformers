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

import tempfile
import unittest

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from transformers.testing_utils import (
    require_accelerate,
    require_optimum_quanto,
    require_read_token,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_accelerate_available, is_optimum_quanto_available, is_torch_available


if is_torch_available():
    import torch

    from transformers import LlamaForCausalLM, LlamaTokenizer

if is_accelerate_available():
    from accelerate import init_empty_weights

if is_optimum_quanto_available():
    from optimum.quanto import QLayerNorm, QLinear

    from transformers.integrations.quanto import replace_with_quanto_layers


class QuantoConfigTest(unittest.TestCase):
    def test_attributes(self):
        pass


@require_optimum_quanto
@require_accelerate
class QuantoTestIntegration(unittest.TestCase):
    model_id = "facebook/opt-350m"

    def setUp(self):
        config = AutoConfig.from_pretrained(self.model_id)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config)
        self.nb_linear = 0
        self.nb_layernorm = 0
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                self.nb_linear += 1
            elif isinstance(module, torch.nn.LayerNorm):
                self.nb_layernorm += 1

    def test_weight_only_quantization_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly when using weight only quantization
        """

        # Try with weight only quantization
        quantization_config = QuantoConfig(weights="int8", activations=None)
        self.model, _ = replace_with_quanto_layers(self.model, quantization_config=quantization_config)

        nb_qlinear = 0
        for module in self.model.modules():
            if isinstance(module, QLinear):
                nb_qlinear += 1

        self.assertEqual(self.nb_linear, nb_qlinear)

    def test_weight_and_activation_quantization_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly when using weight + activation quantization
        """

        # Try with weight + activation quantization
        quantization_config = QuantoConfig(weights="int8", activations="int8")
        self.model, _ = replace_with_quanto_layers(self.model, quantization_config=quantization_config)

        nb_qlinear = 0
        nb_qlayernorm = 0
        for module in self.model.modules():
            if isinstance(module, QLinear):
                nb_qlinear += 1
            if isinstance(module, QLayerNorm):
                nb_qlayernorm += 1

        self.assertEqual(self.nb_linear, nb_qlinear)
        self.assertEqual(self.nb_layernorm, nb_qlayernorm)

    def test_conversion_with_modules_to_not_convert(self):
        """
        Simple test that checks if the quantized model has been converted properly when specifying modules_to_not_convert argument
        """

        # Try with weight + activatioin quantization
        quantization_config = QuantoConfig(weights="int8", activations="int8")
        self.model, _ = replace_with_quanto_layers(
            self.model, quantization_config=quantization_config, modules_to_not_convert=["lm_head"]
        )

        nb_qlinear = 0
        nb_qlayernorm = 0
        for module in self.model.modules():
            if isinstance(module, QLinear):
                nb_qlinear += 1
            if isinstance(module, QLayerNorm):
                nb_qlayernorm += 1

        self.assertEqual(self.nb_linear - 1, nb_qlinear)


@slow
@require_torch_accelerator
@require_optimum_quanto
@require_accelerate
class QuantoQuantizationTest(unittest.TestCase):
    """
    Test 8-bit weights only quantization
    """

    model_name = "bigscience/bloom-560m"

    weights = "int8"
    activations = None
    device_map = "cpu"

    input_text = "Hello my name is"
    EXPECTED_OUTPUTS = "Hello my name is John, I am a professional photographer and I"

    def setUp(self):
        """
        Setup quantized model
        """
        quantization_config = QuantoConfig(
            weights=self.weights,
            activations=self.activations,
        )

        self.quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            quantization_config=quantization_config,
            torch_dtype=torch.float32,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.have_accelerate_hooks = (
            getattr(self.quantized_model, "hf_device_map", False) and len(self.quantized_model.hf_device_map) > 1
        )

    def check_inference_correctness(self, model, device):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        if not self.have_accelerate_hooks:
            model.to(device)
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(device), max_new_tokens=10)
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality_cpu(self):
        """
        Simple test to check the quality of the model on cpu by comparing the generated tokens with the expected tokens
        """
        self.check_inference_correctness(self.quantized_model, "cpu")

    def test_generate_quality_cuda(self):
        """
        Simple test to check the quality of the model on cuda by comparing the generated tokens with the expected tokens
        """
        self.check_inference_correctness(self.quantized_model, "cuda")

    def test_quantized_model_layers(self):
        from optimum.quanto import QBitsTensor, QModuleMixin, QTensor

        """
        Suite of simple test to check if the layers are quantized and are working properly
        """
        # Test the type of the quantized layer
        self.assertTrue(isinstance(self.quantized_model.transformer.h[0].self_attention.query_key_value, QModuleMixin))
        self.assertTrue(
            isinstance(self.quantized_model.transformer.h[0].self_attention.query_key_value.weight, QTensor)
        )
        if self.weights == "int4":
            self.assertTrue(
                isinstance(self.quantized_model.transformer.h[0].self_attention.query_key_value.weight, QBitsTensor)
            )

        # check that the lm_head was indeed not quantized, just like bnb
        self.assertTrue(
            isinstance(self.quantized_model.lm_head, torch.nn.Linear)
            and not isinstance(self.quantized_model.lm_head, QModuleMixin)
        )
        if self.device_map in ["cpu", "cuda"]:
            self.assertEqual(
                self.quantized_model.transformer.h[0].self_attention.query_key_value.weight._data.device.type,
                self.device_map,
            )
            self.quantized_model.to(0)
        self.assertEqual(
            self.quantized_model.transformer.h[0].self_attention.query_key_value.weight._data.device.type, "cuda"
        )

    def test_serialization_bin(self):
        """
        Test the serialization, the loading and the inference of the quantized weights
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(ValueError) as e:
                self.quantized_model.save_pretrained(tmpdirname, safe_serialization=False)
            self.assertIn("The model is quantized with quanto and is not serializable", str(e.exception))
            # TODO: replace by the following when it works
            # quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(
            #     tmpdirname, torch_dtype=torch.float32, device_map="cpu"
            # )
            # self.check_inference_correctness(quantized_model_from_saved, device="cuda")

    def test_serialization_safetensors(self):
        """
        Test the serialization, the loading and the inference of the quantized weights
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(ValueError) as e:
                self.quantized_model.save_pretrained(tmpdirname)
            self.assertIn("The model is quantized with quanto and is not serializable", str(e.exception))
            # quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(
            #     tmpdirname, torch_dtype=torch.float32, device_map="cpu"
            # )
            # self.check_inference_correctness(quantized_model_from_saved, device="cuda")

    def check_same_model(self, model1, model2):
        d0 = dict(model1.named_parameters())
        d1 = dict(model2.named_parameters())
        self.assertTrue(d0.keys() == d1.keys())
        for k in d0.keys():
            self.assertTrue(d0[k].shape == d1[k].shape)
            self.assertTrue(d0[k].device.type == d1[k].device.type)
            self.assertTrue(d0[k].device == d1[k].device)
            self.assertTrue(d0[k].dtype == d1[k].dtype)
            self.assertTrue(torch.equal(d0[k], d1[k].to(d0[k].device)))

    def test_compare_with_quanto(self):
        from optimum.quanto import freeze, qint4, qint8, quantize

        w_mapping = {"int8": qint8, "int4": qint4}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=torch.float32,
        )
        # we do not quantize the lm_head since we don't do that in transformers
        quantize(model.transformer, weights=w_mapping[self.weights])
        freeze(model.transformer)
        self.check_same_model(model, self.quantized_model)
        self.check_inference_correctness(model, device=torch_device)

    @unittest.skip
    def test_load_from_quanto_saved(self):
        from optimum.quanto import freeze, qint4, qint8, quantize

        from transformers import QuantoConfig

        w_mapping = {"int8": qint8, "int4": qint4}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=torch.float32,
        )
        # we do not quantize the lm_head since we don't do that in transformers
        quantize(model.transformer, weights=w_mapping[self.weights])
        freeze(model.transformer)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.config.quantization_config = QuantoConfig(
                weights=self.weights, activations=self.activations, modules_to_not_convert=["lm_head"]
            )
            model.save_pretrained(tmpdirname, safe_serialization=False)
            quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(
                tmpdirname,
                device_map=self.device_map,
                torch_dtype=torch.float32,
            )
        self.check_same_model(model, quantized_model_from_saved)
        self.check_inference_correctness(quantized_model_from_saved, device="cuda")


class QuantoQuantizationOffloadTest(QuantoQuantizationTest):
    device_map = {
        "transformer.word_embeddings": 0,
        "transformer.word_embeddings_layernorm": 0,
        "transformer.ln_f": 0,
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
        "transformer.h.10": 0,
        "transformer.h.11": 0,
        "transformer.h.12": 0,
        "transformer.h.13": 0,
        "transformer.h.14": 0,
        "transformer.h.15": 0,
        "transformer.h.16": 0,
        "transformer.h.17": 0,
        "transformer.h.18": 0,
        "transformer.h.19": 0,
        "transformer.h.20": 0,
        "transformer.h.21": 0,
        "transformer.h.22": "cpu",
        "transformer.h.23": "disk",
        "lm_head": 0,
    }

    @unittest.skip(reason="The execution device is a gpu")
    def test_generate_quality_cpu(self):
        pass

    @unittest.skip(reason="We can't save offloaded values")
    def test_serialization_bin(self):
        pass

    @unittest.skip
    def test_serialization_safetensors(self):
        pass

    @unittest.skip
    def test_compare_with_quanto(self):
        pass

    @unittest.skip
    def test_load_from_quanto_saved(self):
        pass

    def test_check_offload_quantized(self):
        """
        We check that we have unquantized value in the cpu and in the disk
        """
        from optimum.quanto import QBitsTensor, QTensor

        cpu_weights = self.quantized_model.transformer.h[22].self_attention.query_key_value._hf_hook.weights_map[
            "weight"
        ]
        disk_weights = self.quantized_model.transformer.h[23].self_attention.query_key_value._hf_hook.weights_map[
            "weight"
        ]
        self.assertTrue(isinstance(cpu_weights, torch.Tensor) and not isinstance(cpu_weights, QTensor))
        self.assertTrue(isinstance(disk_weights, torch.Tensor) and not isinstance(disk_weights, QTensor))
        if self.weights == "int4":
            self.assertTrue(isinstance(cpu_weights, torch.Tensor) and not isinstance(disk_weights, QBitsTensor))
            self.assertTrue(isinstance(disk_weights, torch.Tensor) and not isinstance(disk_weights, QBitsTensor))


@unittest.skip(reason="Skipping test class because serialization is not supported yet")
class QuantoQuantizationSerializationTest(QuantoQuantizationTest):
    """
    Perform the same tests as in QuantoQuantizationTest but with a serialized model.
    """

    def setUp(self):
        """
        Setup quantized model
        """
        quantization_config = QuantoConfig(
            weights=self.weights,
            activations=self.activations,
        )
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            quantization_config=quantization_config,
            torch_dtype=torch.float32,
        )
        with tempfile.TemporaryDirectory() as tmpdirname:
            quantized_model.save_pretrained(tmpdirname, safe_serialization=False)
            self.quantized_model = AutoModelForCausalLM.from_pretrained(
                tmpdirname, torch_dtype=torch.float32, device_map=self.device_map
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.have_accelerate_hooks = (
            getattr(self.quantized_model, "hf_device_map", False) and len(self.quantized_model.hf_device_map) > 1
        )


@unittest.skip(reason="Skipping test class because serialization is not supported yet")
class QuantoQuantizationSerializationCudaTest(QuantoQuantizationTest):
    """
    Perform the same tests as in QuantoQuantizationTest but with model on cuda
    """

    device_map = "cuda:0"


class QuantoQuantizationQBitsTensorTest(QuantoQuantizationTest):
    EXPECTED_OUTPUTS = "Hello my name is John, I am a professional photographer, I"
    weights = "int4"


class QuantoQuantizationQBitsTensorOffloadTest(QuantoQuantizationOffloadTest):
    EXPECTED_OUTPUTS = "Hello my name is John, I am a professional photographer, I"
    weights = "int4"


@unittest.skip(reason="Skipping test class because serialization is not supported yet")
class QuantoQuantizationQBitsTensorSerializationTest(QuantoQuantizationSerializationTest):
    EXPECTED_OUTPUTS = "Hello my name is John, I am a professional photographer, I"
    weights = "int4"


@require_torch_gpu
class QuantoQuantizationActivationTest(unittest.TestCase):
    def test_quantize_activation(self):
        quantization_config = QuantoConfig(
            weights="int8",
            activations="int8",
        )
        with self.assertRaises(ValueError) as e:
            AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", quantization_config=quantization_config)
        self.assertIn("We don't support quantizing the activations with transformers library", str(e.exception))


@require_optimum_quanto
@require_torch_gpu
class QuantoKVCacheQuantizationTest(unittest.TestCase):
    @slow
    @require_read_token
    def test_quantized_cache(self):
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 1) the speed of light is the same for all observers, and 2) the laws of physics are the same for all observers.\nThe first part of the theory is the most",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", pad_token="</s>", padding_side="left")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map="sequential", torch_dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(torch_device)

        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False, cache_implementation="quantized")
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
