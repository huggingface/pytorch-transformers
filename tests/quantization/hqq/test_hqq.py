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
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig
from transformers.testing_utils import (
    require_accelerate,
    require_hqq,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_hqq_available, is_torch_available


if is_torch_available():
    import torch

if is_hqq_available():
    from hqq.core.quantize import HQQBackend, HQQLinear


class HQQLLMRunner:
    def __init__(self, model_id, quant_config, compute_dtype, device, cache_dir=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            device_map=device,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.device = self.model.device
        HQQLinear.set_backend(HQQBackend.PYTORCH)


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def check_hqqlayer(test_module, hqq_layer, batch_size=1, context_size=1024):
    # Test HQQ layer
    W_dequant = hqq_layer.dequantize()  # Reconstructed weights
    inputs = (
        torch.randn(
            (batch_size, context_size, hqq_layer.meta["shape"][1]),
            device=hqq_layer.device,
            dtype=hqq_layer.compute_dtype,
        )
        / 10.0
    )
    with torch.no_grad():
        outputs = hqq_layer(inputs)
    test_module.assertEqual(outputs.shape[-1], W_dequant.shape[0])
    test_module.assertEqual(outputs.dtype, hqq_layer.compute_dtype)
    del W_dequant, inputs, outputs
    cleanup()


def check_forward(test_module, model, batch_size=1, context_size=1024):
    # Test forward pass
    with torch.no_grad():
        out = model(torch.zeros([batch_size, context_size], device=model.device, dtype=torch.int32)).logits
    test_module.assertEqual(out.shape[0], batch_size)
    test_module.assertEqual(out.shape[1], context_size)
    cleanup()


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@require_torch_gpu
@require_hqq
class HqqConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = HqqConfig()
        hqq_orig_config = quantization_config.to_dict()

        self.assertEqual(quantization_config.quant_config, hqq_orig_config["quant_config"])


@slow
@require_torch_gpu
@require_accelerate
@require_hqq
class HQQTest(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def test_fp16_quantized_model(self):
        """
        Simple LLM model testing fp16
        """
        quant_config = HqqConfig(nbits=8, group_size=64)

        hqq_runner = HQQLLMRunner(
            model_id=MODEL_ID, quant_config=quant_config, compute_dtype=torch.float16, device=torch_device
        )

        check_hqqlayer(self, hqq_runner.model.model.layers[0].self_attn.v_proj)
        check_forward(self, hqq_runner.model)


@slow
@require_torch_gpu
@require_torch_multi_gpu
@require_accelerate
@require_hqq
class HQQTestMultiGPU(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def test_fp16_quantized_model_multipgpu(self):
        """
        Simple LLM model testing fp16 with multi-gpu
        """

        quant_config = HqqConfig(nbits=8, group_size=64)

        hqq_runner = HQQLLMRunner(
            model_id=MODEL_ID, quant_config=quant_config, compute_dtype=torch.float16, device="auto"
        )

        check_hqqlayer(self, hqq_runner.model.model.layers[0].self_attn.v_proj)
        check_forward(self, hqq_runner.model)


@slow
@require_torch_gpu
@require_accelerate
@require_hqq
class HQQSerializationTest(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def test_model_serialization(self):
        """
        Simple HQQ LLM save/load test
        """
        quant_config = HqqConfig(nbits=4, group_size=64)

        hqq_runner = HQQLLMRunner(
            model_id=MODEL_ID, quant_config=quant_config, compute_dtype=torch.float16, device=torch_device
        )

        input_tensor = torch.zeros((1, 8), dtype=torch.int32, device=torch_device)

        with torch.no_grad():
            logits_ref = hqq_runner.model.forward(input_tensor).logits

        # Save
        saved_model_id = "quant_model"
        hqq_runner.model.save_pretrained(saved_model_id)

        # Remove old model
        del hqq_runner.model
        torch.cuda.empty_cache()

        # Load and check if the logits match
        model_loaded = AutoModelForCausalLM.from_pretrained(
            "quant_model", torch_dtype=torch.float16, device_map=torch_device, low_cpu_mem_usage=True
        )

        with torch.no_grad():
            logits_loaded = model_loaded.forward(input_tensor).logits

        self.assertEqual((logits_loaded - logits_ref).abs().mean().item(), 0)
