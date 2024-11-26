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

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitNetConfig,
    OPTForCausalLM,
)
from transformers.testing_utils import (
    require_accelerate,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_accelerate_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import init_empty_weights


@require_torch_gpu
class BitNetConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = BitNetConfig()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])


@slow
@require_torch_gpu
@require_accelerate
class BitNetTest(unittest.TestCase):
    model_name = "HF1BitLLM/Llama3-8B-1.58-100B-tokens"
    device = "cuda"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Load the model
        """
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(cls.model_name, device_map=cls.device)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_replace_with_bitlinear(self):
        from transformers.integrations import BitLinear, replace_with_bitnet_linear

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id)

        with init_empty_weights():
            model = OPTForCausalLM(config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1

        model = replace_with_bitnet_linear(model)
        nb_bitnet_linear = 0
        for module in model.modules():
            if isinstance(module, BitLinear):
                nb_bitnet_linear += 1

        self.assertEqual(nb_linears - 1, nb_bitnet_linear)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_text = "What are we having for dinner?"
        expected_output = "What are we having for dinner? What are we going to do for fun this weekend?"
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")

        output = self.quantized_model.generate(**input_ids, max_new_tokens=11, do_sample=False)
        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), expected_output)

    def test_packing_unpacking(self):
        """
        Simple test the packing and unpacking logic
        """

        from transformers.integrations import pack_weights, unpack_weights

        u = torch.randint(0, 255, (256, 256), dtype=torch.uint8)
        unpacked_u = unpack_weights(u, dtype=torch.bfloat16)
        repacked_u = pack_weights(unpacked_u)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                self.assertEqual(repacked_u[i][j], u[i][j])

    def test_activation_quant(self):
        """
        test the activation function behaviour
        """

        from transformers.integrations import BitLinear

        layer = BitLinear(in_features=4, out_features=2, bias=False, dtype=torch.float32)
        layer.to(self.device)

        input_tensor = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32).to(torch_device)

        # Quantize the input tensor
        quantized_tensor, scale = layer.activation_quant(input_tensor)

        # Verify the output quantized tensor
        for i in range(input_tensor.shape[0]):
            self.assertEqual(quantized_tensor[i] / scale, input_tensor[i])

        # Verify the scale tensor
        self.assertEqual(scale, 127)

    def test_weights_dtype(self):
        """
        test the weights dtype after loading
        """

        self_attn_q = self.quantized_model.model.layers[0].self_attn.q_proj.weight
        self_attn_k = self.quantized_model.model.layers[0].self_attn.k_proj.weight
        self_attn_v = self.quantized_model.model.layers[0].self_attn.v_proj.weight
        self_attn_o = self.quantized_model.model.layers[0].self_attn.o_proj.weight
        mlp_gate = self.quantized_model.model.layers[0].mlp.gate_proj.weight
        mlp_up = self.quantized_model.model.layers[0].mlp.up_proj.weight
        mlp_down = self.quantized_model.model.layers[0].mlp.down_proj.weight

        self.assertEqual(self_attn_q.dtype, torch.uint8)
        self.assertEqual(self_attn_k.dtype, torch.uint8)
        self.assertEqual(self_attn_v.dtype, torch.uint8)
        self.assertEqual(self_attn_o.dtype, torch.uint8)
        self.assertEqual(mlp_up.dtype, torch.uint8)
        self.assertEqual(mlp_gate.dtype, torch.uint8)
        self.assertEqual(mlp_down.dtype, torch.uint8)

    def test_replace_with_bitlinear_shape(self):
        """
        test that the BitNet layer weight shapes are correct, and the weight_scale is correctly initialized to 1
        """

        from transformers.integrations import replace_with_bitnet_linear

        out_features = 1024
        in_features = 512

        class SimpleLinearModule(torch.nn.Module):
            """
            Simple class to test BitLinear
            """

            def __init__(
                self,
                in_features: int = in_features,
                out_features: int = out_features,
                bias: bool = False,
            ):
                super().__init__()
                self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

            def forward(self, x):
                return self.linear(x)

        model = SimpleLinearModule()
        replace_with_bitnet_linear(model)

        self.assertEqual(list(model.linear.weight.shape), [out_features // 4, in_features])
        self.assertEqual(model.linear.weight_scale, 1)


@slow
@require_torch_gpu
@require_accelerate
class BitNetSerializationTest(unittest.TestCase):
    def test_model_serialization(self):
        model_name = "HF1BitLLM/Llama3-8B-1.58-100B-tokens"
        device = "cuda"
        quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        input_tensor = torch.zeros((1, 8), dtype=torch.int32, device=device)

        with torch.no_grad():
            logits_ref = quantized_model.forward(input_tensor).logits

        # Save
        saved_model_id = "quant_model"
        quantized_model.save_pretrained(saved_model_id)

        # Remove old model
        del quantized_model
        torch.cuda.empty_cache()

        # Load and check if the logits match
        model_loaded = AutoModelForCausalLM.from_pretrained("quant_model", device_map=device)

        with torch.no_grad():
            logits_loaded = model_loaded.forward(input_tensor).logits

        self.assertEqual((logits_loaded - logits_ref).abs().mean().item(), 0)
