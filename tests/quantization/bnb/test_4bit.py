# coding=utf-8
# Copyright 2022 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import importlib.metadata
import tempfile
import unittest

from packaging import version

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.testing_utils import (
    apply_skip_if_not_implemented,
    is_bitsandbytes_available,
    is_torch_available,
    require_accelerate,
    require_bitsandbytes,
    require_torch,
    require_torch_gpu_if_bnb_not_multi_backend_enabled,
    require_torch_multi_gpu,
    slow,
    torch_device,
)


def get_some_linear_layer(model):
    if model.config.model_type == "gpt2":
        return model.transformer.h[0].mlp.c_fc
    elif model.config.model_type == "opt":
        try:
            return model.decoder.layers[0].fc1
        except AttributeError:
            # for AutoModelforCausalLM
            return model.model.decoder.layers[0].fc1
    elif model.config.model_type == "llama":
        return model.model.layers[0].mlp.gate_proj
    else:
        return model.transformer.h[0].mlp.dense_4h_to_h


if is_torch_available():
    import torch
    import torch.nn as nn

    class LoRALayer(nn.Module):
        """Wraps a linear layer with LoRA-like adapter - Used for testing purposes only"""

        def __init__(self, module: nn.Module, rank: int):
            super().__init__()
            self.module = module
            self.adapter = nn.Sequential(
                nn.Linear(module.in_features, rank, bias=False),
                nn.Linear(rank, module.out_features, bias=False),
            )
            small_std = (2.0 / (5 * min(module.in_features, module.out_features))) ** 0.5
            nn.init.normal_(self.adapter[0].weight, std=small_std)
            nn.init.zeros_(self.adapter[1].weight)
            self.adapter.to(module.weight.device)

        def forward(self, input, *args, **kwargs):
            return self.module(input, *args, **kwargs) + self.adapter(input)


if is_bitsandbytes_available():
    import bitsandbytes as bnb


@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu_if_bnb_not_multi_backend_enabled
@slow
class Base4bitTest(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "bigscience/bloom-1b7"

    # Constant values
    EXPECTED_RELATIVE_DIFFERENCE = (
        2.109659552692574  # This was obtained on a RTX Titan so the number might slightly change
    )

    input_text = "Hello my name is"
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a professional photographer. I")
    EXPECTED_OUTPUTS.add("Hello my name is John.\nI am a friend of your father.\n")
    EXPECTED_OUTPUTS.add("Hello my name is John Doe, I am a student at the University")
    EXPECTED_OUTPUTS.add("Hello my name is John and I am 25 years old.")
    MAX_NEW_TOKENS = 10

    def setUp(self):
        # Models and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


@apply_skip_if_not_implemented
class Bnb4BitTest(Base4bitTest):
    def setUp(self):
        super().setUp()

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model_4bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True, device_map="auto")

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.model_fp16
        del self.model_4bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_quantization_num_parameters(self):
        r"""
        Test if the number of returned parameters is correct

        See: https://github.com/huggingface/transformers/issues/25978
        """
        num_params_4bit = self.model_4bit.num_parameters()
        num_params_fp16 = self.model_fp16.num_parameters()

        self.assertEqual(num_params_4bit, num_params_fp16)

    def test_quantization_config_json_serialization(self):
        r"""
        A simple test to check if the quantization config is correctly serialized and deserialized
        """
        config = self.model_4bit.config

        self.assertTrue(hasattr(config, "quantization_config"))

        _ = config.to_dict()
        _ = config.to_diff_dict()

        _ = config.to_json_string()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from bitsandbytes.nn import Params4bit

        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_4bit = self.model_4bit.get_memory_footprint()

        self.assertAlmostEqual(mem_fp16 / mem_4bit, self.EXPECTED_RELATIVE_DIFFERENCE, delta=1e-5)
        linear = get_some_linear_layer(self.model_4bit)
        self.assertTrue(linear.weight.__class__ == Params4bit)

    def test_original_dtype(self):
        r"""
        A simple test to check if the model succesfully stores the original dtype
        """
        self.assertTrue(hasattr(self.model_4bit.config, "_pre_quantization_dtype"))
        self.assertFalse(hasattr(self.model_fp16.config, "_pre_quantization_dtype"))
        self.assertTrue(self.model_4bit.config._pre_quantization_dtype == torch.float16)

    def test_linear_are_4bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from transformers import T5PreTrainedModel

        self.model_fp16.get_memory_footprint()
        self.model_4bit.get_memory_footprint()

        for name, module in self.model_4bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name not in ["lm_head"] + T5PreTrainedModel._keep_in_fp32_modules:
                    # 4-bit parameters are packed in uint8 variables
                    self.assertTrue(module.weight.dtype == torch.uint8)

    def test_rwkv_4bit(self):
        r"""
        A simple test to check if 4-bit RWKV inference works as expected.
        """
        model_id = "RWKV/rwkv-4-169m-pile"

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)

        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        tok = AutoTokenizer.from_pretrained(model_id)

        text = "Hello my name is"
        input_ids = tok.encode(text, return_tensors="pt").to(torch_device)

        _ = model.generate(input_ids, max_new_tokens=30)

    def test_generate_quality(self):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = self.model_4bit.generate(
            input_ids=encoded_input["input_ids"].to(torch_device), max_new_tokens=10
        )

        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality_config(self):
        r"""
        Test that loading the model with the config is equivalent
        """
        bnb_config = BitsAndBytesConfig()
        bnb_config.load_in_4bit = True

        model_4bit_from_config = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb_config, device_map="auto"
        )

        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = model_4bit_from_config.generate(
            input_ids=encoded_input["input_ids"].to(torch_device), max_new_tokens=10
        )

        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality_dequantize(self):
        r"""
        Test that loading the model and unquantize it produce correct results
        """
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        model_4bit = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb_config, device_map="auto"
        )

        model_4bit.dequantize()

        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = model_4bit.generate(
            input_ids=encoded_input["input_ids"].to(torch_device), max_new_tokens=10
        )

        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_device_assignment(self):
        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.43.2"):
            self.skipTest(reason="This test requires bitsandbytes >= 0.43.2")

        mem_before = self.model_4bit.get_memory_footprint()

        # Move to CPU
        self.model_4bit.to("cpu")
        self.assertEqual(self.model_4bit.device.type, "cpu")
        self.assertAlmostEqual(self.model_4bit.get_memory_footprint(), mem_before)

        if torch.cuda.is_available():
            # Move back to CUDA device
            self.model_4bit.to("cuda")
            self.assertEqual(self.model_4bit.device.type, "cuda")
            self.assertAlmostEqual(self.model_4bit.get_memory_footprint(), mem_before)

    def test_device_and_dtype_assignment(self):
        r"""
        Test whether attempting to change the device or cast the dtype of a model
        after converting it to 4-bit precision will raise an appropriate error.
        The test ensures that such operations are prohibited on 4-bit models
        to prevent invalid conversions.
        """

        # Moving with `to` or `cuda` is not supported with versions < 0.43.2.
        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.43.2"):
            with self.assertRaises(ValueError):
                # Tries with `str`
                self.model_4bit.to("cpu")

            with self.assertRaises(ValueError):
                # Tries with a `device`
                self.model_4bit.to(torch.device("cuda:0"))

            with self.assertRaises(ValueError):
                # Tries with `cuda`
                self.model_4bit.cuda()

        with self.assertRaises(ValueError):
            # Tries with a `dtype`
            self.model_4bit.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries to cast the 4-bit model to float32 using `float()`
            self.model_4bit.float()

        with self.assertRaises(ValueError):
            # Tries to cast the 4-bit model to float16 using `half()`
            self.model_4bit.half()

        # Test if we did not break anything
        self.model_4bit.to(torch.device(torch_device))

        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        self.model_fp16 = self.model_fp16.to(torch.float32)
        _ = self.model_fp16.generate(input_ids=encoded_input["input_ids"].to(torch_device), max_new_tokens=10)

        if torch.cuda.is_available():
            # Check that this does not throw an error
            _ = self.model_fp16.cuda()

        # Check this does not throw an error
        _ = self.model_fp16.to("cpu")

        # Check this does not throw an error
        _ = self.model_fp16.half()

        # Check this does not throw an error
        _ = self.model_fp16.float()

    def test_fp32_4bit_conversion(self):
        r"""
        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", load_in_4bit=True, device_map="auto")
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.float32)

    def test_bnb_4bit_wrong_config(self):
        r"""
        Test whether creating a bnb config with unsupported values leads to errors.
        """
        with self.assertRaises(ValueError):
            _ = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_storage="add")


@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu_if_bnb_not_multi_backend_enabled
@slow
@apply_skip_if_not_implemented
class Bnb4BitT5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "google-t5/t5-small"
        cls.dense_act_model_name = "google/flan-t5-small"  # flan-t5 uses dense-act instead of dense-relu-dense
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_text = "Translate in German: Hello, my dog is cute"

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        gc.collect()
        torch.cuda.empty_cache()

    def test_inference_without_keep_in_fp32(self):
        r"""
        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        `flan-t5-small` uses `T5DenseGatedActDense` whereas `google-t5/t5-small` uses `T5DenseReluDense`. We need to test
        both cases.
        """
        from transformers import T5ForConditionalGeneration

        modules = T5ForConditionalGeneration._keep_in_fp32_modules
        T5ForConditionalGeneration._keep_in_fp32_modules = None

        # test with `google-t5/t5-small`
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_4bit=True, device_map="auto")
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(model.device)
        _ = model.generate(**encoded_input)

        # test with `flan-t5-small`
        model = T5ForConditionalGeneration.from_pretrained(
            self.dense_act_model_name, load_in_4bit=True, device_map="auto"
        )
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(model.device)
        _ = model.generate(**encoded_input)
        T5ForConditionalGeneration._keep_in_fp32_modules = modules

    def test_inference_with_keep_in_fp32(self):
        r"""
        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        `flan-t5-small` uses `T5DenseGatedActDense` whereas `google-t5/t5-small` uses `T5DenseReluDense`. We need to test
        both cases.
        """
        from transformers import T5ForConditionalGeneration

        # test with `google-t5/t5-small`
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_4bit=True, device_map="auto")

        # there was a bug with decoders - this test checks that it is fixed
        self.assertTrue(isinstance(model.decoder.block[0].layer[0].SelfAttention.q, bnb.nn.Linear4bit))

        encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(model.device)
        _ = model.generate(**encoded_input)

        # test with `flan-t5-small`
        model = T5ForConditionalGeneration.from_pretrained(
            self.dense_act_model_name, load_in_4bit=True, device_map="auto"
        )
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(model.device)
        _ = model.generate(**encoded_input)


@apply_skip_if_not_implemented
class Classes4BitModelTest(Base4bitTest):
    def setUp(self):
        super().setUp()
        # model_name
        self.model_name = "bigscience/bloom-560m"
        self.seq_to_seq_name = "google-t5/t5-small"

        # Different types of model

        self.base_model = AutoModel.from_pretrained(self.model_name, load_in_4bit=True, device_map="auto")
        # Sequence classification model
        self.sequence_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, load_in_4bit=True, device_map="auto"
        )
        # CausalLM model
        self.model_4bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True, device_map="auto")
        # Seq2seq model
        self.seq_to_seq_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.seq_to_seq_name, load_in_4bit=True, device_map="auto"
        )

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.base_model
        del self.sequence_model
        del self.model_4bit
        del self.seq_to_seq_model

        gc.collect()
        torch.cuda.empty_cache()

    def test_correct_head_class(self):
        r"""
        A simple test to check if the last modules for some classes (AutoModelForCausalLM or SequenceClassification)
        are kept in their native class.
        """
        from bitsandbytes.nn import Params4bit

        self.assertTrue(self.base_model.h[-1].mlp.dense_4h_to_h.weight.__class__ == Params4bit)

        # Other heads should be nn.Parameter
        self.assertTrue(self.model_4bit.lm_head.weight.__class__ == torch.nn.Parameter)
        self.assertTrue(self.sequence_model.score.weight.__class__ == torch.nn.Parameter)
        self.assertTrue(self.seq_to_seq_model.lm_head.weight.__class__ == torch.nn.Parameter)


@apply_skip_if_not_implemented
class Pipeline4BitTest(Base4bitTest):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        if hasattr(self, "pipe"):
            del self.pipe

        gc.collect()
        torch.cuda.empty_cache()

    def test_pipeline(self):
        r"""
        The aim of this test is to verify that the mixed 4bit is compatible with `pipeline` from transformers. Since
        we used pipline for inference speed benchmarking we want to make sure that this feature does not break anything
        on pipline.
        """
        # self._clear_cuda_cache()
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={
                "device_map": "auto",
                "load_in_4bit": True,
                # float16 isn't supported on CPU, use bfloat16 instead
                "torch_dtype": torch.bfloat16 if torch_device == "cpu" else torch.float16,
            },
            max_new_tokens=self.MAX_NEW_TOKENS,
        )

        # Real second forward pass
        pipeline_output = self.pipe(self.input_text)
        self.assertIn(pipeline_output[0]["generated_text"], self.EXPECTED_OUTPUTS)


@require_torch_multi_gpu
@apply_skip_if_not_implemented
class Bnb4bitTestMultiGpu(Base4bitTest):
    def setUp(self):
        super().setUp()

    def test_multi_gpu_loading(self):
        r"""
        This tests that the model has been loaded and can be used correctly on a multi-GPU setup.
        Let's just try to load a model on 2 GPUs and see if it works. The model we test has ~2GB of total, 3GB should suffice
        """

        model_parallel = AutoModelForCausalLM.from_pretrained(
            self.model_name, load_in_4bit=True, device_map="balanced"
        )

        # Check correct device map
        self.assertEqual(set(model_parallel.hf_device_map.values()), {0, 1})

        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Second real batch
        output_parallel = model_parallel.generate(
            input_ids=encoded_input["input_ids"].to(torch_device), max_new_tokens=10
        )
        self.assertIn(self.tokenizer.decode(output_parallel[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)


@apply_skip_if_not_implemented
class Bnb4BitTestTraining(Base4bitTest):
    def setUp(self):
        self.model_name = "facebook/opt-350m"
        super().setUp()

    def test_training(self):
        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.37.0"):
            self.skipTest(reason="This test requires bitsandbytes >= 0.37.0")

        # Step 1: freeze all parameters
        model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True)

        if torch.cuda.is_available():
            self.assertEqual(set(model.hf_device_map.values()), {torch.cuda.current_device()})
        elif torch.xpu.is_available():
            self.assertEqual(set(model.hf_device_map.values()), {f"xpu:{torch.xpu.current_device()}"})
        else:
            self.assertTrue(all(param.device.type == "cpu" for param in model.parameters()))

        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        # Step 2: add adapters
        for _, module in model.named_modules():
            if isinstance(module, OPTAttention):
                module.q_proj = LoRALayer(module.q_proj, rank=16)
                module.k_proj = LoRALayer(module.k_proj, rank=16)
                module.v_proj = LoRALayer(module.v_proj, rank=16)

        # Step 3: dummy batch
        batch = self.tokenizer("Test batch ", return_tensors="pt").to(torch_device)

        # Step 4: Check if the gradient is not None
        with torch.autocast(torch_device):
            out = model.forward(**batch)
            out.logits.norm().backward()

        for module in model.modules():
            if isinstance(module, LoRALayer):
                self.assertTrue(module.adapter[1].weight.grad is not None)
                self.assertTrue(module.adapter[1].weight.grad.norm().item() > 0)
            elif isinstance(module, nn.Embedding):
                self.assertTrue(module.weight.grad is None)


@apply_skip_if_not_implemented
@unittest.skipIf(torch_device == "xpu", reason="XPU has precision issue on gpt model, will test it once fixed")
class Bnb4BitGPT2Test(Bnb4BitTest):
    model_name = "openai-community/gpt2-xl"
    EXPECTED_RELATIVE_DIFFERENCE = 3.3191854854152187


@apply_skip_if_not_implemented
class Bnb4BitLlamaTest(Bnb4BitTest):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    EXPECTED_RELATIVE_DIFFERENCE = 2.9461410686392764


@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu_if_bnb_not_multi_backend_enabled
@slow
@apply_skip_if_not_implemented
class BaseSerializationTest(unittest.TestCase):
    model_name = "facebook/opt-125m"
    input_text = "Mars colonists' favorite meals are"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_serialization(self, quant_type="nf4", double_quant=True, safe_serialization=True):
        r"""
        Test whether it is possible to serialize a model in 4-bit. Uses most typical params as default.
        See ExtendedSerializationTest class for more params combinations.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_0 = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            device_map=torch_device,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_0.save_pretrained(tmpdirname, safe_serialization=safe_serialization)

            config = AutoConfig.from_pretrained(tmpdirname)
            self.assertTrue(hasattr(config, "quantization_config"))

            model_1 = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=torch_device)

        # checking quantized linear module weight
        linear = get_some_linear_layer(model_1)
        self.assertTrue(linear.weight.__class__ == bnb.nn.Params4bit)
        self.assertTrue(hasattr(linear.weight, "quant_state"))
        self.assertTrue(linear.weight.quant_state.__class__ == bnb.functional.QuantState)

        # checking memory footpring
        self.assertAlmostEqual(model_0.get_memory_footprint() / model_1.get_memory_footprint(), 1, places=2)

        # Matching all parameters and their quant_state items:
        d0 = dict(model_0.named_parameters())
        d1 = dict(model_1.named_parameters())
        self.assertTrue(d0.keys() == d1.keys())

        for k in d0.keys():
            self.assertTrue(d0[k].shape == d1[k].shape)
            self.assertTrue(d0[k].device.type == d1[k].device.type)
            self.assertTrue(d0[k].device == d1[k].device)
            self.assertTrue(d0[k].dtype == d1[k].dtype)
            self.assertTrue(torch.equal(d0[k], d1[k].to(d0[k].device)))

            if isinstance(d0[k], bnb.nn.modules.Params4bit):
                for v0, v1 in zip(
                    d0[k].quant_state.as_dict().values(),
                    d1[k].quant_state.as_dict().values(),
                ):
                    if isinstance(v0, torch.Tensor):
                        # The absmax will not be saved in the quant_state when using NF4 in CPU
                        if v0.numel() != 0:
                            self.assertTrue(torch.equal(v0, v1.to(v0.device)))
                    else:
                        self.assertTrue(v0 == v1)

        # comparing forward() outputs
        encoded_input = tokenizer(self.input_text, return_tensors="pt").to(torch_device)
        out_0 = model_0(**encoded_input)
        out_1 = model_1(**encoded_input)
        torch.testing.assert_close(out_0["logits"], out_1["logits"], rtol=0.05, atol=0.05)

        # comparing generate() outputs
        encoded_input = tokenizer(self.input_text, return_tensors="pt").to(torch_device)
        output_sequences_0 = model_0.generate(**encoded_input, max_new_tokens=10)
        output_sequences_1 = model_1.generate(**encoded_input, max_new_tokens=10)

        def _decode(token):
            return tokenizer.decode(token, skip_special_tokens=True)

        self.assertEqual(
            [_decode(x) for x in output_sequences_0],
            [_decode(x) for x in output_sequences_1],
        )


@apply_skip_if_not_implemented
class ExtendedSerializationTest(BaseSerializationTest):
    """
    tests more combinations of parameters
    """

    def test_nf4_single_unsafe(self):
        self.test_serialization(quant_type="nf4", double_quant=False, safe_serialization=False)

    def test_nf4_single_safe(self):
        self.test_serialization(quant_type="nf4", double_quant=False, safe_serialization=True)

    def test_nf4_double_unsafe(self):
        self.test_serialization(quant_type="nf4", double_quant=True, safe_serialization=False)

    # nf4 double safetensors quantization is tested in test_serialization() method from the parent class

    def test_fp4_single_unsafe(self):
        self.test_serialization(quant_type="fp4", double_quant=False, safe_serialization=False)

    def test_fp4_single_safe(self):
        self.test_serialization(quant_type="fp4", double_quant=False, safe_serialization=True)

    def test_fp4_double_unsafe(self):
        self.test_serialization(quant_type="fp4", double_quant=True, safe_serialization=False)

    def test_fp4_double_safe(self):
        self.test_serialization(quant_type="fp4", double_quant=True, safe_serialization=True)


class BloomSerializationTest(BaseSerializationTest):
    """
    default BaseSerializationTest config tested with Bloom family model
    """

    model_name = "bigscience/bloom-560m"


class GPTSerializationTest(BaseSerializationTest):
    """
    default BaseSerializationTest config tested with GPT family model
    """

    model_name = "openai-community/gpt2-xl"


class LlamaSerializationTest(BaseSerializationTest):
    """
    default BaseSerializationTest config tested with Llama family model
    """

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@require_bitsandbytes
@require_accelerate
@require_torch_gpu_if_bnb_not_multi_backend_enabled
@slow
@apply_skip_if_not_implemented
class Bnb4BitTestBasicConfigTest(unittest.TestCase):
    def test_load_in_4_and_8_bit_fails(self):
        with self.assertRaisesRegex(ValueError, "load_in_4bit and load_in_8bit are both True"):
            AutoModelForCausalLM.from_pretrained("facebook/opt-125m", load_in_4bit=True, load_in_8bit=True)

    def test_set_load_in_8_bit(self):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        with self.assertRaisesRegex(ValueError, "load_in_4bit and load_in_8bit are both True"):
            quantization_config.load_in_8bit = True
