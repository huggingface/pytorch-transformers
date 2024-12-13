# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import importlib
import os
import tempfile
import unittest

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from packaging import version

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
    logging,
)
from transformers.testing_utils import (
    CaptureLogger,
    require_bitsandbytes,
    require_peft,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_peft
@require_torch
class PeftTesterMixin:
    peft_test_model_ids = ("peft-internal-testing/tiny-OPTForCausalLM-lora",)
    transformers_test_model_ids = ("hf-internal-testing/tiny-random-OPTForCausalLM",)
    transformers_test_model_classes = (AutoModelForCausalLM, OPTForCausalLM)


# TODO: run it with CI after PEFT release.
@slow
class PeftIntegrationTester(unittest.TestCase, PeftTesterMixin):
    """
    A testing suite that makes sure that the PeftModel class is correctly integrated into the transformers library.
    """

    def _check_lora_correctly_converted(self, model):
        """
        Utility method to check if the model has correctly adapters injected on it.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        is_peft_loaded = False

        for _, m in model.named_modules():
            if isinstance(m, BaseTunerLayer):
                is_peft_loaded = True
                break

        return is_peft_loaded

    def test_peft_from_pretrained(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained`.
        This checks if we pass a remote folder that contains an adapter config and adapter weights, it
        should correctly load a model that has adapters injected on it.
        """
        logger = logging.get_logger("transformers.integrations.peft")

        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                with CaptureLogger(logger) as cl:
                    peft_model = transformers_class.from_pretrained(model_id).to(torch_device)
                # ensure that under normal circumstances, there  are no warnings about keys
                self.assertNotIn("unexpected keys", cl.out)
                self.assertNotIn("missing keys", cl.out)

                self.assertTrue(self._check_lora_correctly_converted(peft_model))
                self.assertTrue(peft_model._hf_peft_config_loaded)
                # dummy generation
                _ = peft_model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    def test_peft_state_dict(self):
        """
        Simple test that checks if the returned state dict of `get_adapter_state_dict()` method contains
        the expected keys.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                state_dict = peft_model.get_adapter_state_dict()

                for key in state_dict.keys():
                    self.assertTrue("lora" in key)

    def test_peft_save_pretrained(self):
        """
        Test that checks various combinations of `save_pretrained` with a model that has adapters loaded
        on it. This checks if the saved model contains the expected files (adapter weights and adapter config).
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)

                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))

                    self.assertTrue("config.json" not in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

                    peft_model = transformers_class.from_pretrained(tmpdirname).to(torch_device)
                    self.assertTrue(self._check_lora_correctly_converted(peft_model))

                    peft_model.save_pretrained(tmpdirname, safe_serialization=False)
                    self.assertTrue("adapter_model.bin" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))

                    peft_model = transformers_class.from_pretrained(tmpdirname).to(torch_device)
                    self.assertTrue(self._check_lora_correctly_converted(peft_model))

    def test_peft_enable_disable_adapters(self):
        """
        A test that checks if `enable_adapters` and `disable_adapters` methods work as expected.
        """
        from peft import LoraConfig

        dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                peft_model.add_adapter(peft_config)

                peft_logits = peft_model(dummy_input).logits

                peft_model.disable_adapters()

                peft_logits_disabled = peft_model(dummy_input).logits

                peft_model.enable_adapters()

                peft_logits_enabled = peft_model(dummy_input).logits

                self.assertTrue(torch.allclose(peft_logits, peft_logits_enabled, atol=1e-12, rtol=1e-12))
                self.assertFalse(torch.allclose(peft_logits_enabled, peft_logits_disabled, atol=1e-12, rtol=1e-12))

    def test_peft_add_adapter(self):
        """
        Simple test that tests if `add_adapter` works as expected
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                self.assertTrue(self._check_lora_correctly_converted(model))
                # dummy generation
                _ = model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    def test_peft_add_adapter_from_pretrained(self):
        """
        Simple test that tests if `add_adapter` works as expected
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                self.assertTrue(self._check_lora_correctly_converted(model))
                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    model_from_pretrained = transformers_class.from_pretrained(tmpdirname).to(torch_device)
                    self.assertTrue(self._check_lora_correctly_converted(model_from_pretrained))

    def test_peft_add_adapter_modules_to_save(self):
        """
        Simple test that tests if `add_adapter` works as expected when training with
        modules to save.
        """
        from peft import LoraConfig
        from peft.utils import ModulesToSaveWrapper

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

                model = transformers_class.from_pretrained(model_id).to(torch_device)
                peft_config = LoraConfig(init_lora_weights=False, modules_to_save=["lm_head"])
                model.add_adapter(peft_config)
                self._check_lora_correctly_converted(model)

                _has_modules_to_save_wrapper = False
                for name, module in model.named_modules():
                    if isinstance(module, ModulesToSaveWrapper):
                        _has_modules_to_save_wrapper = True
                        self.assertTrue(module.modules_to_save.default.weight.requires_grad)
                        self.assertTrue("lm_head" in name)
                        break

                self.assertTrue(_has_modules_to_save_wrapper)
                state_dict = model.get_adapter_state_dict()

                self.assertTrue("lm_head.weight" in state_dict.keys())

                logits = model(dummy_input).logits
                loss = logits.mean()
                loss.backward()

                for _, param in model.named_parameters():
                    if param.requires_grad:
                        self.assertTrue(param.grad is not None)

    def test_peft_add_adapter_training_gradient_checkpointing(self):
        """
        Simple test that tests if `add_adapter` works as expected when training with
        gradient checkpointing.
        """
        from peft import LoraConfig

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                self.assertTrue(self._check_lora_correctly_converted(model))

                # When attaching adapters the input embeddings will stay frozen, this will
                # lead to the output embedding having requires_grad=False.
                dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)
                frozen_output = model.get_input_embeddings()(dummy_input)
                self.assertTrue(frozen_output.requires_grad is False)

                model.gradient_checkpointing_enable()

                # Since here we attached the hook, the input should have requires_grad to set
                # properly
                non_frozen_output = model.get_input_embeddings()(dummy_input)
                self.assertTrue(non_frozen_output.requires_grad is True)

                # To repro the Trainer issue
                dummy_input.requires_grad = False

                for name, param in model.named_parameters():
                    if "lora" in name.lower():
                        self.assertTrue(param.requires_grad)

                logits = model(dummy_input).logits
                loss = logits.mean()
                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.assertTrue("lora" in name.lower())
                        self.assertTrue(param.grad is not None)

    def test_peft_add_multi_adapter(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained`. This test tests if
        add_adapter works as expected in multi-adapter setting.
        """
        from peft import LoraConfig
        from peft.tuners.tuners_utils import BaseTunerLayer

        dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

        for model_id in self.transformers_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                is_peft_loaded = False
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                logits_original_model = model(dummy_input).logits

                peft_config = LoraConfig(init_lora_weights=False)

                model.add_adapter(peft_config)

                logits_adapter_1 = model(dummy_input)

                model.add_adapter(peft_config, adapter_name="adapter-2")

                logits_adapter_2 = model(dummy_input)

                for _, m in model.named_modules():
                    if isinstance(m, BaseTunerLayer):
                        is_peft_loaded = True
                        break

                self.assertTrue(is_peft_loaded)

                # dummy generation
                _ = model.generate(input_ids=dummy_input)

                model.set_adapter("default")
                self.assertTrue(model.active_adapters() == ["default"])
                self.assertTrue(model.active_adapter() == "default")

                model.set_adapter("adapter-2")
                self.assertTrue(model.active_adapters() == ["adapter-2"])
                self.assertTrue(model.active_adapter() == "adapter-2")

                # Logits comparison
                self.assertFalse(
                    torch.allclose(logits_adapter_1.logits, logits_adapter_2.logits, atol=1e-6, rtol=1e-6)
                )
                self.assertFalse(torch.allclose(logits_original_model, logits_adapter_2.logits, atol=1e-6, rtol=1e-6))

                model.set_adapter(["adapter-2", "default"])
                self.assertTrue(model.active_adapters() == ["adapter-2", "default"])
                self.assertTrue(model.active_adapter() == "adapter-2")

                logits_adapter_mixed = model(dummy_input)
                self.assertFalse(
                    torch.allclose(logits_adapter_1.logits, logits_adapter_mixed.logits, atol=1e-6, rtol=1e-6)
                )

                self.assertFalse(
                    torch.allclose(logits_adapter_2.logits, logits_adapter_mixed.logits, atol=1e-6, rtol=1e-6)
                )

                # multi active adapter saving not supported
                with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)

    @require_torch_gpu
    @require_bitsandbytes
    def test_peft_from_pretrained_kwargs(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained` + additional kwargs
        and see if the integraiton behaves as expected.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear8bitLt")
                self.assertTrue(peft_model.hf_device_map is not None)

                # dummy generation
                _ = peft_model.generate(input_ids=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device))

    @require_torch_gpu
    @require_bitsandbytes
    def test_peft_save_quantized(self):
        """
        Simple test that tests the basic usage of PEFT model save_pretrained with quantized base models
        """
        # 4bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear4bit")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)
                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

        # 8-bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear8bitLt")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)

                    self.assertTrue("adapter_model.safetensors" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

    @require_torch_gpu
    @require_bitsandbytes
    def test_peft_save_quantized_regression(self):
        """
        Simple test that tests the basic usage of PEFT model save_pretrained with quantized base models
        Regression test to make sure everything works as expected before the safetensors integration.
        """
        # 4bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear4bit")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname, safe_serialization=False)
                    self.assertTrue("adapter_model.bin" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

        # 8-bit
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

                module = peft_model.model.decoder.layers[0].self_attn.v_proj
                self.assertTrue(module.__class__.__name__ == "Linear8bitLt")
                self.assertTrue(peft_model.hf_device_map is not None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname, safe_serialization=False)

                    self.assertTrue("adapter_model.bin" in os.listdir(tmpdirname))
                    self.assertTrue("adapter_config.json" in os.listdir(tmpdirname))
                    self.assertTrue("pytorch_model.bin" not in os.listdir(tmpdirname))
                    self.assertTrue("model.safetensors" not in os.listdir(tmpdirname))

    def test_peft_pipeline(self):
        """
        Simple test that tests the basic usage of PEFT model + pipeline
        """
        from transformers import pipeline

        for model_id in self.peft_test_model_ids:
            pipe = pipeline("text-generation", model_id)
            _ = pipe("Hello")

    def test_peft_add_adapter_with_state_dict(self):
        """
        Simple test that tests the basic usage of PEFT model through `from_pretrained`. This test tests if
        add_adapter works as expected with a state_dict being passed.
        """
        from peft import LoraConfig

        dummy_input = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]]).to(torch_device)

        for model_id, peft_model_id in zip(self.transformers_test_model_ids, self.peft_test_model_ids):
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig(init_lora_weights=False)

                with self.assertRaises(ValueError):
                    model.load_adapter(peft_model_id=None)

                state_dict_path = hf_hub_download(peft_model_id, "adapter_model.bin")

                dummy_state_dict = torch.load(state_dict_path)

                model.load_adapter(adapter_state_dict=dummy_state_dict, peft_config=peft_config)
                with self.assertRaises(ValueError):
                    model.load_adapter(model.load_adapter(adapter_state_dict=dummy_state_dict, peft_config=None))
                self.assertTrue(self._check_lora_correctly_converted(model))

                # dummy generation
                _ = model.generate(input_ids=dummy_input)

    def test_peft_add_adapter_with_state_dict_low_cpu_mem_usage(self):
        """
        Check the usage of low_cpu_mem_usage, which is supported in PEFT >= 0.13.0
        """
        from peft import LoraConfig

        min_version_lcmu = "0.13.0"
        is_lcmu_supported = version.parse(importlib.metadata.version("peft")) >= version.parse(min_version_lcmu)

        for model_id, peft_model_id in zip(self.transformers_test_model_ids, self.peft_test_model_ids):
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig()
                state_dict_path = hf_hub_download(peft_model_id, "adapter_model.bin")
                dummy_state_dict = torch.load(state_dict_path)

                # this should always work
                model.load_adapter(
                    adapter_state_dict=dummy_state_dict, peft_config=peft_config, low_cpu_mem_usage=False
                )

                if is_lcmu_supported:
                    # if supported, this should not raise an error
                    model.load_adapter(
                        adapter_state_dict=dummy_state_dict,
                        adapter_name="other",
                        peft_config=peft_config,
                        low_cpu_mem_usage=True,
                    )
                    # after loading, no meta device should be remaining
                    self.assertFalse(any((p.device.type == "meta") for p in model.parameters()))
                else:
                    err_msg = r"The version of PEFT you are using does not support `low_cpu_mem_usage` yet"
                    with self.assertRaisesRegex(ValueError, err_msg):
                        model.load_adapter(
                            adapter_state_dict=dummy_state_dict,
                            adapter_name="other",
                            peft_config=peft_config,
                            low_cpu_mem_usage=True,
                        )

    def test_peft_from_pretrained_hub_kwargs(self):
        """
        Tests different combinations of PEFT model + from_pretrained + hub kwargs
        """
        peft_model_id = "peft-internal-testing/tiny-opt-lora-revision"

        # This should not work
        with self.assertRaises(OSError):
            _ = AutoModelForCausalLM.from_pretrained(peft_model_id)

        adapter_kwargs = {"revision": "test"}

        # This should work
        model = AutoModelForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

        model = OPTForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

        adapter_kwargs = {"revision": "main", "subfolder": "test_subfolder"}

        model = AutoModelForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

        model = OPTForCausalLM.from_pretrained(peft_model_id, adapter_kwargs=adapter_kwargs)
        self.assertTrue(self._check_lora_correctly_converted(model))

    def test_peft_from_pretrained_unexpected_keys_warning(self):
        """
        Test for warning when loading a PEFT checkpoint with unexpected keys.
        """
        from peft import LoraConfig

        logger = logging.get_logger("transformers.integrations.peft")

        for model_id, peft_model_id in zip(self.transformers_test_model_ids, self.peft_test_model_ids):
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig()
                state_dict_path = hf_hub_download(peft_model_id, "adapter_model.bin")
                dummy_state_dict = torch.load(state_dict_path)

                # add unexpected key
                dummy_state_dict["foobar"] = next(iter(dummy_state_dict.values()))

                with CaptureLogger(logger) as cl:
                    model.load_adapter(
                        adapter_state_dict=dummy_state_dict, peft_config=peft_config, low_cpu_mem_usage=False
                    )

                msg = "Loading adapter weights from state_dict led to unexpected keys not found in the model: foobar"
                self.assertIn(msg, cl.out)

    def test_peft_from_pretrained_missing_keys_warning(self):
        """
        Test for warning when loading a PEFT checkpoint with missing keys.
        """
        from peft import LoraConfig

        logger = logging.get_logger("transformers.integrations.peft")

        for model_id, peft_model_id in zip(self.transformers_test_model_ids, self.peft_test_model_ids):
            for transformers_class in self.transformers_test_model_classes:
                model = transformers_class.from_pretrained(model_id).to(torch_device)

                peft_config = LoraConfig()
                state_dict_path = hf_hub_download(peft_model_id, "adapter_model.bin")
                dummy_state_dict = torch.load(state_dict_path)

                # remove a key so that we have missing keys
                key = next(iter(dummy_state_dict.keys()))
                del dummy_state_dict[key]

                with CaptureLogger(logger) as cl:
                    model.load_adapter(
                        adapter_state_dict=dummy_state_dict,
                        peft_config=peft_config,
                        low_cpu_mem_usage=False,
                        adapter_name="other",
                    )

                # Here we need to adjust the key name a bit to account for PEFT-specific naming.
                # 1. Remove PEFT-specific prefix
                # If merged after dropping Python 3.8, we can use: key = key.removeprefix(peft_prefix)
                peft_prefix = "base_model.model."
                key = key[len(peft_prefix) :]
                # 2. Insert adapter name
                prefix, _, suffix = key.rpartition(".")
                key = f"{prefix}.other.{suffix}"

                msg = f"Loading adapter weights from state_dict led to missing keys in the model: {key}"
                self.assertIn(msg, cl.out)

    def test_peft_load_adapter_training_inference_mode_true(self):
        """
        By default, when loading an adapter, the whole model should be in eval mode and no parameter should have
        requires_grad=False.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)
                    model = transformers_class.from_pretrained(peft_model.config._name_or_path)
                    model.load_adapter(tmpdirname)
                    assert not any(p.requires_grad for p in model.parameters())
                    assert not any(m.training for m in model.modules())
                    del model

    def test_peft_load_adapter_training_inference_mode_false(self):
        """
        When passing is_trainable=True, the LoRA modules should be in training mode and their parameters should have
        requires_grad=True.
        """
        for model_id in self.peft_test_model_ids:
            for transformers_class in self.transformers_test_model_classes:
                peft_model = transformers_class.from_pretrained(model_id).to(torch_device)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    peft_model.save_pretrained(tmpdirname)
                    model = transformers_class.from_pretrained(peft_model.config._name_or_path)
                    model.load_adapter(tmpdirname, is_trainable=True)

                    for name, module in model.named_modules():
                        if len(list(module.children())):
                            # only check leaf modules
                            continue

                        if "lora_" in name:
                            assert module.training
                            assert all(p.requires_grad for p in module.parameters())
                        else:
                            assert not module.training
                            assert all(not p.requires_grad for p in module.parameters())

    def test_prefix_tuning_trainer_load_best_model_at_end_error(self):
        # Original issue: https://github.com/huggingface/peft/issues/2256
        # There is a potential error when using load_best_model_at_end=True with a prompt learning PEFT method. This is
        # because Trainer uses load_adapter under the hood but with some prompt learning methods, there is an
        # optimization on the saved model to remove parameters that are not required for inference, which in turn
        # requires a change to the model architecture. This is why load_adapter will fail in such cases and users should
        # instead set load_best_model_at_end=False and use PeftModel.from_pretrained. As this is not obvious, we now
        # intercept the error and add a helpful error message.
        # This test checks this error message. It also tests the "happy path" (i.e. no error) when using LoRA.
        from peft import LoraConfig, PrefixTuningConfig, TaskType, get_peft_model

        # create a small sequence classification dataset (binary classification)
        dataset = []
        for i, row in enumerate(os.__doc__.splitlines()):
            dataset.append({"text": row, "label": i % 2})
        ds_train = Dataset.from_list(dataset)
        ds_valid = ds_train
        datasets = DatasetDict(
            {
                "train": ds_train,
                "val": ds_valid,
            }
        )

        # tokenizer for peft-internal-testing/tiny-OPTForCausalLM-lora cannot be loaded, thus using
        # hf-internal-testing/tiny-random-OPTForCausalLM
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", model_type="opt")

        def tokenize_function(examples):
            return tokenizer(examples["text"], max_length=128, truncation=True, padding="max_length")

        tokenized_datasets = datasets.map(tokenize_function, batched=True)
        # lora works, prefix-tuning is expected to raise an error
        peft_configs = {
            "lora": LoraConfig(task_type=TaskType.SEQ_CLS),
            "prefix-tuning": PrefixTuningConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                prefix_projection=True,
                num_virtual_tokens=10,
            ),
        }

        for peft_type, peft_config in peft_configs.items():
            base_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
            base_model.config.pad_token_id = tokenizer.pad_token_id
            peft_model = get_peft_model(base_model, peft_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                training_args = TrainingArguments(
                    output_dir=tmpdirname,
                    num_train_epochs=3,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                )
                trainer = Trainer(
                    model=peft_model,
                    args=training_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["val"],
                )

                if peft_type == "lora":
                    # LoRA works with load_best_model_at_end
                    trainer.train()
                else:
                    # prefix tuning does not work, but at least users should get a helpful error message
                    msg = "When using prompt learning PEFT methods such as PREFIX_TUNING"
                    with self.assertRaisesRegex(RuntimeError, msg):
                        trainer.train()
