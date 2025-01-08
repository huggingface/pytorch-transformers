# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen2Audio model."""

import tempfile
import unittest
from io import BytesIO
from urllib.request import urlopen

import librosa

from transformers import (
    AutoProcessor,
    Qwen2AudioConfig,
    Qwen2AudioForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class Qwen2AudioModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        audio_token_index=0,
        seq_length=25,
        feat_seq_length=60,
        text_config={
            "model_type": "qwen2",
            "intermediate_size": 36,
            "initializer_range": 0.02,
            "hidden_size": 32,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "use_labels": True,
            "use_mrope": False,
            "vocab_size": 99,
        },
        is_training=True,
        audio_config={
            "model_type": "qwen2_audio_encoder",
            "d_model": 16,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 16,
            "encoder_layers": 2,
            "num_mel_bins": 80,
            "max_source_positions": 30,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.audio_token_index = audio_token_index
        self.text_config = text_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length

    def get_config(self):
        return Qwen2AudioConfig(
            text_config=self.text_config,
            audio_config=self.audio_config,
            ignore_index=self.ignore_index,
            audio_token_index=self.audio_token_index,
        )

    def prepare_config_and_inputs(self):
        input_features_values = floats_tensor(
            [
                self.batch_size,
                self.audio_config["num_mel_bins"],
                self.feat_seq_length,
            ]
        )
        config = self.get_config()
        feature_attention_mask = torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.long).to(torch_device)
        return config, input_features_values, feature_attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features_values, feature_attention_mask = config_and_inputs
        input_length = (input_features_values.shape[-1] - 1) // 2 + 1
        num_audio_tokens = (input_length - 2) // 2 + 1
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        attention_mask[:, :1] = 0
        # we are giving 3 audios let's make sure we pass in 3 audios tokens
        input_ids[:, 1 : 1 + num_audio_tokens] = config.audio_token_index
        inputs_dict = {
            "input_features": input_features_values,
            "feature_attention_mask": feature_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_qwen2audio_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = Qwen2AudioForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class Qwen2AudioForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen2AudioForConditionalGeneration`.
    """

    all_model_classes = (Qwen2AudioForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Qwen2AudioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen2AudioConfig, has_text_modality=False)

    @unittest.skip(reason="Compile not yet supported because in Qwen2Audio models")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in Qwen2Audio models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        # overwrite because Qwen2 is audio+text model (not vision+text)
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"
                vision_attn = "sdpa" if model.audio_tower._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)
                self.assertTrue(model.audio_tower.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.audio_tower.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


@require_torch
class Qwen2AudioForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test_single(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

        url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": url},
                    {"type": "text", "text": "What's that sound?"},
                ],
            }
        ]

        raw_audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(text=formatted_prompt, audios=[raw_audio], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=32)

        # fmt: off
        EXPECTED_INPUT_IDS = torch.tensor([[
            151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 14755, 220, 16, 25, 220, 151647,
            *[151646] * 101,
            151648, 198, 3838, 594, 429, 5112, 30, 151645, 198, 151644, 77091, 198,
        ]])
        # fmt: on
        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        EXPECTED_DECODED_TEXT = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|>"
            + "<|AUDIO|>" * 101
            + "<|audio_eos|>\nWhat's that sound?<|im_end|>\n<|im_start|>assistant\nIt is the sound of glass breaking.<|im_end|>"
        )

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=False),
            EXPECTED_DECODED_TEXT,
        )

        # test the error when incorrect number of audio tokens
        # fmt: off
        inputs["input_ids"] = torch.tensor([[
            151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 14755, 220, 16, 25, 220, 151647,
            *[151646] * 200,
            151648, 198, 3838, 594, 429, 5112, 30, 151645, 198, 151644, 77091, 198,
        ]])
        # fmt: on
        with self.assertRaisesRegex(
            ValueError, "Audio features and audio tokens do not match: tokens: 200, features 101"
        ):
            model.generate(**inputs, max_new_tokens=32)

    @slow
    def test_small_model_integration_test_batch(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

        conversation1 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                    },
                    {"type": "text", "text": "What's that sound?"},
                ],
            },
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav",
                    },
                    {"type": "text", "text": "What can you hear?"},
                ],
            },
        ]

        conversation2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac",
                    },
                    {"type": "text", "text": "What does the person say?"},
                ],
            },
        ]

        conversations = [conversation1, conversation2]

        text = [
            self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            for conversation in conversations
        ]

        audios = []
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    BytesIO(urlopen(ele["audio_url"]).read()),
                                    sr=self.processor.feature_extractor.sampling_rate,
                                )[0]
                            )

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=32)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nAudio 1: \nWhat's that sound?\nassistant\nIt is the sound of glass shattering.\nuser\nAudio 2: \nWhat can you hear?\nassistant\ncough and throat clearing.",
            "system\nYou are a helpful assistant.\nuser\nAudio 1: \nWhat does the person say?\nassistant\nThe original content of this audio is: 'Mister Quiller is the apostle of the middle classes and we are glad to welcome his gospel.'",
        ]
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_multiturn(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                    },
                    {"type": "text", "text": "What's that sound?"},
                ],
            },
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav",
                    },
                    {"type": "text", "text": "How about this one?"},
                ],
            },
        ]

        formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        audios = []
        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                BytesIO(urlopen(ele["audio_url"]).read()),
                                sr=self.processor.feature_extractor.sampling_rate,
                            )[0]
                        )

        inputs = self.processor(text=formatted_prompt, audios=audios, return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=32, top_k=1)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nAudio 1: \nWhat's that sound?\nassistant\nIt is the sound of glass shattering.\nuser\nAudio 2: \nHow about this one?\nassistant\nThroat clearing.",
        ]
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
