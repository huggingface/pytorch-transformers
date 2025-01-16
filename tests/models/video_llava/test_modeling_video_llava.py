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
"""Testing suite for the PyTorch VideoLlava model."""

import unittest

import numpy as np
import requests
from huggingface_hub import hf_hub_download

from transformers import (
    VideoLlavaConfig,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_bitsandbytes,
    require_torch,
    require_torch_gpu,
    run_test_using_subprocess,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class VideoLlavaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        video_token_index=1,
        projector_hidden_act="gelu",
        seq_length=3,
        num_frames=2,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config={
            "model_type": "llama",
            "seq_length": 13,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 2048,  # we need it high because videos are 8 frames
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 3,
        },
        is_training=True,
        vision_config={
            "model_type": "clip_vision_model",
            "batch_size": 12,
            "image_size": 8,
            "patch_size": 6,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.num_frames = num_frames
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 5
        self.num_channels = 3
        self.image_size = 224

        self.num_image_tokens = (vision_config["image_size"] // vision_config["patch_size"]) ** 2
        self.num_video_tokens = (self.num_image_tokens + 1) * self.num_frames
        self.seq_length = seq_length + self.num_image_tokens + self.num_video_tokens

    def get_config(self):
        return VideoLlavaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            video_token_index=self.video_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
            image_seq_length=self.num_image_tokens,
            video_seq_length=self.num_video_tokens,
        )

    def prepare_config_and_inputs(self):
        pixel_values_videos = floats_tensor(
            [
                self.batch_size,
                self.num_frames,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )

        pixel_values_images = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values_images, pixel_values_videos

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values_images, pixel_values_videos = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)

        input_ids[(input_ids == config.image_token_index) | (input_ids == config.video_token_index)] = (
            self.pad_token_id
        )
        input_ids[:, : self.num_image_tokens] = config.image_token_index
        input_ids[:, self.num_image_tokens : self.num_video_tokens + self.num_image_tokens] = config.video_token_index
        inputs_dict = {
            "pixel_values_videos": pixel_values_videos,
            "pixel_values_images": pixel_values_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class VideoLlavaForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `VideoLlavaForConditionalGeneration`.
    """

    all_model_classes = (VideoLlavaForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (VideoLlavaForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = VideoLlavaVisionText2TextModelTester(self)
        common_properties = ["image_token_index", "video_token_index", "vision_feature_layer", "image_seq_length"]
        self.config_tester = ConfigTester(
            self, config_class=VideoLlavaConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Pass because video-LLava requires `attention_mask is not None`")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Pass because video-LLava requires `attention_mask is not None`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @run_test_using_subprocess
    def test_mixed_input(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            # test that the forward does not fail
            with torch.no_grad():
                _ = model(**inputs)

            # if we remove some images from inputs leaving only one
            # image number mismatch error should raise
            inputs["pixel_values_images"] = inputs["pixel_values_images"][:1]
            with self.assertRaises(ValueError):
                _ = model(**inputs)

    def test_video_only_input(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            # replace image token id with dummy id
            # Error will be raised as num-image-tokens and num-of-image-embeds mismatch
            inputs["input_ids"][:, : self.model_tester.num_image_tokens] = 2
            with self.assertRaises(ValueError):
                _ = model(**inputs)

            inputs["pixel_values_images"] = None
            _ = model(**inputs)

    def test_image_only_input(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            # set dummy id, which is not video token id
            # Error will be raised as num-video-tokens and num-of-video-embeds mismatch
            inputs["input_ids"][
                :,
                self.model_tester.num_image_tokens : self.model_tester.num_image_tokens
                + self.model_tester.num_video_tokens,
            ] = 2
            with self.assertRaises(ValueError):
                _ = model(**inputs)

            inputs["pixel_values_videos"] = None
            _ = model(**inputs)

    def test_batching_equivalence(self):
        def recursive_check(batched_object, single_row_object, model_name, key):
            if isinstance(batched_object, (list, tuple)):
                for batched_object_value, single_row_object_value in zip(batched_object, single_row_object):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            # do not compare returned loss (0-dim tensor) / codebook ids (int) / caching objects
            elif batched_object is None or not isinstance(batched_object, torch.Tensor):
                return
            elif batched_object.dim() == 0:
                return
            else:
                batched_row = batched_object[:1]
                self.assertFalse(
                    torch.isnan(batched_row).any(), f"Batched output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(batched_row).any(), f"Batched output has `inf` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isnan(single_row_object).any(), f"Single row output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(single_row_object).any(), f"Single row output has `inf` in {model_name} for key={key}"
                )
                self.assertTrue(
                    (torch.max(torch.abs(batched_row - single_row_object))) <= 1e-03,
                    msg=(
                        f"Batched and Single row outputs are not equal in {model_name} for key={key}. "
                        f"Difference={torch.max(torch.abs(batched_row - single_row_object))}."
                    ),
                )

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(config).to(torch_device).eval()

            single_row_input = {}
            for key, value in batched_input_prepared.items():
                single_row_input[key] = value[:1]

            with torch.no_grad():
                model_batched_output = model(**batched_input_prepared)
                model_row_output = model(**single_row_input)

            for key in model_batched_output:
                # we can't test videos as their output shapes are linked to number of frames
                # and we don't have to as it is a CLIP model and can be tested from `ClipModelTester` class
                if key == "video_hidden_states":
                    continue
                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values_images"]
            del inputs["pixel_values_videos"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    # while some other models require pixel_values to be present
    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values_images"]
            del inputs["pixel_values_videos"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            _ = model(**input_dict)  # successfull forward with no modifications

            # remove one image but leave the image token in text
            input_dict["pixel_values_images"] = input_dict["pixel_values_images"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = input_dict["input_ids"][:1]
            pixel_values = input_dict["pixel_values_images"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values_images=pixel_values)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            _ = model(input_ids=input_ids, pixel_values_images=pixel_values)


@require_torch
class VideoLlavaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)

        prompt = "USER: <video>\nWhy is this video funny? ASSISTANT:"
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        video_file = np.load(video_file)
        inputs = self.processor(prompt, videos=video_file, return_tensors="pt").to(torch_device)

        EXPECTED_INPUT_IDS = torch.tensor([1,  3148, 1001, 29901, 29871, 13, 11008, 338, 445, 4863, 2090, 1460, 29973, 319, 1799, 9047, 13566, 29901], device=torch_device)  # fmt: skip
        non_video_inputs = inputs["input_ids"][inputs["input_ids"] != 32001]
        self.assertTrue(torch.equal(non_video_inputs, EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        EXPECTED_DECODED_TEXT = "USER: \nWhy is this video funny? ASSISTANT: The video is funny because it shows a baby sitting on a bed and reading a book, which"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_mixed_inputs(self):
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)

        prompts = [
            "USER: <image>\nWhat are the cats in the image doing? ASSISTANT:",
            "USER: <video>\nWhy is this video funny? ASSISTANT:",
        ]
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        video_file = np.load(video_file)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = self.processor(prompts, images=[image], videos=[video_file], padding=True, return_tensors="pt").to(
            torch_device
        )
        output = model.generate(**inputs, do_sample=False, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = [
            'USER: \nWhat are the cats in the image doing? ASSISTANT: The cats in the image are sleeping or resting on a couch.',
            'USER: \nWhy is this video funny? ASSISTANT: The video is funny because it shows a baby sitting on a bed and reading a book. The'
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama(self):
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)
        processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

        prompt = "USER: <video>\nDescribe the video in details. ASSISTANT:"
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        video_file = np.load(video_file)
        inputs = self.processor(prompt, videos=video_file, return_tensors="pt").to(torch_device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        EXPECTED_DECODED_TEXT = "USER: \nDescribe the video in details. ASSISTANT: The video features a young child sitting on a bed, holding a book and reading it. " \
            "The child appears to be enjoying the book, as they are fully engaged in the activity. The bed is located in a bedroom, and there is a chair nearby. The " \
            "child is wearing a blue shirt and glasses, which suggests that they might have a visual impairment. The room is well-lit, and there is a clock on the wall, " \
            "indicating the time. The child's focus on the book indicates that they are interested in the content and are actively participating in the reading process. " \
            "Overall, the video captures a heartwarming moment of a child engaging in a simple yet essential activity, which is reading."  # fmt: skip

        self.assertEqual(
            processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched(self):
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)
        processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
        processor.tokenizer.padding_side = "left"

        prompts = [
            "USER: <video>\nWhat is the baby doing? ASSISTANT:",
            "USER: <video>\nWho is sitting next to the woman? ASSISTANT:",
        ]
        video_1 = np.load(
            hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset")
        )
        video_2 = np.load(
            hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="video_demo_2.npy", repo_type="dataset")
        )

        inputs = processor(prompts, videos=[video_1, video_2], return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = [
            'USER: \nWhat is the baby doing? ASSISTANT: The baby is sitting on a bed and reading a book.',
            'USER: \nWho is sitting next to the woman? ASSISTANT: A small dog is sitting next to the woman.'
        ]  # fmt: skip

        self.assertEqual(processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_video_llava_index_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28032 and makes sure it does not happen anymore
        # Please refer to that PR, or specifically https://github.com/huggingface/transformers/pull/28032#issuecomment-1860650043 for
        # more details
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)

        # Simulate a super long prompt
        user_prompt = "Describe the video:?\n" * 200
        prompt = f"USER: <video>{user_prompt}ASSISTANT:"
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        video_file = np.load(video_file)

        # let's expand it for 16 frames, to check model can handle any number of frames
        video_file = video_file.repeat(2, 0)
        inputs = self.processor(prompt, videos=video_file, return_tensors="pt").to(torch_device, torch.float16)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)

    @slow
    @require_torch_gpu
    def test_video_llava_merge_inputs_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28333 and makes sure it does not happen anymore
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)

        prompt = "USER: <video>\nDescribe the video:? ASSISTANT:"
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        video_file = np.load(video_file)
        inputs = self.processor(prompt, videos=video_file, return_tensors="pt").to(torch_device, torch.float16)

        # Make sure that the loss is properly computed
        loss = model(
            **inputs,
            labels=inputs.input_ids.clone(),
        ).loss
        loss.backward()
