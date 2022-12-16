# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


import json
import os
import tempfile
import unittest

import numpy as np
from datasets import load_dataset

from huggingface_hub import hf_hub_download
from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import prepare_image_inputs


if is_torch_available():
    import torch

    if is_vision_available():
        from transformers import CLIPTokenizer, OneFormerImageProcessor, OneFormerProcessor
        from transformers.models.oneformer.image_processing_oneformer import binary_mask_to_rle
        from transformers.models.oneformer.modeling_oneformer import OneFormerForUniversalSegmentationOutput

if is_vision_available():
    from PIL import Image


def prepare_metadata(class_info_file):
    class_info = json.load(open(hf_hub_download("shi-labs/oneformer_demo", class_info_file, repo_type="dataset"), "r"))
    metadata = {}
    class_names = []
    thing_ids = []
    for key, info in class_info.items():
        metadata[key] = info["name"]
        class_names.append(info["name"])
        if info["isthing"]:
            thing_ids.append(int(key))
    metadata["thing_ids"] = thing_ids
    metadata["class_names"] = class_names
    return metadata


class OneFormerProcessorTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        size=None,
        do_resize=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        num_labels=10,
        reduce_labels=False,
        ignore_index=255,
        max_seq_length=77,
        task_seq_length=77,
        repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        class_info_file="ade20k_panoptic.json",
        num_text=10,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = {"shortest_edge": 32, "longest_edge": 1333} if size is None else size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisor = 0
        self.max_seq_length = max_seq_length
        self.task_seq_length = task_seq_length
        self.class_info_file = class_info_file
        self.metadata = prepare_metadata(class_info_file)
        self.num_text = num_text
        self.repo_path = repo_path

        # for the post_process_functions
        self.batch_size = 2
        self.num_queries = 10
        self.num_classes = 10
        self.height = 3
        self.width = 4
        self.num_labels = num_labels
        self.reduce_labels = reduce_labels
        self.ignore_index = ignore_index

    def prepare_processor_dict(self):
        image_processor_dict = {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "size_divisor": self.size_divisor,
            "num_labels": self.num_labels,
            "reduce_labels": self.reduce_labels,
            "ignore_index": self.ignore_index,
            "class_info_file": self.class_info_file,
            "metadata": self.metadata,
            "num_text": self.num_text,
        }

        image_processor = OneFormerImageProcessor(**image_processor_dict)
        tokenizer = CLIPTokenizer.from_pretrained(self.repo_path)

        return {
            "image_processor": image_processor,
            "tokenizer": tokenizer,
            "max_seq_length": self.max_seq_length,
            "task_seq_length": self.task_seq_length,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to OneFormerImageProcessor,
        assuming do_resize is set to True with a scalar size.
        """
        if not batched:
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                h, w = image.shape[1], image.shape[2]
            if w < h:
                expected_height = int(self.size["shortest_edge"] * h / w)
                expected_width = self.size["shortest_edge"]
            elif w > h:
                expected_height = self.size["shortest_edge"]
                expected_width = int(self.size["shortest_edge"] * w / h)
            else:
                expected_height = self.size["shortest_edge"]
                expected_width = self.size["shortest_edge"]

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width, expected_sequence_length = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width, expected_sequence_length))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        expected_sequence_length = self.max_seq_length

        return expected_height, expected_width, expected_sequence_length

    def get_fake_oneformer_outputs(self):
        return OneFormerForUniversalSegmentationOutput(
            # +1 for null class
            class_queries_logits=torch.randn((self.batch_size, self.num_queries, self.num_classes + 1)),
            masks_queries_logits=torch.randn((self.batch_size, self.num_queries, self.height, self.width)),
        )


@require_torch
@require_vision
class OneFormerProcessingTest(unittest.TestCase):
    processing_class = OneFormerProcessor if (is_vision_available() and is_torch_available()) else None
    # only for test_feat_extracttion_common.test_feat_extract_to_json_string
    feature_extraction_class = processing_class

    def setUp(self):
        self.processing_tester = OneFormerProcessorTester(self)

    @property
    def processor_dict(self):
        return self.processing_tester.prepare_processor_dict()

    def test_feat_extract_properties(self):
        processor = self.processing_class(**self.processor_dict)
        self.assertTrue(hasattr(processor, "image_processor"))
        self.assertTrue(hasattr(processor, "tokenizer"))
        self.assertTrue(hasattr(processor, "max_seq_length"))
        self.assertTrue(hasattr(processor, "task_seq_length"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize processor
        processor = self.processing_class(**self.processor_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(self.processing_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = processor(image_inputs[0], ["semantic"], return_tensors="pt").pixel_values

        expected_height, expected_width, expected_sequence_length = self.processing_tester.get_expected_values(
            image_inputs
        )

        self.assertEqual(
            encoded_images.shape,
            (1, self.processing_tester.num_channels, expected_height, expected_width),
        )

        tokenized_task_inputs = processor(image_inputs[0], ["semantic"], return_tensors="pt").task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (1, expected_sequence_length),
        )

        # Test batched
        expected_height, expected_width, expected_sequence_length = self.processing_tester.get_expected_values(
            image_inputs, batched=True
        )

        encoded_images = processor(image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.processing_tester.batch_size,
                self.processing_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        tokenized_task_inputs = processor(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        ).task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (self.processing_tester.batch_size, expected_sequence_length),
        )

    def test_call_numpy(self):
        # Initialize processor
        processor = self.processing_class(**self.processor_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.processing_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = processor(image_inputs[0], ["semantic"], return_tensors="pt").pixel_values

        expected_height, expected_width, expected_sequence_length = self.processing_tester.get_expected_values(
            image_inputs
        )

        self.assertEqual(
            encoded_images.shape,
            (1, self.processing_tester.num_channels, expected_height, expected_width),
        )

        tokenized_task_inputs = processor(image_inputs[0], ["semantic"], return_tensors="pt").task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (1, expected_sequence_length),
        )

        # Test batched
        expected_height, expected_width, expected_sequence_length = self.processing_tester.get_expected_values(
            image_inputs, batched=True
        )

        encoded_images = processor(image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.processing_tester.batch_size,
                self.processing_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        tokenized_task_inputs = processor(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        ).task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (self.processing_tester.batch_size, expected_sequence_length),
        )

    def test_call_pytorch(self):
        # Initialize processor
        processor = self.processing_class(**self.processor_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.processing_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = processor(image_inputs[0], ["semantic"], return_tensors="pt").pixel_values

        expected_height, expected_width, expected_sequence_length = self.processing_tester.get_expected_values(
            image_inputs
        )

        self.assertEqual(
            encoded_images.shape,
            (1, self.processing_tester.num_channels, expected_height, expected_width),
        )

        tokenized_task_inputs = processor(image_inputs[0], ["semantic"], return_tensors="pt").task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (1, expected_sequence_length),
        )

        # Test batched
        expected_height, expected_width, expected_sequence_length = self.processing_tester.get_expected_values(
            image_inputs, batched=True
        )

        encoded_images = processor(image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.processing_tester.batch_size,
                self.processing_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

        tokenized_task_inputs = processor(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        ).task_inputs

        self.assertEqual(
            tokenized_task_inputs.shape,
            (self.processing_tester.batch_size, expected_sequence_length),
        )

    def test_equivalence_pad_and_create_pixel_mask(self):
        # Initialize processors
        processor_1 = self.processing_class(**self.processor_dict)

        image_processor = OneFormerImageProcessor(
            do_resize=False,
            do_normalize=False,
            do_rescale=False,
            num_labels=self.processing_tester.num_classes,
            class_info_file="ade20k_panoptic.json",
            num_text=self.processing_tester.num_text,
        )
        tokenizer = CLIPTokenizer.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        processor_2 = self.processing_class(
            image_processor=image_processor, tokenizer=tokenizer, max_seq_length=77, task_seq_length=77
        )

        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.processing_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test whether the method "pad_and_return_pixel_mask" and calling the image processor return the same tensors
        encoded_images_with_method = processor_1.encode_inputs(
            image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt"
        )
        encoded_images = processor_2(image_inputs, ["semantic"] * len(image_inputs), return_tensors="pt")

        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_values"], encoded_images["pixel_values"], atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(encoded_images_with_method["pixel_mask"], encoded_images["pixel_mask"], atol=1e-4)
        )

    def comm_get_processor_inputs(self, with_segmentation_maps=False, is_instance_map=False, segmentation_type="np"):
        processor = self.processing_class(**self.processor_dict)
        # prepare image and target
        num_labels = self.processing_tester.num_labels
        annotations = None
        instance_id_to_semantic_id = None
        image_inputs = prepare_image_inputs(self.processing_tester, equal_resolution=False)
        if with_segmentation_maps:
            high = num_labels
            if is_instance_map:
                labels_expanded = list(range(num_labels)) * 2
                instance_id_to_semantic_id = {
                    instance_id: label_id for instance_id, label_id in enumerate(labels_expanded)
                }
            annotations = [
                np.random.randint(0, high * 2, (img.size[1], img.size[0])).astype(np.uint8) for img in image_inputs
            ]
            if segmentation_type == "pil":
                annotations = [Image.fromarray(annotation) for annotation in annotations]

        inputs = processor(
            image_inputs,
            ["semantic"] * len(image_inputs),
            annotations,
            return_tensors="pt",
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            pad_and_return_pixel_mask=True,
        )

        return inputs

    def test_init_without_params(self):
        pass

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            feat_extract_first.save_pretrained(tmpdirname)
            check_json_file_has_correct_format(os.path.join(tmpdirname, "preprocessor_config.json"))
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        self.assertEqual(feat_extract_second.image_processor.to_dict(), feat_extract_first.image_processor.to_dict())
        self.assertIsInstance(feat_extract_first.image_processor, OneFormerImageProcessor)
        self.assertIsInstance(feat_extract_first.tokenizer, CLIPTokenizer)

    def test_with_size_divisor(self):
        size_divisors = [8, 16, 32]
        weird_input_sizes = [(407, 802), (582, 1094)]
        image_processor_dict = {
            "do_resize": self.processing_tester.do_resize,
            "size": self.processing_tester.size,
            "do_normalize": self.processing_tester.do_normalize,
            "image_mean": self.processing_tester.image_mean,
            "image_std": self.processing_tester.image_std,
            "size_divisor": self.processing_tester.size_divisor,
            "num_labels": self.processing_tester.num_labels,
            "reduce_labels": self.processing_tester.reduce_labels,
            "ignore_index": self.processing_tester.ignore_index,
            "class_info_file": self.processing_tester.class_info_file,
            "metadata": self.processing_tester.metadata,
            "num_text": self.processing_tester.num_text,
        }
        tokenizer = CLIPTokenizer.from_pretrained(self.processing_tester.repo_path)

        for size_divisor in size_divisors:
            image_process_dict = {**image_processor_dict, **{"size_divisor": size_divisor}}
            image_processor = OneFormerImageProcessor(**image_process_dict)
            processor = self.processing_class(
                image_processor=image_processor,
                tokenizer=tokenizer,
                max_seq_length=self.processing_tester.max_seq_length,
                task_seq_length=self.processing_tester.task_seq_length,
            )
            for weird_input_size in weird_input_sizes:
                inputs = processor([np.ones((3, *weird_input_size))], ["semantic"], return_tensors="pt")
                pixel_values = inputs["pixel_values"]
                # check if divisible
                self.assertTrue((pixel_values.shape[-1] % size_divisor) == 0)
                self.assertTrue((pixel_values.shape[-2] % size_divisor) == 0)

    def test_call_with_segmentation_maps(self):
        def common(is_instance_map=False, segmentation_type=None):
            inputs = self.comm_get_processor_inputs(
                with_segmentation_maps=True, is_instance_map=is_instance_map, segmentation_type=segmentation_type
            )

            mask_labels = inputs["mask_labels"]
            class_labels = inputs["class_labels"]
            pixel_values = inputs["pixel_values"]
            text_inputs = inputs["text_inputs"]

            # check the batch_size
            for mask_label, class_label, text_input in zip(mask_labels, class_labels, text_inputs):
                self.assertEqual(mask_label.shape[0], class_label.shape[0])
                # this ensure padding has happened
                self.assertEqual(mask_label.shape[1:], pixel_values.shape[2:])
                self.assertEqual(text_inputs.shape[1], self.processing_tester.num_text)

        common()
        common(is_instance_map=True)
        common(is_instance_map=False, segmentation_type="pil")
        common(is_instance_map=True, segmentation_type="pil")

    def test_integration_semantic_segmentation(self):
        # load 2 images and corresponding panoptic annotations from the hub
        dataset = load_dataset("nielsr/ade20k-panoptic-demo")
        image1 = dataset["train"][0]["image"]
        image2 = dataset["train"][1]["image"]
        segments_info1 = dataset["train"][0]["segments_info"]
        segments_info2 = dataset["train"][1]["segments_info"]
        annotation1 = dataset["train"][0]["label"]
        annotation2 = dataset["train"][1]["label"]

        def rgb_to_id(color):
            if isinstance(color, np.ndarray) and len(color.shape) == 3:
                if color.dtype == np.uint8:
                    color = color.astype(np.int32)
                return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
            return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

        def create_panoptic_map(annotation, segments_info):
            annotation = np.array(annotation)
            # convert RGB to segment IDs per pixel
            # 0 is the "ignore" label, for which we don't need to make binary masks
            panoptic_map = rgb_to_id(annotation)

            # create mapping between segment IDs and semantic classes
            inst2class = {segment["id"]: segment["category_id"] for segment in segments_info}

            return panoptic_map, inst2class

        panoptic_map1, inst2class1 = create_panoptic_map(annotation1, segments_info1)
        panoptic_map2, inst2class2 = create_panoptic_map(annotation2, segments_info2)

        image_processor = OneFormerImageProcessor(
            reduce_labels=True,
            ignore_index=0,
            size=(512, 512),
            class_info_file="ade20k_panoptic.json",
            num_text=self.processing_tester.num_text,
        )

        tokenizer = CLIPTokenizer.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

        processor = OneFormerProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_seq_length=77,
            task_seq_length=77,
        )

        # prepare the images and annotations
        pixel_values_list = [np.moveaxis(np.array(image1), -1, 0), np.moveaxis(np.array(image2), -1, 0)]
        inputs = processor.encode_inputs(
            pixel_values_list,
            ["semantic", "semantic"],
            [panoptic_map1, panoptic_map2],
            instance_id_to_semantic_id=[inst2class1, inst2class2],
            return_tensors="pt",
        )

        # verify the pixel values, task inputs, text inputs and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 711))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 711))
        self.assertEqual(inputs["task_inputs"].shape, (2, 77))
        self.assertEqual(inputs["text_inputs"].shape, (2, self.processing_tester.num_text, 77))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"]), 2)
        # fmt: off
        expected_class_labels = torch.tensor([4, 17, 32, 42, 12, 3, 5, 0, 43, 96, 104, 31, 125, 138, 87, 149])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][0], expected_class_labels))
        # fmt: off
        expected_class_labels = torch.tensor([19, 67, 82, 17, 12, 42, 3, 14, 5, 0, 115, 43, 8, 138, 125, 143])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][1], expected_class_labels))

        # verify the task inputs
        self.assertEqual(len(inputs["task_inputs"]), 2)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), 141082)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), inputs["task_inputs"][1].sum().item())

        # verify the text inputs
        self.assertEqual(len(inputs["text_inputs"]), 2)
        self.assertEqual(inputs["text_inputs"][0].sum().item(), 1095752)
        self.assertEqual(inputs["text_inputs"][1].sum().item(), 1062468)

        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"]), 2)
        self.assertEqual(inputs["mask_labels"][0].shape, (16, 512, 711))
        self.assertEqual(inputs["mask_labels"][1].shape, (16, 512, 711))
        self.assertEqual(inputs["mask_labels"][0].sum().item(), 315193.0)
        self.assertEqual(inputs["mask_labels"][1].sum().item(), 350747.0)

    def test_integration_instance_segmentation(self):
        # load 2 images and corresponding panoptic annotations from the hub
        dataset = load_dataset("nielsr/ade20k-panoptic-demo")
        image1 = dataset["train"][0]["image"]
        image2 = dataset["train"][1]["image"]
        segments_info1 = dataset["train"][0]["segments_info"]
        segments_info2 = dataset["train"][1]["segments_info"]
        annotation1 = dataset["train"][0]["label"]
        annotation2 = dataset["train"][1]["label"]

        def rgb_to_id(color):
            if isinstance(color, np.ndarray) and len(color.shape) == 3:
                if color.dtype == np.uint8:
                    color = color.astype(np.int32)
                return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
            return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

        def create_panoptic_map(annotation, segments_info):
            annotation = np.array(annotation)
            # convert RGB to segment IDs per pixel
            # 0 is the "ignore" label, for which we don't need to make binary masks
            panoptic_map = rgb_to_id(annotation)

            # create mapping between segment IDs and semantic classes
            inst2class = {segment["id"]: segment["category_id"] for segment in segments_info}

            return panoptic_map, inst2class

        panoptic_map1, inst2class1 = create_panoptic_map(annotation1, segments_info1)
        panoptic_map2, inst2class2 = create_panoptic_map(annotation2, segments_info2)

        image_processor = OneFormerImageProcessor(
            reduce_labels=True,
            ignore_index=0,
            size=(512, 512),
            class_info_file="ade20k_panoptic.json",
            num_text=self.processing_tester.num_text,
        )

        tokenizer = CLIPTokenizer.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

        processor = OneFormerProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_seq_length=77,
            task_seq_length=77,
        )

        # prepare the images and annotations
        pixel_values_list = [np.moveaxis(np.array(image1), -1, 0), np.moveaxis(np.array(image2), -1, 0)]
        inputs = processor.encode_inputs(
            pixel_values_list,
            ["instance", "instance"],
            [panoptic_map1, panoptic_map2],
            instance_id_to_semantic_id=[inst2class1, inst2class2],
            return_tensors="pt",
        )

        # verify the pixel values, task inputs, text inputs and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 711))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 711))
        self.assertEqual(inputs["task_inputs"].shape, (2, 77))
        self.assertEqual(inputs["text_inputs"].shape, (2, self.processing_tester.num_text, 77))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"]), 2)
        # fmt: off
        expected_class_labels = torch.tensor([32, 42, 42, 42, 42, 42, 42, 42, 32, 12, 12, 12, 12, 12, 42, 42, 12, 12, 12, 42, 12, 12, 12, 12, 12, 12, 12, 12, 12, 42, 42, 42, 12, 42, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 43, 43, 43, 43, 104, 43, 31, 125, 31, 125, 138, 87, 125, 149, 138, 125, 87, 87])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][0], expected_class_labels))
        # fmt: off
        expected_class_labels = torch.tensor([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 67, 82, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 12, 12, 42, 12, 12, 12, 12, 14, 12, 12, 12, 12, 12, 12, 12, 12, 14, 12, 12, 115, 43, 43, 115, 43, 43, 43, 8, 8, 8, 138, 138, 125, 143])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][1], expected_class_labels))

        # verify the task inputs
        self.assertEqual(len(inputs["task_inputs"]), 2)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), 144985)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), inputs["task_inputs"][1].sum().item())

        # verify the text inputs
        self.assertEqual(len(inputs["text_inputs"]), 2)
        self.assertEqual(inputs["text_inputs"][0].sum().item(), 1037040)
        self.assertEqual(inputs["text_inputs"][1].sum().item(), 1044078)

        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"]), 2)
        self.assertEqual(inputs["mask_labels"][0].shape, (73, 512, 711))
        self.assertEqual(inputs["mask_labels"][1].shape, (57, 512, 711))
        self.assertEqual(inputs["mask_labels"][0].sum().item(), 35040.0)
        self.assertEqual(inputs["mask_labels"][1].sum().item(), 98228.0)

    def test_integration_panoptic_segmentation(self):
        # load 2 images and corresponding panoptic annotations from the hub
        dataset = load_dataset("nielsr/ade20k-panoptic-demo")
        image1 = dataset["train"][0]["image"]
        image2 = dataset["train"][1]["image"]
        segments_info1 = dataset["train"][0]["segments_info"]
        segments_info2 = dataset["train"][1]["segments_info"]
        annotation1 = dataset["train"][0]["label"]
        annotation2 = dataset["train"][1]["label"]

        def rgb_to_id(color):
            if isinstance(color, np.ndarray) and len(color.shape) == 3:
                if color.dtype == np.uint8:
                    color = color.astype(np.int32)
                return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
            return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

        def create_panoptic_map(annotation, segments_info):
            annotation = np.array(annotation)
            # convert RGB to segment IDs per pixel
            # 0 is the "ignore" label, for which we don't need to make binary masks
            panoptic_map = rgb_to_id(annotation)

            # create mapping between segment IDs and semantic classes
            inst2class = {segment["id"]: segment["category_id"] for segment in segments_info}

            return panoptic_map, inst2class

        panoptic_map1, inst2class1 = create_panoptic_map(annotation1, segments_info1)
        panoptic_map2, inst2class2 = create_panoptic_map(annotation2, segments_info2)

        image_processor = OneFormerImageProcessor(
            reduce_labels=True,
            ignore_index=0,
            size=(512, 512),
            class_info_file="ade20k_panoptic.json",
            num_text=self.processing_tester.num_text,
        )

        tokenizer = CLIPTokenizer.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

        processor = OneFormerProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_seq_length=77,
            task_seq_length=77,
        )

        # prepare the images and annotations
        pixel_values_list = [np.moveaxis(np.array(image1), -1, 0), np.moveaxis(np.array(image2), -1, 0)]
        inputs = processor.encode_inputs(
            pixel_values_list,
            ["panoptic", "panoptic"],
            [panoptic_map1, panoptic_map2],
            instance_id_to_semantic_id=[inst2class1, inst2class2],
            return_tensors="pt",
        )

        # verify the pixel values, task inputs, text inputs and pixel mask
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 512, 711))
        self.assertEqual(inputs["pixel_mask"].shape, (2, 512, 711))
        self.assertEqual(inputs["task_inputs"].shape, (2, 77))
        self.assertEqual(inputs["text_inputs"].shape, (2, self.processing_tester.num_text, 77))

        # verify the class labels
        self.assertEqual(len(inputs["class_labels"]), 2)
        # fmt: off
        expected_class_labels = torch.tensor([4, 17, 32, 42, 42, 42, 42, 42, 42, 42, 32, 12, 12, 12, 12, 12, 42, 42, 12, 12, 12, 42, 12, 12, 12, 12, 12, 3, 12, 12, 12, 12, 42, 42, 42, 12, 42, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 5, 12, 12, 12, 12, 12, 12, 12, 0, 43, 43, 43, 96, 43, 104, 43, 31, 125, 31, 125, 138, 87, 125, 149, 138, 125, 87, 87])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][0], expected_class_labels))
        # fmt: off
        expected_class_labels = torch.tensor([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 67, 82, 19, 19, 17, 19, 19, 19, 19, 19, 19, 19, 19, 19, 12, 12, 42, 12, 12, 12, 12, 3, 14, 12, 12, 12, 12, 12, 12, 12, 12, 14, 5, 12, 12, 0, 115, 43, 43, 115, 43, 43, 43, 8, 8, 8, 138, 138, 125, 143])  # noqa: E231
        # fmt: on
        self.assertTrue(torch.allclose(inputs["class_labels"][1], expected_class_labels))

        # verify the task inputs
        self.assertEqual(len(inputs["task_inputs"]), 2)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), 136240)
        self.assertEqual(inputs["task_inputs"][0].sum().item(), inputs["task_inputs"][1].sum().item())

        # verify the text inputs
        self.assertEqual(len(inputs["text_inputs"]), 2)
        self.assertEqual(inputs["text_inputs"][0].sum().item(), 1048653)
        self.assertEqual(inputs["text_inputs"][1].sum().item(), 1067160)

        # verify the mask labels
        self.assertEqual(len(inputs["mask_labels"]), 2)
        self.assertEqual(inputs["mask_labels"][0].shape, (79, 512, 711))
        self.assertEqual(inputs["mask_labels"][1].shape, (61, 512, 711))
        self.assertEqual(inputs["mask_labels"][0].sum().item(), 315193.0)
        self.assertEqual(inputs["mask_labels"][1].sum().item(), 350747.0)

    def test_binary_mask_to_rle(self):
        fake_binary_mask = np.zeros((20, 50))
        fake_binary_mask[0, 20:] = 1
        fake_binary_mask[1, :15] = 1
        fake_binary_mask[5, :10] = 1

        rle = binary_mask_to_rle(fake_binary_mask)
        self.assertEqual(len(rle), 4)
        self.assertEqual(rle[0], 21)
        self.assertEqual(rle[1], 45)

    def test_post_process_semantic_segmentation(self):
        image_processor = OneFormerImageProcessor(
            reduce_labels=True,
            ignore_index=0,
            size=(512, 512),
            class_info_file="ade20k_panoptic.json",
            num_text=self.processing_tester.num_text,
        )
        tokenizer = CLIPTokenizer.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        processor = OneFormerProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_seq_length=77,
            task_seq_length=77,
        )

        outputs = self.processing_tester.get_fake_oneformer_outputs()

        segmentation = processor.post_process_semantic_segmentation(outputs)

        self.assertEqual(len(segmentation), self.processing_tester.batch_size)
        self.assertEqual(
            segmentation[0].shape,
            (
                self.processing_tester.height,
                self.processing_tester.width,
            ),
        )

        target_sizes = [(1, 4) for i in range(self.processing_tester.batch_size)]
        segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        self.assertEqual(segmentation[0].shape, target_sizes[0])

    def test_post_process_instance_segmentation(self):
        image_processor = OneFormerImageProcessor(
            reduce_labels=True,
            ignore_index=0,
            size=(512, 512),
            class_info_file="ade20k_panoptic.json",
            num_text=self.processing_tester.num_text,
        )
        tokenizer = CLIPTokenizer.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        processor = OneFormerProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_seq_length=77,
            task_seq_length=77,
        )

        outputs = self.processing_tester.get_fake_oneformer_outputs()
        segmentation = processor.post_process_instance_segmentation(outputs, threshold=0)

        self.assertTrue(len(segmentation) == self.processing_tester.batch_size)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(el["segmentation"].shape, (self.processing_tester.height, self.processing_tester.width))

    def test_post_process_panoptic_segmentation(self):
        image_processor = OneFormerImageProcessor(
            reduce_labels=True,
            ignore_index=0,
            size=(512, 512),
            class_info_file="ade20k_panoptic.json",
            num_text=self.processing_tester.num_text,
        )
        tokenizer = CLIPTokenizer.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        processor = OneFormerProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_seq_length=77,
            task_seq_length=77,
        )

        outputs = self.processing_tester.get_fake_oneformer_outputs()
        segmentation = processor.post_process_panoptic_segmentation(outputs, threshold=0)

        self.assertTrue(len(segmentation) == self.processing_tester.batch_size)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(el["segmentation"].shape, (self.processing_tester.height, self.processing_tester.width))
