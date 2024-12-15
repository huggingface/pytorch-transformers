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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

    if is_vision_available():
        from transformers import OneFormerImageProcessor
        from transformers.models.oneformer.image_processing_oneformer import binary_mask_to_rle, prepare_metadata
        from transformers.models.oneformer.modeling_oneformer import OneFormerForUniversalSegmentationOutput

if is_vision_available():
    from PIL import Image


class OneFormerImageProcessorTester:
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
        do_reduce_labels=False,
        ignore_index=255,
        repo_path="shi-labs/oneformer_demo",
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
        self.class_info_file = class_info_file
        self.num_text = num_text
        self.repo_path = repo_path

        # for the post_process_functions
        self.batch_size = 2
        self.num_queries = 10
        self.num_classes = 10
        self.height = 3
        self.width = 4
        self.num_labels = num_labels
        self.do_reduce_labels = do_reduce_labels
        self.ignore_index = ignore_index

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "num_labels": self.num_labels,
            "do_reduce_labels": self.do_reduce_labels,
            "ignore_index": self.ignore_index,
            "class_info_file": self.class_info_file,
            "num_text": self.num_text,
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
            elif isinstance(image, np.ndarray):
                h, w = image.shape[0], image.shape[1]
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
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width

    def get_fake_oneformer_outputs(self):
        return OneFormerForUniversalSegmentationOutput(
            # +1 for null class
            class_queries_logits=torch.randn((self.batch_size, self.num_queries, self.num_classes + 1)),
            masks_queries_logits=torch.randn((self.batch_size, self.num_queries, self.height, self.width)),
        )

    def expected_output_image_shape(self, images):
        height, width = self.get_expected_values(images, batched=True)
        return self.num_channels, height, width

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class OneFormerImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = OneFormerImageProcessor if (is_vision_available() and is_torch_available()) else None
    # only for test_image_processing_common.test_image_proc_to_json_string
    image_processing_class = image_processing_class

    def setUp(self):
        super().setUp()
        self.image_processor_tester = OneFormerImageProcessorTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_proc_properties(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, "image_mean"))
        self.assertTrue(hasattr(image_processor, "image_std"))
        self.assertTrue(hasattr(image_processor, "do_normalize"))
        self.assertTrue(hasattr(image_processor, "do_resize"))
        self.assertTrue(hasattr(image_processor, "size"))
        self.assertTrue(hasattr(image_processor, "ignore_index"))
        self.assertTrue(hasattr(image_processor, "class_info_file"))
        self.assertTrue(hasattr(image_processor, "num_text"))
        self.assertTrue(hasattr(image_processor, "repo_path"))
        self.assertTrue(hasattr(image_processor, "metadata"))
        self.assertTrue(hasattr(image_processor, "do_reduce_labels"))

    def comm_get_image_processor_inputs(
        self, with_segmentation_maps=False, is_instance_map=False, segmentation_type="np"
    ):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        # prepare image and target
        num_labels = self.image_processor_tester.num_labels
        annotations = None
        instance_id_to_semantic_id = None
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        if with_segmentation_maps:
            high = num_labels
            if is_instance_map:
                labels_expanded = list(range(num_labels)) * 2
                instance_id_to_semantic_id = dict(enumerate(labels_expanded))
            annotations = [
                np.random.randint(0, high * 2, (img.size[1], img.size[0])).astype(np.uint8) for img in image_inputs
            ]
            if segmentation_type == "pil":
                annotations = [Image.fromarray(annotation) for annotation in annotations]

        inputs = image_processor(
            image_inputs,
            ["semantic"] * len(image_inputs),
            annotations,
            return_tensors="pt",
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            pad_and_return_pixel_mask=True,
        )

        return inputs

    @unittest.skip
    def test_init_without_params(self):
        pass

    def test_call_with_segmentation_maps(self):
        def common(is_instance_map=False, segmentation_type=None):
            inputs = self.comm_get_image_processor_inputs(
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
                self.assertEqual(len(text_input), self.image_processor_tester.num_text)

        common()
        common(is_instance_map=True)
        common(is_instance_map=False, segmentation_type="pil")
        common(is_instance_map=True, segmentation_type="pil")

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
        fature_extractor = self.image_processing_class(
            num_labels=self.image_processor_tester.num_classes,
            max_seq_length=77,
            task_seq_length=77,
            class_info_file="ade20k_panoptic.json",
            num_text=self.image_processor_tester.num_text,
            repo_path="shi-labs/oneformer_demo",
        )
        outputs = self.image_processor_tester.get_fake_oneformer_outputs()

        segmentation = fature_extractor.post_process_semantic_segmentation(outputs)

        self.assertEqual(len(segmentation), self.image_processor_tester.batch_size)
        self.assertEqual(
            segmentation[0].shape,
            (
                self.image_processor_tester.height,
                self.image_processor_tester.width,
            ),
        )

        target_sizes = [(1, 4) for i in range(self.image_processor_tester.batch_size)]
        segmentation = fature_extractor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        self.assertEqual(segmentation[0].shape, target_sizes[0])

    def test_post_process_instance_segmentation(self):
        image_processor = self.image_processing_class(
            num_labels=self.image_processor_tester.num_classes,
            max_seq_length=77,
            task_seq_length=77,
            class_info_file="ade20k_panoptic.json",
            num_text=self.image_processor_tester.num_text,
            repo_path="shi-labs/oneformer_demo",
        )
        outputs = self.image_processor_tester.get_fake_oneformer_outputs()
        segmentation = image_processor.post_process_instance_segmentation(outputs, threshold=0)

        self.assertTrue(len(segmentation) == self.image_processor_tester.batch_size)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(
                el["segmentation"].shape, (self.image_processor_tester.height, self.image_processor_tester.width)
            )

        segmentation_with_opts = image_processor.post_process_instance_segmentation(
            outputs,
            threshold=0,
            target_sizes=[(1, 4) for _ in range(self.image_processor_tester.batch_size)],
            task_type="panoptic",
        )
        self.assertTrue(len(segmentation_with_opts) == self.image_processor_tester.batch_size)
        for el in segmentation_with_opts:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(el["segmentation"].shape, (1, 4))

    def test_post_process_panoptic_segmentation(self):
        image_processor = self.image_processing_class(
            num_labels=self.image_processor_tester.num_classes,
            max_seq_length=77,
            task_seq_length=77,
            class_info_file="ade20k_panoptic.json",
            num_text=self.image_processor_tester.num_text,
            repo_path="shi-labs/oneformer_demo",
        )
        outputs = self.image_processor_tester.get_fake_oneformer_outputs()
        segmentation = image_processor.post_process_panoptic_segmentation(outputs, threshold=0)

        self.assertTrue(len(segmentation) == self.image_processor_tester.batch_size)
        for el in segmentation:
            self.assertTrue("segmentation" in el)
            self.assertTrue("segments_info" in el)
            self.assertEqual(type(el["segments_info"]), list)
            self.assertEqual(
                el["segmentation"].shape, (self.image_processor_tester.height, self.image_processor_tester.width)
            )

    def test_can_load_with_local_metadata(self):
        # Create a temporary json file
        class_info = {
            "0": {"isthing": 0, "name": "foo"},
            "1": {"isthing": 0, "name": "bar"},
            "2": {"isthing": 1, "name": "baz"},
        }
        metadata = prepare_metadata(class_info)

        with tempfile.TemporaryDirectory() as tmpdirname:
            metadata_path = os.path.join(tmpdirname, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(class_info, f)

            config_dict = self.image_processor_dict
            config_dict["class_info_file"] = metadata_path
            config_dict["repo_path"] = tmpdirname
            image_processor = self.image_processing_class(**config_dict)

        self.assertEqual(image_processor.metadata, metadata)

    def test_removed_deprecated_kwargs(self):
        image_processor_dict = dict(self.image_processor_dict)
        image_processor_dict.pop("do_reduce_labels", None)
        image_processor_dict["reduce_labels"] = True

        # test we are able to create the image processor with the deprecated kwargs
        image_processor = self.image_processing_class(**image_processor_dict)
        self.assertEqual(image_processor.do_reduce_labels, True)

        # test we still support reduce_labels with config
        image_processor = self.image_processing_class.from_dict(image_processor_dict)
        self.assertEqual(image_processor.do_reduce_labels, True)
