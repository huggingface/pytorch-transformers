# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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
import pathlib
import unittest

import numpy as np
from parameterized import parameterized

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import AnnotationFormatTestMixin, ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import YolosImageProcessor


class YolosImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_rescale=True,
        rescale_factor=1 / 255,
        do_pad=True,
    ):
        # by setting size["longest_edge"] > max_resolution we're effectively not testing this :p
        size = size if size is not None else {"shortest_edge": 18, "longest_edge": 1333}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_pad": self.do_pad,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to YolosImageProcessor,
        assuming do_resize is set to True with a scalar size.
        """
        if not batched:
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, np.ndarray):
                height, width = image.shape[0], image.shape[1]
            else:
                height, width = image.shape[1], image.shape[2]

            size = self.size["shortest_edge"]
            max_size = self.size.get("longest_edge", None)
            if max_size is not None:
                min_original_size = float(min((height, width)))
                max_original_size = float(max((height, width)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if width <= height and width != size:
                height = int(size * height / width)
                width = size
            elif height < width and height != size:
                width = int(size * width / height)
                height = size
            width_mod = width % 16
            height_mod = height % 16
            expected_width = width - width_mod
            expected_height = height - height_mod

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width

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
class YolosImageProcessingTest(AnnotationFormatTestMixin, ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = YolosImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = YolosImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"shortest_edge": 18, "longest_edge": 1333})
        self.assertEqual(image_processor.do_pad, True)

        image_processor = self.image_processing_class.from_dict(
            self.image_processor_dict, size=42, max_size=84, pad_and_return_pixel_mask=False
        )
        self.assertEqual(image_processor.size, {"shortest_edge": 42, "longest_edge": 84})
        self.assertEqual(image_processor.do_pad, False)

    def test_equivalence_padding(self):
        # Initialize image_processings
        image_processing_1 = self.image_processing_class(**self.image_processor_dict)
        image_processing_2 = self.image_processing_class(do_resize=False, do_normalize=False, do_rescale=False)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test whether the method "pad" and calling the image processor return the same tensors
        encoded_images_with_method = image_processing_1.pad(image_inputs, return_tensors="pt")
        encoded_images = image_processing_2(image_inputs, return_tensors="pt")

        torch.testing.assert_close(
            encoded_images_with_method["pixel_values"], encoded_images["pixel_values"], rtol=1e-4, atol=1e-4
        )

    @parameterized.expand(
        [
            ((3, 100, 1500), 1333, 800),
            ((3, 400, 400), 1333, 800),
            ((3, 1500, 1500), 1333, 800),
            ((3, 800, 1333), 1333, 800),
            ((3, 1333, 800), 1333, 800),
            ((3, 800, 800), 400, 400),
        ]
    )
    def test_resize_max_size_respected(self, image_size, longest_edge, shortest_edge):
        image_processor = self.image_processing_class(**self.image_processor_dict)

        # create torch tensors as image
        image = torch.randint(0, 256, image_size, dtype=torch.uint8)
        processed_image = image_processor(
            image,
            size={"longest_edge": longest_edge, "shortest_edge": shortest_edge},
            do_pad=False,
            return_tensors="pt",
        )["pixel_values"]

        shape = list(processed_image.shape[-2:])
        max_size, min_size = max(shape), min(shape)
        self.assertTrue(max_size <= 1333, f"Expected max_size <= 1333, got image shape {shape}")
        self.assertTrue(min_size <= 800, f"Expected min_size <= 800, got image shape {shape}")

    @slow
    def test_call_pytorch_with_coco_detection_annotations(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"image_id": 39769, "annotations": target}

        # encode them
        image_processing = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
        encoding = image_processing(images=image, annotations=target, return_tensors="pt")

        # verify pixel values
        expected_shape = torch.Size([1, 3, 800, 1056])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        torch.testing.assert_close(encoding["pixel_values"][0, 0, 0, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # verify area
        expected_area = torch.tensor([5832.7256, 11144.6689, 484763.2500, 829269.8125, 146579.4531, 164177.6250])
        torch.testing.assert_close(encoding["labels"][0]["area"], expected_area)
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([0.5503, 0.2765, 0.0604, 0.2215])
        torch.testing.assert_close(encoding["labels"][0]["boxes"][0], expected_boxes_slice, rtol=1e-3, atol=1e-3)
        # verify image_id
        expected_image_id = torch.tensor([39769])
        torch.testing.assert_close(encoding["labels"][0]["image_id"], expected_image_id)
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        torch.testing.assert_close(encoding["labels"][0]["iscrowd"], expected_is_crowd)
        # verify class_labels
        expected_class_labels = torch.tensor([75, 75, 63, 65, 17, 17])
        torch.testing.assert_close(encoding["labels"][0]["class_labels"], expected_class_labels)
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        torch.testing.assert_close(encoding["labels"][0]["orig_size"], expected_orig_size)
        # verify size
        expected_size = torch.tensor([800, 1056])
        torch.testing.assert_close(encoding["labels"][0]["size"], expected_size)

    @slow
    def test_call_pytorch_with_coco_panoptic_annotations(self):
        # prepare image, target and masks_path
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_panoptic_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"file_name": "000000039769.png", "image_id": 39769, "segments_info": target}

        masks_path = pathlib.Path("./tests/fixtures/tests_samples/COCO/coco_panoptic")

        # encode them
        image_processing = YolosImageProcessor(format="coco_panoptic")
        encoding = image_processing(images=image, annotations=target, masks_path=masks_path, return_tensors="pt")

        # verify pixel values
        expected_shape = torch.Size([1, 3, 800, 1056])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        torch.testing.assert_close(encoding["pixel_values"][0, 0, 0, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # verify area
        expected_area = torch.tensor([146591.5000, 163974.2500, 480092.2500, 11187.0000, 5824.5000, 7562.5000])
        torch.testing.assert_close(encoding["labels"][0]["area"], expected_area)
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([0.2625, 0.5437, 0.4688, 0.8625])
        torch.testing.assert_close(encoding["labels"][0]["boxes"][0], expected_boxes_slice, rtol=1e-3, atol=1e-3)
        # verify image_id
        expected_image_id = torch.tensor([39769])
        torch.testing.assert_close(encoding["labels"][0]["image_id"], expected_image_id)
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        torch.testing.assert_close(encoding["labels"][0]["iscrowd"], expected_is_crowd)
        # verify class_labels
        expected_class_labels = torch.tensor([17, 17, 63, 75, 75, 93])
        torch.testing.assert_close(encoding["labels"][0]["class_labels"], expected_class_labels)
        # verify masks
        expected_masks_sum = 815161
        self.assertEqual(encoding["labels"][0]["masks"].sum().item(), expected_masks_sum)
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        torch.testing.assert_close(encoding["labels"][0]["orig_size"], expected_orig_size)
        # verify size
        expected_size = torch.tensor([800, 1056])
        torch.testing.assert_close(encoding["labels"][0]["size"], expected_size)

    # Output size is slight different from DETR as yolos takes mod of 16
    @slow
    def test_batched_coco_detection_annotations(self):
        image_0 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        image_1 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png").resize((800, 800))

        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        annotations_0 = {"image_id": 39769, "annotations": target}
        annotations_1 = {"image_id": 39769, "annotations": target}

        # Adjust the bounding boxes for the resized image
        w_0, h_0 = image_0.size
        w_1, h_1 = image_1.size
        for i in range(len(annotations_1["annotations"])):
            coords = annotations_1["annotations"][i]["bbox"]
            new_bbox = [
                coords[0] * w_1 / w_0,
                coords[1] * h_1 / h_0,
                coords[2] * w_1 / w_0,
                coords[3] * h_1 / h_0,
            ]
            annotations_1["annotations"][i]["bbox"] = new_bbox

        images = [image_0, image_1]
        annotations = [annotations_0, annotations_1]

        image_processing = YolosImageProcessor()
        encoding = image_processing(
            images=images,
            annotations=annotations,
            return_segmentation_masks=True,
            return_tensors="pt",  # do_convert_annotations=True
        )

        # Check the pixel values have been padded
        postprocessed_height, postprocessed_width = 800, 1056
        expected_shape = torch.Size([2, 3, postprocessed_height, postprocessed_width])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        # Check the bounding boxes have been adjusted for padded images
        self.assertEqual(encoding["labels"][0]["boxes"].shape, torch.Size([6, 4]))
        self.assertEqual(encoding["labels"][1]["boxes"].shape, torch.Size([6, 4]))
        expected_boxes_0 = torch.tensor(
            [
                [0.6879, 0.4609, 0.0755, 0.3691],
                [0.2118, 0.3359, 0.2601, 0.1566],
                [0.5011, 0.5000, 0.9979, 1.0000],
                [0.5010, 0.5020, 0.9979, 0.9959],
                [0.3284, 0.5944, 0.5884, 0.8112],
                [0.8394, 0.5445, 0.3213, 0.9110],
            ]
        )
        expected_boxes_1 = torch.tensor(
            [
                [0.4169, 0.2765, 0.0458, 0.2215],
                [0.1284, 0.2016, 0.1576, 0.0940],
                [0.3792, 0.4933, 0.7559, 0.9865],
                [0.3794, 0.5002, 0.7563, 0.9955],
                [0.1990, 0.5456, 0.3566, 0.8646],
                [0.5845, 0.4115, 0.3462, 0.7161],
            ]
        )
        torch.testing.assert_close(encoding["labels"][0]["boxes"], expected_boxes_0, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(encoding["labels"][1]["boxes"], expected_boxes_1, rtol=1e-3, atol=1e-3)

        # Check the masks have also been padded
        self.assertEqual(encoding["labels"][0]["masks"].shape, torch.Size([6, 800, 1056]))
        self.assertEqual(encoding["labels"][1]["masks"].shape, torch.Size([6, 800, 1056]))

        # Check if do_convert_annotations=False, then the annotations are not converted to centre_x, centre_y, width, height
        # format and not in the range [0, 1]
        encoding = image_processing(
            images=images,
            annotations=annotations,
            return_segmentation_masks=True,
            do_convert_annotations=False,
            return_tensors="pt",
        )
        self.assertEqual(encoding["labels"][0]["boxes"].shape, torch.Size([6, 4]))
        self.assertEqual(encoding["labels"][1]["boxes"].shape, torch.Size([6, 4]))
        # Convert to absolute coordinates
        unnormalized_boxes_0 = torch.vstack(
            [
                expected_boxes_0[:, 0] * postprocessed_width,
                expected_boxes_0[:, 1] * postprocessed_height,
                expected_boxes_0[:, 2] * postprocessed_width,
                expected_boxes_0[:, 3] * postprocessed_height,
            ]
        ).T
        unnormalized_boxes_1 = torch.vstack(
            [
                expected_boxes_1[:, 0] * postprocessed_width,
                expected_boxes_1[:, 1] * postprocessed_height,
                expected_boxes_1[:, 2] * postprocessed_width,
                expected_boxes_1[:, 3] * postprocessed_height,
            ]
        ).T
        # Convert from centre_x, centre_y, width, height to x_min, y_min, x_max, y_max
        expected_boxes_0 = torch.vstack(
            [
                unnormalized_boxes_0[:, 0] - unnormalized_boxes_0[:, 2] / 2,
                unnormalized_boxes_0[:, 1] - unnormalized_boxes_0[:, 3] / 2,
                unnormalized_boxes_0[:, 0] + unnormalized_boxes_0[:, 2] / 2,
                unnormalized_boxes_0[:, 1] + unnormalized_boxes_0[:, 3] / 2,
            ]
        ).T
        expected_boxes_1 = torch.vstack(
            [
                unnormalized_boxes_1[:, 0] - unnormalized_boxes_1[:, 2] / 2,
                unnormalized_boxes_1[:, 1] - unnormalized_boxes_1[:, 3] / 2,
                unnormalized_boxes_1[:, 0] + unnormalized_boxes_1[:, 2] / 2,
                unnormalized_boxes_1[:, 1] + unnormalized_boxes_1[:, 3] / 2,
            ]
        ).T
        torch.testing.assert_close(encoding["labels"][0]["boxes"], expected_boxes_0, rtol=1, atol=1)
        torch.testing.assert_close(encoding["labels"][1]["boxes"], expected_boxes_1, rtol=1, atol=1)

    # Output size is slight different from DETR as yolos takes mod of 16
    def test_batched_coco_panoptic_annotations(self):
        # prepare image, target and masks_path
        image_0 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        image_1 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png").resize((800, 800))

        with open("./tests/fixtures/tests_samples/COCO/coco_panoptic_annotations.txt", "r") as f:
            target = json.loads(f.read())

        annotation_0 = {"file_name": "000000039769.png", "image_id": 39769, "segments_info": target}
        annotation_1 = {"file_name": "000000039769.png", "image_id": 39769, "segments_info": target}

        w_0, h_0 = image_0.size
        w_1, h_1 = image_1.size
        for i in range(len(annotation_1["segments_info"])):
            coords = annotation_1["segments_info"][i]["bbox"]
            new_bbox = [
                coords[0] * w_1 / w_0,
                coords[1] * h_1 / h_0,
                coords[2] * w_1 / w_0,
                coords[3] * h_1 / h_0,
            ]
            annotation_1["segments_info"][i]["bbox"] = new_bbox

        masks_path = pathlib.Path("./tests/fixtures/tests_samples/COCO/coco_panoptic")

        images = [image_0, image_1]
        annotations = [annotation_0, annotation_1]

        # encode them
        image_processing = YolosImageProcessor(format="coco_panoptic")
        encoding = image_processing(
            images=images,
            annotations=annotations,
            masks_path=masks_path,
            return_tensors="pt",
            return_segmentation_masks=True,
        )

        # Check the pixel values have been padded
        postprocessed_height, postprocessed_width = 800, 1056
        expected_shape = torch.Size([2, 3, postprocessed_height, postprocessed_width])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        # Check the bounding boxes have been adjusted for padded images
        self.assertEqual(encoding["labels"][0]["boxes"].shape, torch.Size([6, 4]))
        self.assertEqual(encoding["labels"][1]["boxes"].shape, torch.Size([6, 4]))
        expected_boxes_0 = torch.tensor(
            [
                [0.2625, 0.5437, 0.4688, 0.8625],
                [0.7719, 0.4104, 0.4531, 0.7125],
                [0.5000, 0.4927, 0.9969, 0.9854],
                [0.1688, 0.2000, 0.2063, 0.0917],
                [0.5492, 0.2760, 0.0578, 0.2187],
                [0.4992, 0.4990, 0.9984, 0.9979],
            ]
        )
        expected_boxes_1 = torch.tensor(
            [
                [0.1591, 0.3262, 0.2841, 0.5175],
                [0.4678, 0.2463, 0.2746, 0.4275],
                [0.3030, 0.2956, 0.6042, 0.5913],
                [0.1023, 0.1200, 0.1250, 0.0550],
                [0.3329, 0.1656, 0.0350, 0.1312],
                [0.3026, 0.2994, 0.6051, 0.5987],
            ]
        )
        torch.testing.assert_close(encoding["labels"][0]["boxes"], expected_boxes_0, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(encoding["labels"][1]["boxes"], expected_boxes_1, rtol=1e-3, atol=1e-3)

        # Check the masks have also been padded
        self.assertEqual(encoding["labels"][0]["masks"].shape, torch.Size([6, 800, 1056]))
        self.assertEqual(encoding["labels"][1]["masks"].shape, torch.Size([6, 800, 1056]))

        # Check if do_convert_annotations=False, then the annotations are not converted to centre_x, centre_y, width, height
        # format and not in the range [0, 1]
        encoding = image_processing(
            images=images,
            annotations=annotations,
            masks_path=masks_path,
            return_segmentation_masks=True,
            do_convert_annotations=False,
            return_tensors="pt",
        )
        self.assertEqual(encoding["labels"][0]["boxes"].shape, torch.Size([6, 4]))
        self.assertEqual(encoding["labels"][1]["boxes"].shape, torch.Size([6, 4]))
        # Convert to absolute coordinates
        unnormalized_boxes_0 = torch.vstack(
            [
                expected_boxes_0[:, 0] * postprocessed_width,
                expected_boxes_0[:, 1] * postprocessed_height,
                expected_boxes_0[:, 2] * postprocessed_width,
                expected_boxes_0[:, 3] * postprocessed_height,
            ]
        ).T
        unnormalized_boxes_1 = torch.vstack(
            [
                expected_boxes_1[:, 0] * postprocessed_width,
                expected_boxes_1[:, 1] * postprocessed_height,
                expected_boxes_1[:, 2] * postprocessed_width,
                expected_boxes_1[:, 3] * postprocessed_height,
            ]
        ).T
        # Convert from centre_x, centre_y, width, height to x_min, y_min, x_max, y_max
        expected_boxes_0 = torch.vstack(
            [
                unnormalized_boxes_0[:, 0] - unnormalized_boxes_0[:, 2] / 2,
                unnormalized_boxes_0[:, 1] - unnormalized_boxes_0[:, 3] / 2,
                unnormalized_boxes_0[:, 0] + unnormalized_boxes_0[:, 2] / 2,
                unnormalized_boxes_0[:, 1] + unnormalized_boxes_0[:, 3] / 2,
            ]
        ).T
        expected_boxes_1 = torch.vstack(
            [
                unnormalized_boxes_1[:, 0] - unnormalized_boxes_1[:, 2] / 2,
                unnormalized_boxes_1[:, 1] - unnormalized_boxes_1[:, 3] / 2,
                unnormalized_boxes_1[:, 0] + unnormalized_boxes_1[:, 2] / 2,
                unnormalized_boxes_1[:, 1] + unnormalized_boxes_1[:, 3] / 2,
            ]
        ).T
        torch.testing.assert_close(encoding["labels"][0]["boxes"], expected_boxes_0, rtol=1)
        torch.testing.assert_close(encoding["labels"][1]["boxes"], expected_boxes_1, rtol=1)

    # Copied from tests.models.detr.test_image_processing_detr.DetrImageProcessingTest.test_max_width_max_height_resizing_and_pad_strategy with Detr->Yolos
    def test_max_width_max_height_resizing_and_pad_strategy(self):
        for image_processing_class in self.image_processor_list:
            image_1 = torch.ones([200, 100, 3], dtype=torch.uint8)

            # do_pad=False, max_height=100, max_width=100, image=200x100 -> 100x50
            image_processor = image_processing_class(
                size={"max_height": 100, "max_width": 100},
                do_pad=False,
            )
            inputs = image_processor(images=[image_1], return_tensors="pt")
            self.assertEqual(inputs["pixel_values"].shape, torch.Size([1, 3, 100, 50]))

            # do_pad=False, max_height=300, max_width=100, image=200x100 -> 200x100
            image_processor = image_processing_class(
                size={"max_height": 300, "max_width": 100},
                do_pad=False,
            )
            inputs = image_processor(images=[image_1], return_tensors="pt")

            # do_pad=True, max_height=100, max_width=100, image=200x100 -> 100x100
            image_processor = image_processing_class(
                size={"max_height": 100, "max_width": 100}, do_pad=True, pad_size={"height": 100, "width": 100}
            )
            inputs = image_processor(images=[image_1], return_tensors="pt")
            self.assertEqual(inputs["pixel_values"].shape, torch.Size([1, 3, 100, 100]))

            # do_pad=True, max_height=300, max_width=100, image=200x100 -> 300x100
            image_processor = image_processing_class(
                size={"max_height": 300, "max_width": 100},
                do_pad=True,
                pad_size={"height": 301, "width": 101},
            )
            inputs = image_processor(images=[image_1], return_tensors="pt")
            self.assertEqual(inputs["pixel_values"].shape, torch.Size([1, 3, 301, 101]))

            ### Check for batch
            image_2 = torch.ones([100, 150, 3], dtype=torch.uint8)

            # do_pad=True, max_height=150, max_width=100, images=[200x100, 100x150] -> 150x100
            image_processor = image_processing_class(
                size={"max_height": 150, "max_width": 100},
                do_pad=True,
                pad_size={"height": 150, "width": 100},
            )
            inputs = image_processor(images=[image_1, image_2], return_tensors="pt")
            self.assertEqual(inputs["pixel_values"].shape, torch.Size([2, 3, 150, 100]))
