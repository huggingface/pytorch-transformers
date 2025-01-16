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
import json
import unittest

import requests

from transformers.testing_utils import require_torch, require_torch_gpu, require_torchvision, require_vision, slow
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image

    from transformers import RTDetrImageProcessor, RTDetrImageProcessorFast

if is_torch_available():
    import torch


class RTDetrImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=4,
        num_channels=3,
        do_resize=True,
        size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=False,
        do_pad=False,
        return_tensors="pt",
    ):
        super().__init__()
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 640, "width": 640}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_pad = do_pad
        self.return_tensors = return_tensors

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "do_pad": self.do_pad,
            "return_tensors": self.return_tensors,
        }

    def get_expected_values(self):
        return self.size["height"], self.size["width"]

    def expected_output_image_shape(self, images):
        height, width = self.get_expected_values()
        return self.num_channels, height, width

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=30,
            max_resolution=400,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class RtDetrImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = RTDetrImageProcessor if is_vision_available() else None
    fast_image_processing_class = RTDetrImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = RTDetrImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "resample"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "return_tensors"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 640, "width": 640})

    def test_valid_coco_detection_annotations(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        params = {"image_id": 39769, "annotations": target}

        for image_processing_class in self.image_processor_list:
            # encode them
            image_processing = image_processing_class.from_pretrained("PekingU/rtdetr_r50vd")

            # legal encodings (single image)
            _ = image_processing(images=image, annotations=params, return_tensors="pt")
            _ = image_processing(images=image, annotations=[params], return_tensors="pt")

            # legal encodings (batch of one image)
            _ = image_processing(images=[image], annotations=params, return_tensors="pt")
            _ = image_processing(images=[image], annotations=[params], return_tensors="pt")

            # legal encoding (batch of more than one image)
            n = 5
            _ = image_processing(images=[image] * n, annotations=[params] * n, return_tensors="pt")

            # example of an illegal encoding (missing the 'image_id' key)
            with self.assertRaises(ValueError) as e:
                image_processing(images=image, annotations={"annotations": target}, return_tensors="pt")

            self.assertTrue(str(e.exception).startswith("Invalid COCO detection annotations"))

            # example of an illegal encoding (unequal lengths of images and annotations)
            with self.assertRaises(ValueError) as e:
                image_processing(images=[image] * n, annotations=[params] * (n - 1), return_tensors="pt")

            self.assertTrue(str(e.exception) == "The number of images (5) and annotations (4) do not match.")

    @slow
    def test_call_pytorch_with_coco_detection_annotations(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"image_id": 39769, "annotations": target}

        for image_processing_class in self.image_processor_list:
            # encode them
            image_processing = image_processing_class.from_pretrained("PekingU/rtdetr_r50vd")
            encoding = image_processing(images=image, annotations=target, return_tensors="pt")

            # verify pixel values
            expected_shape = torch.Size([1, 3, 640, 640])
            self.assertEqual(encoding["pixel_values"].shape, expected_shape)

            expected_slice = torch.tensor([0.5490, 0.5647, 0.5725])
            torch.testing.assert_close(encoding["pixel_values"][0, 0, 0, :3], expected_slice, rtol=1e-4, atol=1e-4)

            # verify area
            expected_area = torch.tensor([2827.9883, 5403.4761, 235036.7344, 402070.2188, 71068.8281, 79601.2812])
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
            expected_size = torch.tensor([640, 640])
            torch.testing.assert_close(encoding["labels"][0]["size"], expected_size)

    @slow
    def test_image_processor_outputs(self):
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")

        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            encoding = image_processing(images=image, return_tensors="pt")

            # verify pixel values: shape
            expected_shape = torch.Size([1, 3, 640, 640])
            self.assertEqual(encoding["pixel_values"].shape, expected_shape)

            # verify pixel values: output values
            expected_slice = torch.tensor([0.5490196347236633, 0.5647059082984924, 0.572549045085907])
            torch.testing.assert_close(encoding["pixel_values"][0, 0, 0, :3], expected_slice, rtol=1e-5, atol=1e-5)

    def test_multiple_images_processor_outputs(self):
        images_urls = [
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "http://images.cocodataset.org/val2017/000000000632.jpg",
            "http://images.cocodataset.org/val2017/000000000724.jpg",
            "http://images.cocodataset.org/val2017/000000000776.jpg",
            "http://images.cocodataset.org/val2017/000000000785.jpg",
            "http://images.cocodataset.org/val2017/000000000802.jpg",
            "http://images.cocodataset.org/val2017/000000000872.jpg",
        ]

        images = []
        for url in images_urls:
            image = Image.open(requests.get(url, stream=True).raw)
            images.append(image)

        for image_processing_class in self.image_processor_list:
            # apply image processing
            image_processing = image_processing_class(**self.image_processor_dict)
            encoding = image_processing(images=images, return_tensors="pt")

            # verify if pixel_values is part of the encoding
            self.assertIn("pixel_values", encoding)

            # verify pixel values: shape
            expected_shape = torch.Size([8, 3, 640, 640])
            self.assertEqual(encoding["pixel_values"].shape, expected_shape)

            # verify pixel values: output values
            expected_slices = torch.tensor(
                [
                    [0.5333333611488342, 0.5568627715110779, 0.5647059082984924],
                    [0.5372549295425415, 0.4705882668495178, 0.4274510145187378],
                    [0.3960784673690796, 0.35686275362968445, 0.3686274588108063],
                    [0.20784315466880798, 0.1882353127002716, 0.15294118225574493],
                    [0.364705890417099, 0.364705890417099, 0.3686274588108063],
                    [0.8078432083129883, 0.8078432083129883, 0.8078432083129883],
                    [0.4431372880935669, 0.4431372880935669, 0.4431372880935669],
                    [0.19607844948768616, 0.21176472306251526, 0.3607843220233917],
                ]
            )
            torch.testing.assert_close(encoding["pixel_values"][:, 1, 0, :3], expected_slices, rtol=1e-5, atol=1e-5)

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

        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class()
            encoding = image_processing(
                images=images,
                annotations=annotations,
                return_segmentation_masks=True,
                return_tensors="pt",  # do_convert_annotations=True
            )

            # Check the pixel values have been padded
            postprocessed_height, postprocessed_width = 640, 640
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
                    [0.5503, 0.2765, 0.0604, 0.2215],
                    [0.1695, 0.2016, 0.2080, 0.0940],
                    [0.5006, 0.4933, 0.9977, 0.9865],
                    [0.5008, 0.5002, 0.9983, 0.9955],
                    [0.2627, 0.5456, 0.4707, 0.8646],
                    [0.7715, 0.4115, 0.4570, 0.7161],
                ]
            )
            torch.testing.assert_close(encoding["labels"][0]["boxes"], expected_boxes_0, rtol=1e-3)
            torch.testing.assert_close(encoding["labels"][1]["boxes"], expected_boxes_1, rtol=1e-3)

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
            torch.testing.assert_close(encoding["labels"][0]["boxes"], expected_boxes_0, rtol=1)
            torch.testing.assert_close(encoding["labels"][1]["boxes"], expected_boxes_1, rtol=1)

    @slow
    @require_torch_gpu
    @require_torchvision
    # Copied from tests.models.detr.test_image_processing_detr.DetrImageProcessingTest.test_fast_processor_equivalence_cpu_gpu_coco_detection_annotations
    def test_fast_processor_equivalence_cpu_gpu_coco_detection_annotations(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"image_id": 39769, "annotations": target}

        processor = self.image_processor_list[1]()
        # 1. run processor on CPU
        encoding_cpu = processor(images=image, annotations=target, return_tensors="pt", device="cpu")
        # 2. run processor on GPU
        encoding_gpu = processor(images=image, annotations=target, return_tensors="pt", device="cuda")

        # verify pixel values
        self.assertEqual(encoding_cpu["pixel_values"].shape, encoding_gpu["pixel_values"].shape)
        self.assertTrue(
            torch.allclose(
                encoding_cpu["pixel_values"][0, 0, 0, :3],
                encoding_gpu["pixel_values"][0, 0, 0, :3].to("cpu"),
                atol=1e-4,
            )
        )
        # verify area
        torch.testing.assert_close(encoding_cpu["labels"][0]["area"], encoding_gpu["labels"][0]["area"].to("cpu"))
        # verify boxes
        self.assertEqual(encoding_cpu["labels"][0]["boxes"].shape, encoding_gpu["labels"][0]["boxes"].shape)
        self.assertTrue(
            torch.allclose(
                encoding_cpu["labels"][0]["boxes"][0], encoding_gpu["labels"][0]["boxes"][0].to("cpu"), atol=1e-3
            )
        )
        # verify image_id
        torch.testing.assert_close(
            encoding_cpu["labels"][0]["image_id"], encoding_gpu["labels"][0]["image_id"].to("cpu")
        )
        # verify is_crowd
        torch.testing.assert_close(
            encoding_cpu["labels"][0]["iscrowd"], encoding_gpu["labels"][0]["iscrowd"].to("cpu")
        )
        # verify class_labels
        self.assertTrue(
            torch.allclose(
                encoding_cpu["labels"][0]["class_labels"], encoding_gpu["labels"][0]["class_labels"].to("cpu")
            )
        )
        # verify orig_size
        torch.testing.assert_close(
            encoding_cpu["labels"][0]["orig_size"], encoding_gpu["labels"][0]["orig_size"].to("cpu")
        )
        # verify size
        torch.testing.assert_close(encoding_cpu["labels"][0]["size"], encoding_gpu["labels"][0]["size"].to("cpu"))
