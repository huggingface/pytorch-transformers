# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import unittest

from datasets import load_dataset
from huggingface_hub import VisualQuestionAnsweringOutputElement

from transformers import MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING, is_vision_available
from transformers.pipelines import pipeline
from transformers.testing_utils import (
    compare_pipeline_output_to_hub_spec,
    is_pipeline_test,
    is_torch_available,
    nested_simplify,
    require_tf,
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)

from .test_pipelines_common import ANY


if is_torch_available():
    import torch

    from transformers.pipelines.pt_utils import KeyDataset


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@is_pipeline_test
@require_torch
@require_vision
class VisualQuestionAnsweringPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        vqa_pipeline = pipeline(
            "visual-question-answering",
            model="hf-internal-testing/tiny-vilt-random-vqa",
            torch_dtype=torch_dtype,
        )
        examples = [
            {
                "image": Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                "question": "How many cats are there?",
            },
            {
                "image": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "question": "How many cats are there?",
            },
        ]
        return vqa_pipeline, examples

    def run_pipeline_test(self, vqa_pipeline, examples):
        outputs = vqa_pipeline(examples, top_k=1)
        self.assertEqual(
            outputs,
            [
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
            ],
        )
        for output in outputs:
            for element in output:
                compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

    @require_torch
    def test_small_model_pt(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question="How many cats are there?", top_k=2)
        self.assertEqual(
            outputs, [{"score": ANY(float), "answer": ANY(str)}, {"score": ANY(float), "answer": ANY(str)}]
        )
        for element in outputs:
            compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

        outputs = vqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            outputs, [{"score": ANY(float), "answer": ANY(str)}, {"score": ANY(float), "answer": ANY(str)}]
        )
        for element in outputs:
            compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

    @require_torch
    @require_torch_accelerator
    def test_small_model_pt_blip2(self):
        vqa_pipeline = pipeline(
            "visual-question-answering", model="hf-internal-testing/tiny-random-Blip2ForConditionalGeneration"
        )
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question=question)
        self.assertEqual(outputs, [{"answer": ANY(str)}])

        outputs = vqa_pipeline({"image": image, "question": question})
        self.assertEqual(outputs, [{"answer": ANY(str)}])

        outputs = vqa_pipeline([{"image": image, "question": question}, {"image": image, "question": question}])
        self.assertEqual(outputs, [[{"answer": ANY(str)}]] * 2)

        vqa_pipeline = pipeline(
            "visual-question-answering",
            model="hf-internal-testing/tiny-random-Blip2ForConditionalGeneration",
            model_kwargs={"torch_dtype": torch.float16},
            device=torch_device,
        )
        self.assertEqual(vqa_pipeline.model.device, torch.device("{}:0".format(torch_device)))
        self.assertEqual(vqa_pipeline.model.language_model.dtype, torch.float16)
        self.assertEqual(vqa_pipeline.model.vision_model.dtype, torch.float16)

        outputs = vqa_pipeline(image=image, question=question)
        self.assertEqual(outputs, [{"answer": ANY(str)}])
        for element in outputs:
            compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

    @slow
    @require_torch
    def test_large_model_pt(self):
        vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "How many cats are there?"

        outputs = vqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4), [{"score": 0.8799, "answer": "2"}, {"score": 0.296, "answer": "1"}]
        )

        outputs = vqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4), [{"score": 0.8799, "answer": "2"}, {"score": 0.296, "answer": "1"}]
        )

        outputs = vqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [[{"score": 0.8799, "answer": "2"}, {"score": 0.296, "answer": "1"}]] * 2,
        )
        for output in outputs:
            for element in output:
                compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

    @slow
    @require_torch
    @require_torch_accelerator
    def test_large_model_pt_blip2(self):
        vqa_pipeline = pipeline(
            "visual-question-answering",
            model="Salesforce/blip2-opt-2.7b",
            model_kwargs={"torch_dtype": torch.float16},
            device=torch_device,
        )
        self.assertEqual(vqa_pipeline.model.device, torch.device("{}:0".format(torch_device)))
        self.assertEqual(vqa_pipeline.model.language_model.dtype, torch.float16)

        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        question = "Question: how many cats are there? Answer:"

        outputs = vqa_pipeline(image=image, question=question)
        self.assertEqual(outputs, [{"answer": "two"}])

        outputs = vqa_pipeline({"image": image, "question": question})
        self.assertEqual(outputs, [{"answer": "two"}])

        outputs = vqa_pipeline([{"image": image, "question": question}, {"image": image, "question": question}])
        self.assertEqual(outputs, [[{"answer": "two"}]] * 2)
        for output in outputs:
            for element in output:
                compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

    @require_torch
    def test_small_model_pt_image_list(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        images = [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000004016.png",
        ]

        outputs = vqa_pipeline(image=images, question="How many cats are there?", top_k=1)
        self.assertEqual(
            outputs, [[{"score": ANY(float), "answer": ANY(str)}], [{"score": ANY(float), "answer": ANY(str)}]]
        )
        for output in outputs:
            for element in output:
                compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

    @require_torch
    def test_small_model_pt_question_list(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        questions = ["How many cats are there?", "Are there any dogs?"]

        outputs = vqa_pipeline(image=image, question=questions, top_k=1)
        self.assertEqual(
            outputs, [[{"score": ANY(float), "answer": ANY(str)}], [{"score": ANY(float), "answer": ANY(str)}]]
        )
        for output in outputs:
            for element in output:
                compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

    @require_torch
    def test_small_model_pt_both_list(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        images = [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000004016.png",
        ]
        questions = ["How many cats are there?", "Are there any dogs?"]

        outputs = vqa_pipeline(image=images, question=questions, top_k=1)
        self.assertEqual(
            outputs,
            [
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
            ],
        )
        for output in outputs:
            for element in output:
                compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

    @require_torch
    def test_small_model_pt_dataset(self):
        vqa_pipeline = pipeline("visual-question-answering", model="hf-internal-testing/tiny-vilt-random-vqa")
        dataset = load_dataset("hf-internal-testing/dummy_image_text_data", split="train[:2]")
        question = "What's in the image?"

        outputs = vqa_pipeline(image=KeyDataset(dataset, "image"), question=question, top_k=1)
        self.assertEqual(
            outputs,
            [
                [{"score": ANY(float), "answer": ANY(str)}],
                [{"score": ANY(float), "answer": ANY(str)}],
            ],
        )
        for output in outputs:
            for element in output:
                compare_pipeline_output_to_hub_spec(element, VisualQuestionAnsweringOutputElement)

    @require_tf
    @unittest.skip(reason="Visual question answering not implemented in TF")
    def test_small_model_tf(self):
        pass
