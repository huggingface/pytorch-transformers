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

from transformers import (
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
    AutoTokenizer,
    is_torch_available,
    is_vision_available,
)
from transformers.pipelines import DocumentQuestionAnsweringPipeline, pipeline
from transformers.pipelines.document_question_answering import apply_tesseract
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_detectron2,
    require_pytesseract,
    require_tf,
    require_torch,
    require_torch_bf16,
    require_vision,
    slow,
)

from .test_pipelines_common import ANY


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers.image_utils import load_image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass

    def load_image(_):
        return None


# This is a pinned image from a specific revision of a document question answering space, hosted by HuggingFace,
# so we can expect it to be available.
INVOICE_URL = (
    "https://huggingface.co/spaces/impira/docquery/resolve/2f6c96314dc84dfda62d40de9da55f2f5165d403/invoice.png"
)


@is_pipeline_test
@require_torch
@require_vision
class DocumentQuestionAnsweringPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING

    @require_pytesseract
    @require_vision
    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        dqa_pipeline = DocumentQuestionAnsweringPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
        )

        image = INVOICE_URL
        word_boxes = list(zip(*apply_tesseract(load_image(image), None, "")))
        question = "What is the placebo?"
        examples = [
            {
                "image": load_image(image),
                "question": question,
            },
            {
                "image": image,
                "question": question,
            },
            {
                "image": image,
                "question": question,
                "word_boxes": word_boxes,
            },
        ]
        return dqa_pipeline, examples

    def run_pipeline_test(self, dqa_pipeline, examples):
        outputs = dqa_pipeline(examples, top_k=2)
        self.assertEqual(
            outputs,
            [
                [
                    {"score": ANY(float), "answer": ANY(str), "start": ANY(int), "end": ANY(int)},
                    {"score": ANY(float), "answer": ANY(str), "start": ANY(int), "end": ANY(int)},
                ]
            ]
            * 3,
        )

    @require_torch
    @require_detectron2
    @require_pytesseract
    def test_small_model_pt(self):
        dqa_pipeline = pipeline(
            "document-question-answering", model="hf-internal-testing/tiny-random-layoutlmv2-for-dqa-test"
        )
        image = INVOICE_URL
        question = "How many cats are there?"

        expected_output = [
            {"score": 0.0001, "answer": "oy 2312/2019", "start": 38, "end": 39},
            {"score": 0.0001, "answer": "oy 2312/2019 DUE", "start": 38, "end": 40},
        ]
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output)

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output)

        # This image does not detect ANY text in it, meaning layoutlmv2 should fail.
        # Empty answer probably
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(outputs, [])

        # We can optionnally pass directly the words and bounding boxes
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        words = []
        boxes = []
        outputs = dqa_pipeline(image=image, question=question, words=words, boxes=boxes, top_k=2)
        self.assertEqual(outputs, [])

    @require_torch
    @require_torch_bf16
    @require_detectron2
    @require_pytesseract
    def test_small_model_pt_bf16(self):
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="hf-internal-testing/tiny-random-layoutlmv2-for-dqa-test",
            torch_dtype=torch.bfloat16,
        )
        image = INVOICE_URL
        question = "How many cats are there?"

        expected_output = [
            {"score": 0.0001, "answer": "oy 2312/2019", "start": 38, "end": 39},
            {"score": 0.0001, "answer": "oy 2312/2019 DUE", "start": 38, "end": 40},
        ]
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output)

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output)

        # This image does not detect ANY text in it, meaning layoutlmv2 should fail.
        # Empty answer probably
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(outputs, [])

        # We can optionnally pass directly the words and bounding boxes
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        words = []
        boxes = []
        outputs = dqa_pipeline(image=image, question=question, words=words, boxes=boxes, top_k=2)
        self.assertEqual(outputs, [])

    # 	 TODO: Enable this once hf-internal-testing/tiny-random-donut is implemented
    #    @require_torch
    #    def test_small_model_pt_donut(self):
    #        dqa_pipeline = pipeline("document-question-answering", model="hf-internal-testing/tiny-random-donut")
    #        # dqa_pipeline = pipeline("document-question-answering", model="../tiny-random-donut")
    #        image = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
    #        question = "How many cats are there?"
    #
    #        outputs = dqa_pipeline(image=image, question=question, top_k=2)
    #        self.assertEqual(
    #            nested_simplify(outputs, decimals=4), [{"score": 0.8799, "answer": "2"}, {"score": 0.296, "answer": "1"}]
    #        )

    @slow
    @require_torch
    @require_detectron2
    @require_pytesseract
    def test_large_model_pt(self):
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",
            revision="9977165",
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9944, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0009, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9944, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0009, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9944, "answer": "us-001", "start": 16, "end": 16},
                    {"score": 0.0009, "answer": "us-001", "start": 16, "end": 16},
                ],
            ]
            * 2,
        )

    @slow
    @require_torch
    @require_detectron2
    @require_pytesseract
    def test_large_model_pt_chunk(self):
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",
            revision="9977165",
            max_seq_len=50,
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9974, "answer": "1110212019", "start": 23, "end": 23},
                {"score": 0.9948, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9974, "answer": "1110212019", "start": 23, "end": 23},
                {"score": 0.9948, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9974, "answer": "1110212019", "start": 23, "end": 23},
                    {"score": 0.9948, "answer": "us-001", "start": 16, "end": 16},
                ]
            ]
            * 2,
        )

    @slow
    @require_torch
    @require_pytesseract
    @require_vision
    def test_large_model_pt_layoutlm(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "impira/layoutlm-document-qa", revision="3dc6de3", add_prefix_space=True
        )
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
            tokenizer=tokenizer,
            revision="3dc6de3",
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=3),
            [
                {"score": 0.425, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.082, "answer": "1110212019", "start": 23, "end": 23},
            ],
        )

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=3),
            [
                {"score": 0.425, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.082, "answer": "1110212019", "start": 23, "end": 23},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=3),
            [
                [
                    {"score": 0.425, "answer": "us-001", "start": 16, "end": 16},
                    {"score": 0.082, "answer": "1110212019", "start": 23, "end": 23},
                ]
            ]
            * 2,
        )

        word_boxes = list(zip(*apply_tesseract(load_image(image), None, "")))

        # This model should also work if `image` is set to None
        outputs = dqa_pipeline({"image": None, "word_boxes": word_boxes, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=3),
            [
                {"score": 0.425, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.082, "answer": "1110212019", "start": 23, "end": 23},
            ],
        )

    @slow
    @require_torch
    @require_pytesseract
    @require_vision
    def test_large_model_pt_layoutlm_chunk(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "impira/layoutlm-document-qa", revision="3dc6de3", add_prefix_space=True
        )
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
            tokenizer=tokenizer,
            revision="3dc6de3",
            max_seq_len=50,
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9999, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.9998, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9999, "answer": "us-001", "start": 16, "end": 16},
                    {"score": 0.9998, "answer": "us-001", "start": 16, "end": 16},
                ]
            ]
            * 2,
        )

        word_boxes = list(zip(*apply_tesseract(load_image(image), None, "")))

        # This model should also work if `image` is set to None
        outputs = dqa_pipeline({"image": None, "word_boxes": word_boxes, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9999, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.9998, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

    @slow
    @require_torch
    def test_large_model_pt_donut(self):
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="naver-clova-ix/donut-base-finetuned-docvqa",
            tokenizer=AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa"),
            image_processor="naver-clova-ix/donut-base-finetuned-docvqa",
        )

        image = INVOICE_URL
        question = "What is the invoice number?"
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), [{"answer": "us-001"}])

    @require_tf
    @unittest.skip(reason="Document question answering not implemented in TF")
    def test_small_model_tf(self):
        pass
