# Copyright 2020 The HuggingFace Team. All rights reserved.
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
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    AutoModelForTableQuestionAnswering,
    AutoTokenizer,
    TableQuestionAnsweringPipeline,
    TFAutoModelForTableQuestionAnswering,
    pipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    require_pandas,
    require_tensorflow_probability,
    require_tf,
    require_torch,
    slow,
)


@is_pipeline_test
class TQAPipelineTests(unittest.TestCase):
    # Putting it there for consistency, but TQA do not have fast tokenizer
    # which are needed to generate automatic tests
    model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING

    @require_tensorflow_probability
    @require_pandas
    @require_tf
    @require_torch
    def test_small_model_tf(self):
        model_id = "lysandre/tiny-tapas-random-wtq"
        model = TFAutoModelForTableQuestionAnswering.from_pretrained(model_id, from_pt=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.assertIsInstance(model.config.aggregation_labels, dict)
        self.assertIsInstance(model.config.no_aggregation_label_index, int)

        table_querier = TableQuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
        outputs = table_querier(
            table={
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            query="how many movies has george clooney played in?",
        )
        self.assertEqual(
            outputs,
            {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
        )
        outputs = table_querier(
            table={
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            query=["how many movies has george clooney played in?", "how old is he?", "what's his date of birth?"],
        )
        self.assertEqual(
            outputs,
            [
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
            ],
        )
        outputs = table_querier(
            table={
                "Repository": ["Transformers", "Datasets", "Tokenizers"],
                "Stars": ["36542", "4512", "3934"],
                "Contributors": ["651", "77", "34"],
                "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
            },
            query=[
                "What repository has the largest number of stars?",
                "Given that the numbers of stars defines if a repository is active, what repository is the most"
                " active?",
                "What is the number of repositories?",
                "What is the average number of stars?",
                "What is the total amount of stars?",
            ],
        )
        self.assertEqual(
            outputs,
            [
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
            ],
        )

        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table=None)
        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table="")
        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table={})
        with self.assertRaises(ValueError):
            table_querier(
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                }
            )
        with self.assertRaises(ValueError):
            table_querier(
                query="",
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                },
            )
        with self.assertRaises(ValueError):
            table_querier(
                query=None,
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                },
            )

    @require_torch
    def test_small_model_pt(self, torch_dtype="float32"):
        model_id = "lysandre/tiny-tapas-random-wtq"
        model = AutoModelForTableQuestionAnswering.from_pretrained(model_id, torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.assertIsInstance(model.config.aggregation_labels, dict)
        self.assertIsInstance(model.config.no_aggregation_label_index, int)

        table_querier = TableQuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
        outputs = table_querier(
            table={
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            query="how many movies has george clooney played in?",
        )
        self.assertEqual(
            outputs,
            {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
        )
        outputs = table_querier(
            table={
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            query=["how many movies has george clooney played in?", "how old is he?", "what's his date of birth?"],
        )
        self.assertEqual(
            outputs,
            [
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
            ],
        )
        outputs = table_querier(
            table={
                "Repository": ["Transformers", "Datasets", "Tokenizers"],
                "Stars": ["36542", "4512", "3934"],
                "Contributors": ["651", "77", "34"],
                "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
            },
            query=[
                "What repository has the largest number of stars?",
                "Given that the numbers of stars defines if a repository is active, what repository is the most"
                " active?",
                "What is the number of repositories?",
                "What is the average number of stars?",
                "What is the total amount of stars?",
            ],
        )
        self.assertEqual(
            outputs,
            [
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
                {"answer": "AVERAGE > ", "coordinates": [], "cells": [], "aggregator": "AVERAGE"},
            ],
        )

        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table=None)
        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table="")
        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table={})
        with self.assertRaises(ValueError):
            table_querier(
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                }
            )
        with self.assertRaises(ValueError):
            table_querier(
                query="",
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                },
            )
        with self.assertRaises(ValueError):
            table_querier(
                query=None,
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                },
            )

    @require_torch
    def test_small_model_pt_fp16(self):
        self.test_small_model_pt(torch_dtype="float16")

    @require_torch
    def test_slow_tokenizer_sqa_pt(self, torch_dtype="float32"):
        model_id = "lysandre/tiny-tapas-random-sqa"
        model = AutoModelForTableQuestionAnswering.from_pretrained(model_id, torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        table_querier = TableQuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

        inputs = {
            "table": {
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            "query": ["how many movies has george clooney played in?", "how old is he?", "what's his date of birth?"],
        }
        sequential_outputs = table_querier(**inputs, sequential=True)
        batch_outputs = table_querier(**inputs, sequential=False)

        self.assertEqual(len(sequential_outputs), 3)
        self.assertEqual(len(batch_outputs), 3)
        self.assertEqual(sequential_outputs[0], batch_outputs[0])
        self.assertNotEqual(sequential_outputs[1], batch_outputs[1])
        # self.assertNotEqual(sequential_outputs[2], batch_outputs[2])

        table_querier = TableQuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
        outputs = table_querier(
            table={
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            query="how many movies has george clooney played in?",
        )
        self.assertEqual(
            outputs,
            {"answer": "7 february 1967", "coordinates": [(0, 3)], "cells": ["7 february 1967"]},
        )
        outputs = table_querier(
            table={
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            query=["how many movies has george clooney played in?", "how old is he?", "what's his date of birth?"],
        )
        self.assertEqual(
            outputs,
            [
                {"answer": "7 february 1967", "coordinates": [(0, 3)], "cells": ["7 february 1967"]},
                {"answer": "7 february 1967", "coordinates": [(0, 3)], "cells": ["7 february 1967"]},
                {"answer": "7 february 1967", "coordinates": [(0, 3)], "cells": ["7 february 1967"]},
            ],
        )
        outputs = table_querier(
            table={
                "Repository": ["Transformers", "Datasets", "Tokenizers"],
                "Stars": ["36542", "4512", "3934"],
                "Contributors": ["651", "77", "34"],
                "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
            },
            query=[
                "What repository has the largest number of stars?",
                "Given that the numbers of stars defines if a repository is active, what repository is the most"
                " active?",
                "What is the number of repositories?",
                "What is the average number of stars?",
                "What is the total amount of stars?",
            ],
        )
        self.assertEqual(
            outputs,
            [
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
            ],
        )

        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table=None)
        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table="")
        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table={})
        with self.assertRaises(ValueError):
            table_querier(
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                }
            )
        with self.assertRaises(ValueError):
            table_querier(
                query="",
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                },
            )
        with self.assertRaises(ValueError):
            table_querier(
                query=None,
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                },
            )

    @require_torch
    def test_slow_tokenizer_sqa_pt_fp16(self):
        self.test_slow_tokenizer_sqa_pt(torch_dtype="float16")

    @require_tf
    @require_tensorflow_probability
    @require_pandas
    @require_torch
    def test_slow_tokenizer_sqa_tf(self):
        model_id = "lysandre/tiny-tapas-random-sqa"
        model = TFAutoModelForTableQuestionAnswering.from_pretrained(model_id, from_pt=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        table_querier = TableQuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

        inputs = {
            "table": {
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            "query": ["how many movies has george clooney played in?", "how old is he?", "what's his date of birth?"],
        }
        sequential_outputs = table_querier(**inputs, sequential=True)
        batch_outputs = table_querier(**inputs, sequential=False)

        self.assertEqual(len(sequential_outputs), 3)
        self.assertEqual(len(batch_outputs), 3)
        self.assertEqual(sequential_outputs[0], batch_outputs[0])
        self.assertNotEqual(sequential_outputs[1], batch_outputs[1])
        # self.assertNotEqual(sequential_outputs[2], batch_outputs[2])

        table_querier = TableQuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
        outputs = table_querier(
            table={
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            query="how many movies has george clooney played in?",
        )
        self.assertEqual(
            outputs,
            {"answer": "7 february 1967", "coordinates": [(0, 3)], "cells": ["7 february 1967"]},
        )
        outputs = table_querier(
            table={
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            query=["how many movies has george clooney played in?", "how old is he?", "what's his date of birth?"],
        )
        self.assertEqual(
            outputs,
            [
                {"answer": "7 february 1967", "coordinates": [(0, 3)], "cells": ["7 february 1967"]},
                {"answer": "7 february 1967", "coordinates": [(0, 3)], "cells": ["7 february 1967"]},
                {"answer": "7 february 1967", "coordinates": [(0, 3)], "cells": ["7 february 1967"]},
            ],
        )
        outputs = table_querier(
            table={
                "Repository": ["Transformers", "Datasets", "Tokenizers"],
                "Stars": ["36542", "4512", "3934"],
                "Contributors": ["651", "77", "34"],
                "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
            },
            query=[
                "What repository has the largest number of stars?",
                "Given that the numbers of stars defines if a repository is active, what repository is the most"
                " active?",
                "What is the number of repositories?",
                "What is the average number of stars?",
                "What is the total amount of stars?",
            ],
        )
        self.assertEqual(
            outputs,
            [
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
                {"answer": "Python, Python", "coordinates": [(0, 3), (1, 3)], "cells": ["Python", "Python"]},
            ],
        )

        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table=None)
        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table="")
        with self.assertRaises(ValueError):
            table_querier(query="What does it do with empty context ?", table={})
        with self.assertRaises(ValueError):
            table_querier(
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                }
            )
        with self.assertRaises(ValueError):
            table_querier(
                query="",
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                },
            )
        with self.assertRaises(ValueError):
            table_querier(
                query=None,
                table={
                    "Repository": ["Transformers", "Datasets", "Tokenizers"],
                    "Stars": ["36542", "4512", "3934"],
                    "Contributors": ["651", "77", "34"],
                    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                },
            )

    @slow
    @require_torch
    def test_integration_wtq_pt(self, torch_dtype="float32"):
        table_querier = pipeline("table-question-answering", torch_dtype=torch_dtype)

        data = {
            "Repository": ["Transformers", "Datasets", "Tokenizers"],
            "Stars": ["36542", "4512", "3934"],
            "Contributors": ["651", "77", "34"],
            "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
        }
        queries = [
            "What repository has the largest number of stars?",
            "Given that the numbers of stars defines if a repository is active, what repository is the most active?",
            "What is the number of repositories?",
            "What is the average number of stars?",
            "What is the total amount of stars?",
        ]

        results = table_querier(data, queries)

        expected_results = [
            {"answer": "Transformers", "coordinates": [(0, 0)], "cells": ["Transformers"], "aggregator": "NONE"},
            {"answer": "Transformers", "coordinates": [(0, 0)], "cells": ["Transformers"], "aggregator": "NONE"},
            {
                "answer": "COUNT > Transformers, Datasets, Tokenizers",
                "coordinates": [(0, 0), (1, 0), (2, 0)],
                "cells": ["Transformers", "Datasets", "Tokenizers"],
                "aggregator": "COUNT",
            },
            {
                "answer": "AVERAGE > 36542, 4512, 3934",
                "coordinates": [(0, 1), (1, 1), (2, 1)],
                "cells": ["36542", "4512", "3934"],
                "aggregator": "AVERAGE",
            },
            {
                "answer": "SUM > 36542, 4512, 3934",
                "coordinates": [(0, 1), (1, 1), (2, 1)],
                "cells": ["36542", "4512", "3934"],
                "aggregator": "SUM",
            },
        ]
        self.assertListEqual(results, expected_results)

    @slow
    @require_torch
    def test_integration_wtq_pt_fp16(self):
        self.test_integration_wtq_pt(torch_dtype="float16")

    @slow
    @require_tensorflow_probability
    @require_pandas
    def test_integration_wtq_tf(self):
        model_id = "google/tapas-base-finetuned-wtq"
        model = TFAutoModelForTableQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        table_querier = pipeline("table-question-answering", model=model, tokenizer=tokenizer)

        data = {
            "Repository": ["Transformers", "Datasets", "Tokenizers"],
            "Stars": ["36542", "4512", "3934"],
            "Contributors": ["651", "77", "34"],
            "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
        }
        queries = [
            "What repository has the largest number of stars?",
            "Given that the numbers of stars defines if a repository is active, what repository is the most active?",
            "What is the number of repositories?",
            "What is the average number of stars?",
            "What is the total amount of stars?",
        ]

        results = table_querier(data, queries)

        expected_results = [
            {"answer": "Transformers", "coordinates": [(0, 0)], "cells": ["Transformers"], "aggregator": "NONE"},
            {"answer": "Transformers", "coordinates": [(0, 0)], "cells": ["Transformers"], "aggregator": "NONE"},
            {
                "answer": "COUNT > Transformers, Datasets, Tokenizers",
                "coordinates": [(0, 0), (1, 0), (2, 0)],
                "cells": ["Transformers", "Datasets", "Tokenizers"],
                "aggregator": "COUNT",
            },
            {
                "answer": "AVERAGE > 36542, 4512, 3934",
                "coordinates": [(0, 1), (1, 1), (2, 1)],
                "cells": ["36542", "4512", "3934"],
                "aggregator": "AVERAGE",
            },
            {
                "answer": "SUM > 36542, 4512, 3934",
                "coordinates": [(0, 1), (1, 1), (2, 1)],
                "cells": ["36542", "4512", "3934"],
                "aggregator": "SUM",
            },
        ]
        self.assertListEqual(results, expected_results)

    @slow
    @require_torch
    def test_integration_sqa_pt(self, torch_dtype="float32"):
        table_querier = pipeline(
            "table-question-answering",
            model="google/tapas-base-finetuned-sqa",
            tokenizer="google/tapas-base-finetuned-sqa",
            torch_dtype=torch_dtype,
        )
        data = {
            "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            "Age": ["56", "45", "59"],
            "Number of movies": ["87", "53", "69"],
            "Date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
        }
        queries = ["How many movies has George Clooney played in?", "How old is he?", "What's his date of birth?"]
        results = table_querier(data, queries, sequential=True)

        expected_results = [
            {"answer": "69", "coordinates": [(2, 2)], "cells": ["69"]},
            {"answer": "59", "coordinates": [(2, 1)], "cells": ["59"]},
            {"answer": "28 november 1967", "coordinates": [(2, 3)], "cells": ["28 november 1967"]},
        ]
        self.assertListEqual(results, expected_results)

    @slow
    @require_torch
    def test_integration_sqa_pt_fp16(self):
        self.test_integration_sqa_pt(torch_dtype="float16")

    @slow
    @require_tensorflow_probability
    @require_pandas
    def test_integration_sqa_tf(self):
        model_id = "google/tapas-base-finetuned-sqa"
        model = TFAutoModelForTableQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        table_querier = pipeline(
            "table-question-answering",
            model=model,
            tokenizer=tokenizer,
        )
        data = {
            "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            "Age": ["56", "45", "59"],
            "Number of movies": ["87", "53", "69"],
            "Date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
        }
        queries = ["How many movies has George Clooney played in?", "How old is he?", "What's his date of birth?"]
        results = table_querier(data, queries, sequential=True)

        expected_results = [
            {"answer": "69", "coordinates": [(2, 2)], "cells": ["69"]},
            {"answer": "59", "coordinates": [(2, 1)], "cells": ["59"]},
            {"answer": "28 november 1967", "coordinates": [(2, 3)], "cells": ["28 november 1967"]},
        ]
        self.assertListEqual(results, expected_results)

    @slow
    @require_torch
    def test_large_model_pt_tapex(self, torch_dtype="float32"):
        model_id = "microsoft/tapex-large-finetuned-wtq"
        table_querier = pipeline(
            "table-question-answering",
            model=model_id,
            torch_dtype=torch_dtype,
        )
        data = {
            "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            "Age": ["56", "45", "59"],
            "Number of movies": ["87", "53", "69"],
            "Date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
        }
        queries = [
            "How many movies has George Clooney played in?",
            "How old is Mr Clooney ?",
            "What's the date of birth of Leonardo ?",
        ]
        results = table_querier(data, queries, sequential=True)

        expected_results = [
            {"answer": " 69"},
            {"answer": " 59"},
            {"answer": " 10 june 1996"},
        ]
        self.assertListEqual(results, expected_results)
