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
    MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TextGenerationPipeline,
    logging,
    pipeline,
)
from transformers.testing_utils import (
    CaptureLogger,
    is_pipeline_test,
    require_accelerate,
    require_tf,
    require_torch,
    require_torch_accelerator,
    require_torch_or_tf,
    torch_device,
)

from .test_pipelines_common import ANY


@is_pipeline_test
@require_torch_or_tf
class TextGenerationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
    tf_model_mapping = TF_MODEL_FOR_CAUSAL_LM_MAPPING

    @require_torch
    def test_small_model_pt(self):
        text_generator = pipeline(task="text-generation", model="sshleifer/tiny-ctrl", framework="pt")
        # Using `do_sample=False` to force deterministic output
        outputs = text_generator("This is a test", do_sample=False)
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": (
                        "This is a test ☃ ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy oscope."
                        " oscope. FiliFili@@"
                    )
                }
            ],
        )

        outputs = text_generator(["This is a test", "This is a second test"])
        self.assertEqual(
            outputs,
            [
                [
                    {
                        "generated_text": (
                            "This is a test ☃ ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy oscope."
                            " oscope. FiliFili@@"
                        )
                    }
                ],
                [
                    {
                        "generated_text": (
                            "This is a second test ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy"
                            " oscope. oscope. FiliFili@@"
                        )
                    }
                ],
            ],
        )

        outputs = text_generator("This is a test", do_sample=True, num_return_sequences=2, return_tensors=True)
        self.assertEqual(
            outputs,
            [
                {"generated_token_ids": ANY(list)},
                {"generated_token_ids": ANY(list)},
            ],
        )

        ## -- test tokenizer_kwargs
        test_str = "testing tokenizer kwargs. using truncation must result in a different generation."
        input_len = len(text_generator.tokenizer(test_str)["input_ids"])
        output_str, output_str_with_truncation = (
            text_generator(test_str, do_sample=False, return_full_text=False, min_new_tokens=1)[0]["generated_text"],
            text_generator(
                test_str,
                do_sample=False,
                return_full_text=False,
                min_new_tokens=1,
                truncation=True,
                max_length=input_len + 1,
            )[0]["generated_text"],
        )
        assert output_str != output_str_with_truncation  # results must be different because one had truncation

        ## -- test kwargs for preprocess_params
        outputs = text_generator("This is a test", do_sample=False, add_special_tokens=False, padding=False)
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": (
                        "This is a test ☃ ☃ segmental segmental segmental 议议eski eski flutter flutter Lacy oscope."
                        " oscope. FiliFili@@"
                    )
                }
            ],
        )

        # -- what is the point of this test? padding is hardcoded False in the pipeline anyway
        text_generator.tokenizer.pad_token_id = text_generator.model.config.eos_token_id
        text_generator.tokenizer.pad_token = "<pad>"
        outputs = text_generator(
            ["This is a test", "This is a second test"],
            do_sample=True,
            num_return_sequences=2,
            batch_size=2,
            return_tensors=True,
        )
        self.assertEqual(
            outputs,
            [
                [
                    {"generated_token_ids": ANY(list)},
                    {"generated_token_ids": ANY(list)},
                ],
                [
                    {"generated_token_ids": ANY(list)},
                    {"generated_token_ids": ANY(list)},
                ],
            ],
        )

    @require_torch
    def test_small_chat_model_pt(self):
        text_generator = pipeline(
            task="text-generation", model="hf-internal-testing/tiny-gpt2-with-chatml-template", framework="pt"
        )
        # Using `do_sample=False` to force deterministic output
        chat1 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a test"},
        ]
        chat2 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a second test"},
        ]
        outputs = text_generator(chat1, do_sample=False, max_new_tokens=10)
        expected_chat1 = chat1 + [
            {
                "role": "assistant",
                "content": " factors factors factors factors factors factors factors factors factors factors",
            }
        ]
        self.assertEqual(
            outputs,
            [
                {"generated_text": expected_chat1},
            ],
        )

        outputs = text_generator([chat1, chat2], do_sample=False, max_new_tokens=10)
        expected_chat2 = chat2 + [
            {
                "role": "assistant",
                "content": " stairs stairs stairs stairs stairs stairs stairs stairs stairs stairs",
            }
        ]

        self.assertEqual(
            outputs,
            [
                [{"generated_text": expected_chat1}],
                [{"generated_text": expected_chat2}],
            ],
        )

    @require_torch
    def test_small_chat_model_continue_final_message(self):
        # Here we check that passing a chat that ends in an assistant message is handled correctly
        # by continuing the final message rather than starting a new one
        text_generator = pipeline(
            task="text-generation", model="hf-internal-testing/tiny-gpt2-with-chatml-template", framework="pt"
        )
        # Using `do_sample=False` to force deterministic output
        chat1 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a test"},
            {"role": "assistant", "content": "This is"},
        ]
        outputs = text_generator(chat1, do_sample=False, max_new_tokens=10)

        # Assert that we continued the last message and there isn't a sneaky <|im_end|>
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": [
                        {"role": "system", "content": "This is a system message."},
                        {"role": "user", "content": "This is a test"},
                        {
                            "role": "assistant",
                            "content": "This is stairs stairs stairs stairs stairs stairs stairs stairs stairs stairs",
                        },
                    ]
                }
            ],
        )

    @require_torch
    def test_small_chat_model_continue_final_message_override(self):
        # Here we check that passing a chat that ends in an assistant message is handled correctly
        # by continuing the final message rather than starting a new one
        text_generator = pipeline(
            task="text-generation", model="hf-internal-testing/tiny-gpt2-with-chatml-template", framework="pt"
        )
        # Using `do_sample=False` to force deterministic output
        chat1 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a test"},
        ]
        outputs = text_generator(chat1, do_sample=False, max_new_tokens=10, continue_final_message=True)

        # Assert that we continued the last message and there isn't a sneaky <|im_end|>
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": [
                        {"role": "system", "content": "This is a system message."},
                        {
                            "role": "user",
                            "content": "This is a test stairs stairs stairs stairs stairs stairs stairs stairs stairs stairs",
                        },
                    ]
                }
            ],
        )

    @require_torch
    def test_small_chat_model_with_dataset_pt(self):
        from torch.utils.data import Dataset

        from transformers.pipelines.pt_utils import KeyDataset

        class MyDataset(Dataset):
            data = [
                [
                    {"role": "system", "content": "This is a system message."},
                    {"role": "user", "content": "This is a test"},
                ],
            ]

            def __len__(self):
                return 1

            def __getitem__(self, i):
                return {"text": self.data[i]}

        text_generator = pipeline(
            task="text-generation", model="hf-internal-testing/tiny-gpt2-with-chatml-template", framework="pt"
        )

        dataset = MyDataset()
        key_dataset = KeyDataset(dataset, "text")

        for outputs in text_generator(key_dataset, do_sample=False, max_new_tokens=10):
            expected_chat = dataset.data[0] + [
                {
                    "role": "assistant",
                    "content": " factors factors factors factors factors factors factors factors factors factors",
                }
            ]
            self.assertEqual(
                outputs,
                [
                    {"generated_text": expected_chat},
                ],
            )

    @require_torch
    def test_small_chat_model_with_iterator_pt(self):
        from transformers.pipelines.pt_utils import PipelineIterator

        text_generator = pipeline(
            task="text-generation", model="hf-internal-testing/tiny-gpt2-with-chatml-template", framework="pt"
        )

        # Using `do_sample=False` to force deterministic output
        chat1 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a test"},
        ]
        chat2 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a second test"},
        ]
        expected_chat1 = chat1 + [
            {
                "role": "assistant",
                "content": " factors factors factors factors factors factors factors factors factors factors",
            }
        ]
        expected_chat2 = chat2 + [
            {
                "role": "assistant",
                "content": " stairs stairs stairs stairs stairs stairs stairs stairs stairs stairs",
            }
        ]

        def data():
            yield from [chat1, chat2]

        outputs = text_generator(data(), do_sample=False, max_new_tokens=10)
        assert isinstance(outputs, PipelineIterator)
        outputs = list(outputs)
        self.assertEqual(
            outputs,
            [
                [{"generated_text": expected_chat1}],
                [{"generated_text": expected_chat2}],
            ],
        )

    @require_tf
    def test_small_model_tf(self):
        text_generator = pipeline(task="text-generation", model="sshleifer/tiny-ctrl", framework="tf")

        # Using `do_sample=False` to force deterministic output
        outputs = text_generator("This is a test", do_sample=False)
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": (
                        "This is a test FeyFeyFey(Croatis.), s.), Cannes Cannes Cannes 閲閲Cannes Cannes Cannes 攵"
                        " please,"
                    )
                }
            ],
        )

        outputs = text_generator(["This is a test", "This is a second test"], do_sample=False)
        self.assertEqual(
            outputs,
            [
                [
                    {
                        "generated_text": (
                            "This is a test FeyFeyFey(Croatis.), s.), Cannes Cannes Cannes 閲閲Cannes Cannes Cannes 攵"
                            " please,"
                        )
                    }
                ],
                [
                    {
                        "generated_text": (
                            "This is a second test Chieftain Chieftain prefecture prefecture prefecture Cannes Cannes"
                            " Cannes 閲閲Cannes Cannes Cannes 攵 please,"
                        )
                    }
                ],
            ],
        )

    @require_tf
    def test_small_chat_model_tf(self):
        text_generator = pipeline(
            task="text-generation", model="hf-internal-testing/tiny-gpt2-with-chatml-template", framework="tf"
        )
        # Using `do_sample=False` to force deterministic output
        chat1 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a test"},
        ]
        chat2 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a second test"},
        ]
        outputs = text_generator(chat1, do_sample=False, max_new_tokens=10)
        expected_chat1 = chat1 + [
            {
                "role": "assistant",
                "content": " factors factors factors factors factors factors factors factors factors factors",
            }
        ]
        self.assertEqual(
            outputs,
            [
                {"generated_text": expected_chat1},
            ],
        )

        outputs = text_generator([chat1, chat2], do_sample=False, max_new_tokens=10)
        expected_chat2 = chat2 + [
            {
                "role": "assistant",
                "content": " stairs stairs stairs stairs stairs stairs stairs stairs stairs stairs",
            }
        ]

        self.assertEqual(
            outputs,
            [
                [{"generated_text": expected_chat1}],
                [{"generated_text": expected_chat2}],
            ],
        )

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        text_generator = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
        )
        return text_generator, ["This is a test", "Another test"]

    def test_stop_sequence_stopping_criteria(self):
        prompt = """Hello I believe in"""
        text_generator = pipeline("text-generation", model="hf-internal-testing/tiny-random-gpt2")
        output = text_generator(prompt)
        self.assertEqual(
            output,
            [{"generated_text": "Hello I believe in fe fe fe fe fe fe fe fe fe fe fe fe"}],
        )

        output = text_generator(prompt, stop_sequence=" fe")
        self.assertEqual(output, [{"generated_text": "Hello I believe in fe"}])

    def run_pipeline_test(self, text_generator, _):
        model = text_generator.model
        tokenizer = text_generator.tokenizer

        outputs = text_generator("This is a test")
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertTrue(outputs[0]["generated_text"].startswith("This is a test"))

        outputs = text_generator("This is a test", return_full_text=False)
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertNotIn("This is a test", outputs[0]["generated_text"])

        text_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        outputs = text_generator("This is a test")
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertNotIn("This is a test", outputs[0]["generated_text"])

        outputs = text_generator("This is a test", return_full_text=True)
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertTrue(outputs[0]["generated_text"].startswith("This is a test"))

        outputs = text_generator(["This is great !", "Something else"], num_return_sequences=2, do_sample=True)
        self.assertEqual(
            outputs,
            [
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
            ],
        )

        if text_generator.tokenizer.pad_token is not None:
            outputs = text_generator(
                ["This is great !", "Something else"], num_return_sequences=2, batch_size=2, do_sample=True
            )
            self.assertEqual(
                outputs,
                [
                    [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                    [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                ],
            )

        with self.assertRaises(ValueError):
            outputs = text_generator("test", return_full_text=True, return_text=True)
        with self.assertRaises(ValueError):
            outputs = text_generator("test", return_full_text=True, return_tensors=True)
        with self.assertRaises(ValueError):
            outputs = text_generator("test", return_text=True, return_tensors=True)

        # Empty prompt is slighly special
        # it requires BOS token to exist.
        # Special case for Pegasus which will always append EOS so will
        # work even without BOS.
        if (
            text_generator.tokenizer.bos_token_id is not None
            or "Pegasus" in tokenizer.__class__.__name__
            or "Git" in model.__class__.__name__
        ):
            outputs = text_generator("")
            self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        else:
            with self.assertRaises((ValueError, AssertionError)):
                outputs = text_generator("", add_special_tokens=False)

        if text_generator.framework == "tf":
            # TF generation does not support max_new_tokens, and it's impossible
            # to control long generation with only max_length without
            # fancy calculation, dismissing tests for now.
            self.skipTest(reason="TF generation does not support max_new_tokens")
        # We don't care about infinite range models.
        # They already work.
        # Skip this test for XGLM, since it uses sinusoidal positional embeddings which are resized on-the-fly.
        EXTRA_MODELS_CAN_HANDLE_LONG_INPUTS = [
            "RwkvForCausalLM",
            "XGLMForCausalLM",
            "GPTNeoXForCausalLM",
            "GPTNeoXJapaneseForCausalLM",
            "FuyuForCausalLM",
            "LlamaForCausalLM",
        ]
        if (
            tokenizer.model_max_length < 10000
            and text_generator.model.__class__.__name__ not in EXTRA_MODELS_CAN_HANDLE_LONG_INPUTS
        ):
            # Handling of large generations
            if str(text_generator.device) == "cpu":
                with self.assertRaises((RuntimeError, IndexError, ValueError, AssertionError)):
                    text_generator("This is a test" * 500, max_new_tokens=20)

            outputs = text_generator("This is a test" * 500, handle_long_generation="hole", max_new_tokens=20)
            # Hole strategy cannot work
            if str(text_generator.device) == "cpu":
                with self.assertRaises(ValueError):
                    text_generator(
                        "This is a test" * 500,
                        handle_long_generation="hole",
                        max_new_tokens=tokenizer.model_max_length + 10,
                    )

    @require_torch
    @require_accelerate
    @require_torch_accelerator
    def test_small_model_pt_bloom_accelerate(self):
        import torch

        # Classic `model_kwargs`
        pipe = pipeline(
            model="hf-internal-testing/tiny-random-bloom",
            model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16},
        )
        self.assertEqual(pipe.model.lm_head.weight.dtype, torch.bfloat16)
        out = pipe("This is a test")
        self.assertEqual(
            out,
            [
                {
                    "generated_text": (
                        "This is a test test test test test test test test test test test test test test test test"
                        " test"
                    )
                }
            ],
        )

        # Upgraded those two to real pipeline arguments (they just get sent for the model as they're unlikely to mean anything else.)
        pipe = pipeline(model="hf-internal-testing/tiny-random-bloom", device_map="auto", torch_dtype=torch.bfloat16)
        self.assertEqual(pipe.model.lm_head.weight.dtype, torch.bfloat16)
        out = pipe("This is a test")
        self.assertEqual(
            out,
            [
                {
                    "generated_text": (
                        "This is a test test test test test test test test test test test test test test test test"
                        " test"
                    )
                }
            ],
        )

        # torch_dtype will be automatically set to float32 if not provided - check: https://github.com/huggingface/transformers/pull/20602
        pipe = pipeline(model="hf-internal-testing/tiny-random-bloom", device_map="auto")
        self.assertEqual(pipe.model.lm_head.weight.dtype, torch.float32)
        out = pipe("This is a test")
        self.assertEqual(
            out,
            [
                {
                    "generated_text": (
                        "This is a test test test test test test test test test test test test test test test test"
                        " test"
                    )
                }
            ],
        )

    @require_torch
    @require_torch_accelerator
    def test_small_model_fp16(self):
        import torch

        pipe = pipeline(
            model="hf-internal-testing/tiny-random-bloom",
            device=torch_device,
            torch_dtype=torch.float16,
        )
        pipe("This is a test")

    @require_torch
    @require_accelerate
    @require_torch_accelerator
    def test_pipeline_accelerate_top_p(self):
        import torch

        pipe = pipeline(
            model="hf-internal-testing/tiny-random-bloom", device_map=torch_device, torch_dtype=torch.float16
        )
        pipe("This is a test", do_sample=True, top_p=0.5)

    def test_pipeline_length_setting_warning(self):
        prompt = """Hello world"""
        text_generator = pipeline("text-generation", model="hf-internal-testing/tiny-random-gpt2")
        if text_generator.model.framework == "tf":
            logger = logging.get_logger("transformers.generation.tf_utils")
        else:
            logger = logging.get_logger("transformers.generation.utils")
        logger_msg = "Both `max_new_tokens`"  # The beggining of the message to be checked in this test

        # Both are set by the user -> log warning
        with CaptureLogger(logger) as cl:
            _ = text_generator(prompt, max_length=10, max_new_tokens=1)
        self.assertIn(logger_msg, cl.out)

        # The user only sets one -> no warning
        with CaptureLogger(logger) as cl:
            _ = text_generator(prompt, max_new_tokens=1)
        self.assertNotIn(logger_msg, cl.out)

        with CaptureLogger(logger) as cl:
            _ = text_generator(prompt, max_length=10)
        self.assertNotIn(logger_msg, cl.out)

    @require_torch
    def test_pipeline_assisted_generation(self):
        """Tests that we can run assisted generation in the pipeline"""
        model = "hf-internal-testing/tiny-random-MistralForCausalLM"
        pipe = pipeline("text-generation", model=model, assistant_model=model)

        # We can run the pipeline
        prompt = "Hello world"
        _ = pipe(prompt)

        # It is running assisted generation under the hood (e.g. flags incompatible with assisted gen will crash)
        with self.assertRaises(ValueError):
            _ = pipe(prompt, generate_kwargs={"num_beams": 2})
