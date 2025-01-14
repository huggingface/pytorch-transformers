# coding=utf-8
# Copyright 2018 HuggingFace Inc..
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
import logging
import os
import sys
from unittest.mock import patch

from transformers import ViTMAEForPreTraining, Wav2Vec2ForPreTraining
from transformers.testing_utils import (
    CaptureLogger,
    TestCasePlus,
    backend_device_count,
    is_torch_fp16_available_on_device,
    slow,
    torch_device,
)


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "text-generation",
        "text-classification",
        "token-classification",
        "language-modeling",
        "multiple-choice",
        "question-answering",
        "summarization",
        "translation",
        "image-classification",
        "speech-recognition",
        "audio-classification",
        "speech-pretraining",
        "image-pretraining",
        "semantic-segmentation",
        "object-detection",
        "instance-segmentation",
    ]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import run_audio_classification
    import run_clm
    import run_generation
    import run_glue
    import run_image_classification
    import run_instance_segmentation
    import run_mae
    import run_mlm
    import run_ner
    import run_object_detection
    import run_qa as run_squad
    import run_semantic_segmentation
    import run_seq2seq_qa as run_squad_seq2seq
    import run_speech_recognition_ctc
    import run_speech_recognition_ctc_adapter
    import run_speech_recognition_seq2seq
    import run_summarization
    import run_swag
    import run_translation
    import run_wav2vec2_pretraining_no_trainer


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_results(output_dir):
    results = {}
    path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"can't find {path}")
    return results


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ExamplesTests(TestCasePlus):
    def test_run_glue(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_glue.py
            --model_name_or_path distilbert/distilbert-base-uncased
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv
            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv
            --do_train
            --do_eval
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=1e-4
            --max_steps=10
            --warmup_steps=2
            --seed=42
            --max_seq_length=128
            """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_glue.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)

    def test_run_clm(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_clm.py
            --model_name_or_path distilbert/distilgpt2
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --do_train
            --do_eval
            --block_size 128
            --per_device_train_batch_size 5
            --per_device_eval_batch_size 5
            --num_train_epochs 2
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if backend_device_count(torch_device) > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        if torch_device == "cpu":
            testargs.append("--use_cpu")

        with patch.object(sys, "argv", testargs):
            run_clm.main()
            result = get_results(tmp_dir)
            self.assertLess(result["perplexity"], 100)

    def test_run_clm_config_overrides(self):
        # test that config_overrides works, despite the misleading dumps of default un-updated
        # config via tokenizer

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_clm.py
            --model_type gpt2
            --tokenizer_name openai-community/gpt2
            --train_file ./tests/fixtures/sample_text.txt
            --output_dir {tmp_dir}
            --config_overrides n_embd=10,n_head=2
            """.split()

        if torch_device == "cpu":
            testargs.append("--use_cpu")

        logger = run_clm.logger
        with patch.object(sys, "argv", testargs):
            with CaptureLogger(logger) as cl:
                run_clm.main()

        self.assertIn('"n_embd": 10', cl.out)
        self.assertIn('"n_head": 2', cl.out)

    def test_run_mlm(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_mlm.py
            --model_name_or_path distilbert/distilroberta-base
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --prediction_loss_only
            --num_train_epochs=1
        """.split()

        if torch_device == "cpu":
            testargs.append("--use_cpu")

        with patch.object(sys, "argv", testargs):
            run_mlm.main()
            result = get_results(tmp_dir)
            self.assertLess(result["perplexity"], 42)

    def test_run_ner(self):
        # with so little data distributed training needs more epochs to get the score on par with 0/1 gpu
        epochs = 7 if backend_device_count(torch_device) > 1 else 2

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_ner.py
            --model_name_or_path google-bert/bert-base-uncased
            --train_file tests/fixtures/tests_samples/conll/sample.json
            --validation_file tests/fixtures/tests_samples/conll/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --warmup_steps=2
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=2
            --num_train_epochs={epochs}
            --seed 7
        """.split()

        if torch_device == "cpu":
            testargs.append("--use_cpu")

        with patch.object(sys, "argv", testargs):
            run_ner.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)
            self.assertLess(result["eval_loss"], 0.5)

    def test_run_squad(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_qa.py
            --model_name_or_path google-bert/bert-base-uncased
            --version_2_with_negative
            --train_file tests/fixtures/tests_samples/SQUAD/sample.json
            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=10
            --warmup_steps=2
            --do_train
            --do_eval
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
        """.split()

        with patch.object(sys, "argv", testargs):
            run_squad.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_f1"], 30)
            self.assertGreaterEqual(result["eval_exact"], 30)

    def test_run_squad_seq2seq(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_seq2seq_qa.py
            --model_name_or_path google-t5/t5-small
            --context_column context
            --question_column question
            --answer_column answers
            --version_2_with_negative
            --train_file tests/fixtures/tests_samples/SQUAD/sample.json
            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=10
            --warmup_steps=2
            --do_train
            --do_eval
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --predict_with_generate
        """.split()

        with patch.object(sys, "argv", testargs):
            run_squad_seq2seq.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_f1"], 30)
            self.assertGreaterEqual(result["eval_exact"], 30)

    def test_run_swag(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_swag.py
            --model_name_or_path google-bert/bert-base-uncased
            --train_file tests/fixtures/tests_samples/swag/sample.json
            --validation_file tests/fixtures/tests_samples/swag/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=20
            --warmup_steps=2
            --do_train
            --do_eval
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
        """.split()

        with patch.object(sys, "argv", testargs):
            run_swag.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.8)

    def test_generation(self):
        testargs = ["run_generation.py", "--prompt=Hello", "--length=10", "--seed=42"]

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        model_type, model_name = (
            "--model_type=gpt2",
            "--model_name_or_path=sshleifer/tiny-gpt2",
        )
        with patch.object(sys, "argv", testargs + [model_type, model_name]):
            result = run_generation.main()
            self.assertGreaterEqual(len(result[0]), 10)

    @slow
    def test_run_summarization(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_summarization.py
            --model_name_or_path google-t5/t5-small
            --train_file tests/fixtures/tests_samples/xsum/sample.json
            --validation_file tests/fixtures/tests_samples/xsum/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=50
            --warmup_steps=8
            --do_train
            --do_eval
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --predict_with_generate
        """.split()

        with patch.object(sys, "argv", testargs):
            run_summarization.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_rouge1"], 10)
            self.assertGreaterEqual(result["eval_rouge2"], 2)
            self.assertGreaterEqual(result["eval_rougeL"], 7)
            self.assertGreaterEqual(result["eval_rougeLsum"], 7)

    @slow
    def test_run_translation(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_translation.py
            --model_name_or_path sshleifer/student_marian_en_ro_6_1
            --source_lang en
            --target_lang ro
            --train_file tests/fixtures/tests_samples/wmt16/sample.json
            --validation_file tests/fixtures/tests_samples/wmt16/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=50
            --warmup_steps=8
            --do_train
            --do_eval
            --learning_rate=3e-3
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --predict_with_generate
            --source_lang en_XX
            --target_lang ro_RO
            --max_source_length 512
        """.split()

        with patch.object(sys, "argv", testargs):
            run_translation.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_bleu"], 30)

    def test_run_image_classification(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_image_classification.py
            --output_dir {tmp_dir}
            --model_name_or_path google/vit-base-patch16-224-in21k
            --dataset_name hf-internal-testing/cats_vs_dogs_sample
            --trust_remote_code
            --do_train
            --do_eval
            --learning_rate 1e-4
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --remove_unused_columns False
            --overwrite_output_dir True
            --dataloader_num_workers 16
            --metric_for_best_model accuracy
            --max_steps 10
            --train_val_split 0.1
            --seed 42
            --label_column_name labels
        """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_image_classification.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.8)

    def test_run_speech_recognition_ctc(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_speech_recognition_ctc.py
            --output_dir {tmp_dir}
            --model_name_or_path hf-internal-testing/tiny-random-wav2vec2
            --dataset_name hf-internal-testing/librispeech_asr_dummy
            --dataset_config_name clean
            --train_split_name validation
            --eval_split_name validation
            --trust_remote_code
            --do_train
            --do_eval
            --learning_rate 1e-4
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --remove_unused_columns False
            --overwrite_output_dir True
            --preprocessing_num_workers 16
            --max_steps 10
            --seed 42
        """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_speech_recognition_ctc.main()
            result = get_results(tmp_dir)
            self.assertLess(result["eval_loss"], result["train_loss"])

    def test_run_speech_recognition_ctc_adapter(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_speech_recognition_ctc_adapter.py
            --output_dir {tmp_dir}
            --model_name_or_path hf-internal-testing/tiny-random-wav2vec2
            --dataset_name hf-internal-testing/librispeech_asr_dummy
            --dataset_config_name clean
            --train_split_name validation
            --eval_split_name validation
            --trust_remote_code
            --do_train
            --do_eval
            --learning_rate 1e-4
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --remove_unused_columns False
            --overwrite_output_dir True
            --preprocessing_num_workers 16
            --max_steps 10
            --target_language tur
            --seed 42
        """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_speech_recognition_ctc_adapter.main()
            result = get_results(tmp_dir)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "./adapter.tur.safetensors")))
            self.assertLess(result["eval_loss"], result["train_loss"])

    def test_run_speech_recognition_seq2seq(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_speech_recognition_seq2seq.py
            --output_dir {tmp_dir}
            --model_name_or_path hf-internal-testing/tiny-random-speech-encoder-decoder
            --dataset_name hf-internal-testing/librispeech_asr_dummy
            --dataset_config_name clean
            --train_split_name validation
            --eval_split_name validation
            --trust_remote_code
            --do_train
            --do_eval
            --learning_rate 1e-4
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 4
            --remove_unused_columns False
            --overwrite_output_dir True
            --preprocessing_num_workers 16
            --max_steps 10
            --seed 42
        """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_speech_recognition_seq2seq.main()
            result = get_results(tmp_dir)
            self.assertLess(result["eval_loss"], result["train_loss"])

    def test_run_audio_classification(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_audio_classification.py
            --output_dir {tmp_dir}
            --model_name_or_path hf-internal-testing/tiny-random-wav2vec2
            --dataset_name anton-l/superb_demo
            --trust_remote_code
            --dataset_config_name ks
            --train_split_name test
            --eval_split_name test
            --audio_column_name audio
            --label_column_name label
            --do_train
            --do_eval
            --learning_rate 1e-4
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --remove_unused_columns False
            --overwrite_output_dir True
            --num_train_epochs 10
            --max_steps 50
            --seed 42
        """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_audio_classification.main()
            result = get_results(tmp_dir)
            self.assertLess(result["eval_loss"], result["train_loss"])

    def test_run_wav2vec2_pretraining(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_wav2vec2_pretraining_no_trainer.py
            --output_dir {tmp_dir}
            --model_name_or_path hf-internal-testing/tiny-random-wav2vec2
            --dataset_name hf-internal-testing/librispeech_asr_dummy
            --dataset_config_names clean
            --dataset_split_names validation
            --trust_remote_code
            --learning_rate 1e-4
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --preprocessing_num_workers 16
            --max_train_steps 2
            --validation_split_percentage 5
            --seed 42
        """.split()

        with patch.object(sys, "argv", testargs):
            run_wav2vec2_pretraining_no_trainer.main()
            model = Wav2Vec2ForPreTraining.from_pretrained(tmp_dir)
            self.assertIsNotNone(model)

    def test_run_vit_mae_pretraining(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_mae.py
            --output_dir {tmp_dir}
            --dataset_name hf-internal-testing/cats_vs_dogs_sample
            --trust_remote_code
            --do_train
            --do_eval
            --learning_rate 1e-4
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --remove_unused_columns False
            --overwrite_output_dir True
            --dataloader_num_workers 16
            --metric_for_best_model accuracy
            --max_steps 10
            --train_val_split 0.1
            --seed 42
        """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_mae.main()
            model = ViTMAEForPreTraining.from_pretrained(tmp_dir)
            self.assertIsNotNone(model)

    def test_run_semantic_segmentation(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_semantic_segmentation.py
            --output_dir {tmp_dir}
            --dataset_name huggingface/semantic-segmentation-test-sample
            --do_train
            --do_eval
            --remove_unused_columns False
            --overwrite_output_dir True
            --max_steps 10
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --seed 32
        """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_semantic_segmentation.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_overall_accuracy"], 0.1)

    @patch.dict(os.environ, {"WANDB_DISABLED": "true"})
    def test_run_object_detection(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_object_detection.py
            --model_name_or_path qubvel-hf/detr-resnet-50-finetuned-10k-cppe5
            --output_dir {tmp_dir}
            --dataset_name qubvel-hf/cppe-5-sample
            --do_train
            --do_eval
            --remove_unused_columns False
            --overwrite_output_dir True
            --eval_do_concat_batches False
            --batch_eval_metrics True
            --max_steps 10
            --learning_rate=1e-6
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --seed 32
        """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_object_detection.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["test_map"], 0.1)

    @patch.dict(os.environ, {"WANDB_DISABLED": "true"})
    def test_run_instance_segmentation(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_instance_segmentation.py
            --model_name_or_path qubvel-hf/finetune-instance-segmentation-ade20k-mini-mask2former
            --output_dir {tmp_dir}
            --dataset_name qubvel-hf/ade20k-nano
            --do_reduce_labels
            --image_height 256
            --image_width 256
            --do_train
            --num_train_epochs 1
            --learning_rate 1e-5
            --lr_scheduler_type constant
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --do_eval
            --evaluation_strategy epoch
            --seed 32
        """.split()

        if is_torch_fp16_available_on_device(torch_device):
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_instance_segmentation.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["test_map"], 0.1)
