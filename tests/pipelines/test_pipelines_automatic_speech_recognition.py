# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import time
import unittest

import numpy as np
import pytest
from datasets import Audio, load_dataset
from huggingface_hub import AutomaticSpeechRecognitionOutput, hf_hub_download, snapshot_download

from transformers import (
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    Speech2TextForConditionalGeneration,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
)
from transformers.pipelines import AutomaticSpeechRecognitionPipeline, pipeline
from transformers.pipelines.audio_utils import chunk_bytes_iter, ffmpeg_microphone_live
from transformers.pipelines.automatic_speech_recognition import _find_timestamp_sequence, chunk_iter
from transformers.testing_utils import (
    compare_pipeline_output_to_hub_spec,
    is_pipeline_test,
    is_torch_available,
    nested_simplify,
    require_pyctcdecode,
    require_tf,
    require_torch,
    require_torch_accelerator,
    require_torchaudio,
    slow,
    torch_device,
)

from .test_pipelines_common import ANY


if is_torch_available():
    import torch


# We can't use this mixin because it assumes TF support.
# from .test_pipelines_common import CustomInputPipelineCommonMixin


@is_pipeline_test
class AutomaticSpeechRecognitionPipelineTests(unittest.TestCase):
    model_mapping = dict(
        (list(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.items()) if MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING else [])
        + (MODEL_FOR_CTC_MAPPING.items() if MODEL_FOR_CTC_MAPPING else [])
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
        if tokenizer is None:
            # Side effect of no Fast Tokenizer class for these model, so skipping
            # But the slow tokenizer test should still run as they're quite small
            self.skipTest(reason="No tokenizer available")

        speech_recognizer = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
        )

        # test with a raw waveform
        audio = np.zeros((34000,))
        audio2 = np.zeros((14000,))
        return speech_recognizer, [audio, audio2]

    def run_pipeline_test(self, speech_recognizer, examples):
        audio = np.zeros((34000,))
        outputs = speech_recognizer(audio)
        self.assertEqual(outputs, {"text": ANY(str)})

        compare_pipeline_output_to_hub_spec(outputs, AutomaticSpeechRecognitionOutput)

        # Striding
        audio = {"raw": audio, "stride": (0, 4000), "sampling_rate": speech_recognizer.feature_extractor.sampling_rate}
        if speech_recognizer.type == "ctc":
            outputs = speech_recognizer(audio)
            self.assertEqual(outputs, {"text": ANY(str)})
        elif "Whisper" in speech_recognizer.model.__class__.__name__:
            outputs = speech_recognizer(audio)
            self.assertEqual(outputs, {"text": ANY(str)})
        else:
            # Non CTC models cannot use striding.
            with self.assertRaises(ValueError):
                outputs = speech_recognizer(audio)

        # Timestamps
        audio = np.zeros((34000,))
        if speech_recognizer.type == "ctc":
            outputs = speech_recognizer(audio, return_timestamps="char")
            self.assertIsInstance(outputs["chunks"], list)
            n = len(outputs["chunks"])
            self.assertEqual(
                outputs,
                {
                    "text": ANY(str),
                    "chunks": [{"text": ANY(str), "timestamp": (ANY(float), ANY(float))} for i in range(n)],
                },
            )

            outputs = speech_recognizer(audio, return_timestamps="word")
            self.assertIsInstance(outputs["chunks"], list)
            n = len(outputs["chunks"])
            self.assertEqual(
                outputs,
                {
                    "text": ANY(str),
                    "chunks": [{"text": ANY(str), "timestamp": (ANY(float), ANY(float))} for i in range(n)],
                },
            )
        elif "Whisper" in speech_recognizer.model.__class__.__name__:
            outputs = speech_recognizer(audio, return_timestamps=True)
            self.assertIsInstance(outputs["chunks"], list)
            nb_chunks = len(outputs["chunks"])
            self.assertGreater(nb_chunks, 0)
            self.assertEqual(
                outputs,
                {
                    "text": ANY(str),
                    "chunks": [{"text": ANY(str), "timestamp": (ANY(float), ANY(float))} for i in range(nb_chunks)],
                },
            )
        else:
            # Non CTC models cannot use return_timestamps
            with self.assertRaisesRegex(
                ValueError, "^We cannot return_timestamps yet on non-CTC models apart from Whisper!$"
            ):
                outputs = speech_recognizer(audio, return_timestamps="char")

    @require_torch
    @slow
    def test_pt_defaults(self):
        pipeline("automatic-speech-recognition", framework="pt")

    @require_torch
    def test_small_model_pt(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/s2t-small-mustc-en-fr-st",
            tokenizer="facebook/s2t-small-mustc-en-fr-st",
            framework="pt",
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform)
        self.assertEqual(output, {"text": "(Applaudissements)"})
        output = speech_recognizer(waveform, chunk_length_s=10)
        self.assertEqual(output, {"text": "(Applaudissements)"})

        # Non CTC models cannot use return_timestamps
        with self.assertRaisesRegex(
            ValueError, "^We cannot return_timestamps yet on non-CTC models apart from Whisper!$"
        ):
            _ = speech_recognizer(waveform, return_timestamps="char")

    @require_torch
    def test_small_model_pt_fp16(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/s2t-small-mustc-en-fr-st",
            tokenizer="facebook/s2t-small-mustc-en-fr-st",
            framework="pt",
            torch_dtype=torch.float16,
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform)
        self.assertEqual(output, {"text": "(Applaudissements)"})
        output = speech_recognizer(waveform, chunk_length_s=10)
        self.assertEqual(output, {"text": "(Applaudissements)"})

        # Non CTC models cannot use return_timestamps
        with self.assertRaisesRegex(
            ValueError, "^We cannot return_timestamps yet on non-CTC models apart from Whisper!$"
        ):
            _ = speech_recognizer(waveform, return_timestamps="char")

    @require_torch
    def test_small_model_pt_bf16(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/s2t-small-mustc-en-fr-st",
            tokenizer="facebook/s2t-small-mustc-en-fr-st",
            framework="pt",
            torch_dtype=torch.bfloat16,
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform)
        self.assertEqual(output, {"text": "(Applaudissements)"})
        output = speech_recognizer(waveform, chunk_length_s=10)
        self.assertEqual(output, {"text": "(Applaudissements)"})

        # Non CTC models cannot use return_timestamps
        with self.assertRaisesRegex(
            ValueError, "^We cannot return_timestamps yet on non-CTC models apart from Whisper!$"
        ):
            _ = speech_recognizer(waveform, return_timestamps="char")

    @slow
    @require_torch_accelerator
    def test_whisper_fp16(self):
        speech_recognizer = pipeline(
            model="openai/whisper-base",
            device=torch_device,
            torch_dtype=torch.float16,
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        speech_recognizer(waveform)

    @require_torch
    def test_small_model_pt_seq2seq(self):
        speech_recognizer = pipeline(
            model="hf-internal-testing/tiny-random-speech-encoder-decoder",
            framework="pt",
        )

        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform)
        self.assertEqual(output, {"text": "あл ش 湯 清 ه ܬ া लᆨしث ल eか u w 全 u"})

    @require_torch
    def test_small_model_pt_seq2seq_gen_kwargs(self):
        speech_recognizer = pipeline(
            model="hf-internal-testing/tiny-random-speech-encoder-decoder",
            framework="pt",
        )

        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform, max_new_tokens=10, generate_kwargs={"num_beams": 2})
        self.assertEqual(output, {"text": "あл † γ ت ב オ 束 泣 足"})

    @slow
    @require_torch
    @require_pyctcdecode
    def test_large_model_pt_with_lm(self):
        dataset = load_dataset("Narsil/asr_dummy", streaming=True, trust_remote_code=True)
        third_item = next(iter(dataset["test"].skip(3)))
        filename = third_item["file"]

        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm",
            framework="pt",
        )
        self.assertEqual(speech_recognizer.type, "ctc_with_lm")

        output = speech_recognizer(filename)
        self.assertEqual(
            output,
            {"text": "y en las ramas medio sumergidas revoloteaban algunos pájaros de quimérico y legendario plumaje"},
        )

        # Override back to pure CTC
        speech_recognizer.type = "ctc"
        output = speech_recognizer(filename)
        # plumajre != plumaje
        self.assertEqual(
            output,
            {
                "text": (
                    "y en las ramas medio sumergidas revoloteaban algunos pájaros de quimérico y legendario plumajre"
                )
            },
        )

        speech_recognizer.type = "ctc_with_lm"
        # Simple test with CTC with LM, chunking + timestamps
        output = speech_recognizer(filename, chunk_length_s=2.0, return_timestamps="word")
        self.assertEqual(
            output,
            {
                "text": (
                    "y en las ramas medio sumergidas revoloteaban algunos pájaros de quimérico y legendario plumajcri"
                ),
                "chunks": [
                    {"text": "y", "timestamp": (0.52, 0.54)},
                    {"text": "en", "timestamp": (0.6, 0.68)},
                    {"text": "las", "timestamp": (0.74, 0.84)},
                    {"text": "ramas", "timestamp": (0.94, 1.24)},
                    {"text": "medio", "timestamp": (1.32, 1.52)},
                    {"text": "sumergidas", "timestamp": (1.56, 2.22)},
                    {"text": "revoloteaban", "timestamp": (2.36, 3.0)},
                    {"text": "algunos", "timestamp": (3.06, 3.38)},
                    {"text": "pájaros", "timestamp": (3.46, 3.86)},
                    {"text": "de", "timestamp": (3.92, 4.0)},
                    {"text": "quimérico", "timestamp": (4.08, 4.6)},
                    {"text": "y", "timestamp": (4.66, 4.68)},
                    {"text": "legendario", "timestamp": (4.74, 5.26)},
                    {"text": "plumajcri", "timestamp": (5.34, 5.74)},
                ],
            },
        )
        # CTC + LM models cannot use return_timestamps="char"
        with self.assertRaisesRegex(
            ValueError, "^CTC with LM can only predict word level timestamps, set `return_timestamps='word'`$"
        ):
            _ = speech_recognizer(filename, return_timestamps="char")

    @require_tf
    def test_small_model_tf(self):
        self.skipTest(reason="Tensorflow not supported yet.")

    @require_torch
    def test_torch_small_no_tokenizer_files(self):
        # test that model without tokenizer file cannot be loaded
        with pytest.raises(OSError):
            pipeline(
                task="automatic-speech-recognition",
                model="patrickvonplaten/tiny-wav2vec2-no-tokenizer",
                framework="pt",
            )

    @require_torch
    @slow
    def test_torch_large(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-base-960h",
            tokenizer="facebook/wav2vec2-base-960h",
            framework="pt",
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform)
        self.assertEqual(output, {"text": ""})

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = speech_recognizer(audio)
        self.assertEqual(output, {"text": "A MAN SAID TO THE UNIVERSE SIR I EXIST"})

    @require_torch
    @slow
    def test_torch_large_with_input_features(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="hf-audio/wav2vec2-bert-CV16-en",
            framework="pt",
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform)
        self.assertEqual(output, {"text": ""})

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = speech_recognizer(audio)
        self.assertEqual(output, {"text": "a man said to the universe sir i exist"})

    @slow
    @require_torch
    def test_return_timestamps_in_preprocess(self):
        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
            chunk_length_s=8,
            stride_length_s=1,
        )
        data = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True)
        sample = next(iter(data))

        res = pipe(sample["audio"]["array"])
        self.assertEqual(res, {"text": " Conquered returned to its place amidst the tents."})

        res = pipe(sample["audio"]["array"], return_timestamps=True)
        self.assertEqual(
            res,
            {
                "text": " Conquered returned to its place amidst the tents.",
                "chunks": [{"timestamp": (0.0, 3.36), "text": " Conquered returned to its place amidst the tents."}],
            },
        )

        res = pipe(sample["audio"]["array"], return_timestamps="word")
        # fmt: off
        self.assertEqual(
            res,
            {
                'text': ' Conquered returned to its place amidst the tents.',
                'chunks': [
                    {'text': ' Conquered', 'timestamp': (0.5, 1.2)},
                    {'text': ' returned', 'timestamp': (1.2, 1.64)},
                    {'text': ' to', 'timestamp': (1.64, 1.84)},
                    {'text': ' its', 'timestamp': (1.84, 2.02)},
                    {'text': ' place', 'timestamp': (2.02, 2.28)},
                    {'text': ' amidst', 'timestamp': (2.28, 2.8)},
                    {'text': ' the', 'timestamp': (2.8, 2.98)},
                    {'text': ' tents.', 'timestamp': (2.98, 3.48)},
                ],
            },
        )
        # fmt: on

    @slow
    @require_torch
    def test_return_timestamps_and_language_in_preprocess(self):
        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
            chunk_length_s=8,
            stride_length_s=1,
            return_language=True,
        )
        data = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True)
        sample = next(iter(data))

        res = pipe(sample["audio"]["array"])
        self.assertEqual(
            res,
            {
                "text": " Conquered returned to its place amidst the tents.",
                "chunks": [{"language": "english", "text": " Conquered returned to its place amidst the tents."}],
            },
        )

        res = pipe(sample["audio"]["array"], return_timestamps=True)
        self.assertEqual(
            res,
            {
                "text": " Conquered returned to its place amidst the tents.",
                "chunks": [
                    {
                        "timestamp": (0.0, 3.36),
                        "language": "english",
                        "text": " Conquered returned to its place amidst the tents.",
                    }
                ],
            },
        )

        res = pipe(sample["audio"]["array"], return_timestamps="word")
        # fmt: off
        self.assertEqual(
            res,
            {
                'text': ' Conquered returned to its place amidst the tents.',
                'chunks': [
                    {"language": "english",'text': ' Conquered', 'timestamp': (0.5, 1.2)},
                    {"language": "english", 'text': ' returned', 'timestamp': (1.2, 1.64)},
                    {"language": "english",'text': ' to', 'timestamp': (1.64, 1.84)},
                    {"language": "english",'text': ' its', 'timestamp': (1.84, 2.02)},
                    {"language": "english",'text': ' place', 'timestamp': (2.02, 2.28)},
                    {"language": "english",'text': ' amidst', 'timestamp': (2.28, 2.8)},
                    {"language": "english",'text': ' the', 'timestamp': (2.8, 2.98)},
                    {"language": "english",'text': ' tents.', 'timestamp': (2.98, 3.48)},
                ],
            },
        )
        # fmt: on

    @slow
    @require_torch
    def test_return_timestamps_in_preprocess_longform(self):
        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny.en",
        )
        data = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True)
        samples = [next(iter(data)) for _ in range(8)]
        audio = np.concatenate([sample["audio"]["array"] for sample in samples])

        res = pipe(audio)
        expected_output = {
            "text": " Concord returned to its place amidst the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst "
            "the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst "
            "the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst "
            "the tents. Concord returned to its place amidst the tents."
        }
        self.assertEqual(res, expected_output)
        res = pipe(audio, return_timestamps=True)
        self.assertEqual(
            res,
            {
                "text": " Concord returned to its place amidst the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst the tents. Concord returned to its place amidst the tents.",
                "chunks": [
                    {"timestamp": (0.0, 3.22), "text": " Concord returned to its place amidst the tents."},
                    {"timestamp": (3.22, 6.74), "text": " Concord returned to its place amidst the tents."},
                    {"timestamp": (6.74, 10.26), "text": " Concord returned to its place amidst the tents."},
                    {"timestamp": (10.26, 13.78), "text": " Concord returned to its place amidst the tents."},
                    {"timestamp": (13.78, 17.3), "text": " Concord returned to its place amidst the tents."},
                    {"timestamp": (17.3, 20.82), "text": " Concord returned to its place amidst the tents."},
                    {"timestamp": (20.82, 24.34), "text": " Concord returned to its place amidst the tents."},
                    {"timestamp": (24.34, 27.86), "text": " Concord returned to its place amidst the tents."},
                ],
            },
        )
        pipe.model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
        res = pipe(audio, return_timestamps="word")

        # fmt: off
        self.assertEqual(
            res["chunks"][:15],
            [
                {"text": " Concord", "timestamp": (0.5, 0.94)},
                {"text": " returned", "timestamp": (0.94, 1.52)},
                {"text": " to", "timestamp": (1.52, 1.78)},
                {"text": " its", "timestamp": (1.78, 1.98)},
                {"text": " place", "timestamp": (1.98, 2.16)},
                {"text": " amidst", "timestamp": (2.16, 2.5)},
                {"text": " the", "timestamp": (2.5, 2.9)},
                {"text": " tents.", "timestamp": (2.9, 4.2)},
                {"text": " Concord", "timestamp": (4.2, 4.5)},
                {"text": " returned", "timestamp": (4.5, 5.0)},
                {"text": " to", "timestamp": (5.0, 5.28)},
                {"text": " its", "timestamp": (5.28, 5.48)},
                {"text": " place", "timestamp": (5.48, 5.7)},
                {"text": " amidst", "timestamp": (5.7, 6.02)},
                {"text": " the", "timestamp": (6.02, 6.4)}


            ],
        )
        # fmt: on

    @require_torch
    def test_return_timestamps_in_init(self):
        # segment-level timestamps are accepted
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")

        dummy_speech = np.ones(100)

        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chunk_length_s=8,
            stride_length_s=1,
            return_timestamps=True,
        )

        _ = pipe(dummy_speech)

        # word-level timestamps are accepted
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chunk_length_s=8,
            stride_length_s=1,
            return_timestamps="word",
        )

        _ = pipe(dummy_speech)

        # char-level timestamps are not accepted
        with self.assertRaisesRegex(
            ValueError,
            "^Whisper cannot return `char` timestamps, only word level or segment level timestamps. "
            "Use `return_timestamps='word'` or `return_timestamps=True` respectively.$",
        ):
            pipe = pipeline(
                task="automatic-speech-recognition",
                model=model,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                chunk_length_s=8,
                stride_length_s=1,
                return_timestamps="char",
            )

            _ = pipe(dummy_speech)

    @require_torch
    @slow
    def test_torch_whisper(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
            framework="pt",
        )
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = speech_recognizer(audio)
        self.assertEqual(output, {"text": " A man said to the universe, Sir, I exist."})

        output = speech_recognizer([ds[40]["audio"]], chunk_length_s=5, batch_size=4)
        self.assertEqual(output, [{"text": " A man said to the universe, Sir, I exist."}])

    @require_torch
    @slow
    def test_torch_whisper_batched(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
            framework="pt",
        )
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:2]")
        EXPECTED_OUTPUT = [
            {"text": " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."},
            {"text": " Nor is Mr. Quilters' manner less interesting than his matter."},
        ]

        output = speech_recognizer(ds["audio"], batch_size=2)
        self.assertEqual(output, EXPECTED_OUTPUT)

    @slow
    def test_find_longest_common_subsequence(self):
        max_source_positions = 1500
        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")

        previous_sequence = [[51492, 406, 3163, 1953, 466, 13, 51612, 51612]]
        self.assertEqual(
            processor.decode(previous_sequence[0], output_offsets=True),
            {
                "text": " not worth thinking about.",
                "offsets": [{"text": " not worth thinking about.", "timestamp": (22.56, 24.96)}],
            },
        )

        # Merge when the previous sequence is a suffix of the next sequence
        # fmt: off
        next_sequences_1 = [
            [50364, 295, 6177, 3391, 11, 19817, 3337, 507, 307, 406, 3163, 1953, 466, 13, 50614, 50614, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50834, 50257]
        ]
        # fmt: on
        self.assertEqual(
            processor.decode(next_sequences_1[0], output_offsets=True),
            {
                "text": (
                    " of spectators, retrievality is not worth thinking about. His instant panic was followed by a"
                    " small, sharp blow high on his chest.<|endoftext|>"
                ),
                "offsets": [
                    {"text": " of spectators, retrievality is not worth thinking about.", "timestamp": (0.0, 5.0)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (5.0, 9.4),
                    },
                ],
            },
        )
        merge = _find_timestamp_sequence(
            [[previous_sequence, (480_000, 0, 0)], [next_sequences_1, (480_000, 120_000, 0)]],
            processor.tokenizer,
            processor.feature_extractor,
            max_source_positions,
        )

        # fmt: off
        self.assertEqual(
            merge,
            [51492, 406, 3163, 1953, 466, 13, 51739, 51739, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 51959],
        )
        # fmt: on
        self.assertEqual(
            processor.decode(merge, output_offsets=True),
            {
                "text": (
                    " not worth thinking about. His instant panic was followed by a small, sharp blow high on his"
                    " chest."
                ),
                "offsets": [
                    {"text": " not worth thinking about.", "timestamp": (22.56, 27.5)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (27.5, 31.900000000000002),
                    },
                ],
            },
        )

        # Merge when the sequence is in the middle of the 1st next sequence
        # fmt: off
        next_sequences_2 = [
            [50364, 295, 6177, 3391, 11, 19817, 3337, 507, 307, 406, 3163, 1953, 466, 13, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50834, 50257]
        ]
        # fmt: on
        # {'text': ' of spectators, retrievality is not worth thinking about. His instant panic was followed by a small, sharp blow high on his chest.','timestamp': (0.0, 9.4)}
        merge = _find_timestamp_sequence(
            [[previous_sequence, (480_000, 0, 0)], [next_sequences_2, (480_000, 120_000, 0)]],
            processor.tokenizer,
            processor.feature_extractor,
            max_source_positions,
        )
        # fmt: off
        self.assertEqual(
            merge,
            [51492, 406, 3163, 1953, 466, 13, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 51959],
        )
        # fmt: on
        self.assertEqual(
            processor.decode(merge, output_offsets=True),
            {
                "text": (
                    " not worth thinking about. His instant panic was followed by a small, sharp blow high on his"
                    " chest."
                ),
                "offsets": [
                    {
                        "text": (
                            " not worth thinking about. His instant panic was followed by a small, sharp blow high on"
                            " his chest."
                        ),
                        "timestamp": (22.56, 31.900000000000002),
                    },
                ],
            },
        )

        # Merge when the previous sequence is not included in the current sequence
        next_sequences_3 = [[50364, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50584, 50257]]  # fmt: skip
        # {'text': ' His instant panic was followed by a small, sharp blow high on his chest.','timestamp': (0.0, 9.4)}
        merge = _find_timestamp_sequence(
            [[previous_sequence, (480_000, 0, 0)], [next_sequences_3, (480_000, 120_000, 0)]],
            processor.tokenizer,
            processor.feature_extractor,
            max_source_positions,
        )
        self.assertEqual(
            merge,
            [51492, 406, 3163, 1953, 466, 13, 51612, 51612, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 51832],
        )  # fmt: skip
        self.assertEqual(
            processor.decode(merge, output_offsets=True),
            {
                "text": (
                    " not worth thinking about. His instant panic was followed by a small, sharp blow high on his"
                    " chest."
                ),
                "offsets": [
                    {"text": " not worth thinking about.", "timestamp": (22.56, 24.96)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (24.96, 29.36),
                    },
                ],
            },
        )
        # last case is when the sequence is not in the first next predicted start and end of timestamp
        next_sequences_3 = [
            [50364, 2812, 9836, 14783, 390, 406, 3163, 1953, 466, 13, 50634, 50634, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50934]
        ]  # fmt: skip
        merge = _find_timestamp_sequence(
            [[previous_sequence, (480_000, 0, 0)], [next_sequences_3, (480_000, 167_000, 0)]],
            processor.tokenizer,
            processor.feature_extractor,
            max_source_positions,
        )
        self.assertEqual(
            merge,
            [51492, 406, 3163, 1953, 466, 13, 51612, 51612, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 51912]
        )  # fmt: skip
        self.assertEqual(
            processor.decode(merge, output_offsets=True),
            {
                "text": (
                    " not worth thinking about. His instant panic was followed by a small, sharp blow high on his"
                    " chest."
                ),
                "offsets": [
                    {"text": " not worth thinking about.", "timestamp": (22.56, 24.96)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (24.96, 30.96),
                    },
                ],
            },
        )

    @slow
    @require_torch
    def test_whisper_timestamp_prediction(self):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        array = np.concatenate(
            [ds[40]["audio"]["array"], ds[41]["audio"]["array"], ds[42]["audio"]["array"], ds[43]["audio"]["array"]]
        )
        pipe = pipeline(
            model="openai/whisper-small",
            return_timestamps=True,
        )

        output = pipe(ds[40]["audio"])
        self.assertDictEqual(
            output,
            {
                "text": " A man said to the universe, Sir, I exist.",
                "chunks": [{"text": " A man said to the universe, Sir, I exist.", "timestamp": (0.0, 4.26)}],
            },
        )

        output = pipe(array, chunk_length_s=10)
        self.assertDictEqual(
            nested_simplify(output),
            {
                "chunks": [
                    {"text": " A man said to the universe, Sir, I exist.", "timestamp": (0.0, 5.5)},
                    {
                        "text": (
                            " Sweat covered Brion's body, trickling into the "
                            "tight-loan cloth that was the only garment he wore, the "
                            "cut"
                        ),
                        "timestamp": (5.5, 11.95),
                    },
                    {
                        "text": (
                            " on his chest still dripping blood, the ache of his "
                            "overstrained eyes, even the soaring arena around him "
                            "with"
                        ),
                        "timestamp": (11.95, 19.61),
                    },
                    {
                        "text": " the thousands of spectators, retrievality is not worth thinking about.",
                        "timestamp": (19.61, 25.0),
                    },
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (25.0, 29.4),
                    },
                ],
                "text": (
                    " A man said to the universe, Sir, I exist. Sweat covered Brion's "
                    "body, trickling into the tight-loan cloth that was the only garment "
                    "he wore, the cut on his chest still dripping blood, the ache of his "
                    "overstrained eyes, even the soaring arena around him with the "
                    "thousands of spectators, retrievality is not worth thinking about. "
                    "His instant panic was followed by a small, sharp blow high on his "
                    "chest."
                ),
            },
        )

        output = pipe(array)
        self.assertDictEqual(
            output,
            {
                "chunks": [
                    {"text": " A man said to the universe, Sir, I exist.", "timestamp": (0.0, 5.5)},
                    {
                        "text": (
                            " Sweat covered Brion's body, trickling into the "
                            "tight-loan cloth that was the only garment"
                        ),
                        "timestamp": (5.5, 10.18),
                    },
                    {"text": " he wore.", "timestamp": (10.18, 11.68)},
                    {"text": " The cut on his chest still dripping blood.", "timestamp": (11.68, 14.92)},
                    {"text": " The ache of his overstrained eyes.", "timestamp": (14.92, 17.6)},
                    {
                        "text": (
                            " Even the soaring arena around him with the thousands of spectators were trivialities"
                        ),
                        "timestamp": (17.6, 22.56),
                    },
                    {"text": " not worth thinking about.", "timestamp": (22.56, 24.96)},
                ],
                "text": (
                    " A man said to the universe, Sir, I exist. Sweat covered Brion's "
                    "body, trickling into the tight-loan cloth that was the only garment "
                    "he wore. The cut on his chest still dripping blood. The ache of his "
                    "overstrained eyes. Even the soaring arena around him with the "
                    "thousands of spectators were trivialities not worth thinking about."
                ),
            },
        )

    @slow
    @require_torch
    def test_whisper_large_timestamp_prediction(self):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        array = np.concatenate(
            [ds[40]["audio"]["array"], ds[41]["audio"]["array"], ds[42]["audio"]["array"], ds[43]["audio"]["array"]]
        )
        pipe = pipeline(model="openai/whisper-large-v3", return_timestamps=True)

        output = pipe(ds[40]["audio"])
        self.assertDictEqual(
            output,
            {
                "text": " A man said to the universe, Sir, I exist.",
                "chunks": [{"text": " A man said to the universe, Sir, I exist.", "timestamp": (0.0, 4.08)}],
            },
        )

        output = pipe(array, chunk_length_s=10)

        self.assertDictEqual(
            nested_simplify(output),
            {
                "chunks": [
                    {"timestamp": (0.0, 2.0), "text": (" A man said to the universe,")},
                    {"timestamp": (2.0, 4.1), "text": (" Sir, I exist.")},
                    {"timestamp": (5.14, 5.96), "text": (" Sweat covered")},
                    {"timestamp": (5.96, 8.02), "text": (" Breon's body, trickling into")},
                    {"timestamp": (8.02, 10.67), "text": (" the tight loincloth that was the only garment he wore,")},
                    {"timestamp": (10.67, 13.67), "text": (" the cut on his chest still dripping blood,")},
                    {"timestamp": (13.67, 17.61), "text": (" the ache of his overstrained eyes.")},
                    {
                        "timestamp": (17.61, 24.0),
                        "text": (
                            " Even the soaring arena around him with thousands of spectators were trivialities not worth thinking about."
                        ),
                    },
                    {
                        "timestamp": (24.0, 29.94),
                        "text": (" His instant of panic was followed by a small, sharp blow high on his chest."),
                    },
                ],
                "text": (
                    " A man said to the universe, Sir, I exist. Sweat covered Breon's"
                    " body, trickling into the tight loincloth that was the only garment"
                    " he wore, the cut on his chest still dripping blood, the ache of his"
                    " overstrained eyes. Even the soaring arena around him with thousands"
                    " of spectators were trivialities not worth thinking about. His "
                    "instant of panic was followed by a small, sharp blow high on his chest."
                ),
            },
        )

        output = pipe(array)
        self.assertDictEqual(
            output,
            {
                "chunks": [
                    {"timestamp": (0.0, 1.96), "text": " A man said to the universe,"},
                    {"timestamp": (2.7, 4.1), "text": " Sir, I exist."},
                    {"timestamp": (5.14, 6.84), "text": " Sweat covered Brion's body,"},
                    {
                        "timestamp": (7.4, 10.68),
                        "text": " trickling into the tight loincloth that was the only garment he wore,",
                    },
                    {"timestamp": (11.6, 13.94), "text": " the cut on his chest still dripping blood,"},
                    {"timestamp": (14.78, 16.72), "text": " the ache of his overstrained eyes,"},
                    {
                        "timestamp": (17.32, 21.16),
                        "text": " even the soaring arena around him with the thousands of spectators",
                    },
                    {"timestamp": (21.16, 23.94), "text": " were trivialities not worth thinking about."},
                    {
                        "timestamp": (24.42, 29.94),
                        "text": " His instant panic was followed by a small sharp blow high on his chest.",
                    },
                ],
                "text": (
                    " A man said to the universe, Sir, I exist. Sweat covered Brion's body,"
                    " trickling into the tight loincloth that was the only garment he wore, "
                    "the cut on his chest still dripping blood, the ache of his overstrained "
                    "eyes, even the soaring arena around him with the thousands of spectators "
                    "were trivialities not worth thinking about. His instant panic was followed "
                    "by a small sharp blow high on his chest."
                ),
            },
        )

    @slow
    @require_torch
    def test_whisper_word_timestamps_batched(self):
        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
            chunk_length_s=3,
            return_timestamps="word",
        )
        data = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = data[0]["audio"]

        # not the same output as test_simple_whisper_asr because of chunking
        EXPECTED_OUTPUT = {
            "text": " Mr. Quilder is the apostle of the middle classes and we are glad to welcome his gospel.",
            "chunks": [
                {"text": " Mr.", "timestamp": (0.48, 0.96)},
                {"text": " Quilder", "timestamp": (0.96, 1.24)},
                {"text": " is", "timestamp": (1.24, 1.5)},
                {"text": " the", "timestamp": (1.5, 1.72)},
                {"text": " apostle", "timestamp": (1.72, 1.98)},
                {"text": " of", "timestamp": (1.98, 2.32)},
                {"text": " the", "timestamp": (2.32, 2.5)},
                {"text": " middle", "timestamp": (2.5, 2.68)},
                {"text": " classes", "timestamp": (2.68, 3.2)},
                {"text": " and", "timestamp": (3.2, 3.56)},
                {"text": " we", "timestamp": (3.56, 3.68)},
                {"text": " are", "timestamp": (3.68, 3.8)},
                {"text": " glad", "timestamp": (3.8, 4.1)},
                {"text": " to", "timestamp": (4.1, 4.34)},
                {"text": " welcome", "timestamp": (4.3, 4.6)},
                {"text": " his", "timestamp": (4.6, 4.94)},
                {"text": " gospel.", "timestamp": (4.94, 5.82)},
            ],
        }

        # batch size 1: copy the audio sample since pipeline consumes it
        output = pipe(sample.copy(), batch_size=1)
        self.assertDictEqual(output, EXPECTED_OUTPUT)

        # batch size 2: input audio is chunked into smaller pieces so it's testing batching
        output = pipe(sample, batch_size=2)
        self.assertDictEqual(output, EXPECTED_OUTPUT)

    @slow
    @require_torch
    def test_whisper_large_word_timestamps_batched(self):
        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-large-v3",
            return_timestamps="word",
        )
        data = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = data[0]["audio"]

        # not the same output as test_simple_whisper_asr because of chunking
        EXPECTED_OUTPUT = {
            "text": " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
            "chunks": [
                {"text": " Mr.", "timestamp": (0.0, 0.74)},
                {"text": " Quilter", "timestamp": (0.74, 1.04)},
                {"text": " is", "timestamp": (1.04, 1.3)},
                {"text": " the", "timestamp": (1.3, 1.44)},
                {"text": " apostle", "timestamp": (1.44, 1.74)},
                {"text": " of", "timestamp": (1.74, 2.18)},
                {"text": " the", "timestamp": (2.18, 2.28)},
                {"text": " middle", "timestamp": (2.28, 2.5)},
                {"text": " classes,", "timestamp": (2.5, 3.0)},
                {"text": " and", "timestamp": (3.0, 3.4)},
                {"text": " we", "timestamp": (3.4, 3.5)},
                {"text": " are", "timestamp": (3.5, 3.6)},
                {"text": " glad", "timestamp": (3.6, 3.84)},
                {"text": " to", "timestamp": (3.84, 4.1)},
                {"text": " welcome", "timestamp": (4.1, 4.4)},
                {"text": " his", "timestamp": (4.4, 4.7)},
                {"text": " gospel.", "timestamp": (4.7, 5.34)},
            ],
        }

        # batch size 1: copy the audio sample since pipeline consumes it
        output = pipe(sample.copy(), batch_size=1)
        self.assertDictEqual(output, EXPECTED_OUTPUT)

        # batch size 2: input audio is chunked into smaller pieces so it's testing batching
        output = pipe(sample, batch_size=2)
        self.assertDictEqual(output, EXPECTED_OUTPUT)

    @require_torch
    @slow
    def test_torch_speech_encoder_decoder(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/s2t-wav2vec2-large-en-de",
            feature_extractor="facebook/s2t-wav2vec2-large-en-de",
            framework="pt",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = speech_recognizer(audio)
        self.assertEqual(output, {"text": 'Ein Mann sagte zum Universum : " Sir, ich existiert! "'})

    @slow
    @require_torch
    def test_simple_wav2vec2(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

        asr = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = asr(waveform)
        self.assertEqual(output, {"text": ""})

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = asr(audio)
        self.assertEqual(output, {"text": "A MAN SAID TO THE UNIVERSE SIR I EXIST"})

        data = Audio().encode_example(ds[40]["audio"])["bytes"]
        output = asr(data)
        self.assertEqual(output, {"text": "A MAN SAID TO THE UNIVERSE SIR I EXIST"})

    @slow
    @require_torch
    @require_torchaudio
    def test_simple_s2t(self):
        model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-mustc-en-it-st")
        tokenizer = AutoTokenizer.from_pretrained("facebook/s2t-small-mustc-en-it-st")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/s2t-small-mustc-en-it-st")

        asr = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)

        output = asr(waveform)
        self.assertEqual(output, {"text": "(Applausi)"})

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = asr(audio)
        self.assertEqual(output, {"text": "Un uomo disse all'universo: \"Signore, io esisto."})

        data = Audio().encode_example(ds[40]["audio"])["bytes"]
        output = asr(data)
        self.assertEqual(output, {"text": "Un uomo disse all'universo: \"Signore, io esisto."})

    @slow
    @require_torch
    @require_torchaudio
    def test_simple_whisper_asr(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            framework="pt",
        )
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio = ds[0]["audio"]
        output = speech_recognizer(audio)
        self.assertEqual(
            output,
            {"text": " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."},
        )
        output = speech_recognizer(ds[0]["audio"], return_timestamps=True)
        self.assertEqual(
            output,
            {
                "text": " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
                "chunks": [
                    {
                        "text": (
                            " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
                        ),
                        "timestamp": (0.0, 5.44),
                    }
                ],
            },
        )
        speech_recognizer.model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
        output = speech_recognizer(ds[0]["audio"], return_timestamps="word")
        # fmt: off
        self.assertEqual(
            output,
            {
                'text': ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.',
                'chunks': [
                    {'text': ' Mr.', 'timestamp': (0.38, 1.04)},
                    {'text': ' Quilter', 'timestamp': (1.04, 1.18)},
                    {'text': ' is', 'timestamp': (1.18, 1.44)},
                    {'text': ' the', 'timestamp': (1.44, 1.58)},
                    {'text': ' apostle', 'timestamp': (1.58, 1.98)},
                    {'text': ' of', 'timestamp': (1.98, 2.32)},
                    {'text': ' the', 'timestamp': (2.32, 2.46)},
                    {'text': ' middle', 'timestamp': (2.46, 2.56)},
                    {'text': ' classes,', 'timestamp': (2.56, 3.4)},
                    {'text': ' and', 'timestamp': (3.4, 3.54)},
                    {'text': ' we', 'timestamp': (3.54, 3.62)},
                    {'text': ' are', 'timestamp': (3.62, 3.72)},
                    {'text': ' glad', 'timestamp': (3.72, 4.0)},
                    {'text': ' to', 'timestamp': (4.0, 4.26)},
                    {'text': ' welcome', 'timestamp': (4.26, 4.56)},
                    {'text': ' his', 'timestamp': (4.56, 4.92)},
                    {'text': ' gospel.', 'timestamp': (4.92, 5.84)}
                ]
            }
        )
        # fmt: on

        # Whisper can only predict segment level timestamps or word level, not character level
        with self.assertRaisesRegex(
            ValueError,
            "^Whisper cannot return `char` timestamps, only word level or segment level timestamps. "
            "Use `return_timestamps='word'` or `return_timestamps=True` respectively.$",
        ):
            _ = speech_recognizer(audio, return_timestamps="char")

    @slow
    @require_torch
    @require_torchaudio
    def test_simple_whisper_translation(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-large",
            framework="pt",
        )
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = speech_recognizer(audio)
        self.assertEqual(output, {"text": " A man said to the universe, Sir, I exist."})

        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
        tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large")
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large")

        speech_recognizer_2 = AutomaticSpeechRecognitionPipeline(
            model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
        )
        output_2 = speech_recognizer_2(ds[40]["audio"])
        self.assertEqual(output, output_2)

        # either use generate_kwargs or set the model's generation_config
        # model.generation_config.task = "transcribe"
        # model.generation_config.lang = "<|it|>"
        speech_translator = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            generate_kwargs={"task": "transcribe", "language": "<|it|>"},
        )
        output_3 = speech_translator(ds[40]["audio"])
        self.assertEqual(output_3, {"text": " Un uomo ha detto all'universo, Sir, esiste."})

    @slow
    @require_torch
    def test_whisper_language(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            framework="pt",
        )
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio = ds[0]["audio"]

        # 1. English-only model compatible with no language argument
        output = speech_recognizer(audio)
        self.assertEqual(
            output,
            {"text": " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."},
        )

        # 2. English-only Whisper does not accept the language argument
        with self.assertRaisesRegex(
            ValueError,
            "Cannot specify `task` or `language` for an English-only model. If the model is intended to be multilingual, "
            "pass `is_multilingual=True` to generate, or update the generation config.",
        ):
            _ = speech_recognizer(ds[0]["audio"], generate_kwargs={"language": "en"})

        # 3. Multilingual model accepts language argument
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
            framework="pt",
        )
        output = speech_recognizer(ds[0]["audio"], generate_kwargs={"language": "en"})
        self.assertEqual(
            output,
            {"text": " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."},
        )

    @slow
    def test_speculative_decoding_whisper_non_distil(self):
        # Load data:
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]", trust_remote_code=True
        )
        sample = dataset[0]["audio"]

        # Load model:
        model_id = "openai/whisper-large-v2"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            use_safetensors=True,
        )

        # Load assistant:
        assistant_model_id = "openai/whisper-tiny"
        assistant_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            assistant_model_id,
            use_safetensors=True,
        )

        # Load pipeline:
        pipe = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            generate_kwargs={"language": "en"},
        )

        start_time = time.time()
        transcription_non_ass = pipe(sample.copy(), generate_kwargs={"assistant_model": assistant_model})["text"]
        total_time_assist = time.time() - start_time

        start_time = time.time()
        transcription_ass = pipe(sample)["text"]
        total_time_non_assist = time.time() - start_time

        self.assertEqual(transcription_ass, transcription_non_ass)
        self.assertEqual(
            transcription_ass,
            " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.",
        )
        self.assertTrue(total_time_non_assist > total_time_assist, "Make sure that assistant decoding is faster")

    @slow
    def test_speculative_decoding_whisper_distil(self):
        # Load data:
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]", trust_remote_code=True
        )
        sample = dataset[0]["audio"]

        # Load model:
        model_id = "openai/whisper-large-v2"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            use_safetensors=True,
        )

        # Load assistant:
        assistant_model_id = "distil-whisper/distil-large-v2"
        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_id,
            use_safetensors=True,
        )

        # Load pipeline:
        pipe = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            generate_kwargs={"language": "en"},
        )

        start_time = time.time()
        transcription_non_ass = pipe(sample.copy(), generate_kwargs={"assistant_model": assistant_model})["text"]
        total_time_assist = time.time() - start_time

        start_time = time.time()
        transcription_ass = pipe(sample)["text"]
        total_time_non_assist = time.time() - start_time

        self.assertEqual(transcription_ass, transcription_non_ass)
        self.assertEqual(
            transcription_ass,
            " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.",
        )
        self.assertEqual(total_time_non_assist > total_time_assist, "Make sure that assistant decoding is faster")

    @slow
    @require_torch
    @require_torchaudio
    def test_xls_r_to_en(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-xls-r-1b-21-to-en",
            feature_extractor="facebook/wav2vec2-xls-r-1b-21-to-en",
            framework="pt",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = speech_recognizer(audio)
        self.assertEqual(output, {"text": "A man said to the universe: “Sir, I exist."})

    @slow
    @require_torch
    @require_torchaudio
    def test_xls_r_from_en(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-xls-r-1b-en-to-15",
            feature_extractor="facebook/wav2vec2-xls-r-1b-en-to-15",
            framework="pt",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = speech_recognizer(audio)
        self.assertEqual(output, {"text": "Ein Mann sagte zu dem Universum, Sir, ich bin da."})

    @slow
    @require_torch
    @require_torchaudio
    def test_speech_to_text_leveraged(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="patrickvonplaten/wav2vec2-2-bart-base",
            feature_extractor="patrickvonplaten/wav2vec2-2-bart-base",
            tokenizer=AutoTokenizer.from_pretrained("patrickvonplaten/wav2vec2-2-bart-base"),
            framework="pt",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]
        output = speech_recognizer(audio)
        self.assertEqual(output, {"text": "a man said to the universe sir i exist"})

    @slow
    @require_torch_accelerator
    def test_wav2vec2_conformer_float16(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-conformer-rope-large-960h-ft",
            device=torch_device,
            torch_dtype=torch.float16,
            framework="pt",
        )

        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = dataset[0]["audio"]

        output = speech_recognizer(sample)
        self.assertEqual(
            output,
            {"text": "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"},
        )

    @require_torch
    def test_chunking_fast(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="hf-internal-testing/tiny-random-wav2vec2",
            chunk_length_s=10.0,
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 2
        audio_tiled = np.tile(audio, n_repeats)
        output = speech_recognizer([audio_tiled], batch_size=2)
        self.assertEqual(output, [{"text": ANY(str)}])
        self.assertEqual(output[0]["text"][:6], "ZBT ZC")

    @require_torch
    def test_input_parameter_passthrough(self):
        """Test that chunked vs non chunked versions of ASR pipelines returns the same structure for the same inputs."""
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="hf-internal-testing/tiny-random-wav2vec2",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        inputs = {"raw": audio, "sampling_rate": 16_000, "id": 1}

        chunked_output = speech_recognizer(inputs.copy(), chunk_length_s=30)
        non_chunked_output = speech_recognizer(inputs.copy())
        assert (
            chunked_output.keys() == non_chunked_output.keys()
        ), "The output structure should be the same for chunked vs non-chunked versions of asr pipelines."

    @require_torch
    def test_return_timestamps_ctc_fast(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="hf-internal-testing/tiny-random-wav2vec2",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        # Take short audio to keep the test readable
        audio = ds[40]["audio"]["array"][:800]

        output = speech_recognizer(audio, return_timestamps="char")
        self.assertEqual(
            output,
            {
                "text": "ZBT ZX G",
                "chunks": [
                    {"text": " ", "timestamp": (0.0, 0.012)},
                    {"text": "Z", "timestamp": (0.012, 0.016)},
                    {"text": "B", "timestamp": (0.016, 0.02)},
                    {"text": "T", "timestamp": (0.02, 0.024)},
                    {"text": " ", "timestamp": (0.024, 0.028)},
                    {"text": "Z", "timestamp": (0.028, 0.032)},
                    {"text": "X", "timestamp": (0.032, 0.036)},
                    {"text": " ", "timestamp": (0.036, 0.04)},
                    {"text": "G", "timestamp": (0.04, 0.044)},
                ],
            },
        )

        output = speech_recognizer(audio, return_timestamps="word")
        self.assertEqual(
            output,
            {
                "text": "ZBT ZX G",
                "chunks": [
                    {"text": "ZBT", "timestamp": (0.012, 0.024)},
                    {"text": "ZX", "timestamp": (0.028, 0.036)},
                    {"text": "G", "timestamp": (0.04, 0.044)},
                ],
            },
        )

    @require_torch
    @require_pyctcdecode
    def test_chunking_fast_with_lm(self):
        speech_recognizer = pipeline(
            model="hf-internal-testing/processor_with_lm",
            chunk_length_s=10.0,
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 2
        audio_tiled = np.tile(audio, n_repeats)
        # Batch_size = 1
        output1 = speech_recognizer([audio_tiled], batch_size=1)
        self.assertEqual(output1, [{"text": ANY(str)}])
        self.assertEqual(output1[0]["text"][:6], "<s> <s")

        # batch_size = 2
        output2 = speech_recognizer([audio_tiled], batch_size=2)
        self.assertEqual(output2, [{"text": ANY(str)}])
        self.assertEqual(output2[0]["text"][:6], "<s> <s")

        # TODO There is an offby one error because of the ratio.
        # Maybe logits get affected by the padding on this random
        # model is more likely. Add some masking ?
        # self.assertEqual(output1, output2)

    @require_torch
    @require_pyctcdecode
    def test_with_lm_fast(self):
        speech_recognizer = pipeline(
            model="hf-internal-testing/processor_with_lm",
        )
        self.assertEqual(speech_recognizer.type, "ctc_with_lm")

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 2
        audio_tiled = np.tile(audio, n_repeats)

        output = speech_recognizer([audio_tiled], batch_size=2)
        self.assertEqual(output, [{"text": ANY(str)}])
        self.assertEqual(output[0]["text"][:6], "<s> <s")

        # Making sure the argument are passed to the decoder
        # Since no change happens in the result, check the error comes from
        # the `decode_beams` function.
        with self.assertRaises(TypeError) as e:
            output = speech_recognizer([audio_tiled], decoder_kwargs={"num_beams": 2})
            self.assertContains(e.msg, "TypeError: decode_beams() got an unexpected keyword argument 'num_beams'")
        output = speech_recognizer([audio_tiled], decoder_kwargs={"beam_width": 2})

    @require_torch
    @require_pyctcdecode
    def test_with_local_lm_fast(self):
        local_dir = snapshot_download("hf-internal-testing/processor_with_lm")
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model=local_dir,
        )
        self.assertEqual(speech_recognizer.type, "ctc_with_lm")

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 2
        audio_tiled = np.tile(audio, n_repeats)

        output = speech_recognizer([audio_tiled], batch_size=2)

        self.assertEqual(output, [{"text": ANY(str)}])
        self.assertEqual(output[0]["text"][:6], "<s> <s")

    @require_torch
    @slow
    def test_whisper_prompted(self):
        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model = model.to(torch_device)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
        )

        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        sample = dataset[0]["audio"]

        # prompt the model to misspell "Mr Quilter" as "Mr Quillter"
        whisper_prompt = "Mr. Quillter."
        prompt_ids = pipe.tokenizer.get_prompt_ids(whisper_prompt, return_tensors="pt").to(torch_device)

        unprompted_result = pipe(sample.copy())["text"]
        prompted_result = pipe(sample, generate_kwargs={"prompt_ids": prompt_ids})["text"]

        # fmt: off
        EXPECTED_UNPROMPTED_RESULT = " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins work is really Greek after all and can discover in it but little of rocky Ithaca. Lennils, pictures are a sort of upguards and atom paintings and Mason's exquisite itals are as national as a jingo poem. Mr. Birkut Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap on the back before he says like a shampoo or a Turkish bath. Next man"
        EXPECTED_PROMPTED_RESULT = " Mr. Quillter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quillter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins work is really great after all, and can discover in it but little of rocky Ithaca. Lennils, pictures are a sort of upguards and atom paintings, and Mason's exquisite itals are as national as a jingo poem. Mr. Birkut Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. Mr. John Collier gives his sitter a cheerful slap on the back before he says like a shampoo or a Turkish bath. Next man."
        # fmt: on

        self.assertEqual(unprompted_result, EXPECTED_UNPROMPTED_RESULT)
        self.assertEqual(prompted_result, EXPECTED_PROMPTED_RESULT)

    @require_torch
    @slow
    def test_whisper_longform(self):
        # fmt: off
        EXPECTED_RESULT = " Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on Saturday, Rusty Cargo, container down by the Wharf, and challenge toothless drifters to the godless bughouse lets of tournament that is my segment. MUSIC Meanwhile!"
        # fmt: on

        processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model = model.to(torch_device)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            device=torch_device,
            return_timestamps=True,  # to allow longform generation
        )

        ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        audio = ds[:1]["audio"]

        result = pipe(audio)[0]["text"]

        assert result == EXPECTED_RESULT

    @require_torch
    @slow
    def test_seamless_v2(self):
        pipe = pipeline(
            "automatic-speech-recognition",
            model="facebook/seamless-m4t-v2-large",
            device=torch_device,
        )

        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = dataset[0]["audio"]

        result = pipe(sample, generate_kwargs={"tgt_lang": "eng"})
        EXPECTED_RESULT = "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"

        assert result["text"] == EXPECTED_RESULT

    @require_torch
    @slow
    def test_chunking_and_timestamps(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            framework="pt",
            chunk_length_s=10.0,
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 10
        audio_tiled = np.tile(audio, n_repeats)
        output = speech_recognizer([audio_tiled], batch_size=2)
        self.assertEqual(output, [{"text": ("A MAN SAID TO THE UNIVERSE SIR I EXIST " * n_repeats).strip()}])

        output = speech_recognizer(audio, return_timestamps="char")
        self.assertEqual(audio.shape, (74_400,))
        self.assertEqual(speech_recognizer.feature_extractor.sampling_rate, 16_000)
        # The audio is 74_400 / 16_000 = 4.65s long.
        self.assertEqual(
            output,
            {
                "text": "A MAN SAID TO THE UNIVERSE SIR I EXIST",
                "chunks": [
                    {"text": "A", "timestamp": (0.6, 0.62)},
                    {"text": " ", "timestamp": (0.62, 0.66)},
                    {"text": "M", "timestamp": (0.68, 0.7)},
                    {"text": "A", "timestamp": (0.78, 0.8)},
                    {"text": "N", "timestamp": (0.84, 0.86)},
                    {"text": " ", "timestamp": (0.92, 0.98)},
                    {"text": "S", "timestamp": (1.06, 1.08)},
                    {"text": "A", "timestamp": (1.14, 1.16)},
                    {"text": "I", "timestamp": (1.16, 1.18)},
                    {"text": "D", "timestamp": (1.2, 1.24)},
                    {"text": " ", "timestamp": (1.24, 1.28)},
                    {"text": "T", "timestamp": (1.28, 1.32)},
                    {"text": "O", "timestamp": (1.34, 1.36)},
                    {"text": " ", "timestamp": (1.38, 1.42)},
                    {"text": "T", "timestamp": (1.42, 1.44)},
                    {"text": "H", "timestamp": (1.44, 1.46)},
                    {"text": "E", "timestamp": (1.46, 1.5)},
                    {"text": " ", "timestamp": (1.5, 1.56)},
                    {"text": "U", "timestamp": (1.58, 1.62)},
                    {"text": "N", "timestamp": (1.64, 1.68)},
                    {"text": "I", "timestamp": (1.7, 1.72)},
                    {"text": "V", "timestamp": (1.76, 1.78)},
                    {"text": "E", "timestamp": (1.84, 1.86)},
                    {"text": "R", "timestamp": (1.86, 1.9)},
                    {"text": "S", "timestamp": (1.96, 1.98)},
                    {"text": "E", "timestamp": (1.98, 2.02)},
                    {"text": " ", "timestamp": (2.02, 2.06)},
                    {"text": "S", "timestamp": (2.82, 2.86)},
                    {"text": "I", "timestamp": (2.94, 2.96)},
                    {"text": "R", "timestamp": (2.98, 3.02)},
                    {"text": " ", "timestamp": (3.06, 3.12)},
                    {"text": "I", "timestamp": (3.5, 3.52)},
                    {"text": " ", "timestamp": (3.58, 3.6)},
                    {"text": "E", "timestamp": (3.66, 3.68)},
                    {"text": "X", "timestamp": (3.68, 3.7)},
                    {"text": "I", "timestamp": (3.9, 3.92)},
                    {"text": "S", "timestamp": (3.94, 3.96)},
                    {"text": "T", "timestamp": (4.0, 4.02)},
                    {"text": " ", "timestamp": (4.06, 4.1)},
                ],
            },
        )
        output = speech_recognizer(audio, return_timestamps="word")
        self.assertEqual(
            output,
            {
                "text": "A MAN SAID TO THE UNIVERSE SIR I EXIST",
                "chunks": [
                    {"text": "A", "timestamp": (0.6, 0.62)},
                    {"text": "MAN", "timestamp": (0.68, 0.86)},
                    {"text": "SAID", "timestamp": (1.06, 1.24)},
                    {"text": "TO", "timestamp": (1.28, 1.36)},
                    {"text": "THE", "timestamp": (1.42, 1.5)},
                    {"text": "UNIVERSE", "timestamp": (1.58, 2.02)},
                    {"text": "SIR", "timestamp": (2.82, 3.02)},
                    {"text": "I", "timestamp": (3.5, 3.52)},
                    {"text": "EXIST", "timestamp": (3.66, 4.02)},
                ],
            },
        )
        output = speech_recognizer(audio, return_timestamps="word", chunk_length_s=2.0)
        self.assertEqual(
            output,
            {
                "text": "A MAN SAID TO THE UNIVERSE SIR I EXIST",
                "chunks": [
                    {"text": "A", "timestamp": (0.6, 0.62)},
                    {"text": "MAN", "timestamp": (0.68, 0.86)},
                    {"text": "SAID", "timestamp": (1.06, 1.24)},
                    {"text": "TO", "timestamp": (1.3, 1.36)},
                    {"text": "THE", "timestamp": (1.42, 1.48)},
                    {"text": "UNIVERSE", "timestamp": (1.58, 2.02)},
                    # Tiny change linked to chunking.
                    {"text": "SIR", "timestamp": (2.84, 3.02)},
                    {"text": "I", "timestamp": (3.5, 3.52)},
                    {"text": "EXIST", "timestamp": (3.66, 4.02)},
                ],
            },
        )
        # CTC models must specify return_timestamps type - cannot set `return_timestamps=True` blindly
        with self.assertRaisesRegex(
            ValueError,
            "^CTC can either predict character level timestamps, or word level timestamps. "
            "Set `return_timestamps='char'` or `return_timestamps='word'` as required.$",
        ):
            _ = speech_recognizer(audio, return_timestamps=True)

    @require_torch
    @slow
    def test_chunking_with_lm(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="patrickvonplaten/wav2vec2-base-100h-with-lm",
            chunk_length_s=10.0,
        )
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 10
        audio = np.tile(audio, n_repeats)
        output = speech_recognizer([audio], batch_size=2)
        expected_text = "A MAN SAID TO THE UNIVERSE SIR I EXIST " * n_repeats
        expected = [{"text": expected_text.strip()}]
        self.assertEqual(output, expected)

    @require_torch
    def test_chunk_iterator(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        inputs = torch.arange(100).long()
        outs = list(chunk_iter(inputs, feature_extractor, 100, 0, 0))

        self.assertEqual(len(outs), 1)
        self.assertEqual([o["stride"] for o in outs], [(100, 0, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 100)])
        self.assertEqual([o["is_last"] for o in outs], [True])

        # two chunks no stride
        outs = list(chunk_iter(inputs, feature_extractor, 50, 0, 0))
        self.assertEqual(len(outs), 2)
        self.assertEqual([o["stride"] for o in outs], [(50, 0, 0), (50, 0, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 50), (1, 50)])
        self.assertEqual([o["is_last"] for o in outs], [False, True])

        # two chunks incomplete last
        outs = list(chunk_iter(inputs, feature_extractor, 80, 0, 0))
        self.assertEqual(len(outs), 2)
        self.assertEqual([o["stride"] for o in outs], [(80, 0, 0), (20, 0, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 80), (1, 20)])
        self.assertEqual([o["is_last"] for o in outs], [False, True])

        # one chunk since first is also last, because it contains only data
        # in the right strided part we just mark that part as non stride
        # This test is specifically crafted to trigger a bug if next chunk
        # would be ignored by the fact that all the data would be
        # contained in the strided left data.
        outs = list(chunk_iter(inputs, feature_extractor, 105, 5, 5))
        self.assertEqual(len(outs), 1)
        self.assertEqual([o["stride"] for o in outs], [(100, 0, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 100)])
        self.assertEqual([o["is_last"] for o in outs], [True])

    @require_torch
    def test_chunk_iterator_stride(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        inputs = torch.arange(100).long()
        input_values = feature_extractor(inputs, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")[
            "input_values"
        ]
        outs = list(chunk_iter(inputs, feature_extractor, 100, 20, 10))
        self.assertEqual(len(outs), 1)
        self.assertEqual([o["stride"] for o in outs], [(100, 0, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 100)])
        self.assertEqual([o["is_last"] for o in outs], [True])

        outs = list(chunk_iter(inputs, feature_extractor, 80, 20, 10))
        self.assertEqual(len(outs), 2)
        self.assertEqual([o["stride"] for o in outs], [(80, 0, 10), (50, 20, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 80), (1, 50)])
        self.assertEqual([o["is_last"] for o in outs], [False, True])

        outs = list(chunk_iter(inputs, feature_extractor, 90, 20, 0))
        self.assertEqual(len(outs), 2)
        self.assertEqual([o["stride"] for o in outs], [(90, 0, 0), (30, 20, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 90), (1, 30)])

        outs = list(chunk_iter(inputs, feature_extractor, 36, 6, 6))
        self.assertEqual(len(outs), 4)
        self.assertEqual([o["stride"] for o in outs], [(36, 0, 6), (36, 6, 6), (36, 6, 6), (28, 6, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 36), (1, 36), (1, 36), (1, 28)])

        inputs = torch.LongTensor([i % 2 for i in range(100)])
        input_values = feature_extractor(inputs, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")[
            "input_values"
        ]
        outs = list(chunk_iter(inputs, feature_extractor, 30, 5, 5))
        self.assertEqual(len(outs), 5)
        self.assertEqual([o["stride"] for o in outs], [(30, 0, 5), (30, 5, 5), (30, 5, 5), (30, 5, 5), (20, 5, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 30), (1, 30), (1, 30), (1, 30), (1, 20)])
        self.assertEqual([o["is_last"] for o in outs], [False, False, False, False, True])
        # (0, 25)
        self.assertEqual(nested_simplify(input_values[:, :30]), nested_simplify(outs[0]["input_values"]))
        # (25, 45)
        self.assertEqual(nested_simplify(input_values[:, 20:50]), nested_simplify(outs[1]["input_values"]))
        # (45, 65)
        self.assertEqual(nested_simplify(input_values[:, 40:70]), nested_simplify(outs[2]["input_values"]))
        # (65, 85)
        self.assertEqual(nested_simplify(input_values[:, 60:90]), nested_simplify(outs[3]["input_values"]))
        # (85, 100)
        self.assertEqual(nested_simplify(input_values[:, 80:100]), nested_simplify(outs[4]["input_values"]))

    @require_torch
    def test_stride(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="hf-internal-testing/tiny-random-wav2vec2",
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 10)
        output = speech_recognizer({"raw": waveform, "stride": (0, 0), "sampling_rate": 16_000})
        self.assertEqual(output, {"text": "OB XB  B EB BB  B EB B OB X"})

        # 0 effective ids Just take the middle one
        output = speech_recognizer({"raw": waveform, "stride": (5000, 5000), "sampling_rate": 16_000})
        self.assertEqual(output, {"text": ""})

        # Only 1 arange.
        output = speech_recognizer({"raw": waveform, "stride": (0, 9000), "sampling_rate": 16_000})
        self.assertEqual(output, {"text": "OB"})

        # 2nd arange
        output = speech_recognizer({"raw": waveform, "stride": (1000, 8000), "sampling_rate": 16_000})
        self.assertEqual(output, {"text": "XB"})

    @slow
    @require_torch_accelerator
    def test_slow_unfinished_sequence(self):
        from transformers import GenerationConfig

        pipe = pipeline(
            "automatic-speech-recognition",
            model="vasista22/whisper-hindi-large-v2",
            device=torch_device,
        )

        # the audio is 4 seconds long
        audio = hf_hub_download("Narsil/asr_dummy", filename="hindi.ogg", repo_type="dataset")

        # Original model wasn't trained with timestamps and has incorrect generation config
        out = pipe(
            audio,
            return_timestamps=True,
            generate_kwargs={"generation_config": GenerationConfig.from_pretrained("openai/whisper-large-v2")},
        )
        self.assertEqual(
            out,
            {
                "text": "मिर्ची में कितने विभिन्न प्रजातियां हैं",
                "chunks": [{"timestamp": (0.58, None), "text": "मिर्ची में कितने विभिन्न प्रजातियां हैं"}],
            },
        )


def require_ffmpeg(test_case):
    """
    Decorator marking a test that requires FFmpeg.

    These tests are skipped when FFmpeg isn't installed.

    """
    import subprocess

    try:
        subprocess.check_output(["ffmpeg", "-h"], stderr=subprocess.DEVNULL)
        return test_case
    except Exception:
        return unittest.skip(reason="test requires ffmpeg")(test_case)


def bytes_iter(chunk_size, chunks):
    for i in range(chunks):
        yield bytes(range(i * chunk_size, (i + 1) * chunk_size))


@require_ffmpeg
class AudioUtilsTest(unittest.TestCase):
    def test_chunk_bytes_iter_too_big(self):
        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=2), 10, stride=(0, 0)))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02\x03\x04\x05", "stride": (0, 0)})
        with self.assertRaises(StopIteration):
            next(iter_)

    def test_chunk_bytes_iter(self):
        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=2), 3, stride=(0, 0)))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 0)})
        self.assertEqual(next(iter_), {"raw": b"\x03\x04\x05", "stride": (0, 0)})
        with self.assertRaises(StopIteration):
            next(iter_)

    def test_chunk_bytes_iter_stride(self):
        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=2), 3, stride=(1, 1)))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 1)})
        self.assertEqual(next(iter_), {"raw": b"\x01\x02\x03", "stride": (1, 1)})
        self.assertEqual(next(iter_), {"raw": b"\x02\x03\x04", "stride": (1, 1)})
        # This is finished, but the chunk_bytes doesn't know it yet.
        self.assertEqual(next(iter_), {"raw": b"\x03\x04\x05", "stride": (1, 1)})
        self.assertEqual(next(iter_), {"raw": b"\x04\x05", "stride": (1, 0)})
        with self.assertRaises(StopIteration):
            next(iter_)

    def test_chunk_bytes_iter_stride_stream(self):
        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=2), 5, stride=(1, 1), stream=True))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 0), "partial": True})
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02\x03\x04", "stride": (0, 1), "partial": False})
        self.assertEqual(next(iter_), {"raw": b"\x03\x04\x05", "stride": (1, 0), "partial": False})
        with self.assertRaises(StopIteration):
            next(iter_)

        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=3), 5, stride=(1, 1), stream=True))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 0), "partial": True})
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02\x03\x04", "stride": (0, 1), "partial": False})
        self.assertEqual(next(iter_), {"raw": b"\x03\x04\x05\x06\x07", "stride": (1, 1), "partial": False})
        self.assertEqual(next(iter_), {"raw": b"\x06\x07\x08", "stride": (1, 0), "partial": False})
        with self.assertRaises(StopIteration):
            next(iter_)

        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=3), 10, stride=(1, 1), stream=True))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 0), "partial": True})
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02\x03\x04\x05", "stride": (0, 0), "partial": True})
        self.assertEqual(
            next(iter_), {"raw": b"\x00\x01\x02\x03\x04\x05\x06\x07\x08", "stride": (0, 0), "partial": True}
        )
        self.assertEqual(
            next(iter_), {"raw": b"\x00\x01\x02\x03\x04\x05\x06\x07\x08", "stride": (0, 0), "partial": False}
        )
        with self.assertRaises(StopIteration):
            next(iter_)

    def test_ffmpeg_no_additional_args(self):
        mic = ffmpeg_microphone_live(16000, 2.0)
        mic.close()

    def test_ffmpeg_additional_args(self):
        mic = ffmpeg_microphone_live(16000, 2.0, ffmpeg_additional_args=["-nostdin"])
        mic.close()
