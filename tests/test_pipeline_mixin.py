# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import copy
import inspect
import json
import os
import random
import re
import unittest
from dataclasses import fields, is_dataclass
from pathlib import Path
from textwrap import dedent
from typing import get_args

from huggingface_hub import (
    AudioClassificationInput,
    AutomaticSpeechRecognitionInput,
    DepthEstimationInput,
    ImageClassificationInput,
    ImageSegmentationInput,
    ImageToTextInput,
    ObjectDetectionInput,
    QuestionAnsweringInput,
    TableQuestionAnsweringInput,
    VideoClassificationInput,
    ZeroShotImageClassificationInput,
)

from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
from transformers.pipelines import (
    AudioClassificationPipeline,
    AutomaticSpeechRecognitionPipeline,
    DepthEstimationPipeline,
    ImageClassificationPipeline,
    ImageSegmentationPipeline,
    ImageToTextPipeline,
    ObjectDetectionPipeline,
    QuestionAnsweringPipeline,
    TableQuestionAnsweringPipeline,
    VideoClassificationPipeline,
    ZeroShotImageClassificationPipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    require_av,
    require_pytesseract,
    require_timm,
    require_torch,
    require_torch_or_tf,
    require_vision,
)
from transformers.utils import direct_transformers_import, logging

from .pipelines.test_pipelines_audio_classification import AudioClassificationPipelineTests
from .pipelines.test_pipelines_automatic_speech_recognition import AutomaticSpeechRecognitionPipelineTests
from .pipelines.test_pipelines_depth_estimation import DepthEstimationPipelineTests
from .pipelines.test_pipelines_document_question_answering import DocumentQuestionAnsweringPipelineTests
from .pipelines.test_pipelines_feature_extraction import FeatureExtractionPipelineTests
from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests
from .pipelines.test_pipelines_image_classification import ImageClassificationPipelineTests
from .pipelines.test_pipelines_image_feature_extraction import ImageFeatureExtractionPipelineTests
from .pipelines.test_pipelines_image_segmentation import ImageSegmentationPipelineTests
from .pipelines.test_pipelines_image_text_to_text import ImageTextToTextPipelineTests
from .pipelines.test_pipelines_image_to_image import ImageToImagePipelineTests
from .pipelines.test_pipelines_image_to_text import ImageToTextPipelineTests
from .pipelines.test_pipelines_mask_generation import MaskGenerationPipelineTests
from .pipelines.test_pipelines_object_detection import ObjectDetectionPipelineTests
from .pipelines.test_pipelines_question_answering import QAPipelineTests
from .pipelines.test_pipelines_summarization import SummarizationPipelineTests
from .pipelines.test_pipelines_table_question_answering import TQAPipelineTests
from .pipelines.test_pipelines_text2text_generation import Text2TextGenerationPipelineTests
from .pipelines.test_pipelines_text_classification import TextClassificationPipelineTests
from .pipelines.test_pipelines_text_generation import TextGenerationPipelineTests
from .pipelines.test_pipelines_text_to_audio import TextToAudioPipelineTests
from .pipelines.test_pipelines_token_classification import TokenClassificationPipelineTests
from .pipelines.test_pipelines_translation import TranslationPipelineTests
from .pipelines.test_pipelines_video_classification import VideoClassificationPipelineTests
from .pipelines.test_pipelines_visual_question_answering import VisualQuestionAnsweringPipelineTests
from .pipelines.test_pipelines_zero_shot import ZeroShotClassificationPipelineTests
from .pipelines.test_pipelines_zero_shot_audio_classification import ZeroShotAudioClassificationPipelineTests
from .pipelines.test_pipelines_zero_shot_image_classification import ZeroShotImageClassificationPipelineTests
from .pipelines.test_pipelines_zero_shot_object_detection import ZeroShotObjectDetectionPipelineTests


pipeline_test_mapping = {
    "audio-classification": {"test": AudioClassificationPipelineTests},
    "automatic-speech-recognition": {"test": AutomaticSpeechRecognitionPipelineTests},
    "depth-estimation": {"test": DepthEstimationPipelineTests},
    "document-question-answering": {"test": DocumentQuestionAnsweringPipelineTests},
    "feature-extraction": {"test": FeatureExtractionPipelineTests},
    "fill-mask": {"test": FillMaskPipelineTests},
    "image-classification": {"test": ImageClassificationPipelineTests},
    "image-feature-extraction": {"test": ImageFeatureExtractionPipelineTests},
    "image-segmentation": {"test": ImageSegmentationPipelineTests},
    "image-text-to-text": {"test": ImageTextToTextPipelineTests},
    "image-to-image": {"test": ImageToImagePipelineTests},
    "image-to-text": {"test": ImageToTextPipelineTests},
    "mask-generation": {"test": MaskGenerationPipelineTests},
    "object-detection": {"test": ObjectDetectionPipelineTests},
    "question-answering": {"test": QAPipelineTests},
    "summarization": {"test": SummarizationPipelineTests},
    "table-question-answering": {"test": TQAPipelineTests},
    "text2text-generation": {"test": Text2TextGenerationPipelineTests},
    "text-classification": {"test": TextClassificationPipelineTests},
    "text-generation": {"test": TextGenerationPipelineTests},
    "text-to-audio": {"test": TextToAudioPipelineTests},
    "token-classification": {"test": TokenClassificationPipelineTests},
    "translation": {"test": TranslationPipelineTests},
    "video-classification": {"test": VideoClassificationPipelineTests},
    "visual-question-answering": {"test": VisualQuestionAnsweringPipelineTests},
    "zero-shot": {"test": ZeroShotClassificationPipelineTests},
    "zero-shot-audio-classification": {"test": ZeroShotAudioClassificationPipelineTests},
    "zero-shot-image-classification": {"test": ZeroShotImageClassificationPipelineTests},
    "zero-shot-object-detection": {"test": ZeroShotObjectDetectionPipelineTests},
}

task_to_pipeline_and_spec_mapping = {
    # Adding a task to this list will cause its pipeline input signature to be checked against the corresponding
    # task spec in the HF Hub
    "audio-classification": (AudioClassificationPipeline, AudioClassificationInput),
    "automatic-speech-recognition": (AutomaticSpeechRecognitionPipeline, AutomaticSpeechRecognitionInput),
    "depth-estimation": (DepthEstimationPipeline, DepthEstimationInput),
    "image-classification": (ImageClassificationPipeline, ImageClassificationInput),
    "image-segmentation": (ImageSegmentationPipeline, ImageSegmentationInput),
    "image-to-text": (ImageToTextPipeline, ImageToTextInput),
    "object-detection": (ObjectDetectionPipeline, ObjectDetectionInput),
    "question-answering": (QuestionAnsweringPipeline, QuestionAnsweringInput),
    "table-question-answering": (TableQuestionAnsweringPipeline, TableQuestionAnsweringInput),
    "video-classification": (VideoClassificationPipeline, VideoClassificationInput),
    "zero-shot-image-classification": (ZeroShotImageClassificationPipeline, ZeroShotImageClassificationInput),
}

for task, task_info in pipeline_test_mapping.items():
    test = task_info["test"]
    task_info["mapping"] = {
        "pt": getattr(test, "model_mapping", None),
        "tf": getattr(test, "tf_model_mapping", None),
    }


# The default value `hf-internal-testing` is for running the pipeline testing against the tiny models on the Hub.
# For debugging purpose, we can specify a local path which is the `output_path` argument of a previous run of
# `utils/create_dummy_models.py`.
TRANSFORMERS_TINY_MODEL_PATH = os.environ.get("TRANSFORMERS_TINY_MODEL_PATH", "hf-internal-testing")
if TRANSFORMERS_TINY_MODEL_PATH == "hf-internal-testing":
    TINY_MODEL_SUMMARY_FILE_PATH = os.path.join(Path(__file__).parent.parent, "tests/utils/tiny_model_summary.json")
else:
    TINY_MODEL_SUMMARY_FILE_PATH = os.path.join(TRANSFORMERS_TINY_MODEL_PATH, "reports", "tiny_model_summary.json")
with open(TINY_MODEL_SUMMARY_FILE_PATH) as fp:
    tiny_model_summary = json.load(fp)


PATH_TO_TRANSFORMERS = os.path.join(Path(__file__).parent.parent, "src/transformers")


# Dynamically import the Transformers module to grab the attribute classes of the processor form their names.
transformers_module = direct_transformers_import(PATH_TO_TRANSFORMERS)

logger = logging.get_logger(__name__)


class PipelineTesterMixin:
    model_tester = None
    pipeline_model_mapping = None
    supported_frameworks = ["pt", "tf"]

    def run_task_tests(self, task, torch_dtype="float32"):
        """Run pipeline tests for a specific `task`

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
            torch_dtype (`str`, `optional`, defaults to `'float32'`):
                The torch dtype to use for the model. Can be used for FP16/other precision inference.
        """
        if task not in self.pipeline_model_mapping:
            self.skipTest(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')}_{torch_dtype} is skipped: `{task}` is not in "
                f"`self.pipeline_model_mapping` for `{self.__class__.__name__}`."
            )

        model_architectures = self.pipeline_model_mapping[task]
        if not isinstance(model_architectures, tuple):
            model_architectures = (model_architectures,)

        # We are going to run tests for multiple model architectures, some of them might be skipped
        # with this flag we are control if at least one model were tested or all were skipped
        at_least_one_model_is_tested = False

        for model_architecture in model_architectures:
            model_arch_name = model_architecture.__name__
            model_type = model_architecture.config_class.model_type

            # Get the canonical name
            for _prefix in ["Flax", "TF"]:
                if model_arch_name.startswith(_prefix):
                    model_arch_name = model_arch_name[len(_prefix) :]
                    break

            if model_arch_name not in tiny_model_summary:
                continue

            tokenizer_names = tiny_model_summary[model_arch_name]["tokenizer_classes"]

            # Sort image processors and feature extractors from tiny-models json file
            image_processor_names = []
            feature_extractor_names = []

            processor_classes = tiny_model_summary[model_arch_name]["processor_classes"]
            for cls_name in processor_classes:
                if "ImageProcessor" in cls_name:
                    image_processor_names.append(cls_name)
                elif "FeatureExtractor" in cls_name:
                    feature_extractor_names.append(cls_name)

            # Processor classes are not in tiny models JSON file, so extract them from the mapping
            # processors are mapped to instance, e.g. "XxxProcessor"
            processor_names = PROCESSOR_MAPPING_NAMES.get(model_type, None)
            if not isinstance(processor_names, (list, tuple)):
                processor_names = [processor_names]

            commit = None
            if model_arch_name in tiny_model_summary and "sha" in tiny_model_summary[model_arch_name]:
                commit = tiny_model_summary[model_arch_name]["sha"]

            repo_name = f"tiny-random-{model_arch_name}"
            if TRANSFORMERS_TINY_MODEL_PATH != "hf-internal-testing":
                repo_name = model_arch_name

            self.run_model_pipeline_tests(
                task,
                repo_name,
                model_architecture,
                tokenizer_names=tokenizer_names,
                image_processor_names=image_processor_names,
                feature_extractor_names=feature_extractor_names,
                processor_names=processor_names,
                commit=commit,
                torch_dtype=torch_dtype,
            )
            at_least_one_model_is_tested = True

        if task in task_to_pipeline_and_spec_mapping:
            pipeline, hub_spec = task_to_pipeline_and_spec_mapping[task]
            compare_pipeline_args_to_hub_spec(pipeline, hub_spec)

        if not at_least_one_model_is_tested:
            self.skipTest(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')}_{torch_dtype} is skipped: Could not find any "
                f"model architecture in the tiny models JSON file for `{task}`."
            )

    def run_model_pipeline_tests(
        self,
        task,
        repo_name,
        model_architecture,
        tokenizer_names,
        image_processor_names,
        feature_extractor_names,
        processor_names,
        commit,
        torch_dtype="float32",
    ):
        """Run pipeline tests for a specific `task` with the give model class and tokenizer/processor class names

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
            repo_name (`str`):
                A model repository id on the Hub.
            model_architecture (`type`):
                A subclass of `PretrainedModel` or `PretrainedModel`.
            tokenizer_names (`List[str]`):
                A list of names of a subclasses of `PreTrainedTokenizerFast` or `PreTrainedTokenizer`.
            image_processor_names (`List[str]`):
                A list of names of subclasses of `BaseImageProcessor`.
            feature_extractor_names (`List[str]`):
                A list of names of subclasses of `FeatureExtractionMixin`.
            processor_names (`List[str]`):
                A list of names of subclasses of `ProcessorMixin`.
            commit (`str`):
                The commit hash of the model repository on the Hub.
            torch_dtype (`str`, `optional`, defaults to `'float32'`):
                The torch dtype to use for the model. Can be used for FP16/other precision inference.
        """
        # Get an instance of the corresponding class `XXXPipelineTests` in order to use `get_test_pipeline` and
        # `run_pipeline_test`.
        pipeline_test_class_name = pipeline_test_mapping[task]["test"].__name__

        # If no image processor or feature extractor is found, we still need to test the pipeline with None
        # otherwise for any empty list we might skip all the tests
        tokenizer_names = tokenizer_names or [None]
        image_processor_names = image_processor_names or [None]
        feature_extractor_names = feature_extractor_names or [None]
        processor_names = processor_names or [None]

        test_cases = [
            {
                "tokenizer_name": tokenizer_name,
                "image_processor_name": image_processor_name,
                "feature_extractor_name": feature_extractor_name,
                "processor_name": processor_name,
            }
            for tokenizer_name in tokenizer_names
            for image_processor_name in image_processor_names
            for feature_extractor_name in feature_extractor_names
            for processor_name in processor_names
        ]

        for test_case in test_cases:
            tokenizer_name = test_case["tokenizer_name"]
            image_processor_name = test_case["image_processor_name"]
            feature_extractor_name = test_case["feature_extractor_name"]
            processor_name = test_case["processor_name"]

            do_skip_test_case = self.is_pipeline_test_to_skip(
                pipeline_test_class_name,
                model_architecture.config_class,
                model_architecture,
                tokenizer_name,
                image_processor_name,
                feature_extractor_name,
                processor_name,
            )

            if do_skip_test_case:
                logger.warning(
                    f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')}_{torch_dtype} is skipped: test is "
                    f"currently known to fail for: model `{model_architecture.__name__}` | tokenizer "
                    f"`{tokenizer_name}` | image processor `{image_processor_name}` | feature extractor {feature_extractor_name}."
                )
                continue

            self.run_pipeline_test(
                task,
                repo_name,
                model_architecture,
                tokenizer_name=tokenizer_name,
                image_processor_name=image_processor_name,
                feature_extractor_name=feature_extractor_name,
                processor_name=processor_name,
                commit=commit,
                torch_dtype=torch_dtype,
            )

    def run_pipeline_test(
        self,
        task,
        repo_name,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
        commit,
        torch_dtype="float32",
    ):
        """Run pipeline tests for a specific `task` with the give model class and tokenizer/processor class name

        The model will be loaded from a model repository on the Hub.

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
            repo_name (`str`):
                A model repository id on the Hub.
            model_architecture (`type`):
                A subclass of `PretrainedModel` or `PretrainedModel`.
            tokenizer_name (`str`):
                The name of a subclass of `PreTrainedTokenizerFast` or `PreTrainedTokenizer`.
            image_processor_name (`str`):
                The name of a subclass of `BaseImageProcessor`.
            feature_extractor_name (`str`):
                The name of a subclass of `FeatureExtractionMixin`.
            processor_name (`str`):
                The name of a subclass of `ProcessorMixin`.
            commit (`str`):
                The commit hash of the model repository on the Hub.
            torch_dtype (`str`, `optional`, defaults to `'float32'`):
                The torch dtype to use for the model. Can be used for FP16/other precision inference.
        """
        repo_id = f"{TRANSFORMERS_TINY_MODEL_PATH}/{repo_name}"
        model_type = model_architecture.config_class.model_type

        if TRANSFORMERS_TINY_MODEL_PATH != "hf-internal-testing":
            repo_id = os.path.join(TRANSFORMERS_TINY_MODEL_PATH, model_type, repo_name)

        # -------------------- Load model --------------------

        # TODO: We should check if a model file is on the Hub repo. instead.
        try:
            model = model_architecture.from_pretrained(repo_id, revision=commit)
        except Exception:
            logger.warning(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')}_{torch_dtype} is skipped: Could not find or load "
                f"the model from `{repo_id}` with `{model_architecture}`."
            )
            self.skipTest(f"Could not find or load the model from {repo_id} with {model_architecture}.")

        # -------------------- Load tokenizer --------------------

        tokenizer = None
        if tokenizer_name is not None:
            tokenizer_class = getattr(transformers_module, tokenizer_name)
            tokenizer = tokenizer_class.from_pretrained(repo_id, revision=commit)

        # -------------------- Load processors --------------------

        processors = {}
        for key, name in zip(
            ["image_processor", "feature_extractor", "processor"],
            [image_processor_name, feature_extractor_name, processor_name],
        ):
            if name is not None:
                try:
                    # Can fail if some extra dependencies are not installed
                    processor_class = getattr(transformers_module, name)
                    processor = processor_class.from_pretrained(repo_id, revision=commit)
                    processors[key] = processor
                except Exception:
                    logger.warning(
                        f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')}_{torch_dtype} is skipped: "
                        f"Could not load the {key} from `{repo_id}` with `{name}`."
                    )
                    self.skipTest(f"Could not load the {key} from {repo_id} with {name}.")

        # ---------------------------------------------------------

        # TODO: Maybe not upload such problematic tiny models to Hub.
        if tokenizer is None and "image_processor" not in processors and "feature_extractor" not in processors:
            logger.warning(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')}_{torch_dtype} is skipped: Could not find or load "
                f"any tokenizer / image processor / feature extractor from `{repo_id}`."
            )
            self.skipTest(f"Could not find or load any tokenizer / processor from {repo_id}.")

        pipeline_test_class_name = pipeline_test_mapping[task]["test"].__name__
        if self.is_pipeline_test_to_skip_more(pipeline_test_class_name, model.config, model, tokenizer, **processors):
            logger.warning(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')}_{torch_dtype} is skipped: test is "
                f"currently known to fail for: model `{model_architecture.__name__}` | tokenizer "
                f"`{tokenizer_name}` | image processor `{image_processor_name}` | feature extractor `{feature_extractor_name}`."
            )
            self.skipTest(
                f"Test is known to fail for: model `{model_architecture.__name__}` | tokenizer `{tokenizer_name}` "
                f"| image processor `{image_processor_name}` | feature extractor `{feature_extractor_name}`."
            )

        # validate
        validate_test_components(model, tokenizer)

        if hasattr(model, "eval"):
            model = model.eval()

        # Get an instance of the corresponding class `XXXPipelineTests` in order to use `get_test_pipeline` and
        # `run_pipeline_test`.
        task_test = pipeline_test_mapping[task]["test"]()

        pipeline, examples = task_test.get_test_pipeline(model, tokenizer, **processors, torch_dtype=torch_dtype)
        if pipeline is None:
            # The test can disable itself, but it should be very marginal
            # Concerns: Wav2Vec2ForCTC without tokenizer test (FastTokenizer don't exist)
            logger.warning(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')}_{torch_dtype} is skipped: Could not get the "
                "pipeline for testing."
            )
            self.skipTest(reason="Could not get the pipeline for testing.")

        task_test.run_pipeline_test(pipeline, examples)

        def run_batch_test(pipeline, examples):
            # Need to copy because `Conversation` are stateful
            if pipeline.tokenizer is not None and pipeline.tokenizer.pad_token_id is None:
                return  # No batching for this and it's OK

            # 10 examples with batch size 4 means there needs to be a unfinished batch
            # which is important for the unbatcher
            def data(n):
                for _ in range(n):
                    # Need to copy because Conversation object is mutated
                    yield copy.deepcopy(random.choice(examples))

            out = []
            for item in pipeline(data(10), batch_size=4):
                out.append(item)
            self.assertEqual(len(out), 10)

        run_batch_test(pipeline, examples)

    @is_pipeline_test
    def test_pipeline_audio_classification(self):
        self.run_task_tests(task="audio-classification")

    @is_pipeline_test
    @require_torch
    def test_pipeline_audio_classification_fp16(self):
        self.run_task_tests(task="audio-classification", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_automatic_speech_recognition(self):
        self.run_task_tests(task="automatic-speech-recognition")

    @is_pipeline_test
    @require_torch
    def test_pipeline_automatic_speech_recognition_fp16(self):
        self.run_task_tests(task="automatic-speech-recognition", torch_dtype="float16")

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_depth_estimation(self):
        self.run_task_tests(task="depth-estimation")

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_depth_estimation_fp16(self):
        self.run_task_tests(task="depth-estimation", torch_dtype="float16")

    @is_pipeline_test
    @require_pytesseract
    @require_torch
    @require_vision
    def test_pipeline_document_question_answering(self):
        self.run_task_tests(task="document-question-answering")

    @is_pipeline_test
    @require_pytesseract
    @require_torch
    @require_vision
    def test_pipeline_document_question_answering_fp16(self):
        self.run_task_tests(task="document-question-answering", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_feature_extraction(self):
        self.run_task_tests(task="feature-extraction")

    @is_pipeline_test
    @require_torch
    def test_pipeline_feature_extraction_fp16(self):
        self.run_task_tests(task="feature-extraction", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_fill_mask(self):
        self.run_task_tests(task="fill-mask")

    @is_pipeline_test
    @require_torch
    def test_pipeline_fill_mask_fp16(self):
        self.run_task_tests(task="fill-mask", torch_dtype="float16")

    @is_pipeline_test
    @require_torch_or_tf
    @require_vision
    def test_pipeline_image_classification(self):
        self.run_task_tests(task="image-classification")

    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_image_classification_fp16(self):
        self.run_task_tests(task="image-classification", torch_dtype="float16")

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_image_segmentation(self):
        self.run_task_tests(task="image-segmentation")

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_image_segmentation_fp16(self):
        self.run_task_tests(task="image-segmentation", torch_dtype="float16")

    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_image_text_to_text(self):
        self.run_task_tests(task="image-text-to-text")

    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_image_text_to_text_fp16(self):
        self.run_task_tests(task="image-text-to-text", torch_dtype="float16")

    @is_pipeline_test
    @require_vision
    def test_pipeline_image_to_text(self):
        self.run_task_tests(task="image-to-text")

    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_image_to_text_fp16(self):
        self.run_task_tests(task="image-to-text", torch_dtype="float16")

    @is_pipeline_test
    @require_timm
    @require_vision
    @require_torch
    def test_pipeline_image_feature_extraction(self):
        self.run_task_tests(task="image-feature-extraction")

    @is_pipeline_test
    @require_timm
    @require_vision
    @require_torch
    def test_pipeline_image_feature_extraction_fp16(self):
        self.run_task_tests(task="image-feature-extraction", torch_dtype="float16")

    @unittest.skip(reason="`run_pipeline_test` is currently not implemented.")
    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_mask_generation(self):
        self.run_task_tests(task="mask-generation")

    @unittest.skip(reason="`run_pipeline_test` is currently not implemented.")
    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_mask_generation_fp16(self):
        self.run_task_tests(task="mask-generation", torch_dtype="float16")

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_object_detection(self):
        self.run_task_tests(task="object-detection")

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_object_detection_fp16(self):
        self.run_task_tests(task="object-detection", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_question_answering(self):
        self.run_task_tests(task="question-answering")

    @is_pipeline_test
    @require_torch
    def test_pipeline_question_answering_fp16(self):
        self.run_task_tests(task="question-answering", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_summarization(self):
        self.run_task_tests(task="summarization")

    @is_pipeline_test
    @require_torch
    def test_pipeline_summarization_fp16(self):
        self.run_task_tests(task="summarization", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_table_question_answering(self):
        self.run_task_tests(task="table-question-answering")

    @is_pipeline_test
    @require_torch
    def test_pipeline_table_question_answering_fp16(self):
        self.run_task_tests(task="table-question-answering", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_text2text_generation(self):
        self.run_task_tests(task="text2text-generation")

    @is_pipeline_test
    @require_torch
    def test_pipeline_text2text_generation_fp16(self):
        self.run_task_tests(task="text2text-generation", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_text_classification(self):
        self.run_task_tests(task="text-classification")

    @is_pipeline_test
    @require_torch
    def test_pipeline_text_classification_fp16(self):
        self.run_task_tests(task="text-classification", torch_dtype="float16")

    @is_pipeline_test
    @require_torch_or_tf
    def test_pipeline_text_generation(self):
        self.run_task_tests(task="text-generation")

    @is_pipeline_test
    @require_torch
    def test_pipeline_text_generation_fp16(self):
        self.run_task_tests(task="text-generation", torch_dtype="float16")

    @is_pipeline_test
    @require_torch
    def test_pipeline_text_to_audio(self):
        self.run_task_tests(task="text-to-audio")

    @is_pipeline_test
    @require_torch
    def test_pipeline_text_to_audio_fp16(self):
        self.run_task_tests(task="text-to-audio", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_token_classification(self):
        self.run_task_tests(task="token-classification")

    @is_pipeline_test
    @require_torch
    def test_pipeline_token_classification_fp16(self):
        self.run_task_tests(task="token-classification", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_translation(self):
        self.run_task_tests(task="translation")

    @is_pipeline_test
    @require_torch
    def test_pipeline_translation_fp16(self):
        self.run_task_tests(task="translation", torch_dtype="float16")

    @is_pipeline_test
    @require_torch_or_tf
    @require_vision
    @require_av
    def test_pipeline_video_classification(self):
        self.run_task_tests(task="video-classification")

    @is_pipeline_test
    @require_vision
    @require_torch
    @require_av
    def test_pipeline_video_classification_fp16(self):
        self.run_task_tests(task="video-classification", torch_dtype="float16")

    @is_pipeline_test
    @require_torch
    @require_vision
    def test_pipeline_visual_question_answering(self):
        self.run_task_tests(task="visual-question-answering")

    @is_pipeline_test
    @require_torch
    @require_vision
    def test_pipeline_visual_question_answering_fp16(self):
        self.run_task_tests(task="visual-question-answering", torch_dtype="float16")

    @is_pipeline_test
    def test_pipeline_zero_shot(self):
        self.run_task_tests(task="zero-shot")

    @is_pipeline_test
    @require_torch
    def test_pipeline_zero_shot_fp16(self):
        self.run_task_tests(task="zero-shot", torch_dtype="float16")

    @is_pipeline_test
    @require_torch
    def test_pipeline_zero_shot_audio_classification(self):
        self.run_task_tests(task="zero-shot-audio-classification")

    @is_pipeline_test
    @require_torch
    def test_pipeline_zero_shot_audio_classification_fp16(self):
        self.run_task_tests(task="zero-shot-audio-classification", torch_dtype="float16")

    @is_pipeline_test
    @require_vision
    def test_pipeline_zero_shot_image_classification(self):
        self.run_task_tests(task="zero-shot-image-classification")

    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_zero_shot_image_classification_fp16(self):
        self.run_task_tests(task="zero-shot-image-classification", torch_dtype="float16")

    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_zero_shot_object_detection(self):
        self.run_task_tests(task="zero-shot-object-detection")

    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_zero_shot_object_detection_fp16(self):
        self.run_task_tests(task="zero-shot-object-detection", torch_dtype="float16")

    # This contains the test cases to be skipped without model architecture being involved.
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        """Skip some tests based on the classes or their names without the instantiated objects.

        This is to avoid calling `from_pretrained` (so reducing the runtime) if we already know the tests will fail.
        """
        # No fix is required for this case.
        if (
            pipeline_test_case_name == "DocumentQuestionAnsweringPipelineTests"
            and tokenizer_name is not None
            and not tokenizer_name.endswith("Fast")
        ):
            # `DocumentQuestionAnsweringPipelineTests` requires a fast tokenizer.
            return True

        return False

    def is_pipeline_test_to_skip_more(
        self,
        pipeline_test_case_name,
        config,
        model,
        tokenizer,
        image_processor=None,
        feature_extractor=None,
        processor=None,
    ):  # noqa
        """Skip some more tests based on the information from the instantiated objects."""
        # No fix is required for this case.
        if (
            pipeline_test_case_name == "QAPipelineTests"
            and tokenizer is not None
            and getattr(tokenizer, "pad_token", None) is None
            and not tokenizer.__class__.__name__.endswith("Fast")
        ):
            # `QAPipelineTests` doesn't work with a slow tokenizer that has no pad token.
            return True

        return False


def validate_test_components(model, tokenizer):
    # TODO: Move this to tiny model creation script
    # head-specific (within a model type) necessary changes to the config
    # 1. for `BlenderbotForCausalLM`
    if model.__class__.__name__ == "BlenderbotForCausalLM":
        model.config.encoder_no_repeat_ngram_size = 0

    # TODO: Change the tiny model creation script: don't create models with problematic tokenizers
    # Avoid `IndexError` in embedding layers
    CONFIG_WITHOUT_VOCAB_SIZE = ["CanineConfig"]
    if tokenizer is not None:
        # Removing `decoder=True` in `get_text_config` can lead to conflicting values e.g. in MusicGen
        config_vocab_size = getattr(model.config.get_text_config(decoder=True), "vocab_size", None)
        # For CLIP-like models
        if config_vocab_size is None:
            if hasattr(model.config, "text_encoder"):
                config_vocab_size = getattr(model.config.text_config, "vocab_size", None)
        if config_vocab_size is None and model.config.__class__.__name__ not in CONFIG_WITHOUT_VOCAB_SIZE:
            raise ValueError(
                "Could not determine `vocab_size` from model configuration while `tokenizer` is not `None`."
            )


def get_arg_names_from_hub_spec(hub_spec, first_level=True):
    # This util is used in pipeline tests, to verify that a pipeline's documented arguments
    # match the Hub specification for that task
    arg_names = []
    for field in fields(hub_spec):
        # Recurse into nested fields, but max one level
        if is_dataclass(field.type):
            arg_names.extend([field.name for field in fields(field.type)])
            continue
        # Next, catch nested fields that are part of a Union[], which is usually caused by Optional[]
        for param_type in get_args(field.type):
            if is_dataclass(param_type):
                # Again, recurse into nested fields, but max one level
                arg_names.extend([field.name for field in fields(param_type)])
                break
        else:
            # Finally, this line triggers if it's not a nested field
            arg_names.append(field.name)
    return arg_names


def parse_args_from_docstring_by_indentation(docstring):
    # This util is used in pipeline tests, to extract the argument names from a google-format docstring
    # to compare them against the Hub specification for that task. It uses indentation levels as a primary
    # source of truth, so these have to be correct!
    docstring = dedent(docstring)
    lines_by_indent = [
        (len(line) - len(line.lstrip()), line.strip()) for line in docstring.split("\n") if line.strip()
    ]
    args_lineno = None
    args_indent = None
    args_end = None
    for lineno, (indent, line) in enumerate(lines_by_indent):
        if line == "Args:":
            args_lineno = lineno
            args_indent = indent
            continue
        elif args_lineno is not None and indent == args_indent:
            args_end = lineno
            break
    if args_lineno is None:
        raise ValueError("No args block to parse!")
    elif args_end is None:
        args_block = lines_by_indent[args_lineno + 1 :]
    else:
        args_block = lines_by_indent[args_lineno + 1 : args_end]
    outer_indent_level = min(line[0] for line in args_block)
    outer_lines = [line for line in args_block if line[0] == outer_indent_level]
    arg_names = [re.match(r"(\w+)\W", line[1]).group(1) for line in outer_lines]
    return arg_names


def compare_pipeline_args_to_hub_spec(pipeline_class, hub_spec):
    ALLOWED_TRANSFORMERS_ONLY_ARGS = ["timeout"]

    docstring = inspect.getdoc(pipeline_class.__call__).strip()
    docstring_args = set(parse_args_from_docstring_by_indentation(docstring))
    hub_args = set(get_arg_names_from_hub_spec(hub_spec))

    # Special casing: We allow the name of this arg to differ
    js_generate_args = [js_arg for js_arg in hub_args if js_arg.startswith("generate")]
    docstring_generate_args = [
        docstring_arg for docstring_arg in docstring_args if docstring_arg.startswith("generate")
    ]
    if (
        len(js_generate_args) == 1
        and len(docstring_generate_args) == 1
        and js_generate_args != docstring_generate_args
    ):
        hub_args.remove(js_generate_args[0])
        docstring_args.remove(docstring_generate_args[0])

    # Special casing 2: We permit some transformers-only arguments that don't affect pipeline output
    for arg in ALLOWED_TRANSFORMERS_ONLY_ARGS:
        if arg in docstring_args and arg not in hub_args:
            docstring_args.remove(arg)

    if hub_args != docstring_args:
        error = [f"{pipeline_class.__name__} differs from JS spec {hub_spec.__name__}"]
        matching_args = hub_args & docstring_args
        huggingface_hub_only = hub_args - docstring_args
        transformers_only = docstring_args - hub_args
        if matching_args:
            error.append(f"Matching args: {matching_args}")
        if huggingface_hub_only:
            error.append(f"Huggingface Hub only: {huggingface_hub_only}")
        if transformers_only:
            error.append(f"Transformers only: {transformers_only}")
        raise ValueError("\n".join(error))
