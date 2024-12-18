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

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)


_import_structure = {
    "auto_factory": ["get_values"],
    "configuration_auto": ["CONFIG_MAPPING", "MODEL_NAMES_MAPPING", "AutoConfig"],
    "feature_extraction_auto": ["FEATURE_EXTRACTOR_MAPPING", "AutoFeatureExtractor"],
    "image_processing_auto": ["IMAGE_PROCESSOR_MAPPING", "AutoImageProcessor"],
    "processing_auto": ["PROCESSOR_MAPPING", "AutoProcessor"],
    "tokenization_auto": ["TOKENIZER_MAPPING", "AutoTokenizer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_auto"] = [
        "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
        "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING",
        "MODEL_FOR_AUDIO_XVECTOR_MAPPING",
        "MODEL_FOR_BACKBONE_MAPPING",
        "MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING",
        "MODEL_FOR_CAUSAL_LM_MAPPING",
        "MODEL_FOR_CTC_MAPPING",
        "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
        "MODEL_FOR_DEPTH_ESTIMATION_MAPPING",
        "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
        "MODEL_FOR_IMAGE_MAPPING",
        "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING",
        "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING",
        "MODEL_FOR_KEYPOINT_DETECTION_MAPPING",
        "MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING",
        "MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
        "MODEL_FOR_MASKED_LM_MAPPING",
        "MODEL_FOR_MASK_GENERATION_MAPPING",
        "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
        "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
        "MODEL_FOR_OBJECT_DETECTION_MAPPING",
        "MODEL_FOR_PRETRAINING_MAPPING",
        "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
        "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
        "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
        "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
        "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
        "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
        "MODEL_FOR_TEXT_ENCODING_MAPPING",
        "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING",
        "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING",
        "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
        "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING",
        "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING",
        "MODEL_FOR_VISION_2_SEQ_MAPPING",
        "MODEL_FOR_RETRIEVAL_MAPPING",
        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING",
        "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING",
        "MODEL_MAPPING",
        "MODEL_WITH_LM_HEAD_MAPPING",
        "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
        "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING",
        "MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING",
        "MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING",
        "AutoModel",
        "AutoBackbone",
        "AutoModelForAudioClassification",
        "AutoModelForAudioFrameClassification",
        "AutoModelForAudioXVector",
        "AutoModelForCausalLM",
        "AutoModelForCTC",
        "AutoModelForDepthEstimation",
        "AutoModelForImageClassification",
        "AutoModelForImageSegmentation",
        "AutoModelForImageToImage",
        "AutoModelForInstanceSegmentation",
        "AutoModelForKeypointDetection",
        "AutoModelForMaskGeneration",
        "AutoModelForTextEncoding",
        "AutoModelForMaskedImageModeling",
        "AutoModelForMaskedLM",
        "AutoModelForMultipleChoice",
        "AutoModelForNextSentencePrediction",
        "AutoModelForObjectDetection",
        "AutoModelForPreTraining",
        "AutoModelForQuestionAnswering",
        "AutoModelForSemanticSegmentation",
        "AutoModelForSeq2SeqLM",
        "AutoModelForSequenceClassification",
        "AutoModelForSpeechSeq2Seq",
        "AutoModelForTableQuestionAnswering",
        "AutoModelForTextToSpectrogram",
        "AutoModelForTextToWaveform",
        "AutoModelForTokenClassification",
        "AutoModelForUniversalSegmentation",
        "AutoModelForVideoClassification",
        "AutoModelForVision2Seq",
        "AutoModelForVisualQuestionAnswering",
        "AutoModelForDocumentQuestionAnswering",
        "AutoModelWithLMHead",
        "AutoModelForZeroShotImageClassification",
        "AutoModelForZeroShotObjectDetection",
        "AutoModelForImageTextToText",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_auto"] = [
        "TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
        "TF_MODEL_FOR_CAUSAL_LM_MAPPING",
        "TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
        "TF_MODEL_FOR_MASK_GENERATION_MAPPING",
        "TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
        "TF_MODEL_FOR_MASKED_LM_MAPPING",
        "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
        "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
        "TF_MODEL_FOR_PRETRAINING_MAPPING",
        "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
        "TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
        "TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
        "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
        "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
        "TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
        "TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
        "TF_MODEL_FOR_TEXT_ENCODING_MAPPING",
        "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
        "TF_MODEL_FOR_VISION_2_SEQ_MAPPING",
        "TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
        "TF_MODEL_MAPPING",
        "TF_MODEL_WITH_LM_HEAD_MAPPING",
        "TFAutoModel",
        "TFAutoModelForAudioClassification",
        "TFAutoModelForCausalLM",
        "TFAutoModelForImageClassification",
        "TFAutoModelForMaskedImageModeling",
        "TFAutoModelForMaskedLM",
        "TFAutoModelForMaskGeneration",
        "TFAutoModelForMultipleChoice",
        "TFAutoModelForNextSentencePrediction",
        "TFAutoModelForPreTraining",
        "TFAutoModelForDocumentQuestionAnswering",
        "TFAutoModelForQuestionAnswering",
        "TFAutoModelForSemanticSegmentation",
        "TFAutoModelForSeq2SeqLM",
        "TFAutoModelForSequenceClassification",
        "TFAutoModelForSpeechSeq2Seq",
        "TFAutoModelForTableQuestionAnswering",
        "TFAutoModelForTextEncoding",
        "TFAutoModelForTokenClassification",
        "TFAutoModelForVision2Seq",
        "TFAutoModelForZeroShotImageClassification",
        "TFAutoModelWithLMHead",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_auto"] = [
        "FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_CAUSAL_LM_MAPPING",
        "FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_MASKED_LM_MAPPING",
        "FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
        "FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
        "FLAX_MODEL_FOR_PRETRAINING_MAPPING",
        "FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
        "FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
        "FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
        "FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING",
        "FLAX_MODEL_MAPPING",
        "FlaxAutoModel",
        "FlaxAutoModelForCausalLM",
        "FlaxAutoModelForImageClassification",
        "FlaxAutoModelForMaskedLM",
        "FlaxAutoModelForMultipleChoice",
        "FlaxAutoModelForNextSentencePrediction",
        "FlaxAutoModelForPreTraining",
        "FlaxAutoModelForQuestionAnswering",
        "FlaxAutoModelForSeq2SeqLM",
        "FlaxAutoModelForSequenceClassification",
        "FlaxAutoModelForSpeechSeq2Seq",
        "FlaxAutoModelForTokenClassification",
        "FlaxAutoModelForVision2Seq",
    ]


if TYPE_CHECKING:
    from .auto_factory import get_values
    from .configuration_auto import CONFIG_MAPPING, MODEL_NAMES_MAPPING, AutoConfig
    from .feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
    from .image_processing_auto import IMAGE_PROCESSOR_MAPPING, AutoImageProcessor
    from .processing_auto import PROCESSOR_MAPPING, AutoProcessor
    from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_auto import (
            MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_XVECTOR_MAPPING,
            MODEL_FOR_BACKBONE_MAPPING,
            MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_CTC_MAPPING,
            MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
            MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_IMAGE_MAPPING,
            MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
            MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
            MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
            MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
            MODEL_FOR_MASK_GENERATION_MAPPING,
            MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_OBJECT_DETECTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_RETRIEVAL_MAPPING,
            MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_TEXT_ENCODING_MAPPING,
            MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
            MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
            MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
            MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
            MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
            MODEL_FOR_VISION_2_SEQ_MAPPING,
            MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
            MODEL_MAPPING,
            MODEL_WITH_LM_HEAD_MAPPING,
            AutoBackbone,
            AutoModel,
            AutoModelForAudioClassification,
            AutoModelForAudioFrameClassification,
            AutoModelForAudioXVector,
            AutoModelForCausalLM,
            AutoModelForCTC,
            AutoModelForDepthEstimation,
            AutoModelForDocumentQuestionAnswering,
            AutoModelForImageClassification,
            AutoModelForImageSegmentation,
            AutoModelForImageTextToText,
            AutoModelForImageToImage,
            AutoModelForInstanceSegmentation,
            AutoModelForKeypointDetection,
            AutoModelForMaskedImageModeling,
            AutoModelForMaskedLM,
            AutoModelForMaskGeneration,
            AutoModelForMultipleChoice,
            AutoModelForNextSentencePrediction,
            AutoModelForObjectDetection,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSemanticSegmentation,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForSpeechSeq2Seq,
            AutoModelForTableQuestionAnswering,
            AutoModelForTextEncoding,
            AutoModelForTextToSpectrogram,
            AutoModelForTextToWaveform,
            AutoModelForTokenClassification,
            AutoModelForUniversalSegmentation,
            AutoModelForVideoClassification,
            AutoModelForVision2Seq,
            AutoModelForVisualQuestionAnswering,
            AutoModelForZeroShotImageClassification,
            AutoModelForZeroShotObjectDetection,
            AutoModelWithLMHead,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_auto import (
            TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_MASK_GENERATION_MAPPING,
            TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            TF_MODEL_FOR_MASKED_LM_MAPPING,
            TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            TF_MODEL_FOR_PRETRAINING_MAPPING,
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_TEXT_ENCODING_MAPPING,
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
            TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_MAPPING,
            TF_MODEL_WITH_LM_HEAD_MAPPING,
            TFAutoModel,
            TFAutoModelForAudioClassification,
            TFAutoModelForCausalLM,
            TFAutoModelForDocumentQuestionAnswering,
            TFAutoModelForImageClassification,
            TFAutoModelForMaskedImageModeling,
            TFAutoModelForMaskedLM,
            TFAutoModelForMaskGeneration,
            TFAutoModelForMultipleChoice,
            TFAutoModelForNextSentencePrediction,
            TFAutoModelForPreTraining,
            TFAutoModelForQuestionAnswering,
            TFAutoModelForSemanticSegmentation,
            TFAutoModelForSeq2SeqLM,
            TFAutoModelForSequenceClassification,
            TFAutoModelForSpeechSeq2Seq,
            TFAutoModelForTableQuestionAnswering,
            TFAutoModelForTextEncoding,
            TFAutoModelForTokenClassification,
            TFAutoModelForVision2Seq,
            TFAutoModelForZeroShotImageClassification,
            TFAutoModelWithLMHead,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_auto import (
            FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_MASKED_LM_MAPPING,
            FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            FLAX_MODEL_FOR_PRETRAINING_MAPPING,
            FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
            FLAX_MODEL_MAPPING,
            FlaxAutoModel,
            FlaxAutoModelForCausalLM,
            FlaxAutoModelForImageClassification,
            FlaxAutoModelForMaskedLM,
            FlaxAutoModelForMultipleChoice,
            FlaxAutoModelForNextSentencePrediction,
            FlaxAutoModelForPreTraining,
            FlaxAutoModelForQuestionAnswering,
            FlaxAutoModelForSeq2SeqLM,
            FlaxAutoModelForSequenceClassification,
            FlaxAutoModelForSpeechSeq2Seq,
            FlaxAutoModelForTokenClassification,
            FlaxAutoModelForVision2Seq,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
