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

from typing import TYPE_CHECKING

from ..utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_tf_available, is_torch_available


_import_structure = {
    "configuration_utils": [
        "BaseWatermarkingConfig",
        "CompileConfig",
        "GenerationConfig",
        "GenerationMode",
        "SynthIDTextWatermarkingConfig",
        "WatermarkingConfig",
    ],
    "streamers": ["AsyncTextIteratorStreamer", "TextIteratorStreamer", "TextStreamer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["beam_constraints"] = [
        "Constraint",
        "ConstraintListState",
        "DisjunctiveConstraint",
        "PhrasalConstraint",
    ]
    _import_structure["beam_search"] = [
        "BeamHypotheses",
        "BeamScorer",
        "BeamSearchScorer",
        "ConstrainedBeamSearchScorer",
    ]
    _import_structure["candidate_generator"] = [
        "AssistedCandidateGenerator",
        "CandidateGenerator",
        "EarlyExitCandidateGenerator",
        "PromptLookupCandidateGenerator",
    ]
    _import_structure["logits_process"] = [
        "AlternatingCodebooksLogitsProcessor",
        "ClassifierFreeGuidanceLogitsProcessor",
        "EncoderNoRepeatNGramLogitsProcessor",
        "EncoderRepetitionPenaltyLogitsProcessor",
        "EpsilonLogitsWarper",
        "EtaLogitsWarper",
        "ExponentialDecayLengthPenalty",
        "ForcedBOSTokenLogitsProcessor",
        "ForcedEOSTokenLogitsProcessor",
        "HammingDiversityLogitsProcessor",
        "InfNanRemoveLogitsProcessor",
        "LogitNormalization",
        "LogitsProcessor",
        "LogitsProcessorList",
        "MinLengthLogitsProcessor",
        "MinNewTokensLengthLogitsProcessor",
        "MinPLogitsWarper",
        "NoBadWordsLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "SequenceBiasLogitsProcessor",
        "SuppressTokensLogitsProcessor",
        "SuppressTokensAtBeginLogitsProcessor",
        "SynthIDTextWatermarkLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
        "TypicalLogitsWarper",
        "UnbatchedClassifierFreeGuidanceLogitsProcessor",
        "WhisperTimeStampLogitsProcessor",
        "WatermarkLogitsProcessor",
    ]
    _import_structure["stopping_criteria"] = [
        "MaxLengthCriteria",
        "MaxTimeCriteria",
        "ConfidenceCriteria",
        "EosTokenCriteria",
        "StoppingCriteria",
        "StoppingCriteriaList",
        "validate_stopping_criteria",
        "StopStringCriteria",
    ]
    _import_structure["utils"] = [
        "GenerationMixin",
        "GreedySearchEncoderDecoderOutput",
        "GreedySearchDecoderOnlyOutput",
        "SampleEncoderDecoderOutput",
        "SampleDecoderOnlyOutput",
        "BeamSearchEncoderDecoderOutput",
        "BeamSearchDecoderOnlyOutput",
        "BeamSampleEncoderDecoderOutput",
        "BeamSampleDecoderOnlyOutput",
        "ContrastiveSearchEncoderDecoderOutput",
        "ContrastiveSearchDecoderOnlyOutput",
        "GenerateBeamDecoderOnlyOutput",
        "GenerateBeamEncoderDecoderOutput",
        "GenerateDecoderOnlyOutput",
        "GenerateEncoderDecoderOutput",
    ]
    _import_structure["watermarking"] = [
        "WatermarkDetector",
        "WatermarkDetectorOutput",
        "BayesianDetectorModel",
        "BayesianDetectorConfig",
        "SynthIDTextWatermarkDetector",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tf_logits_process"] = [
        "TFForcedBOSTokenLogitsProcessor",
        "TFForcedEOSTokenLogitsProcessor",
        "TFForceTokensLogitsProcessor",
        "TFLogitsProcessor",
        "TFLogitsProcessorList",
        "TFLogitsWarper",
        "TFMinLengthLogitsProcessor",
        "TFNoBadWordsLogitsProcessor",
        "TFNoRepeatNGramLogitsProcessor",
        "TFRepetitionPenaltyLogitsProcessor",
        "TFSuppressTokensAtBeginLogitsProcessor",
        "TFSuppressTokensLogitsProcessor",
        "TFTemperatureLogitsWarper",
        "TFTopKLogitsWarper",
        "TFTopPLogitsWarper",
    ]
    _import_structure["tf_utils"] = [
        "TFGenerationMixin",
        "TFGreedySearchDecoderOnlyOutput",
        "TFGreedySearchEncoderDecoderOutput",
        "TFSampleEncoderDecoderOutput",
        "TFSampleDecoderOnlyOutput",
        "TFBeamSearchEncoderDecoderOutput",
        "TFBeamSearchDecoderOnlyOutput",
        "TFBeamSampleEncoderDecoderOutput",
        "TFBeamSampleDecoderOnlyOutput",
        "TFContrastiveSearchEncoderDecoderOutput",
        "TFContrastiveSearchDecoderOnlyOutput",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["flax_logits_process"] = [
        "FlaxForcedBOSTokenLogitsProcessor",
        "FlaxForcedEOSTokenLogitsProcessor",
        "FlaxForceTokensLogitsProcessor",
        "FlaxLogitsProcessor",
        "FlaxLogitsProcessorList",
        "FlaxLogitsWarper",
        "FlaxMinLengthLogitsProcessor",
        "FlaxSuppressTokensAtBeginLogitsProcessor",
        "FlaxSuppressTokensLogitsProcessor",
        "FlaxTemperatureLogitsWarper",
        "FlaxTopKLogitsWarper",
        "FlaxTopPLogitsWarper",
        "FlaxWhisperTimeStampLogitsProcessor",
        "FlaxNoRepeatNGramLogitsProcessor",
    ]
    _import_structure["flax_utils"] = [
        "FlaxGenerationMixin",
        "FlaxGreedySearchOutput",
        "FlaxSampleOutput",
        "FlaxBeamSearchOutput",
    ]

if TYPE_CHECKING:
    from .configuration_utils import (
        BaseWatermarkingConfig,
        CompileConfig,
        GenerationConfig,
        GenerationMode,
        SynthIDTextWatermarkingConfig,
        WatermarkingConfig,
    )
    from .streamers import AsyncTextIteratorStreamer, TextIteratorStreamer, TextStreamer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .beam_constraints import Constraint, ConstraintListState, DisjunctiveConstraint, PhrasalConstraint
        from .beam_search import BeamHypotheses, BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
        from .candidate_generator import (
            AssistedCandidateGenerator,
            CandidateGenerator,
            EarlyExitCandidateGenerator,
            PromptLookupCandidateGenerator,
        )
        from .logits_process import (
            AlternatingCodebooksLogitsProcessor,
            ClassifierFreeGuidanceLogitsProcessor,
            EncoderNoRepeatNGramLogitsProcessor,
            EncoderRepetitionPenaltyLogitsProcessor,
            EpsilonLogitsWarper,
            EtaLogitsWarper,
            ExponentialDecayLengthPenalty,
            ForcedBOSTokenLogitsProcessor,
            ForcedEOSTokenLogitsProcessor,
            HammingDiversityLogitsProcessor,
            InfNanRemoveLogitsProcessor,
            LogitNormalization,
            LogitsProcessor,
            LogitsProcessorList,
            MinLengthLogitsProcessor,
            MinNewTokensLengthLogitsProcessor,
            MinPLogitsWarper,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SequenceBiasLogitsProcessor,
            SuppressTokensAtBeginLogitsProcessor,
            SuppressTokensLogitsProcessor,
            SynthIDTextWatermarkLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TypicalLogitsWarper,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
            WatermarkLogitsProcessor,
            WhisperTimeStampLogitsProcessor,
        )
        from .stopping_criteria import (
            ConfidenceCriteria,
            EosTokenCriteria,
            MaxLengthCriteria,
            MaxTimeCriteria,
            StoppingCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
            validate_stopping_criteria,
        )
        from .utils import (
            BeamSampleDecoderOnlyOutput,
            BeamSampleEncoderDecoderOutput,
            BeamSearchDecoderOnlyOutput,
            BeamSearchEncoderDecoderOutput,
            ContrastiveSearchDecoderOnlyOutput,
            ContrastiveSearchEncoderDecoderOutput,
            GenerateBeamDecoderOnlyOutput,
            GenerateBeamEncoderDecoderOutput,
            GenerateDecoderOnlyOutput,
            GenerateEncoderDecoderOutput,
            GenerationMixin,
            GreedySearchDecoderOnlyOutput,
            GreedySearchEncoderDecoderOutput,
            SampleDecoderOnlyOutput,
            SampleEncoderDecoderOutput,
        )
        from .watermarking import (
            BayesianDetectorConfig,
            BayesianDetectorModel,
            SynthIDTextWatermarkDetector,
            WatermarkDetector,
            WatermarkDetectorOutput,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tf_logits_process import (
            TFForcedBOSTokenLogitsProcessor,
            TFForcedEOSTokenLogitsProcessor,
            TFForceTokensLogitsProcessor,
            TFLogitsProcessor,
            TFLogitsProcessorList,
            TFLogitsWarper,
            TFMinLengthLogitsProcessor,
            TFNoBadWordsLogitsProcessor,
            TFNoRepeatNGramLogitsProcessor,
            TFRepetitionPenaltyLogitsProcessor,
            TFSuppressTokensAtBeginLogitsProcessor,
            TFSuppressTokensLogitsProcessor,
            TFTemperatureLogitsWarper,
            TFTopKLogitsWarper,
            TFTopPLogitsWarper,
        )
        from .tf_utils import (
            TFBeamSampleDecoderOnlyOutput,
            TFBeamSampleEncoderDecoderOutput,
            TFBeamSearchDecoderOnlyOutput,
            TFBeamSearchEncoderDecoderOutput,
            TFContrastiveSearchDecoderOnlyOutput,
            TFContrastiveSearchEncoderDecoderOutput,
            TFGenerationMixin,
            TFGreedySearchDecoderOnlyOutput,
            TFGreedySearchEncoderDecoderOutput,
            TFSampleDecoderOnlyOutput,
            TFSampleEncoderDecoderOutput,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .flax_logits_process import (
            FlaxForcedBOSTokenLogitsProcessor,
            FlaxForcedEOSTokenLogitsProcessor,
            FlaxForceTokensLogitsProcessor,
            FlaxLogitsProcessor,
            FlaxLogitsProcessorList,
            FlaxLogitsWarper,
            FlaxMinLengthLogitsProcessor,
            FlaxNoRepeatNGramLogitsProcessor,
            FlaxSuppressTokensAtBeginLogitsProcessor,
            FlaxSuppressTokensLogitsProcessor,
            FlaxTemperatureLogitsWarper,
            FlaxTopKLogitsWarper,
            FlaxTopPLogitsWarper,
            FlaxWhisperTimeStampLogitsProcessor,
        )
        from .flax_utils import FlaxBeamSearchOutput, FlaxGenerationMixin, FlaxGreedySearchOutput, FlaxSampleOutput
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
