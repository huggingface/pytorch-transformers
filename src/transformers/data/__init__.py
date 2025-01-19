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

from .data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForMultipleChoice,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForSOP,
    DataCollatorForTokenClassification,
    DataCollatorForWholeWordMask,
    DataCollatorWithFlattening,
    DataCollatorWithPadding,
    DefaultDataCollator,
    default_data_collator,
)
from .metrics import glue_compute_metrics, xnli_compute_metrics
from .processors import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    squad_convert_examples_to_features,
    xnli_output_modes,
    xnli_processors,
    xnli_tasks_num_labels,
)
