# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team.
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

from tempfile import TemporaryDirectory

from transformers import TrainingArguments
from transformers.testing_utils import TestCasePlus, is_accelerate_available, require_accelerate


if is_accelerate_available():
    from accelerate.utils import patch_environment


@require_accelerate
class TrainingArgsTest(TestCasePlus):
    """
    Tests the core `TrainingArguments` class for pre and post processing.
    """

    def test_mixed_precision(self):
        with TemporaryDirectory() as temp_dir:
            # First with no env
            TrainingArguments(fp16=True, output_dir=temp_dir)
            args = TrainingArguments(output_dir=temp_dir, fp16=False)
            self.assertEqual(args.fp16, False)

            # Then with env
            with patch_environment(accelerate_mixed_precision="fp16"):
                args = TrainingArguments(output_dir=temp_dir)
                self.assertEqual(args.fp16, True)
