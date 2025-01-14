# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import shutil
import tempfile
import unittest

import numpy as np

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        AutoProcessor,
        ProPainterProcessor,
        ProPainterVideoProcessor,
    )


@require_vision
class ProPainterProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ProPainterProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        video_processor = ProPainterVideoProcessor()

        processor = ProPainterProcessor(video_processor=video_processor)
        processor.save_pretrained(self.tmpdirname)

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    def prepare_mask_inputs(self):
        """This function prepares a list of numpy arrays of masks for all the frames of videos."""
        mask_inputs = [np.random.randint(2, size=(1, 30, 400), dtype=np.uint8)] * 8
        mask_inputs = [mask_inputs] * 3  # batch-size=3
        return mask_inputs

    def test_video_processor(self):
        video_processor = self.get_video_processor()

        processor = ProPainterProcessor(video_processor=video_processor)

        video_input = self.prepare_video_inputs()
        mask_inptut = self.prepare_mask_inputs()

        input_video_proc = video_processor(video_input, masks=mask_inptut, return_tensors="np")
        input_processor = processor(videos=video_input, masks=mask_inptut, return_tensors="np")

        for key in input_video_proc.keys():
            self.assertAlmostEqual(input_video_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_model_input_names(self):
        video_processor = self.get_video_processor()

        processor = ProPainterProcessor(video_processor=video_processor)

        video_input = self.prepare_video_inputs()
        mask_inptut = self.prepare_mask_inputs()
        inputs = processor(videos=video_input, masks=mask_inptut, return_tensors="pt")

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)
