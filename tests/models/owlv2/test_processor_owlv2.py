import shutil
import tempfile
import unittest

import pytest

from transformers import Owlv2Processor
from transformers.testing_utils import require_scipy

from ...test_processing_common import ProcessorTesterMixin


@require_scipy
class Owlv2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Owlv2Processor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained("google/owlv2-base-patch16-ensemble")
        processor.save_pretrained(self.tmpdirname)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_processor_query_images_positional(self):
        processor_components = self.prepare_components()
        processor = Owlv2Processor(**processor_components)

        image_input = self.prepare_image_inputs()
        query_images = self.prepare_image_inputs()

        inputs = processor(None, image_input, query_images)

        self.assertListEqual(list(inputs.keys()), ["query_pixel_values", "pixel_values"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()
