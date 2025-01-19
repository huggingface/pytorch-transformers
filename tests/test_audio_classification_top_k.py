import unittest

import numpy as np

from transformers import pipeline
from transformers.testing_utils import require_torch


@require_torch
class AudioClassificationTopKTest(unittest.TestCase):
    def test_top_k_none_returns_all_labels(self):
        model_name = "superb/wav2vec2-base-superb-ks"  # model with more than 5 labels
        classification_pipeline = pipeline(
            "audio-classification",
            model=model_name,
            top_k=None,
        )

        # Create dummy input
        sampling_rate = 16000
        signal = np.zeros((sampling_rate,), dtype=np.float32)

        result = classification_pipeline(signal)
        num_labels = classification_pipeline.model.config.num_labels

        self.assertEqual(len(result), num_labels, "Should return all labels when top_k is None")

    def test_top_k_none_with_few_labels(self):
        model_name = "superb/hubert-base-superb-er"  # model with fewer labels
        classification_pipeline = pipeline(
            "audio-classification",
            model=model_name,
            top_k=None,
        )

        # Create dummy input
        sampling_rate = 16000
        signal = np.zeros((sampling_rate,), dtype=np.float32)

        result = classification_pipeline(signal)
        num_labels = classification_pipeline.model.config.num_labels

        self.assertEqual(len(result), num_labels, "Should handle models with fewer labels correctly")

    def test_top_k_greater_than_labels(self):
        model_name = "superb/hubert-base-superb-er"
        classification_pipeline = pipeline(
            "audio-classification",
            model=model_name,
            top_k=100,  # intentionally large number
        )

        # Create dummy input
        sampling_rate = 16000
        signal = np.zeros((sampling_rate,), dtype=np.float32)

        result = classification_pipeline(signal)
        num_labels = classification_pipeline.model.config.num_labels

        self.assertEqual(len(result), num_labels, "Should cap top_k to number of labels")
