import unittest

import numpy as np

from transformers.generation.candidate_generator import AssistedCandidateGeneratorDifferentTokenizers


class TestAssistedCandidateGeneratorDifferentTokenizers(unittest.TestCase):
    def test_no_intersection(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[4, 5, 6]])
        result = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(prompt, prompt_plus_new_tokens)
        self.assertEqual(result, (None, None, None))

    def test_complete_overlap(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[1, 2, 3, 4, 5]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[4, 5]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))

    def test_partial_overlap(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[2, 3, 4, 5]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[4, 5]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))

    def test_no_new_tokens(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[1, 2, 3]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))
