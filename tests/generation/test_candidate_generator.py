import gc
import threading
import unittest
import weakref
from unittest.mock import MagicMock

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.candidate_generator import (
    AssistantToTargetTranslator,
    AssistantVocabTranslatorCache,
    AssistedCandidateGeneratorDifferentTokenizers,
    UniversalSpeculativeDecodingGenerator,
)


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


class TestAssistantToTargetTranslator(unittest.TestCase):
    def setUp(self):
        # Create mock tokenizers with predefined vocabularies
        self.target_tokenizer = MagicMock()
        self.assistant_tokenizer = MagicMock()

        # Define mock vocabularies for the tokenizers
        self.target_vocab = {"hello": 0, "world": 1, "foo": 2, "bar": 3}
        self.assistant_vocab = {"hello": 0, "world": 1, "foo": 2, "baz": 4}

        self.target_tokenizer.get_vocab.return_value = self.target_vocab
        self.assistant_tokenizer.get_vocab.return_value = self.assistant_vocab
        self.assistant_model_device = "cpu"
        self.target_vocab_size = 6

        # Instantiate the class under test
        self.translator = AssistantToTargetTranslator(
            target_tokenizer=self.target_tokenizer,
            assistant_tokenizer=self.assistant_tokenizer,
            assistant_model_device=self.assistant_model_device,
            target_vocab_size=self.target_vocab_size,
        )

    def test_get_assistant_to_target_input_ids(self):
        """Test the mapping from assistant tokens to target tokens."""
        expected_mapping = [0, 1, 2, self.translator.suppress_tokens_id, self.translator.suppress_tokens_id]
        actual_mapping = self.translator._assistant_to_target_input_ids.tolist()
        self.assertEqual(actual_mapping, expected_mapping)

    def test_get_suppress_input_ids(self):
        """Test the suppression of assistant input IDs not present in the target vocabulary."""
        expected_suppress_ids = [3, 4]
        actual_suppress_ids = self.translator._get_suppress_input_ids().tolist()
        self.assertEqual(actual_suppress_ids, expected_suppress_ids)

    def test_get_target_ids(self):
        """Test the translation of assistant candidate IDs to target candidate IDs."""
        assistant_input_ids = torch.LongTensor([[0, 1, 2]])  # 'hello world foo' in assistant tokenizer
        target_input_ids = torch.LongTensor([[0, 1, 2]])  # 'hello world foo' in target tokenizer
        assistant_candidate_ids = torch.LongTensor([[0, 1, 2, 4]])  # 'hello world foo baz' in assistant tokenizer

        expected_target_ids = torch.LongTensor(
            [[0, 1, 2, self.translator.suppress_tokens_id]]
        )  # 'hello world foo baz' in target tokenizer (baz is mapped to self.translator.suppress_tokens_id since it does not exist in target vocab)

        actual_target_ids = self.translator.get_target_ids(
            assistant_input_ids, target_input_ids, assistant_candidate_ids
        )
        self.assertTrue(torch.equal(actual_target_ids, expected_target_ids))

    def test_get_target_logits(self):
        """Test the conversion of assistant logits to target logits."""
        # Assistant logits for IDs 0, 1, 2
        assistant_logits = torch.FloatTensor([[[0.1, 0.2, 0.3, 0.4, self.translator.filter_value]]])  # Shape (1, 1, 5)

        # Expected target logits (target_vocab_size = 4)
        expected_target_logits = torch.full((1, 1, self.target_vocab_size), self.translator.filter_value)
        expected_target_logits[0, 0, 0] = 0.1  # 'hello'
        expected_target_logits[0, 0, 1] = 0.2  # 'world'
        expected_target_logits[0, 0, 2] = 0.3  # 'foo'
        # The 'bar' token in target vocab remains at -inf

        actual_target_logits = self.translator.get_target_logits(assistant_logits)
        self.assertTrue(torch.equal(actual_target_logits, expected_target_logits))


class MockTokenizer:
    """A simple mock tokenizer class that supports weak references."""

    def __init__(self, vocab=None):
        self._vocab = vocab or {}

    def get_vocab(self):
        return self._vocab

    def __call__(self, text, add_special_tokens=True):
        # Mock implementation of the __call__ method
        tokens = text.split()
        input_ids = [self._vocab.get(token, 0) for token in tokens]
        return {"input_ids": input_ids}


class TestAssistantVocabTranslatorCache(unittest.TestCase):
    def setUp(self):
        # Clear the cache before each test
        AssistantVocabTranslatorCache._cache.clear()
        # Create mock tokenizers with different vocabularies
        self.target_tokenizer = MockTokenizer({"hello": 0, "world": 1})
        self.assistant_tokenizer = MockTokenizer({"hello": 0, "world": 1, "foo": 2})
        self.other_target_tokenizer = MockTokenizer({"foo": 2, "bar": 3})
        self.other_assistant_tokenizer = MockTokenizer({"baz": 4, "qux": 5})
        self.assistant_model_device = "cpu"
        self.target_vocab_size = 6

    def test_same_instance_for_same_tokenizers(self):
        """Test that the same translator is returned for the same tokenizers."""
        translator1 = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            assistant_model_device=self.assistant_model_device,
            target_vocab_size=self.target_vocab_size,
        )
        translator2 = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            assistant_model_device=self.assistant_model_device,
            target_vocab_size=self.target_vocab_size,
        )
        self.assertIs(translator1, translator2, "Translators should be cached and identical")

    def test_different_instances_for_different_tokenizers(self):
        """Test that different tokenizers produce different translators."""
        translator1 = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            assistant_model_device=self.assistant_model_device,
            target_vocab_size=self.target_vocab_size,
        )
        translator2 = AssistantVocabTranslatorCache.get_translator(
            self.other_target_tokenizer,
            self.other_assistant_tokenizer,
            assistant_model_device=self.assistant_model_device,
            target_vocab_size=self.target_vocab_size,
        )
        self.assertIsNot(translator1, translator2, "Translators should differ for different tokenizers")

    def test_cache_with_weakref_key(self):
        """Ensure that the cache uses weak references as keys."""
        initial_cache_size = len(AssistantVocabTranslatorCache._cache)
        target_tokenizer = MockTokenizer({"hello": 0})
        assistant_tokenizer = MockTokenizer({"hello": 0})

        # Store translator in a local variable to avoid it being kept alive
        translator = AssistantVocabTranslatorCache.get_translator(
            target_tokenizer,
            assistant_tokenizer,
            assistant_model_device=self.assistant_model_device,
            target_vocab_size=self.target_vocab_size,
        )
        self.assertEqual(len(AssistantVocabTranslatorCache._cache), initial_cache_size + 1)

        # Delete all strong references
        del target_tokenizer
        del assistant_tokenizer
        del translator

        # Force garbage collection
        gc.collect()

        # Call cleanup to remove dead entries
        AssistantVocabTranslatorCache.cleanup()

        # The cache size remains increased due to strong references
        self.assertEqual(len(AssistantVocabTranslatorCache._cache), initial_cache_size + 1)

    def test_weakref_cache_cleanup(self):
        """Test that the cache cleans up translators when tokenizers are garbage collected."""

        def create_translator():
            target_tokenizer = MockTokenizer({"hello": 0})
            assistant_tokenizer = MockTokenizer({"hello": 0})
            translator = AssistantVocabTranslatorCache.get_translator(
                target_tokenizer,
                assistant_tokenizer,
                assistant_model_device=self.assistant_model_device,
                target_vocab_size=self.target_vocab_size,
            )
            # Create weak references before returning
            refs = (weakref.ref(translator), weakref.ref(target_tokenizer), weakref.ref(assistant_tokenizer))
            # Remove strong references inside the function
            del target_tokenizer
            del assistant_tokenizer
            del translator
            return refs

        translator_ref, target_ref, assistant_ref = create_translator()

        # Force garbage collection
        gc.collect()

        # Call cleanup to remove dead entries
        AssistantVocabTranslatorCache.cleanup()

        # The tokenizers and translator are not garbage collected due to strong references
        self.assertIsNotNone(target_ref(), "Target tokenizer should still be alive due to strong references")
        self.assertIsNotNone(assistant_ref(), "Assistant tokenizer should still be alive due to strong references")
        self.assertIsNotNone(translator_ref(), "Translator should still be alive due to strong references")

    def test_thread_safety(self):
        """Test that get_translator is thread-safe."""
        translators = []

        def get_translator():
            translator = AssistantVocabTranslatorCache.get_translator(
                self.target_tokenizer,
                self.assistant_tokenizer,
                assistant_model_device=self.assistant_model_device,
                target_vocab_size=self.target_vocab_size,
            )
            translators.append(translator)

        threads = [threading.Thread(target=get_translator) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All translators should be the same instance
        for translator in translators:
            self.assertIs(translators[0], translator, "All translators should be identical across threads")


class TestUniversalSpeculativeDecoding(unittest.TestCase):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setUpClass(cls):
        cls.assistant_model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(
            cls.device
        )
        cls.main_tokenizer = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B-SFT")
        cls.assistant_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        cls.generation_config = GenerationConfig()

        # Ensure required tokens exist
        if cls.main_tokenizer.pad_token_id is None:
            cls.main_tokenizer.pad_token_id = cls.main_tokenizer.eos_token_id
        if cls.main_tokenizer.bos_token_id is None:
            cls.main_tokenizer.bos_token_id = cls.main_tokenizer.eos_token_id

    def setUp(self):
        self.input_ids = torch.tensor([[1, 2, 3]]).to(self.device)
        self.model_kwargs = {
            "attention_mask": torch.ones_like(self.input_ids).to(self.device),
        }
        self.generator = UniversalSpeculativeDecodingGenerator(
            input_ids=self.input_ids,
            assistant_model=self.assistant_model,
            target_tokenizer=self.main_tokenizer,
            assistant_tokenizer=self.assistant_tokenizer,
            generation_config=self.generation_config,
            model_kwargs=self.model_kwargs,
            target_vocab_size=self.main_tokenizer.vocab_size,
        )

    def test_basic_generation(self):
        """Test basic speculative decoding works"""
        input_text = "The quick brown fox"
        input_ids = self.main_tokenizer.encode(input_text, return_tensors="pt")
        self.generator.input_ids = input_ids
        candidates, scores = self.generator.get_candidates(input_ids)

        self.assertIsNotNone(candidates)
        self.assertIsNotNone(scores)
        self.assertTrue(torch.is_tensor(candidates))
        self.assertTrue(torch.is_tensor(scores))

    def test_mismatched_vocabularies(self):
        """Test handling of mismatched vocabularies between models"""
        # Create input with tokens present in main but not assistant vocab
        # Find a token that is not in the assistant tokenizer but in
        # the main tokenizer.
        missing_token = next(
            token
            for token in self.main_tokenizer.get_vocab()
            if token not in self.assistant_tokenizer.get_vocab()
            and token not in self.main_tokenizer.all_special_tokens
            and "reserved_" not in token
        )
        input_ids = torch.tensor([[self.main_tokenizer.convert_tokens_to_ids(missing_token)]])
        self.generator.input_ids = input_ids
        candidates, scores = self.generator.get_candidates(input_ids)
        self.assertIsNotNone(candidates)

    def test_speculation_depth(self):
        """Test different speculation depths"""
        input_ids = self.main_tokenizer.encode("Test text", return_tensors="pt")
        self.generator.input_ids = input_ids

        for depth in [1, 8, 17]:
            self.generator.num_assistant_tokens = depth
            candidates, scores = self.generator.get_candidates(input_ids)
            self.assertLessEqual(candidates.shape[1] - input_ids.shape[1], depth)

    def test_device_consistency(self):
        """Test handling of inputs on different devices"""
        if torch.cuda.is_available():
            input_ids = torch.tensor([[1, 2, 3]]).to(self.generator.assistant_model.device)
            self.generator.input_ids = input_ids
            candidates, scores = self.generator.get_candidates(input_ids)
            self.assertEqual(candidates.device, input_ids.device)
