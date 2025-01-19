import unittest

import torch

from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


class TestAttentionMaskIssue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "gpt2"
        cls.model = GPT2LMHeadModel.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

    def prepare_data(self, packing=False):
        texts = [
            "Hello, how are you?",
            "When is the next holiday?",
            "China is a great country.",
        ]
        encoded = self.tokenizer(texts)

        if packing:
            total_length = sum(len(x) for x in encoded["input_ids"])
            input_ids = torch.zeros((1, total_length), dtype=torch.long)
            # Create 4D attention mask with proper shape
            attention_mask = torch.full(
                (1, 1, total_length, total_length), dtype=torch.float32, fill_value=float("-inf")
            )

            offset = 0
            for i, (ids, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"])):
                length = len(ids)
                input_ids[0, offset : offset + length] = torch.tensor(ids)
                # Set valid attention positions to 0
                attention_mask[0, 0, offset : offset + length, : offset + length] = 0.0
                offset += length

            return input_ids, attention_mask
        else:
            # Regular batched processing
            max_length = max(len(x) for x in encoded["input_ids"])
            input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
            attention_mask = torch.zeros((len(texts), max_length), dtype=torch.long)

            for i, (ids, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"])):
                input_ids[i, : len(ids)] = torch.tensor(ids)
                attention_mask[i, : len(mask)] = torch.tensor(mask)

            return input_ids, attention_mask

    def test_attention_mask_shapes(self):
        # Test both regular and packed versions
        input_ids_regular, mask_regular = self.prepare_data(packing=False)
        output_regular = self.model(input_ids=input_ids_regular, attention_mask=mask_regular)

        input_ids_packed, mask_packed = self.prepare_data(packing=True)
        output_packed = self.model(input_ids=input_ids_packed, attention_mask=mask_packed)

        # Verify outputs have expected shapes
        self.assertEqual(output_regular.logits.shape[:-1], input_ids_regular.shape)
        self.assertEqual(output_packed.logits.shape[:-1], input_ids_packed.shape)

    def test_attention_patterns(self):
        # Test that attention patterns are preserved
        input_ids, mask_4d = self.prepare_data(packing=True)

        # Create equivalent 2D mask
        mask_2d = (mask_4d[:, 0, 0] > -1).float()

        # Compare outputs
        output_4d = self.model(input_ids, attention_mask=mask_4d)
        output_2d = self.model(input_ids, attention_mask=mask_2d)

        # Outputs should be nearly identical
        torch.testing.assert_close(output_4d.logits, output_2d.logits)

    def test_causal_attention(self):
        # Test causal attention is preserved with 4D masks
        input_ids, mask_4d = self.prepare_data(packing=True)
        outputs = self.model(input_ids, attention_mask=mask_4d, output_attentions=True)

        # Verify upper triangle is masked
        attentions = outputs.attentions[0]  # First layer
        upper_triangle = torch.triu(attentions, diagonal=1)
        assert torch.all(upper_triangle == 0)

    def test_batch_consistency(self):
        # Test consistency across different batch sizes
        input_ids, mask_4d = self.prepare_data(packing=True)

        # Single batch
        single_output = self.model(input_ids[:1], attention_mask=mask_4d[:1])

        # Multiple batches
        multi_output = self.model(input_ids, attention_mask=mask_4d)

        # First batch should give same results
        torch.testing.assert_close(single_output.logits, multi_output.logits[:1], rtol=1e-5, atol=1e-5)

    def test_edge_cases(self):
        # Test edge cases

        # 1. Empty sequence (just padding)
        empty_ids = torch.zeros((1, 10), dtype=torch.long)
        empty_mask = torch.full((1, 1, 10, 10), float("-inf"))
        outputs = self.model(empty_ids, attention_mask=empty_mask)
        self.assertEqual(outputs.logits.shape, (1, 10, self.model.config.vocab_size))

        # 2. Single token
        single_token = torch.tensor([[1]], dtype=torch.long)
        single_mask = torch.zeros((1, 1, 1, 1))
        outputs = self.model(single_token, attention_mask=single_mask)
        self.assertEqual(outputs.logits.shape, (1, 1, self.model.config.vocab_size))

        # 3. Maximum context length
        max_length = self.model.config.max_position_embeddings
        long_ids = torch.ones((1, max_length), dtype=torch.long)
        long_mask = torch.zeros((1, 1, max_length, max_length))
        outputs = self.model(long_ids, attention_mask=long_mask)
        self.assertEqual(outputs.logits.shape, (1, max_length, self.model.config.vocab_size))

    def test_4d_mask_handling(self):
        """Critical test: Verify 4D attention mask is handled correctly"""
        # Prepare packed sequence with 4D mask
        input_ids, mask_4d = self.prepare_packed_sequence()

        # Should run without errors and produce valid outputs
        try:
            outputs = self.model(input_ids, attention_mask=mask_4d)
            self.assertIsNotNone(outputs.logits)
        except Exception as e:
            self.fail(f"Failed to handle 4D mask: {e}")

    def test_2d_vs_4d_mask_behavior(self):
        """Test that 2D and 4D masks produce consistent behavior"""
        model = GPT2LMHeadModel.from_pretrained(self.model_name, attn_implementation="eager")

        # Create a simple sequence with both 2D and 4D masks
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        # 2D mask: [1, 1, 0, 0] (masking last two tokens)
        mask_2d = torch.tensor([[1, 1, 0, 0]], dtype=torch.float)

        # Equivalent 4D mask
        mask_4d = torch.full((1, 1, 4, 4), float("-inf"))
        mask_4d[0, 0, :2, :2] = 0  # Allow attention for first two tokens

        # Get outputs for both masks
        outputs_2d = model(input_ids, attention_mask=mask_2d, output_attentions=True)

        outputs_4d = model(input_ids, attention_mask=mask_4d, output_attentions=True)

        print("\n2D mask attention patterns:")
        print(outputs_2d.attentions[0][0, 0])  # First layer, first batch, first head

        print("\n4D mask attention patterns:")
        print(outputs_4d.attentions[0][0, 0])

        print("\n2D mask:")
        print(mask_2d)

        print("\n4D mask:")
        print(mask_4d[0, 0])

    def prepare_packed_sequence(self):
        """Helper to prepare a packed sequence with 4D attention mask"""
        texts = ["Hello world", "This is a test"]
        encoded = self.tokenizer(texts)

        total_length = sum(len(x) for x in encoded["input_ids"])
        input_ids = torch.zeros((1, total_length), dtype=torch.long)

        # Create 4D attention mask initialized to -inf
        mask_4d = torch.full((1, 1, total_length, total_length), float("-inf"), dtype=torch.float)

        offset = 0
        for ids in encoded["input_ids"]:
            length = len(ids)
            input_ids[0, offset : offset + length] = torch.tensor(ids)
            # Set valid attention positions to 0.0
            mask_4d[0, 0, offset : offset + length, offset : offset + length] = 0.0
            offset += length

        # Add debugging print
        print("Mask statistics:")
        print("- Total positions:", mask_4d.numel())
        print("- Masked positions:", (mask_4d == float("-inf")).sum().item())
        print("- Unmasked positions:", (mask_4d == 0).sum().item())

        return input_ids, mask_4d


if __name__ == "__main__":
    unittest.main()
