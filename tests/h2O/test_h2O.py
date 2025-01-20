import unittest

from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.cache_utils import H2OCache
from transformers.testing_utils import require_torch, torch_device


@require_torch
class TestH2OCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        cls.model = LlamaForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="eager",
        )
        cls.tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    @unittest.skipIf(torch_device == "cpu", "Requires CUDA")
    def test_h2o_cache_response(self):
        past_key_values = H2OCache(max_cache_len=50, device=torch_device)

        messages = [
            {"role": "system", "content": "You are a friendly chatbot."},
            {"role": "user", "content": "Tell me a joke."},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=50, past_key_values=past_key_values)
        response = self.tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertGreater(len(response.strip()), 0, "Response should not be empty.")


if __name__ == "__main__":
    unittest.main()
