import gc
import unittest

from transformers import AutoModelForCausalLM
from transformers.testing_utils import require_compressed_tensors, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_compressed_tensors
@require_torch
class CompressedTensorsTest(unittest.TestCase):
    tinyllama_w4a16 = "nm-testing/tinyllama-w4a16-compressed-hf-quantizer"
    tinyllama_w8a8 = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"

    prompt = "Paris is the capital of which country?"

    stubs = [tinyllama_w4a16, tinyllama_w8a8]

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_default_run_compressed__True(self):
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        from compressed_tensors.quantization.utils import iter_named_leaf_modules

        for stub in self.stubs:
            model = AutoModelForCausalLM.from_pretrained(
                stub,
            )
            compressed_linear_counts = 0

            for _, submodule in iter_named_leaf_modules(
                model,
            ):
                if isinstance(submodule, CompressedLinear):
                    compressed_linear_counts += 1

            # some linear models are not compressed - ex. lm_head
            assert compressed_linear_counts > 0

    def test_default_run_compressed__False(self):
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        from compressed_tensors.quantization.utils import iter_named_leaf_modules

        from transformers.utils.quantization_config import CompressedTensorsConfig

        quantization_config = CompressedTensorsConfig(run_compressed=False)

        for stub in self.stubs:
            model = AutoModelForCausalLM.from_pretrained(
                stub,
                quantization_config=quantization_config,
            )
            compressed_linear_counts = 0

            for _, submodule in iter_named_leaf_modules(
                model,
            ):
                if isinstance(submodule, CompressedLinear):
                    compressed_linear_counts += 1

            # No modules should be CompressedLinear
            assert compressed_linear_counts == 0

    def test_run_compressed_outputs_match(self):
        """Check that run_compressed=True/False output are the same"""

        from transformers import AutoTokenizer
        from transformers.utils.quantization_config import CompressedTensorsConfig

        quantization_config = CompressedTensorsConfig(run_compressed=False)

        for stub in self.stubs:
            tokenizer = AutoTokenizer.from_pretrained(stub)
            input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids

            model_run_compressed__True = AutoModelForCausalLM.from_pretrained(
                stub,
            )
            output_rc_true = model_run_compressed__True.generate(input_ids, max_new_tokens=100)

            model_run_compressed__False = AutoModelForCausalLM.from_pretrained(
                stub,
                quantization_config=quantization_config,
            )
            output_rc_false = model_run_compressed__False.generate(input_ids, max_new_tokens=100)

            assert tokenizer.decode(output_rc_true[0]) == tokenizer.decode(output_rc_false[0])
