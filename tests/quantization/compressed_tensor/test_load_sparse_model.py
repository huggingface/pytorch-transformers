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
    model_sparse_uncompressed = "horheynm/llama2.c_stories15M_pruned_50.2of4_uncompressed"
    model_sparse_compressed = "horheynm/llama2.c_stories15M_pruned_50.2of4_compressed"

    prompt = "Paris is the capital of which country?"

    stubs = [model_sparse_uncompressed, model_sparse_compressed]

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_compressed_uncompressed_model_shapes(self):
        """
        Check that the weights are the same between
         uncompressed and compressed-decompressed model
        Sparse compressed modules' weights are "packed" and shape/value will
         differ
        """

        def _has_nested_attr(obj, attr_path):
            attrs = attr_path.split(".")
            for attr in attrs:
                if not hasattr(obj, attr):
                    return None
                obj = getattr(obj, attr)
            return obj

        from compressed_tensors.quantization.utils import iter_named_leaf_modules

        uncompressed_model = AutoModelForCausalLM.from_pretrained(
            self.model_sparse_uncompressed,
        )

        compressed_model_decompressed = AutoModelForCausalLM.from_pretrained(
            self.model_sparse_compressed,
        )

        for name, submodule in iter_named_leaf_modules(
            uncompressed_model,
        ):
            if comp_decomp_obj := _has_nested_attr(compressed_model_decompressed, name):
                if hasattr(submodule, "weight"):
                    assert torch.equal(submodule.weight, comp_decomp_obj.weight)

    def test_run_compressed_outputs_match(self):
        """Check that uncompressed and compressed-decompressed model outputs are the same"""

        from transformers import AutoTokenizer

        for stub in self.stubs:
            tokenizer = AutoTokenizer.from_pretrained(stub)
            input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids

            uncompressed_model = AutoModelForCausalLM.from_pretrained(
                self.model_sparse_uncompressed,
            )
            output_rc_true = uncompressed_model.generate(input_ids, max_new_tokens=100)

            compressed_model_decompressed = AutoModelForCausalLM.from_pretrained(
                self.model_sparse_compressed,
            )
            output_rc_false = compressed_model_decompressed.generate(input_ids, max_new_tokens=100)

            assert tokenizer.decode(output_rc_true[0]) == tokenizer.decode(output_rc_false[0])
