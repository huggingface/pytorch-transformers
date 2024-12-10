# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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


from __future__ import annotations

import inspect
import json
import os
import random
import tempfile
import unittest
import unittest.mock as mock

from huggingface_hub import HfFolder, Repository, snapshot_download
from requests.exceptions import HTTPError

from transformers import is_tf_available, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.testing_utils import (  # noqa: F401
    TOKEN,
    USER,
    CaptureLogger,
    TemporaryHubRepo,
    _tf_gpu_memory_limit,
    is_pt_tf_cross_test,
    is_staging_test,
    require_safetensors,
    require_tf,
    require_torch,
    slow,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_INDEX_NAME,
    TF2_WEIGHTS_NAME,
    logging,
)


logger = logging.get_logger(__name__)


if is_tf_available():
    import h5py
    import numpy as np
    import tensorflow as tf

    from transformers import (
        BertConfig,
        PreTrainedModel,
        PushToHubCallback,
        RagRetriever,
        TFAutoModel,
        TFBertForMaskedLM,
        TFBertForSequenceClassification,
        TFBertModel,
        TFPreTrainedModel,
        TFRagModel,
    )
    from transformers.modeling_tf_utils import keras, tf_shard_checkpoint, unpack_inputs
    from transformers.tf_utils import stable_softmax

    tf.config.experimental.enable_tensor_float_32_execution(False)

    if _tf_gpu_memory_limit is not None:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            # Restrict TensorFlow to only allocate x GB of memory on the GPUs
            try:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=_tf_gpu_memory_limit)]
                )
                logical_gpus = tf.config.list_logical_devices("GPU")
                print("Logical GPUs", logical_gpus)
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

if is_torch_available():
    from transformers import BertModel


@require_tf
class TFModelUtilsTest(unittest.TestCase):
    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            _ = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
            # This check we did call the fake head request
            mock_head.assert_called()

    # tests whether the unpack_inputs function behaves as expected
    def test_unpack_inputs(self):
        class DummyModel:
            def __init__(self):
                config_kwargs = {"output_attentions": False, "output_hidden_states": False, "return_dict": False}
                self.config = PretrainedConfig(**config_kwargs)
                self.main_input_name = "input_ids"

            @unpack_inputs
            def call(
                self,
                input_ids=None,
                past_key_values=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
            ):
                return input_ids, past_key_values, output_attentions, output_hidden_states, return_dict

            @unpack_inputs
            def foo(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None):
                return pixel_values, output_attentions, output_hidden_states, return_dict

        dummy_model = DummyModel()
        input_ids = tf.constant([0, 1, 2, 3], dtype=tf.int32)
        past_key_values = tf.constant([4, 5, 6, 7], dtype=tf.int32)
        pixel_values = tf.constant([8, 9, 10, 11], dtype=tf.int32)

        # test case 1: Pass inputs as keyword arguments; Booleans are inherited from the config.
        output = dummy_model.call(input_ids=input_ids, past_key_values=past_key_values)
        tf.debugging.assert_equal(output[0], input_ids)
        tf.debugging.assert_equal(output[1], past_key_values)
        self.assertFalse(output[2])
        self.assertFalse(output[3])
        self.assertFalse(output[4])

        # test case 2: Same as above, but with positional arguments.
        output = dummy_model.call(input_ids, past_key_values)
        tf.debugging.assert_equal(output[0], input_ids)
        tf.debugging.assert_equal(output[1], past_key_values)
        self.assertFalse(output[2])
        self.assertFalse(output[3])
        self.assertFalse(output[4])

        # test case 3: We can also pack everything in the first input.
        output = dummy_model.call(input_ids={"input_ids": input_ids, "past_key_values": past_key_values})
        tf.debugging.assert_equal(output[0], input_ids)
        tf.debugging.assert_equal(output[1], past_key_values)
        self.assertFalse(output[2])
        self.assertFalse(output[3])
        self.assertFalse(output[4])

        # test case 4: Explicit boolean arguments should override the config.
        output = dummy_model.call(
            input_ids=input_ids, past_key_values=past_key_values, output_attentions=False, return_dict=True
        )
        tf.debugging.assert_equal(output[0], input_ids)
        tf.debugging.assert_equal(output[1], past_key_values)
        self.assertFalse(output[2])
        self.assertFalse(output[3])
        self.assertTrue(output[4])

        # test case 5: Unexpected arguments should raise an exception.
        with self.assertRaises(ValueError):
            output = dummy_model.call(input_ids=input_ids, past_key_values=past_key_values, foo="bar")

        # test case 6: the decorator is independent from `main_input_name` -- it treats the first argument of the
        # decorated function as its main input.
        output = dummy_model.foo(pixel_values=pixel_values)
        tf.debugging.assert_equal(output[0], pixel_values)
        self.assertFalse(output[1])
        self.assertFalse(output[2])
        self.assertFalse(output[3])

    # Tests whether the stable softmax is stable on CPU, with and without XLA
    def test_xla_stable_softmax(self):
        large_penalty = -1e9
        n_tokens = 10
        batch_size = 8

        def masked_softmax(x, boolean_mask):
            numerical_mask = (1.0 - tf.cast(boolean_mask, dtype=tf.float32)) * large_penalty
            masked_x = x + numerical_mask
            return stable_softmax(masked_x)

        xla_masked_softmax = tf.function(masked_softmax, jit_compile=True)
        xla_stable_softmax = tf.function(stable_softmax, jit_compile=True)
        x = tf.random.normal((batch_size, n_tokens))

        # Same outcome regardless of the boolean mask here
        masked_tokens = random.randint(0, n_tokens)
        boolean_mask = tf.convert_to_tensor([[1] * (n_tokens - masked_tokens) + [0] * masked_tokens], dtype=tf.int32)

        # We can randomly mask a random numerical input OUTSIDE XLA
        numerical_mask = (1.0 - tf.cast(boolean_mask, dtype=tf.float32)) * large_penalty
        masked_x = x + numerical_mask
        xla_out = xla_stable_softmax(masked_x)
        out = stable_softmax(masked_x)
        assert tf.experimental.numpy.allclose(xla_out, out)

        # The stable softmax has the same output as the original softmax
        unstable_out = tf.nn.softmax(masked_x)
        assert tf.experimental.numpy.allclose(unstable_out, out)

        # We can randomly mask a random numerical input INSIDE XLA
        xla_out = xla_masked_softmax(x, boolean_mask)
        out = masked_softmax(x, boolean_mask)
        assert tf.experimental.numpy.allclose(xla_out, out)

    def test_checkpoint_sharding_from_hub(self):
        model = TFBertModel.from_pretrained("ArthurZ/tiny-random-bert-sharded")
        # the model above is the same as the model below, just a sharded version.
        ref_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        for p1, p2 in zip(model.weights, ref_model.weights):
            assert np.allclose(p1.numpy(), p2.numpy())

    def test_sharded_checkpoint_with_prefix(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert", load_weight_prefix="a/b")
        sharded_model = TFBertModel.from_pretrained("ArthurZ/tiny-random-bert-sharded", load_weight_prefix="a/b")
        for p1, p2 in zip(model.weights, sharded_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))
            self.assertTrue(p1.name.startswith("a/b/"))
            self.assertTrue(p2.name.startswith("a/b/"))

    def test_sharded_checkpoint_transfer(self):
        # If this doesn't throw an error then the test passes
        TFBertForSequenceClassification.from_pretrained("ArthurZ/tiny-random-bert-sharded")

    @is_pt_tf_cross_test
    def test_checkpoint_sharding_local_from_pt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _ = Repository(local_dir=tmp_dir, clone_from="hf-internal-testing/tiny-random-bert-sharded")
            model = TFBertModel.from_pretrained(tmp_dir, from_pt=True)
            # the model above is the same as the model below, just a sharded pytorch version.
            ref_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
            for p1, p2 in zip(model.weights, ref_model.weights):
                assert np.allclose(p1.numpy(), p2.numpy())

    @is_pt_tf_cross_test
    def test_checkpoint_loading_with_prefix_from_pt(self):
        model = TFBertModel.from_pretrained(
            "hf-internal-testing/tiny-random-bert", from_pt=True, load_weight_prefix="a/b"
        )
        ref_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert", from_pt=True)
        for p1, p2 in zip(model.weights, ref_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))
            self.assertTrue(p1.name.startswith("a/b/"))

    @is_pt_tf_cross_test
    def test_checkpoint_sharding_hub_from_pt(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded", from_pt=True)
        # the model above is the same as the model below, just a sharded pytorch version.
        ref_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        for p1, p2 in zip(model.weights, ref_model.weights):
            assert np.allclose(p1.numpy(), p2.numpy())

    def test_shard_checkpoint(self):
        # This is the model we will use, total size 340,000 bytes.
        model = keras.Sequential(
            [
                keras.layers.Dense(200, use_bias=False),  # size 80,000
                keras.layers.Dense(200, use_bias=False),  # size 160,000
                keras.layers.Dense(100, use_bias=False),  # size 80,000
                keras.layers.Dense(50, use_bias=False),  # size 20,000
            ]
        )
        inputs = tf.zeros((1, 100), dtype=tf.float32)
        model(inputs)
        weights = model.weights
        weights_dict = {w.name: w for w in weights}
        with self.subTest("No shard when max size is bigger than model size"):
            shards, index = tf_shard_checkpoint(weights)
            self.assertIsNone(index)
            self.assertDictEqual(shards, {TF2_WEIGHTS_NAME: weights})

        with self.subTest("Test sharding, no weights bigger than max size"):
            shards, index = tf_shard_checkpoint(weights, max_shard_size="300kB")
            # Split is first two layers then last two.
            self.assertDictEqual(
                index,
                {
                    "metadata": {"total_size": 340000},
                    "weight_map": {
                        "dense/kernel:0": "tf_model-00001-of-00002.h5",
                        "dense_1/kernel:0": "tf_model-00001-of-00002.h5",
                        "dense_2/kernel:0": "tf_model-00002-of-00002.h5",
                        "dense_3/kernel:0": "tf_model-00002-of-00002.h5",
                    },
                },
            )

            shard1 = [weights_dict["dense/kernel:0"], weights_dict["dense_1/kernel:0"]]
            shard2 = [weights_dict["dense_2/kernel:0"], weights_dict["dense_3/kernel:0"]]
            self.assertDictEqual(shards, {"tf_model-00001-of-00002.h5": shard1, "tf_model-00002-of-00002.h5": shard2})

        with self.subTest("Test sharding with weights bigger than max size"):
            shards, index = tf_shard_checkpoint(weights, max_shard_size="100kB")
            # Split is first layer, second layer then last 2.
            self.assertDictEqual(
                index,
                {
                    "metadata": {"total_size": 340000},
                    "weight_map": {
                        "dense/kernel:0": "tf_model-00001-of-00003.h5",
                        "dense_1/kernel:0": "tf_model-00002-of-00003.h5",
                        "dense_2/kernel:0": "tf_model-00003-of-00003.h5",
                        "dense_3/kernel:0": "tf_model-00003-of-00003.h5",
                    },
                },
            )

            shard1 = [weights_dict["dense/kernel:0"]]
            shard2 = [weights_dict["dense_1/kernel:0"]]
            shard3 = [weights_dict["dense_2/kernel:0"], weights_dict["dense_3/kernel:0"]]
            self.assertDictEqual(
                shards,
                {
                    "tf_model-00001-of-00003.h5": shard1,
                    "tf_model-00002-of-00003.h5": shard2,
                    "tf_model-00003-of-00003.h5": shard3,
                },
            )

    @slow
    def test_special_layer_name_sharding(self):
        retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
        model = TFRagModel.from_pretrained("facebook/rag-token-nq", retriever=retriever)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for max_size in ["150kB", "150kiB", "200kB", "200kiB"]:
                model.save_pretrained(tmp_dir, max_shard_size=max_size)
                ref_model = TFRagModel.from_pretrained(tmp_dir, retriever=retriever)
                for p1, p2 in zip(model.weights, ref_model.weights):
                    assert np.allclose(p1.numpy(), p2.numpy())

    @require_safetensors
    def test_checkpoint_sharding_local(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # We use the same folder for various sizes to make sure a new save erases the old checkpoint.
            for max_size in ["150kB", "150kiB", "200kB", "200kiB"]:
                model.save_pretrained(tmp_dir, max_shard_size=max_size)

                # Get each shard file and its size
                shard_to_size = {}
                for shard in os.listdir(tmp_dir):
                    if shard.endswith(".h5"):
                        shard_file = os.path.join(tmp_dir, shard)
                        shard_to_size[shard_file] = os.path.getsize(shard_file)

                index_file = os.path.join(tmp_dir, TF2_WEIGHTS_INDEX_NAME)
                # Check there is an index but no regular weight file
                self.assertTrue(os.path.isfile(index_file))
                self.assertFalse(os.path.isfile(os.path.join(tmp_dir, TF2_WEIGHTS_NAME)))

                # Check a file is bigger than max_size only when it has a single weight
                for shard_file, size in shard_to_size.items():
                    if max_size.endswith("kiB"):
                        max_size_int = int(max_size[:-3]) * 2**10
                    else:
                        max_size_int = int(max_size[:-2]) * 10**3
                    # Note: pickle adds some junk so the weight of the file can end up being slightly bigger than
                    # the size asked for (since we count parameters)
                    if size >= max_size_int + 50000:
                        with h5py.File(shard_file, "r") as state_file:
                            self.assertEqual(len(state_file), 1)

                # Check the index and the shard files found match
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.loads(f.read())

                all_shards = set(index["weight_map"].values())
                shards_found = {f for f in os.listdir(tmp_dir) if f.endswith(".h5")}
                self.assertSetEqual(all_shards, shards_found)

                # Finally, check the model can be reloaded
                new_model = TFBertModel.from_pretrained(tmp_dir)

                model.build_in_name_scope()
                new_model.build_in_name_scope()

                for p1, p2 in zip(model.weights, new_model.weights):
                    self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    def test_safetensors_checkpoint_sharding_local(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # We use the same folder for various sizes to make sure a new save erases the old checkpoint.
            for max_size in ["150kB", "150kiB", "200kB", "200kiB"]:
                model.save_pretrained(tmp_dir, max_shard_size=max_size, safe_serialization=True)

                # Get each shard file and its size
                shard_to_size = {}
                for shard in os.listdir(tmp_dir):
                    if shard.endswith(".h5"):
                        shard_file = os.path.join(tmp_dir, shard)
                        shard_to_size[shard_file] = os.path.getsize(shard_file)

                index_file = os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)
                # Check there is an index but no regular weight file
                self.assertTrue(os.path.isfile(index_file))
                self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
                self.assertFalse(os.path.isfile(os.path.join(tmp_dir, TF2_WEIGHTS_NAME)))
                self.assertFalse(os.path.isfile(os.path.join(tmp_dir, TF2_WEIGHTS_INDEX_NAME)))

                # Check the index and the shard files found match
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.loads(f.read())

                all_shards = set(index["weight_map"].values())
                shards_found = {f for f in os.listdir(tmp_dir) if f.endswith(".safetensors")}
                self.assertSetEqual(all_shards, shards_found)

                # Finally, check the model can be reloaded
                new_model = TFBertModel.from_pretrained(tmp_dir)

                model.build_in_name_scope()
                new_model.build_in_name_scope()

                for p1, p2 in zip(model.weights, new_model.weights):
                    self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @is_pt_tf_cross_test
    @require_safetensors
    def test_bfloat16_torch_loading(self):
        # Assert that neither of these raise an error - both repos contain bfloat16 tensors
        model1 = TFAutoModel.from_pretrained("Rocketknight1/tiny-random-gpt2-bfloat16-pt", from_pt=True)
        model2 = TFAutoModel.from_pretrained("Rocketknight1/tiny-random-gpt2-bfloat16")  # PT-format safetensors
        # Check that PT and safetensors loading paths end up with the same values
        for weight1, weight2 in zip(model1.weights, model2.weights):
            self.assertTrue(tf.reduce_all(weight1 == weight2))

    @slow
    def test_save_pretrained_signatures(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Short custom TF signature function.
        # `input_signature` is specific to BERT.
        @tf.function(
            input_signature=[
                [
                    tf.TensorSpec([None, None], tf.int32, name="input_ids"),
                    tf.TensorSpec([None, None], tf.int32, name="token_type_ids"),
                    tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
                ]
            ]
        )
        def serving_fn(input):
            return model(input)

        # Using default signature (default behavior) overrides 'serving_default'
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, saved_model=True, signatures=None)
            model_loaded = keras.models.load_model(f"{tmp_dir}/saved_model/1")
            self.assertTrue("serving_default" in list(model_loaded.signatures.keys()))

        # Providing custom signature function
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, saved_model=True, signatures={"custom_signature": serving_fn})
            model_loaded = keras.models.load_model(f"{tmp_dir}/saved_model/1")
            self.assertTrue("custom_signature" in list(model_loaded.signatures.keys()))

        # Providing multiple custom signature function
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(
                tmp_dir,
                saved_model=True,
                signatures={"custom_signature_1": serving_fn, "custom_signature_2": serving_fn},
            )
            model_loaded = keras.models.load_model(f"{tmp_dir}/saved_model/1")
            self.assertTrue("custom_signature_1" in list(model_loaded.signatures.keys()))
            self.assertTrue("custom_signature_2" in list(model_loaded.signatures.keys()))

    @require_safetensors
    def test_safetensors_save_and_load(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            # No tf_model.h5 file, only a model.safetensors
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, TF2_WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, TF2_WEIGHTS_INDEX_NAME)))

            new_model = TFBertModel.from_pretrained(tmp_dir)

            # Check models are equal
            for p1, p2 in zip(model.weights, new_model.weights):
                self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_sharded_save_and_load(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="150kB")
            # No tf weights or index file, only a safetensors index
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, TF2_WEIGHTS_NAME)))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, TF2_WEIGHTS_INDEX_NAME)))

            new_model = TFBertModel.from_pretrained(tmp_dir)

            # Check models are equal
            for p1, p2 in zip(model.weights, new_model.weights):
                self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @is_pt_tf_cross_test
    def test_safetensors_save_and_load_pt_to_tf(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        pt_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pt_model.save_pretrained(tmp_dir, safe_serialization=True)
            # Check we have a model.safetensors file
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))

            new_model = TFBertModel.from_pretrained(tmp_dir)

            # Check models are equal
            for p1, p2 in zip(model.weights, new_model.weights):
                self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @is_pt_tf_cross_test
    def test_sharded_safetensors_save_and_load_pt_to_tf(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        pt_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pt_model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="150kB")
            # Check we have a safetensors shard index file
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))

            new_model = TFBertModel.from_pretrained(tmp_dir)

            # Check models are equal
            for p1, p2 in zip(model.weights, new_model.weights):
                self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_load_from_hub(self):
        tf_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Can load from the TF-formatted checkpoint
        safetensors_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert-safetensors-tf")

        # Check models are equal
        for p1, p2 in zip(safetensors_model.weights, tf_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

        # Can load from the PyTorch-formatted checkpoint
        safetensors_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-random-bert-safetensors")

        # Check models are equal
        for p1, p2 in zip(safetensors_model.weights, tf_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_tf_from_tf(self):
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-bert-tf-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = TFBertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.weights, new_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    @is_pt_tf_cross_test
    def test_safetensors_tf_from_torch(self):
        hub_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-bert-tf-only")
        model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = TFBertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(hub_model.weights, new_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_tf_from_sharded_h5_with_sharded_safetensors_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = snapshot_download("hf-internal-testing/tiny-bert-tf-safetensors-h5-sharded", cache_dir=tmp_dir)

            # This should not raise even if there are two types of sharded weights
            TFBertModel.from_pretrained(path)

    @require_safetensors
    def test_safetensors_tf_from_sharded_h5_with_sharded_safetensors_hub(self):
        # Confirm that we can correctly load the safetensors weights from a sharded hub repo even when TF weights present
        TFBertModel.from_pretrained("hf-internal-testing/tiny-bert-tf-safetensors-h5-sharded", use_safetensors=True)
        # Confirm that we can access the TF weights too
        TFBertModel.from_pretrained("hf-internal-testing/tiny-bert-tf-safetensors-h5-sharded", use_safetensors=False)

    @require_safetensors
    def test_safetensors_load_from_local(self):
        """
        This test checks that we can load safetensors from a checkpoint that only has those on the Hub
        """
        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download("hf-internal-testing/tiny-bert-tf-only", cache_dir=tmp)
            tf_model = TFBertModel.from_pretrained(location)

        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download("hf-internal-testing/tiny-bert-tf-safetensors-only", cache_dir=tmp)
            safetensors_model = TFBertModel.from_pretrained(location)

        for p1, p2 in zip(tf_model.weights, safetensors_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_load_from_hub_from_safetensors_pt(self):
        """
        This test checks that we can load safetensors from a checkpoint that only has those on the Hub.
        saved in the "pt" format.
        """
        tf_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-bert-h5")

        # Can load from the PyTorch-formatted checkpoint
        safetensors_model = TFBertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-safetensors")
        for p1, p2 in zip(tf_model.weights, safetensors_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_load_from_local_from_safetensors_pt(self):
        """
        This test checks that we can load safetensors from a local checkpoint that only has those
        saved in the "pt" format.
        """
        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download("hf-internal-testing/tiny-bert-h5", cache_dir=tmp)
            tf_model = TFBertModel.from_pretrained(location)

        # Can load from the PyTorch-formatted checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download("hf-internal-testing/tiny-bert-pt-safetensors", cache_dir=tmp)
            safetensors_model = TFBertModel.from_pretrained(location)

        for p1, p2 in zip(tf_model.weights, safetensors_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_load_from_hub_h5_before_safetensors(self):
        """
        This test checks that we'll first download h5 weights before safetensors
        The safetensors file on that repo is a pt safetensors and therefore cannot be loaded without PyTorch
        """
        TFBertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-safetensors-msgpack")

    @require_safetensors
    def test_safetensors_load_from_local_h5_before_safetensors(self):
        """
        This test checks that we'll first download h5 weights before safetensors
        The safetensors file on that repo is a pt safetensors and therefore cannot be loaded without PyTorch
        """
        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download("hf-internal-testing/tiny-bert-pt-safetensors-msgpack", cache_dir=tmp)
            TFBertModel.from_pretrained(location)


@require_tf
@is_staging_test
class TFModelPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = TFBertModel(config)
            # Make sure model is properly initialized
            model.build_in_name_scope()

            logging.set_verbosity_info()
            logger = logging.get_logger("transformers.utils.hub")
            with CaptureLogger(logger) as cl:
                model.push_to_hub(tmp_repo.repo_id, token=self._token)
            logging.set_verbosity_warning()
            # Check the model card was created and uploaded.
            self.assertIn("Uploading the following files to __DUMMY_TRANSFORMERS_USER__/test-model-tf", cl.out)

            new_model = TFBertModel.from_pretrained(tmp_repo.repo_id)
            models_equal = True
            for p1, p2 in zip(model.weights, new_model.weights):
                if not tf.math.reduce_all(p1 == p2):
                    models_equal = False
                    break
            self.assertTrue(models_equal)

    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = TFBertModel(config)
            # Make sure model is properly initialized
            model.build_in_name_scope()

            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_model = TFBertModel.from_pretrained(tmp_repo.repo_id)
            models_equal = True
            for p1, p2 in zip(model.weights, new_model.weights):
                if not tf.math.reduce_all(p1 == p2):
                    models_equal = False
                    break
            self.assertTrue(models_equal)

    @is_pt_tf_cross_test
    def test_push_to_hub_callback(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = TFBertForMaskedLM(config)
            model.compile()

            with tempfile.TemporaryDirectory() as tmp_dir:
                push_to_hub_callback = PushToHubCallback(
                    output_dir=tmp_dir,
                    hub_model_id=tmp_repo.repo_id,
                    hub_token=self._token,
                )
                model.fit(model.dummy_inputs, model.dummy_inputs, epochs=1, callbacks=[push_to_hub_callback])

            new_model = TFBertForMaskedLM.from_pretrained(tmp_repo.repo_id)
            models_equal = True
            for p1, p2 in zip(model.weights, new_model.weights):
                if not tf.math.reduce_all(p1 == p2):
                    models_equal = False
                    break
            self.assertTrue(models_equal)

            tf_push_to_hub_params = dict(inspect.signature(TFPreTrainedModel.push_to_hub).parameters)
            tf_push_to_hub_params.pop("base_model_card_args")
            pt_push_to_hub_params = dict(inspect.signature(PreTrainedModel.push_to_hub).parameters)
            pt_push_to_hub_params.pop("deprecated_kwargs")
            self.assertDictEaual(tf_push_to_hub_params, pt_push_to_hub_params)

    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = TFBertModel(config)
            # Make sure model is properly initialized
            model.build_in_name_scope()

            model.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_model = TFBertModel.from_pretrained(tmp_repo.repo_id)
            models_equal = True
            for p1, p2 in zip(model.weights, new_model.weights):
                if not tf.math.reduce_all(p1 == p2):
                    models_equal = False
                    break
            self.assertTrue(models_equal)

    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = TFBertModel(config)
            # Make sure model is properly initialized
            model.build_in_name_scope()

            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir, push_to_hub=True, token=self._token, repo_id=tmp_repo.repo_id)

            new_model = TFBertModel.from_pretrained(tmp_repo.repo_id)
            models_equal = True
            for p1, p2 in zip(model.weights, new_model.weights):
                if not tf.math.reduce_all(p1 == p2):
                    models_equal = False
                    break
            self.assertTrue(models_equal)
