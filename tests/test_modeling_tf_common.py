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

import copy
import inspect
import json
import os
import random
import tempfile
import unittest
from importlib import import_module
from math import isnan
from typing import List, Tuple

from datasets import Dataset

from transformers import is_tf_available, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import (  # noqa: F401
    CaptureLogger,
    _tf_gpu_memory_limit,
    is_pt_tf_cross_test,
    require_tf,
    require_tf2onnx,
    slow,
    torch_device,
)
from transformers.utils import CONFIG_NAME, GENERATION_CONFIG_NAME, logging
from transformers.utils.generic import ModelOutput


logger = logging.get_logger(__name__)


if is_tf_available():
    import numpy as np
    import tensorflow as tf

    from transformers import (
        TF_MODEL_FOR_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
        TF_MODEL_FOR_MASKED_LM_MAPPING,
        TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        TF_MODEL_FOR_PRETRAINING_MAPPING,
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TFAutoModel,
        TFAutoModelForSequenceClassification,
        TFSharedEmbeddings,
    )
    from transformers.generation import (
        TFBeamSampleDecoderOnlyOutput,
        TFBeamSampleEncoderDecoderOutput,
        TFBeamSearchDecoderOnlyOutput,
        TFBeamSearchEncoderDecoderOutput,
        TFGreedySearchDecoderOnlyOutput,
        TFGreedySearchEncoderDecoderOutput,
        TFSampleDecoderOnlyOutput,
        TFSampleEncoderDecoderOutput,
    )
    from transformers.modeling_tf_utils import keras

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
    import torch


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key:
            setattr(configs_no_init, key, 0.0)
    return configs_no_init


@require_tf
class TFModelTesterMixin:
    model_tester = None
    all_model_classes = ()
    all_generative_model_classes = ()
    test_mismatched_shapes = True
    test_resize_embeddings = True
    test_head_masking = True
    is_encoder_decoder = False
    has_attentions = True

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False) -> dict:
        inputs_dict = copy.deepcopy(inputs_dict)

        if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
            inputs_dict = {
                k: tf.tile(tf.expand_dims(v, 1), (1, self.model_tester.num_choices) + (1,) * (v.ndim - 1))
                if isinstance(v, tf.Tensor) and v.ndim > 0
                else v
                for k, v in inputs_dict.items()
            }

        if return_labels:
            if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = tf.ones(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [
                *get_values(TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING),
                *get_values(TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING),
            ]:
                inputs_dict["start_positions"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
                inputs_dict["end_positions"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [
                *get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING):
                inputs_dict["next_sentence_label"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [
                *get_values(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(TF_MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(TF_MODEL_FOR_PRETRAINING_MAPPING),
                *get_values(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
                *get_values(TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING),
            ] and "labels" in dict(inspect.signature(model_class.call).parameters):
                inputs_dict["labels"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int32
                )
            elif model_class in get_values(TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING):
                num_patches = self.model_tester.image_size // self.model_tester.patch_size
                inputs_dict["bool_masked_pos"] = tf.zeros(
                    (self.model_tester.batch_size, num_patches**2), dtype=tf.int32
                )
            elif model_class in get_values(TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING):
                batch_size, num_channels, height, width = inputs_dict["pixel_values"].shape
                inputs_dict["labels"] = tf.zeros((self.model_tester.batch_size, height, width), dtype=tf.int32)
            elif model_class.__name__.endswith("ForCTC"):
                # When we have enough CTC models for an AutoClass, we should use their mapping instead of name checks
                inputs_dict["labels"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int32
                )

        return inputs_dict

    def test_initialization(self):
        pass

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)

                # the config file (and the generation config file, if it can generate) should be saved
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))
                self.assertEqual(
                    model.can_generate(), os.path.exists(os.path.join(tmpdirname, GENERATION_CONFIG_NAME))
                )

                model = model_class.from_pretrained(tmpdirname)
                after_outputs = model(self._prepare_for_class(inputs_dict, model_class))

                self.assert_outputs_same(after_outputs, outputs)

    def test_save_load_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            model_config = model.get_config()
            # make sure that returned config is jsonifiable, which is required by keras
            json.dumps(model_config)
            new_model = model_class.from_config(model.get_config())
            # make sure it also accepts a normal config
            _ = model_class.from_config(model.config)
            _ = new_model(self._prepare_for_class(inputs_dict, model_class))  # Build model
            new_model.set_weights(model.get_weights())
            after_outputs = new_model(self._prepare_for_class(inputs_dict, model_class))

            self.assert_outputs_same(after_outputs, outputs)

    @slow
    def test_saved_model_creation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = False
        config.output_attentions = False

        if hasattr(config, "use_cache"):
            config.use_cache = False

        model_class = self.all_model_classes[0]

        class_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
        model = model_class(config)

        model(class_inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, saved_model=True)
            saved_model_dir = os.path.join(tmpdirname, "saved_model", "1")
            self.assertTrue(os.path.exists(saved_model_dir))

    def test_prepare_serving_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)
            outputs = model(inputs)
            serving_outputs = model.serving_output(outputs)

            for k, v in serving_outputs.items():
                # Check that we have one of three possible outputs: None, tuple of tensors or a tensor
                if isinstance(v, tuple):
                    self.assertTrue(all(isinstance(elem, tf.Tensor) for elem in v))
                elif v is not None:
                    self.assertIsInstance(v, tf.Tensor)
                else:
                    self.assertIsNone(v)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = [
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
                expected_arg_names.extend(["decoder_position_ids"] if "decoder_position_ids" in arg_names else [])
                expected_arg_names.extend(
                    ["head_mask", "decoder_head_mask"] if "head_mask" and "decoder_head_mask" in arg_names else []
                )
                expected_arg_names.extend(
                    ["cross_attn_head_mask", "encoder_outputs"]
                    if "cross_attn_head_mask" in arg_names
                    else ["encoder_outputs"]
                )
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

            else:
                expected_arg_names = ["input_ids"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_onnx_compliancy(self):
        if not self.test_onnx:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        INTERNAL_OPS = [
            "Assert",
            "AssignVariableOp",
            "EmptyTensorList",
            "ReadVariableOp",
            "ResourceGather",
            "TruncatedNormal",
            "VarHandleOp",
            "VarIsInitializedOp",
        ]
        onnx_ops = []

        with open(os.path.join(".", "utils", "tf_ops", "onnx.json")) as f:
            onnx_opsets = json.load(f)["opsets"]

        for i in range(1, self.onnx_min_opset + 1):
            onnx_ops.extend(onnx_opsets[str(i)])

        for model_class in self.all_model_classes:
            model_op_names = set()

            with tf.Graph().as_default() as g:
                model = model_class(config)
                model.build_in_name_scope()

                for op in g.get_operations():
                    model_op_names.add(op.node_def.op)

            model_op_names = sorted(model_op_names)
            incompatible_ops = []

            for op in model_op_names:
                if op not in onnx_ops and op not in INTERNAL_OPS:
                    incompatible_ops.append(op)

            self.assertEqual(len(incompatible_ops), 0, incompatible_ops)

    # `tf2onnx` issue page: https://github.com/onnx/tensorflow-onnx/issues/2172
    # TODO: undo skip once a fix is done in `tf2onnx`
    @unittest.skip("`tf2onnx` broke with TF 2.13")
    @require_tf2onnx
    @slow
    def test_onnx_runtime_optimize(self):
        if not self.test_onnx:
            return

        import onnxruntime
        import tf2onnx

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes[:2]:
            model = model_class(config)
            model.build_in_name_scope()

            onnx_model_proto, _ = tf2onnx.convert.from_keras(model, opset=self.onnx_min_opset)

            onnxruntime.InferenceSession(onnx_model_proto.SerializeToString())

    def test_keras_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        tf_main_layer_classes = {
            module_member
            for model_class in self.all_model_classes
            for module in (import_module(model_class.__module__),)
            for module_member_name in dir(module)
            if module_member_name.endswith("MainLayer")
            # This condition is required, since `modeling_tf_clip.py` has 3 classes whose names end with `MainLayer`.
            and module_member_name[: -len("MainLayer")] == model_class.__name__[: -len("Model")]
            for module_member in (getattr(module, module_member_name),)
            if isinstance(module_member, type)
            and keras.layers.Layer in module_member.__bases__
            and getattr(module_member, "_keras_serializable", False)
        }
        for main_layer_class in tf_main_layer_classes:
            # T5MainLayer needs an embed_tokens parameter when called without the inputs_embeds parameter
            if "T5" in main_layer_class.__name__:
                # Take the same values than in TFT5ModelTester for this shared layer
                shared = TFSharedEmbeddings(99, 32, name="shared")
                config.use_cache = inputs_dict.pop("use_cache", None)
                main_layer = main_layer_class(config, embed_tokens=shared)
            else:
                main_layer = main_layer_class(config)

            symbolic_inputs = {
                name: keras.Input(tensor.shape[1:], dtype=tensor.dtype)
                for name, tensor in inputs_dict.items()
                if tf.is_tensor(tensor)
            }

            model = keras.Model(symbolic_inputs, outputs=main_layer(symbolic_inputs))
            outputs = model(inputs_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                filepath = os.path.join(tmpdirname, "keras_model.h5")
                model.save(filepath)
                if "T5" in main_layer_class.__name__:
                    model = keras.models.load_model(
                        filepath,
                        custom_objects={
                            main_layer_class.__name__: main_layer_class,
                            "TFSharedEmbeddings": TFSharedEmbeddings,
                        },
                    )
                else:
                    model = keras.models.load_model(
                        filepath, custom_objects={main_layer_class.__name__: main_layer_class}
                    )
                assert isinstance(model, keras.Model)
                after_outputs = model(inputs_dict)
                self.assert_outputs_same(after_outputs, outputs)

    def assert_outputs_same(self, after_outputs, outputs):
        # Make sure we don't have nans
        if isinstance(after_outputs, tf.Tensor):
            out_1 = after_outputs.numpy()
        elif isinstance(after_outputs, dict):
            out_1 = after_outputs[list(after_outputs.keys())[0]].numpy()
        else:
            out_1 = after_outputs[0].numpy()
        out_2 = outputs[0].numpy()
        self.assertEqual(out_1.shape, out_2.shape)
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, 1e-5)

    # Don't copy this method to model specific test file!
    # TODO: remove this method once the issues are all fixed!
    def _make_attention_mask_non_null(self, inputs_dict):
        """Make sure no sequence has all zeros as attention mask"""

        for k in ["attention_mask", "encoder_attention_mask", "decoder_attention_mask"]:
            if k in inputs_dict:
                attention_mask = inputs_dict[k]

                # Make sure no all 0s attention masks - to avoid failure at this moment.
                # Put `1` at the beginning of sequences to make it still work when combining causal attention masks.
                # TODO: remove this line once a fix regarding large negative values for attention mask is done.
                attention_mask = tf.concat(
                    [tf.ones_like(attention_mask[:, :1], dtype=attention_mask.dtype), attention_mask[:, 1:]], axis=-1
                )

                # Here we make the first sequence with all 0s as attention mask.
                # Currently, this will fail for `TFWav2Vec2Model`. This is caused by the different large negative
                # values, like `1e-4`, `1e-9`, `1e-30` and `-inf` for attention mask across models/frameworks.
                # TODO: enable this block once the large negative values thing is cleaned up.
                # (see https://github.com/huggingface/transformers/issues/14859)
                # attention_mask = tf.concat(
                #     [
                #         tf.zeros_like(attention_mask[:1], dtype=tf.int32),
                #         tf.cast(attention_mask[1:], dtype=tf.int32)
                #     ],
                #     axis=0
                # )

                inputs_dict[k] = attention_mask

    # Don't copy this method to model specific test file!
    # TODO: remove this method once the issues are all fixed!
    def _postprocessing_to_ignore_test_cases(self, tf_outputs, pt_outputs, model_class):
        """For temporarily ignoring some failed test cases (issues to be fixed)"""

        tf_keys = {k for k, v in tf_outputs.items() if v is not None}
        pt_keys = {k for k, v in pt_outputs.items() if v is not None}

        key_differences = tf_keys.symmetric_difference(pt_keys)

        if model_class.__name__ in [
            "TFFlaubertWithLMHeadModel",
            "TFFunnelForPreTraining",
            "TFElectraForPreTraining",
            "TFXLMWithLMHeadModel",
        ]:
            for k in key_differences:
                if k in ["loss", "losses"]:
                    tf_keys.discard(k)
                    pt_keys.discard(k)
        elif model_class.__name__.startswith("TFGPT2"):
            # `TFGPT2` has `past_key_values` as a tensor while `GPT2` has it as a tuple.
            tf_keys.discard("past_key_values")
            pt_keys.discard("past_key_values")

        # create new outputs from the remaining fields
        new_tf_outputs = type(tf_outputs)(**{k: tf_outputs[k] for k in tf_keys})
        new_pt_outputs = type(pt_outputs)(**{k: pt_outputs[k] for k in pt_keys})

        return new_tf_outputs, new_pt_outputs

    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=1e-4, name="outputs", attributes=None):
        """Check the outputs from PyTorch and TensorFlow models are close enough. Checks are done in a recursive way.

        Args:
            model_class: The class of the model that is currently testing. For example, `TFBertModel`,
                TFBertForMaskedLM`, `TFBertForSequenceClassification`, etc. Mainly used for providing more informative
                error messages.
            name (`str`): The name of the output. For example, `output.hidden_states`, `output.attentions`, etc.
            attributes (`Tuple[str]`): The names of the output's element if the output is a tuple/list with each element
                being a named field in the output.
        """
        from transformers.cache_utils import DynamicCache

        self.assertEqual(type(name), str)
        if attributes is not None:
            self.assertEqual(type(attributes), tuple, f"{name}: The argument `attributes` should be a `tuple`")

        # Allow `ModelOutput` (e.g. `CLIPOutput` has `text_model_output` and `vision_model_output`).
        if isinstance(tf_outputs, ModelOutput):
            self.assertTrue(
                isinstance(pt_outputs, ModelOutput),
                f"{name}: `pt_outputs` should an instance of `ModelOutput` when `tf_outputs` is",
            )

            # Don't copy this block to model specific test file!
            # TODO: remove this method and this line after issues are fixed
            tf_outputs, pt_outputs = self._postprocessing_to_ignore_test_cases(tf_outputs, pt_outputs, model_class)

            tf_keys = [k for k, v in tf_outputs.items() if v is not None]
            pt_keys = [k for k, v in pt_outputs.items() if v is not None]

            self.assertEqual(tf_keys, pt_keys, f"{name}: Output keys differ between TF and PyTorch")

            # convert to the case of `tuple`
            # appending each key to the current (string) `names`
            attributes = tuple([f"{name}.{k}" for k in tf_keys])
            self.check_pt_tf_outputs(
                tf_outputs.to_tuple(), pt_outputs.to_tuple(), model_class, tol=tol, name=name, attributes=attributes
            )

        # Allow `list` (e.g. `TransfoXLModelOutput.mems` is a list of tensors.)
        elif type(tf_outputs) in [tuple, list]:
            self.assertEqual(type(tf_outputs), type(pt_outputs), f"{name}: Output types differ between TF and PyTorch")
            self.assertEqual(len(tf_outputs), len(pt_outputs), f"{name}: Output lengths differ between TF and PyTorch")

            if attributes is not None:
                # case 1: each output has assigned name (e.g. a tuple form of a `ModelOutput`)
                self.assertEqual(
                    len(attributes),
                    len(tf_outputs),
                    f"{name}: The tuple `names` should have the same length as `tf_outputs`",
                )
            else:
                # case 2: each output has no assigned name (e.g. hidden states of each layer) -> add an index to `names`
                attributes = tuple([f"{name}_{idx}" for idx in range(len(tf_outputs))])

            for tf_output, pt_output, attr in zip(tf_outputs, pt_outputs, attributes):
                if isinstance(pt_output, DynamicCache):
                    pt_output = pt_output.to_legacy_cache()
                self.check_pt_tf_outputs(tf_output, pt_output, model_class, tol=tol, name=attr)

        elif isinstance(tf_outputs, tf.Tensor):
            self.assertTrue(
                isinstance(pt_outputs, torch.Tensor), f"{name}: `pt_outputs` should a tensor when `tf_outputs` is"
            )

            tf_outputs = tf_outputs.numpy()
            pt_outputs = pt_outputs.detach().to("cpu").numpy()

            self.assertEqual(
                tf_outputs.shape, pt_outputs.shape, f"{name}: Output shapes differ between TF and PyTorch"
            )

            # deal with NumPy's scalars to make replacing nan values by 0 work.
            if np.isscalar(tf_outputs):
                tf_outputs = np.array([tf_outputs])
                pt_outputs = np.array([pt_outputs])

            tf_nans = np.isnan(tf_outputs)
            pt_nans = np.isnan(pt_outputs)

            pt_outputs[tf_nans] = 0
            tf_outputs[tf_nans] = 0
            pt_outputs[pt_nans] = 0
            tf_outputs[pt_nans] = 0

            max_diff = np.amax(np.abs(tf_outputs - pt_outputs))
            self.assertLessEqual(max_diff, tol, f"{name}: Difference between torch and tf is {max_diff} (>= {tol}).")
        else:
            raise ValueError(
                "`tf_outputs` should be an instance of `tf.Tensor`, a `tuple`, or an instance of `tf.Tensor`. Got"
                f" {type(tf_outputs)} instead."
            )

    def prepare_pt_inputs_from_tf_inputs(self, tf_inputs_dict):
        pt_inputs_dict = {}
        for name, key in tf_inputs_dict.items():
            if isinstance(key, bool):
                pt_inputs_dict[name] = key
            elif name == "input_values":
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            elif name == "pixel_values":
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            elif name == "input_features":
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            # other general float inputs
            elif tf_inputs_dict[name].dtype.is_floating:
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            else:
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)

        return pt_inputs_dict

    def check_pt_tf_models(self, tf_model, pt_model, tf_inputs_dict):
        pt_inputs_dict = self.prepare_pt_inputs_from_tf_inputs(tf_inputs_dict)

        # send pytorch inputs to the correct device
        pt_inputs_dict = {
            k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs_dict.items()
        }

        # send pytorch model to the correct device
        pt_model.to(torch_device)

        # Check predictions on first output (logits/hidden-states) are close enough given low-level computational differences
        pt_model.eval()

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs_dict)
        tf_outputs = tf_model(tf_inputs_dict)

        # tf models returned loss is usually a tensor rather than a scalar.
        # (see `hf_compute_loss`: it uses `keras.losses.Reduction.NONE`)
        # Change it here to a scalar to match PyTorch models' loss
        tf_loss = getattr(tf_outputs, "loss", None)
        if tf_loss is not None:
            tf_outputs.loss = tf.math.reduce_mean(tf_loss)

        self.check_pt_tf_outputs(tf_outputs, pt_outputs, type(tf_model))

    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self, allow_missing_keys=False):
        import transformers

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # Output all for aggressive testing
            config.output_hidden_states = True
            config.output_attentions = self.has_attentions

            # Make sure no sequence has all zeros as attention mask, otherwise some tests fail due to the inconsistency
            # of the usage `1e-4`, `1e-9`, `1e-30`, `-inf`.
            # TODO: Use a uniform value for all models, make sure all tests pass without this processing, and remove it.
            self._make_attention_mask_non_null(inputs_dict)

            pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beginning
            pt_model_class = getattr(transformers, pt_model_class_name)

            tf_model = model_class(config)
            pt_model = pt_model_class(config)

            tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            tf_inputs_dict_with_labels = self._prepare_for_class(
                inputs_dict,
                model_class,
                # Not all models accept "labels" in the forward pass (yet :) )
                return_labels=True if "labels" in inspect.signature(model_class.call).parameters.keys() else False,
            )

            # For some models (e.g. base models), there is no label returned.
            # Set the input dict to `None` to avoid check outputs twice for the same input dicts.
            if not set(tf_inputs_dict_with_labels.keys()).symmetric_difference(tf_inputs_dict.keys()):
                tf_inputs_dict_with_labels = None

            # Check we can load pt model in tf and vice-versa with model => model functions
            tf_model = transformers.load_pytorch_model_in_tf2_model(
                tf_model, pt_model, tf_inputs=tf_inputs_dict, allow_missing_keys=allow_missing_keys
            )
            pt_model = transformers.load_tf2_model_in_pytorch_model(
                pt_model, tf_model, allow_missing_keys=allow_missing_keys
            )

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)
            # check with `labels`
            if tf_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict_with_labels)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(
                    tf_model, pt_checkpoint_path, allow_missing_keys=allow_missing_keys
                )

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(
                    pt_model, tf_checkpoint_path, allow_missing_keys=allow_missing_keys
                )

            # Original test: check without `labels`
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)
            # check with `labels`
            if tf_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict_with_labels)

    @slow
    def test_compile_tf_model(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes[:2]:
            # Prepare our model
            model = model_class(config)
            # These are maximally general inputs for the model, with multiple None dimensions
            # Hopefully this will catch any conditionals that fail for flexible shapes
            functional_inputs = {
                key: keras.Input(shape=val.shape[1:], dtype=val.dtype, name=key)
                for key, val in model.input_signature.items()
                if key in model.dummy_inputs
            }
            outputs_dict = model(functional_inputs)

            hidden_states = outputs_dict[0]

            # Compile extended model
            functional_model = keras.Model(inputs=functional_inputs, outputs=hidden_states)
            model_out = functional_model.predict(model.dummy_inputs)  # Check we can pass inputs with the Keras API
            self.assertTrue(model_out is not None)
            with tempfile.TemporaryDirectory() as tmpdirname:
                functional_model.save(tmpdirname)  # Ensure we can save/export the whole functional model

    def test_keyword_and_dict_args(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)

            outputs_dict = model(inputs)

            inputs_keywords = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
            outputs_keywords = model(**inputs_keywords)
            output_dict = outputs_dict[0].numpy()
            output_keywords = outputs_keywords[0].numpy()

            self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-6)

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", self.model_tester.seq_length)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", self.model_tester.seq_length)
        decoder_key_length = getattr(self.model_tester, "key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        def check_decoder_attentions_output(outputs):
            out_len = len(outputs)
            self.assertEqual(min(out_len % 2, out_len % 5), 0)  # differentiation due to newly added cross_attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

        def check_encoder_attentions_output(outputs):
            attentions = [
                t.numpy() for t in (outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions)
            ]
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            out_len = len(outputs)
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)

            if self.is_encoder_decoder:
                model = model_class(config)
                outputs = model(self._prepare_for_class(inputs_dict, model_class))
                self.assertEqual(config.output_hidden_states, False)
                check_decoder_attentions_output(outputs)

            # Check that output attentions can also be changed via the config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            config.output_hidden_states = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + (2 if self.is_encoder_decoder else 1), len(outputs))
            self.assertEqual(model.config.output_hidden_states, True)
            check_encoder_attentions_output(outputs)

    def test_headmasking(self):
        if not self.test_head_masking:
            return

        random.Random().seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        random.Random().seed()

        inputs_dict["output_attentions"] = True
        config.output_hidden_states = True
        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)

            # Prepare head_mask
            def prepare_layer_head_mask(i, attention_heads, num_hidden_layers):
                if i == 0:
                    return tf.concat(
                        (tf.zeros(1, dtype=tf.float32), tf.ones(attention_heads - 1, dtype=tf.float32)), 0
                    )
                elif i == num_hidden_layers - 1:
                    return tf.concat(
                        (tf.zeros(attention_heads - 1, dtype=tf.float32), tf.ones(1, dtype=tf.float32)), 0
                    )
                else:
                    return tf.ones(attention_heads, dtype=tf.float32)

            head_mask = tf.stack(
                [
                    prepare_layer_head_mask(i, config.num_attention_heads, config.num_hidden_layers)
                    for i in range(config.num_hidden_layers)
                ],
                0,
            )

            inputs = self._prepare_for_class(inputs_dict, model_class).copy()
            inputs["head_mask"] = head_mask
            if model.config.is_encoder_decoder:
                signature = inspect.signature(model.call)
                arg_names = [*signature.parameters.keys()]
                if "decoder_head_mask" in arg_names:  # necessary diferentiation because of T5 model
                    inputs["decoder_head_mask"] = head_mask
                if "cross_attn_head_mask" in arg_names:
                    inputs["cross_attn_head_mask"] = head_mask

            outputs = model(**inputs, return_dict=True)

            def check_attentions_validity(attentions):
                # Remove Nan
                for t in attentions:
                    self.assertLess(
                        (tf.math.reduce_sum(tf.cast(tf.math.is_nan(t), tf.float32))).numpy(), (tf.size(t) / 4).numpy()
                    )  # Check we don't have more than 25% nans (arbitrary)

                attentions = [
                    tf.where(tf.math.is_nan(t), 0.0, t) for t in attentions
                ]  # remove them (the test is less complete)

                self.assertAlmostEqual(tf.math.reduce_sum(attentions[0][..., 0, :, :]).numpy(), 0.0)
                self.assertNotEqual(tf.math.reduce_sum(attentions[0][..., -1, :, :]).numpy(), 0.0)
                if len(attentions) > 2:  # encoder-decodere models have only 2 layers in each modules
                    self.assertNotEqual(tf.math.reduce_sum(attentions[1][..., 0, :, :]).numpy(), 0.0)
                self.assertAlmostEqual(tf.math.reduce_sum(attentions[-1][..., -2, :, :]).numpy(), 0.0)
                self.assertNotEqual(tf.math.reduce_sum(attentions[-1][..., -1, :, :]).numpy(), 0.0)

            if model.config.is_encoder_decoder:
                check_attentions_validity(outputs.encoder_attentions)
                check_attentions_validity(outputs.decoder_attentions)
                if "cross_attn_head_mask" in arg_names:
                    check_attentions_validity(outputs.cross_attentions)
            else:
                check_attentions_validity(outputs.attentions)

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_hidden_states_output(config, inputs_dict, model_class):
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )

            if model.config.is_encoder_decoder:
                encoder_hidden_states = outputs.encoder_hidden_states
                decoder_hidden_states = outputs.decoder_hidden_states

                self.assertEqual(config.output_attentions, False)
                self.assertEqual(len(encoder_hidden_states), expected_num_layers)
                self.assertListEqual(
                    list(encoder_hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size],
                )
                self.assertEqual(len(decoder_hidden_states), expected_num_layers)
                self.assertListEqual(
                    list(decoder_hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size],
                )
            else:
                hidden_states = outputs.hidden_states
                self.assertEqual(config.output_attentions, False)
                self.assertEqual(len(hidden_states), expected_num_layers)
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size],
                )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(config, inputs_dict, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(config, inputs_dict, model_class)

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        text_in_text_out_models = (
            get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING)
            + get_values(TF_MODEL_FOR_MASKED_LM_MAPPING)
            + get_values(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING)
        )
        speech_in_text_out_models = get_values(TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING)

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), keras.layers.Layer)

            legacy_text_in_text_out = model.get_lm_head() is not None
            if model_class in text_in_text_out_models or legacy_text_in_text_out:
                out_embeddings = model.get_output_embeddings()
                self.assertIsInstance(out_embeddings, keras.layers.Layer)
                bias = model.get_bias()
                if bias is not None:
                    self.assertIsInstance(bias, dict)
                    for _, v in bias.items():
                        self.assertIsInstance(v, tf.Variable)
            elif model_class in speech_in_text_out_models:
                out_embeddings = model.get_output_embeddings()
                self.assertIsInstance(out_embeddings, keras.layers.Layer)
                bias = model.get_bias()
                self.assertIsNone(bias)
            else:
                out_embeddings = model.get_output_embeddings()
                assert out_embeddings is None
                bias = model.get_bias()
                self.assertIsNone(bias)

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            first, second = (
                model(self._prepare_for_class(inputs_dict, model_class), training=False)[0],
                model(self._prepare_for_class(inputs_dict, model_class), training=False)[0],
            )
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            tuple_output = model(tuple_inputs, return_dict=False, **additional_kwargs)
            dict_output = model(dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

            def recursive_check(tuple_object, dict_object):
                if isinstance(tuple_object, (List, Tuple)):
                    for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                        recursive_check(tuple_iterable_value, dict_iterable_value)
                elif tuple_object is None:
                    return
                else:
                    self.assertTrue(
                        all(tf.equal(tuple_object, dict_object)),
                        msg=(
                            "Tuple and dict output are not equal. Difference:"
                            f" {tf.math.reduce_max(tf.abs(tuple_object - dict_object))}"
                        ),
                    )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            # Not all models accept "labels" in the forward pass (yet :) )
            if "labels" in inspect.signature(model.call).parameters.keys():
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs)

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

                if self.has_attentions:
                    tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                    tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    check_equivalence(
                        model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                    )

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            inputs = copy.deepcopy(inputs_dict)

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
            else:
                inputs["inputs_embeds"] = model.get_input_embeddings()(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = model.get_input_embeddings()(decoder_input_ids)

            inputs = self._prepare_for_class(inputs, model_class)

            model(inputs)

    def test_numpy_arrays_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def prepare_numpy_arrays(inputs_dict):
            inputs_np_dict = {}
            for k, v in inputs_dict.items():
                if tf.is_tensor(v):
                    inputs_np_dict[k] = v.numpy()
                else:
                    inputs_np_dict[k] = np.array(k)

            return inputs_np_dict

        for model_class in self.all_model_classes:
            model = model_class(config)

            inputs = self._prepare_for_class(inputs_dict, model_class)
            inputs_np = prepare_numpy_arrays(inputs)

            output_for_dict_input = model(inputs_np)
            output_for_kw_input = model(**inputs_np)
            self.assert_outputs_same(output_for_dict_input, output_for_kw_input)

    def test_valid_input_signature_and_dummies(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            call_args = inspect.signature(model.call).parameters
            for key in model.input_signature:
                self.assertIn(key, call_args)
            for key in model.dummy_inputs:
                self.assertIn(key, call_args)

    def test_resize_token_embeddings(self):
        # TODO (joao): after the embeddings refactor is complete, rework this test so as to rely exclusively on
        # keras.layers.Embedding

        if not self.test_resize_embeddings:
            return
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def _get_word_embedding_weight(model, embedding_layer):
            if isinstance(embedding_layer, keras.layers.Embedding):
                # builds the embeddings layer
                model.build_in_name_scope()
                return embedding_layer.embeddings
            else:
                return model._get_word_embedding_weight(embedding_layer)

        for model_class in self.all_model_classes:
            for size in [config.vocab_size - 10, config.vocab_size + 10, None]:
                # build the embeddings
                model = model_class(config=copy.deepcopy(config))  # `resize_token_embeddings` mutates `config`
                old_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                old_bias = model.get_bias()
                old_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())
                # reshape the embeddings
                model.resize_token_embeddings(size)
                new_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                new_bias = model.get_bias()
                new_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())

                # check that the resized embeddings size matches the desired size.
                assert_size = size if size is not None else config.vocab_size
                self.assertEqual(new_input_embeddings.shape[0], assert_size)

                # check that weights remain the same after resizing
                models_equal = True
                for p1, p2 in zip(old_input_embeddings.value(), new_input_embeddings.value()):
                    if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                        models_equal = False
                self.assertTrue(models_equal)

                if old_bias is not None and new_bias is not None:
                    for old_weight, new_weight in zip(old_bias.values(), new_bias.values()):
                        self.assertEqual(new_weight.shape[-1], assert_size)

                        models_equal = True
                        for p1, p2 in zip(tf.squeeze(old_weight), tf.squeeze(new_weight)):
                            if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                                models_equal = False
                        self.assertTrue(models_equal)

                if old_output_embeddings is not None and new_output_embeddings is not None:
                    self.assertEqual(new_output_embeddings.shape[0], assert_size)
                    self.assertEqual(new_output_embeddings.shape[1], old_output_embeddings.shape[1])

                    models_equal = True
                    for p1, p2 in zip(old_output_embeddings.value(), new_output_embeddings.value()):
                        if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                            models_equal = False
                    self.assertTrue(models_equal)

    # TODO (Joao): this test is not slow, but it's tagged as such to keep track of failures on the scheduled CI runs,
    # while passing push CI. Fix the underlying issues and remove the tag.
    @slow
    def test_save_load_after_resize_token_embeddings(self):
        if not self.test_resize_embeddings:
            return
        config, original_inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # create a model with resized (expended) embeddings
            new_tokens_size = 10
            old_total_size = config.vocab_size
            new_total_size = old_total_size + new_tokens_size
            model = model_class(config=copy.deepcopy(config))  # `resize_token_embeddings` mutates `config`
            model.build_in_name_scope()
            model.resize_token_embeddings(new_total_size)

            # fetch the output for an input exclusively made of new members of the vocabulary
            inputs_dict = copy.deepcopy(original_inputs_dict)
            ids_feat_name = None
            if "input_ids" in inputs_dict:
                ids_feat_name = "input_ids"
            elif "decoder_input_ids" in inputs_dict:
                ids_feat_name = "decoder_input_ids"
            else:
                assert False, "No input ids feature found in the inputs dict"

            new_vocab_input_ids = ids_tensor(inputs_dict[ids_feat_name].shape, new_tokens_size)
            new_vocab_input_ids += old_total_size
            inputs_dict[ids_feat_name] = new_vocab_input_ids
            if "input_ids" in inputs_dict:
                inputs_dict["input_ids"] = new_vocab_input_ids
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"] = new_vocab_input_ids
            prepared_inputs = self._prepare_for_class(inputs_dict, model_class)
            outputs = model(**prepared_inputs)

            # save and load the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                model = model_class.from_pretrained(tmpdirname)
                restored_model_outputs = model(**prepared_inputs)

                # check that the output for the restored model is the same
                self.assert_outputs_same(restored_model_outputs, outputs)

    @unittest.skipIf(
        not is_tf_available() or len(tf.config.list_physical_devices("GPU")) == 0,
        reason="This test always passes on CPU.",
    )
    def test_embeddings_out_of_bounds_raise_exception(self):
        # TF embeddings layers don't raise an exception when an index is out of bounds on GPU, so we manually raise it.
        # This test should only fail on GPU for models where we haven't added the safety check.
        if not self.test_resize_embeddings:
            return
        config, original_inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            inputs_dict = copy.deepcopy(original_inputs_dict)
            if "input_ids" in inputs_dict:
                inputs_dict["input_ids"] = inputs_dict["input_ids"] * int(1e9)
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"] = inputs_dict["decoder_input_ids"] * int(1e9)
            prepared_inputs = self._prepare_for_class(inputs_dict, model_class)
            with self.assertRaises(tf.errors.InvalidArgumentError):
                model(**prepared_inputs)

    def test_lm_head_model_random_no_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get("input_ids", None)

        # iterate over all generative models
        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            if config.bos_token_id is None:
                # if bos token id is not defined model needs input_ids
                with self.assertRaises(ValueError):
                    model.generate(do_sample=True, max_length=5)
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(input_ids, do_sample=True))
            elif model_class.__name__ not in ["TFSpeech2TextForConditionalGeneration"]:
                # Models with non-text inputs won't work here; num_return_sequences = 1
                self._check_generated_ids(model.generate(do_sample=True, max_length=5))

            with self.assertRaises(ValueError):
                # generating multiple sequences when no beam search generation
                # is not allowed as it would always generate the same sequences
                model.generate(input_ids, do_sample=False, num_return_sequences=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(model.generate(input_ids, do_sample=True, num_return_sequences=2))

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(
                input_ids, do_sample=True, bad_words_ids=bad_words_ids, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_ids.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    def test_lm_head_model_no_beam_search_generate_dict_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get("input_ids", None)
        if input_ids is None:
            input_ids = inputs_dict.get("input_features", None)

        # iterate over all generative models
        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            output_greedy = model.generate(
                input_ids,
                do_sample=False,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            output_sample = model.generate(
                input_ids,
                do_sample=True,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_greedy, TFGreedySearchEncoderDecoderOutput)
                self.assertIsInstance(output_sample, TFSampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_greedy, TFGreedySearchDecoderOnlyOutput)
                self.assertIsInstance(output_sample, TFSampleDecoderOnlyOutput)

    def test_lm_head_model_random_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get("input_ids", None)

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            if config.bos_token_id is None:
                # if bos token id is not defined model needs input_ids, num_return_sequences = 1
                self._check_generated_ids(model.generate(input_ids, do_sample=True, num_beams=2))
            else:
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(do_sample=True, max_length=5, num_beams=2))

            with self.assertRaises(ValueError):
                # generating more sequences than having beams leads is not possible
                model.generate(input_ids, do_sample=False, num_return_sequences=3, num_beams=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(
                model.generate(
                    input_ids,
                    do_sample=True,
                    num_beams=2,
                    num_return_sequences=2,
                )
            )
            # num_return_sequences > 1, greedy
            self._check_generated_ids(model.generate(input_ids, do_sample=False, num_beams=2, num_return_sequences=2))

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(
                input_ids, do_sample=False, bad_words_ids=bad_words_ids, num_beams=2, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_ids.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    def test_lm_head_model_beam_search_generate_dict_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get("input_ids", None)
        if input_ids is None:
            input_ids = inputs_dict.get("input_features", None)

        # iterate over all generative models
        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            output_beam_search = model.generate(
                input_ids,
                num_beams=2,
                do_sample=False,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            output_beam_sample = model.generate(
                input_ids,
                num_beams=2,
                do_sample=True,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_beam_search, TFBeamSearchEncoderDecoderOutput)
                self.assertIsInstance(output_beam_sample, TFBeamSampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_beam_search, TFBeamSearchDecoderOnlyOutput)
                self.assertIsInstance(output_beam_sample, TFBeamSampleDecoderOnlyOutput)

    def test_loss_computation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            # The number of elements in the loss should be the same as the number of elements in the label
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            added_label_names = sorted(prepared_for_class.keys() - inputs_dict.keys(), reverse=True)
            if not added_label_names:
                continue  # This test is only for models with easily-separable labels
            added_label = prepared_for_class[added_label_names[0]]
            expected_loss_size = added_label.shape.as_list()[:1]

            # Test that model correctly compute the loss with kwargs
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            possible_input_names = {"input_ids", "pixel_values", "input_features", "input_values"}
            input_name = possible_input_names.intersection(set(prepared_for_class)).pop()
            model_input = prepared_for_class.pop(input_name)

            outputs = model(model_input, **prepared_for_class)
            if not isinstance(outputs, ModelOutput) or not hasattr(outputs, "loss"):
                continue

            loss = outputs.loss
            self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

            # Test that model correctly compute the loss when we mask some positions
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            possible_input_names = {"input_ids", "pixel_values", "input_features", "input_values"}
            input_name = possible_input_names.intersection(set(prepared_for_class)).pop()
            model_input = prepared_for_class.pop(input_name)
            if "labels" in prepared_for_class:
                labels = prepared_for_class["labels"].numpy()
                if len(labels.shape) > 1 and labels.shape[1] != 1:
                    labels[0] = -100
                    prepared_for_class["labels"] = tf.convert_to_tensor(labels)
                    loss = model(model_input, **prepared_for_class)[0]
                    self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])
                    self.assertTrue(not np.any(np.isnan(loss.numpy())))

            # Test that model correctly compute the loss with a dict
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            loss = model(prepared_for_class)[0]
            self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

            # Test that model correctly compute the loss with a tuple
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)

            # Get keys that were added with the _prepare_for_class function
            label_keys = prepared_for_class.keys() - inputs_dict.keys()
            signature = inspect.signature(model.call).parameters
            signature_names = list(signature.keys())

            # Create a dictionary holding the location of the tensors in the tuple
            tuple_index_mapping = {0: input_name}
            for label_key in label_keys:
                label_key_index = signature_names.index(label_key)
                tuple_index_mapping[label_key_index] = label_key
            sorted_tuple_index_mapping = sorted(tuple_index_mapping.items())
            # Initialize a list with their default values, update the values and convert to a tuple
            list_input = []

            for name in signature_names:
                if name != "kwargs":
                    list_input.append(signature[name].default)

            for index, value in sorted_tuple_index_mapping:
                list_input[index] = prepared_for_class[value]

            tuple_input = tuple(list_input)

            # Send to model
            loss = model(tuple_input[:-1])[0]

            self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

    def check_keras_fit_results(self, val_loss1, val_loss2, atol=1e-2, rtol=1e-3):
        self.assertTrue(np.allclose(val_loss1, val_loss2, atol=atol, rtol=rtol))

    @slow
    def test_keras_fit(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            # Test that model correctly compute the loss with kwargs
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            # We also remove "return_loss" as this is covered by the train_step when using fit()
            prepared_for_class = {
                key: val
                for key, val in prepared_for_class.items()
                if key not in ("head_mask", "decoder_head_mask", "cross_attn_head_mask", "return_loss")
            }
            if "labels" in prepared_for_class and "decoder_input_ids" in prepared_for_class:
                del prepared_for_class["decoder_input_ids"]

            accuracy_classes = [
                "ForPreTraining",
                "ForCausalLM",
                "ForMaskedLM",
                "ForQuestionAnswering",
                "ForMultipleChoice",
                "ForSequenceClassification",
                "ForTokenClassification",
                "ForNextSentencePrediction",
                "LMHeadModel",
            ]
            for accuracy_class in accuracy_classes:
                if model.__class__.__name__.endswith(accuracy_class):
                    metrics = [keras.metrics.SparseCategoricalAccuracy()]
                    break
            else:
                metrics = []

            if hasattr(self.model_tester, "batch_size"):
                sample_weight = tf.convert_to_tensor([0.5] * self.model_tester.batch_size, dtype=tf.float32)
            else:
                sample_weight = None
            # Build the model so we can get some constant weights and check outputs
            outputs = model(prepared_for_class)
            if getattr(outputs, "loss", None) is None:
                continue
            model_weights = model.get_weights()

            # Run eagerly to save some expensive compilation times
            model.compile(optimizer=keras.optimizers.SGD(0.0), run_eagerly=True, metrics=metrics)
            # Make sure the model fits without crashing regardless of where we pass the labels
            history1 = model.fit(
                prepared_for_class,
                validation_data=prepared_for_class,
                sample_weight=sample_weight,
                steps_per_epoch=1,
                validation_steps=1,
                shuffle=False,
            )
            val_loss1 = history1.history["val_loss"][0]
            self.assertTrue(not isnan(val_loss1))
            accuracy1 = {key: val[0] for key, val in history1.history.items() if key.endswith("accuracy")}

            possible_label_cols = {
                "labels",
                "label",
                "label_ids",
                "start_positions",
                "start_position",
                "end_positions",
                "end_position",
                "next_sentence_label",
            }
            label_names = possible_label_cols.intersection(set(prepared_for_class))
            if len(label_names) == 0:
                # The next tests only make sense for models with separate inputs and labels, and do not make
                # sense for models that don't clearly distinguish between the two (e.g. CLIP)
                return
            labels = {key: val for key, val in prepared_for_class.items() if key in label_names}
            inputs_minus_labels = {key: val for key, val in prepared_for_class.items() if key not in label_names}
            self.assertGreater(len(inputs_minus_labels), 0)

            # We reinitialize the model here even though our learning rate was zero
            # because BatchNorm updates weights by means other than gradient descent.
            model.set_weights(model_weights)

            history2 = model.fit(
                inputs_minus_labels,
                labels,
                validation_data=(inputs_minus_labels, labels),
                sample_weight=sample_weight,
                steps_per_epoch=1,
                validation_steps=1,
                shuffle=False,
            )
            val_loss2 = history2.history["val_loss"][0]
            self.assertTrue(not isnan(val_loss2))
            accuracy2 = {key: val[0] for key, val in history2.history.items() if key.endswith("accuracy")}
            self.check_keras_fit_results(val_loss1, val_loss2)
            self.assertEqual(history1.history.keys(), history2.history.keys())
            for key in history1.history.keys():
                if not key.startswith("val_"):
                    self.assertTrue("val_" + key in history1.history.keys(), "Outputs differ in train/test step!")
            if metrics:
                self.assertTrue(len(accuracy1) == len(accuracy2) > 0, "Missing metrics!")

    def test_int_support(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            prepared_for_class = self._prepare_for_class(
                inputs_dict.copy(),
                model_class,
                return_labels=True if "labels" in inspect.signature(model_class.call).parameters.keys() else False,
            )
            if not any(
                tensor.dtype.is_integer for tensor in prepared_for_class.values() if isinstance(tensor, tf.Tensor)
            ):
                return  # No integer inputs means no need for this test

            prepared_for_class = {
                key: tf.cast(tensor, tf.int64) if isinstance(tensor, tf.Tensor) and tensor.dtype.is_integer else tensor
                for key, tensor in prepared_for_class.items()
            }
            model = model_class(config)
            model(**prepared_for_class)  # No assertion, we're just checking this doesn't throw an error
            int32_prepared_for_class = {
                key: tf.cast(tensor, tf.int32) if isinstance(tensor, tf.Tensor) and tensor.dtype.is_integer else tensor
                for key, tensor in prepared_for_class.items()
            }
            model(**int32_prepared_for_class)  # No assertion, we're just checking this doesn't throw an error

            # After testing that the model accepts all int inputs, confirm that its dummies are int32
            for key, tensor in model.dummy_inputs.items():
                self.assertTrue(
                    isinstance(tensor, tf.Tensor) or keras.backend.is_keras_tensor(tensor),
                    "Dummy inputs should be tf.Tensor!",
                )
                if tensor.dtype.is_integer:
                    self.assertTrue(tensor.dtype == tf.int32, "Integer dummy inputs should be tf.int32!")

            # Also confirm that the input_signature uses int32
            for key, tensor_spec in model.input_signature.items():
                if tensor_spec.dtype.is_integer:
                    self.assertTrue(tensor_spec.dtype == tf.int32, "Input signatures should use tf.int32 for ints!")

    def test_generate_with_headmasking(self):
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            # We want to test only encoder-decoder models
            if not config.is_encoder_decoder:
                continue

            head_masking = {
                "head_mask": tf.zeros((config.encoder_layers, config.encoder_attention_heads)),
                "decoder_head_mask": tf.zeros((config.decoder_layers, config.decoder_attention_heads)),
                "cross_attn_head_mask": tf.zeros((config.decoder_layers, config.decoder_attention_heads)),
            }

            signature = inspect.signature(model.call)
            if set(head_masking.keys()) < {*signature.parameters.keys()}:
                continue

            for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
                out = model.generate(
                    inputs_dict["input_ids"],
                    num_beams=1,
                    max_length=inputs_dict["input_ids"] + 5,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    **{name: mask},
                )
                # We check the state of decoder_attentions and cross_attentions just from the last step
                attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
                self.assertEqual(sum([tf.reduce_sum(w).numpy() for w in attn_weights]), 0.0)

    def test_load_with_mismatched_shapes(self):
        if not self.test_mismatched_shapes:
            return
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class not in get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING):
                continue

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(config)
                    inputs = self._prepare_for_class(inputs_dict, model_class)
                    _ = model(**inputs)
                    model.save_pretrained(tmp_dir)

                    # Fails when we don't set ignore_mismatched_sizes=True
                    with self.assertRaises(ValueError):
                        new_model = TFAutoModelForSequenceClassification.from_pretrained(tmp_dir, num_labels=42)
                    with self.assertRaises(ValueError):
                        new_model_without_prefix = TFAutoModel.from_pretrained(tmp_dir, vocab_size=10)

                    logger = logging.get_logger("transformers.modeling_tf_utils")
                    with CaptureLogger(logger) as cl:
                        new_model = TFAutoModelForSequenceClassification.from_pretrained(
                            tmp_dir, num_labels=42, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)

                    logits = new_model(**inputs).logits
                    self.assertEqual(logits.shape[1], 42)

                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = TFAutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)

                    # Although Tf models always have a prefix pointing to `MainLayer`,
                    # we still add this "without prefix" test to keep a consistency between tf and pt tests.
                    input_ids = ids_tensor((2, 8), 10)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)

    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "call"))
            # The main input is the name of the argument after `self`
            observed_main_input_name = list(model_signature.parameters.keys())[1]
            self.assertEqual(model_class.main_input_name, observed_main_input_name)

    def test_dataset_conversion(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class, return_labels=False)
            if "labels" in tf_inputs_dict:
                return  # This is some kinda funky decoder model that needs labels in its forward pass
            tf_inputs_dict = {
                key: val
                for key, val in tf_inputs_dict.items()
                if "head_mask" not in key and isinstance(val, tf.Tensor)
            }
            tf_inputs_dict["extra_unwanted_column"] = list(tf_inputs_dict.values())[0]  # Use a random other tensor
            input_dataset = Dataset.from_dict(tf_inputs_dict)
            tf_dataset = model.prepare_tf_dataset(
                input_dataset, batch_size=len(input_dataset), drop_remainder=False, shuffle=False
            )
            test_batch = next(iter(tf_dataset))
            if isinstance(test_batch, tf.Tensor):
                self.assertEqual(len(test_batch), len(input_dataset))  # Assert we didn't lose any data
            elif isinstance(test_batch, dict):
                # Assert we discarded the unwanted extra column but kept everything else
                self.assertEqual(len(test_batch), len(input_dataset.features) - 1)
                self.assertNotIn("extra_unwanted_column", test_batch)
                for tensor in test_batch.values():
                    self.assertTrue(isinstance(tensor, tf.Tensor))
                    self.assertEqual(len(tensor), len(input_dataset))  # Assert we didn't lose any data
            model(test_batch, training=False)

            if "labels" in inspect.signature(model_class.call).parameters.keys():
                tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                if "labels" not in tf_inputs_dict:
                    return  # This model isn't giving us labels after all, don't try training with it
                tf_inputs_dict = {
                    key: val
                    for key, val in tf_inputs_dict.items()
                    if "head_mask" not in key and isinstance(val, tf.Tensor)
                }
                tf_inputs_dict["extra_unwanted_column"] = list(tf_inputs_dict.values())[0]  # Use a random other tensor
                input_dataset = Dataset.from_dict(tf_inputs_dict)
                tf_dataset = model.prepare_tf_dataset(
                    input_dataset, batch_size=len(input_dataset), drop_remainder=False, shuffle=False
                )
                test_batch, test_batch_labels = next(iter(tf_dataset))
                self.assertGreater(len(test_batch_labels), 0)  # Assert the labels are present
                feature_columns = 1 if isinstance(test_batch, tf.Tensor) else len(test_batch)
                label_columns = 1 if isinstance(test_batch_labels, tf.Tensor) else len(test_batch_labels)
                # Assert we discarded the unwanted extra column but kept everything else
                self.assertEqual(feature_columns + label_columns, len(input_dataset.features) - 1)
                if isinstance(test_batch, dict):
                    self.assertNotIn("extra_unwanted_column", test_batch)
                if isinstance(test_batch_labels, dict):
                    self.assertNotIn("extra_unwanted_column", test_batch_labels)
                model.compile(optimizer="sgd", run_eagerly=True)
                model.train_on_batch(test_batch, test_batch_labels)

    def _test_xla_generate(self, **generate_kwargs):
        def _generate_and_check_results(model, inputs, is_input_ids):
            # make sure there are no pad tokens in prompt, which may trigger unwanted behavior
            if is_input_ids:
                if model.generation_config.pad_token_id is not None:
                    if config.pad_token_id == 0:
                        new_pad_token = model.generation_config.pad_token_id + 1
                    else:
                        new_pad_token = model.generation_config.pad_token_id - 1
                else:
                    new_pad_token = None
                inputs = tf.where(inputs != model.generation_config.pad_token_id, inputs, new_pad_token)

            generated = model.generate(inputs, **generate_kwargs).numpy()
            generate_xla = tf.function(model.generate, jit_compile=True)
            generated_xla = generate_xla(inputs, **generate_kwargs).numpy()

            # Due to numerical instability, let's fail the test only if there are more than 10% of input sequences give
            # different outputs between XLA and non-XLA versions. If there are less than 10 examples, let's be strict
            # and not allow any difference.
            diff = [[], []]
            for _generated, _generated_xla in zip(generated.tolist(), generated_xla.tolist()):
                if _generated != _generated_xla:
                    diff[0].append(_generated)
                    diff[1].append(_generated_xla)
            ratio = len(diff[0]) / len(generated)
            if ratio > 0.1 or (len(diff[0]) > 0 and len(generated) < 10):
                self.assertListEqual(diff[0], diff[1])

        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.eos_token_id = None  # Generate until max length
            config.do_sample = False

            # extract the input to the model
            is_input_ids = "input_ids" in inputs_dict
            is_input_features = "input_features" in inputs_dict
            if not (is_input_ids or is_input_features):
                raise ValueError("No valid generate input found in inputs_dict")
            inputs = inputs_dict["input_ids"] if is_input_ids else inputs_dict["input_features"]

            # fix config for models with additional sequence-length limiting settings
            seq_len = inputs.get_shape()[1]
            for var_name in ["max_position_embeddings", "max_target_positions"]:
                attr = getattr(config, var_name, None)
                if attr is not None and attr < seq_len + generate_kwargs["max_new_tokens"]:
                    try:
                        setattr(config, var_name, seq_len + generate_kwargs["max_new_tokens"])
                    except NotImplementedError:
                        # xlnet will raise an exception when trying to set
                        # max_position_embeddings.
                        pass

            model = model_class(config)

            if model.supports_xla_generation:
                _generate_and_check_results(model, inputs, is_input_ids)
            else:
                with self.assertRaises(ValueError):
                    _generate_and_check_results(model, inputs, is_input_ids)

    def test_xla_generate_fast(self):
        """
        Basic quick test for generate-compatible classes that confirms that XLA-generated tokens are the same as their
        non XLA counterparts.

        Either the model supports XLA generation and passes the inner test, or it raises an appropriate exception
        """
        self._test_xla_generate(num_beams=1, num_return_sequences=1, max_new_tokens=3)

    @slow
    def test_xla_generate_contrastive(self):
        """
        Slow and challenging version of `test_xla_generate_fast` for contrastive search -- contrastive search directly
        manipulates the model cache and other outputs, and this test ensures that they are in a valid format that is
        also supported by XLA.

        Either the model supports XLA generation and passes the inner test, or it raises an appropriate exception
        """
        self._test_xla_generate(num_beams=1, num_return_sequences=1, max_new_tokens=16, penalty_alpha=0.5, top_k=4)

    @slow
    def test_xla_generate_slow(self):
        """
        Slow and challenging version of `test_xla_generate_fast` -- this test asks for several long sequences using
        beam search, with and without XLA. The two outputs should match, and a failure in this test indicates that the
        model may need further analysis if it is to be used for XLA generation.

        Either the model supports XLA generation and passes the inner test, or it raises an appropriate exception
        """
        self._test_xla_generate(num_beams=8, num_return_sequences=2, max_new_tokens=128)

    def _generate_random_bad_tokens(self, num_bad_tokens, model):
        # special tokens cannot be bad tokens
        special_tokens = []
        if model.config.bos_token_id is not None:
            special_tokens.append(model.config.bos_token_id)
        if model.config.pad_token_id is not None:
            special_tokens.append(model.config.pad_token_id)
        if model.config.eos_token_id is not None:
            special_tokens.append(model.config.eos_token_id)

        # create random bad tokens that are not special tokens
        bad_tokens = []
        while len(bad_tokens) < num_bad_tokens:
            token = tf.squeeze(ids_tensor((1, 1), self.model_tester.vocab_size), 0).numpy()[0]
            if token not in special_tokens:
                bad_tokens.append(token)
        return bad_tokens

    def _check_generated_ids(self, output_ids):
        for token_id in output_ids[0].numpy().tolist():
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, self.model_tester.vocab_size)

    def _check_match_tokens(self, generated_ids, bad_words_ids):
        # for all bad word tokens
        for bad_word_ids in bad_words_ids:
            # for all slices in batch
            for generated_ids_slice in generated_ids:
                # for all word idx
                for i in range(len(bad_word_ids), len(generated_ids_slice)):
                    # if tokens match
                    if generated_ids_slice[i - len(bad_word_ids) : i] == bad_word_ids:
                        return True
        return False


def ids_tensor(shape, vocab_size, rng=None, name=None, dtype=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    output = tf.constant(values, shape=shape, dtype=dtype if dtype is not None else tf.int32)

    return output


def random_attention_mask(shape, rng=None, name=None, dtype=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=None, name=None, dtype=dtype)
    # Mark the first token as 1 (matches behaviour of PyTorch/Flax function)
    attn_mask = tf.concat([tf.ones_like(attn_mask[:, :1]), attn_mask[:, 1:]], axis=1)
    return attn_mask


def floats_tensor(shape, scale=1.0, rng=None, name=None, dtype=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return tf.reshape(tf.constant(values, dtype=dtype if dtype is not None else tf.float32), shape=shape)
