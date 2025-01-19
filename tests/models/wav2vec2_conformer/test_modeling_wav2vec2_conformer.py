# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Wav2Vec2-Conformer model."""

import math
import tempfile
import unittest

import numpy as np
from datasets import load_dataset

from transformers import Wav2Vec2ConformerConfig, is_torch_available
from transformers.testing_utils import (
    is_pt_flax_cross_test,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Wav2Vec2ConformerForAudioFrameClassification,
        Wav2Vec2ConformerForCTC,
        Wav2Vec2ConformerForPreTraining,
        Wav2Vec2ConformerForSequenceClassification,
        Wav2Vec2ConformerForXVector,
        Wav2Vec2ConformerModel,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
    )
    from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
        Wav2Vec2ConformerGumbelVectorQuantizer,
        _compute_mask_indices,
        _sample_negative_indices,
    )


class Wav2Vec2ConformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,  # speech is longer
        is_training=False,
        hidden_size=16,
        feat_extract_norm="group",
        feat_extract_dropout=0.0,
        feat_extract_activation="gelu",
        conv_dim=(32, 32, 32),
        conv_stride=(4, 4, 4),
        conv_kernel=(8, 8, 8),
        conv_bias=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout_prob=0.1,
        intermediate_size=20,
        layer_norm_eps=1e-5,
        post_layer_norm=True,
        hidden_act="gelu",
        initializer_range=0.02,
        mask_time_prob=0.5,
        mask_time_length=2,
        vocab_size=32,
        do_stable_layer_norm=False,
        num_adapter_layers=1,
        adapter_stride=2,
        tdnn_dim=(32, 32),
        tdnn_kernel=(5, 3),
        tdnn_dilation=(1, 2),
        xvector_output_dim=32,
        position_embeddings_type="relative",
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_dropout = feat_extract_dropout
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.post_layer_norm = post_layer_norm
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.num_adapter_layers = num_adapter_layers
        self.adapter_stride = adapter_stride
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.scope = scope
        self.tdnn_dim = tdnn_dim
        self.tdnn_kernel = tdnn_kernel
        self.tdnn_dilation = tdnn_dilation
        self.xvector_output_dim = xvector_output_dim
        self.position_embeddings_type = position_embeddings_type

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

        self.adapter_output_seq_length = (self.output_seq_length - 1) // adapter_stride + 1

    def prepare_config_and_inputs(self, position_embeddings_type="relative"):
        input_values = floats_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config(position_embeddings_type=position_embeddings_type)

        return config, input_values, attention_mask

    def get_config(self, position_embeddings_type="relative"):
        return Wav2Vec2ConformerConfig(
            hidden_size=self.hidden_size,
            feat_extract_norm=self.feat_extract_norm,
            feat_extract_dropout=self.feat_extract_dropout,
            feat_extract_activation=self.feat_extract_activation,
            conv_dim=self.conv_dim,
            conv_stride=self.conv_stride,
            conv_kernel=self.conv_kernel,
            conv_bias=self.conv_bias,
            mask_time_prob=self.mask_time_prob,
            mask_time_length=self.mask_time_length,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            post_layer_norm=self.post_layer_norm,
            do_stable_layer_norm=self.do_stable_layer_norm,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            num_adapter_layers=self.num_adapter_layers,
            adapter_stride=self.adapter_stride,
            tdnn_dim=self.tdnn_dim,
            tdnn_kernel=self.tdnn_kernel,
            tdnn_dilation=self.tdnn_dilation,
            xvector_output_dim=self.xvector_output_dim,
            position_embeddings_type=position_embeddings_type,
        )

    def create_and_check_model(self, config, input_values, attention_mask):
        model = Wav2Vec2ConformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_model_with_feature_projection(self, config, input_values, attention_mask):
        config.hidden_size = 32
        assert config.conv_dim[-1] == config.hidden_size
        model = Wav2Vec2ConformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, config.hidden_size)
        )

    def create_and_check_model_with_no_post_layer_norm(self, config, input_values, attention_mask):
        config.post_layer_norm = False
        model = Wav2Vec2ConformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_model_with_adapter(self, config, input_values, attention_mask):
        config.add_adapter = True
        model = Wav2Vec2ConformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.adapter_output_seq_length, self.hidden_size)
        )

    def create_and_check_model_with_adapter_for_ctc(self, config, input_values, attention_mask):
        config.add_adapter = True
        config.output_hidden_size = 2 * config.hidden_size
        model = Wav2Vec2ConformerForCTC(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.adapter_output_seq_length, self.vocab_size)
        )

    def create_and_check_model_with_adapter_proj_dim(self, config, input_values, attention_mask):
        config.add_adapter = True
        config.output_hidden_size = 8
        model = Wav2Vec2ConformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.adapter_output_seq_length, config.output_hidden_size),
        )

    def create_and_check_model_float16(self, config, input_values, attention_mask):
        model = Wav2Vec2ConformerModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = Wav2Vec2ConformerModel.from_pretrained(tmpdirname, torch_dtype=torch.float16)

        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            result = model(input_values.type(dtype=torch.float16), attention_mask=attention_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_batch_inference(self, config, input_values, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        model = Wav2Vec2ConformerModel(config=config)
        model.to(torch_device)
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.bool)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0.0

        batch_outputs = model(input_values, attention_mask=attention_mask).last_hidden_state

        for i in range(input_values.shape[0]):
            input_slice = input_values[i : i + 1, : input_lengths[i]]
            output = model(input_slice).last_hidden_state

            batch_output = batch_outputs[i : i + 1, : output.shape[1]]
            self.parent.assertTrue(torch.allclose(output, batch_output, atol=1e-3))

    def check_ctc_loss(self, config, input_values, *args):
        model = Wav2Vec2ConformerForCTC(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.long)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], min(max_length_labels) - 1), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))

    def check_seq_classifier_loss(self, config, input_values, *args):
        model = Wav2Vec2ConformerForSequenceClassification(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.long)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        masked_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()
        unmasked_loss = model(input_values, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(masked_loss, float))
        self.parent.assertTrue(isinstance(unmasked_loss, float))
        self.parent.assertTrue(masked_loss != unmasked_loss)

    def check_ctc_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = Wav2Vec2ConformerForCTC(config=config)
        model.to(torch_device)
        model.train()

        # freeze feature encoder
        model.freeze_feature_encoder()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

            if max_length_labels[i] < labels.shape[-1]:
                # it's important that we make sure that target lengths are at least
                # one shorter than logit lengths to prevent -inf
                labels[i, max_length_labels[i] - 1 :] = -100

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_seq_classifier_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = Wav2Vec2ConformerForSequenceClassification(config=config)
        model.to(torch_device)
        model.train()

        # freeze everything but the classification head
        model.freeze_base_model()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_xvector_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = Wav2Vec2ConformerForXVector(config=config)
        model.to(torch_device)
        model.train()

        # freeze everything but the classification head
        model.freeze_base_model()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_labels_out_of_vocab(self, config, input_values, *args):
        model = Wav2Vec2ConformerForCTC(config)
        model.to(torch_device)
        model.train()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size + 100)

        with self.parent.assertRaises(ValueError):
            model(input_values, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class Wav2Vec2ConformerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Wav2Vec2ConformerForCTC,
            Wav2Vec2ConformerModel,
            Wav2Vec2ConformerForSequenceClassification,
            Wav2Vec2ConformerForPreTraining,
            Wav2Vec2ConformerForAudioFrameClassification,
            Wav2Vec2ConformerForXVector,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "audio-classification": Wav2Vec2ConformerForSequenceClassification,
            "automatic-speech-recognition": Wav2Vec2ConformerForCTC,
            "feature-extraction": Wav2Vec2ConformerModel,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Wav2Vec2ConformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Wav2Vec2ConformerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_relative(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type="relative")
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_rotary(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type="rotary")
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_no_rel_pos(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type=None)
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_feature_projection(*config_and_inputs)

    def test_model_with_no_post_layer_norm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_no_post_layer_norm(*config_and_inputs)

    def test_model_with_adapter(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter(*config_and_inputs)

    def test_model_with_adapter_for_ctc(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter_for_ctc(*config_and_inputs)

    def test_model_with_adapter_proj_dim(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter_proj_dim(*config_and_inputs)

    @require_torch_accelerator
    @require_torch_fp16
    def test_model_float16_with_relative(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type="relative")
        self.model_tester.create_and_check_model_float16(*config_and_inputs)

    @require_torch_accelerator
    @require_torch_fp16
    def test_model_float16_with_rotary(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type="rotary")
        self.model_tester.create_and_check_model_float16(*config_and_inputs)

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_seq_classifier_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_loss(*config_and_inputs)

    def test_ctc_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_seq_classifier_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_training(*config_and_inputs)

    def test_xvector_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_xvector_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    @unittest.skip(reason="Wav2Vec2Conformer has not inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Wav2Vec2Conformer has input_values instead of input_ids")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="Wav2Vec2Conformer has not token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Wav2Vec2Conformer has not inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @is_pt_flax_cross_test
    @unittest.skip(reason="Non-robust architecture does not exist in Flax")
    def test_equivalence_flax_to_pt(self):
        pass

    @is_pt_flax_cross_test
    @unittest.skip(reason="Non-robust architecture does not exist in Flax")
    def test_equivalence_pt_to_flax(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        # set layer drop to 0
        model.config.layerdrop = 0.0

        input_values = inputs_dict["input_values"]

        input_lengths = torch.tensor(
            [input_values.shape[1] for _ in range(input_values.shape[0])], dtype=torch.long, device=torch_device
        )
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)

        labels = ids_tensor((input_values.shape[0], output_lengths[0] - 2), self.model_tester.vocab_size)
        inputs_dict["attention_mask"] = torch.ones_like(inputs_dict["attention_mask"])
        inputs_dict["labels"] = labels

        outputs = model(**inputs_dict)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        hidden_states.retain_grad()
        attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(attentions.grad)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.weight",
                    "conv.parametrizations.weight",
                    "masked_spec_embed",
                    "codevectors",
                    "quantizer.weight_proj.weight",
                    "project_hid.weight",
                    "project_hid.bias",
                    "project_q.weight",
                    "project_q.bias",
                    "pos_bias_v",
                    "pos_bias_u",
                    "pointwise_conv1",
                    "pointwise_conv2",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                    "objective.weight",
                ]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, "pos_bias_u") and module.pos_bias_u is not None:
            module.pos_bias_u.data.fill_(3)
        if hasattr(module, "pos_bias_v") and module.pos_bias_v is not None:
            module.pos_bias_v.data.fill_(3)
        if hasattr(module, "codevectors") and module.codevectors is not None:
            module.codevectors.data.fill_(3)
        if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

    def test_mask_feature_prob_ctc(self):
        model = Wav2Vec2ConformerForCTC.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2-conformer", mask_feature_prob=0.2, mask_feature_length=2
        )
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2-conformer", return_attention_mask=True
        )

        batch_duration_in_seconds = [1, 3, 2, 6]
        input_features = [np.random.random(16_000 * s) for s in batch_duration_in_seconds]

        batch = processor(
            input_features, padding=True, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
        )

        logits = model(
            input_values=batch["input_values"].to(torch_device),
            attention_mask=batch["attention_mask"].to(torch_device),
        ).logits

        self.assertEqual(logits.shape, (4, 1498, 32))

    def test_mask_time_prob_ctc(self):
        model = Wav2Vec2ConformerForCTC.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2-conformer", mask_time_prob=0.2, mask_time_length=2
        )
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2-conformer", return_attention_mask=True
        )

        batch_duration_in_seconds = [1, 3, 2, 6]
        input_features = [np.random.random(16_000 * s) for s in batch_duration_in_seconds]

        batch = processor(
            input_features, padding=True, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
        )

        logits = model(
            input_values=batch["input_values"].to(torch_device),
            attention_mask=batch["attention_mask"].to(torch_device),
        ).logits

        self.assertEqual(logits.shape, (4, 1498, 32))

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")
        self.assertIsNotNone(model)


@require_torch
class Wav2Vec2ConformerUtilsTest(unittest.TestCase):
    def test_compute_mask_indices(self):
        batch_size = 4
        sequence_length = 60
        mask_prob = 0.5
        mask_length = 1

        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
        mask = torch.from_numpy(mask).to(torch_device)

        self.assertListEqual(mask.sum(axis=-1).tolist(), [mask_prob * sequence_length for _ in range(batch_size)])

    def test_compute_mask_indices_low_prob(self):
        # with these settings num_masked_spans=0.5, which means probabilistic rounding
        # ensures that in 5 out of 10 method calls, num_masked_spans=0, and in
        # the other 5 out of 10, cases num_masked_spans=1
        n_trials = 100
        batch_size = 4
        sequence_length = 100
        mask_prob = 0.05
        mask_length = 10

        count_dimensions_masked = 0
        count_dimensions_not_masked = 0

        for _ in range(n_trials):
            mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
            mask = torch.from_numpy(mask).to(torch_device)

            num_masks = torch.sum(mask).item()

            if num_masks > 0:
                count_dimensions_masked += 1
            else:
                count_dimensions_not_masked += 1

        # as we test for at least 10 masked dimension and at least
        # 10 non-masked dimension, this test could fail with probability:
        # P(100 coin flips, at most 9 heads) = 1.66e-18
        self.assertGreater(count_dimensions_masked, int(n_trials * 0.1))
        self.assertGreater(count_dimensions_not_masked, int(n_trials * 0.1))

    def test_compute_mask_indices_overlap(self):
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4

        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
        mask = torch.from_numpy(mask).to(torch_device)

        # because of overlap mask don't have to add up exactly to `mask_prob * sequence_length`, but have to be smaller or equal
        for batch_sum in mask.sum(axis=-1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)

    def test_compute_mask_indices_attn_mask_overlap(self):
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4

        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=torch_device)
        attention_mask[:2, sequence_length // 2 :] = 0

        mask = _compute_mask_indices(
            (batch_size, sequence_length), mask_prob, mask_length, attention_mask=attention_mask
        )
        mask = torch.from_numpy(mask).to(torch_device)

        for batch_sum in mask.sum(axis=-1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)

        self.assertTrue(mask[:2, sequence_length // 2 :].sum() == 0)

    def test_compute_mask_indices_short_audio(self):
        batch_size = 4
        sequence_length = 100
        mask_prob = 0.05
        mask_length = 10

        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=torch_device)
        # force one example to be heavily padded
        attention_mask[0, 5:] = 0

        mask = _compute_mask_indices(
            (batch_size, sequence_length), mask_prob, mask_length, attention_mask=attention_mask, min_masks=2
        )

        # make sure that non-padded examples cannot be padded
        self.assertFalse(mask[0][attention_mask[0].to(torch.bool).cpu()].any())

    def test_compute_perplexity(self):
        probs = torch.arange(100, device=torch_device).reshape(2, 5, 10) / 100

        ppl = Wav2Vec2ConformerGumbelVectorQuantizer._compute_perplexity(probs)
        self.assertTrue(abs(ppl.item() - 141.4291) < 1e-3)

        # mask half of the input
        mask = torch.ones((2,), device=torch_device, dtype=torch.bool)
        mask[0] = 0

        ppl = Wav2Vec2ConformerGumbelVectorQuantizer._compute_perplexity(probs, mask)
        self.assertTrue(abs(ppl.item() - 58.6757) < 1e-3)

    def test_sample_negatives(self):
        batch_size = 2
        sequence_length = 10
        hidden_size = 4
        num_negatives = 3

        features = (torch.arange(sequence_length * hidden_size, device=torch_device) // hidden_size).view(
            sequence_length, hidden_size
        )  # each value in vector consits of same value
        features = features[None, :].expand(batch_size, sequence_length, hidden_size).contiguous()

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices((batch_size, sequence_length), num_negatives, None)
        sampled_negative_indices = torch.from_numpy(sampled_negative_indices).to(torch_device)
        negatives = features.view(-1, hidden_size)[sampled_negative_indices.long().view(-1)]
        negatives = negatives.view(batch_size, sequence_length, -1, hidden_size).permute(2, 0, 1, 3)
        self.assertTrue(negatives.shape == (num_negatives, batch_size, sequence_length, hidden_size))

        # make sure no negatively sampled vector is actually a positive one
        for negative in negatives:
            self.assertTrue(((negative - features) == 0).sum() == 0.0)

        # make sure that full vectors are sampled and not values of vectors => this means that `unique()` yields a single value for `hidden_size` dim
        self.assertTrue(negatives.unique(dim=-1).shape, (num_negatives, batch_size, sequence_length, 1))

    def test_sample_negatives_with_mask(self):
        batch_size = 2
        sequence_length = 10
        hidden_size = 4
        num_negatives = 3

        # second half of last input tensor is padded
        mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=torch_device)
        mask[-1, sequence_length // 2 :] = 0

        features = (torch.arange(sequence_length * hidden_size, device=torch_device) // hidden_size).view(
            sequence_length, hidden_size
        )  # each value in vector consits of same value
        features = features[None, :].expand(batch_size, sequence_length, hidden_size).contiguous()

        # replace masked feature vectors with -100 to test that those are not sampled
        features = torch.where(mask[:, :, None].expand(features.shape).bool(), features, -100)

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            (batch_size, sequence_length), num_negatives, mask.cpu().numpy()
        )
        sampled_negative_indices = torch.from_numpy(sampled_negative_indices).to(torch_device)
        negatives = features.view(-1, hidden_size)[sampled_negative_indices.long().view(-1)]
        negatives = negatives.view(batch_size, sequence_length, -1, hidden_size).permute(2, 0, 1, 3)

        self.assertTrue((negatives >= 0).all().item())

        self.assertTrue(negatives.shape == (num_negatives, batch_size, sequence_length, hidden_size))

        # make sure no negatively sampled vector is actually a positive one
        for negative in negatives:
            self.assertTrue(((negative - features) == 0).sum() == 0.0)

        # make sure that full vectors are sampled and not values of vectors => this means that `unique()` yields a single value for `hidden_size` dim
        self.assertTrue(negatives.unique(dim=-1).shape, (num_negatives, batch_size, sequence_length, 1))


@require_torch
@slow
class Wav2Vec2ConformerModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)])
        speech_samples = speech_samples[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_ctc_normal_batched_rel_pos(self):
        model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
        model.to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-conformer-rel-pos-large-960h-ft", do_lower_case=True
        )

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loincloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_normal_batched_rope(self):
        model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
        model.to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-conformer-rope-large-960h-ft", do_lower_case=True
        )

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_pretrained(self):
        model = Wav2Vec2ConformerForPreTraining.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")
        model.to(torch_device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-conformer-rel-pos-large", return_attention_mask=True
        )
        input_speech = self._load_datasamples(2)

        inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

        batch_size = inputs_dict["input_values"].shape[0]
        feature_seq_length = int(model._get_feat_extract_output_lengths(inputs_dict["input_values"].shape[1]))

        features_shape = (batch_size, feature_seq_length)

        torch.manual_seed(0)
        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            min_masks=2,
        )
        mask_time_indices = torch.from_numpy(mask_time_indices).to(torch_device)

        with torch.no_grad():
            outputs = model(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # compute cosine similarity
        cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        # retrieve cosine sim of masked features
        cosine_sim_masked = cosine_sim[mask_time_indices]

        # ... now compare to randomly initialized model

        config = Wav2Vec2ConformerConfig.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")
        model_rand = Wav2Vec2ConformerForPreTraining(config).to(torch_device).eval()

        with torch.no_grad():
            outputs_rand = model_rand(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # compute cosine similarity
        cosine_sim_rand = torch.cosine_similarity(
            outputs_rand.projected_states, outputs_rand.projected_quantized_states, dim=-1
        )

        # retrieve cosine sim of masked features
        cosine_sim_masked_rand = cosine_sim_rand[mask_time_indices]

        # a pretrained wav2vec2_conformer model has learned to predict the quantized latent states
        # => the cosine similarity between quantized states and predicted states > 0.5
        # a random wav2vec2_conformer model has not learned to predict the quantized latent states
        # => the cosine similarity between quantized states and predicted states is very likely < 0.1
        self.assertTrue(cosine_sim_masked.mean().item() - 5 * cosine_sim_masked_rand.mean().item() > 0)
