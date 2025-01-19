# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Wav2Vec2-BERT model."""

import tempfile
import unittest

from datasets import load_dataset

from transformers import Wav2Vec2BertConfig, is_torch_available
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
        AutoFeatureExtractor,
        Wav2Vec2BertForAudioFrameClassification,
        Wav2Vec2BertForCTC,
        Wav2Vec2BertForSequenceClassification,
        Wav2Vec2BertForXVector,
        Wav2Vec2BertModel,
    )
    from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
        _compute_mask_indices,
        _sample_negative_indices,
    )


class Wav2Vec2BertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=200,  # speech is longer
        is_training=False,
        hidden_size=16,
        feature_projection_input_dim=16,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout_prob=0.1,
        intermediate_size=20,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        mask_time_prob=0.5,
        mask_time_length=2,
        vocab_size=32,
        do_stable_layer_norm=False,
        num_adapter_layers=2,
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
        self.feature_projection_input_dim = feature_projection_input_dim
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
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

        self.output_seq_length = self.seq_length
        self.encoder_seq_length = self.output_seq_length

        self.adapter_output_seq_length = self.output_seq_length

        for _ in range(num_adapter_layers):
            self.adapter_output_seq_length = (self.adapter_output_seq_length - 1) // adapter_stride + 1

    def prepare_config_and_inputs(self, position_embeddings_type="relative"):
        input_shape = [self.batch_size, self.seq_length, self.feature_projection_input_dim]

        input_features = floats_tensor(input_shape, self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config(position_embeddings_type=position_embeddings_type)

        return config, input_features, attention_mask

    def get_config(self, position_embeddings_type="relative"):
        return Wav2Vec2BertConfig(
            hidden_size=self.hidden_size,
            feature_projection_input_dim=self.feature_projection_input_dim,
            mask_time_prob=self.mask_time_prob,
            mask_time_length=self.mask_time_length,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
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

    def create_and_check_model(self, config, input_features, attention_mask):
        model = Wav2Vec2BertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_model_with_adapter(self, config, input_features, attention_mask):
        config.add_adapter = True
        model = Wav2Vec2BertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.adapter_output_seq_length, self.hidden_size)
        )

    def create_and_check_model_with_adapter_for_ctc(self, config, input_features, attention_mask):
        config.add_adapter = True
        config.output_hidden_size = 2 * config.hidden_size
        model = Wav2Vec2BertForCTC(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.adapter_output_seq_length, self.vocab_size)
        )

    def create_and_check_model_with_intermediate_ffn_before_adapter(self, config, input_features, attention_mask):
        config.add_adapter = True
        config.use_intermediate_ffn_before_adapter = True
        model = Wav2Vec2BertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.adapter_output_seq_length, config.output_hidden_size),
        )

        # also try with different adapter proj dim
        config.output_hidden_size = 8
        model = Wav2Vec2BertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.adapter_output_seq_length, config.output_hidden_size),
        )

    def create_and_check_model_with_adapter_proj_dim(self, config, input_features, attention_mask):
        config.add_adapter = True
        config.output_hidden_size = 8
        model = Wav2Vec2BertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.adapter_output_seq_length, config.output_hidden_size),
        )

    def create_and_check_model_float16(self, config, input_features, attention_mask):
        model = Wav2Vec2BertModel(config=config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = Wav2Vec2BertModel.from_pretrained(tmpdirname, torch_dtype=torch.float16)

        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            result = model(input_features.type(dtype=torch.float16), attention_mask=attention_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_batch_inference(self, config, input_features, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        model = Wav2Vec2BertModel(config=config)
        model.to(torch_device)
        model.eval()

        input_features = input_features[:3]
        attention_mask = torch.ones(input_features.shape, device=torch_device, dtype=torch.bool)

        input_lengths = [input_features.shape[-1] // i for i in [4, 2, 1]]

        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0.0

        batch_outputs = model(input_features, attention_mask=attention_mask).last_hidden_state

        for i in range(input_features.shape[0]):
            input_slice = input_features[i : i + 1, : input_lengths[i]]
            output = model(input_slice).last_hidden_state

            batch_output = batch_outputs[i : i + 1, : output.shape[1]]
            self.parent.assertTrue(torch.allclose(output, batch_output, atol=1e-3))

    def check_ctc_loss(self, config, input_features, *args):
        model = Wav2Vec2BertForCTC(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_features = input_features[:3]
        attention_mask = torch.ones(input_features.shape[:2], device=torch_device, dtype=torch.long)

        input_lengths = [input_features.shape[1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_features.shape[0], min(max_length_labels) - 1), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_features, attention_mask=attention_mask, labels=labels).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_features, attention_mask=attention_mask, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))

    def check_seq_classifier_loss(self, config, input_features, *args):
        model = Wav2Vec2BertForSequenceClassification(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_features = input_features[:3]
        attention_mask = torch.ones(input_features.shape[:2], device=torch_device, dtype=torch.long)

        input_lengths = [input_features.shape[1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_features.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        masked_loss = model(input_features, attention_mask=attention_mask, labels=labels).loss.item()
        unmasked_loss = model(input_features, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(masked_loss, float))
        self.parent.assertTrue(isinstance(unmasked_loss, float))
        self.parent.assertTrue(masked_loss != unmasked_loss)

    def check_ctc_training(self, config, input_features, *args):
        config.ctc_zero_infinity = True
        model = Wav2Vec2BertForCTC(config=config)
        model.to(torch_device)
        model.train()

        input_features = input_features[:3]

        input_lengths = [input_features.shape[1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_features.shape[0], max(max_length_labels) - 2), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0

            if max_length_labels[i] < labels.shape[-1]:
                # it's important that we make sure that target lengths are at least
                # one shorter than logit lengths to prevent -inf
                labels[i, max_length_labels[i] - 1 :] = -100

        loss = model(input_features, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_seq_classifier_training(self, config, input_features, *args):
        config.ctc_zero_infinity = True
        model = Wav2Vec2BertForSequenceClassification(config=config)
        model.to(torch_device)
        model.train()

        # freeze everything but the classification head
        model.freeze_base_model()

        input_features = input_features[:3]

        input_lengths = [input_features.shape[1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_features.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0

        loss = model(input_features, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_xvector_training(self, config, input_features, *args):
        config.ctc_zero_infinity = True
        model = Wav2Vec2BertForXVector(config=config)
        model.to(torch_device)
        model.train()

        # freeze everything but the classification head
        model.freeze_base_model()

        input_features = input_features[:3]

        input_lengths = [input_features.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_features.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_features[i, input_lengths[i] :] = 0.0

        loss = model(input_features, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_labels_out_of_vocab(self, config, input_features, *args):
        model = Wav2Vec2BertForCTC(config)
        model.to(torch_device)
        model.train()

        input_features = input_features[:3]

        input_lengths = [input_features.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_features.shape[0], max(max_length_labels) - 2), model.config.vocab_size + 100)

        with self.parent.assertRaises(ValueError):
            model(input_features, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_features": input_features, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class Wav2Vec2BertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Wav2Vec2BertForCTC,
            Wav2Vec2BertModel,
            Wav2Vec2BertForSequenceClassification,
            Wav2Vec2BertForAudioFrameClassification,
            Wav2Vec2BertForXVector,
        )
        if is_torch_available()
        else ()
    )

    pipeline_model_mapping = (
        {
            "audio-classification": Wav2Vec2BertForSequenceClassification,
            "automatic-speech-recognition": Wav2Vec2BertForCTC,
            "feature-extraction": Wav2Vec2BertModel,
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Wav2Vec2BertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Wav2Vec2BertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_relative(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type="relative")
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_relative_key(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type="relative_key")
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_rotary(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type="rotary")
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_no_rel_pos(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type=None)
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_adapter(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter(*config_and_inputs)

    def test_model_with_adapter_for_ctc(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter_for_ctc(*config_and_inputs)

    def test_model_with_intermediate_ffn_before_adapter(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_intermediate_ffn_before_adapter(*config_and_inputs)

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
    def test_model_float16_with_relative_key(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(position_embeddings_type="relative_key")
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

    @unittest.skip(reason="Wav2Vec2Bert has no inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="`input_ids` is renamed to `input_features`")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="Wav2Vec2Bert has no tokens embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Wav2Vec2Bert has no inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="non-robust architecture does not exist in Flax")
    @is_pt_flax_cross_test
    def test_equivalence_flax_to_pt(self):
        pass

    @unittest.skip(reason="non-robust architecture does not exist in Flax")
    @is_pt_flax_cross_test
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

        input_features = inputs_dict["input_features"]

        input_lengths = torch.tensor(
            [input_features.shape[1] for _ in range(input_features.shape[0])], dtype=torch.long, device=torch_device
        )
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)

        labels = ids_tensor((input_features.shape[0], output_lengths[0] - 2), self.model_tester.vocab_size)
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

    @unittest.skip(reason="Kept to make #Copied from working")
    def test_mask_feature_prob_ctc(self):
        pass

    @unittest.skip(reason="Kept to make #Copied from working")
    def test_mask_time_prob_ctc(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        self.assertIsNotNone(model)


@require_torch
class Wav2Vec2BertUtilsTest(unittest.TestCase):
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

    @unittest.skip(reason="Kept to make #Copied from working. Test a class used for pretraining, not yet supported.")
    def test_compute_perplexity(self):
        pass

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
class Wav2Vec2BertModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)])
        speech_samples = speech_samples[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_w2v2_bert(self):
        model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        model.to(torch_device)
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        input_speech = self._load_datasamples(2)

        inputs = feature_extractor(input_speech, return_tensors="pt", padding=True).to(torch_device)

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # fmt: off
        expected_slice_0 = torch.tensor(
            [[-0.0098, -0.0570, -0.1286,  0.0439, -0.1037, -0.0235],
            [-0.0767,  0.0574, -0.3224,  0.0482,  0.0440, -0.0193],
            [ 0.0220, -0.0878, -0.2027, -0.0028, -0.0666,  0.0721],
            [ 0.0307, -0.1099,  0.0273, -0.0416, -0.0715,  0.0094],
            [ 0.0758, -0.0291,  0.1084,  0.0004, -0.0751, -0.0116],
            [ 0.0349, -0.0343, -0.0098,  0.0415, -0.0617,  0.0241],
            [-0.0193, -0.0171,  0.1965,  0.0797, -0.0308,  0.2033],
            [-0.0323, -0.0315,  0.0948,  0.0944, -0.0254,  0.1241],
            [-0.0493,  0.0010, -0.1762,  0.0034, -0.0787,  0.0832],
            [ 0.0043, -0.1228, -0.0739,  0.0266, -0.0337, -0.0068]]
        ).to(torch_device)
        # fmt: on

        # fmt: off
        expected_slice_1 = torch.tensor(
            [[-0.0348, -0.0521, -0.3036,  0.0285, -0.0715, -0.0453],
            [-0.0102,  0.0114, -0.3266,  0.0027, -0.0558,  0.0038],
            [ 0.0454,  0.0148, -0.2418, -0.0392, -0.0455,  0.0478],
            [-0.0013,  0.0825, -0.1730, -0.0091, -0.0426,  0.0360],
            [-0.0227,  0.0687, -0.1168,  0.0569, -0.0160,  0.0759],
            [-0.0318,  0.0562, -0.0508,  0.0605,  0.0150,  0.0953],
            [-0.0415,  0.0438,  0.0233,  0.0336,  0.0262,  0.0860],
            [-0.0163,  0.0048,  0.0807,  0.0119,  0.0712,  0.0158],
            [ 0.0244, -0.0145,  0.0262, -0.0237,  0.0283, -0.0125],
            [-0.0587, -0.0516, -0.0368, -0.0196,  0.0307, -0.1434]]
        ).to(torch_device)
        # fmt: on

        self.assertTrue((outputs.last_hidden_state[0, 25:35, 4:10] - expected_slice_0).abs().max() <= 1e-4)
        self.assertTrue((outputs.last_hidden_state[1, 25:35, 4:10] - expected_slice_1).abs().max() <= 1e-4)

        self.assertAlmostEqual(outputs.last_hidden_state[1].mean().item(), 3.3123e-05)
        self.assertAlmostEqual(outputs.last_hidden_state[1].std().item(), 0.1545, delta=2e-5)

        self.assertListEqual(list(outputs.last_hidden_state.shape), [2, 326, 1024])
