# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Wav2Vec2 model."""

import math
import multiprocessing
import os
import pickle
import tempfile
import traceback
import unittest

import numpy as np
from datasets import load_dataset
from pytest import mark

from transformers import Wav2Vec2Config, is_torch_available
from transformers.testing_utils import (
    CaptureLogger,
    cleanup,
    is_pt_flax_cross_test,
    is_pyctcdecode_available,
    is_torchaudio_available,
    require_flash_attn,
    require_pyctcdecode,
    require_soundfile,
    require_torch,
    require_torch_gpu,
    require_torchaudio,
    run_test_in_subprocess,
    slow,
    torch_device,
)
from transformers.utils import is_torch_fx_available

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
    from safetensors.torch import save_file as safe_save_file

    from transformers import (
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForAudioFrameClassification,
        Wav2Vec2ForCTC,
        Wav2Vec2ForMaskedLM,
        Wav2Vec2ForPreTraining,
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2ForXVector,
        Wav2Vec2Model,
        Wav2Vec2Processor,
    )
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        WAV2VEC2_ADAPTER_PT_FILE,
        WAV2VEC2_ADAPTER_SAFE_FILE,
        Wav2Vec2GumbelVectorQuantizer,
        _compute_mask_indices,
        _sample_negative_indices,
    )


if is_torchaudio_available():
    import torchaudio


if is_pyctcdecode_available():
    import pyctcdecode.decoder

    from transformers import Wav2Vec2ProcessorWithLM
    from transformers.models.wav2vec2_with_lm import processing_wav2vec2_with_lm


if is_torch_fx_available():
    from transformers.utils.fx import symbolic_trace


def _test_wav2vec2_with_lm_invalid_pool(in_queue, out_queue, timeout):
    error = None
    try:
        _ = in_queue.get(timeout=timeout)

        ds = load_dataset(
            "mozilla-foundation/common_voice_11_0", "es", split="test", streaming=True, trust_remote_code=True
        )
        sample = next(iter(ds))

        resampled_audio = torchaudio.functional.resample(
            torch.tensor(sample["audio"]["array"]), 48_000, 16_000
        ).numpy()

        model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm").to(
            torch_device
        )
        processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm")

        input_values = processor(resampled_audio, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values.to(torch_device)).logits

        # use a spawn pool, which should trigger a warning if different than fork
        with CaptureLogger(pyctcdecode.decoder.logger) as cl, multiprocessing.get_context("spawn").Pool(1) as pool:
            transcription = processor.batch_decode(logits.cpu().numpy(), pool).text

        unittest.TestCase().assertIn("Falling back to sequential decoding.", cl.out)
        unittest.TestCase().assertEqual(transcription[0], "habitan aguas poco profundas y rocosas")

        # force batch_decode to internally create a spawn pool, which should trigger a warning if different than fork
        multiprocessing.set_start_method("spawn", force=True)
        with CaptureLogger(processing_wav2vec2_with_lm.logger) as cl:
            transcription = processor.batch_decode(logits.cpu().numpy()).text

        unittest.TestCase().assertIn("Falling back to sequential decoding.", cl.out)
        unittest.TestCase().assertEqual(transcription[0], "habitan aguas poco profundas y rocosas")
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


class Wav2Vec2ModelTester:
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
        hidden_dropout_prob=0.1,  # this is most likely not correctly set yet
        intermediate_size=20,
        layer_norm_eps=1e-5,
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

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

        self.adapter_output_seq_length = (self.output_seq_length - 1) // adapter_stride + 1

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_values, attention_mask

    def get_config(self):
        return Wav2Vec2Config(
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
        )

    def create_and_check_model(self, config, input_values, attention_mask):
        model = Wav2Vec2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_model_with_adapter(self, config, input_values, attention_mask):
        config.add_adapter = True
        model = Wav2Vec2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.adapter_output_seq_length, self.hidden_size)
        )

    def create_and_check_model_with_adapter_for_ctc(self, config, input_values, attention_mask):
        config.add_adapter = True
        config.output_hidden_size = 2 * config.hidden_size
        model = Wav2Vec2ForCTC(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.adapter_output_seq_length, self.vocab_size)
        )

    def create_and_check_model_with_adapter_proj_dim(self, config, input_values, attention_mask):
        config.add_adapter = True
        config.output_hidden_size = 8
        model = Wav2Vec2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.adapter_output_seq_length, config.output_hidden_size),
        )

    def create_and_check_model_with_attn_adapter(self, config, input_values, attention_mask):
        config.adapter_attn_dim = 16
        model = Wav2Vec2ForCTC(config=config)

        self.parent.assertIsNotNone(model._get_adapters())

        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.output_seq_length, self.vocab_size))

    def create_and_check_batch_inference(self, config, input_values, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        model = Wav2Vec2Model(config=config)
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
        model = Wav2Vec2ForCTC(config=config)
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
        model = Wav2Vec2ForSequenceClassification(config=config)
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
        model = Wav2Vec2ForCTC(config=config)
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
        model = Wav2Vec2ForSequenceClassification(config=config)
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
        model = Wav2Vec2ForXVector(config=config)
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
        model = Wav2Vec2ForCTC(config)
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
class Wav2Vec2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2ForMaskedLM, Wav2Vec2ForSequenceClassification, Wav2Vec2ForPreTraining)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "audio-classification": Wav2Vec2ForSequenceClassification,
            "automatic-speech-recognition": Wav2Vec2ForCTC,
            "feature-extraction": Wav2Vec2Model,
            "fill-mask": Wav2Vec2ForMaskedLM,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = True
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        self.model_tester = Wav2Vec2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Wav2Vec2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_adapter(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter(*config_and_inputs)

    def test_model_with_adapter_for_ctc(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter_for_ctc(*config_and_inputs)

    def test_model_with_adapter_proj_dim(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter_proj_dim(*config_and_inputs)

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

    @unittest.skip(reason="Model has no inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Model has input_values instead of input_ids")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="Model has no tokens embeds")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Model has no inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @is_pt_flax_cross_test
    @unittest.skip(reason="Non-rubst architecture does not exist in Flax")
    def test_equivalence_flax_to_pt(self):
        pass

    @is_pt_flax_cross_test
    @unittest.skip(reason="Non-rubst architecture does not exist in Flax")
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
        if hasattr(module, "codevectors") and module.codevectors is not None:
            module.codevectors.data.fill_(3)
        if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

    def test_mask_feature_prob_ctc(self):
        model = Wav2Vec2ForCTC.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", mask_feature_prob=0.2, mask_feature_length=2
        )
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
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
        model = Wav2Vec2ForCTC.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", mask_time_prob=0.2, mask_time_length=2
        )
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
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
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsNotNone(model)

    # Wav2Vec2 cannot be torchscripted because of group norm.
    def _create_and_check_torch_fx_tracing(self, config, inputs_dict, output_loss=False):
        # TODO: fix it
        self.skipTest(reason="torch 2.1 breaks torch fx tests for wav2vec2/hubert.")

        if not is_torch_fx_available() or not self.fx_compatible:
            self.skipTest(reason="torch fx not available or not compatible with this model")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.return_dict = False

        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=output_loss)

            try:
                input_names = [
                    "attention_mask",
                    "bbox",
                    "input_features",
                    "input_ids",
                    "input_values",
                    "pixel_values",
                    "token_type_ids",
                    "visual_feats",
                    "visual_pos",
                ]

                labels = inputs.get("labels", None)
                start_positions = inputs.get("start_positions", None)
                end_positions = inputs.get("end_positions", None)
                if labels is not None:
                    input_names.append("labels")
                if start_positions is not None:
                    input_names.append("start_positions")
                if end_positions is not None:
                    input_names.append("end_positions")

                filtered_inputs = {k: v for (k, v) in inputs.items() if k in input_names}
                input_names = list(filtered_inputs.keys())

                model_output = model(**filtered_inputs)

                if (
                    isinstance(model, Wav2Vec2ForSequenceClassification)
                    and not hasattr(model.config, "problem_type")
                    or model.config.problem_type is None
                ):
                    model.config.problem_type = "single_label_classification"

                traced_model = symbolic_trace(model, input_names)
                traced_output = traced_model(**filtered_inputs)

            except Exception as e:
                self.fail(f"Couldn't trace module: {e}")

            def flatten_output(output):
                flatten = []
                for x in output:
                    if isinstance(x, (tuple, list)):
                        flatten += flatten_output(x)
                    elif not isinstance(x, torch.Tensor):
                        continue
                    else:
                        flatten.append(x)
                return flatten

            model_output = flatten_output(model_output)
            traced_output = flatten_output(traced_output)
            num_outputs = len(model_output)

            for i in range(num_outputs):
                self.assertTrue(
                    torch.allclose(model_output[i], traced_output[i]),
                    f"traced {i}th output doesn't match model {i}th output for {model_class}",
                )

            # Test that the model can be serialized and restored properly
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pkl_file_name = os.path.join(tmp_dir_name, "model.pkl")
                try:
                    with open(pkl_file_name, "wb") as f:
                        pickle.dump(traced_model, f)
                    with open(pkl_file_name, "rb") as f:
                        loaded = pickle.load(f)
                except Exception as e:
                    self.fail(f"Couldn't serialize / deserialize the traced model: {e}")

                loaded_output = loaded(**filtered_inputs)
                loaded_output = flatten_output(loaded_output)

                for i in range(num_outputs):
                    self.assertTrue(
                        torch.allclose(model_output[i], loaded_output[i]),
                        f"serialized model {i}th output doesn't match model {i}th output for {model_class}",
                    )

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()

    @unittest.skip(
        "Need to investigate why config.do_stable_layer_norm is set to False here when it doesn't seem to be supported"
    )
    def test_flax_from_pt_safetensors(self):
        return


@require_torch
class Wav2Vec2RobustModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Wav2Vec2ForCTC,
            Wav2Vec2Model,
            Wav2Vec2ForMaskedLM,
            Wav2Vec2ForSequenceClassification,
            Wav2Vec2ForPreTraining,
            Wav2Vec2ForAudioFrameClassification,
            Wav2Vec2ForXVector,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        self.model_tester = Wav2Vec2ModelTester(
            self, conv_stride=(3, 3, 3), feat_extract_norm="layer", do_stable_layer_norm=True
        )
        self.config_tester = ConfigTester(self, config_class=Wav2Vec2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_adapter(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter(*config_and_inputs)

    def test_model_with_adapter_proj_dim(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter_proj_dim(*config_and_inputs)

    def test_model_with_attn_adapter(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_attn_adapter(*config_and_inputs)

    def test_batched_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_batch_inference(*config_and_inputs)

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

    @unittest.skip(reason="Model has no input_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Model has input_values instead of input_ids")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="Model has no token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Model has no input_embeds")
    def test_model_get_set_embeddings(self):
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
        if hasattr(module, "codevectors") and module.codevectors is not None:
            module.codevectors.data.fill_(3)
        if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

    def test_model_for_pretraining(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Wav2Vec2ForPreTraining(config).to(torch_device)

        batch_size = inputs_dict["input_values"].shape[0]
        feature_seq_length = int(model._get_feat_extract_output_lengths(inputs_dict["input_values"].shape[1]))

        features_shape = (batch_size, feature_seq_length)

        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            min_masks=2,
        )
        sampled_negative_indices = _sample_negative_indices(features_shape, 10, mask_time_indices)

        mask_time_indices = torch.from_numpy(mask_time_indices).to(torch_device)
        sampled_negative_indices = torch.from_numpy(sampled_negative_indices).to(torch_device)

        loss = model(
            inputs_dict["input_values"],
            attention_mask=inputs_dict["attention_mask"],
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
        ).loss

        # more losses
        mask_time_indices[:, : mask_time_indices.shape[-1] // 2] = True

        sampled_negative_indices = _sample_negative_indices(features_shape, 10, mask_time_indices.cpu().numpy())
        sampled_negative_indices = torch.from_numpy(sampled_negative_indices).to(torch_device)
        loss_more_masked = model(
            inputs_dict["input_values"],
            attention_mask=inputs_dict["attention_mask"],
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
        ).loss

        # loss_more_masked has to be bigger or equal loss since more masked inputs have to be predicted
        self.assertTrue(loss.detach().item() <= loss_more_masked.detach().item())

    def test_mask_feature_prob_ctc(self):
        model = Wav2Vec2ForCTC.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", mask_feature_prob=0.2, mask_feature_length=2
        )
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
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
        model = Wav2Vec2ForCTC.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", mask_time_prob=0.2, mask_time_length=2
        )
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
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

    def test_mask_time_feature_prob_ctc_single_batch(self):
        model = Wav2Vec2ForCTC.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2",
            mask_time_prob=0.2,
            mask_feature_prob=0.2,
            mask_time_length=2,
            mask_feature_length=2,
        )
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
        )

        batch_duration_in_seconds = [6]
        input_features = [np.random.random(16_000 * s) for s in batch_duration_in_seconds]

        batch = processor(
            input_features, padding=True, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
        )

        logits = model(
            input_values=batch["input_values"].to(torch_device),
            attention_mask=batch["attention_mask"].to(torch_device),
        ).logits

        self.assertEqual(logits.shape, (1, 1498, 32))

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    def test_load_and_set_attn_adapter(self):
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
        )

        def get_logits(model, input_features):
            model = model.to(torch_device)
            batch = processor(
                input_features,
                padding=True,
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = model(
                    input_values=batch["input_values"].to(torch_device),
                    attention_mask=batch["attention_mask"].to(torch_device),
                ).logits
            return logits

        input_features = [np.random.random(16_000 * s) for s in [1, 3, 2, 6]]

        model = Wav2Vec2ForCTC.from_pretrained("hf-internal-testing/tiny-random-wav2vec2-adapter", target_lang="it")

        logits = get_logits(model, input_features)

        model_2 = Wav2Vec2ForCTC.from_pretrained("hf-internal-testing/tiny-random-wav2vec2-adapter")
        model_2.load_adapter("it")

        logits_2 = get_logits(model_2, input_features)

        torch.testing.assert_close(logits, logits_2, rtol=1e-3, atol=1e-3)

    # test that loading adapter weights with mismatched vocab sizes can be loaded
    def test_load_target_lang_with_mismatched_size(self):
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
        )

        def get_logits(model, input_features):
            model = model.to(torch_device)
            batch = processor(
                input_features,
                padding=True,
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = model(
                    input_values=batch["input_values"].to(torch_device),
                    attention_mask=batch["attention_mask"].to(torch_device),
                ).logits
            return logits

        input_features = [np.random.random(16_000 * s) for s in [1, 3, 2, 6]]

        model = Wav2Vec2ForCTC.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2-adapter", target_lang="fr", ignore_mismatched_sizes=True
        )

        logits = get_logits(model, input_features)

        model_2 = Wav2Vec2ForCTC.from_pretrained("hf-internal-testing/tiny-random-wav2vec2-adapter")
        model_2.load_adapter("fr")

        logits_2 = get_logits(model_2, input_features)

        torch.testing.assert_close(logits, logits_2, rtol=1e-3, atol=1e-3)

    def test_load_attn_adapter(self):
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
        )

        def get_logits(model, input_features):
            model = model.to(torch_device)
            batch = processor(
                input_features,
                padding=True,
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = model(
                    input_values=batch["input_values"].to(torch_device),
                    attention_mask=batch["attention_mask"].to(torch_device),
                ).logits
            return logits

        input_features = [np.random.random(16_000 * s) for s in [1, 3, 2, 6]]

        model = Wav2Vec2ForCTC.from_pretrained("hf-internal-testing/tiny-random-wav2vec2", adapter_attn_dim=16)

        with tempfile.TemporaryDirectory() as tempdir:
            model.save_pretrained(tempdir)
            model = Wav2Vec2ForCTC.from_pretrained(tempdir)

            logits = get_logits(model, input_features)
            adapter_weights = model._get_adapters()

            # save safe weights
            safe_filepath = os.path.join(tempdir, WAV2VEC2_ADAPTER_SAFE_FILE.format("eng"))
            safe_save_file(adapter_weights, safe_filepath, metadata={"format": "pt"})

            model.load_adapter("eng")
            model.load_adapter("eng", use_safetensors=True)

            with self.assertRaises(OSError):
                model.load_adapter("eng", use_safetensors=False)
            with self.assertRaises(Exception):
                model.load_adapter("ita", use_safetensors=True)
            logits_2 = get_logits(model, input_features)

            torch.testing.assert_close(logits, logits_2, rtol=1e-3, atol=1e-3)

        with tempfile.TemporaryDirectory() as tempdir:
            model.save_pretrained(tempdir)
            model = Wav2Vec2ForCTC.from_pretrained(tempdir)

            logits = get_logits(model, input_features)
            adapter_weights = model._get_adapters()

            # save pt weights
            pt_filepath = os.path.join(tempdir, WAV2VEC2_ADAPTER_PT_FILE.format("eng"))
            torch.save(adapter_weights, pt_filepath)

            model.load_adapter("eng")
            model.load_adapter("eng", use_safetensors=False)

            with self.assertRaises(OSError):
                model.load_adapter("eng", use_safetensors=True)

            logits_2 = get_logits(model, input_features)

            torch.testing.assert_close(logits, logits_2, rtol=1e-3, atol=1e-3)

        model = Wav2Vec2ForCTC.from_pretrained("hf-internal-testing/tiny-random-wav2vec2-adapter")
        logits = get_logits(model, input_features)

        model.load_adapter("eng")
        model.load_adapter("eng", use_safetensors=False)
        model.load_adapter("eng", use_safetensors=True)

        logits_2 = get_logits(model, input_features)

        torch.testing.assert_close(logits, logits_2, rtol=1e-3, atol=1e-3)

    @slow
    def test_model_from_pretrained(self):
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsNotNone(model)


@require_torch
class Wav2Vec2UtilsTest(unittest.TestCase):
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

        ppl = Wav2Vec2GumbelVectorQuantizer._compute_perplexity(probs)
        self.assertTrue(abs(ppl.item() - 141.4291) < 1e-3)

        # mask half of the input
        mask = torch.ones((2,), device=torch_device, dtype=torch.bool)
        mask[0] = 0

        ppl = Wav2Vec2GumbelVectorQuantizer._compute_perplexity(probs, mask)
        self.assertTrue(abs(ppl.item() - 58.6757) < 1e-3)

    def test_sample_negatives(self):
        batch_size = 2
        sequence_length = 10
        hidden_size = 4
        num_negatives = 3
        sequence = torch.div(
            torch.arange(sequence_length * hidden_size, device=torch_device), hidden_size, rounding_mode="floor"
        )
        features = sequence.view(sequence_length, hidden_size)  # each value in vector consits of same value
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
        self.assertEqual(negatives.unique(dim=-1).shape, (num_negatives, batch_size, sequence_length, 1))

    def test_sample_negatives_with_mask(self):
        batch_size = 2
        sequence_length = 10
        hidden_size = 4
        num_negatives = 3

        # second half of last input tensor is padded
        mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=torch_device)
        mask[-1, sequence_length // 2 :] = 0

        sequence = torch.div(
            torch.arange(sequence_length * hidden_size, device=torch_device), hidden_size, rounding_mode="floor"
        )
        features = sequence.view(sequence_length, hidden_size)  # each value in vector consits of same value
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
        self.assertEqual(negatives.unique(dim=-1).shape, (num_negatives, batch_size, sequence_length, 1))


@require_torch
@require_soundfile
@slow
class Wav2Vec2ModelIntegrationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        cleanup(torch_device, gc_collect=True)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def _load_superb(self, task, num_samples):
        ds = load_dataset("anton-l/superb_dummy", task, split="test", trust_remote_code=True)

        return ds[:num_samples]

    def test_inference_ctc_normal(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)
        input_speech = self._load_datasamples(1)

        input_values = processor(input_speech, return_tensors="pt").input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_normal_batched(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight lowing cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_robust_batched(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", do_lower_case=True)

        input_speech = self._load_datasamples(4)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
            "the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around"
            " him with the thousands of spectators were trivialities not worth thinking about",
            "his instant panic was followed by a small sharp blow high on his chest",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    @unittest.skipIf(torch_device != "cpu", "cannot make deterministic on GPU")
    def test_inference_integration(self):
        model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
        model.to(torch_device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        input_speech = self._load_datasamples(2)

        inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

        batch_size = inputs_dict["input_values"].shape[0]
        feature_seq_length = int(model._get_feat_extract_output_lengths(inputs_dict["input_values"].shape[1]))

        features_shape = (batch_size, feature_seq_length)

        np.random.seed(4)
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
                mask_time_indices=mask_time_indices,
            )

        # compute cosine similarity
        cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        # retrieve cosine sim of masked features
        cosine_sim_masked = cosine_sim[mask_time_indices]

        # cosine similarity of model is all > 0.5 as model is
        # pre-trained on contrastive loss
        # fmt: off
        expected_cosine_sim_masked = torch.tensor([
            0.8523, 0.5860, 0.6905, 0.5557, 0.7456, 0.5249, 0.6639, 0.7654, 0.7565,
            0.8167, 0.8222, 0.7960, 0.8034, 0.8166, 0.8310, 0.8263, 0.8274, 0.8258,
            0.8179, 0.8412, 0.8536, 0.5098, 0.4728, 0.6461, 0.4498, 0.6002, 0.5774,
            0.6457, 0.7123, 0.5668, 0.6866, 0.4960, 0.6293, 0.7423, 0.7419, 0.7526,
            0.7768, 0.4898, 0.5393, 0.8183
        ], device=torch_device)
        # fmt: on

        torch.testing.assert_close(cosine_sim_masked, expected_cosine_sim_masked, rtol=1e-3, atol=1e-3)

    def test_inference_pretrained(self):
        model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
        model.to(torch_device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
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

        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
        model_rand = Wav2Vec2ForPreTraining(config).to(torch_device).eval()

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

        # a pretrained wav2vec2 model has learned to predict the quantized latent states
        # => the cosine similarity between quantized states and predicted states > 0.5
        # a random wav2vec2 model has not learned to predict the quantized latent states
        # => the cosine similarity between quantized states and predicted states is very likely < 0.1
        self.assertTrue(cosine_sim_masked.mean().item() - 5 * cosine_sim_masked_rand.mean().item() > 0)

    @unittest.skipIf(torch_device != "cpu", "cannot make deterministic on GPU")
    def test_loss_pretraining(self):
        model = Wav2Vec2ForPreTraining.from_pretrained(
            "facebook/wav2vec2-base",
            attention_dropout=0.0,
            feat_proj_dropout=0.0,
            hidden_dropout=0.0,
            layerdrop=0.0,
        )
        model.to(torch_device).train()

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        input_speech = self._load_datasamples(2)

        inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

        batch_size = inputs_dict["input_values"].shape[0]
        feature_seq_length = int(model._get_feat_extract_output_lengths(inputs_dict["input_values"].shape[1]))

        features_shape = (batch_size, feature_seq_length)

        torch.manual_seed(0)
        np.random.seed(0)

        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            min_masks=2,
        )
        sampled_negative_indices = _sample_negative_indices(
            mask_time_indices.shape, model.config.num_negatives, mask_time_indices
        )

        mask_time_indices = torch.from_numpy(mask_time_indices).to(torch_device)
        sampled_negative_indices = torch.from_numpy(sampled_negative_indices).to(torch_device)

        with torch.no_grad():
            outputs = model(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
                sampled_negative_indices=sampled_negative_indices,
            )

        # check diversity loss
        num_codevectors = model.config.num_codevectors_per_group * model.config.num_codevector_groups
        diversity_loss = (num_codevectors - outputs.codevector_perplexity) / num_codevectors
        self.assertTrue(abs(diversity_loss.item() - 0.9538) < 1e-3)

        # check overall loss (contrastive loss + diversity loss)
        expected_loss = 116.7094

        self.assertTrue(abs(outputs.loss.item() - expected_loss) < 1e-3)

    def test_inference_keyword_spotting(self):
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks").to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
        input_data = self._load_superb("ks", 4)
        inputs = processor(input_data["speech"], return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
        predicted_logits, predicted_ids = torch.max(outputs.logits, dim=-1)

        expected_labels = [7, 6, 10, 9]
        # s3prl logits for the same batch
        expected_logits = torch.tensor([6.1186, 11.8961, 10.2931, 6.0898], device=torch_device)

        self.assertListEqual(predicted_ids.tolist(), expected_labels)
        torch.testing.assert_close(predicted_logits, expected_logits, rtol=1e-2, atol=1e-2)

    def test_inference_intent_classification(self):
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ic").to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ic")
        input_data = self._load_superb("ic", 4)
        inputs = processor(input_data["speech"], return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)

        predicted_logits_action, predicted_ids_action = torch.max(outputs.logits[:, :6], dim=-1)
        predicted_logits_object, predicted_ids_object = torch.max(outputs.logits[:, 6:20], dim=-1)
        predicted_logits_location, predicted_ids_location = torch.max(outputs.logits[:, 20:24], dim=-1)

        expected_labels_action = [0, 0, 2, 3]
        expected_logits_action = torch.tensor([0.4568, 11.0848, 1.6621, 9.3841], device=torch_device)
        expected_labels_object = [3, 10, 3, 4]
        expected_logits_object = torch.tensor([1.5322, 10.7094, 5.2469, 22.1318], device=torch_device)
        expected_labels_location = [0, 0, 0, 1]
        expected_logits_location = torch.tensor([1.5335, 6.5096, 10.5704, 11.0569], device=torch_device)

        self.assertListEqual(predicted_ids_action.tolist(), expected_labels_action)
        self.assertListEqual(predicted_ids_object.tolist(), expected_labels_object)
        self.assertListEqual(predicted_ids_location.tolist(), expected_labels_location)

        torch.testing.assert_close(predicted_logits_action, expected_logits_action, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(predicted_logits_object, expected_logits_object, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(predicted_logits_location, expected_logits_location, rtol=1e-2, atol=1e-2)

    def test_inference_speaker_identification(self):
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid").to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
        input_data = self._load_superb("si", 4)

        output_logits = []
        with torch.no_grad():
            for example in input_data["speech"]:
                input = processor(example, return_tensors="pt", padding=True)
                output = model(input.input_values.to(torch_device), attention_mask=None)
                output_logits.append(output.logits[0])
        output_logits = torch.stack(output_logits)
        predicted_logits, predicted_ids = torch.max(output_logits, dim=-1)

        expected_labels = [251, 1, 1, 3]
        # s3prl logits for the same batch
        expected_logits = torch.tensor([37.5627, 71.6362, 64.2419, 31.7778], device=torch_device)

        self.assertListEqual(predicted_ids.tolist(), expected_labels)
        torch.testing.assert_close(predicted_logits, expected_logits, rtol=1e-2, atol=1e-2)

    def test_inference_emotion_recognition(self):
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er").to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
        input_data = self._load_superb("er", 4)
        inputs = processor(input_data["speech"], return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
        predicted_logits, predicted_ids = torch.max(outputs.logits, dim=-1)

        expected_labels = [1, 1, 2, 2]
        # s3prl logits for the same batch
        expected_logits = torch.tensor([2.1722, 3.0779, 8.0287, 6.6797], device=torch_device)

        self.assertListEqual(predicted_ids.tolist(), expected_labels)
        torch.testing.assert_close(predicted_logits, expected_logits, rtol=1e-2, atol=1e-2)

    def test_phoneme_recognition(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

        input_speech = self._load_datasamples(4)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "ɐ m æ n s ɛ d t ə ð ə j uː n ɪ v ɚ s s ɚ aɪ ɛ ɡ z ɪ s t",
            "s w ɛ t k ʌ v ɚ d b ɹ iː ɔ n z b ɑː d i t ɹ ɪ k l ɪ ŋ ɪ n t ə ð ə t aɪ t l oɪ n k l ɑː θ ð æ w ʌ z ð ɪ oʊ"
            " n l i ɡ ɑːɹ m ə n t h iː w ɔːɹ",
            "ð ə k aɪ t ɔ n h ɪ z tʃ ɛ s t s t ɪ l d ɹ ɪ p ɪ ŋ b l ʌ d ð ɪ eɪ k ʌ v h ɪ z oʊ v ɚ s t ɹ eɪ n d aɪ z iː"
            " v ə n ð ə s ɔːɹ ɹ ɪ ŋ ɐ ɹ iː n ɐ ɚ ɹ aʊ n d h ɪ m w ɪ ð ə θ aʊ z ə n d z ʌ v s p ɛ k t eɪ ɾ ɚ z w ɜː t ɹ"
            " ɪ v ɪ æ l ᵻ ɾ i z n ɑː t w ɜː θ θ ɪ ŋ k ɪ ŋ ɐ b aʊ t",
            "h ɪ z ɪ n s t ə n t v p æ n ɪ k w ʌ z f ɑː l oʊ d b aɪ ɐ s m ɔː l ʃ ɑːɹ p b l oʊ h aɪ ɔ n h ɪ z tʃ ɛ s t",
        ]
        # should correspond to =>:
        # [
        # "a man said to the universe sir i exist",
        # "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
        # "the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with the thousands of spectators were trivialities not worth thinking about",
        # "his instant panic was followed by a small sharp blow high on his chest",
        # ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    @require_pyctcdecode
    @require_torchaudio
    def test_wav2vec2_with_lm(self):
        ds = load_dataset(
            "mozilla-foundation/common_voice_11_0", "es", split="test", streaming=True, trust_remote_code=True
        )
        sample = next(iter(ds))

        resampled_audio = torchaudio.functional.resample(
            torch.tensor(sample["audio"]["array"]), 48_000, 16_000
        ).numpy()

        model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm").to(
            torch_device
        )
        processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm")

        input_values = processor(resampled_audio, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values.to(torch_device)).logits

        transcription = processor.batch_decode(logits.cpu().numpy()).text

        self.assertEqual(transcription[0], "habitan aguas poco profundas y rocosas")

    @require_pyctcdecode
    @require_torchaudio
    def test_wav2vec2_with_lm_pool(self):
        ds = load_dataset(
            "mozilla-foundation/common_voice_11_0", "es", split="test", streaming=True, trust_remote_code=True
        )
        sample = next(iter(ds))

        resampled_audio = torchaudio.functional.resample(
            torch.tensor(sample["audio"]["array"]), 48_000, 16_000
        ).numpy()

        model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm").to(
            torch_device
        )
        processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm")

        input_values = processor(resampled_audio, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values.to(torch_device)).logits

        # test user-managed pool
        with multiprocessing.get_context("fork").Pool(2) as pool:
            transcription = processor.batch_decode(logits.cpu().numpy(), pool).text

        self.assertEqual(transcription[0], "habitan aguas poco profundas y rocosas")

        # user-managed pool + num_processes should trigger a warning
        with CaptureLogger(processing_wav2vec2_with_lm.logger) as cl, multiprocessing.get_context("fork").Pool(
            2
        ) as pool:
            transcription = processor.batch_decode(logits.cpu().numpy(), pool, num_processes=2).text

        self.assertIn("num_process", cl.out)
        self.assertIn("it will be ignored", cl.out)

        self.assertEqual(transcription[0], "habitan aguas poco profundas y rocosas")

    @require_pyctcdecode
    @require_torchaudio
    def test_wav2vec2_with_lm_invalid_pool(self):
        run_test_in_subprocess(test_case=self, target_func=_test_wav2vec2_with_lm_invalid_pool, inputs=None)

    def test_inference_diarization(self):
        model = Wav2Vec2ForAudioFrameClassification.from_pretrained("anton-l/wav2vec2-base-superb-sd").to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sd")
        input_data = self._load_superb("sd", 4)
        inputs = processor(input_data["speech"], return_tensors="pt", padding=True, sampling_rate=16_000)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
        # labels is a one-hot array of shape (num_frames, num_speakers)
        labels = (outputs.logits > 0).long()

        # s3prl logits for the same batch
        expected_logits = torch.tensor(
            [
                [[-5.2807, -5.1272], [-5.4059, -4.7757], [-5.2764, -4.9621], [-5.0117, -4.5851]],
                [[-1.7643, -0.5462], [-1.7369, -0.2649], [-1.5066, -0.6200], [-4.5703, -2.4863]],
                [[-0.8656, -0.4783], [-0.8899, -0.3289], [-0.9267, -0.5781], [-0.7817, -0.4619]],
                [[-4.8625, -2.5316], [-5.2339, -2.2155], [-4.9835, -2.0344], [-4.4727, -1.8421]],
            ],
            device=torch_device,
        )
        self.assertEqual(labels[0, :, 0].sum(), 555)
        self.assertEqual(labels[0, :, 1].sum(), 299)
        torch.testing.assert_close(outputs.logits[:, :4], expected_logits, rtol=1e-2, atol=1e-2)

    def test_inference_speaker_verification(self):
        model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv").to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
        input_data = self._load_superb("si", 4)

        inputs = processor(input_data["speech"], return_tensors="pt", padding=True, sampling_rate=16_000)
        labels = torch.tensor([5, 1, 1, 3], device=torch_device).T

        with torch.no_grad():
            input_values = inputs.input_values.to(torch_device)
            attention_mask = inputs.attention_mask.to(torch_device)
            outputs = model(input_values, attention_mask=attention_mask, labels=labels)
        embeddings = torch.nn.functional.normalize(outputs.embeddings, dim=-1).cpu()

        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        # id10002 vs id10002
        self.assertAlmostEqual(cosine_sim(embeddings[1], embeddings[2]).numpy(), 0.9758, 3)
        # id10006 vs id10002
        self.assertAlmostEqual(cosine_sim(embeddings[0], embeddings[1]).numpy(), 0.7579, 3)
        # id10002 vs id10004
        self.assertAlmostEqual(cosine_sim(embeddings[2], embeddings[3]).numpy(), 0.7594, 3)

        self.assertAlmostEqual(outputs.loss.item(), 17.7963, 2)

    @require_torchaudio
    def test_inference_mms_1b_all(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")

        LANG_MAP = {"it": "ita", "es": "spa", "fr": "fra", "en": "eng"}

        def run_model(lang):
            ds = load_dataset(
                "mozilla-foundation/common_voice_11_0", lang, split="test", streaming=True, trust_remote_code=True
            )
            sample = next(iter(ds))

            wav2vec2_lang = LANG_MAP[lang]

            model.load_adapter(wav2vec2_lang)
            processor.tokenizer.set_target_lang(wav2vec2_lang)

            resampled_audio = torchaudio.functional.resample(
                torch.tensor(sample["audio"]["array"]), 48_000, 16_000
            ).numpy()

            inputs = processor(resampled_audio, sampling_rate=16_000, return_tensors="pt")
            input_values = inputs.input_values.to(torch_device)
            attention_mask = inputs.attention_mask.to(torch_device)

            with torch.no_grad():
                outputs = model(input_values, attention_mask=attention_mask).logits

            ids = torch.argmax(outputs, dim=-1)[0]

            transcription = processor.decode(ids)
            return transcription

        TRANSCRIPTIONS = {
            "it": "il libro ha suscitato molte polemiche a causa dei suoi contenuti",
            "es": "habitan aguas poco profundas y rocosas",
            "fr": "ce dernier est volé tout au long de l'histoire romaine",
            "en": "joe keton disapproved of films and buster also had reservations about the media",
        }

        for lang in LANG_MAP.keys():
            assert run_model(lang) == TRANSCRIPTIONS[lang]

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    def test_inference_ctc_fa2(self):
        model_fa = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
        )
        model_fa.to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)
        input_speech = self._load_datasamples(1)

        input_values = processor(input_speech, return_tensors="pt").input_values.to(torch_device)

        with torch.no_grad():
            logits = model_fa(input_values.to(torch.bfloat16)).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    def test_inference_ctc_fa2_batched(self):
        model_fa = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
        )
        model_fa.to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = inputs.to(torch_device)

        with torch.no_grad():
            logits = model_fa(inputs.input_values.to(torch.bfloat16), attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight lowing cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)
