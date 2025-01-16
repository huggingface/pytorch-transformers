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
"""Testing suite for the PyTorch LayoutLMv2 model."""

import unittest

from transformers.testing_utils import (
    require_detectron2,
    require_non_xpu,
    require_torch,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_detectron2_available, is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import (
        LayoutLMv2Config,
        LayoutLMv2ForQuestionAnswering,
        LayoutLMv2ForSequenceClassification,
        LayoutLMv2ForTokenClassification,
        LayoutLMv2Model,
    )

if is_detectron2_available():
    from detectron2.structures.image_list import ImageList


class LayoutLMv2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=4,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=36,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        image_feature_pool_shape=[7, 7, 256],
        coordinate_size=6,
        shape_size=6,
        num_labels=3,
        num_choices=4,
        scope=None,
        range_bbox=1000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.image_feature_pool_shape = image_feature_pool_shape
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.range_bbox = range_bbox

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        bbox = ids_tensor([self.batch_size, self.seq_length, 4], self.range_bbox)
        # Ensure that bbox is legal
        for i in range(bbox.shape[0]):
            for j in range(bbox.shape[1]):
                if bbox[i, j, 3] < bbox[i, j, 1]:
                    t = bbox[i, j, 3]
                    bbox[i, j, 3] = bbox[i, j, 1]
                    bbox[i, j, 1] = t
                if bbox[i, j, 2] < bbox[i, j, 0]:
                    t = bbox[i, j, 2]
                    bbox[i, j, 2] = bbox[i, j, 0]
                    bbox[i, j, 0] = t

        image = ImageList(
            torch.zeros(self.batch_size, self.num_channels, self.image_size, self.image_size, device=torch_device),
            self.image_size,
        )

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = LayoutLMv2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            image_feature_pool_shape=self.image_feature_pool_shape,
            coordinate_size=self.coordinate_size,
            shape_size=self.shape_size,
        )

        # use smaller resnet backbone to make tests faster
        config.detectron2_config_args["MODEL.RESNETS.DEPTH"] = 18
        config.detectron2_config_args["MODEL.RESNETS.RES2_OUT_CHANNELS"] = 64
        config.detectron2_config_args["MODEL.RESNETS.NUM_GROUPS"] = 1

        return config, input_ids, bbox, image, token_type_ids, input_mask, sequence_labels, token_labels

    def create_and_check_model(
        self, config, input_ids, bbox, image, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        model = LayoutLMv2Model(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, bbox=bbox, image=image, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, bbox=bbox, image=image, token_type_ids=token_type_ids)
        result = model(input_ids, bbox=bbox, image=image)

        # LayoutLMv2 has a different expected sequence length, namely also visual tokens are added
        expected_seq_len = self.seq_length + self.image_feature_pool_shape[0] * self.image_feature_pool_shape[1]
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, expected_seq_len, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, bbox, image, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = LayoutLMv2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            image=image,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, bbox, image, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = LayoutLMv2ForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            image=image,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self, config, input_ids, bbox, image, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        model = LayoutLMv2ForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            image=image,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            bbox,
            image,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "image": image,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_non_xpu
@require_torch
@require_detectron2
class LayoutLMv2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_pruning = False
    test_torchscript = True
    test_mismatched_shapes = False

    all_model_classes = (
        (
            LayoutLMv2Model,
            LayoutLMv2ForSequenceClassification,
            LayoutLMv2ForTokenClassification,
            LayoutLMv2ForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"document-question-answering": LayoutLMv2ForQuestionAnswering, "feature-extraction": LayoutLMv2Model}
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = LayoutLMv2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LayoutLMv2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @require_torch_multi_gpu
    @unittest.skip(
        reason=(
            "LayoutLMV2 and its dependency `detectron2` have some layers using `add_module` which doesn't work well"
            " with `nn.DataParallel`"
        )
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        # LayoutLMv2 has a different expected sequence length
        expected_seq_len = (
            self.model_tester.seq_length
            + self.model_tester.image_feature_pool_shape[0] * self.model_tester.image_feature_pool_shape[1]
        )

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, expected_seq_len, expected_seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, expected_seq_len, expected_seq_len],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            # LayoutLMv2 has a different expected sequence length
            expected_seq_len = (
                self.model_tester.seq_length
                + self.model_tester.image_feature_pool_shape[0] * self.model_tester.image_feature_pool_shape[1]
            )

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [expected_seq_len, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="We cannot configure detectron2 to output a smaller backbone")
    def test_model_is_small(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "microsoft/layoutlmv2-base-uncased"
        model = LayoutLMv2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "backbone" in name or "visual_segment_embedding" in name:
                    continue

                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_batching_equivalence(self):
        def equivalence(tensor1, tensor2):
            return 1.0 - F.cosine_similarity(tensor1.float().flatten(), tensor2.float().flatten(), dim=0, eps=0)

        def recursive_check(batched_object, single_row_object, model_name, key):
            if isinstance(batched_object, (list, tuple)):
                for batched_object_value, single_row_object_value in zip(batched_object, single_row_object):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            elif batched_object is None:
                return
            else:
                batched_row = batched_object[:1]
                self.assertFalse(
                    torch.isnan(batched_row).any(), f"Batched output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(batched_row).any(), f"Batched output has `inf` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isnan(single_row_object).any(), f"Single row output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(single_row_object).any(), f"Single row output has `inf` in {model_name} for key={key}"
                )
                self.assertTrue(
                    (equivalence(batched_row, single_row_object)) <= 1e-03,
                    msg=(
                        f"Batched and Single row outputs are not equal in {model_name} for key={key}. "
                        f"Difference={equivalence(batched_row, single_row_object)}."
                    ),
                )

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(config).to(torch_device).eval()
            batch_size = self.model_tester.batch_size

            single_row_input = {}
            for key, value in batched_input_prepared.items():
                if isinstance(value, torch.Tensor) and value.shape[0] % batch_size == 0:
                    single_batch_shape = value.shape[0] // batch_size
                    single_row_input[key] = value[:single_batch_shape]
                elif hasattr(value, "tensor"):
                    # layoutlmv2uses ImageList intead of pixel values (needs for torchscript)
                    single_row_input[key] = value.tensor[:single_batch_shape]

            with torch.no_grad():
                model_batched_output = model(**batched_input_prepared)
                model_row_output = model(**single_row_input)

            for key in model_batched_output:
                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)


def prepare_layoutlmv2_batch_inputs():
    # Here we prepare a batch of 2 sequences to test a LayoutLMv2 forward pass on:
    # fmt: off
    input_ids = torch.tensor([[101,1019,1014,1016,1037,12849,4747,1004,14246,2278,5439,4524,5002,2930,2193,2930,4341,3208,1005,1055,2171,2848,11300,3531,102],[101,4070,4034,7020,1024,3058,1015,1013,2861,1013,6070,19274,2772,6205,27814,16147,16147,4343,2047,10283,10969,14389,1012,2338,102]])  # noqa: E231
    bbox = torch.tensor([[[0,0,0,0],[423,237,440,251],[427,272,441,287],[419,115,437,129],[961,885,992,912],[256,38,330,58],[256,38,330,58],[336,42,353,57],[360,39,401,56],[360,39,401,56],[411,39,471,59],[479,41,528,59],[533,39,630,60],[67,113,134,131],[141,115,209,132],[68,149,133,166],[141,149,187,164],[195,148,287,165],[195,148,287,165],[195,148,287,165],[295,148,349,165],[441,149,492,166],[497,149,546,164],[64,201,125,218],[1000,1000,1000,1000]],[[0,0,0,0],[662,150,754,166],[665,199,742,211],[519,213,554,228],[519,213,554,228],[134,433,187,454],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[314,469,376,482],[504,684,582,706],[941,825,973,900],[941,825,973,900],[941,825,973,900],[941,825,973,900],[610,749,652,765],[130,659,168,672],[176,657,237,672],[238,657,312,672],[443,653,628,672],[443,653,628,672],[716,301,825,317],[1000,1000,1000,1000]]])  # noqa: E231
    image = ImageList(torch.randn((2,3,224,224)), image_sizes=[(224,224), (224,224)])  # noqa: E231
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],])  # noqa: E231
    token_type_ids = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])  # noqa: E231
    # fmt: on

    return input_ids, bbox, image, attention_mask, token_type_ids


@require_torch
@require_detectron2
class LayoutLMv2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased").to(torch_device)

        (
            input_ids,
            bbox,
            image,
            attention_mask,
            token_type_ids,
        ) = prepare_layoutlmv2_batch_inputs()

        # forward pass
        outputs = model(
            input_ids=input_ids.to(torch_device),
            bbox=bbox.to(torch_device),
            image=image.to(torch_device),
            attention_mask=attention_mask.to(torch_device),
            token_type_ids=token_type_ids.to(torch_device),
        )

        # verify the sequence output
        expected_shape = torch.Size(
            (
                2,
                input_ids.shape[1]
                + model.config.image_feature_pool_shape[0] * model.config.image_feature_pool_shape[1],
                model.config.hidden_size,
            )
        )
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.1087, 0.0727, -0.3075], [0.0799, -0.0427, -0.0751], [-0.0367, 0.0480, -0.1358]], device=torch_device
        )
        torch.testing.assert_close(outputs.last_hidden_state[0, :3, :3], expected_slice, rtol=1e-3, atol=1e-3)

        # verify the pooled output
        expected_shape = torch.Size((2, model.config.hidden_size))
        self.assertEqual(outputs.pooler_output.shape, expected_shape)
