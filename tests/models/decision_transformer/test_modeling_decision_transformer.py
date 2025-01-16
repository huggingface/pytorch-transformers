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
"""Testing suite for the PyTorch DecisionTransformer model."""

import inspect
import unittest

from transformers import DecisionTransformerConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import DecisionTransformerModel


class DecisionTransformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        act_dim=6,
        state_dim=17,
        hidden_size=23,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        states = floats_tensor((self.batch_size, self.seq_length, self.state_dim))
        actions = floats_tensor((self.batch_size, self.seq_length, self.act_dim))
        rewards = floats_tensor((self.batch_size, self.seq_length, 1))
        returns_to_go = floats_tensor((self.batch_size, self.seq_length, 1))
        timesteps = ids_tensor((self.batch_size, self.seq_length), vocab_size=1000)
        attention_mask = random_attention_mask((self.batch_size, self.seq_length))

        config = self.get_config()

        return (
            config,
            states,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            attention_mask,
        )

    def get_config(self):
        return DecisionTransformerConfig(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            hidden_size=self.hidden_size,
        )

    def create_and_check_model(
        self,
        config,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask,
    ):
        model = DecisionTransformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(states, actions, rewards, returns_to_go, timesteps, attention_mask)

        self.parent.assertEqual(result.state_preds.shape, states.shape)
        self.parent.assertEqual(result.action_preds.shape, actions.shape)
        self.parent.assertEqual(result.return_preds.shape, returns_to_go.shape)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.seq_length * 3, self.hidden_size)
        )  # seq length *3 as there are 3 modelities: states, returns and actions

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            states,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            attention_mask,
        ) = config_and_inputs
        inputs_dict = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class DecisionTransformerModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DecisionTransformerModel,) if is_torch_available() else ()
    all_generative_model_classes = ()
    pipeline_model_mapping = {"feature-extraction": DecisionTransformerModel} if is_torch_available() else {}

    # Ignoring of a failing test from GenerationTesterMixin, as the model does not use inputs_ids
    test_generate_without_input_ids = False

    # Ignoring of a failing tests from ModelTesterMixin, as the model does not implement these features
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_attention_outputs = False
    test_hidden_states_output = False
    test_inputs_embeds = False
    test_gradient_checkpointing = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = DecisionTransformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DecisionTransformerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "edbeeching/decision-transformer-gym-hopper-medium"
        model = DecisionTransformerModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "states",
                "actions",
                "rewards",
                "returns_to_go",
                "timesteps",
                "attention_mask",
            ]

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass


@require_torch
class DecisionTransformerModelIntegrationTest(unittest.TestCase):
    @slow
    def test_autoregressive_prediction(self):
        """
        An integration test that performs autoregressive prediction of state, action and return
        from a sequence of state, actions and returns. Test is performed over two timesteps.

        """

        NUM_STEPS = 2  # number of steps of autoregressive prediction we will perform
        TARGET_RETURN = 10  # defined by the RL environment, may be normalized
        model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-expert")
        model = model.to(torch_device)
        config = model.config
        torch.manual_seed(0)
        state = torch.randn(1, 1, config.state_dim).to(device=torch_device, dtype=torch.float32)  # env.reset()

        expected_outputs = torch.tensor(
            [[0.242793, -0.28693074, 0.8742613], [0.67815274, -0.08101085, -0.12952147]], device=torch_device
        )

        returns_to_go = torch.tensor(TARGET_RETURN, device=torch_device, dtype=torch.float32).reshape(1, 1, 1)
        states = state
        actions = torch.zeros(1, 0, config.act_dim, device=torch_device, dtype=torch.float32)
        rewards = torch.zeros(1, 0, device=torch_device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=torch_device, dtype=torch.long).reshape(1, 1)

        for step in range(NUM_STEPS):
            actions = torch.cat([actions, torch.zeros(1, 1, config.act_dim, device=torch_device)], dim=1)
            rewards = torch.cat([rewards, torch.zeros(1, 1, device=torch_device)], dim=1)

            attention_mask = torch.ones(1, states.shape[1]).to(dtype=torch.long, device=states.device)

            with torch.no_grad():
                _, action_pred, _ = model(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    return_dict=False,
                )

            self.assertEqual(action_pred.shape, actions.shape)
            torch.testing.assert_close(action_pred[0, -1], expected_outputs[step], rtol=1e-4, atol=1e-4)
            state, reward, _, _ = (  # env.step(action)
                torch.randn(1, 1, config.state_dim).to(device=torch_device, dtype=torch.float32),
                1.0,
                False,
                {},
            )

            actions[-1] = action_pred[0, -1]
            states = torch.cat([states, state], dim=1)
            pred_return = returns_to_go[0, -1] - reward
            returns_to_go = torch.cat([returns_to_go, pred_return.reshape(1, 1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps, torch.ones((1, 1), device=torch_device, dtype=torch.long) * (step + 1)], dim=1
            )
