# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import CamembertModel


@require_torch
@require_sentencepiece
@require_tokenizers
class CamembertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_output_embeds_base_model(self):
        model = CamembertModel.from_pretrained("almanach/camembert-base", attn_implementation="eager")
        model.to(torch_device)

        input_ids = torch.tensor(
            [[5, 121, 11, 660, 16, 730, 25543, 110, 83, 6]],
            device=torch_device,
            dtype=torch.long,
        )  # J'aime le camembert !
        with torch.no_grad():
            output = model(input_ids)["last_hidden_state"]
        expected_shape = torch.Size((1, 10, 768))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[-0.0254, 0.0235, 0.1027], [0.0606, -0.1811, -0.0418], [-0.1561, -0.1127, 0.2687]]],
            device=torch_device,
            dtype=torch.float,
        )
        # camembert = torch.hub.load('pytorch/fairseq', 'camembert.v0')
        # camembert.eval()
        # expected_slice = roberta.model.forward(input_ids)[0][:, :3, :3].detach()

        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    @require_torch_sdpa
    def test_output_embeds_base_model_sdpa(self):
        input_ids = torch.tensor(
            [[5, 121, 11, 660, 16, 730, 25543, 110, 83, 6]],
            device=torch_device,
            dtype=torch.long,
        )  # J'aime le camembert !

        expected_slice = torch.tensor(
            [[[-0.0254, 0.0235, 0.1027], [0.0606, -0.1811, -0.0418], [-0.1561, -0.1127, 0.2687]]],
            device=torch_device,
            dtype=torch.float,
        )

        model = CamembertModel.from_pretrained("almanach/camembert-base", attn_implementation="sdpa").to(torch_device)
        with torch.no_grad():
            output = model(input_ids)["last_hidden_state"].detach()

        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
