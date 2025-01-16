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
)


if is_torch_available():
    import torch

    from transformers import XLMRobertaModel


@require_sentencepiece
@require_tokenizers
@require_torch
class XLMRobertaModelIntegrationTest(unittest.TestCase):
    @slow
    def test_xlm_roberta_base(self):
        model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base", attn_implementation="eager")
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        # The dog is cute and lives in the garden house

        expected_output_shape = torch.Size((1, 12, 768))  # batch_size, sequence_length, embedding_vector_dim
        expected_output_values_last_dim = torch.tensor(
            [[-0.0101, 0.1218, -0.0803, 0.0801, 0.1327, 0.0776, -0.1215, 0.2383, 0.3338, 0.3106, 0.0300, 0.0252]]
        )
        #  xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.base')
        #  xlmr.eval()
        #  expected_output_values_last_dim = xlmr.extract_features(input_ids[0])[:, :, -1]
        with torch.no_grad():
            output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        torch.testing.assert_close(output[:, :, -1], expected_output_values_last_dim, rtol=1e-3, atol=1e-3)

    @require_torch_sdpa
    def test_xlm_roberta_base_sdpa(self):
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        # The dog is cute and lives in the garden house

        expected_output_shape = torch.Size((1, 12, 768))  # batch_size, sequence_length, embedding_vector_dim
        expected_output_values_last_dim = torch.tensor(
            [[-0.0101, 0.1218, -0.0803, 0.0801, 0.1327, 0.0776, -0.1215, 0.2383, 0.3338, 0.3106, 0.0300, 0.0252]]
        )

        model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base", attn_implementation="sdpa")
        with torch.no_grad():
            output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        torch.testing.assert_close(output[:, :, -1], expected_output_values_last_dim, rtol=1e-3, atol=1e-3)

    @slow
    def test_xlm_roberta_large(self):
        model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-large")
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        # The dog is cute and lives in the garden house

        expected_output_shape = torch.Size((1, 12, 1024))  # batch_size, sequence_length, embedding_vector_dim
        expected_output_values_last_dim = torch.tensor(
            [[-0.0699, -0.0318, 0.0705, -0.1241, 0.0999, -0.0520, 0.1004, -0.1838, -0.4704, 0.1437, 0.0821, 0.0126]]
        )
        #  xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
        #  xlmr.eval()
        #  expected_output_values_last_dim = xlmr.extract_features(input_ids[0])[:, :, -1]
        with torch.no_grad():
            output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        torch.testing.assert_close(output[:, :, -1], expected_output_values_last_dim, rtol=1e-3, atol=1e-3)
