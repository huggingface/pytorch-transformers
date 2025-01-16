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
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.activations import gelu_new, gelu_python, get_activation


@require_torch
class TestActivations(unittest.TestCase):
    def test_gelu_versions(self):
        x = torch.tensor([-100, -1, -0.1, 0, 0.1, 1.0, 100])
        torch_builtin = get_activation("gelu")
        torch.testing.assert_close(gelu_python(x), torch_builtin(x))
        self.assertFalse(torch.allclose(gelu_python(x), gelu_new(x)))

    def test_gelu_10(self):
        x = torch.tensor([-100, -1, -0.1, 0, 0.1, 1.0, 100])
        torch_builtin = get_activation("gelu")
        gelu10 = get_activation("gelu_10")

        y_gelu = torch_builtin(x)
        y_gelu_10 = gelu10(x)

        clipped_mask = torch.where(y_gelu_10 < 10.0, 1, 0)

        self.assertTrue(torch.max(y_gelu_10).item() == 10.0)
        torch.testing.assert_close(y_gelu * clipped_mask, y_gelu_10 * clipped_mask)

    def test_get_activation(self):
        get_activation("gelu")
        get_activation("gelu_10")
        get_activation("gelu_fast")
        get_activation("gelu_new")
        get_activation("gelu_python")
        get_activation("gelu_pytorch_tanh")
        get_activation("linear")
        get_activation("mish")
        get_activation("quick_gelu")
        get_activation("relu")
        get_activation("sigmoid")
        get_activation("silu")
        get_activation("swish")
        get_activation("tanh")
        with self.assertRaises(KeyError):
            get_activation("bogus")
        with self.assertRaises(KeyError):
            get_activation(None)

    def test_activations_are_distinct_objects(self):
        act1 = get_activation("gelu")
        act1.a = 1
        act2 = get_activation("gelu")
        self.assertEqual(act1.a, 1)
        with self.assertRaises(AttributeError):
            _ = act2.a
