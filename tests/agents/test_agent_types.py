# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from transformers.agents.agent_types import AgentAudio, AgentImage, AgentText
from transformers.testing_utils import get_tests_dir, require_soundfile, require_torch, require_vision
from transformers.utils import is_soundfile_available, is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf

if is_vision_available():
    from PIL import Image


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


@require_soundfile
@require_torch
class AgentAudioTests(unittest.TestCase):
    def test_from_tensor(self):
        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        agent_type = AgentAudio(tensor)
        path = str(agent_type.to_string())

        # Ensure that the tensor and the agent_type's tensor are the same
        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=1e-4))

        del agent_type

        # Ensure the path remains even after the object deletion
        self.assertTrue(os.path.exists(path))

        # Ensure that the file contains the same value as the original tensor
        new_tensor, _ = sf.read(path)
        self.assertTrue(torch.allclose(tensor, torch.tensor(new_tensor), atol=1e-4))

    def test_from_string(self):
        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        path = get_new_path(suffix=".wav")
        sf.write(path, tensor, 16000)

        agent_type = AgentAudio(path)

        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=1e-4))
        self.assertEqual(agent_type.to_string(), path)


@require_vision
@require_torch
class AgentImageTests(unittest.TestCase):
    def test_from_tensor(self):
        tensor = torch.randint(0, 256, (64, 64, 3))
        agent_type = AgentImage(tensor)
        path = str(agent_type.to_string())

        # Ensure that the tensor and the agent_type's tensor are the same
        self.assertTrue(torch.allclose(tensor, agent_type._tensor, atol=1e-4))

        self.assertIsInstance(agent_type.to_raw(), Image.Image)

        # Ensure the path remains even after the object deletion
        del agent_type
        self.assertTrue(os.path.exists(path))

    def test_from_string(self):
        path = Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png"
        image = Image.open(path)
        agent_type = AgentImage(path)

        self.assertTrue(path.samefile(agent_type.to_string()))
        self.assertTrue(image == agent_type.to_raw())

        # Ensure the path remains even after the object deletion
        del agent_type
        self.assertTrue(os.path.exists(path))

    def test_from_image(self):
        path = Path(get_tests_dir("fixtures/tests_samples/COCO")) / "000000039769.png"
        image = Image.open(path)
        agent_type = AgentImage(image)

        self.assertFalse(path.samefile(agent_type.to_string()))
        self.assertTrue(image == agent_type.to_raw())

        # Ensure the path remains even after the object deletion
        del agent_type
        self.assertTrue(os.path.exists(path))


class AgentTextTests(unittest.TestCase):
    def test_from_string(self):
        string = "Hey!"
        agent_type = AgentText(string)

        self.assertEqual(string, agent_type.to_string())
        self.assertEqual(string, agent_type.to_raw())
        self.assertEqual(string, agent_type)
