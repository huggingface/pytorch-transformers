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


import os
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        Adafactor,
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_inverse_sqrt_schedule,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
        get_wsd_schedule,
    )


def unwrap_schedule(scheduler, num_steps=10):
    lrs = []
    for _ in range(num_steps):
        lrs.append(scheduler.get_lr()[0])
        scheduler.step()
    return lrs


def unwrap_and_save_reload_schedule(scheduler, num_steps=10):
    lrs = []
    for step in range(num_steps):
        lrs.append(scheduler.get_lr()[0])
        scheduler.step()
        if step == num_steps // 2:
            with tempfile.TemporaryDirectory() as tmpdirname:
                file_name = os.path.join(tmpdirname, "schedule.bin")
                torch.save(scheduler.state_dict(), file_name)

                state_dict = torch.load(file_name)
                scheduler.load_state_dict(state_dict)
    return lrs


@require_torch
class OptimizationTest(unittest.TestCase):
    def assertListAlmostEqual(self, list1, list2, tol):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol)

    def test_adam_w(self):
        w = torch.tensor([0.1, -0.2, -0.1], requires_grad=True)
        target = torch.tensor([0.4, 0.2, -0.5])
        criterion = nn.MSELoss()
        # No warmup, constant schedule, no gradient clipping
        optimizer = AdamW(params=[w], lr=2e-1, weight_decay=0.0)
        for _ in range(100):
            loss = criterion(w, target)
            loss.backward()
            optimizer.step()
            w.grad.detach_()  # No zero_grad() function on simple tensors. we do it ourselves.
            w.grad.zero_()
        self.assertListAlmostEqual(w.tolist(), [0.4, 0.2, -0.5], tol=1e-2)

    def test_adafactor(self):
        w = torch.tensor([0.1, -0.2, -0.1], requires_grad=True)
        target = torch.tensor([0.4, 0.2, -0.5])
        criterion = nn.MSELoss()
        # No warmup, constant schedule, no gradient clipping
        optimizer = Adafactor(
            params=[w],
            lr=1e-2,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        for _ in range(1000):
            loss = criterion(w, target)
            loss.backward()
            optimizer.step()
            w.grad.detach_()  # No zero_grad() function on simple tensors. we do it ourselves.
            w.grad.zero_()
        self.assertListAlmostEqual(w.tolist(), [0.4, 0.2, -0.5], tol=1e-2)


@require_torch
class ScheduleInitTest(unittest.TestCase):
    m = nn.Linear(50, 50) if is_torch_available() else None
    optimizer = AdamW(m.parameters(), lr=10.0) if is_torch_available() else None
    num_steps = 10

    def assertListAlmostEqual(self, list1, list2, tol, msg=None):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol, msg=msg)

    def test_schedulers(self):
        common_kwargs = {"num_warmup_steps": 2, "num_training_steps": 10}
        # schedulers doct format
        # function: (sched_args_dict, expected_learning_rates)
        scheds = {
            get_constant_schedule: ({}, [10.0] * self.num_steps),
            get_constant_schedule_with_warmup: (
                {"num_warmup_steps": 4},
                [0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            ),
            get_linear_schedule_with_warmup: (
                {**common_kwargs},
                [0.0, 5.0, 10.0, 8.75, 7.5, 6.25, 5.0, 3.75, 2.5, 1.25],
            ),
            get_cosine_schedule_with_warmup: (
                {**common_kwargs},
                [0.0, 5.0, 10.0, 9.61, 8.53, 6.91, 5.0, 3.08, 1.46, 0.38],
            ),
            get_cosine_with_hard_restarts_schedule_with_warmup: (
                {**common_kwargs, "num_cycles": 2},
                [0.0, 5.0, 10.0, 8.53, 5.0, 1.46, 10.0, 8.53, 5.0, 1.46],
            ),
            get_polynomial_decay_schedule_with_warmup: (
                {**common_kwargs, "power": 2.0, "lr_end": 1e-7},
                [0.0, 5.0, 10.0, 7.656, 5.625, 3.906, 2.5, 1.406, 0.625, 0.156],
            ),
            get_inverse_sqrt_schedule: (
                {"num_warmup_steps": 2},
                [0.0, 5.0, 10.0, 8.165, 7.071, 6.325, 5.774, 5.345, 5.0, 4.714],
            ),
            get_wsd_schedule: (
                {**common_kwargs, "num_decay_steps": 2, "min_lr_ratio": 0.0},
                [0.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0],
            ),
        }

        for scheduler_func, data in scheds.items():
            kwargs, expected_learning_rates = data

            scheduler = scheduler_func(self.optimizer, **kwargs)
            self.assertEqual(len([scheduler.get_lr()[0]]), 1)
            lrs_1 = unwrap_schedule(scheduler, self.num_steps)
            self.assertListAlmostEqual(
                lrs_1,
                expected_learning_rates,
                tol=1e-2,
                msg=f"failed for {scheduler_func} in normal scheduler",
            )

            scheduler = scheduler_func(self.optimizer, **kwargs)
            if scheduler_func.__name__ != "get_constant_schedule":
                LambdaScheduleWrapper.wrap_scheduler(scheduler)  # wrap to test picklability of the schedule
            lrs_2 = unwrap_and_save_reload_schedule(scheduler, self.num_steps)
            self.assertListEqual(lrs_1, lrs_2, msg=f"failed for {scheduler_func} in save and reload")

    def test_get_scheduler(self):
        test_params = [
            {
                "name": "warmup_stable_decay",
                "optimizer": self.optimizer,
                "num_warmup_steps": 2,
                "num_training_steps": 10,
                "scheduler_specific_kwargs": {
                    "num_decay_steps": 2,
                    "warmup_type": "linear",
                    "decay_type": "linear",
                },
            },
            {
                "name": "warmup_stable_decay",
                "optimizer": self.optimizer,
                "num_warmup_steps": 2,
                "num_training_steps": 10,
                "scheduler_specific_kwargs": {
                    "num_decay_steps": 2,
                    "warmup_type": "cosine",
                    "decay_type": "cosine",
                },
            },
            {
                "name": "warmup_stable_decay",
                "optimizer": self.optimizer,
                "num_warmup_steps": 2,
                "num_training_steps": 10,
                "scheduler_specific_kwargs": {
                    "num_decay_steps": 2,
                    "warmup_type": "1-sqrt",
                    "decay_type": "1-sqrt",
                },
            },
            {"name": "cosine", "optimizer": self.optimizer, "num_warmup_steps": 2, "num_training_steps": 10},
        ]

        for param in test_params:
            self.assertTrue(get_scheduler(**param), msg=f"failed for {param['name']} in get_scheduler")


class LambdaScheduleWrapper:
    """See https://github.com/huggingface/transformers/issues/21689"""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    @classmethod
    def wrap_scheduler(cls, scheduler):
        scheduler.lr_lambdas = list(map(cls, scheduler.lr_lambdas))
