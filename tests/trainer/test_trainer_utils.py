# coding=utf-8
# Copyright 2018 the HuggingFace Inc. team.
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

import copy
import unittest

import numpy as np

from transformers.data.data_collator import default_data_collator
from transformers.testing_utils import require_accelerate, require_torch
from transformers.trainer_utils import RemoveColumnsCollator, find_executable_batch_size
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import IterableDataset

    from transformers.modeling_outputs import SequenceClassifierOutput
    from transformers.tokenization_utils_base import BatchEncoding
    from transformers.trainer_pt_utils import (
        DistributedLengthGroupedSampler,
        DistributedSamplerWithLoop,
        DistributedTensorGatherer,
        EvalLoopContainer,
        IterableDatasetShard,
        LabelSmoother,
        LengthGroupedSampler,
        SequentialDistributedSampler,
        ShardSampler,
        get_parameter_names,
        numpy_pad_and_concatenate,
        torch_pad_and_concatenate,
    )

    class TstLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)
            self.bias = nn.Parameter(torch.zeros(hidden_size))

        def forward(self, x):
            h = self.ln1(nn.functional.relu(self.linear1(x)))
            h = nn.functional.relu(self.linear2(x))
            return self.ln2(x + h + self.bias)

    class RandomIterableDataset(IterableDataset):
        # For testing, an iterable dataset of random length
        def __init__(self, p_stop=0.01, max_length=1000):
            self.p_stop = p_stop
            self.max_length = max_length
            self.generator = torch.Generator()

        def __iter__(self):
            count = 0
            stop = False
            while not stop and count < self.max_length:
                yield count
                count += 1
                number = torch.rand(1, generator=self.generator).item()
                stop = number < self.p_stop


@require_torch
class TrainerUtilsTest(unittest.TestCase):
    def test_distributed_tensor_gatherer(self):
        # Simulate a result with a dataset of size 21, 4 processes and chunks of lengths 2, 3, 1
        world_size = 4
        num_samples = 21
        input_indices = [
            [0, 1, 6, 7, 12, 13, 18, 19],
            [2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 0, 1],
            [5, 11, 17, 2],
        ]

        predictions = np.random.normal(size=(num_samples, 13))
        gatherer = DistributedTensorGatherer(world_size=world_size, num_samples=num_samples)
        for indices in input_indices:
            gatherer.add_arrays(predictions[indices])
        result = gatherer.finalize()
        self.assertTrue(np.array_equal(result, predictions))

        # With nested tensors
        gatherer = DistributedTensorGatherer(world_size=world_size, num_samples=num_samples)
        for indices in input_indices:
            gatherer.add_arrays([predictions[indices], [predictions[indices], predictions[indices]]])
        result = gatherer.finalize()
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result[1], list))
        self.assertEqual(len(result[1]), 2)
        self.assertTrue(np.array_equal(result[0], predictions))
        self.assertTrue(np.array_equal(result[1][0], predictions))
        self.assertTrue(np.array_equal(result[1][1], predictions))

    def test_distributed_tensor_gatherer_different_shapes(self):
        # Simulate a result with a dataset of size 21, 4 processes and chunks of lengths 2, 3, 1
        world_size = 4
        num_samples = 21
        input_indices = [
            [0, 1, 6, 7, 12, 13, 18, 19],
            [2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 0, 1],
            [5, 11, 17, 2],
        ]
        sequence_lengths = [8, 10, 13]

        predictions = np.random.normal(size=(num_samples, 13))
        gatherer = DistributedTensorGatherer(world_size=world_size, num_samples=num_samples)
        for indices, seq_length in zip(input_indices, sequence_lengths):
            gatherer.add_arrays(predictions[indices, :seq_length])
        result = gatherer.finalize()

        # Remove the extra samples added at the end for a round multiple of num processes.
        actual_indices = [input_indices[0], input_indices[1][:-2], input_indices[2][:-1]]
        for indices, seq_length in zip(actual_indices, sequence_lengths):
            self.assertTrue(np.array_equal(result[indices, :seq_length], predictions[indices, :seq_length]))

        # With nested tensors
        predictions = np.random.normal(size=(num_samples, 13))
        gatherer = DistributedTensorGatherer(world_size=world_size, num_samples=num_samples)
        for indices, seq_length in zip(input_indices, sequence_lengths):
            gatherer.add_arrays([predictions[indices, :seq_length], predictions[indices]])
        result = gatherer.finalize()

        for indices, seq_length in zip(actual_indices, sequence_lengths):
            self.assertTrue(np.array_equal(result[0][indices, :seq_length], predictions[indices, :seq_length]))
        self.assertTrue(np.array_equal(result[1], predictions))

        # Check if works if varying seq_length is second
        gatherer = DistributedTensorGatherer(world_size=world_size, num_samples=num_samples)
        for indices, seq_length in zip(input_indices, sequence_lengths):
            gatherer.add_arrays([predictions[indices], predictions[indices, :seq_length]])
        result = gatherer.finalize()

        self.assertTrue(np.array_equal(result[0], predictions))
        for indices, seq_length in zip(actual_indices, sequence_lengths):
            self.assertTrue(np.array_equal(result[1][indices, :seq_length], predictions[indices, :seq_length]))

    def test_label_smoothing(self):
        epsilon = 0.1
        num_labels = 12
        random_logits = torch.randn(4, 5, num_labels)
        random_labels = torch.randint(0, num_labels, (4, 5))
        loss = nn.functional.cross_entropy(random_logits.view(-1, num_labels), random_labels.view(-1))
        model_output = SequenceClassifierOutput(logits=random_logits)
        label_smoothed_loss = LabelSmoother(0.1)(model_output, random_labels)
        log_probs = -nn.functional.log_softmax(random_logits, dim=-1)
        expected_loss = (1 - epsilon) * loss + epsilon * log_probs.mean()
        torch.testing.assert_close(label_smoothed_loss, expected_loss)

        # With a few -100 labels
        random_labels[0, 1] = -100
        random_labels[2, 1] = -100
        random_labels[2, 3] = -100

        loss = nn.functional.cross_entropy(random_logits.view(-1, num_labels), random_labels.view(-1))
        model_output = SequenceClassifierOutput(logits=random_logits)
        label_smoothed_loss = LabelSmoother(0.1)(model_output, random_labels)
        log_probs = -nn.functional.log_softmax(random_logits, dim=-1)
        # Mask the log probs with the -100 labels
        log_probs[0, 1] = 0.0
        log_probs[2, 1] = 0.0
        log_probs[2, 3] = 0.0
        expected_loss = (1 - epsilon) * loss + epsilon * log_probs.sum() / (num_labels * 17)
        torch.testing.assert_close(label_smoothed_loss, expected_loss)

    def test_group_by_length(self):
        # Get some inputs of random lengths
        lengths = torch.randint(0, 25, (100,)).tolist()
        # Put one bigger than the others to check it ends up in first position
        lengths[32] = 50

        indices = list(LengthGroupedSampler(4, lengths=lengths))
        # The biggest element should be first
        self.assertEqual(lengths[indices[0]], 50)
        # The indices should be a permutation of range(100)
        self.assertEqual(sorted(indices), list(range(100)))

    def test_group_by_length_with_dict(self):
        # Get some inputs of random lengths
        data = []
        for _ in range(6):
            input_ids = torch.randint(0, 25, (100,)).tolist()
            data.append({"input_ids": input_ids})
        # Put one bigger than the others to check it ends up in first position
        data[3]["input_ids"] = torch.randint(0, 25, (105,)).tolist()

        indices = list(LengthGroupedSampler(4, dataset=data))
        # The biggest element should be first
        self.assertEqual(len(data[indices[0]]["input_ids"]), 105)
        # The indices should be a permutation of range(6)
        self.assertEqual(sorted(indices), list(range(6)))

    def test_group_by_length_with_batch_encoding(self):
        # Get some inputs of random lengths
        data = []
        for _ in range(6):
            input_ids = torch.randint(0, 25, (100,)).tolist()
            data.append(BatchEncoding({"input_ids": input_ids}))
        # Put one bigger than the others to check it ends up in first position
        data[3]["input_ids"] = torch.randint(0, 25, (105,)).tolist()

        indices = list(LengthGroupedSampler(4, dataset=data))
        # The biggest element should be first
        self.assertEqual(len(data[indices[0]]["input_ids"]), 105)
        # The indices should be a permutation of range(6)
        self.assertEqual(sorted(indices), list(range(6)))

    def test_distributed_length_grouped(self):
        # Get some inputs of random lengths
        lengths = torch.randint(0, 25, (100,)).tolist()
        # Put one bigger than the others to check it ends up in first position
        lengths[32] = 50

        indices_process_0 = list(DistributedLengthGroupedSampler(4, num_replicas=2, rank=0, lengths=lengths))
        indices_process_1 = list(DistributedLengthGroupedSampler(4, num_replicas=2, rank=1, lengths=lengths))
        # The biggest element should be first
        self.assertEqual(lengths[indices_process_0[0]], 50)
        # The indices should be a permutation of range(100)
        self.assertEqual(sorted(indices_process_0 + indices_process_1), list(range(100)))

    def test_get_parameter_names(self):
        model = nn.Sequential(TstLayer(128), nn.ModuleList([TstLayer(128), TstLayer(128)]))
        # fmt: off
        self.assertEqual(
            get_parameter_names(model, [nn.LayerNorm]),
            ['0.linear1.weight', '0.linear1.bias', '0.linear2.weight', '0.linear2.bias', '0.bias', '1.0.linear1.weight', '1.0.linear1.bias', '1.0.linear2.weight', '1.0.linear2.bias', '1.0.bias', '1.1.linear1.weight', '1.1.linear1.bias', '1.1.linear2.weight', '1.1.linear2.bias', '1.1.bias']
        )
        # fmt: on

    def test_distributed_sampler_with_loop(self):
        batch_size = 16
        for length in [23, 64, 123]:
            dataset = list(range(length))
            shard1 = DistributedSamplerWithLoop(dataset, batch_size, num_replicas=2, rank=0)
            shard2 = DistributedSamplerWithLoop(dataset, batch_size, num_replicas=2, rank=1)

            # Set seeds
            shard1.set_epoch(0)
            shard2.set_epoch(0)

            # Sample
            samples1 = list(shard1)
            samples2 = list(shard2)

            self.assertTrue(len(samples1) % batch_size == 0)
            self.assertTrue(len(samples2) % batch_size == 0)

            total = []
            for sample1, sample2 in zip(samples1, samples2):
                total += [sample1, sample2]

            self.assertEqual(set(total[:length]), set(dataset))
            self.assertEqual(set(total[length:]), set(total[: (len(total) - length)]))

    def test_sequential_distributed_sampler(self):
        batch_size = 16
        for length in [23, 64, 123]:
            dataset = list(range(length))
            shard1 = SequentialDistributedSampler(dataset, num_replicas=2, rank=0)
            shard2 = SequentialDistributedSampler(dataset, num_replicas=2, rank=1)

            # Sample
            samples1 = list(shard1)
            samples2 = list(shard2)

            total = samples1 + samples2

            self.assertListEqual(total[:length], dataset)
            self.assertListEqual(total[length:], dataset[: (len(total) - length)])

            # With a batch_size passed
            shard1 = SequentialDistributedSampler(dataset, num_replicas=2, rank=0, batch_size=batch_size)
            shard2 = SequentialDistributedSampler(dataset, num_replicas=2, rank=1, batch_size=batch_size)

            # Sample
            samples1 = list(shard1)
            samples2 = list(shard2)

            self.assertTrue(len(samples1) % batch_size == 0)
            self.assertTrue(len(samples2) % batch_size == 0)

            total = samples1 + samples2

            self.assertListEqual(total[:length], dataset)
            self.assertListEqual(total[length:], dataset[: (len(total) - length)])

    def check_iterable_dataset_shard(self, dataset, batch_size, drop_last, num_processes=2, epoch=0):
        # Set the seed for the base dataset to get the proper reference.
        dataset.generator.manual_seed(epoch)
        reference = list(dataset)

        shards = [
            IterableDatasetShard(
                dataset, batch_size=batch_size, drop_last=drop_last, num_processes=num_processes, process_index=i
            )
            for i in range(num_processes)
        ]
        for shard in shards:
            shard.set_epoch(epoch)
        shard_lists = [list(shard) for shard in shards]

        for shard in shard_lists:
            # All shards have a number of samples that is a round multiple of batch size
            self.assertTrue(len(shard) % batch_size == 0)
            # All shards have the same number of samples
            self.assertEqual(len(shard), len(shard_lists[0]))

        for shard in shards:
            # All shards know the total number of samples
            self.assertEqual(shard.num_examples, len(reference))

        observed = []
        for idx in range(0, len(shard_lists[0]), batch_size):
            for shard in shard_lists:
                observed += shard[idx : idx + batch_size]

        # If drop_last is False we loop through samples at the beginning to have a size that is a round multiple of
        # batch_size
        if not drop_last:
            while len(reference) < len(observed):
                reference += reference
        self.assertListEqual(observed, reference[: len(observed)])

        # Check equivalence between IterableDataset and ShardSampler
        dataset.generator.manual_seed(epoch)
        reference = list(dataset)

        sampler_shards = [
            ShardSampler(
                reference, batch_size=batch_size, drop_last=drop_last, num_processes=num_processes, process_index=i
            )
            for i in range(num_processes)
        ]
        for shard, sampler_shard in zip(shard_lists, sampler_shards):
            self.assertListEqual(shard, list(sampler_shard))

    def test_iterable_dataset_shard(self):
        dataset = RandomIterableDataset()

        self.check_iterable_dataset_shard(dataset, 4, drop_last=True, num_processes=2, epoch=0)
        self.check_iterable_dataset_shard(dataset, 4, drop_last=False, num_processes=2, epoch=0)

        self.check_iterable_dataset_shard(dataset, 4, drop_last=True, num_processes=3, epoch=42)
        self.check_iterable_dataset_shard(dataset, 4, drop_last=False, num_processes=3, epoch=42)

    def test_iterable_dataset_shard_with_length(self):
        sampler_shards = [
            IterableDatasetShard(list(range(100)), batch_size=4, drop_last=True, num_processes=2, process_index=i)
            for i in range(2)
        ]

        # Build expected shards: each process will have batches of size 4 until there is not enough elements to
        # form two full batches (so we stop at 96 = (100 // (4 * 2)) * 4)
        expected_shards = [[], []]
        current_shard = 0
        for i in range(0, 96, 4):
            expected_shards[current_shard].extend(list(range(i, i + 4)))
            current_shard = 1 - current_shard

        self.assertListEqual([list(shard) for shard in sampler_shards], expected_shards)
        self.assertListEqual([len(shard) for shard in sampler_shards], [len(shard) for shard in expected_shards])

        sampler_shards = [
            IterableDatasetShard(list(range(100)), batch_size=4, drop_last=False, num_processes=2, process_index=i)
            for i in range(2)
        ]
        # When drop_last=False, we get two last full batches by looping back to the beginning.
        expected_shards[0].extend(list(range(96, 100)))
        expected_shards[1].extend(list(range(0, 4)))

        self.assertListEqual([list(shard) for shard in sampler_shards], expected_shards)
        self.assertListEqual([len(shard) for shard in sampler_shards], [len(shard) for shard in expected_shards])

    def check_shard_sampler(self, dataset, batch_size, drop_last, num_processes=2):
        shards = [
            ShardSampler(
                dataset, batch_size=batch_size, drop_last=drop_last, num_processes=num_processes, process_index=i
            )
            for i in range(num_processes)
        ]
        shard_lists = [list(shard) for shard in shards]

        for shard in shard_lists:
            # All shards have a number of samples that is a round multiple of batch size
            self.assertTrue(len(shard) % batch_size == 0)
            # All shards have the same number of samples
            self.assertEqual(len(shard), len(shard_lists[0]))

        observed = []
        for idx in range(0, len(shard_lists[0]), batch_size):
            for shard in shard_lists:
                observed += shard[idx : idx + batch_size]

        # If drop_last is False we loop through samples at the beginning to have a size that is a round multiple of
        # batch_size
        reference = copy.copy(dataset)
        if not drop_last:
            while len(reference) < len(observed):
                reference += reference
        self.assertListEqual(observed, reference[: len(observed)])

    def test_shard_sampler(self):
        for n_elements in [64, 123]:
            dataset = list(range(n_elements))

            self.check_shard_sampler(dataset, 4, drop_last=True, num_processes=2)
            self.check_shard_sampler(dataset, 4, drop_last=False, num_processes=2)

            self.check_shard_sampler(dataset, 4, drop_last=True, num_processes=3)
            self.check_shard_sampler(dataset, 4, drop_last=False, num_processes=3)

    @require_accelerate
    def test_executable_batch_size(self):
        batch_sizes = []

        @find_executable_batch_size(starting_batch_size=64, auto_find_batch_size=True)
        def mock_training_loop_function(batch_size):
            nonlocal batch_sizes
            batch_sizes.append(batch_size)
            if batch_size > 16:
                raise RuntimeError("CUDA out of memory.")

        mock_training_loop_function()
        self.assertEqual(batch_sizes, [64, 32, 16])

    @require_accelerate
    def test_executable_batch_size_no_search(self):
        batch_sizes = []

        @find_executable_batch_size(starting_batch_size=64, auto_find_batch_size=False)
        def mock_training_loop_function(batch_size):
            nonlocal batch_sizes
            batch_sizes.append(batch_size)

        mock_training_loop_function()
        self.assertEqual(batch_sizes, [64])

    @require_accelerate
    def test_executable_batch_size_with_error(self):
        @find_executable_batch_size(starting_batch_size=64, auto_find_batch_size=False)
        def mock_training_loop_function(batch_size):
            raise RuntimeError("CUDA out of memory.")

        with self.assertRaises(RuntimeError) as cm:
            mock_training_loop_function()
            self.assertEqual("CUDA out of memory", cm.args[0])

    def test_pad_and_concatenate_with_1d(self):
        """Tests whether pad_and_concatenate works with scalars."""
        array1 = 1.0
        array2 = 2.0
        result = numpy_pad_and_concatenate(array1, array2)
        self.assertTrue(np.array_equal(np.array([1.0, 2.0]), result))

        tensor1 = torch.tensor(1.0)
        tensor2 = torch.tensor(2.0)
        result = torch_pad_and_concatenate(tensor1, tensor2)
        self.assertTrue(torch.equal(result, torch.Tensor([1.0, 2.0])))

    def test_remove_columns_collator(self):
        class MockLogger:
            def __init__(self) -> None:
                self.called = 0

            def info(self, msg):
                self.called += 1
                self.last_msg = msg

        data_batch = [
            {"col1": 1, "col2": 2, "col3": 3},
            {"col1": 1, "col2": 2, "col3": 3},
        ]
        logger = MockLogger()
        remove_columns_collator = RemoveColumnsCollator(
            default_data_collator, ["col1", "col2"], logger, "model", "training"
        )

        self.assertNotIn("col3", remove_columns_collator(data_batch))
        # check that the logging message is printed out only once
        remove_columns_collator(data_batch)
        remove_columns_collator(data_batch)
        self.assertEqual(logger.called, 1)
        self.assertIn("col3", logger.last_msg)

    def test_eval_loop_container(self):
        batch_1 = [
            torch.ones([8, 5]),
            {"loss": torch.tensor(1.0)},
            (torch.ones([8, 2, 3]), torch.ones([8, 2])),
        ]
        batch_2 = [
            torch.ones([4, 5]),
            {"loss": torch.tensor(2.0)},
            (torch.ones([4, 2, 3]), torch.ones([4, 6])),
        ]

        concat_container = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        concat_container.add(batch_1)
        concat_container.add(batch_2)
        concat_container.to_cpu_and_numpy()
        arrays = concat_container.get_arrays()

        # Test two nested batches concatenation
        self.assertIsInstance(arrays, list)
        self.assertEqual(len(arrays), 3)
        self.assertIsInstance(arrays[0], np.ndarray)
        self.assertEqual(arrays[0].shape, (12, 5))
        self.assertIsInstance(arrays[1], dict)
        self.assertIsInstance(arrays[1]["loss"], np.ndarray)
        self.assertEqual(arrays[1]["loss"].shape, (2,))
        self.assertTrue(np.allclose(arrays[1]["loss"], np.array([1.0, 2.0])))
        self.assertIsInstance(arrays[2], tuple)
        self.assertEqual(len(arrays[2]), 2)
        self.assertEqual(arrays[2][0].shape, (12, 2, 3))
        self.assertEqual(arrays[2][1].shape, (12, 6))
        # check that first batch padded with padding index -100 after concatenation
        self.assertEqual(arrays[2][1][0][2], -100)

        # Test two batches with no concatenation
        list_container = EvalLoopContainer(do_nested_concat=False)
        list_container.add(batch_1)
        list_container.add(batch_2)
        list_container.to_cpu_and_numpy()
        arrays = list_container.get_arrays()

        self.assertEqual(len(arrays), 2)
        self.assertIsInstance(arrays, list)
        np_batch_1, np_batch_2 = arrays

        self.assertIsInstance(np_batch_1, list)
        self.assertEqual(len(np_batch_1), 3)
        self.assertIsInstance(np_batch_1[0], np.ndarray)
        self.assertIsInstance(np_batch_1[1], dict)
        self.assertIsInstance(np_batch_1[2], tuple)
        self.assertEqual(np_batch_1[0].shape, (8, 5))
        self.assertEqual(np_batch_1[1]["loss"].shape, ())
        self.assertEqual(np_batch_1[2][0].shape, (8, 2, 3))
        self.assertEqual(np_batch_1[2][1].shape, (8, 2))

        self.assertIsInstance(np_batch_2, list)
        self.assertEqual(len(np_batch_2), 3)
        self.assertIsInstance(np_batch_2[0], np.ndarray)
        self.assertIsInstance(np_batch_2[1], dict)
        self.assertIsInstance(np_batch_2[2], tuple)
        self.assertEqual(np_batch_2[0].shape, (4, 5))
        self.assertEqual(np_batch_2[1]["loss"].shape, ())
        self.assertEqual(np_batch_2[2][0].shape, (4, 2, 3))
        self.assertEqual(np_batch_2[2][1].shape, (4, 6))

        # Test no batches
        none_arr = EvalLoopContainer(do_nested_concat=True, padding_index=-100).get_arrays()
        self.assertIsNone(none_arr)

        none_arr = EvalLoopContainer(do_nested_concat=False).get_arrays()
        self.assertIsNone(none_arr)

        # Test one batch
        concat_container = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        concat_container.add(batch_1)
        arrays = concat_container.get_arrays()
        self.assertIsInstance(arrays, list)
        self.assertEqual(len(arrays), 3)
        self.assertIsInstance(arrays[0], np.ndarray)
        self.assertEqual(arrays[0].shape, (8, 5))
        self.assertIsInstance(arrays[1], dict)
        self.assertIsInstance(arrays[1]["loss"], np.ndarray)
        self.assertEqual(arrays[1]["loss"].shape, ())
        self.assertTrue(np.allclose(arrays[1]["loss"], np.array([1.0])))
        self.assertIsInstance(arrays[2], tuple)
        self.assertEqual(len(arrays[2]), 2)
        self.assertEqual(arrays[2][0].shape, (8, 2, 3))
        self.assertEqual(arrays[2][1].shape, (8, 2))
