import unittest

import torch

from stereo_tokenizer.mode_sampling import (
    MODE_IDS,
    DatasetSource,
    MixedModeBatchSampler,
    MixedModeDataset,
    mode_for_update,
    mode_occurrences_before,
    parse_positive_int_spec,
    parse_weight_spec,
)


class _Source:
    def __init__(self, eye_mode, views, length=17):
        self.eye_mode = eye_mode
        self.views = views
        self.length = length

    def __len__(self):
        return self.length

    def get_mode_item(self, index, temporal_mode):
        views = self.views
        eyes = 1 if self.eye_mode == "mono" else 2
        frames = 1 if temporal_mode == "single_frame" else 4
        return {"video": torch.zeros(views, eyes, 3, frames, 8, 8)}


class ThreeSourceScheduleTest(unittest.TestCase):
    def test_positive_integer_contract_preserves_absolute_values(self):
        self.assertEqual(
            parse_positive_int_spec("48:48:48:24", MODE_IDS),
            dict(zip(MODE_IDS, (48, 48, 48, 24))),
        )

    def test_weighted_mode_and_dataset_counts_are_exact_per_cycle(self):
        weights = parse_weight_spec("35:35:15:15", MODE_IDS)
        self.assertEqual(
            mode_occurrences_before(7, 20, weights),
            dict(zip(MODE_IDS, (7, 7, 3, 3))),
        )
        batches = list(
            MixedModeBatchSampler(
                {"hy": 31, "libero": 11, "umi": 23},
                batch_size=2,
                seed=7,
                updates_per_epoch=100,
                mode_weights=weights,
                mono_dataset_weights={"hy": 9, "libero": 1},
            )
        )
        dataset_ids = [batch[0][1] for batch in batches]
        self.assertEqual(dataset_ids.count("umi"), 30)
        self.assertEqual(dataset_ids.count("hy"), 63)
        self.assertEqual(dataset_ids.count("libero"), 7)

    def test_node_local_manifests_keep_equal_updates_and_distinct_indices(self):
        kwargs = dict(
            source_lengths={"hy": 10, "libero": 10, "umi": 10},
            batch_size=3,
            seed=1234,
            updates_per_epoch=25,
            shard_num_replicas=2,
        )
        rank0 = list(MixedModeBatchSampler(shard_rank=0, **kwargs))
        rank1 = list(MixedModeBatchSampler(shard_rank=1, **kwargs))
        self.assertEqual(len(rank0), len(rank1))
        for batch0, batch1 in zip(rank0, rank1):
            self.assertEqual(batch0[0][:2], batch1[0][:2])
            self.assertTrue(all(item[2] % 2 == 0 for item in batch0))
            self.assertTrue(all(item[2] % 2 == 1 for item in batch1))

    def test_mode_aware_sampler_emits_consecutive_micro_batches(self):
        weights = parse_weight_spec("35:35:15:15", MODE_IDS)
        batch_sizes = dict(zip(MODE_IDS, (2, 2, 2, 1)))
        accumulation = dict(zip(MODE_IDS, (1, 1, 1, 2)))
        sampler = MixedModeBatchSampler(
            {"hy": 101, "libero": 101, "umi": 101},
            batch_size=2,
            mode_batch_sizes=batch_sizes,
            mode_accumulation_factors=accumulation,
            seed=7,
            updates_per_epoch=20,
            mode_weights=weights,
        )
        batches = list(sampler)
        self.assertEqual(len(sampler), 23)
        cursor = 0
        for update_index in range(20):
            mode_id = mode_for_update(7, update_index, weights)
            logical_batches = batches[cursor : cursor + accumulation[mode_id]]
            self.assertEqual(len(logical_batches), accumulation[mode_id])
            self.assertTrue(
                all(len(batch) == batch_sizes[mode_id] for batch in logical_batches)
            )
            self.assertTrue(
                all(item[0] == mode_id for batch in logical_batches for item in batch)
            )
            if accumulation[mode_id] == 2:
                self.assertNotEqual(logical_batches[0], logical_batches[1])
            cursor += accumulation[mode_id]
        self.assertEqual(cursor, len(batches))

    def test_resume_suffix_preserves_physical_batch_stream(self):
        kwargs = dict(
            source_lengths={"hy": 101, "libero": 101, "umi": 101},
            batch_size=2,
            mode_batch_sizes=dict(zip(MODE_IDS, (2, 2, 2, 1))),
            mode_accumulation_factors=dict(zip(MODE_IDS, (1, 1, 1, 2))),
            seed=19,
            mode_weights=parse_weight_spec("35:35:15:15", MODE_IDS),
        )
        full = list(MixedModeBatchSampler(updates_per_epoch=20, **kwargs))
        prefix = list(MixedModeBatchSampler(updates_per_epoch=5, **kwargs))
        resumed = list(
            MixedModeBatchSampler(
                updates_per_epoch=15,
                start_update=5,
                **kwargs,
            )
        )
        self.assertEqual(resumed, full[len(prefix) :])

    def test_dispatches_hy_libero_and_umi_without_shape_conversion(self):
        dataset = MixedModeDataset(
            {
                "hy": DatasetSource("mono", _Source("mono", 3)),
                "libero": DatasetSource("mono", _Source("mono", 2)),
                "umi": DatasetSource("stereo", _Source("stereo", 3)),
            }
        )
        self.assertEqual(dataset[("mono/four_frame", "hy", 0)]["dataset_id"], "hy")
        self.assertEqual(
            dataset[("mono/single_frame", "libero", 0)]["dataset_id"], "libero"
        )
        self.assertEqual(dataset[("stereo/four_frame", "umi", 0)]["dataset_id"], "umi")


if __name__ == "__main__":
    unittest.main()
