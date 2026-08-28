import unittest

import torch

from stereo_tokenizer.mode_sampling import (
    MODE_IDS,
    DatasetSource,
    MixedModeBatchSampler,
    MixedModeDataset,
    mode_occurrences_before,
    parse_weight_spec,
)


class _Source:
    def __init__(self, eye_mode, length=17):
        self.eye_mode = eye_mode
        self.length = length

    def __len__(self):
        return self.length

    def get_mode_item(self, index, temporal_mode):
        views, eyes = ((1, 1) if self.eye_mode == "mono" else (3, 2))
        frames = 1 if temporal_mode == "single_frame" else 4
        return {"video": torch.zeros(views, eyes, 3, frames, 8, 8)}


class ThreeSourceScheduleTest(unittest.TestCase):
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

    def test_dispatches_hy_libero_and_umi_without_shape_conversion(self):
        dataset = MixedModeDataset(
            {
                "hy": DatasetSource("mono", _Source("mono")),
                "libero": DatasetSource("mono", _Source("mono")),
                "umi": DatasetSource("stereo", _Source("stereo")),
            }
        )
        self.assertEqual(dataset[("mono/four_frame", "hy", 0)]["dataset_id"], "hy")
        self.assertEqual(
            dataset[("mono/single_frame", "libero", 0)]["dataset_id"], "libero"
        )
        self.assertEqual(dataset[("stereo/four_frame", "umi", 0)]["dataset_id"], "umi")


if __name__ == "__main__":
    unittest.main()
