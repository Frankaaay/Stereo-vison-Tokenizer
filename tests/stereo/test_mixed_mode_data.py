import unittest

from stereo_tokenizer.mode_sampling import (
    MODE_IDS,
    MixedModeBatchSampler,
    mode_for_update,
    mode_occurrences_before,
    mode_order_for_cycle,
)


class MixedModeScheduleTest(unittest.TestCase):
    def test_each_seeded_cycle_contains_all_four_modes_once(self) -> None:
        for cycle in range(8):
            self.assertEqual(set(mode_order_for_cycle(1234, cycle)), set(MODE_IDS))

    def test_schedule_is_deterministic_and_not_fixed_rotation(self) -> None:
        first = [mode_for_update(1234, update) for update in range(16)]
        second = [mode_for_update(1234, update) for update in range(16)]
        self.assertEqual(first, second)
        self.assertNotEqual(first[:4], first[4:8])

    def test_resume_occurrences_match_prefix(self) -> None:
        for stop in range(17):
            expected = {mode_id: 0 for mode_id in MODE_IDS}
            for update in range(stop):
                expected[mode_for_update(9, update)] += 1
            self.assertEqual(mode_occurrences_before(9, stop), expected)

    def test_ddp_ranks_share_modes_but_not_source_indices(self) -> None:
        kwargs = dict(
            source_lengths={"mono": 101, "stereo": 103},
            batch_size=24,
            seed=77,
            updates_per_epoch=8,
            start_update=5,
            num_replicas=2,
        )
        rank_zero = list(MixedModeBatchSampler(rank=0, **kwargs))
        rank_one = list(MixedModeBatchSampler(rank=1, **kwargs))
        self.assertEqual(
            [batch[0][0] for batch in rank_zero],
            [batch[0][0] for batch in rank_one],
        )
        for batch_zero, batch_one in zip(rank_zero, rank_one):
            self.assertEqual(len(batch_zero), 24)
            self.assertEqual(len(batch_one), 24)
            self.assertTrue(all(request[0] == batch_zero[0][0] for request in batch_zero))
            self.assertTrue(all(request[0] == batch_one[0][0] for request in batch_one))
            self.assertTrue(
                set(index for _, index in batch_zero).isdisjoint(
                    index for _, index in batch_one
                )
            )

    def test_fixed_48_sample_two_rank_smoke_has_two_super_cycles(self) -> None:
        schedules = []
        for rank in (0, 1):
            batches = list(
                MixedModeBatchSampler(
                    {"mono": 48, "stereo": 48},
                    batch_size=24,
                    seed=1234,
                    updates_per_epoch=8,
                    num_replicas=2,
                    rank=rank,
                )
            )
            schedules.append([batch[0][0] for batch in batches])
            self.assertEqual(
                {mode_id: schedules[-1].count(mode_id) for mode_id in MODE_IDS},
                {mode_id: 2 for mode_id in MODE_IDS},
            )
            for batch in batches:
                self.assertEqual(len({index for _, index in batch}), 24)
        self.assertEqual(schedules[0], schedules[1])

    def test_fixed_48_sample_eight_rank_bs24_repeats_rank_local_shard(self) -> None:
        rank_source_indices = []
        schedules = []
        for rank in range(8):
            batches = list(
                MixedModeBatchSampler(
                    {"mono": 48, "stereo": 48},
                    batch_size=24,
                    seed=1234,
                    updates_per_epoch=4,
                    num_replicas=8,
                    rank=rank,
                )
            )
            schedules.append([batch[0][0] for batch in batches])
            rank_indices = set(range(rank, 48, 8))
            rank_source_indices.append(rank_indices)
            self.assertEqual(len(rank_indices), 6)
            for batch in batches:
                indices = [index for _, index in batch]
                self.assertEqual(set(indices), rank_indices)
                self.assertEqual(
                    {index: indices.count(index) for index in rank_indices},
                    {index: 4 for index in rank_indices},
                )
        self.assertTrue(all(schedule == schedules[0] for schedule in schedules))
        self.assertEqual(set().union(*rank_source_indices), set(range(48)))
        for left in range(8):
            for right in range(left + 1, 8):
                self.assertTrue(
                    rank_source_indices[left].isdisjoint(
                        rank_source_indices[right]
                    )
                )


if __name__ == "__main__":
    unittest.main()
