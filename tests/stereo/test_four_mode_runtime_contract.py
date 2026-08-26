import unittest
from types import SimpleNamespace

from train_stereo_vae import _validate_four_mode_batch_contract


class FourModeRuntimeContractTest(unittest.TestCase):
    def test_accepts_two_rank_12_plus_12_recipe(self):
        _validate_four_mode_batch_contract(
            SimpleNamespace(
                batch_size=6,
                grad_accumulates=1,
                devices=2,
                num_nodes=1,
                mixed_mono_sample_limit=12,
                mixed_stereo_sample_limit=12,
            )
        )

    def test_accepts_two_rank_6_plus_6_recipe(self):
        _validate_four_mode_batch_contract(
            SimpleNamespace(
                batch_size=3,
                grad_accumulates=1,
                devices=2,
                num_nodes=1,
                mixed_mono_sample_limit=6,
                mixed_stereo_sample_limit=6,
            )
        )

    def test_rejects_limits_that_cannot_form_equal_rank_shards(self):
        with self.assertRaisesRegex(ValueError, "divisible by DDP world size"):
            _validate_four_mode_batch_contract(
                SimpleNamespace(
                    batch_size=3,
                    grad_accumulates=1,
                    devices=2,
                    num_nodes=1,
                    mixed_mono_sample_limit=5,
                    mixed_stereo_sample_limit=6,
                )
            )

    def test_rejects_batch_larger_than_rank_local_source_shard(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed rank-local"):
            _validate_four_mode_batch_contract(
                SimpleNamespace(
                    batch_size=4,
                    grad_accumulates=1,
                    devices=2,
                    num_nodes=1,
                    mixed_mono_sample_limit=6,
                    mixed_stereo_sample_limit=6,
                )
            )


if __name__ == "__main__":
    unittest.main()
