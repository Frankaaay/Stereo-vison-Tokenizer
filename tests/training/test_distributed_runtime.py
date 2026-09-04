import unittest
from unittest import mock
from types import SimpleNamespace

from stereo_tokenizer.data import StereoDataModule
from stereo_tokenizer.training.provenance import _is_global_zero_process
from stereo_tokenizer.training.runtime import validate_distributed_runtime_args


class DistributedRuntimeContractTest(unittest.TestCase):
    def test_lightning_subprocess_without_global_rank_is_not_global_zero(self):
        self.assertFalse(
            _is_global_zero_process({"NODE_RANK": "0", "LOCAL_RANK": "3"})
        )
        self.assertTrue(
            _is_global_zero_process({"NODE_RANK": "0", "LOCAL_RANK": "0"})
        )

    def test_global_rank_takes_precedence(self):
        self.assertFalse(
            _is_global_zero_process(
                {"RANK": "7", "NODE_RANK": "0", "LOCAL_RANK": "0"}
            )
        )

    def test_single_node_lightning_falls_back_to_configured_device_count(self):
        module = object.__new__(StereoDataModule)
        module.args = SimpleNamespace(devices=8)
        with mock.patch.dict(
            "os.environ", {"LOCAL_RANK": "3"}, clear=True
        ):
            self.assertEqual(module._local_shard(), (8, 3))

    def test_single_mode_requires_one_node(self):
        args = SimpleNamespace(distributed_mode="single", devices=8, num_nodes=2)
        with self.assertRaisesRegex(ValueError, "num_nodes=1"):
            validate_distributed_runtime_args(args, {})

    def test_ib_mode_accepts_two_nodes_and_matching_torchrun_world(self):
        args = SimpleNamespace(distributed_mode="ib", devices=2, num_nodes=2)
        validate_distributed_runtime_args(
            args,
            {
                "NODE_RANK": "1",
                "MASTER_ADDR": "214.30.239.40",
                "MASTER_PORT": "29641",
                "WORLD_SIZE": "4",
                "LOCAL_WORLD_SIZE": "2",
            },
        )

    def test_ib_mode_rejects_world_size_mismatch(self):
        args = SimpleNamespace(distributed_mode="ib", devices=2, num_nodes=2)
        with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
            validate_distributed_runtime_args(
                args,
                {
                    "NODE_RANK": "0",
                    "MASTER_ADDR": "214.30.239.40",
                    "MASTER_PORT": "29641",
                    "WORLD_SIZE": "2",
                    "LOCAL_WORLD_SIZE": "2",
                },
            )

    def test_ib_mode_requires_rendezvous_environment(self):
        args = SimpleNamespace(distributed_mode="ib", devices=2, num_nodes=2)
        with self.assertRaisesRegex(ValueError, "MASTER_ADDR"):
            validate_distributed_runtime_args(
                args,
                {
                    "NODE_RANK": "0",
                    "WORLD_SIZE": "4",
                },
            )


if __name__ == "__main__":
    unittest.main()
