import unittest
from types import SimpleNamespace

from train_stereo_vae import validate_distributed_runtime_args


class DistributedRuntimeContractTest(unittest.TestCase):
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
