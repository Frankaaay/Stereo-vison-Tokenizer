import unittest
from argparse import Namespace

from train_stereo_vae import _validate_four_mode_batch_contract


class ThreeSourceRuntimeContractTest(unittest.TestCase):
    def _args(self, **updates):
        values = dict(
            grad_accumulates=1,
            batch_size=24,
            devices=8,
            num_nodes=1,
            mode_update_weights="35:35:15:15",
            mono_dataset_weights="9:1",
            node_manifest_contracts=None,
        )
        values.update(updates)
        return Namespace(**values)

    def test_single_node_accepts_weighted_contract(self):
        _validate_four_mode_batch_contract(self._args())

    def test_rejects_gradient_accumulation(self):
        with self.assertRaisesRegex(ValueError, "grad_accumulates=1"):
            _validate_four_mode_batch_contract(self._args(grad_accumulates=2))

    def test_dual_node_requires_global_manifest_contract(self):
        with self.assertRaisesRegex(ValueError, "node_manifest_contracts"):
            _validate_four_mode_batch_contract(self._args(num_nodes=2))
        _validate_four_mode_batch_contract(
            self._args(num_nodes=2, node_manifest_contracts='{"0":{},"1":{}}')
        )

    def test_rejects_invalid_weights(self):
        with self.assertRaises(ValueError):
            _validate_four_mode_batch_contract(self._args(mono_dataset_weights="9:0"))


if __name__ == "__main__":
    unittest.main()
