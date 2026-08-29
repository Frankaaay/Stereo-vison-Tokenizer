import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

from train_stereo_vae import StepTimingCallback, _validate_four_mode_batch_contract


class ThreeSourceRuntimeContractTest(unittest.TestCase):
    def _args(self, **updates):
        values = dict(
            grad_accumulates=1,
            batch_size=24,
            mode_batch_sizes="48:48:48:24",
            mode_grad_accumulates="1:1:1:2",
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
        with self.assertRaisesRegex(ValueError, "global grad_accumulates=1"):
            _validate_four_mode_batch_contract(self._args(grad_accumulates=2))

    def test_accepts_equal_effective_global_batches(self):
        _validate_four_mode_batch_contract(self._args())

    def test_rejects_unequal_effective_global_batches(self):
        with self.assertRaisesRegex(ValueError, "effective global batch sizes"):
            _validate_four_mode_batch_contract(
                self._args(mode_batch_sizes="48:48:48:25")
            )

    def test_rejects_mono_accumulation(self):
        with self.assertRaisesRegex(ValueError, "mono modes"):
            _validate_four_mode_batch_contract(
                self._args(
                    mode_batch_sizes="24:48:48:24",
                    mode_grad_accumulates="2:1:1:2",
                )
            )

    def test_dual_node_requires_global_manifest_contract(self):
        with self.assertRaisesRegex(ValueError, "node_manifest_contracts"):
            _validate_four_mode_batch_contract(self._args(num_nodes=2))
        _validate_four_mode_batch_contract(
            self._args(num_nodes=2, node_manifest_contracts='{"0":{},"1":{}}')
        )

    def test_rejects_invalid_weights(self):
        with self.assertRaises(ValueError):
            _validate_four_mode_batch_contract(self._args(mono_dataset_weights="9:0"))

    @mock.patch("train_stereo_vae.torch.cuda.reset_peak_memory_stats")
    @mock.patch("train_stereo_vae.torch.cuda.max_memory_reserved", return_value=20)
    @mock.patch("train_stereo_vae.torch.cuda.max_memory_allocated", return_value=10)
    @mock.patch("train_stereo_vae.torch.cuda.synchronize")
    def test_step_timing_separates_micro_and_logical_updates(
        self,
        synchronize,
        max_allocated,
        max_reserved,
        reset_peak,
    ):
        callback = StepTimingCallback("unused.json", warmup_updates=0)
        module = SimpleNamespace(
            generator_updates=0,
            batch_updates=0,
            _micro_step=0,
            last_mode_id="stereo/four_frame",
            last_temporal_mode="four_frame",
            last_micro_step_index=1,
            last_accumulation_factor=2,
            last_microbatch_size=24,
            last_logical_global_samples=0,
            log_dict=mock.Mock(),
        )
        trainer = SimpleNamespace(world_size=8)
        callback.on_train_start(trainer, module)
        callback.on_train_batch_start(trainer, module, None, 0)
        module.batch_updates = 1
        callback.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(len(callback.micro_timings), 1)
        self.assertEqual(callback.timings, [])

        module._micro_step = 1
        module.last_micro_step_index = 2
        callback.on_train_batch_start(trainer, module, None, 1)
        module._micro_step = 0
        module.batch_updates = 2
        module.generator_updates = 1
        module.last_logical_global_samples = 384
        callback.on_train_batch_end(trainer, module, None, None, 1)

        self.assertEqual(len(callback.micro_timings), 2)
        self.assertEqual(len(callback.timings), 1)
        self.assertEqual(callback.timings[0]["global_samples"], 384)
        self.assertGreater(callback.timings[0]["samples_per_s"], 0)


if __name__ == "__main__":
    unittest.main()
