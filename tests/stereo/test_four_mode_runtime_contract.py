import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

from stereo_tokenizer.mode_sampling import MODE_IDS, mode_for_update, parse_weight_spec
from train_stereo_vae import (
    DiscriminatorExpansionOptimizerCallback,
    StepTimingCallback,
    _load_discriminator_expansion_checkpoint,
    _load_stage_transition_checkpoint,
    _resolve_val_check_interval,
    _validate_four_mode_batch_contract,
)


class ThreeSourceRuntimeContractTest(unittest.TestCase):
    def test_discriminator_expansion_preserves_image_and_adds_only_video(self):
        checkpoint = {
            "state_dict": {
                "encoder.weight": object(),
                "image_discriminator.weight": object(),
            },
            "stereo_update_counters": {"discriminator_updates": 7},
            "optimizer_states": [{"generator": True}, {"discriminator": True}],
        }

        class Model:
            @staticmethod
            def state_dict():
                return {
                    "encoder.weight": object(),
                    "image_discriminator.weight": object(),
                    "video_discriminator.weight": object(),
                }

            def load_state_dict(self, state_dict, strict):
                self.asserted_state_dict = state_dict
                self.asserted_strict = strict
                return SimpleNamespace(
                    missing_keys=["video_discriminator.weight"],
                    unexpected_keys=[],
                )

            def on_load_checkpoint(self, loaded):
                self.loaded_checkpoint = loaded

        model = Model()
        _load_discriminator_expansion_checkpoint(
            model, checkpoint, "stage-b.ckpt"
        )

        self.assertIs(model.loaded_checkpoint, checkpoint)
        self.assertFalse(model.asserted_strict)
        self.assertIs(
            model._discriminator_expansion_optimizer_states,
            checkpoint["optimizer_states"],
        )
        self.assertTrue(
            model.discriminator_expansion_source.endswith("stage-b.ckpt")
        )

    def test_discriminator_expansion_rejects_existing_video_weights(self):
        checkpoint = {
            "state_dict": {
                "image_discriminator.weight": object(),
                "video_discriminator.weight": object(),
            }
        }
        with self.assertRaisesRegex(ValueError, "already has video"):
            _load_discriminator_expansion_checkpoint(
                SimpleNamespace(), checkpoint, "stage-c.ckpt"
            )

    def test_discriminator_expansion_restores_partial_optimizer_state(self):
        class Optimizer:
            def __init__(self, state):
                self.current = state
                self.loaded = None

            def state_dict(self):
                return self.current

            def load_state_dict(self, state):
                self.loaded = state

        generator_optimizer = Optimizer(
            {"state": {}, "param_groups": [{"params": [0, 1]}]}
        )
        discriminator_optimizer = Optimizer(
            {"state": {}, "param_groups": [{"params": [0, 1, 2]}]}
        )
        generator_source = {
            "state": {5: {"exp_avg": "generator"}},
            "param_groups": [{"params": [5, 6]}],
        }
        discriminator_source = {
            "state": {
                7: {"exp_avg": "image-0"},
                9: {"exp_avg": "image-1"},
            },
            "param_groups": [{"params": [7, 9]}],
        }
        schedulers = [SimpleNamespace(step_update=mock.Mock()) for _ in range(2)]
        module = SimpleNamespace(
            _discriminator_expansion_optimizer_states=[
                generator_source,
                discriminator_source,
            ],
            image_discriminator=SimpleNamespace(parameters=lambda: [1, 2]),
            video_discriminator=SimpleNamespace(parameters=lambda: [3]),
            generator_updates=100,
            discriminator_updates=56,
            _as_sequence=lambda value: value,
            lr_schedulers=lambda: schedulers,
        )
        trainer = SimpleNamespace(
            optimizers=[generator_optimizer, discriminator_optimizer]
        )

        DiscriminatorExpansionOptimizerCallback().on_train_start(trainer, module)

        self.assertEqual(generator_optimizer.loaded, generator_source)
        self.assertEqual(
            discriminator_optimizer.loaded["param_groups"][0]["params"],
            [0, 1, 2],
        )
        self.assertEqual(
            discriminator_optimizer.loaded["state"],
            {
                0: {"exp_avg": "image-0"},
                1: {"exp_avg": "image-1"},
            },
        )
        schedulers[0].step_update.assert_called_once_with(100)
        schedulers[1].step_update.assert_called_once_with(56)
        self.assertTrue(module.discriminator_expansion_optimizer_restored)
        self.assertFalse(
            hasattr(module, "_discriminator_expansion_optimizer_states")
        )

    def test_stage_transition_loads_only_new_discriminator_as_missing(self):
        checkpoint = {
            "state_dict": {"encoder.weight": object()},
            "stereo_update_counters": {"discriminator_updates": 0},
        }

        class Model:
            def __init__(self):
                self.loaded_checkpoint = None

            @staticmethod
            def state_dict():
                return {
                    "encoder.weight": object(),
                    "image_discriminator.weight": object(),
                }

            def load_state_dict(self, state_dict, strict):
                self.asserted_state_dict = state_dict
                self.asserted_strict = strict
                return SimpleNamespace(
                    missing_keys=["image_discriminator.weight"],
                    unexpected_keys=[],
                )

            def on_load_checkpoint(self, loaded):
                self.loaded_checkpoint = loaded

        model = Model()
        _load_stage_transition_checkpoint(model, checkpoint, "stage-a.ckpt")

        self.assertIs(model.loaded_checkpoint, checkpoint)
        self.assertFalse(model.asserted_strict)
        self.assertTrue(model.stage_transition_source.endswith("stage-a.ckpt"))

    def test_stage_transition_rejects_existing_discriminator_weights(self):
        checkpoint = {
            "state_dict": {"image_discriminator.weight": object()},
            "stereo_update_counters": {"discriminator_updates": 0},
        }
        with self.assertRaisesRegex(ValueError, "strict resume"):
            _load_stage_transition_checkpoint(
                SimpleNamespace(), checkpoint, "gan.ckpt"
            )

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
            four_mode_mixed_training=True,
            online_val_check_interval_steps=340,
            max_steps=340,
            mode_schedule_start_update=0,
            mode_schedule_seed=7,
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

    def test_validation_interval_expands_to_complete_logical_updates(self):
        args = self._args()
        weights = parse_weight_spec(args.mode_update_weights, MODE_IDS)
        expected = sum(
            2
            if mode_for_update(args.mode_schedule_seed, update, weights)
            == "stereo/four_frame"
            else 1
            for update in range(args.max_steps)
        )
        self.assertEqual(_resolve_val_check_interval(args), expected)
        self.assertGreater(expected, args.online_val_check_interval_steps)

    def test_periodic_validation_requires_complete_schedule_cycles(self):
        with self.assertRaisesRegex(ValueError, "whole mode-schedule cycles"):
            _resolve_val_check_interval(
                self._args(max_steps=1000, online_val_check_interval_steps=341)
            )

    def test_periodic_validation_expands_complete_schedule_cycles(self):
        interval = _resolve_val_check_interval(
            self._args(max_steps=1000, online_val_check_interval_steps=500)
        )
        self.assertEqual(interval, 575)

    def test_periodic_validation_requires_cycle_aligned_resume(self):
        with self.assertRaisesRegex(ValueError, "cycle-aligned schedule start"):
            _resolve_val_check_interval(
                self._args(
                    max_steps=1001,
                    mode_schedule_start_update=1,
                    online_val_check_interval_steps=500,
                )
            )

    def test_non_mixed_validation_interval_is_unchanged(self):
        args = self._args(four_mode_mixed_training=False)
        self.assertEqual(_resolve_val_check_interval(args), 340)

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
