import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

import torch

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.mode_sampling import (
    MODE_IDS,
    mode_for_update,
    mode_occurrences_before,
    parse_weight_spec,
)
from stereo_tokenizer.modules.attention import PEG
from train_stereo_vae import _load_continuation_checkpoint


class StereoVAEIntegrationTest(unittest.TestCase):
    def _args(self) -> Namespace:
        return Namespace(
            resolution=32,
            image_channels=3,
            norm_type="group",
            embedding_dim=32,
            latent_channels=48,
            patch_size=8,
            patch_embed="linear",
            enc_block="tt",
            dec_block="tt",
            twod_window_size=2,
            defer_temporal_pool=False,
            defer_spatial_pool=False,
            spatial_pos="rope",
            spatial_depth=2,
            temporal_depth=1,
            causal_in_peg=True,
            causal_in_temporal_transformer=False,
            peg_backend="conv3d_contiguous",
            dim_head=8,
            heads=4,
            attn_dropout=0.0,
            ff_dropout=0.0,
            ff_mult=4.0,
            initialize_vit=True,
            stereo_num_views=3,
            stereo_num_frames=4,
            single_frame_source_index=0,
            stereo_search_radii=(1, 1, 1),
            stereo_search_direction="left",
            rgb_weight=1.0,
            relative_depth_weight=1.0,
            relative_gradient_weight=1.0,
            relative_depth_epsilon=1e-6,
            kl_weight=1e-6,
            kl_warmup_steps=100,
            smooth_l1_beta=1.0,
            recon_loss_type="l1",
            perceptual_weight=0.0,
            perceptual_frame_microbatch=2,
            perceptual_channels_last=0,
            perceptual_compile=0,
            gan_enabled=False,
            image_gan_weight=0.0,
            video_gan_weight=0.0,
            gan_feat_weight=0.0,
            discriminator_iter_start=0,
            disc_channels=32,
            disc_layers=2,
            disc_loss_type="hinge",
            sigmoid_in_disc=False,
            activation_in_disc="leaky_relu",
            apply_noise=False,
            apply_diffaug=False,
            grad_accumulates=1,
            batch_size=24,
            mode_batch_sizes=None,
            mode_grad_accumulates=None,
            four_mode_mixed_training=False,
            mode_update_weights="35:35:15:15",
            mono_dataset_weights="9:1",
            node_manifest_contracts=None,
            mode_schedule_seed=1234,
            devices=1,
            num_nodes=1,
            grad_clip_val=1.0,
            grad_clip_val_disc=1.0,
            lr=3e-4,
            lr_min=0.0,
            max_steps=100,
            warmup_steps=0,
            warmup_lr_init=0.0,
            dis_lr_multiplier=1.0,
            dis_minlr_multiplier=False,
            dis_warmup_steps=0,
        )

    def _batch(self):
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        disparity = torch.rand(1, 3, 1, 4, 32, 32) * 20.0 + 0.5
        return {
            "video": video,
            "disparity": disparity,
            "valid_mask": torch.ones_like(disparity, dtype=torch.bool),
            "fx": torch.tensor(((100.0, 110.0, 120.0),)),
            "baseline_m": torch.tensor(((0.1, 0.1, 0.1),)),
            "mode_id": ["stereo/four_frame"],
            "eye_mode": ["stereo"],
            "temporal_mode": ["four_frame"],
            "view_count": torch.tensor([3]),
            "teacher_kind": ["foundation_stereo"],
        }

    def _single_batch(self):
        batch = self._batch()
        selected = dict(batch)
        selected["video"] = batch["video"][..., :1, :, :]
        selected["disparity"] = batch["disparity"][..., :1, :, :]
        selected["valid_mask"] = batch["valid_mask"][..., :1, :, :]
        selected["mode_id"] = ["stereo/single_frame"]
        selected["temporal_mode"] = ["single_frame"]
        return selected

    def test_forward_uses_structured_48_channel_one_slot_latent(self) -> None:
        model = StereoVAE(self._args()).eval()
        output = model(
            self._batch()["video"],
            eye_mode="stereo",
            temporal_mode="four_frame",
            sample_posterior=False,
        )
        self.assertEqual(output.latent.shape, (1, 3, 48, 1, 4, 4))
        self.assertEqual(output.rgb.shape, (1, 3, 3, 4, 32, 32))
        self.assertEqual(
            output.raw_relative_log_depth.shape, (1, 3, 1, 4, 32, 32)
        )

    def test_mono_two_views_use_one_four_frame_forward(self) -> None:
        model = StereoVAE(self._args()).eval()
        output = model(
            self._batch()["video"][:, :2, :1],
            eye_mode="mono",
            temporal_mode="four_frame",
            sample_posterior=False,
        )
        self.assertIsNone(output.fusion)
        self.assertEqual(output.latent.shape, (1, 2, 48, 1, 4, 4))
        self.assertEqual(output.rgb.shape, (1, 2, 3, 4, 32, 32))
        self.assertEqual(
            output.raw_relative_log_depth.shape, (1, 2, 1, 4, 32, 32)
        )

    def test_constructor_applies_selected_peg_backend_to_reduced_model(self) -> None:
        args = self._args()
        args.peg_backend = "conv2d_t1_slice"

        model = StereoVAE(args)
        peg_modules = [module for module in model.modules() if isinstance(module, PEG)]

        self.assertTrue(peg_modules)
        self.assertTrue(
            all(module._backend == "conv2d_t1_slice" for module in peg_modules)
        )
        self.assertFalse(
            any(
                isinstance(module, PEG)
                for module in model.encoder.enc_temporal_transformer.modules()
            )
        )
        self.assertFalse(
            any(
                isinstance(module, PEG)
                for module in model.decoder.dec_temporal_transformer.modules()
            )
        )

    def test_eval_default_uses_posterior_mean(self) -> None:
        model = StereoVAE(self._args()).eval()
        video = self._batch()["video"]
        first = model.encode(
            video,
            eye_mode="stereo",
            temporal_mode="four_frame",
        )
        second = model.encode(
            video,
            eye_mode="stereo",
            temporal_mode="four_frame",
        )
        torch.testing.assert_close(first.latent, first.posterior.mean)
        torch.testing.assert_close(first.latent, second.latent)

    def test_core_loss_backpropagates_through_main_model(self) -> None:
        model = StereoVAE(self._args()).train()
        result = model.compute_core_loss(
            self._batch(),
            eye_mode="stereo",
            temporal_mode="four_frame",
            sample_posterior=True,
        )
        result.loss.total.backward()
        parameters = (
            model.encoder.enc_temporal_transformer.layers[0][1].to_q.weight,
            model.encoder.stereo_temporal_projection[1].weight,
            model.posterior_projection[1].weight,
            model.decoder.stereo_temporal_expansion[1].weight,
            model.decoder.dec_temporal_transformer.layers[0][1].to_q.weight,
            model.decoder.dec_spatial_transformer.layers[0][1].to_q.weight,
            model.decoder.stereo_rgb_head.weight,
            model.decoder.relative_log_depth_head.weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_main_model_has_no_codebook_or_legacy_image_mode(self) -> None:
        model = StereoVAE(self._args())
        self.assertFalse(hasattr(model, "codebook"))
        with self.assertRaises(ValueError):
            model(
                torch.randn(1, 3, 32, 32),
                eye_mode="stereo",
                temporal_mode="four_frame",
            )

    def test_checkpoint_hyperparameters_preserve_constructor_args(self) -> None:
        args = self._args()
        model = StereoVAE(args)

        self.assertEqual(set(model.hparams), {"args"})
        self.assertIsInstance(model.hparams["args"], Namespace)
        self.assertEqual(vars(model.hparams["args"]), vars(args))

    def test_train_keeps_perceptual_model_in_eval_mode(self) -> None:
        model = StereoVAE(self._args())
        model.perceptual_model = torch.nn.Dropout(p=0.5)

        model.train()

        self.assertTrue(model.training)
        self.assertFalse(model.perceptual_model.training)

    def test_perceptual_loss_chunks_views_and_frames_without_changing_mean(
        self,
    ) -> None:
        class RecordingPerceptual(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.batch_shapes = []

            def forward(self, prediction, target):
                self.batch_shapes.append(tuple(prediction.shape))
                return (prediction - target).square().mean(
                    dim=(1, 2, 3), keepdim=True
                )

        model = StereoVAE(self._args())
        perceptual_model = RecordingPerceptual()
        model.perceptual_model = perceptual_model
        model.perceptual_weight = 0.5
        prediction = torch.randn(2, 3, 3, 4, 8, 8, requires_grad=True)
        target = torch.randn_like(prediction)

        actual = model._perceptual_loss(prediction, target)
        expected = ((prediction - target) * 2.0).square().mean() * 0.5

        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(len(perceptual_model.batch_shapes), 12)
        self.assertEqual(
            set(perceptual_model.batch_shapes),
            {(2, 3, 8, 8)},
        )

    def test_perceptual_loss_supports_channels_last_and_compile(self) -> None:
        class RecordingPerceptual(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.channels_last = []

            def forward(self, prediction, target):
                self.channels_last.append(
                    prediction.is_contiguous(memory_format=torch.channels_last)
                    and target.is_contiguous(memory_format=torch.channels_last)
                )
                return (prediction - target).square().mean(
                    dim=(1, 2, 3), keepdim=True
                )

        args = self._args()
        args.perceptual_frame_microbatch = 5
        args.perceptual_channels_last = 1
        args.perceptual_compile = 1
        model = StereoVAE(args)
        perceptual_model = RecordingPerceptual()
        model.perceptual_model = perceptual_model
        model.perceptual_weight = 1.0
        prediction = torch.randn(2, 3, 3, 4, 8, 8)
        target = torch.randn_like(prediction)

        with mock.patch(
            "torch.compile", side_effect=lambda module, **_: module
        ) as compile_mock:
            actual = model._perceptual_loss(prediction, target)
            second = model._perceptual_loss(prediction, target)

        expected = ((prediction - target) * 2.0).square().mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(second, expected)
        compile_mock.assert_called_once_with(perceptual_model, dynamic=False)
        self.assertEqual(len(perceptual_model.channels_last), 10)
        self.assertTrue(all(perceptual_model.channels_last))

    def test_gan_builds_only_weighted_discriminators(self) -> None:
        image_args = self._args()
        image_args.gan_enabled = True
        image_args.image_gan_weight = 1.0
        image_model = StereoVAE(image_args)
        self.assertIsNotNone(image_model.image_discriminator)
        self.assertIsNone(image_model.video_discriminator)

        video_args = self._args()
        video_args.gan_enabled = True
        video_args.video_gan_weight = 1.0
        video_model = StereoVAE(video_args)
        self.assertIsNone(video_model.image_discriminator)
        self.assertIsNotNone(video_model.video_discriminator)
        final_block = getattr(
            video_model.video_discriminator,
            f"model{video_model.video_discriminator.n_layers + 1}",
        )
        self.assertEqual(len(final_block), 1)
        self.assertIsInstance(final_block[0], torch.nn.Conv3d)

    def test_generator_updates_control_kl_warmup_and_gan_activation(self) -> None:
        args = self._args()
        args.gan_enabled = True
        args.image_gan_weight = 1.0
        args.discriminator_iter_start = 5
        args.kl_weight = 1e-4
        args.kl_warmup_steps = 10
        model = StereoVAE(args)

        model.generator_updates = 4
        self.assertAlmostEqual(model._effective_kl_weight(), 4e-5)
        self.assertFalse(model._gan_is_active())

        model.generator_updates = 5
        self.assertAlmostEqual(model._effective_kl_weight(), 5e-5)
        self.assertTrue(model._gan_is_active())

        optimizers, schedulers = model.configure_optimizers()
        self.assertEqual(len(optimizers), 2)
        self.assertEqual(len(schedulers), 2)
        self.assertEqual(schedulers[0]["scheduler"].t_initial, 100)
        self.assertEqual(schedulers[1]["scheduler"].t_initial, 95)

    def test_update_counters_round_trip_through_checkpoint_hook(self) -> None:
        model = StereoVAE(self._args())
        model.generator_updates = 7
        model.discriminator_updates = 0
        model.four_frame_updates = 4
        model.single_frame_updates = 3
        model.mode_updates = {
            "mono/single_frame": 2,
            "mono/four_frame": 2,
            "stereo/single_frame": 1,
            "stereo/four_frame": 2,
        }
        model.mode_samples = {
            mode_id: count * 24 for mode_id, count in model.mode_updates.items()
        }
        model.batch_updates = 11
        checkpoint = {}
        model.on_save_checkpoint(checkpoint)

        restored = StereoVAE(self._args())
        restored.on_load_checkpoint(checkpoint)

        self.assertEqual(restored.generator_updates, 7)
        self.assertEqual(restored.discriminator_updates, 0)
        self.assertEqual(restored.four_frame_updates, 4)
        self.assertEqual(restored.single_frame_updates, 3)
        self.assertEqual(restored.batch_updates, 11)

    def test_checkpoint_without_temporal_counters_is_rejected(self) -> None:
        model = StereoVAE(self._args())

        with self.assertRaisesRegex(ValueError, "temporal-mode update counters"):
            model.on_load_checkpoint({"global_step": 9})

    def test_checkpoint_rejects_incomplete_logical_update(self) -> None:
        model = StereoVAE(self._args())
        model._micro_step = 1

        with self.assertRaisesRegex(RuntimeError, "incomplete logical update"):
            model.on_save_checkpoint({})

    def test_single_frame_forward_and_metadata(self) -> None:
        model = StereoVAE(self._args()).eval()
        single_batch = self._single_batch()
        single_batch = model._prepare_temporal_batch(
            single_batch, temporal_mode="single_frame"
        )
        output = model(
            single_batch["video"],
            eye_mode="stereo",
            temporal_mode="single_frame",
            sample_posterior=False,
        )

        self.assertEqual(single_batch["video"].shape[4], 1)
        self.assertEqual(single_batch["disparity"].shape[3], 1)
        self.assertEqual(single_batch["valid_mask"].shape[3], 1)
        self.assertEqual(output.latent.shape, (1, 3, 48, 1, 4, 4))
        self.assertEqual(output.rgb.shape, (1, 3, 3, 1, 32, 32))
        self.assertEqual(
            output.raw_relative_log_depth.shape, (1, 3, 1, 1, 32, 32)
        )
        self.assertEqual(output.eye_mode, "stereo")
        self.assertEqual(output.temporal_mode, "single_frame")
        self.assertEqual(output.source_num_frames, 1)

    def test_single_frame_rejects_a_predecoded_four_frame_batch(self):
        model = StereoVAE(self._args())
        with self.assertRaisesRegex(ValueError, "already be decoded with T=1"):
            model._prepare_temporal_batch(
                self._batch(), temporal_mode="single_frame"
            )

    def test_single_core_loss_backpropagates_only_single_temporal_path(self) -> None:
        model = StereoVAE(self._args()).train()
        single_batch = self._single_batch()
        result = model.compute_core_loss(
            single_batch,
            eye_mode="stereo",
            temporal_mode="single_frame",
            sample_posterior=True,
        )
        result.loss.total.backward()

        for parameter in (
            model.encoder.single_frame_projection[1].weight,
            model.posterior_projection[1].weight,
            model.decoder.single_frame_expansion[1].weight,
            model.decoder.dec_spatial_transformer.layers[0][1].to_q.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)
        self.assertIsNone(
            model.encoder.enc_temporal_transformer.layers[0][1].to_q.weight.grad
        )

    def test_mode_is_read_from_uniform_batch_metadata(self) -> None:
        self.assertEqual(
            StereoVAE._mode_from_batch(self._single_batch()),
            ("stereo/single_frame", "stereo", "single_frame"),
        )
        mixed = self._single_batch()
        mixed["mode_id"] = ["stereo/single_frame", "stereo/four_frame"]
        with self.assertRaisesRegex(ValueError, "entries for B=1"):
            StereoVAE._mode_from_batch(mixed)

        wrong_views = self._single_batch()
        wrong_views["view_count"] = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, "view_count metadata"):
            StereoVAE._mode_from_batch(wrong_views)

        wrong_teacher = self._single_batch()
        wrong_teacher["teacher_kind"] = ["da3"]
        with self.assertRaisesRegex(ValueError, "teacher_kind metadata"):
            StereoVAE._mode_from_batch(wrong_teacher)

    def test_checkpoint_resume_preserves_four_mode_counters(self) -> None:
        args = self._args()
        args.four_mode_mixed_training = True
        args.mode_batch_sizes = "48:48:48:24"
        args.mode_grad_accumulates = "1:1:1:2"
        args.devices = 8
        model = StereoVAE(args)
        model.generator_updates = 20
        model.mode_updates = mode_occurrences_before(
            args.mode_schedule_seed,
            model.generator_updates,
            parse_weight_spec(args.mode_update_weights, MODE_IDS),
        )
        model.four_frame_updates = sum(
            count
            for mode_id, count in model.mode_updates.items()
            if mode_id.endswith("four_frame")
        )
        model.single_frame_updates = model.generator_updates - model.four_frame_updates
        model.mode_samples = {
            mode_id: count * 384 for mode_id, count in model.mode_updates.items()
        }
        model.batch_updates = sum(
            count * model.mode_grad_accumulates[mode_id]
            for mode_id, count in model.mode_updates.items()
        )
        checkpoint = {}
        model.on_save_checkpoint(checkpoint)

        restored = StereoVAE(args)
        restored.on_load_checkpoint(checkpoint)

        self.assertEqual(restored.mode_updates, model.mode_updates)
        self.assertEqual(restored.mode_samples, model.mode_samples)

    def test_continuation_checkpoint_preserves_historical_samples_across_new_batch(self):
        source_args = self._args()
        source_args.four_mode_mixed_training = True
        source_args.mode_batch_sizes = "24:24:24:24"
        source_args.mode_grad_accumulates = "1:1:1:1"
        source_args.devices = 8
        source_args.node_manifest_contracts = "old"
        source = StereoVAE(source_args)
        source.generator_updates = 20
        source.mode_updates = mode_occurrences_before(
            source_args.mode_schedule_seed,
            source.generator_updates,
            parse_weight_spec(source_args.mode_update_weights, MODE_IDS),
        )
        source.four_frame_updates = sum(
            count
            for mode_id, count in source.mode_updates.items()
            if mode_id.endswith("four_frame")
        )
        source.single_frame_updates = (
            source.generator_updates - source.four_frame_updates
        )
        source.mode_samples = {
            mode_id: count * 192
            for mode_id, count in source.mode_updates.items()
        }
        source.batch_updates = 20
        checkpoint = {"state_dict": source.state_dict()}
        source.on_save_checkpoint(checkpoint)

        target_args = self._args()
        target_args.four_mode_mixed_training = True
        target_args.mode_batch_sizes = "48:48:48:24"
        target_args.mode_grad_accumulates = "1:1:1:2"
        target_args.devices = 8
        target_args.node_manifest_contracts = "new"
        target_args.continuation_checkpoint = "stage-a.ckpt"
        target = StereoVAE(target_args)
        _load_continuation_checkpoint(target, checkpoint, "stage-a.ckpt")

        self.assertEqual(target.generator_updates, 20)
        self.assertEqual(target.mode_samples, source.mode_samples)
        next_mode = mode_for_update(
            target_args.mode_schedule_seed,
            target.generator_updates,
            parse_weight_spec(target_args.mode_update_weights, MODE_IDS),
        )
        target.generator_updates += 1
        target.mode_updates[next_mode] += 1
        target.mode_samples[next_mode] += 384
        target.batch_updates += target.mode_grad_accumulates[next_mode]
        if next_mode.endswith("four_frame"):
            target.four_frame_updates += 1
        else:
            target.single_frame_updates += 1
        continued = {}
        target.on_save_checkpoint(continued)

        restored = StereoVAE(target_args)
        restored.on_load_checkpoint(continued)
        self.assertEqual(restored.generator_updates, 21)
        self.assertEqual(restored.mode_samples, target.mode_samples)

    def test_stereo_four_ga2_steps_once_and_counts_384_samples(self) -> None:
        args = self._args()
        args.four_mode_mixed_training = True
        args.mode_batch_sizes = "48:48:48:24"
        args.mode_grad_accumulates = "1:1:1:2"
        args.devices = 8
        args.grad_clip_val = None
        model = StereoVAE(args)
        model._trainer = SimpleNamespace(world_size=8, should_stop=False)
        optimizer = mock.Mock()
        scheduler = mock.Mock()
        zero = torch.zeros(())
        result = SimpleNamespace(
            model=SimpleNamespace(rgb=torch.zeros(24, 1, 3, 1, 1, 1)),
            loss=SimpleNamespace(total=torch.tensor(4.0, requires_grad=True)),
        )
        batch = {
            "video": torch.zeros(24, 1, 1, 3, 4, 1, 1),
            "mode_id": ["stereo/four_frame"] * 24,
            "dataset_id": ["umi"] * 24,
        }
        adversarial_zeros = (zero, zero, zero, zero)
        with (
            mock.patch(
                "stereo_tokenizer.model.mode_for_update",
                return_value="stereo/four_frame",
            ),
            mock.patch(
                "stereo_tokenizer.model.dataset_for_mode_occurrence",
                return_value="umi",
            ),
            mock.patch.object(
                model,
                "_mode_from_batch",
                return_value=("stereo/four_frame", "stereo", "four_frame"),
            ),
            mock.patch.object(
                model,
                "_prepare_temporal_batch",
                side_effect=lambda value, **_: value,
            ),
            mock.patch.object(model, "compute_core_loss", return_value=result),
            mock.patch.object(model, "_perceptual_loss", return_value=zero),
            mock.patch.object(
                model,
                "_generator_adversarial_loss",
                return_value=adversarial_zeros,
            ),
            mock.patch.object(model, "_gan_is_active", return_value=False),
            mock.patch.object(model, "optimizers", return_value=[optimizer]),
            mock.patch.object(model, "lr_schedulers", return_value=[scheduler]),
            mock.patch.object(model, "manual_backward") as backward,
            mock.patch.object(model, "_log_loss_breakdown"),
            mock.patch.object(model, "log_dict"),
        ):
            model._profiled_training_step(batch)
            self.assertEqual(model.generator_updates, 0)
            optimizer.step.assert_not_called()
            model._profiled_training_step(batch)

        self.assertEqual(backward.call_count, 2)
        self.assertEqual(backward.call_args_list[0].args[0].item(), 2.0)
        optimizer.step.assert_called_once_with()
        optimizer.zero_grad.assert_called_once_with()
        scheduler.step_update.assert_called_once_with(1)
        self.assertEqual(model.generator_updates, 1)
        self.assertEqual(model.batch_updates, 2)
        self.assertEqual(model.mode_updates["stereo/four_frame"], 1)
        self.assertEqual(model.mode_samples["stereo/four_frame"], 384)


if __name__ == "__main__":
    unittest.main()
