import unittest
from argparse import Namespace

import torch

from stereo_tokenizer import StereoVAE


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
            temporal_patch_size=4,
            defer_temporal_pool=False,
            defer_spatial_pool=False,
            spatial_pos="rope",
            spatial_depth=2,
            temporal_depth=1,
            causal_in_peg=True,
            dim_head=8,
            heads=4,
            attn_dropout=0.0,
            ff_dropout=0.0,
            ff_mult=4.0,
            initialize_vit=True,
            stereo_num_views=3,
            stereo_num_frames=4,
            stereo_search_radii=(1, 1, 1),
            stereo_search_direction="left",
            stereo_disparity_scale=(128.0, 128.0, 128.0),
            stereo_disparity_bias=-2.572,
            stereo_disparity_epsilon=1e-6,
            stereo_mode="stereo",
            rgb_weight=1.0,
            disparity_weight=1.0,
            gradient_weight=1.0,
            kl_weight=1e-6,
            kl_warmup_steps=100,
            smooth_l1_beta=1.0,
            recon_loss_type="l1",
            geometry_gradient_scale_px=16.0,
            perceptual_weight=0.0,
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
        }

    def test_forward_uses_structured_48_channel_one_slot_latent(self) -> None:
        model = StereoVAE(self._args()).eval()
        output = model(self._batch()["video"], sample_posterior=False)
        self.assertEqual(output.latent.shape, (1, 3, 48, 1, 4, 4))
        self.assertEqual(output.rgb.shape, (1, 3, 3, 4, 32, 32))
        self.assertEqual(output.disparity.shape, (1, 3, 1, 4, 32, 32))

    def test_mono_entrypoint_uses_same_four_frame_temporal_path(self) -> None:
        model = StereoVAE(self._args()).eval()
        output = model(
            self._batch()["video"], mode="mono", sample_posterior=False
        )
        self.assertIsNone(output.fusion)
        self.assertEqual(output.latent.shape, (1, 3, 48, 1, 4, 4))
        self.assertEqual(output.rgb.shape, (1, 3, 3, 4, 32, 32))

    def test_eval_default_uses_posterior_mean(self) -> None:
        model = StereoVAE(self._args()).eval()
        video = self._batch()["video"]
        first = model.encode(video)
        second = model.encode(video)
        torch.testing.assert_close(first.latent, first.posterior.mean)
        torch.testing.assert_close(first.latent, second.latent)

    def test_core_loss_backpropagates_through_main_model(self) -> None:
        model = StereoVAE(self._args()).train()
        result = model.compute_core_loss(self._batch(), sample_posterior=True)
        result.loss.total.backward()
        parameters = (
            model.encoder.enc_temporal_transformer.layers[0][1].to_q.weight,
            model.encoder.stereo_temporal_projection[1].weight,
            model.posterior_projection[1].weight,
            model.decoder.stereo_temporal_expansion[1].weight,
            model.decoder.dec_temporal_transformer.layers[0][1].to_q.weight,
            model.decoder.dec_spatial_transformer.layers[0][1].to_q.weight,
            model.decoder.stereo_rgb_head.weight,
            model.decoder.stereo_disparity_head.weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_main_model_has_no_codebook_or_legacy_image_mode(self) -> None:
        model = StereoVAE(self._args())
        self.assertFalse(hasattr(model, "codebook"))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 32, 32))

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
        model.batch_updates = 11
        checkpoint = {}
        model.on_save_checkpoint(checkpoint)

        restored = StereoVAE(self._args())
        restored.on_load_checkpoint(checkpoint)

        self.assertEqual(restored.generator_updates, 7)
        self.assertEqual(restored.discriminator_updates, 0)
        self.assertEqual(restored.batch_updates, 11)

    def test_legacy_non_gan_checkpoint_recovers_generator_updates(self) -> None:
        model = StereoVAE(self._args())

        model.on_load_checkpoint({"global_step": 9})

        self.assertEqual(model.generator_updates, 9)
        self.assertEqual(model.discriminator_updates, 0)
        self.assertEqual(model.batch_updates, 9)

    def test_legacy_gan_checkpoint_without_counters_is_rejected(self) -> None:
        args = self._args()
        args.gan_enabled = True
        args.image_gan_weight = 1.0
        model = StereoVAE(args)

        with self.assertRaisesRegex(ValueError, "independent optimizer counters"):
            model.on_load_checkpoint({"global_step": 9})


if __name__ == "__main__":
    unittest.main()
