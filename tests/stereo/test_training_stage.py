import unittest

import torch
import torch.nn as nn

from OmniTokenizer.stereo.model import StereoTokenizer, StereoTokenizerConfig
from OmniTokenizer.stereo.training import (
    StereoAdversarialConfig,
    StereoOptimizerConfig,
    StereoTokenizerTrainingCore,
    StereoTokenizerTrainingStage,
    StereoTrainingLossConfig,
    StereoValidationPolicy,
)


def tiny_config() -> StereoTokenizerConfig:
    return StereoTokenizerConfig(
        search_radii=(0, 1, 2),
        search_direction="left",
        disparity_scale=(128.0, 128.0, 128.0),
        disparity_head_bias=-2.572,
        image_size=(32, 32),
        patch_size=(8, 8),
        embedding_dim=32,
        encoder_block="tw",
        decoder_block="tt",
        window_size=2,
        temporal_depth=1,
        attention_heads=4,
        head_dim=8,
    )


def loss_config(perceptual_weight: float = 0.5) -> StereoTrainingLossConfig:
    return StereoTrainingLossConfig(
        mode="stereo",
        rgb_weight=1.0,
        disparity_weight=1.0,
        gradient_weight=0.1,
        kl_target_weight=0.01,
        perceptual_weight=perceptual_weight,
        kl_warmup_steps=10,
        smooth_l1_beta=1.0,
        geometry_gradient_scale_px=16.0,
        rgb_loss_type="l1",
    )


def stereo_batch(batch_size: int = 1):
    video = torch.rand(batch_size, 3, 2, 3, 4, 32, 32) - 0.5
    disparity = torch.full((batch_size, 3, 1, 4, 32, 32), 9.42)
    valid_mask = torch.ones_like(disparity, dtype=torch.bool)
    return {
        "video": video,
        "disparity": disparity,
        "valid_mask": valid_mask,
    }


class RecordingPerceptual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shapes = []
        self.last_target = None

    def forward(self, target: torch.Tensor, prediction: torch.Tensor):
        self.shapes.append((tuple(target.shape), tuple(prediction.shape)))
        self.last_target = target.detach().clone()
        return (target - prediction).abs().mean(dim=(1, 2, 3), keepdim=True)


class RecordingDiscriminator(nn.Module):
    def __init__(self, expected_ndim: int) -> None:
        super().__init__()
        self.expected_ndim = expected_ndim
        self.scale = nn.Parameter(torch.ones(()))
        self.shapes = []

    def forward(self, inputs: torch.Tensor, apply_diffaug: bool = False):
        if inputs.ndim != self.expected_ndim:
            raise ValueError("unexpected discriminator input rank")
        self.shapes.append((tuple(inputs.shape), apply_diffaug))
        logits = inputs.mean(
            dim=tuple(range(1, inputs.ndim)), keepdim=True
        ) * self.scale
        return logits, [inputs * self.scale, logits]


class StereoTrainingStageTest(unittest.TestCase):
    def test_lpips_flattens_all_views_and_frames_and_scales_range(self) -> None:
        batch = stereo_batch(batch_size=2)
        batch["video"][:, :, 0] = 0.25
        perceptual = RecordingPerceptual()
        core = StereoTokenizerTrainingCore(
            StereoTokenizer(tiny_config()), loss_config()
        )
        stage = StereoTokenizerTrainingStage(
            core,
            StereoAdversarialConfig(
                enabled=False,
                start_step=0,
                image_weight=0.0,
                video_weight=0.0,
                feature_matching_weight=0.0,
            ),
            perceptual_model=perceptual,
        )

        output = stage.generator_step(batch, global_step=0)

        self.assertEqual(
            perceptual.shapes,
            [((24, 3, 32, 32), (24, 3, 32, 32))],
        )
        torch.testing.assert_close(
            perceptual.last_target, torch.full_like(perceptual.last_target, 0.5)
        )
        self.assertFalse(output.loss.gan_active)
        self.assertIsNone(stage.image_discriminator)
        self.assertIsNone(stage.video_discriminator)

    def test_shared_discriminators_receive_flattened_views(self) -> None:
        batch = stereo_batch(batch_size=2)
        perceptual = RecordingPerceptual()
        image_discriminator = RecordingDiscriminator(expected_ndim=4)
        video_discriminator = RecordingDiscriminator(expected_ndim=5)
        core = StereoTokenizerTrainingCore(
            StereoTokenizer(tiny_config()), loss_config()
        )
        stage = StereoTokenizerTrainingStage(
            core,
            StereoAdversarialConfig(
                enabled=True,
                start_step=10,
                image_weight=1.0,
                video_weight=1.0,
                feature_matching_weight=1.0,
                discriminator_layers=1,
            ),
            perceptual_model=perceptual,
            image_discriminator=image_discriminator,
            video_discriminator=video_discriminator,
        )

        inactive = stage.generator_step(batch, global_step=9)
        self.assertFalse(inactive.loss.gan_active)
        self.assertEqual(image_discriminator.shapes, [])
        self.assertEqual(video_discriminator.shapes, [])

        active = stage.generator_step(batch, global_step=10)
        self.assertTrue(active.loss.gan_active)
        self.assertEqual(image_discriminator.shapes[0][0], (24, 3, 32, 32))
        self.assertEqual(video_discriminator.shapes[0][0], (6, 3, 4, 32, 32))
        active.loss.total.backward()
        self.assertIsNone(image_discriminator.scale.grad)
        self.assertIsNone(video_discriminator.scale.grad)

        discriminator = stage.discriminator_step(
            batch, active.core.model.rgb, global_step=10
        )
        self.assertTrue(discriminator.active)
        self.assertTrue(torch.isfinite(discriminator.total))

    def test_adam_betas_and_schedulers_are_explicit(self) -> None:
        core = StereoTokenizerTrainingCore(
            StereoTokenizer(tiny_config()), loss_config(perceptual_weight=0.0)
        )
        stage = StereoTokenizerTrainingStage(
            core,
            StereoAdversarialConfig(
                enabled=False,
                start_step=0,
                image_weight=0.0,
                video_weight=0.0,
                feature_matching_weight=0.0,
            ),
        )
        bundle = stage.build_optimizers(
            StereoOptimizerConfig(
                autoencoder_lr=1e-4,
                discriminator_lr=1e-4,
                autoencoder_min_lr=1e-5,
                discriminator_min_lr=1e-5,
                autoencoder_warmup_start_lr=1e-6,
                discriminator_warmup_start_lr=1e-6,
                autoencoder_warmup_steps=10,
                discriminator_warmup_steps=10,
                total_steps=100,
            )
        )
        self.assertEqual(bundle.autoencoder.param_groups[0]["betas"], (0.5, 0.9))
        self.assertIsNone(bundle.discriminator)
        self.assertIsNone(bundle.discriminator_scheduler)

    def test_validation_policy_rejects_missing_split(self) -> None:
        disabled = StereoValidationPolicy(
            enabled=False, manifest_path=None, split_name=None
        )
        self.assertFalse(disabled.should_run(epoch_end=True))
        with self.assertRaisesRegex(ValueError, "requires a manifest"):
            StereoValidationPolicy(
                enabled=True, manifest_path=None, split_name=None
            ).validate()


if __name__ == "__main__":
    unittest.main()
