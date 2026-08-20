import copy
import unittest

import torch

from OmniTokenizer.stereo.model import StereoTokenizer, StereoTokenizerConfig
from OmniTokenizer.stereo.training import (
    StereoTokenizerTrainingCore,
    StereoTrainingLossConfig,
)


def tiny_config() -> StereoTokenizerConfig:
    return StereoTokenizerConfig(
        search_radii=(0, 1, 2),
        search_direction="left",
        disparity_scale=(1.0, 2.0, 3.0),
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


class StereoTokenizerShapeTest(unittest.TestCase):
    def test_decoder_unpatch_preserves_element_order(self) -> None:
        config = tiny_config()
        decoder = StereoTokenizer(config).decoder
        grid_height = config.image_size[0] // config.patch_size[0]
        grid_width = config.image_size[1] // config.patch_size[1]
        patch_width = (
            config.image_channels
            * config.num_frames
            * config.patch_size[0]
            * config.patch_size[1]
        )
        patches = torch.arange(
            config.num_views * grid_height * grid_width * patch_width
        ).reshape(1, config.num_views, 1, grid_height, grid_width, patch_width)
        pixels = decoder._unpatch(patches, config.image_channels)
        roundtrip = (
            pixels.reshape(
                1,
                config.num_views,
                config.image_channels,
                config.num_frames,
                grid_height,
                config.patch_size[0],
                grid_width,
                config.patch_size[1],
            )
            .permute(0, 1, 4, 6, 2, 3, 5, 7)
            .contiguous()
            .reshape_as(patches)
        )
        torch.testing.assert_close(roundtrip, patches)

    def test_stereo_forward_contract(self) -> None:
        model = StereoTokenizer(tiny_config()).eval()
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        with torch.no_grad():
            output = model(video, mode="stereo")

        self.assertEqual(output.latent.shape, (1, 3, 48, 1, 4, 4))
        self.assertEqual(output.rgb.shape, (1, 3, 3, 4, 32, 32))
        self.assertEqual(output.disparity.shape, (1, 3, 1, 4, 32, 32))
        self.assertEqual(
            output.normalized_disparity.shape, (1, 3, 1, 4, 32, 32)
        )
        self.assertTrue((output.disparity > 0).all())
        self.assertIsNotNone(output.fusion)

    def test_mono_mode_uses_only_left_eye(self) -> None:
        model = StereoTokenizer(tiny_config()).eval()
        left = torch.randn(1, 3, 1, 3, 4, 32, 32)
        two_eye = torch.cat((left, torch.randn_like(left)), dim=2)
        with torch.no_grad():
            one_eye_output = model.encode(left, mode="mono")
            two_eye_output = model.encode(two_eye, mode="mono")
        torch.testing.assert_close(
            one_eye_output.latent, two_eye_output.latent, rtol=0.0, atol=0.0
        )
        self.assertIsNone(one_eye_output.fusion)

    def test_new_class_rejects_non_four_frame_input(self) -> None:
        model = StereoTokenizer(tiny_config())
        video = torch.randn(1, 3, 2, 3, 1, 32, 32)
        with self.assertRaisesRegex(ValueError, "expected T=4"):
            model(video, mode="stereo")

    def test_eval_uses_posterior_mean(self) -> None:
        model = StereoTokenizer(tiny_config()).eval()
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        with torch.no_grad():
            first = model.encode(video, mode="stereo")
            second = model.encode(video, mode="stereo")
        torch.testing.assert_close(first.latent, first.posterior.mean)
        torch.testing.assert_close(first.latent, second.latent)

    def test_strict_checkpoint_roundtrip_is_deterministic(self) -> None:
        config = tiny_config()
        model = StereoTokenizer(config).eval()
        restored = StereoTokenizer(config).eval()
        state = copy.deepcopy(model.state_dict())
        restored.load_state_dict(state, strict=True)

        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        with torch.no_grad():
            expected = model(video, mode="stereo")
            actual = restored(video, mode="stereo")
        torch.testing.assert_close(expected.latent, actual.latent)
        torch.testing.assert_close(expected.rgb, actual.rgb)
        torch.testing.assert_close(expected.disparity, actual.disparity)

    def test_backward_reaches_temporal_reducer_and_gate(self) -> None:
        model = StereoTokenizer(tiny_config()).train()
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        output = model(video, mode="stereo", sample_posterior=False)
        loss = output.rgb.mean() + output.normalized_disparity.mean()
        loss.backward()
        self.assertIsNotNone(model.temporal_encoder.reducer[1].weight.grad)
        self.assertIsNotNone(model.stereo_fusion.alpha.grad)

    def test_kl_warmup_is_explicit(self) -> None:
        model = StereoTokenizer(tiny_config())
        loss_config = StereoTrainingLossConfig(
            mode="stereo",
            rgb_weight=1.0,
            disparity_weight=2.0,
            gradient_weight=0.1,
            kl_target_weight=0.01,
            perceptual_weight=0.5,
            kl_warmup_steps=100,
            smooth_l1_beta=1.0,
            geometry_gradient_scale_px=16.0,
            rgb_loss_type="l1",
        )
        core = StereoTokenizerTrainingCore(model, loss_config)
        self.assertEqual(core.kl_weight_at(0), 0.0)
        self.assertEqual(core.kl_weight_at(50), 0.005)
        self.assertEqual(core.kl_weight_at(100), 0.01)
        self.assertEqual(core.kl_weight_at(200), 0.01)

    def test_disparity_head_uses_resolved_raw_bias(self) -> None:
        model = StereoTokenizer(tiny_config())
        expected = torch.full_like(model.decoder.disparity_head.bias, -2.572)
        torch.testing.assert_close(model.decoder.disparity_head.bias, expected)

    def test_disparity_head_rejects_non_finite_bias(self) -> None:
        config = copy.copy(tiny_config())
        object.__setattr__(config, "disparity_head_bias", float("nan"))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
