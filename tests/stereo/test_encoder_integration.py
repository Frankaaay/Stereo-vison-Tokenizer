import unittest

import torch

from OmniTokenizer.omnitokenizer import OmniTokenizer_Encoder


class StructuredStereoEncoderTest(unittest.TestCase):
    def _encoder(self) -> OmniTokenizer_Encoder:
        return OmniTokenizer_Encoder(
            image_size=32,
            image_channel=3,
            norm_type="group",
            block="tt",
            window_size=2,
            spatial_pos="rope",
            patch_embed="linear",
            patch_size=8,
            temporal_patch_size=4,
            spatial_depth=2,
            temporal_depth=1,
            causal_in_temporal_transformer=True,
            causal_in_peg=True,
            dim=32,
            dim_head=8,
            heads=4,
            stereo_num_views=3,
            stereo_num_frames=4,
            stereo_search_radii=(1, 1, 1),
            stereo_search_direction="left",
        )

    def test_stereo_encoder_returns_one_slot_per_view(self) -> None:
        encoder = self._encoder().eval()
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        output = encoder.forward_stereo(video, mode="stereo")
        self.assertEqual(output.features.shape, (3, 32, 1, 4, 4))
        self.assertIsNotNone(output.fusion)

    def test_mono_mode_ignores_right_eye(self) -> None:
        encoder = self._encoder().eval()
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        changed_right = video.clone()
        changed_right[:, :, 1].normal_(mean=100.0, std=1.0)
        left_only = encoder.forward_stereo(video, mode="mono")
        changed = encoder.forward_stereo(changed_right, mode="mono")
        torch.testing.assert_close(left_only.features, changed.features)
        self.assertIsNone(left_only.fusion)

    def test_temporal_projection_and_fusion_receive_gradients(self) -> None:
        encoder = self._encoder().train()
        with torch.no_grad():
            encoder.stereo_fusion.alpha.fill_(1.0)
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        output = encoder.forward_stereo(video, mode="stereo")
        output.features.mean().backward()
        self.assertIsNotNone(
            encoder.stereo_temporal_projection[1].weight.grad
        )
        self.assertIsNotNone(encoder.stereo_fusion.to_v.weight.grad)


if __name__ == "__main__":
    unittest.main()
