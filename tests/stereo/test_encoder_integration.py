import unittest

import torch

from stereo_tokenizer.model import StereoEncoder
from stereo_tokenizer.modules.attention import Attention


class StructuredStereoEncoderTest(unittest.TestCase):
    def test_rope_cache_rebuilds_after_device_migration(self) -> None:
        attention = Attention(
            dim=8,
            dim_head=8,
            heads=1,
            spatial_pos="rope",
        ).eval()
        attention.freqs_cis = torch.empty(
            (4, 4), dtype=torch.complex64, device="meta"
        )

        output = attention(torch.randn(1, 4, 8))

        self.assertEqual(output.device.type, "cpu")
        self.assertEqual(attention.freqs_cis.device.type, "cpu")

    def _encoder(self) -> StereoEncoder:
        return StereoEncoder(
            image_size=32,
            image_channel=3,
            block="tt",
            window_size=2,
            spatial_pos="rope",
            patch_embed="linear",
            patch_size=8,
            temporal_patch_size=4,
            spatial_depth=2,
            temporal_depth=1,
            causal_in_peg=True,
            dim=32,
            dim_head=8,
            heads=4,
            stereo_num_views=3,
            stereo_num_frames=4,
            stereo_search_radii=(1, 1, 1),
            stereo_search_direction="left",
        )

    def assert_nonzero_finite_gradient(self, parameter: torch.Tensor) -> None:
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_stereo_encoder_returns_one_slot_per_view(self) -> None:
        encoder = self._encoder().eval()
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        output = encoder.forward_stereo(
            video,
            eye_mode="stereo",
            temporal_mode="four_frame",
        )
        self.assertEqual(output.features.shape, (3, 32, 1, 4, 4))
        self.assertIsNotNone(output.fusion)

    def test_mono_mode_uses_one_view_one_eye_without_fusion(self) -> None:
        encoder = self._encoder().eval()
        video = torch.randn(2, 1, 1, 3, 4, 32, 32)
        output = encoder.forward_stereo(
            video, eye_mode="mono", temporal_mode="four_frame"
        )
        self.assertEqual(output.features.shape, (2, 32, 1, 4, 4))
        self.assertIsNone(output.fusion)

    def test_spatial_encoder_keeps_frames_isolated(self) -> None:
        torch.manual_seed(7)
        encoder = self._encoder().eval()
        frames = torch.randn(4, 3, 32, 32)
        changed = frames.clone()
        changed[3, :, :8, :8] += torch.randn_like(changed[3, :, :8, :8])

        with torch.no_grad():
            baseline = encoder._encode_stereo_frames(frames)
            perturbed = encoder._encode_stereo_frames(changed)

        # Spatial Transformer 将每帧视为独立样本，因此第 4 帧扰动不能影响前三帧。
        torch.testing.assert_close(baseline[:3], perturbed[:3], rtol=0, atol=0)
        self.assertGreater((baseline[3] - perturbed[3]).abs().max().item(), 1e-6)

    def test_temporal_attention_is_bidirectional_and_precedes_sampler(self) -> None:
        torch.manual_seed(11)
        encoder = self._encoder().eval()
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        changed = video.clone()
        changed[:, :, 0, :, 3, :8, :8] += torch.randn_like(
            changed[:, :, 0, :, 3, :8, :8]
        )

        order = []
        temporal_outputs = []
        temporal_lengths = []
        sampler_widths = []

        def record_temporal(_module, inputs, output):
            order.append("temporal")
            temporal_lengths.append(inputs[0].shape[1])
            temporal_outputs.append(output.detach().clone())

        def record_sampler(_module, inputs, _output):
            order.append("sampler")
            sampler_widths.append(inputs[0].shape[-1])

        handles = [
            encoder.enc_spatial_transformer.register_forward_hook(
                lambda *_: order.append("spatial")
            ),
            encoder.stereo_fusion.register_forward_hook(
                lambda *_: order.append("fusion")
            ),
            encoder.enc_temporal_transformer.register_forward_hook(
                record_temporal
            ),
            encoder.stereo_temporal_projection.register_forward_hook(
                record_sampler
            ),
        ]
        try:
            with torch.no_grad():
                encoder.forward_stereo(
                    video,
                    eye_mode="stereo",
                    temporal_mode="four_frame",
                )
            self.assertEqual(order, ["spatial", "fusion", "temporal", "sampler"])
            self.assertEqual(temporal_lengths, [4])
            self.assertEqual(sampler_widths, [4 * encoder.stereo_embedding_dim])

            order.clear()
            temporal_outputs.clear()
            mono_video = video[:, :1, :1]
            mono_changed = changed[:, :1, :1]
            with torch.no_grad():
                encoder.forward_stereo(
                    mono_video, eye_mode="mono", temporal_mode="four_frame"
                )
                baseline_temporal = temporal_outputs[-1]
                encoder.forward_stereo(
                    mono_changed, eye_mode="mono", temporal_mode="four_frame"
                )
                changed_temporal = temporal_outputs[-1]
        finally:
            for handle in handles:
                handle.remove()

        # 只改变第 4 帧后，第 1 帧的 Sampler 前特征也应改变，证明注意力是双向的。
        first_frame_delta = (
            baseline_temporal[:, 0] - changed_temporal[:, 0]
        ).abs().max()
        self.assertGreater(first_frame_delta.item(), 1e-6)

    def test_views_remain_isolated_after_temporal_encoding(self) -> None:
        torch.manual_seed(13)
        encoder = self._encoder().eval()
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        changed = video.clone()
        changed[:, 0, 0] += torch.randn_like(changed[:, 0, 0])

        with torch.no_grad():
            baseline = encoder.forward_stereo(
                video, eye_mode="stereo", temporal_mode="four_frame"
            ).features
            perturbed = encoder.forward_stereo(
                changed, eye_mode="stereo", temporal_mode="four_frame"
            ).features

        torch.testing.assert_close(baseline[1:], perturbed[1:], rtol=0, atol=0)
        self.assertGreater((baseline[0] - perturbed[0]).abs().max().item(), 1e-6)

    def test_temporal_projection_and_fusion_receive_gradients(self) -> None:
        encoder = self._encoder().train()
        with torch.no_grad():
            encoder.stereo_fusion.alpha.fill_(1.0)
        video = torch.randn(1, 3, 2, 3, 4, 32, 32)
        output = encoder.forward_stereo(
            video,
            eye_mode="stereo",
            temporal_mode="four_frame",
        )
        output.features.square().mean().backward()

        self.assert_nonzero_finite_gradient(
            encoder.enc_temporal_transformer.layers[0][1].to_q.weight
        )
        self.assert_nonzero_finite_gradient(
            encoder.stereo_temporal_projection[1].weight
        )
        self.assert_nonzero_finite_gradient(encoder.stereo_fusion.to_v.weight)

    def test_single_frame_skips_four_frame_temporal_modules(self) -> None:
        encoder = self._encoder().eval()
        video = torch.randn(1, 3, 2, 3, 1, 32, 32)
        calls = []
        handles = [
            encoder.stereo_fusion.register_forward_hook(
                lambda *_: calls.append("fusion")
            ),
            encoder.enc_temporal_transformer.register_forward_hook(
                lambda *_: calls.append("temporal")
            ),
            encoder.stereo_temporal_projection.register_forward_hook(
                lambda *_: calls.append("four_sampler")
            ),
            encoder.single_frame_projection.register_forward_hook(
                lambda *_: calls.append("single_projection")
            ),
        ]
        try:
            output = encoder.forward_stereo(
                video,
                eye_mode="stereo",
                temporal_mode="single_frame",
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(output.features.shape, (3, 32, 1, 4, 4))
        self.assertEqual(calls, ["fusion", "single_projection"])

    def test_single_frame_shared_and_private_paths_receive_gradients(self) -> None:
        encoder = self._encoder().train()
        with torch.no_grad():
            encoder.stereo_fusion.alpha.fill_(1.0)
        video = torch.randn(1, 3, 2, 3, 1, 32, 32)
        output = encoder.forward_stereo(
            video,
            eye_mode="stereo",
            temporal_mode="single_frame",
        )
        output.features.square().mean().backward()

        self.assert_nonzero_finite_gradient(
            encoder.single_frame_projection[1].weight
        )
        self.assert_nonzero_finite_gradient(encoder.stereo_fusion.to_v.weight)
        self.assertIsNone(
            encoder.enc_temporal_transformer.layers[0][1].to_q.weight.grad
        )

    def test_temporal_mode_requires_matching_input_length(self) -> None:
        encoder = self._encoder()
        with self.assertRaisesRegex(ValueError, "requires T=1"):
            encoder.forward_stereo(
                torch.randn(1, 3, 2, 3, 4, 32, 32),
                eye_mode="stereo",
                temporal_mode="single_frame",
            )


if __name__ == "__main__":
    unittest.main()
