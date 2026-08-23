import unittest

import torch

from stereo_tokenizer.model import StereoDecoder


class StructuredStereoDecoderTest(unittest.TestCase):
    def _decoder(self) -> StereoDecoder:
        return StereoDecoder(
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
            causal_in_peg=True,
            dim=32,
            dim_head=8,
            heads=4,
            stereo_num_views=3,
            stereo_num_frames=4,
            stereo_disparity_scale=(1.0, 2.0, 3.0),
            stereo_disparity_bias=-2.572,
        )

    def assert_nonzero_finite_gradient(self, parameter: torch.Tensor) -> None:
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_one_slot_decodes_four_frames_per_view(self) -> None:
        decoder = self._decoder().eval()
        output = decoder.forward_stereo(
            torch.randn(3, 32, 1, 4, 4),
            temporal_mode="four_frame",
        )
        self.assertEqual(output.rgb.shape, (1, 3, 3, 4, 32, 32))
        self.assertEqual(output.disparity.shape, (1, 3, 1, 4, 32, 32))
        self.assertTrue((output.disparity > 0).all())

    def test_temporal_expansion_attention_and_spatial_order(self) -> None:
        decoder = self._decoder().eval()
        order = []
        temporal_lengths = []
        spatial_shapes = []
        head_times = []

        def spatial_pre_hook(_module, inputs, kwargs):
            order.append("spatial")
            spatial_shapes.append(kwargs["video_shape"])

        def temporal_hook(_module, inputs, _output):
            order.append("temporal")
            temporal_lengths.append(inputs[0].shape[1])

        def rgb_head_hook(_module, inputs, _output):
            order.append("rgb_head")
            head_times.append(inputs[0].shape[1])

        handles = [
            decoder.stereo_temporal_expansion.register_forward_hook(
                lambda *_: order.append("temporal_expand")
            ),
            decoder.dec_temporal_transformer.register_forward_hook(temporal_hook),
            decoder.dec_spatial_transformer.register_forward_pre_hook(
                spatial_pre_hook, with_kwargs=True
            ),
            decoder.stereo_rgb_head.register_forward_hook(rgb_head_hook),
        ]
        try:
            with torch.no_grad():
                decoder.forward_stereo(
                    torch.randn(3, 32, 1, 4, 4),
                    temporal_mode="four_frame",
                )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(order, ["temporal_expand", "temporal", "spatial", "rgb_head"])
        self.assertEqual(temporal_lengths, [4])
        self.assertEqual(spatial_shapes, [(12, 1, 4, 4)])
        self.assertEqual(head_times, [4])

    def test_disparity_head_uses_resolved_bias(self) -> None:
        decoder = self._decoder()
        expected = torch.full_like(decoder.stereo_disparity_head.bias, -2.572)
        torch.testing.assert_close(decoder.stereo_disparity_head.bias, expected)

    def test_temporal_and_spatial_decoder_paths_receive_gradients(self) -> None:
        decoder = self._decoder().train()
        tokens = torch.randn(3, 32, 1, 4, 4, requires_grad=True)
        output = decoder.forward_stereo(tokens, temporal_mode="four_frame")
        loss = output.rgb.square().mean() + output.normalized_disparity.square().mean()
        loss.backward()

        self.assert_nonzero_finite_gradient(
            decoder.stereo_temporal_expansion[1].weight
        )
        self.assert_nonzero_finite_gradient(
            decoder.dec_temporal_transformer.layers[0][1].to_q.weight
        )
        self.assert_nonzero_finite_gradient(
            decoder.dec_spatial_transformer.layers[0][1].to_q.weight
        )
        self.assert_nonzero_finite_gradient(decoder.stereo_rgb_head.weight)
        self.assert_nonzero_finite_gradient(decoder.stereo_disparity_head.weight)
        self.assert_nonzero_finite_gradient(tokens)

    def test_single_frame_skips_four_frame_temporal_modules(self) -> None:
        decoder = self._decoder().eval()
        calls = []
        handles = [
            decoder.single_frame_expansion.register_forward_hook(
                lambda *_: calls.append("single_expand")
            ),
            decoder.stereo_temporal_expansion.register_forward_hook(
                lambda *_: calls.append("four_expand")
            ),
            decoder.dec_temporal_transformer.register_forward_hook(
                lambda *_: calls.append("temporal")
            ),
            decoder.dec_spatial_transformer.register_forward_hook(
                lambda *_: calls.append("spatial")
            ),
        ]
        try:
            output = decoder.forward_stereo(
                torch.randn(3, 32, 1, 4, 4),
                temporal_mode="single_frame",
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(calls, ["single_expand", "spatial"])
        self.assertEqual(output.rgb.shape, (1, 3, 3, 1, 32, 32))
        self.assertEqual(output.disparity.shape, (1, 3, 1, 1, 32, 32))

    def test_single_frame_shared_and_private_paths_receive_gradients(self) -> None:
        decoder = self._decoder().train()
        tokens = torch.randn(3, 32, 1, 4, 4, requires_grad=True)
        output = decoder.forward_stereo(tokens, temporal_mode="single_frame")
        loss = output.rgb.square().mean() + output.normalized_disparity.square().mean()
        loss.backward()

        self.assert_nonzero_finite_gradient(
            decoder.single_frame_expansion[1].weight
        )
        self.assert_nonzero_finite_gradient(
            decoder.dec_spatial_transformer.layers[0][1].to_q.weight
        )
        self.assert_nonzero_finite_gradient(decoder.stereo_rgb_head.weight)
        self.assert_nonzero_finite_gradient(decoder.stereo_disparity_head.weight)
        self.assertIsNone(
            decoder.dec_temporal_transformer.layers[0][1].to_q.weight.grad
        )


if __name__ == "__main__":
    unittest.main()
