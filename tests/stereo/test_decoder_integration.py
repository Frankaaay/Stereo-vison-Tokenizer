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
            causal_in_temporal_transformer=True,
            causal_in_peg=True,
            dim=32,
            dim_head=8,
            heads=4,
            stereo_num_views=3,
            stereo_num_frames=4,
            stereo_disparity_scale=(1.0, 2.0, 3.0),
            stereo_disparity_bias=-2.572,
        )

    def test_one_slot_decodes_four_frames_per_view(self) -> None:
        decoder = self._decoder().eval()
        output = decoder.forward_stereo(torch.randn(3, 32, 1, 4, 4))
        self.assertEqual(output.rgb.shape, (1, 3, 3, 4, 32, 32))
        self.assertEqual(output.disparity.shape, (1, 3, 1, 4, 32, 32))
        self.assertTrue((output.disparity > 0).all())

    def test_disparity_head_uses_resolved_bias(self) -> None:
        decoder = self._decoder()
        expected = torch.full_like(decoder.stereo_disparity_head.bias, -2.572)
        torch.testing.assert_close(decoder.stereo_disparity_head.bias, expected)

    def test_both_heads_backpropagate_through_shared_decoder(self) -> None:
        decoder = self._decoder().train()
        tokens = torch.randn(3, 32, 1, 4, 4, requires_grad=True)
        output = decoder.forward_stereo(tokens)
        (output.rgb.mean() + output.normalized_disparity.mean()).backward()
        self.assertIsNotNone(decoder.stereo_rgb_head.weight.grad)
        self.assertIsNotNone(decoder.stereo_disparity_head.weight.grad)
        self.assertIsNotNone(tokens.grad)


if __name__ == "__main__":
    unittest.main()
