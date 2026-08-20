import unittest

import torch

from OmniTokenizer.stereo.fusion import StereoFusion


class StereoFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = StereoFusion(
            dim=8,
            heads=2,
            head_dim=4,
            search_radii=(0, 1, 2),
            search_direction="left",
        )

    def test_per_view_and_boundary_mask(self) -> None:
        candidate_x, valid = self.fusion._candidate_layout(
            width=4, device=torch.device("cpu")
        )
        torch.testing.assert_close(
            candidate_x,
            torch.tensor(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [2, 1, 0],
                    [3, 2, 1],
                ]
            ),
        )
        torch.testing.assert_close(
            valid.sum(dim=-1),
            torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 2, 2, 2],
                    [1, 2, 3, 3],
                ]
            ),
        )

    def test_zero_gate_is_exact_left_identity(self) -> None:
        left = torch.randn(2, 3, 4, 2, 4, 8)
        right = torch.randn_like(left)
        output = self.fusion(left, right)
        torch.testing.assert_close(output.features, left, rtol=0.0, atol=0.0)

        expanded_mask = output.valid_mask[None, :, None, None, :, None, :]
        invalid_attention = output.attention.masked_select(~expanded_mask)
        torch.testing.assert_close(
            invalid_attention, torch.zeros_like(invalid_attention)
        )
        self.assertTrue(torch.isfinite(output.confidence).all())
        self.assertTrue((output.confidence >= 0).all())
        self.assertTrue((output.confidence <= 1).all())

        # At x=0, only the zero-offset candidate is valid for every view.
        torch.testing.assert_close(
            output.confidence[..., 0],
            torch.ones_like(output.confidence[..., 0]),
        )

    def test_search_radius_must_fit_feature_width(self) -> None:
        left = torch.randn(1, 3, 4, 1, 2, 8)
        with self.assertRaisesRegex(ValueError, "smaller than feature width"):
            self.fusion(left, left)

    def test_entropy_gate_is_detached_but_matching_path_trains(self) -> None:
        with torch.no_grad():
            self.fusion.alpha.fill_(1.0)
        left = torch.randn(1, 3, 4, 1, 4, 8, requires_grad=True)
        right = torch.randn_like(left, requires_grad=True)
        output = self.fusion(left, right)
        output.confidence.retain_grad()
        output.features.sum().backward()
        self.assertIsNone(output.confidence.grad)
        self.assertIsNotNone(self.fusion.to_q.weight.grad)
        self.assertIsNotNone(self.fusion.to_v.weight.grad)


if __name__ == "__main__":
    unittest.main()
