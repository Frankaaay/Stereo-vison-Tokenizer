import unittest

import torch

from OmniTokenizer.modules.stereo_geometry import disparity_to_depth
from OmniTokenizer.stereo.losses import (
    masked_disparity_gradient_loss,
    masked_smooth_l1_disparity_loss,
    posterior_kl_loss,
)
from OmniTokenizer.stereo.training import StereoTrainingLossConfig


class _PosteriorStub:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def kl(self) -> torch.Tensor:
        return self.values


class StereoLossTest(unittest.TestCase):
    def test_disparity_is_normalized_per_view_before_average(self) -> None:
        prediction = torch.zeros(1, 3, 1, 1, 2, 3)
        target = torch.stack(
            (
                torch.ones(1, 1, 1, 2, 3),
                torch.full((1, 1, 1, 2, 3), 2.0),
                torch.full((1, 1, 1, 2, 3), 3.0),
            ),
            dim=1,
        )
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[:, 0, :, :, :, 1:] = False
        valid[:, 1, :, :, :, 2:] = False

        result = masked_smooth_l1_disparity_loss(
            prediction, target, valid, beta=1.0
        )
        torch.testing.assert_close(
            result.per_view, torch.tensor((0.5, 1.5, 2.5))
        )
        torch.testing.assert_close(result.loss, torch.tensor(1.5))
        torch.testing.assert_close(result.valid_count, torch.tensor((2, 4, 6)))

    def test_empty_view_fails_closed(self) -> None:
        prediction = torch.zeros(1, 3, 1, 1, 2, 2)
        target = torch.zeros_like(prediction)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[:, 2] = False
        with self.assertRaisesRegex(ValueError, "empty view indices"):
            masked_smooth_l1_disparity_loss(
                prediction, target, valid, beta=1.0
            )

    def test_gradient_loss_uses_only_valid_neighbor_pairs(self) -> None:
        prediction = torch.zeros(1, 3, 1, 1, 2, 3)
        target = torch.zeros_like(prediction)
        target[..., 1:] = 1.0
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[..., -1] = False
        result = masked_disparity_gradient_loss(
            prediction, target, valid, scale_px=16.0
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue((result.valid_count > 0).all())

    def test_gradient_loss_uses_independent_pixel_scale(self) -> None:
        prediction = torch.zeros(1, 3, 1, 1, 2, 2)
        target = torch.zeros_like(prediction)
        target[..., 1] = 16.0
        valid = torch.ones_like(target, dtype=torch.bool)
        result = masked_disparity_gradient_loss(
            prediction, target, valid, scale_px=16.0
        )
        torch.testing.assert_close(result.loss, torch.tensor(0.5))

    def test_kl_sums_are_averaged_across_batch_and_view(self) -> None:
        values = torch.tensor(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
        torch.testing.assert_close(
            posterior_kl_loss(_PosteriorStub(values)), torch.tensor(3.5)
        )


class GeometryTest(unittest.TestCase):
    def test_metric_depth_and_mask(self) -> None:
        disparity = torch.full((1, 3, 1, 2, 2, 2), 10.0)
        disparity[..., 0, 0] = 0.0
        fx = torch.tensor(((100.0, 200.0, 300.0),))
        baseline = torch.tensor(((0.1, 0.1, 0.1),))
        output = disparity_to_depth(disparity, fx, baseline)
        self.assertFalse(output.valid_mask[..., 0, 0].any())
        torch.testing.assert_close(
            output.depth[0, :, 0, :, 0, 1],
            torch.tensor(((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))),
        )

    def test_loss_config_has_no_implicit_weights(self) -> None:
        with self.assertRaises(TypeError):
            StereoTrainingLossConfig(mode="stereo")


if __name__ == "__main__":
    unittest.main()
