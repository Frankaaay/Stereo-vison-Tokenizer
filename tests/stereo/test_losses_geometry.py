import unittest

import torch

from stereo_tokenizer.modules.relative_depth import (
    center_relative_log_depth,
    relative_prediction_from_raw,
    relative_target_from_da3,
    relative_target_from_foundation_stereo,
)
from stereo_tokenizer.modules.stereo_losses import (
    StereoReconstructionKLLoss,
    masked_relative_gradient_loss,
    masked_smooth_l1_relative_depth_loss,
    posterior_kl_loss,
)


class _PosteriorStub:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def kl(self) -> torch.Tensor:
        return self.values


class StereoLossTest(unittest.TestCase):
    def test_existing_core_objective_accepts_single_frame_tensors(self) -> None:
        objective = StereoReconstructionKLLoss(
            rgb_weight=1.0,
            relative_depth_weight=1.0,
            relative_gradient_weight=1.0,
            kl_weight=1e-6,
            smooth_l1_beta=1.0,
            rgb_loss_type="l1",
        )
        rgb_prediction = torch.randn(
            1, 3, 3, 1, 4, 4, requires_grad=True
        )
        rgb_target = torch.zeros_like(rgb_prediction)
        relative_prediction = torch.ones(
            1, 3, 1, 1, 4, 4, requires_grad=True
        )
        relative_target = torch.full_like(relative_prediction, 0.5)
        valid = torch.ones_like(relative_prediction, dtype=torch.bool)

        result = objective(
            rgb_prediction=rgb_prediction,
            rgb_target=rgb_target,
            relative_depth_prediction=relative_prediction,
            relative_depth_target=relative_target,
            valid_mask=valid,
            posterior=_PosteriorStub(torch.zeros(1, 3)),
        )
        result.total.backward()

        self.assertTrue(torch.isfinite(result.total))
        self.assertTrue(torch.isfinite(rgb_prediction.grad).all())
        self.assertTrue(torch.isfinite(relative_prediction.grad).all())

    def test_relative_depth_is_normalized_per_sample_view_before_average(self) -> None:
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

        result = masked_smooth_l1_relative_depth_loss(
            prediction, target, valid, beta=1.0
        )
        torch.testing.assert_close(
            result.per_view, torch.tensor((0.5, 1.5, 2.5))
        )
        torch.testing.assert_close(result.loss, torch.tensor(1.5))
        torch.testing.assert_close(result.valid_count, torch.tensor((2, 4, 6)))

    def test_empty_views_are_ignored_with_sample_equal_weight(self) -> None:
        prediction = torch.zeros(2, 3, 1, 1, 2, 2)
        target = torch.empty_like(prediction)
        target[0, 0] = 1.0
        target[0, 1] = 2.0
        target[0, 2] = 100.0
        target[1, 0] = 100.0
        target[1, 1] = 100.0
        target[1, 2] = 3.0
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[0, 2] = False
        valid[1, :2] = False
        result = masked_smooth_l1_relative_depth_loss(
            prediction, target, valid, beta=1.0
        )
        torch.testing.assert_close(result.loss, torch.tensor(1.75))
        torch.testing.assert_close(
            result.per_view, torch.tensor((0.5, 1.5, 2.5))
        )
        torch.testing.assert_close(
            result.supervised_sample_count, torch.tensor((1, 1, 1))
        )

    def test_sample_with_all_views_empty_fails_closed(self) -> None:
        prediction = torch.zeros(2, 3, 1, 1, 2, 2)
        target = torch.zeros_like(prediction)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[1] = False
        with self.assertRaisesRegex(ValueError, r"empty \[B\].*\[1\]"):
            masked_smooth_l1_relative_depth_loss(
                prediction, target, valid, beta=1.0
            )

    def test_invalid_nonfinite_disparity_cannot_poison_smooth_l1_gradient(
        self,
    ) -> None:
        prediction = torch.ones(1, 3, 1, 1, 2, 2, requires_grad=True)
        target = torch.zeros_like(prediction)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        target[..., 0, 0] = torch.nan
        valid[..., 0, 0] = False

        result = masked_smooth_l1_relative_depth_loss(
            prediction, target, valid, beta=1.0
        )
        result.loss.backward()

        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_gradient_loss_uses_only_valid_neighbor_pairs(self) -> None:
        prediction = torch.zeros(1, 3, 1, 1, 2, 3)
        target = torch.zeros_like(prediction)
        target[..., 1:] = 1.0
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[..., -1] = False
        result = masked_relative_gradient_loss(
            prediction, target, valid, beta=1.0
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue((result.valid_count > 0).all())

    def test_gradient_loss_uses_equal_x_y_smooth_l1(self) -> None:
        prediction = torch.zeros(1, 3, 1, 1, 2, 2)
        target = torch.zeros_like(prediction)
        target[..., 1] = 1.0
        valid = torch.ones_like(target, dtype=torch.bool)
        result = masked_relative_gradient_loss(
            prediction, target, valid, beta=1.0
        )
        torch.testing.assert_close(result.loss, torch.tensor(0.25))

    def test_gradient_loss_ignores_empty_sample_view(self) -> None:
        prediction = torch.zeros(1, 3, 1, 1, 2, 2)
        target = torch.ones_like(prediction)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[:, 2] = False
        result = masked_relative_gradient_loss(
            prediction, target, valid, beta=1.0
        )
        self.assertTrue(torch.isfinite(result.loss))
        torch.testing.assert_close(
            result.supervised_sample_count, torch.tensor((1, 1, 0))
        )

    def test_four_frames_do_not_receive_four_times_single_frame_weight(self) -> None:
        single_prediction = torch.tensor(
            [[[[[[0.0, 1.0], [2.0, 3.0]]]]]], dtype=torch.float32
        )
        single_target = torch.zeros_like(single_prediction)
        single_valid = torch.ones_like(single_prediction, dtype=torch.bool)
        four_prediction = single_prediction.repeat(1, 1, 1, 4, 1, 1)
        four_target = single_target.repeat(1, 1, 1, 4, 1, 1)
        four_valid = single_valid.repeat(1, 1, 1, 4, 1, 1)
        single = masked_smooth_l1_relative_depth_loss(
            single_prediction, single_target, single_valid, beta=1.0
        )
        four = masked_smooth_l1_relative_depth_loss(
            four_prediction, four_target, four_valid, beta=1.0
        )
        torch.testing.assert_close(single.loss, four.loss)

    def test_invalid_nonfinite_disparity_is_sanitized_before_differences(
        self,
    ) -> None:
        prediction = torch.ones(1, 3, 1, 1, 2, 2, requires_grad=True)
        target = torch.zeros_like(prediction)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        target[..., 0, 0] = torch.nan
        valid[..., 0, 0] = False

        result = masked_relative_gradient_loss(
            prediction, target, valid, beta=1.0
        )
        result.loss.backward()

        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_kl_sums_are_averaged_across_batch_and_view(self) -> None:
        values = torch.tensor(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
        torch.testing.assert_close(
            posterior_kl_loss(_PosteriorStub(values)), torch.tensor(3.5)
        )


class RelativeDepthTargetTest(unittest.TestCase):
    def test_center_ignores_empty_view(self) -> None:
        log_depth = torch.tensor((1.0, 3.0, 100.0)).reshape(
            1, 3, 1, 1, 1, 1
        )
        valid = torch.ones_like(log_depth, dtype=torch.bool)
        valid[:, 2] = False
        relative, center = center_relative_log_depth(
            log_depth, valid, require_all_finite=True
        )
        torch.testing.assert_close(center.flatten(), torch.tensor((2.0,)))
        torch.testing.assert_close(
            relative.flatten(), torch.tensor((-1.0, 1.0, 98.0))
        )

    def test_center_rejects_sample_with_all_views_empty(self) -> None:
        log_depth = torch.zeros(2, 3, 1, 1, 1, 1)
        valid = torch.ones_like(log_depth, dtype=torch.bool)
        valid[1] = False
        with self.assertRaisesRegex(ValueError, r"empty \[B\].*\[1\]"):
            center_relative_log_depth(
                log_depth, valid, require_all_finite=True
            )

    def test_da3_target_is_invariant_to_positive_scale(self) -> None:
        depth = torch.tensor(
            [[[[[[1.0, 2.0], [4.0, 8.0]]]]]], dtype=torch.float32
        )
        valid = torch.ones_like(depth, dtype=torch.bool)
        first = relative_target_from_da3(depth, valid, epsilon=1e-6)
        second = relative_target_from_da3(depth * 17.0, valid, epsilon=1e-6)
        torch.testing.assert_close(
            first.relative_log_depth, second.relative_log_depth
        )

    def test_other_batch_sample_does_not_change_target(self) -> None:
        depth = torch.arange(1, 17, dtype=torch.float32).reshape(2, 1, 1, 2, 2, 2)
        valid = torch.ones_like(depth, dtype=torch.bool)
        baseline = relative_target_from_da3(depth, valid, epsilon=1e-6)
        changed = depth.clone()
        changed[1] *= 1000.0
        updated = relative_target_from_da3(changed, valid, epsilon=1e-6)
        torch.testing.assert_close(
            baseline.relative_log_depth[0], updated.relative_log_depth[0]
        )

    def test_stereo_fx_baseline_corrects_cross_view_scale(self) -> None:
        metric_depth = torch.tensor((2.0, 4.0, 8.0)).reshape(1, 3, 1, 1, 1, 1)
        fx = torch.tensor(((100.0, 200.0, 400.0),))
        baseline = torch.tensor(((0.1, 0.1, 0.1),))
        disparity = (
            (fx * baseline).reshape(1, 3, 1, 1, 1, 1) / metric_depth
        ).expand(1, 3, 1, 1, 2, 2)
        valid = torch.ones_like(disparity, dtype=torch.bool)
        target = relative_target_from_foundation_stereo(
            disparity, valid, fx, baseline, epsilon=1e-6
        )
        expected = metric_depth.log()
        expected = expected - expected.mean(dim=1, keepdim=True)
        torch.testing.assert_close(
            target.relative_log_depth[..., :1, :1], expected
        )

    def test_da3_and_foundation_targets_share_relative_semantics(self) -> None:
        depth = torch.tensor(
            [[[[[[1.0, 2.0], [4.0, 8.0]]]]]], dtype=torch.float32
        )
        valid = torch.ones_like(depth, dtype=torch.bool)
        fx = torch.tensor(((120.0,),))
        baseline = torch.tensor(((0.08,),))
        disparity = (fx * baseline).reshape(1, 1, 1, 1, 1, 1) / depth
        da3 = relative_target_from_da3(depth, valid, epsilon=1e-6)
        foundation = relative_target_from_foundation_stereo(
            disparity, valid, fx, baseline, epsilon=1e-6
        )
        torch.testing.assert_close(
            da3.relative_log_depth, foundation.relative_log_depth
        )

    def test_student_center_is_differentiable_and_shared_across_frames(self) -> None:
        raw = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 2, 2, 2)
        raw.requires_grad_()
        valid = torch.ones_like(raw, dtype=torch.bool)
        relative, center = relative_prediction_from_raw(raw, valid)
        self.assertEqual(center.shape, (1, 1, 1, 1, 1, 1))
        torch.testing.assert_close(relative.mean(), torch.tensor(0.0))
        relative.square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

if __name__ == "__main__":
    unittest.main()
