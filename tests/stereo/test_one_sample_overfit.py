import unittest

import torch

from train_stereo_vae_one_sample_overfit import (
    _validate_milestones,
    mode_for_step,
    reconstruction_metrics,
)
from stereo_tokenizer.mode_sampling import MODE_IDS


class OneSampleOverfitTest(unittest.TestCase):
    def test_joint_schedule_is_strict_one_to_one_rotation(self):
        observed = [mode_for_step("joint", step) for step in range(1, 9)]
        self.assertEqual(observed, list(MODE_IDS) * 2)

    def test_joint_milestones_are_per_mode(self):
        _validate_milestones("joint", 16000, (100, 500, 1000, 2000, 4000))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            _validate_milestones("joint", 4000, (4000,))

    def test_metrics_record_temporal_and_stereo_differences(self):
        source = torch.zeros(2, 2, 3, 2, 2, 2)
        source[:, 0, :, 1] = 0.25
        source[:, 1] = source[:, 0] + 0.1
        target = source[:, 0]
        prediction = target + 0.05
        metrics = reconstruction_metrics(source, target, prediction)
        self.assertAlmostEqual(metrics["mae"], 0.05, places=6)
        self.assertEqual(len(metrics["gt_adjacent_frame_mae"]), 1)
        self.assertEqual(len(metrics["prediction_adjacent_frame_mae"]), 1)
        self.assertEqual(len(metrics["input_left_right_mae_per_view_frame"]), 2)


if __name__ == "__main__":
    unittest.main()
