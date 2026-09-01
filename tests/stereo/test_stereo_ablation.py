import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import eval_stereo_vae as evaluation
from scripts.data.build_canonical_v3_stereo_manifest import split_for_episode
from stereo_tokenizer.ablation import (
    AblationCondition,
    apply_student_condition,
    expand_conditions,
    paired_bootstrap,
    teacher_target_checksum,
    write_ablation_report,
    zero_fill_horizontal_shift,
)
from stereo_tokenizer.canonical_v3_data import (
    EpisodeSpan,
    fixed_episode_window_pairs,
)


class StereoAblationTest(unittest.TestCase):
    def test_condition_order_and_shift_expansion_are_deterministic(self):
        conditions = expand_conditions(
            (
                "real_stereo",
                "copy_left",
                "fusion_off",
                "wrong_right",
                "shift_right",
                "time_reverse",
            ),
            (32, -16, 16, -32),
        )
        self.assertEqual(
            tuple(condition.name for condition in conditions),
            (
                "real_stereo",
                "copy_left",
                "fusion_off",
                "wrong_right",
                "shift_right_-32",
                "shift_right_-16",
                "shift_right_+16",
                "shift_right_+32",
                "time_reverse",
            ),
        )
        self.assertEqual(conditions[2].fusion_scale_override, 0.0)

    def test_horizontal_shift_zero_fills_without_wrap(self):
        value = torch.arange(5).reshape(1, 5)
        torch.testing.assert_close(
            zero_fill_horizontal_shift(value, 2),
            torch.tensor([[0, 0, 0, 1, 2]]),
        )
        torch.testing.assert_close(
            zero_fill_horizontal_shift(value, -2),
            torch.tensor([[2, 3, 4, 0, 0]]),
        )

    def test_student_conditions_never_modify_teacher_tensors(self):
        video = torch.arange(1 * 1 * 2 * 1 * 4 * 1 * 5).reshape(
            1, 1, 2, 1, 4, 1, 5
        ).float()
        disparity = torch.randn(1, 1, 1, 4, 1, 5)
        valid = torch.ones_like(disparity, dtype=torch.bool)
        batch = {
            "video": video,
            "wrong_right_video": torch.full_like(video[:, :, 1], 99),
            "disparity": disparity,
            "valid_mask": valid,
        }
        checksum = teacher_target_checksum(
            {**batch, "sample_id": ["sample"]}
        )
        for condition in (
            AblationCondition("copy_left", "copy_left"),
            AblationCondition("wrong_right", "wrong_right"),
            AblationCondition("shift_right_+2", "shift_right", shift_px=2),
            AblationCondition("time_reverse", "time_reverse"),
        ):
            changed = apply_student_condition(batch, condition)
            self.assertIs(changed["disparity"], disparity)
            self.assertIs(changed["valid_mask"], valid)
            self.assertEqual(
                checksum,
                teacher_target_checksum(
                    {**changed, "sample_id": ["sample"]}
                ),
            )
        copied = apply_student_condition(
            batch, AblationCondition("copy_left", "copy_left")
        )
        torch.testing.assert_close(
            copied["video"][:, :, 1], copied["video"][:, :, 0]
        )
        wrong = apply_student_condition(
            batch, AblationCondition("wrong_right", "wrong_right")
        )
        torch.testing.assert_close(
            wrong["video"][:, :, 1], batch["wrong_right_video"]
        )

    def test_episode_pairing_is_deterministic_and_deranged(self):
        records = [{"episode_id": f"episode-{index}"} for index in range(4)]
        spans = [
            EpisodeSpan(index, index * 5, 5, f"file-{index}")
            for index in range(4)
        ]
        dataset = SimpleNamespace(records=records, episode_spans=spans)
        pairs = fixed_episode_window_pairs(dataset, 4, 2, 1234)
        self.assertEqual(pairs, fixed_episode_window_pairs(dataset, 4, 2, 1234))
        self.assertEqual(len(pairs), 8)
        for left, right in pairs:
            self.assertNotEqual(left // 5, right // 5)

    def test_hash_split_has_no_identity_overlap(self):
        groups = {"train": set(), "val": set(), "test": set()}
        for index in range(10_000):
            identity = f"episode-{index}"
            groups[split_for_episode(identity, 1234)].add(identity)
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        self.assertEqual(sum(map(len, groups.values())), 10_000)

    @staticmethod
    def _records():
        conditions = {
            "real_stereo": 0.90,
            "copy_left": 1.00,
            "fusion_off": 1.02,
            "wrong_right": 1.10,
            "shift_right_-32": 1.20,
            "shift_right_-16": 1.10,
            "shift_right_+16": 1.11,
            "shift_right_+32": 1.21,
            "time_reverse": 1.08,
        }
        records = []
        for episode in range(8):
            for condition, l1 in conditions.items():
                for view in ("head", "lefthand", "righthand"):
                    records.append(
                        {
                            "sample_id": f"sample-{episode}",
                            "episode_id": f"episode-{episode}",
                            "condition": condition,
                            "mode_id": "stereo/four_frame",
                            "view": view,
                            "teacher_checksum": f"checksum-{episode}",
                            "relative_log_l1": l1,
                            "relative_log_rmse": l1,
                            "relative_log_silog": l1,
                            "rgb_l1": 0.1,
                            "rgb_psnr_db": 30.0,
                            "temporal_delta_l1": 0.1,
                            "latent_l2": 0.0,
                            "latent_cosine": 1.0,
                            "fusion_alpha": 0.5,
                            "fusion_confidence_mean": 0.7,
                            "fusion_attention_entropy": 0.3,
                            "fusion_boundary_offset_rate": 0.1,
                            "fusion_offset_teacher_mae": 0.5,
                            "fusion_offset_teacher_spearman": 0.6,
                        }
                    )
        return records

    def test_bootstrap_and_self_contained_html_share_metrics(self):
        records = self._records()
        bootstrap = paired_bootstrap(records, iterations=100, seed=1234)
        shuffled = list(records)
        random.Random(1234).shuffle(shuffled)
        self.assertEqual(
            bootstrap,
            paired_bootstrap(shuffled, iterations=100, seed=1234),
        )
        self.assertEqual(bootstrap["iterations"], 100)
        copy = next(
            entry
            for entry in bootstrap["entries"]
            if entry["condition"] == "copy_left"
            and entry["mode_id"] == "all"
            and entry["view"] == "macro"
        )
        self.assertAlmostEqual(copy["stereo_gain_percent"], 10.0)
        with tempfile.TemporaryDirectory() as parent:
            report_dir = Path(parent) / "report"
            decision = write_ablation_report(
                report_dir,
                records=records,
                evaluation_result={"modes": {}},
                provenance={"git_sha": "abc"},
                bootstrap_iterations=100,
                seed=1234,
            )
            self.assertEqual(decision["status"], "pass")
            html = (report_dir / "index.html").read_text(encoding="utf-8")
            metrics = json.loads(
                (report_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("10.00%", html)
            self.assertEqual(metrics["decision"]["status"], "pass")
            self.assertTrue((report_dir / "paired_samples.csv").is_file())
            self.assertNotIn("https://", html)

    def test_requested_keys_preserve_legacy_and_expand_ablation(self):
        legacy = SimpleNamespace(
            eval_eye_mode="stereo",
            eval_temporal_mode="both",
            single_frame_source_index=0,
            single_frame_source_indices=None,
            ablation_condition=None,
        )
        self.assertEqual(
            evaluation.requested_evaluation_keys(legacy),
            evaluation.requested_mode_ids(legacy),
        )
        ablation = SimpleNamespace(
            **vars(legacy),
        )
        ablation.ablation_condition = [
            "real_stereo",
            "copy_left",
            "fusion_off",
            "wrong_right",
            "shift_right",
            "time_reverse",
        ]
        ablation.right_shift_px = [-32, -16, 16, 32]
        keys = evaluation.requested_evaluation_keys(ablation)
        self.assertIn("real_stereo/stereo/single_frame", keys)
        self.assertIn("time_reverse/stereo/four_frame", keys)
        self.assertNotIn("time_reverse/stereo/single_frame", keys)


if __name__ == "__main__":
    unittest.main()
