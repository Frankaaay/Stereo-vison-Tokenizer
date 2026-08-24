import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stereo_tokenizer.lerobot_data import (
    EpisodeSequentialDistributedSampler,
    LeRobotStereoDataset,
    VIDEO_KEYS,
    fixed_episode_subset_indices,
    window_count,
)
from stereo_tokenizer.online_gt import FoundationStereoOnlineTeacher


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts" / "data" / "build_lerobot_stereo_manifest.py"
SPEC = importlib.util.spec_from_file_location("lerobot_manifest_builder", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
SELECTION_PATH = ROOT / "scripts" / "data" / "build_lerobot_teacher_selection.py"
SELECTION_SPEC = importlib.util.spec_from_file_location(
    "lerobot_teacher_selection", SELECTION_PATH
)
SELECTION = importlib.util.module_from_spec(SELECTION_SPEC)
SELECTION_SPEC.loader.exec_module(SELECTION)


def camera(fx=200.0, tx=0.0):
    return {
        "K": [fx, 0, 320, 0, fx, 240, 0, 0, 1],
        "D": [0.0] * 8,
        "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "P": [fx, 0, 320, tx, 0, fx, 240, 0, 0, 0, 1, 0],
        "width": 640,
        "height": 480,
        "distortion_model": "rational_polynomial",
    }


def calibration():
    return {
        view: {"left": camera(), "right": camera(tx=-11.0)}
        for view in ("head", "lefthand", "righthand")
    }


def record(episode_id, shard_id, length, split, audit_sha):
    videos = {
        key: {
            "relative_path": f"{shard_id}/videos/{key}/chunk-000/file-000.mp4",
            "from_timestamp": 0.0,
            "to_timestamp": length / 30,
        }
        for key in VIDEO_KEYS.values()
    }
    return {
        "schema": "lerobot-stereo-episode-v1",
        "episode_id": episode_id,
        "shard_id": shard_id,
        "episode_index": 0,
        "length": length,
        "window_count": window_count(length),
        "split": split,
        "videos": videos,
        "calibration": calibration(),
        "rectification": {
            "mode": "verified_pre_rectified",
            "audit_sha256": audit_sha,
        },
        "contract_sha256": "b" * 64,
    }


class LeRobotOnlineContractTest(unittest.TestCase):
    def test_teacher_selection_can_be_limited_to_resident_shards(self):
        records = [
            {
                "episode_id": f"episode-{index}",
                "shard_id": f"shard_{index:04d}",
                "first_dataset_index": index * 10,
                "window_count": 10,
            }
            for index in range(4)
        ]
        filtered = SELECTION.filter_by_maximum_shard_index(records, 1)
        self.assertEqual(
            [record["shard_id"] for record in filtered],
            ["shard_0000", "shard_0001"],
        )
        self.assertEqual(filtered[1]["first_dataset_index"], 10)
        with self.assertRaisesRegex(ValueError, "maximum shard index"):
            SELECTION.filter_by_maximum_shard_index(records, -1)

    def test_failed_audit_requires_explicit_provisional_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema": "lerobot-stereo-rectification-audit-v1",
                        "dataset_root": str(root),
                        "result": "fail",
                        "selected_mode": None,
                        "representative_pair_count": 12,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "audit did not pass"):
                BUILDER._load_rectification_audit(audit, root)
            payload, decision = BUILDER._load_rectification_audit(
                audit,
                root,
                allow_provisional_pre_rectified=True,
            )
            self.assertEqual(payload["result"], "fail")
            self.assertEqual(decision["mode"], "verified_pre_rectified")
            self.assertEqual(
                decision["status"], "provisional_user_assumption"
            )
            self.assertEqual(decision["source_audit_result"], "fail")
            self.assertEqual(len(decision["audit_sha256"]), 64)

    def test_window_count_preserves_point_one_second_frame_offsets(self):
        self.assertEqual(window_count(9), 0)
        self.assertEqual(window_count(10), 1)
        self.assertEqual(window_count(21), 1)
        self.assertEqual(window_count(22), 2)

    def test_episode_split_assignment_is_deterministic_and_exact(self):
        records = [
            {"episode_id": f"episode-{index}", "shard_id": f"shard_{index:04d}"}
            for index in range(100)
        ]
        first = [dict(item) for item in records]
        second = [dict(item) for item in records]
        BUILDER.assign_splits(first, 1234)
        BUILDER.assign_splits(second, 1234)
        self.assertEqual(first, second)
        counts = {
            split: sum(item["split"] == split for item in first)
            for split in ("train", "val", "test")
        }
        self.assertEqual(counts, {"train": 90, "val": 5, "test": 5})

    def test_manifest_requires_matching_rectification_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "episodes.jsonl"
            manifest.write_text(
                json.dumps(record("episode", "shard_0000", 22, "train", "a" * 64))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "audit SHA mismatch"):
                LeRobotStereoDataset(
                    manifest,
                    root,
                    split="train",
                    expected_rectification_audit_sha256="c" * 64,
                )

    def test_output_fx_is_scaled_but_baseline_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "episodes.jsonl"
            item = record("episode", "shard_0000", 22, "train", "a" * 64)
            manifest.write_text(json.dumps(item) + "\n", encoding="utf-8")
            dataset = LeRobotStereoDataset(
                manifest,
                root,
                split="train",
                expected_rectification_audit_sha256="a" * 64,
            )
            fx, baseline = dataset._output_calibration(item)
            np.testing.assert_allclose(fx, [80.0, 80.0, 80.0])
            np.testing.assert_allclose(baseline, [0.055, 0.055, 0.055])

    def test_sampler_keeps_each_episode_in_time_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "episodes.jsonl"
            items = [
                record("a", "shard_0000", 34, "train", "a" * 64),
                record("b", "shard_0001", 34, "train", "a" * 64),
            ]
            manifest.write_text(
                "".join(json.dumps(item) + "\n" for item in items),
                encoding="utf-8",
            )
            dataset = LeRobotStereoDataset(
                manifest,
                root,
                split="train",
                expected_rectification_audit_sha256="a" * 64,
            )
            sampler = EpisodeSequentialDistributedSampler(
                dataset, shuffle=False, seed=1234, num_replicas=1, rank=0
            )
            self.assertEqual(list(sampler), list(range(len(dataset))))

    def test_fixed_validation_subset_uses_distinct_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "episodes.jsonl"
            items = [
                record(
                    f"episode-{index}",
                    f"shard_{index:04d}",
                    34,
                    "val",
                    "a" * 64,
                )
                for index in range(8)
            ]
            manifest.write_text(
                "".join(json.dumps(item) + "\n" for item in items),
                encoding="utf-8",
            )
            dataset = LeRobotStereoDataset(
                manifest,
                root,
                split="val",
                expected_rectification_audit_sha256="a" * 64,
            )
            indices = fixed_episode_subset_indices(dataset, 5, seed=1234)
            episode_ids = {
                dataset._sample_address(index)[0]["episode_id"] for index in indices
            }
            self.assertEqual(len(indices), 5)
            self.assertEqual(len(episode_ids), 5)

    def test_lr_consistency_accepts_matching_positive_disparity(self):
        disparity = torch.full((1, 1, 256, 256), 4.0)
        residual, valid = FoundationStereoOnlineTeacher.lr_consistency(
            disparity, disparity
        )
        self.assertTrue(valid[:, :, 32:224, 8:-8].all())
        torch.testing.assert_close(
            residual[:, :, 32:224, 8:-8],
            torch.zeros_like(residual[:, :, 32:224, 8:-8]),
        )
        self.assertFalse(valid[:, :, :32].any())
        self.assertFalse(valid[:, :, 224:].any())


if __name__ == "__main__":
    unittest.main()
