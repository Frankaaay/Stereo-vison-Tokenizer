import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluation.stage_a_contract import (
    CANONICAL_SPLIT_SCHEMA,
    canonical_sha256,
    read_identity_contract,
    read_selection,
    select_episode_windows,
    write_selection,
)
from evaluation.stage_a_manifest import _apportion_counts, assign_splits
from evaluation.stage_a_metrics import StageA1MetricSuite, _content_crop
from evaluation.tokenizer_stage_a import _run_parser


class StageASelectionTest(unittest.TestCase):
    def _identity(self):
        return {
            "identity_contract_path": "/contracts/umi-split-identities.json",
            "identity_contract_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
        }

    def _candidates(self):
        return [
            {
                "legacy_episode_id": f"episode-{index}",
                "legacy_group": "table_000",
                "canonical_config": "/configs/umi_table_000.yaml",
                "canonical_config_sha256": "c" * 64,
                "canonical_episode_index": index,
                "canonical_rgb_target_length": 24 + index,
                "source_fps": 30.0,
                "window_count": 5,
            }
            for index in range(8)
        ]

    def test_canonical_split_counts_and_assignment_are_exact_and_deterministic(self):
        self.assertEqual(
            _apportion_counts(90174),
            {"train": 81156, "val": 4509, "test": 4509},
        )
        records = [{"episode_id": f"episode-{index}"} for index in range(20)]
        first, counts = assign_splits(records, dataset_id="umi", seed=3407)
        second, _ = assign_splits(
            list(reversed(records)), dataset_id="umi", seed=3407
        )
        self.assertEqual(first, second)
        self.assertEqual(counts, {"train": 18, "val": 1, "test": 1})

    def test_canonical_manifest_detects_tampering(self):
        payload = {
            "schema": CANONICAL_SPLIT_SCHEMA,
            "dataset_id": "umi",
            "split_policy": {
                "name": "sha256_global_episode_rank_exact_90_5_5_v1",
                "counts": {"train": 1, "val": 1, "test": 1},
            },
            "canonical_loader": {
                "git_sha": "d51377ac450b0066bc0c8eb13939bcfae47275ff"
            },
            "records": [
                {
                    "episode_id": identity,
                    "split": split,
                    "canonical_config": "/configs/umi.yaml",
                    "canonical_config_sha256": "a" * 64,
                    "canonical_episode_index": index,
                    "canonical_rgb_target_length": 16,
                    "source_fps": 30.0,
                }
                for index, (identity, split) in enumerate(
                    (("train", "train"), ("val", "val"), ("test", "test"))
                )
            ],
        }
        payload["manifest_sha256"] = canonical_sha256(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload))
            loaded = read_identity_contract(path, dataset_id="umi")
            self.assertEqual(
                loaded["source_manifest_sha256"], payload["manifest_sha256"]
            )
            payload["records"][0]["split"] = "test"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                read_identity_contract(path, dataset_id="umi")

    def test_selection_is_deterministic_distinct_and_four_frame(self):
        first = select_episode_windows(
            self._candidates(),
            dataset_id="umi",
            split="test",
            sample_count=4,
            seed=1234,
            identity_contract=self._identity(),
        )
        second = select_episode_windows(
            reversed(self._candidates()),
            dataset_id="umi",
            split="test",
            sample_count=4,
            seed=1234,
            identity_contract=self._identity(),
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["records"]), 4)
        self.assertEqual(
            len({row["legacy_episode_id"] for row in first["records"]}), 4
        )
        for row in first["records"]:
            self.assertEqual(row["anchor_rgb_index"] % 4, 0)
            self.assertEqual(row["expected_frame_offsets"], [0, 1, 2, 3])
            start = row["anchor_rgb_index"] * 3
            self.assertEqual(
                row["expected_source_frame_indices"],
                [start, start + 3, start + 6, start + 9],
            )

    def test_written_selection_detects_tampering(self):
        payload = select_episode_windows(
            self._candidates(),
            dataset_id="umi",
            split="test",
            sample_count=2,
            seed=1234,
            identity_contract=self._identity(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            write_selection(path, payload)
            self.assertEqual(read_selection(path)["selection_sha256"], payload["selection_sha256"])
            value = json.loads(path.read_text())
            value["records"][0]["anchor_rgb_index"] += 4
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                read_selection(path)

    def test_selection_digest_is_canonical(self):
        payload = select_episode_windows(
            self._candidates(),
            dataset_id="umi",
            split="test",
            sample_count=2,
            seed=1234,
            identity_contract=self._identity(),
        )
        digest = payload.pop("selection_sha256")
        self.assertEqual(digest, canonical_sha256(payload))

    def test_selection_resamples_decode_failures_deterministically(self):
        def validator(row):
            if int(row["legacy_episode_id"].rsplit("-", 1)[1]) % 2:
                raise RuntimeError("row identity mismatch")

        payload = select_episode_windows(
            self._candidates(),
            dataset_id="umi",
            split="test",
            sample_count=4,
            seed=1234,
            identity_contract=self._identity(),
            candidate_validator=validator,
        )
        self.assertTrue(
            all(
                int(row["legacy_episode_id"].rsplit("-", 1)[1]) % 2 == 0
                for row in payload["records"]
            )
        )
        self.assertEqual(payload["decode_validation"]["accepted_count"], 4)
        self.assertGreater(payload["decode_validation"]["rejected_count"], 0)

    def test_stage_a_parser_does_not_require_training_hyperparameters(self):
        args = _run_parser().parse_args(
            [
                "--stereo_vae_ckpt", "checkpoint.ckpt",
                "--output_json", "metrics.json",
                "--eval_temporal_mode", "both",
                "--stage-a-dataset-id", "umi",
                "--stage-a-selection", "selection.json",
                "--canonical-loader-root", "loader",
                "--single_frame_source_indices", "0", "1", "2", "3",
            ]
        )
        self.assertIsNone(args.image_gan_weight)
        self.assertIsNone(args.stereo_search_radii)


class _DummyLPIPS(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean((1, 2, 3), keepdim=True)


class StageA1MetricTest(unittest.TestCase):
    def test_masked_rgb_temporal_health_and_abi(self):
        target = torch.zeros(2, 1, 1, 3, 4, 16, 16)
        prediction = torch.full((2, 1, 3, 4, 16, 16), 0.1)
        mask = torch.zeros(2, 1, 1, 4, 16, 16, dtype=torch.bool)
        mask[..., 2:14, :] = True
        batch = {
            "video": target,
            "rgb_valid_mask": mask,
            "sample_id": ["sample-0", "sample-1"],
        }
        output = SimpleNamespace(
            rgb=prediction,
            latent=torch.zeros(2, 1, 48, 1, 4, 4),
            raw_relative_log_depth=torch.zeros(2, 1, 1, 4, 16, 16),
        )
        suite = StageA1MetricSuite(relative_depth_epsilon=1e-6)
        suite.update(
            "mono/four_frame",
            batch,
            output,
            ("cam_head_left",),
            _DummyLPIPS(),
        )
        result = suite.finalize("mono/four_frame", ("cam_head_left",))
        macro = result["per_sample_macro"]
        self.assertAlmostEqual(macro["rgb_l1"]["mean"], 0.1, places=6)
        self.assertAlmostEqual(macro["rgb_mse"]["mean"], 0.01, places=6)
        self.assertAlmostEqual(macro["temporal_delta_l1"]["mean"], 0.0, places=6)
        self.assertEqual(result["output_health"]["nan_count"], 0)
        self.assertEqual(result["output_health"]["inf_count"], 0)
        self.assertEqual(result["latent_abi"]["tokens_per_window"], 16)
        self.assertEqual(
            result["valid_rgb_values"], 2 * 1 * 3 * 4 * 12 * 16
        )

    def test_content_crop_rejects_non_rectangular_mask(self):
        target = torch.zeros(3, 1, 16, 16)
        mask = torch.ones(1, 16, 16, dtype=torch.bool)
        mask[0, 4, 4] = False
        with self.assertRaisesRegex(ValueError, "rectangular"):
            _content_crop(target, target, mask)


if __name__ == "__main__":
    unittest.main()
