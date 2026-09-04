import hashlib
import gzip
import json
import tempfile
import unittest
from unittest import mock
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
from evaluation.stage_a_metrics import (
    StageA1MetricSuite,
    _content_crop,
    _rgb_overshoot_stats,
    _teacher_targets,
    _temporal_flow_metrics,
    _temporal_geometry_metrics,
)
from evaluation.stage_a_data import HY_SCHEMA, _read_hy_manifest_matches
from evaluation.tokenizer_stage_a import (
    _checkpoint_provenance,
    _percentile_summary,
    _report_command,
    _run_parser,
    _stage_a_visualization_batch,
)


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
                "--checkpoint-sha256", "a" * 64,
                "--raft-checkpoint", "raft.pth",
                "--raft-checkpoint-sha256", "b" * 64,
                "--single_frame_source_indices", "0", "1", "2", "3",
            ]
        )
        self.assertIsNone(args.image_gan_weight)
        self.assertIsNone(args.stereo_search_radii)


class _DummyLPIPS(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean((1, 2, 3), keepdim=True)


class _QueuedFlow:
    def __init__(self, flows):
        self.flows = list(flows)

    def __call__(self, first, second):
        if not self.flows:
            raise AssertionError("unexpected flow inference")
        flow = self.flows.pop(0).to(first)
        return flow.expand(first.shape[0], -1, -1, -1).clone()


class StageA1MetricTest(unittest.TestCase):
    def test_hy_manifest_reader_supports_gzip_and_normalizes_table_names(self):
        record = {
            "schema": HY_SCHEMA,
            "table_name": "Table014",
            "episode_index": 7,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hy.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            matched, provenance = _read_hy_manifest_matches(
                path, digest, {("table_014", 7)}
            )
        self.assertEqual(matched[("table_014", 7)]["table_name"], "table_014")
        self.assertEqual(provenance["sha256"], digest)

    def test_temporal_flow_metrics_separate_static_and_dynamic_domains(self):
        target = torch.zeros(4, 3, 4, 5)
        prediction = target.clone()
        for frame_index in range(4):
            prediction[frame_index] = 0.1 * frame_index
        zero = torch.zeros(1, 2, 4, 5)
        static, _ = _temporal_flow_metrics(
            target, prediction, _QueuedFlow([zero, zero, zero])
        )
        self.assertAlmostEqual(static["clamped_optical_flow_warp_l1"], 0.1)
        self.assertAlmostEqual(static["clamped_static_flicker_l1"], 0.1)
        self.assertNotIn("clamped_motion_flow_epe_px", static)
        self.assertGreater(static["static_flow_valid_pixels"], 0)

        forward = torch.zeros(1, 2, 4, 5)
        forward[:, 0] = 1.0
        backward = -forward
        reconstruction_forward = forward.clone()
        reconstruction_forward[:, 0] = 2.0
        dynamic, _ = _temporal_flow_metrics(
            target,
            target,
            _QueuedFlow([forward, backward, reconstruction_forward]),
        )
        self.assertAlmostEqual(dynamic["clamped_motion_flow_epe_px"], 1.0)
        self.assertGreater(dynamic["dynamic_flow_valid_pixels"], 0)

    def test_temporal_geometry_uses_flow_aligned_relative_change(self):
        target = torch.arange(4.0).view(4, 1, 1, 1).expand(-1, -1, 4, 5)
        prediction = target * 2.0
        valid = torch.ones_like(target, dtype=torch.bool)
        zero_flow = torch.zeros(3, 2, 4, 5)
        values = _temporal_geometry_metrics(
            target,
            prediction,
            valid,
            valid,
            {
                "backward": zero_flow,
                "backward_valid": torch.ones(3, 4, 5, dtype=torch.bool),
            },
            "reconstruction_teacher",
        )
        self.assertAlmostEqual(
            values["reconstruction_teacher_temporal_geometry_warp_l1"], 1.0
        )
        for pair in ("pair_01", "pair_12", "pair_23"):
            self.assertAlmostEqual(
                values[
                    f"reconstruction_teacher_temporal_geometry_warp_l1_{pair}"
                ],
                1.0,
            )

    def test_checkpoint_provenance_accepts_declared_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.ckpt"
            checkpoint = {
                "epoch": 0,
                "global_step": 80000,
                "stereo_update_counters": {
                    "generator_updates": 124000,
                    "discriminator_updates": 0,
                    "batch_updates": 136000,
                    "four_frame_updates": 62000,
                    "single_frame_updates": 62000,
                },
            }
            torch.save(checkpoint, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = _checkpoint_provenance(path, digest)
            self.assertEqual(result["sha256"], digest)
            self.assertEqual(result["global_step"], 80000)
            self.assertEqual(
                result["stereo_update_counters"]["generator_updates"], 124000
            )

    def test_stereo_visualization_uses_unit_calibration_when_canonical_has_none(self):
        batch = {"disparity": torch.ones(1, 3, 1, 4, 8, 8)}
        adapted = _stage_a_visualization_batch(batch)
        self.assertNotIn("fx", batch)
        self.assertNotIn("baseline_m", batch)
        self.assertEqual(tuple(adapted["fx"].shape), (1, 3))
        self.assertTrue(torch.equal(adapted["fx"], adapted["baseline_m"]))
        self.assertTrue(torch.all(adapted["fx"] == 1))

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
        self.assertAlmostEqual(macro["clamped_rgb_l1"]["mean"], 0.1, places=6)
        self.assertAlmostEqual(macro["clamped_rgb_mse"]["mean"], 0.01, places=6)
        self.assertAlmostEqual(macro["clamped_temporal_delta_l1"]["mean"], 0.0, places=6)
        for pair in ("pair_01", "pair_12", "pair_23"):
            self.assertAlmostEqual(
                macro[f"clamped_temporal_delta_l1_{pair}"]["mean"], 0.0, places=6
            )
            self.assertAlmostEqual(
                macro[f"clamped_temporal_delta_lpips_{pair}"]["mean"], 0.0, places=6
            )
        self.assertEqual(result["output_health"]["nan_count"], 0)
        self.assertEqual(result["output_health"]["inf_count"], 0)
        self.assertEqual(result["latent_abi"]["tokens_per_window"], 16)
        self.assertEqual(
            result["valid_rgb_values"], 2 * 1 * 3 * 4 * 12 * 16
        )


    def test_rgb_v2_clamp_padding_overshoot_and_fail_closed_contracts(self):
        class RecordingLPIPS(_DummyLPIPS):
            def __init__(self):
                super().__init__()
                self.ranges = []

            def forward(self, prediction, target):
                self.ranges.append(
                    (
                        float(prediction.min()),
                        float(prediction.max()),
                        float(target.min()),
                        float(target.max()),
                    )
                )
                return super().forward(prediction, target)

        target = torch.zeros(1, 1, 1, 3, 1, 16, 16)
        prediction = torch.zeros(1, 1, 3, 1, 16, 16)
        mask = torch.zeros(1, 1, 1, 1, 16, 16, dtype=torch.bool)
        mask[..., 2:14, :] = True
        prediction[:, :, :, :, 0, :] = 100.0
        prediction[0, 0, 0, 0, 2, 0] = 0.75
        prediction[0, 0, 2, 0, 2, 1] = -1.0
        batch = {
            "video": target,
            "rgb_valid_mask": mask,
            "sample_id": ["spike"],
        }
        output = SimpleNamespace(
            rgb=prediction,
            latent=torch.zeros(1, 1, 48, 1, 4, 4),
            raw_relative_log_depth=torch.zeros(1, 1, 1, 1, 16, 16),
        )
        lpips = RecordingLPIPS()
        suite = StageA1MetricSuite(relative_depth_epsilon=1e-6)
        suite.update(
            "mono/single_frame/source_0",
            batch,
            output,
            ("head",),
            lpips,
        )
        result = suite.finalize("mono/single_frame/source_0", ("head",))
        metric = result["per_view"]["head"]
        denominator = 3 * 12 * 16
        self.assertAlmostEqual(metric["raw_rgb_l1"]["mean"], 1.75 / denominator)
        self.assertAlmostEqual(metric["raw_rgb_mse"]["mean"], 1.5625 / denominator)
        self.assertAlmostEqual(metric["clamped_rgb_l1"]["mean"], 1.0 / denominator)
        self.assertAlmostEqual(metric["clamped_rgb_mse"]["mean"], 0.5 / denominator)
        self.assertLessEqual(
            metric["clamped_rgb_l1"]["mean"], metric["raw_rgb_l1"]["mean"]
        )
        self.assertLessEqual(
            metric["clamped_rgb_mse"]["mean"], metric["raw_rgb_mse"]["mean"]
        )
        self.assertAlmostEqual(
            metric["rgb_out_of_range_pixel_ratio"]["mean"], 2 / (12 * 16)
        )
        self.assertAlmostEqual(metric["rgb_overshoot_positive_p50"]["mean"], 0.375)
        self.assertAlmostEqual(metric["rgb_overshoot_positive_p90"]["mean"], 0.475)
        self.assertAlmostEqual(metric["rgb_overshoot_positive_p99"]["mean"], 0.4975)
        self.assertAlmostEqual(metric["rgb_overshoot_positive_max"]["mean"], 0.5)
        health = result["output_health"]
        self.assertEqual(health["out_of_range_pixel_count"], 2)
        self.assertEqual(health["valid_pixel_count"], 12 * 16)
        self.assertEqual(health["valid_value_count"], denominator)
        self.assertEqual(health["all_raw_max"], 100.0)
        self.assertEqual(health["valid_raw_min"], -1.0)
        self.assertEqual(health["valid_raw_max"], 0.75)
        self.assertTrue(
            all(-1.0 <= value <= 1.0 for bounds in lpips.ranges for value in bounds)
        )

        no_overshoot = _rgb_overshoot_stats(
            torch.zeros(3, 1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool)
        )
        self.assertEqual(no_overshoot["rgb_out_of_range_pixels"], 0)
        for name in (
            "rgb_overshoot_positive_p50",
            "rgb_overshoot_positive_p90",
            "rgb_overshoot_positive_p99",
            "rgb_overshoot_positive_max",
        ):
            self.assertEqual(no_overshoot[name], 0.0)

        bad_target = dict(batch)
        bad_target["video"] = target.clone()
        bad_target["video"][0, 0, 0, 0, 0, 2, 0] = 0.6
        with self.assertRaisesRegex(ValueError, "normalized"):
            StageA1MetricSuite(relative_depth_epsilon=1e-6).update(
                "mono/single_frame/source_0",
                bad_target,
                output,
                ("head",),
                _DummyLPIPS(),
            )

        nan_prediction = prediction.clone()
        nan_prediction[0, 0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            StageA1MetricSuite(relative_depth_epsilon=1e-6).update(
                "mono/single_frame/source_0",
                batch,
                SimpleNamespace(
                    rgb=nan_prediction,
                    latent=output.latent,
                    raw_relative_log_depth=output.raw_relative_log_depth,
                ),
                ("head",),
                _DummyLPIPS(),
            )

    def test_temporal_delta_uses_clamped_frames(self):
        target = torch.zeros(1, 1, 1, 3, 4, 16, 16)
        prediction = torch.zeros(1, 1, 3, 4, 16, 16)
        prediction[:, :, :, 1] = 2.0
        prediction[:, :, :, 2] = -2.0
        mask = torch.ones(1, 1, 1, 4, 16, 16, dtype=torch.bool)
        suite = StageA1MetricSuite(relative_depth_epsilon=1e-6)
        suite.update(
            "mono/four_frame",
            {
                "video": target,
                "rgb_valid_mask": mask,
                "sample_id": ["temporal"],
            },
            SimpleNamespace(
                rgb=prediction,
                latent=torch.zeros(1, 1, 48, 1, 4, 4),
                raw_relative_log_depth=torch.zeros(1, 1, 1, 4, 16, 16),
            ),
            ("head",),
            _DummyLPIPS(),
        )
        metric = suite.finalize("mono/four_frame", ("head",))["per_view"]["head"]
        self.assertAlmostEqual(metric["clamped_temporal_delta_l1"]["mean"], 2 / 3)
        self.assertAlmostEqual(
            metric["clamped_temporal_delta_l1_pair_01"]["mean"], 0.5
        )
        self.assertAlmostEqual(
            metric["clamped_temporal_delta_l1_pair_12"]["mean"], 1.0
        )
        self.assertAlmostEqual(
            metric["clamped_temporal_delta_l1_pair_23"]["mean"], 0.5
        )
        self.assertNotIn("temporal_delta_l1", metric)

    def test_empty_teacher_view_is_recorded_without_dropping_rgb_sample(self):
        batch_size, views, frames, height, width = 2, 2, 1, 16, 16
        video = torch.zeros(batch_size, views, 2, 3, frames, height, width)
        rgb_valid = torch.ones(
            batch_size, views, 1, frames, height, width, dtype=torch.bool
        )
        teacher_valid = torch.ones_like(rgb_valid)
        teacher_valid[1, 0] = False
        batch = {
            "video": video,
            "rgb_valid_mask": rgb_valid,
            "valid_mask": teacher_valid,
            "disparity": torch.ones_like(teacher_valid, dtype=torch.float32),
            "sample_id": ["sample-valid", "sample-head-empty"],
        }
        output = SimpleNamespace(
            rgb=torch.zeros(batch_size, views, 3, frames, height, width),
            latent=torch.zeros(batch_size, views, 48, 1, 4, 4),
            raw_relative_log_depth=torch.zeros_like(teacher_valid, dtype=torch.float32),
        )
        suite = StageA1MetricSuite(relative_depth_epsilon=1e-6)
        suite.update(
            "stereo/single_frame/source_0",
            batch,
            output,
            ("head", "wrist"),
            _DummyLPIPS(),
        )
        result = suite.finalize(
            "stereo/single_frame/source_0", ("head", "wrist")
        )
        metric = "depth_head_teacher_relative_log_l1"
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["per_view"]["head"][metric]["count"], 1)
        self.assertEqual(result["per_view"]["wrist"][metric]["count"], 2)
        self.assertEqual(result["per_sample_macro"][metric]["count"], 1)
        self.assertEqual(result["teacher_invalid_count"], 1)
        self.assertEqual(
            result["teacher_invalid_samples"],
            [
                {
                    "sample_id": "sample-head-empty",
                    "view": "head",
                    "reason": "empty_teacher_mask",
                }
            ],
        )
        self.assertGreater(result["valid_rgb_values"], 0)

    def test_teacher_metric_semantics_are_explicit(self):
        depth = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 1, 1, 2, 2)
        valid = torch.ones_like(depth, dtype=torch.bool)
        target, reconstruction, kind = _teacher_targets(
            {
                "valid_mask": valid,
                "da3_relative_depth": depth,
                "reconstruction_da3_relative_depth": depth * 1.1,
                "reconstruction_valid_mask": valid,
            },
            1e-6,
        )
        self.assertEqual(kind, "reconstruction_teacher")
        self.assertIsNotNone(target)
        self.assertIsNotNone(reconstruction)

        target, reconstruction, kind = _teacher_targets(
            {"valid_mask": valid, "disparity": depth}, 1e-6
        )
        self.assertEqual(kind, "depth_head_teacher")
        self.assertIsNotNone(target)
        self.assertIsNone(reconstruction)

    def test_benchmark_percentiles(self):
        result = _percentile_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["mean_ms"], 2.5)
        self.assertAlmostEqual(result["p50_ms"], 2.5)
        self.assertAlmostEqual(result["p90_ms"], 3.7)

    def test_report_fails_closed_on_incomplete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "quality").mkdir()
            (root / "benchmark").mkdir()
            with self.assertRaisesRegex(ValueError, "exactly 10 quality"):
                _report_command(
                    [
                        "--artifact-root",
                        str(root),
                        "--output",
                        str(root / "report.md"),
                    ]
                )
            self.assertFalse((root / "report.md").exists())

    def test_report_renders_only_complete_verified_artifacts(self):
        cameras = [
            None,
            "observation.images.cam_head_left",
            "observation.images.cam_head_right",
            "observation.images.cam_left_wrist_left",
            "observation.images.cam_left_wrist_right",
            "observation.images.cam_right_wrist_left",
            "observation.images.cam_right_wrist_right",
        ]
        cells = [("umi", "stereo", cameras[0])]
        cells.extend(("umi", "mono", camera) for camera in cameras[1:])
        cells.extend(
            ("libero", "mono", camera)
            for camera in (
                "observation.images.cam_head_left",
                "observation.images.cam_left_wrist_left",
            )
        )
        cells.append(("hy", "mono", None))

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        def summary(value=0.1, count=1):
            return {
                "count": count,
                "mean": value,
                "p50": value,
                "p90": value,
                "p99": value,
            }

        def mode(eye_mode, four_frame, view_name, count):
            geometry = (
                "reconstruction_teacher" if eye_mode == "mono"
                else "depth_head_teacher"
            )
            metrics = {
                "raw_rgb_l1": summary(0.12, count),
                "raw_rgb_mse": summary(0.02, count),
                "clamped_rgb_l1": summary(0.1, count),
                "clamped_rgb_mse": summary(0.01, count),
                "clamped_psnr_db": summary(20.0, count),
                "clamped_ssim": summary(0.8, count),
                "clamped_lpips": summary(0.2, count),
                "rgb_valid_ratio": summary(1.0, count),
                "rgb_out_of_range_pixel_ratio": summary(0.0, count),
                "rgb_overshoot_positive_p50": summary(0.0, count),
                "rgb_overshoot_positive_p90": summary(0.0, count),
                "rgb_overshoot_positive_p99": summary(0.0, count),
                "rgb_overshoot_positive_max": summary(0.0, count),
                f"{geometry}_relative_log_l1": summary(0.3, count),
                f"{geometry}_relative_log_rmse": summary(0.4, count),
                f"{geometry}_relative_log_silog": summary(0.2, count),
                f"{geometry}_valid_ratio": summary(0.9, count),
            }
            if four_frame:
                metrics["clamped_temporal_delta_l1"] = summary(0.05, count)
                metrics["clamped_temporal_delta_lpips"] = summary(0.06, count)
                for pair in ("pair_01", "pair_12", "pair_23"):
                    metrics[f"clamped_temporal_delta_l1_{pair}"] = summary(0.05, count)
                    metrics[f"clamped_temporal_delta_lpips_{pair}"] = summary(0.06, count)
                    metrics[f"clamped_optical_flow_warp_l1_{pair}"] = summary(0.04, count)
                    metrics[f"clamped_static_flicker_l1_{pair}"] = summary(0.03, count)
                    metrics[f"clamped_motion_flow_epe_px_{pair}"] = summary(0.7, count)
                metrics["clamped_optical_flow_warp_l1"] = summary(0.04, count)
                metrics["clamped_static_flicker_l1"] = summary(0.03, count)
                metrics["clamped_motion_flow_epe_px"] = summary(0.7, count)
                metrics["optical_flow_valid_ratio"] = summary(0.9, count)
                metrics["static_flow_valid_ratio"] = summary(0.4, count)
                metrics["dynamic_flow_valid_ratio"] = summary(0.3, count)
                metrics[f"{geometry}_temporal_geometry_warp_l1"] = summary(0.08, count)
                metrics[f"{geometry}_temporal_geometry_valid_ratio"] = summary(0.8, count)
                for pair in ("pair_01", "pair_12", "pair_23"):
                    metrics[f"{geometry}_temporal_geometry_warp_l1_{pair}"] = summary(0.08, count)
            return {
                "sample_count": 1,
                "per_view": {view_name: metrics},
                "per_sample_macro": metrics,
                "valid_rgb_values": 300,
                "valid_teacher_pixels": 90,
                "teacher_invalid_count": 0,
                "teacher_invalid_samples": [],
                "output_health": {
                    "nan_count": 0,
                    "inf_count": 0,
                    "invalid_sample_count": 0,
                    "invalid_sample_ids": [],
                    "all_value_count": 300,
                    "all_raw_min": -0.5,
                    "all_raw_max": 0.5,
                    "valid_value_count": 300,
                    "valid_raw_min": -0.5,
                    "valid_raw_max": 0.5,
                    "valid_pixel_count": 100,
                    "out_of_range_pixel_count": 0,
                    "out_of_range_pixel_ratio": 0.0,
                },
                "latent_abi": {
                    "input_shape_without_batch": [1, 1, 3, 4, 256, 256],
                    "latent_shape_without_batch": [1, 48, 1, 16, 16],
                    "input_dtype": "torch.float32",
                    "latent_dtype": "torch.float32",
                    "latent_channels": 48,
                    "tokens_per_window": 256,
                    "tokens_per_input_frame": 64.0,
                    "spatial_compression_ratio": 256.0,
                    "temporal_compression_ratio": 4.0,
                    "view_compression_ratio": 1.0,
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quality_dir = root / "quality"
            benchmark_dir = root / "benchmark"
            log_dir = root / "logs"
            quality_dir.mkdir()
            benchmark_dir.mkdir()
            log_dir.mkdir()
            source_patch = root / "source.patch"
            source_patch.write_text("synthetic patch\n")
            metric_backbone = root / "vgg16-397923af.pth"
            metric_backbone.write_bytes(b"synthetic VGG16 fixture")
            raft_checkpoint = root / "raft-large.pth"
            raft_checkpoint.write_bytes(b"synthetic RAFT fixture")
            environment = {
                "python": "3.12.11 test",
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "gpu_capability": [9, 0],
                "torch_cuda": "12.6",
                "cudnn": 90100,
                "uv_lock_sha256": "4" * 64,
                "metric_backbone": {
                    "name": "torchvision.vgg16.IMAGENET1K_V1",
                    "role": "torchmetrics LPIPS VGG feature backbone",
                    "path": str(metric_backbone),
                    "sha256": digest(metric_backbone),
                    "preprocessing": "torchmetrics LPIPS vgg normalize=False on RGB [-1,1]",
                },
                "packages": {"torch": "2.7.1"},
            }
            provenance = {
                "cwd": "/repo",
                "git_branch": "hezhou-las2-h",
                "git_commit": "b" * 40,
                "git_diff_sha256": digest(source_patch),
                "git_status_porcelain": " M evaluation/tokenizer_stage_a.py",
                "environment": environment,
                "resolved_args": {},
            }
            selections = {}
            configs = {}
            hy_manifest = root / "hy.jsonl.gz"
            hy_manifest.write_bytes(b"synthetic compressed manifest fixture")
            for dataset_id in ("umi", "libero", "hy"):
                selection = root / f"{dataset_id}-selection.json"
                count = 1024 if dataset_id in {"umi", "hy"} else 256
                selection_payload = {
                            "sample_count": count,
                            "records": [{} for _ in range(count)],
                            "decode_validation": {
                                "accepted_count": count,
                                "checked_candidate_count": count + 2,
                                "rejected_count": 2,
                                "rejected_episode_ids_sha256": "3" * 64,
                            }
                }
                if dataset_id == "hy":
                    selection_payload.update(
                        {
                            "hy_manifest": {
                                "path": str(hy_manifest),
                                "sha256": digest(hy_manifest),
                            },
                            "excluded_source_groups": {
                                "groups": ["table_014"],
                                "episode_count": 1,
                                "episode_ids_sha256": "6" * 64,
                                "reason": "fixture",
                            },
                            "included_source_groups": [
                                "table_012", "table_016", "table_018", "table_020"
                            ],
                        }
                    )
                selection.write_text(json.dumps(selection_payload))
                selections[dataset_id] = selection
                if dataset_id != "hy":
                    config = root / f"{dataset_id}.yaml"
                    config.write_text(f"dataset: {dataset_id}\n")
                    configs[dataset_id] = config

            artifacts = []
            for index, (dataset_id, eye_mode, camera) in enumerate(cells):
                count = 1024 if dataset_id in {"umi", "hy"} else 256
                view_name = camera or "head_pair"
                modes = {
                    f"{eye_mode}/single_frame/source_{source}": mode(
                        eye_mode, False, view_name, count
                    )
                    for source in range(4)
                }
                modes[f"{eye_mode}/four_frame"] = mode(
                    eye_mode, True, view_name, count
                )
                for value in modes.values():
                    value["sample_count"] = count
                visualizations = []
                resolved_args = {}
                if (dataset_id, eye_mode, camera) in {
                    ("umi", "stereo", None),
                    ("libero", "mono", "observation.images.cam_head_left"),
                }:
                    visual_dir = root / "visualizations" / f"case-{index}"
                    visual_dir.mkdir(parents=True)
                    visualizations = []
                    for slot in range(8):
                        for source in range(4):
                            rgb = f"case-{slot:02d}-source-{source}.png"
                            geometry = f"depth-case-{slot:02d}-source-{source}.png"
                            (visual_dir / rgb).write_bytes(b"png")
                            (visual_dir / geometry).write_bytes(b"png")
                            visualizations.append(
                                {
                                    "slot": slot,
                                    "selection_index": slot,
                                    "sample_id": f"sample-{slot}",
                                    "source_frame_index": source,
                                    "rgb_file": rgb,
                                    "geometry_file": geometry,
                                }
                            )
                    (visual_dir / "cases.json").write_text(
                        json.dumps(visualizations)
                    )
                    resolved_args["visualization_dir"] = str(visual_dir)
                payload = {
                    "schema": "stereo-tokenizer-stage-a1-result-v3",
                    "status": "formal",
                    "posterior": "mean",
                    "quality_precision": "fp32",
                    "requested_modes": list(modes),
                    "single_frame_source_indices": [0, 1, 2, 3],
                    "checkpoint": {
                        "path": "/checkpoint.ckpt",
                        "sha256": "9" * 64,
                        "global_step": 125000,
                        "epoch": 1,
                        "stereo_update_counters": {"generator_updates": 162500},
                    },
                    "dataset": {
                        "dataset_id": dataset_id,
                        "eye_mode": eye_mode,
                        "camera_key": camera,
                        "sample_count": count,
                        "selection_path": str(selections[dataset_id]),
                        "selection_sha256": ({"umi": "e", "libero": "f", "hy": "7"}[dataset_id]) * 64,
                        "selection_file_sha256": digest(selections[dataset_id]),
                        "identity_contract": {
                            "sha256": "1" * 64,
                            "source_manifest_sha256": "2" * 64,
                        },
                        **(
                            {
                                "data_backend": "hy_lance_manifest",
                                "hy_manifest": {
                                    "path": str(hy_manifest),
                                    "sha256": digest(hy_manifest),
                                },
                                "excluded_source_groups": selection_payload["excluded_source_groups"],
                                "included_source_groups": selection_payload["included_source_groups"],
                            }
                            if dataset_id == "hy"
                            else {
                                "canonical_config_sha256": {
                                    str(configs[dataset_id]): digest(configs[dataset_id])
                                },
                                "canonical_loader": {
                                    "git_sha": "d51377ac450b0066bc0c8eb13939bcfae47275ff"
                                },
                            }
                        ),
                    },
                    "teacher": (
                        {
                            "source_sha": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
                            "checkpoint_sha256": "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5",
                        }
                        if eye_mode == "mono" else
                        {
                            "backend": "las2_h",
                            "source_sha": "8c97bd4c4da3712c2ac60003a23201dfdb5935f4",
                            "checkpoint_sha256": "758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4",
                        }
                    ),
                    "flow_teacher": {
                        "name": "torchvision.raft_large",
                        "architecture": "RAFT-Large",
                        "transform_contract": "Raft_Large_Weights.C_T_SKHT_V2.transforms",
                        "checkpoint": str(raft_checkpoint),
                        "checkpoint_sha256": digest(raft_checkpoint),
                        "microbatch": 3,
                        "precision": "fp32",
                        "flow_unit": "content-crop pixels",
                        "static_flow_max_px": 0.5,
                        "dynamic_flow_min_px": 1.0,
                        "forward_backward_relative_threshold": 0.01,
                        "forward_backward_absolute_threshold_px2": 0.5,
                    },
                    "tokenizer_parameters": {
                        "total": 100,
                        "architecturally_trainable": 100,
                        "runtime_requires_grad": 0,
                        "evaluation_state": "eval_inference_mode_posterior_mean",
                    },
                    "metrics": modes,
                    "visualizations": visualizations,
                    "provenance": {**provenance, "resolved_args": resolved_args},
                }
                result_path = quality_dir / f"quality-{index}.json"
                result_path.write_text(json.dumps(payload))
                artifacts.append(result_path)

            for eye_mode in ("mono", "stereo"):
                timing = {
                    "count": 300,
                    "mean_ms": 10.0,
                    "p50_ms": 10.0,
                    "p90_ms": 11.0,
                    "peak_allocated_bytes": 2**30,
                    "peak_reserved_bytes": 2**30,
                }
                mode_payload = {
                    name: {
                        "encode_including_posterior_mean": timing,
                        "cached_posterior_mean": timing,
                        "decode": timing,
                        "end_to_end": timing,
                        "throughput": {
                            "samples_per_second": 100.0,
                            "frames_per_second": 100.0,
                        },
                    }
                    for name in ("single_frame", "four_frame")
                }
                payload = {
                    "schema": "stereo-tokenizer-stage-a1-benchmark-v1",
                    "status": "formal",
                    "warmup": 20,
                    "iterations": 100,
                    "repeats": 3,
                    "precision": "bf16",
                    "batch_size": 1,
                    "posterior": "mean",
                    "timing_scope": "model_only_excludes_data_decode_and_teacher",
                    "checkpoint": {
                        "sha256": "9" * 64
                    },
                    "dataset": {"dataset_id": "umi", "eye_mode": eye_mode},
                    "modes": mode_payload,
                    "provenance": provenance,
                }
                result_path = benchmark_dir / f"benchmark-{eye_mode}.json"
                result_path.write_text(json.dumps(payload))
                artifacts.append(result_path)

            jobs = []
            for index, artifact in enumerate(artifacts):
                log = log_dir / f"job-{index}.log"
                log.write_text("completed\n")
                jobs.append(
                    {
                        "artifact": str(artifact.relative_to(root)),
                        "sha256": digest(artifact),
                        "log": str(log.relative_to(root)),
                        "log_sha256": digest(log),
                        "job_id": str(1000 + index),
                        "state": "COMPLETED",
                        "exit_code": 0,
                    }
                )
            (root / "job-status.json").write_text(
                json.dumps(
                    {
                        "schema": "stereo-tokenizer-stage-a1-job-status-v1",
                        "jobs": jobs,
                    }
                )
            )
            output = root / "report.md"
            with mock.patch(
                "evaluation.tokenizer_stage_a.VGG16_CHECKPOINT_SHA256",
                digest(metric_backbone),
            ):
                _report_command(
                    ["--artifact-root", str(root), "--output", str(output)]
                )
            rendered = output.read_text()
            self.assertIn("Stage A1 Baseline", rendered)
            self.assertIn("depth_head_teacher", rendered)
            self.assertIn("显式排除 Table014", rendered)
            self.assertIn("置信度：80%", rendered)
            self.assertIn("Clamp-domain RGB 图像质量", rendered)
            self.assertNotIn("| abs(output)>1 |", rendered)

            legacy_path = quality_dir / "quality-0.json"
            legacy = json.loads(legacy_path.read_text())
            legacy["schema"] = "stereo-tokenizer-stage-a1-result-v1"
            legacy_path.write_text(json.dumps(legacy))
            for job in jobs:
                if job["artifact"] == str(legacy_path.relative_to(root)):
                    job["sha256"] = digest(legacy_path)
            (root / "job-status.json").write_text(
                json.dumps(
                    {
                        "schema": "stereo-tokenizer-stage-a1-job-status-v1",
                        "jobs": jobs,
                    }
                )
            )
            output.unlink()
            with mock.patch(
                "evaluation.tokenizer_stage_a.VGG16_CHECKPOINT_SHA256",
                digest(metric_backbone),
            ):
                with self.assertRaisesRegex(ValueError, "quality result schema mismatch"):
                    _report_command(
                        ["--artifact-root", str(root), "--output", str(output)]
                    )
            self.assertFalse(output.exists())

    def test_content_crop_rejects_non_rectangular_mask(self):
        target = torch.zeros(3, 1, 16, 16)
        mask = torch.ones(1, 16, 16, dtype=torch.bool)
        mask[0, 4, 4] = False
        with self.assertRaisesRegex(ValueError, "rectangular"):
            _content_crop(target, target, mask)


if __name__ == "__main__":
    unittest.main()
