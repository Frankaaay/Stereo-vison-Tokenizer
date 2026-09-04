"""Stage A artifact validation and scorecard report command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (
    DA3_CHECKPOINT_SHA256,
    DA3_SOURCE_SHA,
    LAS2_H_CHECKPOINT_SHA256,
    LAS2_H_SOURCE_SHA,
    VGG16_CHECKPOINT_SHA256,
)
from .contract import sha256_file
from .metrics import (
    DYNAMIC_FLOW_MIN_PX,
    FLOW_FB_ABSOLUTE_THRESHOLD_PX2,
    FLOW_FB_RELATIVE_THRESHOLD,
    STATIC_FLOW_MAX_PX,
)


def _report_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tokenizer_stage_a report")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.artifact_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    quality_paths = sorted((root / "quality").glob("*.json"))
    benchmark_paths = sorted((root / "benchmark").glob("*.json"))
    if len(quality_paths) != 10 or len(benchmark_paths) != 2:
        raise ValueError("Stage A report requires exactly 10 quality and 2 benchmark JSON files")
    quality = [json.loads(path.read_text()) for path in quality_paths]
    benchmarks = [json.loads(path.read_text()) for path in benchmark_paths]

    def require_sha256(value, label):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{label} must be one SHA256 digest")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{label} must be hexadecimal") from error

    def require_metric_backbone(environment, label):
        backbone = environment.get("metric_backbone")
        if not isinstance(backbone, dict):
            raise ValueError(f"{label} metric backbone provenance is missing")
        if (
            backbone.get("name") != "torchvision.vgg16.IMAGENET1K_V1"
            or backbone.get("role") != "torchmetrics LPIPS VGG feature backbone"
            or backbone.get("preprocessing")
            != "torchmetrics LPIPS vgg normalize=False on RGB [-1,1]"
        ):
            raise ValueError(f"{label} metric backbone contract mismatch")
        require_sha256(backbone.get("sha256"), f"{label} VGG16 hash")
        if backbone["sha256"] != VGG16_CHECKPOINT_SHA256:
            raise ValueError(f"{label} VGG16 hash does not match the frozen contract")
        if not str(backbone.get("path", "")):
            raise ValueError(f"{label} VGG16 path is missing")
        return backbone

    def require_summary(container, name, expected_count, label):
        summary = container.get(name)
        if not isinstance(summary, dict):
            raise ValueError(f"{label} is missing metric {name}")
        if int(summary.get("count", -1)) != expected_count:
            raise ValueError(f"{label}/{name} sample count mismatch")
        for field in ("mean", "p50", "p90", "p99"):
            value = summary.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{label}/{name}/{field} must be finite")
        if not summary["p50"] <= summary["p90"] <= summary["p99"]:
            raise ValueError(f"{label}/{name} percentiles are not monotonic")
        return summary

    def require_nonempty_summary(container, name, maximum_count, label):
        summary = container.get(name)
        if not isinstance(summary, dict):
            raise ValueError(f"{label} is missing metric {name}")
        count = int(summary.get("count", 0))
        if not 1 <= count <= maximum_count:
            raise ValueError(f"{label}/{name} coverage is empty or oversized")
        for field in ("mean", "p50", "p90", "p99"):
            value = summary.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{label}/{name}/{field} must be finite")
        return summary

    def require_rgb_v2_metrics(container, expected_count, *, four_frame, label):
        required = (
            "raw_rgb_l1",
            "raw_rgb_mse",
            "clamped_rgb_l1",
            "clamped_rgb_mse",
            "clamped_psnr_db",
            "clamped_ssim",
            "clamped_lpips",
            "rgb_valid_ratio",
            "rgb_out_of_range_pixel_ratio",
            "rgb_overshoot_positive_p50",
            "rgb_overshoot_positive_p90",
            "rgb_overshoot_positive_p99",
            "rgb_overshoot_positive_max",
        )
        summaries = {
            name: require_summary(container, name, expected_count, label)
            for name in required
        }
        for field in ("mean", "p50", "p90", "p99"):
            if summaries["clamped_rgb_l1"][field] > summaries["raw_rgb_l1"][field] + 1e-8:
                raise ValueError(f"{label}: clamped L1 exceeds raw L1")
            if summaries["clamped_rgb_mse"][field] > summaries["raw_rgb_mse"][field] + 1e-8:
                raise ValueError(f"{label}: clamped MSE exceeds raw MSE")
            ratio = summaries["rgb_out_of_range_pixel_ratio"][field]
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{label}: out-of-range ratio is outside [0,1]")
            if not (
                summaries["rgb_overshoot_positive_p50"][field]
                <= summaries["rgb_overshoot_positive_p90"][field]
                <= summaries["rgb_overshoot_positive_p99"][field]
                <= summaries["rgb_overshoot_positive_max"][field]
            ):
                raise ValueError(f"{label}: overshoot summaries are not monotonic")
        legacy = {
            "rgb_l1", "rgb_mse", "psnr_db", "ssim", "lpips",
            "temporal_delta_l1", "temporal_delta_lpips",
        }
        if legacy.intersection(container) or any(
            name.startswith("temporal_delta_") for name in container
        ):
            raise ValueError(f"{label}: legacy or mixed-domain metrics are present")
        if four_frame:
            for name in (
                "clamped_temporal_delta_l1",
                "clamped_temporal_delta_lpips",
                "clamped_temporal_delta_l1_pair_01",
                "clamped_temporal_delta_lpips_pair_01",
                "clamped_temporal_delta_l1_pair_12",
                "clamped_temporal_delta_lpips_pair_12",
                "clamped_temporal_delta_l1_pair_23",
                "clamped_temporal_delta_lpips_pair_23",
            ):
                require_summary(container, name, expected_count, label)
            for name in (
                "optical_flow_valid_ratio",
                "static_flow_valid_ratio",
                "dynamic_flow_valid_ratio",
                "clamped_optical_flow_warp_l1",
            ):
                require_summary(container, name, expected_count, label)
            for name in (
                "clamped_static_flicker_l1",
                "clamped_motion_flow_epe_px",
            ):
                require_nonempty_summary(container, name, expected_count, label)
            for base in (
                "clamped_optical_flow_warp_l1",
                "clamped_static_flicker_l1",
                "clamped_motion_flow_epe_px",
            ):
                for pair in ("pair_01", "pair_12", "pair_23"):
                    require_nonempty_summary(
                        container, f"{base}_{pair}", expected_count, label
                    )

    status_path = root / "job-status.json"
    if not status_path.is_file():
        raise FileNotFoundError("A1 report requires job-status.json")
    job_status = json.loads(status_path.read_text())
    if job_status.get("schema") != "stereo-tokenizer-stage-a1-job-status-v1":
        raise ValueError("job status schema mismatch")
    expected_artifacts = {
        str(path.relative_to(root)) for path in (*quality_paths, *benchmark_paths)
    }
    jobs = job_status.get("jobs", [])
    actual_artifacts = {job.get("artifact") for job in jobs}
    if len(jobs) != len(expected_artifacts) or actual_artifacts != expected_artifacts:
        raise ValueError("job status does not cover every result artifact exactly once")
    for job in jobs:
        if job.get("state") != "COMPLETED" or int(job.get("exit_code", -1)) != 0:
            raise ValueError("one or more formal Stage A jobs did not complete successfully")
        if not str(job.get("job_id", "")):
            raise ValueError("job status is missing one Slurm job ID")
        artifact = root / job["artifact"]
        require_sha256(job.get("sha256"), f"{artifact} status hash")
        if sha256_file(artifact) != job["sha256"]:
            raise ValueError(f"result artifact hash mismatch: {artifact}")
        log_path = root / str(job.get("log", ""))
        if not log_path.is_file():
            raise FileNotFoundError(f"formal job log is missing: {log_path}")
        require_sha256(job.get("log_sha256"), f"{log_path} status hash")
        if sha256_file(log_path) != job["log_sha256"]:
            raise ValueError(f"formal job log hash mismatch: {log_path}")

    expected = {
        ("umi", "stereo", None),
        *(("umi", "mono", camera) for camera in (
            "observation.images.cam_head_left",
            "observation.images.cam_head_right",
            "observation.images.cam_left_wrist_left",
            "observation.images.cam_left_wrist_right",
            "observation.images.cam_right_wrist_left",
            "observation.images.cam_right_wrist_right",
        )),
        ("libero", "mono", "observation.images.cam_head_left"),
        ("libero", "mono", "observation.images.cam_left_wrist_left"),
        ("hy", "mono", None),
    }
    actual = set()
    visualization_slots = set()
    source_fingerprints = set()
    environment_fingerprints = set()
    checkpoint_fingerprint = None
    checkpoint_sha256 = None
    flow_fingerprint = None
    selection_rows = {}
    for result in quality:
        if result.get("schema") != "stereo-tokenizer-stage-a1-result-v3":
            raise ValueError("quality result schema mismatch")
        dataset = result["dataset"]
        key = (dataset["dataset_id"], dataset["eye_mode"], dataset["camera_key"])
        actual.add(key)
        expected_count = 1024 if dataset["dataset_id"] in {"umi", "hy"} else 256
        expected_modes = {
            *(f"{dataset['eye_mode']}/single_frame/source_{index}" for index in range(4)),
            f"{dataset['eye_mode']}/four_frame",
        }
        if result.get("status") != "formal" or dataset["sample_count"] != expected_count:
            raise ValueError(f"non-formal or wrong sample count for {key}")
        if result.get("posterior") != "mean" or result.get("quality_precision") != "fp32":
            raise ValueError(f"quality precision/posterior contract mismatch for {key}")
        if set(result.get("requested_modes", [])) != expected_modes:
            raise ValueError(f"mode coverage mismatch for {key}")
        if result.get("single_frame_source_indices") != [0, 1, 2, 3]:
            raise ValueError(f"single-frame source coverage mismatch for {key}")
        checkpoint = result["checkpoint"]
        require_sha256(checkpoint.get("sha256"), "checkpoint hash")
        counters = checkpoint.get("stereo_update_counters", {})
        if not isinstance(counters, dict) or not isinstance(
            counters.get("generator_updates"), int
        ):
            raise ValueError("checkpoint generator update counter is missing")
        current_checkpoint_fingerprint = json.dumps(checkpoint, sort_keys=True)
        if checkpoint_fingerprint is None:
            checkpoint_fingerprint = current_checkpoint_fingerprint
            checkpoint_sha256 = checkpoint["sha256"]
        elif current_checkpoint_fingerprint != checkpoint_fingerprint:
            raise ValueError("checkpoint provenance mismatch across quality results")
        require_sha256(dataset.get("selection_sha256"), f"{key} selection semantic hash")
        require_sha256(dataset.get("selection_file_sha256"), f"{key} selection file hash")
        selection_path = Path(dataset["selection_path"])
        if not selection_path.is_file() or sha256_file(selection_path) != dataset["selection_file_sha256"]:
            raise ValueError(f"selection file drift for {key}")
        selection_payload = json.loads(selection_path.read_text())
        if (
            int(selection_payload.get("sample_count", -1)) != expected_count
            or len(selection_payload.get("records", [])) != expected_count
        ):
            raise ValueError(f"selection sample count mismatch for {key}")
        decode_validation = selection_payload.get("decode_validation", {})
        if int(decode_validation.get("accepted_count", -1)) != expected_count:
            raise ValueError(f"selection decode audit mismatch for {key}")
        require_sha256(
            decode_validation.get("rejected_episode_ids_sha256"),
            f"{key} rejected episode IDs hash",
        )
        identity = dataset.get("identity_contract", {})
        require_sha256(identity.get("sha256"), f"{key} identity contract hash")
        require_sha256(identity.get("source_manifest_sha256"), f"{key} manifest hash")
        manifest_sha = identity["source_manifest_sha256"]
        if dataset["dataset_id"] == "hy":
            if (
                dataset.get("data_backend") != "hy_lance_manifest"
                or dataset.get("camera_key") is not None
                or dataset.get("excluded_source_groups", {}).get("groups")
                != ["table_014"]
                or not dataset.get("included_source_groups")
                or "table_014" in dataset["included_source_groups"]
            ):
                raise ValueError("Hy manifest backend/exclusion contract mismatch")
            hy_manifest = dataset.get("hy_manifest", {})
            require_sha256(hy_manifest.get("sha256"), "Hy production manifest hash")
            hy_manifest_path = Path(str(hy_manifest.get("path", "")))
            if (
                not hy_manifest_path.is_file()
                or sha256_file(hy_manifest_path) != hy_manifest["sha256"]
                or selection_payload.get("hy_manifest") != hy_manifest
                or selection_payload.get("excluded_source_groups")
                != dataset["excluded_source_groups"]
                or selection_payload.get("included_source_groups")
                != dataset["included_source_groups"]
            ):
                raise ValueError("Hy production manifest or exclusion provenance drifted")
            manifest_sha = hy_manifest["sha256"]
        else:
            config_hashes = dataset.get("canonical_config_sha256", {})
            if not config_hashes:
                raise ValueError(f"canonical config hashes are missing for {key}")
            for config_path, digest in config_hashes.items():
                require_sha256(digest, f"{config_path} config hash")
                if sha256_file(Path(config_path)) != digest:
                    raise ValueError(f"canonical config drift for {config_path}")
            loader_sha = dataset.get("canonical_loader", {}).get("git_sha")
            if loader_sha != "d51377ac450b0066bc0c8eb13939bcfae47275ff":
                raise ValueError("canonical loader SHA mismatch")
        selection_rows.setdefault(
            dataset["dataset_id"],
            {
                "semantic_sha256": dataset["selection_sha256"],
                "file_sha256": dataset["selection_file_sha256"],
                "manifest_sha256": manifest_sha,
                "sample_count": expected_count,
                "decode_checked": int(
                    decode_validation.get("checked_candidate_count", -1)
                ),
                "decode_rejected": int(
                    decode_validation.get("rejected_count", -1)
                ),
                "rejected_ids_sha256": decode_validation[
                    "rejected_episode_ids_sha256"
                ],
                "excluded": (
                    dataset.get("excluded_source_groups")
                    if dataset["dataset_id"] == "hy"
                    else None
                ),
            },
        )
        if selection_rows[dataset["dataset_id"]]["semantic_sha256"] != dataset["selection_sha256"]:
            raise ValueError("one dataset used multiple selections")

        teacher = result.get("teacher", {})
        if dataset["eye_mode"] == "mono":
            if teacher.get("source_sha") != DA3_SOURCE_SHA or teacher.get("checkpoint_sha256") != DA3_CHECKPOINT_SHA256:
                raise ValueError(f"DA3 provenance mismatch for {key}")
        elif (
            teacher.get("backend") != "las2_h"
            or teacher.get("source_sha") != LAS2_H_SOURCE_SHA
            or teacher.get("checkpoint_sha256") != LAS2_H_CHECKPOINT_SHA256
        ):
            raise ValueError("LAS2-H provenance mismatch")

        flow_teacher = result.get("flow_teacher", {})
        if (
            flow_teacher.get("name") != "torchvision.raft_large"
            or flow_teacher.get("architecture") != "RAFT-Large"
            or flow_teacher.get("precision") != "fp32"
            or flow_teacher.get("flow_unit") != "content-crop pixels"
            or float(flow_teacher.get("static_flow_max_px", -1))
            != STATIC_FLOW_MAX_PX
            or float(flow_teacher.get("dynamic_flow_min_px", -1))
            != DYNAMIC_FLOW_MIN_PX
            or float(flow_teacher.get("forward_backward_relative_threshold", -1))
            != FLOW_FB_RELATIVE_THRESHOLD
            or float(flow_teacher.get("forward_backward_absolute_threshold_px2", -1))
            != FLOW_FB_ABSOLUTE_THRESHOLD_PX2
        ):
            raise ValueError(f"RAFT flow contract mismatch for {key}")
        require_sha256(
            flow_teacher.get("checkpoint_sha256"), f"{key} RAFT checkpoint hash"
        )
        current_flow_fingerprint = json.dumps(flow_teacher, sort_keys=True)
        if flow_fingerprint is None:
            flow_fingerprint = current_flow_fingerprint
        elif current_flow_fingerprint != flow_fingerprint:
            raise ValueError("RAFT provenance mismatch across quality results")

        parameters = result.get("tokenizer_parameters", {})
        if (
            int(parameters.get("total", 0)) <= 0
            or int(parameters.get("architecturally_trainable", 0)) <= 0
            or int(parameters.get("runtime_requires_grad", -1)) != 0
            or parameters.get("evaluation_state") != "eval_inference_mode_posterior_mean"
        ):
            raise ValueError(f"Tokenizer freeze/parameter provenance mismatch for {key}")
        if set(result.get("metrics", {})) != expected_modes:
            raise ValueError(f"metric mode coverage mismatch for {key}")
        for mode_id, mode in result["metrics"].items():
            if mode["sample_count"] != expected_count:
                raise ValueError(f"metric sample count mismatch for {key}")
            if int(mode.get("valid_rgb_values", 0)) <= 0 or int(mode.get("valid_teacher_pixels", 0)) <= 0:
                raise ValueError(f"empty metric mask for {key}")
            health = mode["output_health"]
            if (
                health["nan_count"]
                or health["inf_count"]
                or health.get("invalid_sample_count")
                or health.get("invalid_sample_ids")
            ):
                raise ValueError(f"invalid output in {key}")
            expected_health = {
                "all_value_count",
                "all_raw_min",
                "all_raw_max",
                "valid_value_count",
                "valid_raw_min",
                "valid_raw_max",
                "valid_pixel_count",
                "out_of_range_pixel_count",
                "out_of_range_pixel_ratio",
            }
            if not expected_health.issubset(health):
                raise ValueError(f"RGB v2 output health is incomplete for {key}")
            if {"value_count", "raw_min", "raw_max", "abs_gt_one_count", "abs_gt_one_ratio"}.intersection(health):
                raise ValueError(f"legacy output health is present for {key}")
            for name in ("all_raw_min", "all_raw_max", "valid_raw_min", "valid_raw_max", "out_of_range_pixel_ratio"):
                if not np.isfinite(health[name]):
                    raise ValueError(f"non-finite output health field {name} for {key}")
            if (
                int(health["all_value_count"]) <= 0
                or int(health["valid_value_count"]) != int(mode["valid_rgb_values"])
                or int(health["valid_pixel_count"]) * 3 != int(health["valid_value_count"])
                or not 0 <= int(health["out_of_range_pixel_count"]) <= int(health["valid_pixel_count"])
                or not 0.0 <= float(health["out_of_range_pixel_ratio"]) <= 1.0
                or abs(
                    float(health["out_of_range_pixel_ratio"])
                    - int(health["out_of_range_pixel_count"]) / int(health["valid_pixel_count"])
                ) > 1e-12
                or float(health["all_raw_min"]) > float(health["all_raw_max"])
                or float(health["valid_raw_min"]) > float(health["valid_raw_max"])
            ):
                raise ValueError(f"RGB v2 output health contract mismatch for {key}")
            for view_name, view_metrics in mode.get("per_view", {}).items():
                require_rgb_v2_metrics(
                    view_metrics,
                    expected_count,
                    four_frame=mode_id.endswith("/four_frame"),
                    label=f"{key}/{mode_id}/{view_name}",
                )
            require_rgb_v2_metrics(
                mode.get("per_sample_macro", {}),
                expected_count,
                four_frame=mode_id.endswith("/four_frame"),
                label=f"{key}/{mode_id}/macro",
            )
            if mode_id.endswith("/four_frame"):
                geometry_prefix = (
                    "reconstruction_teacher"
                    if dataset["eye_mode"] == "mono"
                    else "depth_head_teacher"
                )
                for scope_name, scope_metrics in (
                    *tuple(mode.get("per_view", {}).items()),
                    ("macro", mode.get("per_sample_macro", {})),
                ):
                    require_summary(
                        scope_metrics,
                        f"{geometry_prefix}_temporal_geometry_valid_ratio",
                        expected_count,
                        f"{key}/{mode_id}/{scope_name}",
                    )
                    require_nonempty_summary(
                        scope_metrics,
                        f"{geometry_prefix}_temporal_geometry_warp_l1",
                        expected_count,
                        f"{key}/{mode_id}/{scope_name}",
                    )
            teacher_invalid = mode.get("teacher_invalid_samples", [])
            if int(mode.get("teacher_invalid_count", -1)) != len(teacher_invalid):
                raise ValueError(f"teacher-invalid count mismatch for {key}")
            identities = set()
            for entry in teacher_invalid:
                if (
                    not isinstance(entry, dict)
                    or not str(entry.get("sample_id", ""))
                    or not str(entry.get("view", ""))
                    or entry.get("reason") != "empty_teacher_mask"
                ):
                    raise ValueError(f"invalid teacher exclusion record for {key}")
                identity = (entry["sample_id"], entry["view"])
                if identity in identities:
                    raise ValueError(f"duplicate teacher exclusion record for {key}")
                identities.add(identity)

        provenance = result.get("provenance", {})
        for name in ("cwd", "git_branch", "git_commit", "git_diff_sha256", "git_status_porcelain"):
            if name not in provenance:
                raise ValueError(f"quality provenance is missing {name}")
        require_sha256(provenance["git_diff_sha256"], "source diff hash")
        if len(provenance["git_commit"]) != 40:
            raise ValueError("source commit must be one full Git SHA")
        source_fingerprints.add(tuple(provenance[name] for name in (
            "cwd", "git_branch", "git_commit", "git_diff_sha256", "git_status_porcelain"
        )))
        environment = provenance.get("environment", {})
        if not environment.get("python", "").startswith("3.12.") or "H100" not in str(environment.get("gpu_name", "")):
            raise ValueError("formal Stage A quality must use Python 3.12 on H100")
        require_sha256(environment.get("uv_lock_sha256"), "quality uv.lock hash")
        require_metric_backbone(environment, "quality")
        environment_fingerprints.add(json.dumps(environment, sort_keys=True))

        cases = result.get("visualizations", [])
        wants_cases = key in {
            ("umi", "stereo", None),
            ("libero", "mono", "observation.images.cam_head_left"),
        }
        if wants_cases:
            expected_cases = {(slot, source) for slot in range(8) for source in range(4)}
            actual_cases = {(case["slot"], case["source_frame_index"]) for case in cases}
            if len(cases) != 32 or actual_cases != expected_cases:
                raise ValueError(f"fixed visualization coverage mismatch for {key}")
            visual_dir = Path(provenance["resolved_args"]["visualization_dir"])
            case_index = visual_dir / "cases.json"
            if not case_index.is_file() or json.loads(case_index.read_text()) != cases:
                raise ValueError(f"visualization index mismatch for {key}")
            for case in cases:
                for field in ("rgb_file", "geometry_file"):
                    image = visual_dir / case[field]
                    if not image.is_file() or image.stat().st_size == 0:
                        raise FileNotFoundError(f"visualization file missing: {image}")
                visualization_slots.add((dataset["dataset_id"], case["slot"]))
        elif cases:
            raise ValueError(f"unexpected visualizations for {key}")
    if actual != expected:
        raise ValueError(f"quality coverage mismatch: missing={expected-actual}, extra={actual-expected}")
    if visualization_slots != {
        *(("umi", index) for index in range(8)),
        *(("libero", index) for index in range(8)),
    }:
        raise ValueError("A1 report requires 8 fixed UMI and 8 fixed LIBERO visualizations")

    benchmark_eyes = set()
    for result in benchmarks:
        if result.get("schema") != "stereo-tokenizer-stage-a1-benchmark-v1":
            raise ValueError("benchmark result schema mismatch")
        dataset = result.get("dataset", {})
        benchmark_eyes.add(dataset.get("eye_mode"))
        if dataset.get("dataset_id") != "umi":
            raise ValueError("efficiency benchmark must use representative UMI input")
        if result.get("status") != "formal" or (
            result.get("warmup"), result.get("iterations"), result.get("repeats")
        ) != (20, 100, 3):
            raise ValueError("benchmark contract mismatch")
        if (
            result.get("precision") != "bf16"
            or int(result.get("batch_size", -1)) != 1
            or result.get("posterior") != "mean"
            or result.get("timing_scope") != "model_only_excludes_data_decode_and_teacher"
            or set(result.get("modes", {})) != {"single_frame", "four_frame"}
            or result.get("checkpoint", {}).get("sha256") != checkpoint_sha256
        ):
            raise ValueError("benchmark precision/scope/checkpoint mismatch")
        provenance = result.get("provenance", {})
        require_sha256(provenance.get("git_diff_sha256"), "benchmark source diff hash")
        source_fingerprints.add(tuple(provenance.get(name) for name in (
            "cwd", "git_branch", "git_commit", "git_diff_sha256", "git_status_porcelain"
        )))
        environment = provenance.get("environment", {})
        if not environment.get("python", "").startswith("3.12.") or "H100" not in str(environment.get("gpu_name", "")):
            raise ValueError("formal benchmark must use Python 3.12 on H100")
        require_sha256(environment.get("uv_lock_sha256"), "benchmark uv.lock hash")
        require_metric_backbone(environment, "benchmark")
        environment_fingerprints.add(json.dumps(environment, sort_keys=True))
    if benchmark_eyes != {"mono", "stereo"}:
        raise ValueError("benchmark must cover UMI mono and stereo")
    if len(source_fingerprints) != 1 or len(environment_fingerprints) != 1:
        raise ValueError("formal jobs used inconsistent source or environments")

    source = quality[0]["provenance"]
    source_patch = root / "source.patch"
    if (
        not source_patch.is_file()
        or sha256_file(source_patch) != source["git_diff_sha256"]
    ):
        raise ValueError("source.patch does not match the recorded Git diff SHA256")
    environment = source["environment"]
    metric_backbone = require_metric_backbone(environment, "formal")
    metric_backbone_path = Path(metric_backbone["path"])
    if (
        not metric_backbone_path.is_file()
        or sha256_file(metric_backbone_path) != metric_backbone["sha256"]
    ):
        raise ValueError("frozen LPIPS VGG16 file is missing or has changed")
    checkpoint = quality[0]["checkpoint"]
    flow_teacher = quality[0]["flow_teacher"]
    flow_checkpoint_path = Path(flow_teacher["checkpoint"])
    if (
        not flow_checkpoint_path.is_file()
        or sha256_file(flow_checkpoint_path) != flow_teacher["checkpoint_sha256"]
    ):
        raise ValueError("frozen RAFT checkpoint is missing or has changed")
    parameters = quality[0]["tokenizer_parameters"]
    package_text = ", ".join(
        f"{name}={version}" for name, version in sorted(environment["packages"].items())
    )
    status_text = source["git_status_porcelain"].replace("\n", "; ")
    lines = [
        "# Stereo Tokenizer Stage A1 Baseline（Preliminary）",
        "",
        "> 状态：PRELIMINARY。v6 增加 HY（显式排除 Table014）、RAFT warp/static flicker/motion consistency 和 teacher-relative temporal geometry；rFID 暂不执行。",
        "",
        "> v4 失效原因：raw L1/MSE/PSNR/LPIPS 与 clamp 后 SSIM 不在同一图像域，且旧越界阈值 abs(output)>1 与合法域 [-0.5,0.5] 不一致。v4 artifact 仅保留作审计。",
        "",
        "## 实验合同与 provenance",
        "",
        f"- Artifact 根目录：`{root}`",
        f"- 实际 cwd：`{source['cwd']}`",
        f"- Git branch / commit：`{source['git_branch']}` / `{source['git_commit']}`",
        f"- 未提交代码 diff：`{source_patch}`；SHA256：`{source['git_diff_sha256']}`",
        f"- `git status --porcelain`：`{status_text}`",
        f"- Checkpoint：`{checkpoint['path']}`",
        f"- Checkpoint SHA256：`{checkpoint['sha256']}`；global_step={checkpoint['global_step']}；epoch={checkpoint['epoch']}",
        f"- 直接训练计数：`{json.dumps(checkpoint['stereo_update_counters'], sort_keys=True)}`",
        "- 质量：FP32；效率：BF16；posterior mean；Tokenizer `eval + inference_mode` 且运行时冻结。",
        f"- Python：`{environment['python'].split()[0]}`；GPU：`{environment['gpu_name']}`；CUDA：`{environment['torch_cuda']}`；cuDNN：`{environment['cudnn']}`",
        f"- `uv.lock` SHA256：`{environment['uv_lock_sha256']}`",
        f"- LPIPS VGG16：`{metric_backbone['path']}`；SHA256：`{metric_backbone['sha256']}`；预处理：`{metric_backbone['preprocessing']}`",
        f"- RAFT-Large：`{flow_teacher['checkpoint']}`；SHA256：`{flow_teacher['checkpoint_sha256']}`；FP32；static≤{flow_teacher['static_flow_max_px']} px；dynamic≥{flow_teacher['dynamic_flow_min_px']} px。",
        f"- 关键包：{package_text}",
        f"- Tokenizer 参数：total={parameters['total']:,}；架构可训练={parameters['architecturally_trainable']:,}；运行时 requires_grad={parameters['runtime_requires_grad']:,}",
        f"- DA3：source `{DA3_SOURCE_SHA}`；weights `{DA3_CHECKPOINT_SHA256}`",
        f"- LAS2-H：source `{LAS2_H_SOURCE_SHA}`；weights `{LAS2_H_CHECKPOINT_SHA256}`",
        "",
        "### 数据与哈希",
        "",
        "| Dataset | Windows/cell | Decode checked/rejected | Explicit exclusion | Selection semantic SHA256 | Selection file SHA256 | Manifest SHA256 |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for dataset_id, row in sorted(selection_rows.items()):
        lines.append(
            f"| {dataset_id} | {row['sample_count']} | "
            f"{row['decode_checked']}/{row['decode_rejected']} | "
            f"{('none' if row['excluded'] is None else ','.join(row['excluded']['groups']) + ':' + str(row['excluded']['episode_count']))} | "
            f"`{row['semantic_sha256']}` | `{row['file_sha256']}` | "
            f"`{row['manifest_sha256']}` |"
        )
    lines.extend([
        "",
        "### 覆盖矩阵",
        "",
        "| Dataset | Eye | Camera/view cell | Windows | Modes | Macro inclusion |",
        "| --- | --- | --- | ---: | --- | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        camera = dataset["camera_key"] or "3 canonical stereo pairs"
        lines.append(
            f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {camera} | "
            f"{dataset['sample_count']} | single source 0/1/2/3 + four-frame | yes |"
        )
    lines.extend([
        "",
        "## Clamp-domain RGB 图像质量（per camera/view）",
        "",
        "下表所有 L1/MSE/PSNR/SSIM/LPIPS 都使用 `prediction.clamp(-0.5, 0.5)`；PSNR data_range=1.0。",
        "",
        "| Dataset | Eye | Camera/view | Mode | L1 mean | P50 | P90 | P99 | MSE | PSNR | SSIM | LPIPS | RGB mask |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            for view_name, metric in mode["per_view"].items():
                lines.append(
                    f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | {mode_id} | "
                    f"{metric['clamped_rgb_l1']['mean']:.6f} | {metric['clamped_rgb_l1']['p50']:.6f} | "
                    f"{metric['clamped_rgb_l1']['p90']:.6f} | {metric['clamped_rgb_l1']['p99']:.6f} | "
                    f"{metric['clamped_rgb_mse']['mean']:.6f} | {metric['clamped_psnr_db']['mean']:.3f} | "
                    f"{metric['clamped_ssim']['mean']:.6f} | {metric['clamped_lpips']['mean']:.6f} | "
                    f"{metric['rgb_valid_ratio']['mean']:.6f} |"
                )
    lines.extend([
        "",
        "### Dataset/eye/mode 等权 macro（clamp-domain）",
        "",
        "| Dataset | Eye | Mode | RGB L1 | MSE | PSNR | SSIM | LPIPS |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    macro_cells = {}
    for result in quality:
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            key = (dataset["dataset_id"], dataset["eye_mode"], mode_id)
            macro_cells.setdefault(key, []).append(mode["per_sample_macro"])
    for key, cells in sorted(macro_cells.items()):
        means = {
            name: float(np.mean([cell[name]["mean"] for cell in cells]))
            for name in (
                "clamped_rgb_l1",
                "clamped_rgb_mse",
                "clamped_psnr_db",
                "clamped_ssim",
                "clamped_lpips",
            )
        }
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {means['clamped_rgb_l1']:.6f} | "
            f"{means['clamped_rgb_mse']:.6f} | {means['clamped_psnr_db']:.3f} | "
            f"{means['clamped_ssim']:.6f} | {means['clamped_lpips']:.6f} |"
        )
    lines.extend([
        "",
        "## Raw RGB 数值稳定性诊断",
        "",
        "Raw L1/MSE 只用于诊断 decoder 数值稳定性，不作为正式图像质量结论。",
        "",
        "| Dataset | Eye | Camera/view | Mode | Raw L1 mean/P50/P90/P99 | Raw MSE mean/P50/P90/P99 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            for view_name, metric in mode["per_view"].items():
                l1 = metric["raw_rgb_l1"]
                mse = metric["raw_rgb_mse"]
                lines.append(
                    f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | {mode_id} | "
                    f"{l1['mean']:.6f}/{l1['p50']:.6f}/{l1['p90']:.6f}/{l1['p99']:.6f} | "
                    f"{mse['mean']:.6f}/{mse['p50']:.6f}/{mse['p90']:.6f}/{mse['p99']:.6f} |"
                )
    lines.extend([
        "",
        "## RGB 越界与 overshoot 诊断",
        "",
        "越界像素定义为 valid 时空像素中任一 RGB channel 超出 [-0.5,0.5]。每个 sample/view 先对正 overshoot `max_channel(relu(abs(output)-0.5))` 的全部越界像素精确求 P50/P90/P99/max，再汇总样本分布；不是全局近似分位数。无越界样本的 positive count 和这些统计均为 0。",
        "",
        "| Dataset | Eye | Camera/view | Mode | OOR ratio mean/P50/P90/P99 | sample P50 mean/P50/P90/P99 | sample P90 mean/P50/P90/P99 | sample P99 mean/P50/P90/P99 | sample max mean/P50/P90/P99 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            for view_name, metric in mode["per_view"].items():
                def render_summary(name):
                    value = metric[name]
                    return (
                        f"{value['mean']:.8f}/{value['p50']:.8f}/"
                        f"{value['p90']:.8f}/{value['p99']:.8f}"
                    )
                lines.append(
                    f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | {mode_id} | "
                    f"{render_summary('rgb_out_of_range_pixel_ratio')} | "
                    f"{render_summary('rgb_overshoot_positive_p50')} | "
                    f"{render_summary('rgb_overshoot_positive_p90')} | "
                    f"{render_summary('rgb_overshoot_positive_p99')} | "
                    f"{render_summary('rgb_overshoot_positive_max')} |"
                )
    lines.extend([
        "",
        "## Four-frame 时间一致性（clamp-domain）",
        "",
        "先逐帧 clamp 到 [-0.5,0.5]，再计算相邻帧 temporal delta。",
        "",
        "| Dataset | Eye | Camera/view | Δ L1 | Δ LPIPS | Δ01 L1/LPIPS | Δ12 L1/LPIPS | Δ23 L1/LPIPS |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        mode = result["metrics"][f"{dataset['eye_mode']}/four_frame"]
        for view_name, metric in mode["per_view"].items():
            pairs = []
            for pair in ("pair_01", "pair_12", "pair_23"):
                pairs.append(
                    f"{metric['clamped_temporal_delta_l1_' + pair]['mean']:.6f}/"
                    f"{metric['clamped_temporal_delta_lpips_' + pair]['mean']:.6f}"
                )
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | "
                f"{metric['clamped_temporal_delta_l1']['mean']:.6f} | "
                f"{metric['clamped_temporal_delta_lpips']['mean']:.6f} | "
                f"{pairs[0]} | {pairs[1]} | {pairs[2]} |"
            )
    lines.extend([
        "",
        "### RAFT flow-aware 时间指标",
        "",
        "Warp L1 使用目标视频的 backward flow 对齐相邻帧；static flicker 仅统计目标 flow≤0.5 px；motion EPE 仅统计目标 flow≥1.0 px。0.5–1.0 px 灰区不进入后二者。",
        "",
        "| Dataset | Eye | Camera/view | Warp L1 | Static flicker L1 | Motion EPE px | Flow/static/dynamic coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        mode = result["metrics"][f"{dataset['eye_mode']}/four_frame"]
        for view_name, metric in mode["per_view"].items():
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | "
                f"{metric['clamped_optical_flow_warp_l1']['mean']:.6f} | "
                f"{metric['clamped_static_flicker_l1']['mean']:.6f} | "
                f"{metric['clamped_motion_flow_epe_px']['mean']:.6f} | "
                f"{metric['optical_flow_valid_ratio']['mean']:.4f}/"
                f"{metric['static_flow_valid_ratio']['mean']:.4f}/"
                f"{metric['dynamic_flow_valid_ratio']['mean']:.4f} |"
            )
    lines.extend([
        "",
        "## Teacher-relative 几何（非真实 GT accuracy）",
        "",
        "| Dataset | Eye | Camera/view | Mode | Metric kind | log-L1 | RMSE | SILog | Mask coverage | Valid samples |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        prefix = "reconstruction_teacher" if dataset["eye_mode"] == "mono" else "depth_head_teacher"
        for mode_id, mode in result["metrics"].items():
            for view_name, metric in mode["per_view"].items():
                l1 = metric.get(prefix + "_relative_log_l1")
                rmse = metric.get(prefix + "_relative_log_rmse")
                silog = metric.get(prefix + "_relative_log_silog")
                coverage = metric.get(prefix + "_valid_ratio")
                render = lambda value: "N/A" if value is None else f"{value['mean']:.6f}"
                lines.append(
                    f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | {mode_id} | {prefix} | "
                    f"{render(l1)} | {render(rmse)} | {render(silog)} | "
                    f"{render(coverage)} | {0 if l1 is None else l1['count']} |"
                )
    lines.extend([
        "",
        "### Teacher-relative temporal geometry consistency",
        "",
        "| Dataset | Eye | Camera/view | Flow-aligned log-geometry L1 | Valid coverage | Pairs 01/12/23 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        prefix = "reconstruction_teacher" if dataset["eye_mode"] == "mono" else "depth_head_teacher"
        mode = result["metrics"][f"{dataset['eye_mode']}/four_frame"]
        for view_name, metric in mode["per_view"].items():
            name = f"{prefix}_temporal_geometry_warp_l1"
            pair_values = "/".join(
                f"{metric[f'{name}_pair_{pair}']['mean']:.6f}"
                for pair in ("01", "12", "23")
            )
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | "
                f"{metric[name]['mean']:.6f} | "
                f"{metric[f'{prefix}_temporal_geometry_valid_ratio']['mean']:.4f} | "
                f"{pair_values} |"
            )
    lines.extend([
        "",
        "## Bottleneck 与效率",
        "",
        "| Eye | Mode | Encode P50/P90 ms | Posterior mean P50/P90 ms | Decode P50/P90 ms | E2E P50/P90 ms | samples/s | frames/s | Peak alloc/reserved GiB |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ])
    for result in sorted(benchmarks, key=lambda value: value["dataset"]["eye_mode"]):
        for mode_name, mode in result["modes"].items():
            encode = mode["encode_including_posterior_mean"]
            posterior = mode["cached_posterior_mean"]
            decode = mode["decode"]
            timing = mode["end_to_end"]
            lines.append(
                f"| {result['dataset']['eye_mode']} | {mode_name} | "
                f"{encode['p50_ms']:.3f}/{encode['p90_ms']:.3f} | "
                f"{posterior['p50_ms']:.3f}/{posterior['p90_ms']:.3f} | "
                f"{decode['p50_ms']:.3f}/{decode['p90_ms']:.3f} | "
                f"{timing['p50_ms']:.3f}/{timing['p90_ms']:.3f} | "
                f"{mode['throughput']['samples_per_second']:.3f} | "
                f"{mode['throughput']['frames_per_second']:.3f} | "
                f"{timing['peak_allocated_bytes'] / 2**30:.3f}/"
                f"{timing['peak_reserved_bytes'] / 2**30:.3f} |"
            )
    lines.extend([
        "",
        "### Latent ABI",
        "",
        "| Dataset | Eye | Camera | Mode | Input shape/dtype | Latent shape/dtype | C | Tokens/window | Tokens/input frame | Spatial × | Temporal × | View × |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            abi = mode["latent_abi"]
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | "
                f"{dataset['camera_key'] or 'three canonical pairs'} | {mode_id} | "
                f"`{abi['input_shape_without_batch']}` / `{abi['input_dtype']}` | "
                f"`{abi['latent_shape_without_batch']}` / `{abi['latent_dtype']}` | "
                f"{abi['latent_channels']} | {abi['tokens_per_window']} | "
                f"{abi['tokens_per_input_frame']:.3f} | {abi['spatial_compression_ratio']:.1f} | "
                f"{abi['temporal_compression_ratio']:.1f} | {abi['view_compression_ratio']:.1f} |"
            )
    lines.extend([
        "",
        "## 输出健康、失败与排除样本",
        "",
        "| Dataset | Eye | Camera | Mode | NaN | Inf | Invalid | Teacher-empty | All raw min/max | Valid raw min/max | OOR pixels/valid pixels | OOR ratio | Valid RGB values | Valid teacher pixels |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ])
    teacher_exclusions = []
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            health = mode["output_health"]
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | "
                f"{dataset['camera_key'] or 'three canonical pairs'} | {mode_id} | "
                f"{health['nan_count']} | {health['inf_count']} | "
                f"{health['invalid_sample_count']} | {mode['teacher_invalid_count']} | "
                f"{health['all_raw_min']:.6f}/{health['all_raw_max']:.6f} | "
                f"{health['valid_raw_min']:.6f}/{health['valid_raw_max']:.6f} | "
                f"{health['out_of_range_pixel_count']}/{health['valid_pixel_count']} | "
                f"{health['out_of_range_pixel_ratio']:.8f} | {mode['valid_rgb_values']} | "
                f"{mode['valid_teacher_pixels']} |"
            )
            teacher_exclusions.extend(
                {
                    "dataset": dataset["dataset_id"],
                    "eye": dataset["eye_mode"],
                    "camera": dataset["camera_key"],
                    "mode": mode_id,
                    **entry,
                }
                for entry in mode["teacher_invalid_samples"]
            )
    lines.extend([
        "",
        "Teacher-empty view/frame 不影响同一固定窗口的 RGB 指标；该 view 的 teacher-relative error 缺失，几何汇总的 valid-sample count 会相应减少。",
    ])
    for entry in sorted(teacher_exclusions, key=lambda value: (
        value["dataset"], value["eye"], value["camera"] or "",
        value["mode"], value["sample_id"], value["view"]
    )):
        lines.append(
            f"- teacher exclusion: dataset={entry['dataset']}, eye={entry['eye']}, "
            f"camera={entry['camera'] or 'three canonical pairs'}, mode={entry['mode']}, "
            f"sample={entry['sample_id']}, view={entry['view']}, reason={entry['reason']}"
        )
    lines.extend([
        "",
        "每个 selection 的 decode checked/rejected 与 rejected episode IDs SHA256 已记录在数据表；完整排除原因保存在 selection JSON。",
        "",
        "### 固定案例",
        "",
        "共 16 个固定窗口：UMI 8、LIBERO 8；每个窗口保存四个 source position 的原图/重建与几何图，`cases.json` 和每个 PNG 均已在报告生成时核验。",
        "",
        "## 几何口径",
        "",
        "- Mono：DA3 分别推理原图与重建图，报告 `reconstruction_teacher_relative_*`。",
        "- Stereo：decoder 不重建右眼，报告 `depth_head_teacher_relative_*`；不称为 stereo 重建精度。",
        "- 没有独立真实 depth/disparity GT，因此本报告不声称真实几何 accuracy。",
        "",
        "## 阻断与未完成项",
        "",
        "- 每个 selection 的 decode checked/rejected 与 rejected episode IDs SHA256 已记录；完整排除原因保存在 selection JSON。",
        "- HY：通过训练同款 manifest reader 读取；Table014 因当前资产缺失被显式排除，排除数量和 episode IDs hash 写入 selection。",
        "- rFID：按本轮范围暂不执行。",
        "- rFVD：N/A，现有冻结 I3D-FVD 实现不支持本项目原生 4 帧合同；扩帧/插帧会改变评测对象。",
        "- FVMD：N/A，尚无经验证适用于原生 4 帧的冻结实现。",
        "",
        "## 决策",
        "",
        "1. **值得继续，但仍需补 rFID 与 Table014。** 当前结果可作为扩展后的 preliminary Stage A，不能表述成完整最终标准。",
        "2. **最大风险：** 当前几何指标只有 teacher-relative 证据；若误写为真实 depth/disparity accuracy，会得到错误结论。",
        "3. **最缺的关键证据：** 独立真实几何 GT、rFID，以及 Table014 的可读资产。",
        "4. **下一步：** 固定同一 selection 对 baseline/candidate 重跑，并补齐 Table014 后保持其他合同不变复测 HY。",
        "5. **置信度：80%（中等）。** 对已报告数字和可复现合同置信度较高；因 rFID、Table014 与独立几何 GT 缺失，不给高置信度。",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "quality_results": 10, "benchmarks": 2}, indent=2))
