#!/usr/bin/env python3
"""Compare bidirectional FoundationStereo 32/16/12-iteration teachers."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from stereo_tokenizer.lerobot_data import (  # noqa: E402
    VIEWS,
    LeRobotStereoDataset,
    sha256_file,
)
from stereo_tokenizer.online_gt import FoundationStereoOnlineTeacher  # noqa: E402


SCHEMA = "foundation-stereo-online-teacher-comparison-v1"
EXPECTED_VISUAL_TAGS = {
    "near_object",
    "far_object",
    "low_texture",
    "reflective",
    "occlusion",
    "motion_blur",
    "multi_task_scene",
}


def _distributed_device():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), world_size, local_rank


def _read_selection(
    path: Path,
    dataset: LeRobotStereoDataset,
    expected_count: int,
    *,
    require_visual_review: bool = True,
):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "lerobot-teacher-selection-v1":
        raise ValueError("unsupported teacher selection schema")
    if require_visual_review and payload.get("review_status") != "approved":
        raise ValueError("teacher selection requires approved visual review")
    required_tags = set(payload.get("required_visual_tags", []))
    if require_visual_review and required_tags != EXPECTED_VISUAL_TAGS:
        raise ValueError("teacher selection required visual tags changed")
    coverage = payload.get("coverage_counts", {})
    entries = payload.get("samples", [])
    if len(entries) != expected_count:
        raise ValueError(
            f"teacher selection must contain exactly {expected_count} samples"
        )
    indices = [int(entry["dataset_index"]) for entry in entries]
    if len(set(indices)) != len(indices):
        raise ValueError("teacher selection contains duplicate dataset indices")
    if payload.get("episode_manifest_sha256") != sha256_file(dataset.manifest_path):
        raise ValueError("teacher selection episode manifest SHA256 mismatch")
    actual_coverage = Counter()
    for entry in entries:
        tags = entry.get("visual_tags", [])
        if not isinstance(tags, list):
            raise ValueError("teacher selection visual_tags must be a list")
        unknown = set(tags) - required_tags
        if unknown:
            raise ValueError(
                "teacher selection contains unknown visual tags: "
                + ", ".join(sorted(unknown))
            )
        actual_coverage.update(set(tags))
    declared_coverage = {
        tag: int(coverage.get(tag, 0)) for tag in sorted(required_tags)
    }
    if require_visual_review and declared_coverage != {
        tag: actual_coverage[tag] for tag in sorted(required_tags)
    }:
        raise ValueError("teacher selection coverage_counts do not match samples")
    missing_coverage = [
        tag for tag in sorted(required_tags) if actual_coverage[tag] < 1
    ]
    if require_visual_review and missing_coverage:
        raise ValueError(
            "teacher selection lacks required visual coverage: "
            + ", ".join(missing_coverage)
        )
    for entry, index in zip(entries, indices):
        record, start_frame = dataset._sample_address(index)
        expected = f"{record['episode_id']}:{start_frame:06d}"
        if entry.get("sample_id") != expected:
            raise ValueError(f"selection sample ID mismatch at index {index}")
    return payload, indices


def _batch(dataset, indices, device):
    samples = [dataset[index] for index in indices]
    return {
        "video": torch.stack([sample["video"] for sample in samples]).to(device),
        "sample_id": [sample["sample_id"] for sample in samples],
    }


def _final_mask(args, disparity, residual, base_valid):
    threshold = torch.maximum(
        residual.new_tensor(args.lr_error_abs_threshold_px),
        args.lr_error_relative_threshold * disparity,
    )
    return (
        base_valid
        & torch.isfinite(disparity)
        & torch.isfinite(residual)
        & (disparity >= args.disparity_min_px)
        & (disparity <= args.disparity_max_px)
        & (residual <= threshold)
    )


def _run_configuration(
    args,
    teacher,
    dataset,
    indices,
    valid_iters,
    device,
    visual_sample_ids,
):
    teacher.valid_iters = valid_iters
    warmup = _batch(dataset, indices[: args.sample_batch_size], device)
    teacher.infer(warmup["video"])
    torch.cuda.synchronize(device)
    del warmup
    outputs = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    pair_count = 0
    decode_seconds = 0.0
    teacher_seconds = 0.0
    for start in range(0, len(indices), args.sample_batch_size):
        selected = indices[start : start + args.sample_batch_size]
        decode_started = time.perf_counter()
        batch = _batch(dataset, selected, device)
        torch.cuda.synchronize(device)
        decode_seconds += time.perf_counter() - decode_started
        teacher_started = time.perf_counter()
        disparity, residual, base_valid = teacher.infer(batch["video"])
        torch.cuda.synchronize(device)
        teacher_seconds += time.perf_counter() - teacher_started
        valid = _final_mask(args, disparity, residual, base_valid)
        pair_count += len(selected) * 12
        for item, sample_id in enumerate(batch["sample_id"]):
            result = {
                "sample_id": sample_id,
                "disparity": disparity[item].cpu().numpy().astype(np.float32),
                "residual": residual[item].cpu().numpy().astype(np.float32),
                "valid": valid[item].cpu().numpy(),
            }
            if sample_id in visual_sample_ids:
                rgb = (
                    (batch["video"][item].detach().cpu().numpy() + 0.5) * 255.0
                ).clip(0, 255).astype(np.uint8)
                result["rgb"] = rgb
            outputs.append(result)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return outputs, {
        "valid_iters": valid_iters,
        "sample_count": len(indices),
        "stereo_pair_count": pair_count,
        "wall_seconds": seconds,
        "samples_per_second": len(indices) / seconds,
        "total_seconds_per_stereo_pair": seconds / pair_count,
        "total_seconds_per_training_sample": seconds / len(indices),
        "decode_seconds": decode_seconds,
        "decode_seconds_per_training_sample": decode_seconds / len(indices),
        "teacher_seconds": teacher_seconds,
        "teacher_seconds_per_stereo_pair": teacher_seconds / pair_count,
        "teacher_seconds_per_training_sample": teacher_seconds / len(indices),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _quality(output, reference=None, pixel_stride=16):
    finite_count = 0
    pixel_count = 0
    valid_counts = np.zeros(3, dtype=np.int64)
    view_pixels = np.zeros(3, dtype=np.int64)
    residual_samples = [[], [], []]
    difference_samples = []
    intersection = 0
    union = 0
    temporal_samples = [[], [], []]
    for item_index, item in enumerate(output):
        disparity = item["disparity"].astype(np.float32)
        residual = item["residual"].astype(np.float32)
        valid = item["valid"]
        finite_count += np.isfinite(disparity).sum()
        pixel_count += disparity.size
        for view in range(3):
            valid_counts[view] += valid[view].sum()
            view_pixels[view] += valid[view].size
            selected_residual = residual[view][valid[view]]
            residual_samples[view].append(selected_residual[::pixel_stride])
            for frame in range(3):
                adjacent = valid[view, 0, frame] & valid[view, 0, frame + 1]
                if adjacent.any():
                    difference = np.abs(
                        disparity[view, 0, frame] - disparity[view, 0, frame + 1]
                    )
                    temporal_samples[view].append(difference[adjacent][::pixel_stride])
        if reference is not None:
            baseline = reference[item_index]
            if baseline["sample_id"] != item["sample_id"]:
                raise ValueError("teacher outputs changed sample order")
            baseline_disparity = baseline["disparity"].astype(np.float32)
            common = (
                baseline["valid"]
                & np.isfinite(disparity)
                & np.isfinite(baseline_disparity)
            )
            difference_samples.append(
                np.abs(disparity - baseline_disparity)[common][::pixel_stride]
            )
            intersection += np.logical_and(valid, baseline["valid"]).sum()
            union += np.logical_or(valid, baseline["valid"]).sum()
    return {
        "finite_count": finite_count,
        "pixel_count": pixel_count,
        "valid_counts": valid_counts,
        "view_pixels": view_pixels,
        "residual_samples": [
            np.concatenate(values) if values else np.empty(0, dtype=np.float32)
            for values in residual_samples
        ],
        "difference_samples": (
            np.concatenate(difference_samples)
            if difference_samples
            else np.empty(0, dtype=np.float32)
        ),
        "mask_intersection": intersection,
        "mask_union": union,
        "temporal_samples": [
            np.concatenate(values) if values else np.empty(0, dtype=np.float32)
            for values in temporal_samples
        ],
    }


def _save_visuals(root, valid_iters, outputs, visual_sample_ids):
    destination = root / "visualizations" / f"iters_{valid_iters}"
    destination.mkdir(parents=True, exist_ok=True)
    for item in outputs:
        if item["sample_id"] not in visual_sample_ids:
            continue
        disparity = item["disparity"].astype(np.float32)[:, 0]
        valid = item["valid"][:, 0]
        tiles = []
        for view in range(3):
            row = []
            for frame in range(4):
                values = disparity[view, frame]
                selected = values[valid[view, frame]]
                high = max(float(np.percentile(selected, 99)), 1.0) if selected.size else 1.0
                scaled = np.clip(values / high, 0, 1)
                image = cv2.applyColorMap(
                    (scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO
                )
                image[~valid[view, frame]] = 0
                mask = np.repeat(
                    (valid[view, frame] * 255).astype(np.uint8)[..., None],
                    3,
                    axis=2,
                )
                if "rgb" in item:
                    left = item["rgb"][view, 0, :, frame].transpose(1, 2, 0)
                    right = item["rgb"][view, 1, :, frame].transpose(1, 2, 0)
                    left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
                    right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
                    row.append(np.concatenate([left, right, image, mask], axis=1))
                else:
                    row.append(np.concatenate([image, mask], axis=1))
            tiles.append(np.concatenate(row, axis=1))
        image = np.concatenate(tiles, axis=0)
        safe_name = item["sample_id"].replace(":", "_")
        cv2.imwrite(str(destination / f"{safe_name}.png"), image)


def _write_rank_result(path, performance, quality):
    arrays = {
        "valid_counts": quality["valid_counts"],
        "view_pixels": quality["view_pixels"],
        "difference_samples": quality["difference_samples"],
    }
    for view in range(3):
        arrays[f"residual_samples_{view}"] = quality["residual_samples"][view]
        arrays[f"temporal_samples_{view}"] = quality["temporal_samples"][view]
    metadata = {
        "performance": performance,
        "finite_count": int(quality["finite_count"]),
        "pixel_count": int(quality["pixel_count"]),
        "mask_intersection": int(quality["mask_intersection"]),
        "mask_union": int(quality["mask_union"]),
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def _percentiles(values, label):
    if values.size == 0:
        raise ValueError(f"no valid values were collected for {label}")
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _aggregate(args, world_size):
    summary = {"schema": SCHEMA, "configurations": {}}
    for valid_iters in args.valid_iters:
        rank_payloads = []
        for rank in range(world_size):
            path = args.output_root / f"rank_{rank}_iters_{valid_iters}.npz"
            with np.load(path, allow_pickle=False) as payload:
                rank_payloads.append(
                    {key: payload[key] for key in payload.files}
                )
        metadata = [json.loads(str(payload["metadata_json"])) for payload in rank_payloads]
        total_samples = sum(item["performance"]["sample_count"] for item in metadata)
        slowest_wall = max(item["performance"]["wall_seconds"] for item in metadata)
        valid_counts = sum(payload["valid_counts"] for payload in rank_payloads)
        view_pixels = sum(payload["view_pixels"] for payload in rank_payloads)
        difference = np.concatenate(
            [payload["difference_samples"] for payload in rank_payloads]
        )
        residual = [
            np.concatenate(
                [payload[f"residual_samples_{view}"] for payload in rank_payloads]
            )
            for view in range(3)
        ]
        temporal = [
            np.concatenate(
                [payload[f"temporal_samples_{view}"] for payload in rank_payloads]
            )
            for view in range(3)
        ]
        intersection = sum(item["mask_intersection"] for item in metadata)
        union = sum(item["mask_union"] for item in metadata)
        quality = {
            "finite_disparity_ratio": (
                sum(item["finite_count"] for item in metadata)
                / sum(item["pixel_count"] for item in metadata)
            ),
            "valid_ratio_by_view": {
                view: float(valid_counts[index] / view_pixels[index])
                for index, view in enumerate(VIEWS)
            },
            "lr_residual_px_by_view": {
                view: _percentiles(residual[index], f"{view} LR residual")
                for index, view in enumerate(VIEWS)
            },
            "temporal_unwarped_abs_difference_px_by_view": {
                view: _percentiles(
                    temporal[index], f"{view} temporal difference"
                )
                for index, view in enumerate(VIEWS)
            },
        }
        if valid_iters != 32:
            quality["difference_from_32_px"] = _percentiles(
                difference, f"{valid_iters}-vs-32 disparity difference"
            )
            if union == 0:
                raise ValueError("valid-mask IoU is undefined because union is empty")
            quality["valid_mask_iou_with_32"] = intersection / union
        summary["configurations"][str(valid_iters)] = {
            "performance": {
                "world_size": world_size,
                "sample_count": total_samples,
                "wall_seconds_slowest_rank": slowest_wall,
                "samples_per_second": total_samples / slowest_wall,
                "total_seconds_per_stereo_pair": slowest_wall / (total_samples * 12),
                "total_seconds_per_training_sample": slowest_wall / total_samples,
                "decode_seconds_slowest_rank": max(
                    item["performance"]["decode_seconds"] for item in metadata
                ),
                "teacher_seconds_slowest_rank": max(
                    item["performance"]["teacher_seconds"] for item in metadata
                ),
                "teacher_seconds_per_stereo_pair_slowest_rank": max(
                    item["performance"]["teacher_seconds_per_stereo_pair"]
                    for item in metadata
                ),
                "peak_allocated_bytes_max_rank": max(
                    item["performance"]["peak_allocated_bytes"] for item in metadata
                ),
                "peak_reserved_bytes_max_rank": max(
                    item["performance"]["peak_reserved_bytes"] for item in metadata
                ),
            },
            "quality": quality,
        }
    candidate = summary["configurations"].get("16")
    if candidate:
        metrics = candidate["quality"]
        reference_metrics = summary["configurations"]["32"]["quality"]
        difference = metrics["difference_from_32_px"]
        valid_ratio_change = {
            view: (
                metrics["valid_ratio_by_view"][view]
                - reference_metrics["valid_ratio_by_view"][view]
            )
            for view in VIEWS
        }
        metrics["valid_ratio_change_from_32_by_view"] = valid_ratio_change
        accepted = (
            difference["p50"] < 0.25
            and difference["p95"] < 1.0
            and metrics["valid_mask_iou_with_32"] > 0.98
        )
        summary["project_gate_16_vs_32"] = {
            "automatic_numeric_pass": accepted,
            "requires_visual_review": True,
            "requires_valid_ratio_review": True,
            "final_acceptance": "pending_manual_reviews",
            "thresholds": {
                "difference_p50_lt_px": 0.25,
                "difference_p95_lt_px": 1.0,
                "valid_mask_iou_gt": 0.98,
            },
        }
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rectification-audit-sha256", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-count", type=int, default=512)
    parser.add_argument("--allow-pending-visual-review", action="store_true")
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--foundation-stereo-repo", type=Path, required=True)
    parser.add_argument("--foundation-stereo-checkpoint", type=Path, required=True)
    parser.add_argument("--foundation-stereo-checkpoint-sha256", required=True)
    parser.add_argument("--valid-iters", nargs="+", type=int, default=[32, 16, 12])
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--pair-microbatch", type=int, default=48)
    parser.add_argument("--disparity-min-px", type=float, default=0.5)
    parser.add_argument("--disparity-max-px", type=float, default=112.0)
    parser.add_argument("--lr-error-abs-threshold-px", type=float, default=1.0)
    parser.add_argument("--lr-error-relative-threshold", type=float, default=0.05)
    parser.add_argument("--metric-pixel-stride", type=int, default=16)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.valid_iters != [32, 16, 12]:
        raise ValueError("comparison order is frozen to 32, 16, 12")
    device, world_size, rank = _distributed_device()
    if world_size != args.expected_world_size:
        raise ValueError(
            f"teacher comparison requires world size {args.expected_world_size}, "
            f"got {world_size}"
        )
    if args.selection_count < world_size or args.selection_count % world_size:
        raise ValueError("teacher comparison samples must divide evenly by rank")
    if args.sample_batch_size < 1 or args.pair_microbatch < 1:
        raise ValueError("teacher comparison batch sizes must be positive")
    if args.metric_pixel_stride < 1:
        raise ValueError("metric pixel stride must be positive")
    args.output_root = args.output_root.expanduser().resolve()
    if rank == 0:
        if args.output_root.exists():
            raise FileExistsError(f"refusing to overwrite {args.output_root}")
        args.output_root.mkdir(parents=True)
    if dist.is_initialized():
        dist.barrier()
    dataset = LeRobotStereoDataset(
        args.episode_manifest,
        args.dataset_root,
        split="train",
        expected_rectification_audit_sha256=args.rectification_audit_sha256,
    )
    selection, indices = _read_selection(
        args.selection,
        dataset,
        args.selection_count,
        require_visual_review=not args.allow_pending_visual_review,
    )
    visual_sample_ids = set()
    if not args.allow_pending_visual_review:
        for tag in sorted(EXPECTED_VISUAL_TAGS):
            entry = next(
                item for item in selection["samples"] if tag in item["visual_tags"]
            )
            visual_sample_ids.add(entry["sample_id"])
    local_indices = indices[rank::world_size]
    teacher = FoundationStereoOnlineTeacher(
        args.foundation_stereo_repo,
        args.foundation_stereo_checkpoint,
        args.foundation_stereo_checkpoint_sha256,
        device=device,
        valid_iters=32,
        pair_microbatch=args.pair_microbatch,
    )
    baseline = None
    for valid_iters in args.valid_iters:
        output, performance = _run_configuration(
            args,
            teacher,
            dataset,
            local_indices,
            valid_iters,
            device,
            visual_sample_ids,
        )
        quality = _quality(output, baseline, args.metric_pixel_stride)
        if valid_iters == 32:
            baseline = output
        _save_visuals(
            args.output_root,
            valid_iters,
            output,
            visual_sample_ids,
        )
        _write_rank_result(
            args.output_root / f"rank_{rank}_iters_{valid_iters}.npz",
            performance,
            quality,
        )
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        summary = _aggregate(args, world_size)
        summary["selection"] = str(args.selection.resolve())
        summary["selection_sha256"] = sha256_file(args.selection.resolve())
        summary["selection_schema"] = selection["schema"]
        summary["selection_review_status"] = selection.get("review_status")
        summary["visual_review_required"] = not args.allow_pending_visual_review
        summary["episode_manifest"] = str(args.episode_manifest.resolve())
        summary["episode_manifest_sha256"] = sha256_file(
            args.episode_manifest.resolve()
        )
        summary["foundation_stereo_checkpoint"] = str(
            args.foundation_stereo_checkpoint.resolve()
        )
        summary["foundation_stereo_checkpoint_sha256"] = (
            args.foundation_stereo_checkpoint_sha256
        )
        summary["rectification_audit_sha256"] = (
            args.rectification_audit_sha256
        )
        summary["fixed_configuration"] = {
            "bidirectional": True,
            "lr_consistency": True,
            "valid_iters_order": args.valid_iters,
            "sample_batch_size_per_rank": args.sample_batch_size,
            "pair_microbatch": args.pair_microbatch,
            "metric_pixel_stride": args.metric_pixel_stride,
            "visual_sample_ids": sorted(visual_sample_ids),
        }
        summary_path = args.output_root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
