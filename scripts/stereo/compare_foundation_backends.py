#!/usr/bin/env python3
"""Compare frozen PyTorch and TensorRT FoundationStereo 32-iter backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from compare_online_foundation_teacher import (  # noqa: E402
    EXPECTED_VISUAL_TAGS,
    _batch,
    _distributed_device,
    _percentiles,
    _quality,
    _read_selection,
    _run_configuration,
    _save_visuals,
    _write_rank_result,
)
from stereo_tokenizer.lerobot_data import (  # noqa: E402
    VIEWS,
    LeRobotStereoDataset,
    sha256_file,
)
from stereo_tokenizer.online_gt import (  # noqa: E402
    FoundationStereoOnlineTeacher,
    validate_tensorrt_engine_assets,
)


SCHEMA = "foundation-stereo-backend-comparison-v1"
PYTORCH_BASELINE_TEACHER_SECONDS_PER_PAIR = 0.05338
TENSORRT_TARGET_TEACHER_SECONDS_PER_PAIR = 0.02669


def _equivalence_indices(selection, indices, count, world_size):
    if count < world_size or count % world_size:
        raise ValueError("equivalence sample count must divide evenly by rank")
    required_positions = []
    for tag in sorted(EXPECTED_VISUAL_TAGS):
        position = next(
            index
            for index, entry in enumerate(selection["samples"])
            if tag in entry["visual_tags"]
        )
        if position not in required_positions:
            required_positions.append(position)
    selected_positions = required_positions + [
        index
        for index in range(len(indices))
        if index not in required_positions
    ]
    selected_positions = selected_positions[:count]
    return [indices[index] for index in selected_positions]


def _flatten_pairs(video):
    left = video[:, :, 0].permute(0, 1, 3, 2, 4, 5).contiguous()
    right = video[:, :, 1].permute(0, 1, 3, 2, 4, 5).contiguous()
    channels, height, width = left.shape[-3:]
    left = (left.reshape(-1, channels, height, width) + 0.5) * 255.0
    right = (right.reshape(-1, channels, height, width) + 0.5) * 255.0
    return left, right


def _engine_batch_smoke(teacher, dataset, indices, device):
    if teacher.backend != "tensorrt":
        raise ValueError("engine smoke requires the TensorRT teacher")
    batch = _batch(dataset, indices[:4], device)
    left, right = _flatten_pairs(batch["video"])
    if left.shape[0] < 48:
        raise ValueError("engine smoke needs at least four training samples")
    results = {}
    outputs = {}
    for size in (1, 36, 48):
        output = teacher.runner.infer(left[:size], right[:size])
        torch.cuda.synchronize(device)
        outputs[size] = output
        results[str(size)] = {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "device": str(output.device),
            "finite_ratio": float(torch.isfinite(output).float().mean().item()),
        }
    for size in (1, 36):
        difference = torch.abs(outputs[size] - outputs[48][:size]).float()
        results[str(size)]["difference_from_batch_48_px"] = {
            "max": float(difference.max().item()),
            "p99": float(torch.quantile(difference.flatten(), 0.99).item()),
        }
    return results


def _profile_tensorrt_components(teacher, dataset, indices, device):
    """Profile one sample-batch on the current PyTorch CUDA stream."""
    batch = _batch(dataset, indices[:8], device)
    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in (
            "input_preparation",
            "lr_consistency",
            "bidirectional_teacher_total",
        )
    }
    engine_events = {
        "forward_engine": [],
        "reverse_engine_with_flips": [],
    }
    torch.cuda.synchronize(device)
    with torch.inference_mode():
        events["bidirectional_teacher_total"][0].record()
        events["input_preparation"][0].record()
        left, right = _flatten_pairs(batch["video"])
        events["input_preparation"][1].record()
        left_outputs = []
        right_outputs = []
        for start in range(0, left.shape[0], 48):
            stop = start + 48
            forward = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            reverse = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            forward[0].record()
            disparity_left = teacher.runner.infer(left[start:stop], right[start:stop])
            forward[1].record()
            reverse[0].record()
            disparity_right = teacher.runner.infer(
                torch.flip(right[start:stop], dims=[3]),
                torch.flip(left[start:stop], dims=[3]),
            )
            disparity_right = torch.flip(disparity_right, dims=[3])
            reverse[1].record()
            engine_events["forward_engine"].append(forward)
            engine_events["reverse_engine_with_flips"].append(reverse)
            left_outputs.append(disparity_left.float())
            right_outputs.append(disparity_right.float())
        disparity_left = torch.cat(left_outputs)
        disparity_right = torch.cat(right_outputs)
        events["lr_consistency"][0].record()
        teacher.lr_consistency(disparity_left, disparity_right)
        events["lr_consistency"][1].record()
        events["bidirectional_teacher_total"][1].record()
    torch.cuda.synchronize(device)
    stereo_pairs = int(left.shape[0])
    cuda_seconds = {
        name: started.elapsed_time(finished) / 1000.0
        for name, (started, finished) in events.items()
    }
    cuda_seconds.update(
        {
            name: sum(
                started.elapsed_time(finished) / 1000.0
                for started, finished in pairs
            )
            for name, pairs in engine_events.items()
        }
    )
    cuda_seconds["tensorrt_engine_total"] = (
        cuda_seconds["forward_engine"]
        + cuda_seconds["reverse_engine_with_flips"]
    )
    return {
        "sample_count": int(batch["video"].shape[0]),
        "stereo_pair_count": stereo_pairs,
        "cuda_seconds": cuda_seconds,
        "cuda_seconds_per_stereo_pair": {
            name: seconds / stereo_pairs for name, seconds in cuda_seconds.items()
        },
    }


def _load_rank_payload(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _aggregate_backend(args, world_size, backend, has_reference):
    payloads = [
        _load_rank_payload(args.output_root / f"rank_{rank}_{backend}.npz")
        for rank in range(world_size)
    ]
    metadata = [json.loads(str(payload["metadata_json"])) for payload in payloads]
    total_samples = sum(item["performance"]["sample_count"] for item in metadata)
    slowest_wall = max(item["performance"]["wall_seconds"] for item in metadata)
    valid_counts = sum(payload["valid_counts"] for payload in payloads)
    view_pixels = sum(payload["view_pixels"] for payload in payloads)
    residual = [
        np.concatenate(
            [payload[f"residual_samples_{view}"] for payload in payloads]
        )
        for view in range(3)
    ]
    temporal = [
        np.concatenate(
            [payload[f"temporal_samples_{view}"] for payload in payloads]
        )
        for view in range(3)
    ]
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
            view: _percentiles(temporal[index], f"{view} temporal difference")
            for index, view in enumerate(VIEWS)
        },
    }
    if has_reference:
        difference = np.concatenate(
            [payload["difference_samples"] for payload in payloads]
        )
        intersection = sum(item["mask_intersection"] for item in metadata)
        union = sum(item["mask_union"] for item in metadata)
        if union == 0:
            raise ValueError("backend valid-mask IoU is undefined")
        quality["difference_from_pytorch_px"] = _percentiles(
            difference, "TensorRT-vs-PyTorch disparity difference"
        )
        quality["valid_mask_iou_with_pytorch"] = intersection / union
    performance = {
        "world_size": world_size,
        "sample_count": total_samples,
        "wall_seconds_slowest_rank": slowest_wall,
        "samples_per_second": total_samples / slowest_wall,
        "total_seconds_per_stereo_pair": slowest_wall / (total_samples * 12),
        "teacher_seconds_slowest_rank": max(
            item["performance"]["teacher_seconds"] for item in metadata
        ),
        "teacher_seconds_per_stereo_pair_slowest_rank": max(
            item["performance"]["teacher_seconds_per_stereo_pair"]
            for item in metadata
        ),
        "decode_seconds_slowest_rank": max(
            item["performance"]["decode_seconds"] for item in metadata
        ),
        "peak_allocated_bytes_max_rank": max(
            item["performance"]["peak_allocated_bytes"] for item in metadata
        ),
        "peak_reserved_bytes_max_rank": max(
            item["performance"]["peak_reserved_bytes"] for item in metadata
        ),
    }
    return {"performance": performance, "quality": quality}


def _gate(summary):
    candidate = summary["backends"]["tensorrt"]
    quality = candidate["quality"]
    performance = candidate["performance"]
    checks = {
        "finite_disparity_ratio_eq_1": quality["finite_disparity_ratio"] == 1.0,
        "teacher_seconds_per_pair_le_0_02669": (
            performance["teacher_seconds_per_stereo_pair_slowest_rank"]
            <= TENSORRT_TARGET_TEACHER_SECONDS_PER_PAIR
        ),
        "engine_smoke_batches_1_36_48_finite": all(
            smoke[str(size)]["finite_ratio"] == 1.0
            for smoke in summary["engine_smoke_by_rank"]
            for size in (1, 36, 48)
        ),
        "engine_smoke_shapes_1_36_48": all(
            smoke[str(size)]["shape"] == [size, 1, 256, 256]
            for smoke in summary["engine_smoke_by_rank"]
            for size in (1, 36, 48)
        ),
        "engine_smoke_dtype_matches_manifest": all(
            smoke[str(size)]["dtype"]
            == "torch." + summary["engine_manifest"]["bindings"]["disparity"][
                "dtype"
            ]
            for smoke in summary["engine_smoke_by_rank"]
            for size in (1, 36, 48)
        ),
        "engine_smoke_device_is_cuda": all(
            smoke[str(size)]["device"].startswith("cuda:")
            for smoke in summary["engine_smoke_by_rank"]
            for size in (1, 36, 48)
        ),
        "engine_smoke_batch_consistency_p99_le_0_10_px": all(
            smoke[str(size)]["difference_from_batch_48_px"]["p99"] <= 0.10
            for smoke in summary["engine_smoke_by_rank"]
            for size in (1, 36)
        ),
        "engine_smoke_batch_consistency_max_le_0_50_px": all(
            smoke[str(size)]["difference_from_batch_48_px"]["max"] <= 0.50
            for smoke in summary["engine_smoke_by_rank"]
            for size in (1, 36)
        ),
    }
    if summary["mode"] == "equivalence":
        difference = quality["difference_from_pytorch_px"]
        reference_ratios = summary["backends"]["pytorch"]["quality"][
            "valid_ratio_by_view"
        ]
        changes = {
            view: quality["valid_ratio_by_view"][view] - reference_ratios[view]
            for view in VIEWS
        }
        quality["valid_ratio_change_from_pytorch_by_view"] = changes
        checks.update(
            {
                "difference_p50_le_0_02_px": difference["p50"] <= 0.02,
                "difference_p95_le_0_10_px": difference["p95"] <= 0.10,
                "difference_p99_le_0_50_px": difference["p99"] <= 0.50,
                "valid_mask_iou_ge_0_99": (
                    quality["valid_mask_iou_with_pytorch"] >= 0.99
                ),
                "valid_ratio_change_le_0_002_by_view": all(
                    abs(value) <= 0.002 for value in changes.values()
                ),
            }
        )
    return {"checks": checks, "pass": all(checks.values())}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("equivalence", "tensorrt_benchmark"), required=True
    )
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rectification-audit-sha256", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-count", type=int, default=408)
    parser.add_argument("--equivalence-sample-count", type=int, default=64)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--foundation-stereo-repo", type=Path)
    parser.add_argument("--foundation-stereo-checkpoint", type=Path)
    parser.add_argument("--foundation-stereo-checkpoint-sha256", required=True)
    parser.add_argument("--foundation-stereo-engine", type=Path, required=True)
    parser.add_argument("--foundation-stereo-engine-sha256", required=True)
    parser.add_argument(
        "--foundation-stereo-engine-manifest", type=Path, required=True
    )
    parser.add_argument("--foundation-stereo-engine-manifest-sha256", required=True)
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
    device, world_size, rank = _distributed_device()
    if world_size != args.expected_world_size:
        raise ValueError(
            f"backend comparison requires world size {args.expected_world_size}"
        )
    if args.selection_count != 408:
        raise ValueError("backend pilot is frozen to the approved 408 selection")
    if args.mode == "equivalence" and not 32 <= args.equivalence_sample_count <= 64:
        raise ValueError("equivalence comparison is limited to 32-64 samples")
    if args.mode == "equivalence" and (
        args.foundation_stereo_repo is None
        or args.foundation_stereo_checkpoint is None
    ):
        raise ValueError("equivalence mode requires the frozen PyTorch assets")
    if args.sample_batch_size != 8 or args.pair_microbatch != 48:
        raise ValueError("backend pilot freezes sample batch 8 and pair microbatch 48")
    manifest_payload = validate_tensorrt_engine_assets(
        args.foundation_stereo_engine,
        args.foundation_stereo_engine_sha256,
        args.foundation_stereo_engine_manifest,
        args.foundation_stereo_engine_manifest_sha256,
        args.foundation_stereo_checkpoint_sha256,
    )
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
        args.selection, dataset, args.selection_count, require_visual_review=True
    )
    if args.mode == "equivalence":
        indices = _equivalence_indices(
            selection, indices, args.equivalence_sample_count, world_size
        )
    local_indices = indices[rank::world_size]
    visual_sample_ids = {
        next(
            entry["sample_id"]
            for entry in selection["samples"]
            if tag in entry["visual_tags"]
        )
        for tag in sorted(EXPECTED_VISUAL_TAGS)
    }
    tensorrt_teacher = FoundationStereoOnlineTeacher(
        None,
        None,
        args.foundation_stereo_checkpoint_sha256,
        device=device,
        valid_iters=32,
        pair_microbatch=48,
        backend="tensorrt",
        engine=args.foundation_stereo_engine,
        engine_sha256=args.foundation_stereo_engine_sha256,
        engine_manifest=args.foundation_stereo_engine_manifest,
        engine_manifest_sha256=args.foundation_stereo_engine_manifest_sha256,
    )
    smoke = _engine_batch_smoke(tensorrt_teacher, dataset, local_indices, device)
    component_timing = _profile_tensorrt_components(
        tensorrt_teacher, dataset, local_indices, device
    )
    reference_output = None
    backends = []
    if args.mode == "equivalence":
        pytorch_teacher = FoundationStereoOnlineTeacher(
            args.foundation_stereo_repo,
            args.foundation_stereo_checkpoint,
            args.foundation_stereo_checkpoint_sha256,
            device=device,
            valid_iters=32,
            pair_microbatch=48,
        )
        reference_output, performance = _run_configuration(
            args,
            pytorch_teacher,
            dataset,
            local_indices,
            32,
            device,
            visual_sample_ids,
        )
        _write_rank_result(
            args.output_root / f"rank_{rank}_pytorch.npz",
            performance,
            _quality(reference_output),
        )
        _save_visuals(
            args.output_root, "pytorch32", reference_output, visual_sample_ids
        )
        backends.append("pytorch")
    tensorrt_output, performance = _run_configuration(
        args,
        tensorrt_teacher,
        dataset,
        local_indices,
        32,
        device,
        visual_sample_ids,
    )
    _write_rank_result(
        args.output_root / f"rank_{rank}_tensorrt.npz",
        performance,
        _quality(tensorrt_output, reference_output, args.metric_pixel_stride),
    )
    _save_visuals(
        args.output_root, "tensorrt32", tensorrt_output, visual_sample_ids
    )
    backends.append("tensorrt")
    (args.output_root / f"rank_{rank}_engine_smoke.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / f"rank_{rank}_component_timing.json").write_text(
        json.dumps(component_timing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        summary = {
            "schema": SCHEMA,
            "mode": args.mode,
            "selection": str(args.selection.resolve()),
            "selection_sha256": sha256_file(args.selection.resolve()),
            "selection_count": len(indices),
            "episode_manifest": str(args.episode_manifest.resolve()),
            "episode_manifest_sha256": sha256_file(
                args.episode_manifest.resolve()
            ),
            "checkpoint_sha256": args.foundation_stereo_checkpoint_sha256,
            "engine_sha256": args.foundation_stereo_engine_sha256,
            "engine_manifest_sha256": (
                args.foundation_stereo_engine_manifest_sha256
            ),
            "engine_manifest": manifest_payload,
            "fixed_configuration": {
                "valid_iters": 32,
                "sample_batch_size_per_rank": 8,
                "pair_microbatch": 48,
                "bidirectional": True,
                "lr_consistency": True,
                "cache_enabled": False,
            },
            "published_pytorch_baseline": {
                "sample_count": 408,
                "world_size": 8,
                "teacher_seconds_per_stereo_pair": (
                    PYTORCH_BASELINE_TEACHER_SECONDS_PER_PAIR
                ),
            },
            "engine_smoke_by_rank": [
                json.loads(
                    (args.output_root / f"rank_{item}_engine_smoke.json").read_text(
                        encoding="utf-8"
                    )
                )
                for item in range(world_size)
            ],
            "component_timing_by_rank": [
                json.loads(
                    (
                        args.output_root / f"rank_{item}_component_timing.json"
                    ).read_text(encoding="utf-8")
                )
                for item in range(world_size)
            ],
            "backends": {
                backend: _aggregate_backend(
                    args,
                    world_size,
                    backend,
                    has_reference=backend == "tensorrt"
                    and args.mode == "equivalence",
                )
                for backend in backends
            },
        }
        summary["project_gate"] = _gate(summary)
        summary["backends"]["tensorrt"]["performance"][
            "speedup_vs_published_pytorch_teacher"
        ] = (
            PYTORCH_BASELINE_TEACHER_SECONDS_PER_PAIR
            / summary["backends"]["tensorrt"]["performance"][
                "teacher_seconds_per_stereo_pair_slowest_rank"
            ]
        )
        (args.output_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
