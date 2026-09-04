"""Stage A CUDA latency and throughput benchmark command."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import default_collate

from . import runtime
from .common import (
    _checkpoint_provenance,
    _dataset_provenance,
    _environment_provenance,
    _jsonable,
    _source_provenance,
)
from .data import CanonicalStageADataset
from .quality import _hydrate_checkpoint_semantics, _mode_batch, _run_parser


def _percentile_summary(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("benchmark timings must be finite and non-empty")
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p90_ms": float(np.quantile(values, 0.90)),
    }


def _cuda_benchmark(function, *, warmup: int, iterations: int, repeats: int):
    all_times = []
    allocated = []
    reserved = []
    for _ in range(repeats):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            function()
            end.record()
        torch.cuda.synchronize()
        all_times.extend(start.elapsed_time(end) for start, end in zip(starts, ends))
        allocated.append(torch.cuda.max_memory_allocated())
        reserved.append(torch.cuda.max_memory_reserved())
    summary = _percentile_summary(all_times)
    summary["peak_allocated_bytes"] = max(allocated)
    summary["peak_reserved_bytes"] = max(reserved)
    return summary


def _benchmark_command(argv: list[str]) -> None:
    parser = _run_parser()
    parser.prog = "tokenizer_stage_a benchmark"
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--allow-nonformal-benchmark", action="store_true")
    args = parser.parse_args(argv)
    _hydrate_checkpoint_semantics(args)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage A benchmark requires one allocated CUDA GPU")
    if args.batch_size != 1 or not args.bf16:
        raise ValueError("formal benchmark requires --batch_size 1 --bf16")
    configured = (
        args.benchmark_warmup,
        args.benchmark_iterations,
        args.benchmark_repeats,
    )
    if not args.allow_nonformal_benchmark and configured != (20, 100, 3):
        raise ValueError("formal benchmark is frozen to warmup=20, iterations=100, repeats=3")
    if min(configured) < 1:
        raise ValueError("benchmark counts must be positive")
    environment = _environment_provenance()
    if args.eval_temporal_mode != "both":
        raise ValueError("benchmark requires both temporal modes")
    dataset = CanonicalStageADataset(
        args.stage_a_selection,
        loader_root=args.canonical_loader_root,
        eye_mode=args.eval_eye_mode,
        camera_key=args.stage_a_camera_key,
    )
    batch = default_collate([dataset[0]])
    device = torch.device("cuda")
    video = batch["video"].to(device)
    model = runtime.load_model(args, device)
    model.requires_grad_(False)
    modes = {}
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for temporal_mode, source_index in (("single_frame", 0), ("four_frame", None)):
            mode_batch = _mode_batch(
                {**batch, "video": video}, temporal_mode, source_index
            )
            mode_video = mode_batch["video"]
            encoded = model.encode(
                mode_video,
                eye_mode=args.eval_eye_mode,
                temporal_mode=temporal_mode,
                sample_posterior=False,
            )
            mode = {}
            mode["encode_including_posterior_mean"] = _cuda_benchmark(
                lambda: model.encode(
                    mode_video,
                    eye_mode=args.eval_eye_mode,
                    temporal_mode=temporal_mode,
                    sample_posterior=False,
                ),
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                repeats=args.benchmark_repeats,
            )
            mode["cached_posterior_mean"] = _cuda_benchmark(
                encoded.posterior.mode,
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                repeats=args.benchmark_repeats,
            )
            mode["decode"] = _cuda_benchmark(
                lambda: model.decode(
                    encoded.latent, temporal_mode=temporal_mode
                ),
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                repeats=args.benchmark_repeats,
            )
            mode["end_to_end"] = _cuda_benchmark(
                lambda: model(
                    mode_video,
                    eye_mode=args.eval_eye_mode,
                    temporal_mode=temporal_mode,
                    sample_posterior=False,
                ),
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                repeats=args.benchmark_repeats,
            )
            end_to_end_p50 = mode["end_to_end"]["p50_ms"]
            mode["throughput"] = {
                "samples_per_second": 1000.0 / end_to_end_p50,
                "frames_per_second": (
                    1000.0 * mode_video.shape[-3] / end_to_end_p50
                ),
            }
            mode["input_shape"] = list(mode_video.shape)
            mode["input_dtype"] = str(mode_video.dtype)
            mode["autocast_dtype"] = "torch.bfloat16"
            modes[temporal_mode] = mode
    result = {
        "schema": "stereo-tokenizer-stage-a1-benchmark-v1",
        "status": "smoke" if args.allow_nonformal_benchmark else "formal",
        "checkpoint": _checkpoint_provenance(
            args.stereo_vae_ckpt, args.checkpoint_sha256
        ),
        "dataset": _dataset_provenance(dataset),
        "precision": "bf16",
        "batch_size": 1,
        "warmup": args.benchmark_warmup,
        "iterations": args.benchmark_iterations,
        "repeats": args.benchmark_repeats,
        "posterior": "mean",
        "timing_scope": "model_only_excludes_data_decode_and_teacher",
        "modes": modes,
        "provenance": {
            **_source_provenance(),
            "environment": environment,
            "resolved_args": _jsonable(vars(args)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
