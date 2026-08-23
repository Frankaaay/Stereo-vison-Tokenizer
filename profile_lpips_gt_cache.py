from __future__ import annotations

import argparse
import json
import statistics

import torch

from stereo_tokenizer.modules.lpips import LPIPS


def _run_baseline(model, prediction, target):
    value = prediction.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(value * 2.0, target * 2.0).mean()
    loss.backward()
    return loss.detach().clone(), value.grad.detach().clone()


def _target_features(model, target):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return tuple(
            feature.detach()
            for feature in model.normalized_features(target * 2.0)
        )


def _run_cached(model, prediction, target_features):
    value = prediction.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction_features = model.normalized_features(value * 2.0)
        loss = model.distance_from_normalized_features(
            prediction_features, target_features
        ).mean()
    loss.backward()
    return loss.detach().clone(), value.grad.detach().clone()


def _benchmark(function, warmup: int, iterations: int):
    timings = []
    for index in range(warmup + iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        if index >= warmup:
            timings.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.fmean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    model = LPIPS().eval().requires_grad_(False).cuda()
    prediction = torch.randn(
        args.frames, 3, 256, 256, device="cuda", dtype=torch.bfloat16
    )
    target = torch.randn(
        args.frames, 3, 256, 256, device="cuda", dtype=torch.float32
    )
    target_features = _target_features(model, target)

    baseline_loss, baseline_grad = _run_baseline(model, prediction, target)
    cached_loss, cached_grad = _run_cached(
        model, prediction, target_features
    )
    cache_bytes = sum(
        feature.numel() * feature.element_size() for feature in target_features
    )

    result = {
        "frames": args.frames,
        "loss": {
            "baseline": float(baseline_loss.float().cpu()),
            "cached": float(cached_loss.float().cpu()),
            "max_abs_difference": float(
                (baseline_loss.float() - cached_loss.float()).abs().cpu()
            ),
        },
        "prediction_gradient": {
            "max_abs_difference": float(
                (baseline_grad.float() - cached_grad.float()).abs().max().cpu()
            ),
            "reference_max_abs": float(
                baseline_grad.float().abs().max().cpu()
            ),
        },
        "target_feature_cache": {
            "bytes": cache_bytes,
            "gib": cache_bytes / (1024**3),
            "dtypes": sorted(
                {str(feature.dtype) for feature in target_features}
            ),
            "devices": sorted(
                {str(feature.device) for feature in target_features}
            ),
        },
        "baseline": _benchmark(
            lambda: _run_baseline(model, prediction, target),
            args.warmup,
            args.iterations,
        ),
        "cached": _benchmark(
            lambda: _run_cached(model, prediction, target_features),
            args.warmup,
            args.iterations,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
