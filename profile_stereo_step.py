from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from torch.profiler import ProfilerActivity

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.data import StereoDataModule
from stereo_tokenizer.modules.attention import PEG
from stereo_tokenizer.profiling import profile_region, set_profiling_enabled
from train_stereo_vae import build_parser, validate_runtime_args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _event_time_us(event, name: str) -> float:
    value = getattr(event, name, None)
    if value is not None:
        return float(value)
    legacy = {
        "device_time_total": "cuda_time_total",
        "self_device_time_total": "self_cuda_time_total",
    }.get(name)
    return float(getattr(event, legacy, 0.0)) if legacy else 0.0


class ProfiledCSVLogger(CSVLogger):
    def log_metrics(self, metrics, step=None):
        with profile_region("stereo/logging/csv_logger"):
            return super().log_metrics(metrics, step=step)

    def save(self):
        with profile_region("stereo/logging/csv_save"):
            return super().save()


class StepTraceCallback(Callback):
    def __init__(self, profiler, output_dir: Path):
        self.profiler = profiler
        self.output_dir = output_dir
        self.last_batch_end = None
        self.step_timings = []
        self.pending_loss = None

    def on_train_start(self, trainer, pl_module) -> None:
        self.last_batch_end = time.perf_counter()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        self.pending_loss = None
        if isinstance(outputs, dict) and torch.is_tensor(outputs.get("loss")):
            self.pending_loss = float(outputs["loss"].detach().float().cpu())

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        now = time.perf_counter()
        interval_s = now - self.last_batch_end
        self.last_batch_end = now
        self.step_timings.append(
            {
                "step": int(pl_module.generator_updates),
                "interval_s": interval_s,
                "loss": self.pending_loss,
                "lr": float(trainer.optimizers[0].param_groups[0]["lr"]),
            }
        )
        self.profiler.step()


class TraceWriter:
    def __init__(self, output_dir: Path, configured_active_steps: int):
        self.output_dir = output_dir
        self.configured_active_steps = configured_active_steps
        self.trace_index = 0
        self.region_files = []
        self.operator_files = []
        self.trace_files = []

    @staticmethod
    def _row(event, denominator: int) -> dict:
        return {
            "name": event.key,
            "calls": int(event.count),
            "cpu_total_ms": float(event.cpu_time_total) / 1000.0,
            "cpu_self_ms": float(event.self_cpu_time_total) / 1000.0,
            "device_total_ms": _event_time_us(event, "device_time_total") / 1000.0,
            "device_self_ms": _event_time_us(
                event, "self_device_time_total"
            )
            / 1000.0,
            "cpu_total_ms_per_step": (
                float(event.cpu_time_total) / 1000.0 / denominator
            ),
            "cpu_self_ms_per_step": (
                float(event.self_cpu_time_total) / 1000.0 / denominator
            ),
            "device_total_ms_per_step": (
                _event_time_us(event, "device_time_total") / 1000.0 / denominator
            ),
            "device_self_ms_per_step": (
                _event_time_us(event, "self_device_time_total")
                / 1000.0
                / denominator
            ),
        }

    def __call__(self, profiler) -> None:
        averages = list(profiler.key_averages())
        training_step = next(
            (
                event
                for event in averages
                if event.key == "stereo/step/training_step"
            ),
            None,
        )
        denominator = (
            int(training_step.count)
            if training_step is not None and training_step.count > 0
            else self.configured_active_steps
        )
        regions = [
            self._row(event, denominator)
            for event in averages
            if event.key.startswith("stereo/") or "DataLoader" in event.key
        ]
        regions.sort(
            key=lambda row: (
                row["device_total_ms_per_step"], row["cpu_total_ms_per_step"]
            ),
            reverse=True,
        )
        operators = [self._row(event, denominator) for event in averages]
        operators.sort(
            key=lambda row: row["device_self_ms_per_step"], reverse=True
        )

        suffix = f"trace-{self.trace_index:02d}"
        region_path = self.output_dir / f"{suffix}-regions.json"
        operator_path = self.output_dir / f"{suffix}-top-operators.json"
        trace_path = self.output_dir / f"{suffix}.json.gz"
        _write_json(
            region_path,
            {
                "observed_active_steps": denominator,
                "regions": regions,
            },
        )
        _write_json(
            operator_path,
            {
                "observed_active_steps": denominator,
                "operators": operators[:100],
            },
        )
        profiler.export_chrome_trace(str(trace_path))
        self.region_files.append(str(region_path))
        self.operator_files.append(str(operator_path))
        self.trace_files.append(str(trace_path))
        self.trace_index += 1


def _window_stats(rows: list[dict], start: int, end: int) -> dict:
    values = [
        row["interval_s"] for row in rows if start <= row["step"] <= end
    ]
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "p10_s": ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)],
        "p90_s": ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)],
    }


def build_profile_parser():
    parser = build_parser()
    parser.add_argument("--profile_updates", type=int, default=40)
    parser.add_argument("--profile_wait", type=int, default=15)
    parser.add_argument("--profile_warmup", type=int, default=5)
    parser.add_argument("--profile_active", type=int, default=10)
    parser.add_argument(
        "--profile_peg_backend",
        choices=(
            "conv3d_contiguous",
            "conv3d_channels_last_3d",
            "conv2d_t1_slice",
        ),
        default="conv3d_contiguous",
    )
    parser.add_argument(
        "--profile_preload_data", type=int, choices=(0, 1), default=0
    )
    parser.add_argument(
        "--profile_pin_memory", type=int, choices=(0, 1), default=0
    )
    parser.add_argument(
        "--profile_lpips_gt_cache", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--expected_git_sha", type=str, required=True)
    parser.add_argument("--expected_manifest_sha256", type=str, required=True)
    return parser


def validate_profile_args(args) -> None:
    validate_runtime_args(args)
    if args.devices != 1 or args.num_nodes != 1:
        raise ValueError("step profiling requires exactly one node and one GPU")
    if args.batch_size != 8 or args.num_workers != 0:
        raise ValueError("step profiling freezes batch_size=8 and num_workers=0")
    if args.max_steps != 5000:
        raise ValueError("model max_steps must remain 5000 for scheduler equivalence")
    if args.gan_enabled:
        raise ValueError("the accepted 8-sample contract has GAN disabled")
    scheduled = args.profile_wait + args.profile_warmup + args.profile_active
    if min(
        args.profile_updates,
        args.profile_wait,
        args.profile_warmup,
        args.profile_active,
    ) < 1:
        raise ValueError("all profiling step counts must be positive")
    if scheduled >= args.profile_updates:
        raise ValueError("profile schedule must leave post-profile validation steps")


def main() -> None:
    args = build_profile_parser().parse_args()
    validate_profile_args(args)
    output_dir = Path(args.default_root_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(args.stereo_train_manifest).resolve()
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != args.expected_manifest_sha256:
        raise RuntimeError(
            "selected manifest SHA changed: "
            f"expected {args.expected_manifest_sha256}, got {manifest_sha256}"
        )

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    set_profiling_enabled(True)
    data = StereoDataModule(args, shuffle=False)
    preloaded_sample_count = 0
    if args.profile_preload_data:
        preloaded_sample_count = data.profile_preload_train_dataset()
    dataset = data._dataset(True)
    if len(dataset) != 8:
        raise RuntimeError(f"expected exactly eight samples, got {len(dataset)}")
    model = StereoVAE(args)
    model.set_profile_lpips_gt_cache(bool(args.profile_lpips_gt_cache))
    peg_count = 0
    for module in model.modules():
        if isinstance(module, PEG):
            module.set_profile_backend(args.profile_peg_backend)
            peg_count += 1
    if peg_count != 14:
        raise RuntimeError(f"expected 14 PEG modules, got {peg_count}")

    trace_writer = TraceWriter(output_dir, args.profile_active)
    schedule = torch.profiler.schedule(
        wait=args.profile_wait,
        warmup=args.profile_warmup,
        active=args.profile_active,
        repeat=1,
    )
    profiler = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=trace_writer,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
    )
    callback = StepTraceCallback(profiler, output_dir)
    logger = ProfiledCSVLogger(
        save_dir=str(output_dir), name="csv_metrics", version=0
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy="auto",
        precision="bf16-mixed",
        max_steps=args.profile_updates,
        max_epochs=-1,
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=[callback],
        log_every_n_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    resolved = vars(args).copy()
    resolved.update(
        {
            "git_sha": args.expected_git_sha,
            "manifest_sha256": manifest_sha256,
            "shuffle": False,
            "trainer_max_steps": args.profile_updates,
            "peg_count": peg_count,
            "preloaded_sample_count": preloaded_sample_count,
        }
    )
    _write_json(output_dir / "resolved_config.json", resolved)
    started_at = time.perf_counter()
    with profiler:
        trainer.fit(model, datamodule=data)
    elapsed_s = time.perf_counter() - started_at

    _write_json(output_dir / "step_timings.json", callback.step_timings)
    schedule_end = args.profile_wait + args.profile_warmup + args.profile_active
    lpips_cache = model._profile_lpips_gt_features
    lpips_cache_bytes = (
        sum(feature.numel() * feature.element_size() for feature in lpips_cache)
        if lpips_cache is not None
        else 0
    )
    result = {
        "event": "STEREO_STEP_PROFILE_COMPLETE",
        "git_sha": args.expected_git_sha,
        "manifest_sha256": manifest_sha256,
        "generator_updates": int(model.generator_updates),
        "trainer_global_step": int(trainer.global_step),
        "elapsed_s": elapsed_s,
        "profile_schedule": {
            "wait": args.profile_wait,
            "warmup": args.profile_warmup,
            "active": args.profile_active,
            "total_updates": args.profile_updates,
        },
        "wall_windows": {
            "pre_profile_steady": _window_stats(
                callback.step_timings, 6, args.profile_wait
            ),
            "active_trace": _window_stats(
                callback.step_timings,
                args.profile_wait + args.profile_warmup + 1,
                schedule_end,
            ),
            "post_profile": _window_stats(
                callback.step_timings, schedule_end + 1, args.profile_updates
            ),
        },
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "lpips_gt_cache_bytes": lpips_cache_bytes,
        "trace_files": trace_writer.trace_files,
        "region_files": trace_writer.region_files,
        "operator_files": trace_writer.operator_files,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "lightning": pl.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    _write_json(output_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
