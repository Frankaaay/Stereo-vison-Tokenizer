import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.profiler import ProfilerActivity

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.data import StereoDataModule
from stereo_tokenizer.mode_sampling import MODE_IDS, mode_occurrences_before
from stereo_tokenizer.modules.callbacks import ImageLogger, VideoLogger
from stereo_tokenizer.online_gt import (
    OnlineDepthAnything3GTCallback,
    OnlineFoundationGTCallback,
    validate_tensorrt_engine_assets,
)
from stereo_tokenizer.profiling import set_profiling_enabled


def _timing_summary(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


class StepTimingCallback(Callback):
    def __init__(self, output_path, warmup_updates):
        self.output_path = Path(output_path)
        self.warmup_updates = warmup_updates
        self.last_batch_end = None
        self.last_generator_update = None
        self.timings = []

    def on_train_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.last_batch_end = time.perf_counter()
        self.last_generator_update = int(pl_module.generator_updates)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        generator_update = int(pl_module.generator_updates)
        if generator_update == self.last_generator_update:
            return
        if generator_update != self.last_generator_update + 1:
            raise RuntimeError("step timing observed a non-consecutive generator update")
        torch.cuda.synchronize()
        now = time.perf_counter()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        self.timings.append(
            {
                "step": generator_update,
                "mode_id": pl_module.last_mode_id,
                "temporal_mode": pl_module.last_temporal_mode,
                "interval_s": now - self.last_batch_end,
                "peak_memory_allocated_bytes": peak_allocated,
                "peak_memory_reserved_bytes": peak_reserved,
            }
        )
        mode_prefix = f"train/{pl_module.last_mode_id}"
        pl_module.log_dict(
            {
                f"{mode_prefix}/step_time_s": now - self.last_batch_end,
                f"{mode_prefix}/peak_memory_allocated_bytes": float(
                    peak_allocated
                ),
                f"{mode_prefix}/peak_memory_reserved_bytes": float(
                    peak_reserved
                ),
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )
        torch.cuda.reset_peak_memory_stats()
        self.last_batch_end = now
        self.last_generator_update = generator_update

    def on_train_end(self, trainer, pl_module):
        local_memory = torch.tensor(
            [
                [
                    max(
                        (
                            row["peak_memory_allocated_bytes"]
                            for row in self.timings
                            if row["mode_id"] == mode_id
                        ),
                        default=0,
                    ),
                    max(
                        (
                            row["peak_memory_reserved_bytes"]
                            for row in self.timings
                            if row["mode_id"] == mode_id
                        ),
                        default=0,
                    ),
                ]
                for mode_id in MODE_IDS
            ],
            device=pl_module.device,
            dtype=torch.long,
        )
        rank_memory = [local_memory]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_memory = [
                torch.zeros_like(local_memory) for _ in range(trainer.world_size)
            ]
            torch.distributed.all_gather(rank_memory, local_memory)
        if not trainer.is_global_zero:
            return
        stable = self.timings[self.warmup_updates :]
        values = [row["interval_s"] for row in stable]
        stable_by_temporal_mode = {}
        for temporal_mode in ("single_frame", "four_frame"):
            mode_values = [
                row["interval_s"]
                for row in stable
                if row["temporal_mode"] == temporal_mode
            ]
            if mode_values:
                stable_by_temporal_mode[temporal_mode] = _timing_summary(
                    mode_values
                )
        stable_by_mode = {}
        for mode_id in MODE_IDS:
            mode_values = [
                row["interval_s"]
                for row in stable
                if row["mode_id"] == mode_id
            ]
            if mode_values:
                stable_by_mode[mode_id] = _timing_summary(mode_values)
        payload = {
            "world_size": int(trainer.world_size),
            "per_device_batch_size": int(pl_module.args.batch_size),
            "warmup_updates": self.warmup_updates,
            "peak_memory_bytes_by_rank_and_mode": [
                {
                    "rank": rank,
                    "modes": {
                        mode_id: {
                            "allocated": int(memory[index, 0].item()),
                            "reserved": int(memory[index, 1].item()),
                        }
                        for index, mode_id in enumerate(MODE_IDS)
                    },
                }
                for rank, memory in enumerate(rank_memory)
            ],
            "timings": self.timings,
            "stable": _timing_summary(values),
            "stable_by_temporal_mode": stable_by_temporal_mode,
            "stable_by_mode": stable_by_mode,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _profile_event_time_us(event, name):
    value = getattr(event, name, None)
    if value is not None:
        return float(value)
    legacy = {
        "device_time_total": "cuda_time_total",
        "self_device_time_total": "self_cuda_time_total",
    }.get(name)
    return float(getattr(event, legacy, 0.0)) if legacy else 0.0


class TrainingTraceWriter:
    def __init__(self, output_dir, configured_active_steps):
        self.output_dir = Path(output_dir)
        self.configured_active_steps = configured_active_steps
        self.trace_index = 0

    @staticmethod
    def _row(event, denominator):
        return {
            "name": event.key,
            "calls": int(event.count),
            "cpu_total_ms_per_step": (
                float(event.cpu_time_total) / 1000.0 / denominator
            ),
            "cpu_self_ms_per_step": (
                float(event.self_cpu_time_total) / 1000.0 / denominator
            ),
            "device_total_ms_per_step": (
                _profile_event_time_us(event, "device_time_total")
                / 1000.0
                / denominator
            ),
            "device_self_ms_per_step": (
                _profile_event_time_us(event, "self_device_time_total")
                / 1000.0
                / denominator
            ),
        }

    def __call__(self, profiler):
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
        rows = [self._row(event, denominator) for event in averages]
        regions = [
            row
            for row in rows
            if row["name"].startswith("stereo/")
            or "DataLoader" in row["name"]
        ]
        regions.sort(
            key=lambda row: (
                row["device_total_ms_per_step"],
                row["cpu_total_ms_per_step"],
            ),
            reverse=True,
        )
        rows.sort(key=lambda row: row["device_self_ms_per_step"], reverse=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"rank0-trace-{self.trace_index:02d}"
        for filename, payload in (
            (
                f"{suffix}-regions.json",
                {"observed_active_steps": denominator, "regions": regions},
            ),
            (
                f"{suffix}-top-operators.json",
                {"observed_active_steps": denominator, "operators": rows[:100]},
            ),
        ):
            (self.output_dir / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        profiler.export_chrome_trace(
            str(self.output_dir / f"{suffix}.json.gz")
        )
        self.trace_index += 1


class TrainingProfilerStepCallback(Callback):
    def __init__(self, profiler):
        self.profiler = profiler
        self.last_generator_update = None

    def on_train_start(self, trainer, pl_module):
        self.last_generator_update = int(pl_module.generator_updates)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        generator_update = int(pl_module.generator_updates)
        if generator_update == self.last_generator_update:
            return
        if generator_update != self.last_generator_update + 1:
            raise RuntimeError("profiler observed a non-consecutive generator update")
        self.last_generator_update = generator_update
        self.profiler.step()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--disable_media_logging", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="stereo-vae")
    parser.add_argument("--image_log_every_n_steps", type=int, default=750)
    parser.add_argument("--video_log_every_n_steps", type=int, default=1500)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=500)
    parser.add_argument("--step_timing_output", type=str, default=None)
    parser.add_argument("--step_timing_warmup", type=int, default=5)
    parser.add_argument("--torch_profile_output_dir", type=Path, default=None)
    parser.add_argument("--torch_profile_wait", type=int, default=5)
    parser.add_argument("--torch_profile_warmup", type=int, default=2)
    parser.add_argument("--torch_profile_active", type=int, default=4)
    parser.add_argument("--online_gt_enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--foundation_stereo_backend",
        choices=("pytorch", "tensorrt"),
        default="pytorch",
    )
    parser.add_argument("--foundation_stereo_repo", type=str, default=None)
    parser.add_argument("--foundation_stereo_checkpoint", type=str, default=None)
    parser.add_argument(
        "--foundation_stereo_checkpoint_sha256", type=str, default=None
    )
    parser.add_argument(
        "--foundation_stereo_valid_iters",
        type=int,
        choices=(12, 16, 32),
        default=16,
    )
    parser.add_argument(
        "--foundation_stereo_pair_microbatch", type=int, default=48
    )
    parser.add_argument("--foundation_stereo_engine", type=str, default=None)
    parser.add_argument(
        "--foundation_stereo_engine_sha256", type=str, default=None
    )
    parser.add_argument(
        "--foundation_stereo_engine_manifest", type=str, default=None
    )
    parser.add_argument(
        "--foundation_stereo_engine_manifest_sha256", type=str, default=None
    )
    parser.add_argument(
        "--online_gt_cache_enabled", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--online_gt_cache_root", type=str, default=None)
    parser.add_argument(
        "--online_val_check_interval_steps", type=int, default=500
    )
    parser.add_argument("--da3_repo", type=str, default=None)
    parser.add_argument("--da3_source_sha", type=str, default=None)
    parser.add_argument("--da3_checkpoint", type=str, default=None)
    parser.add_argument("--da3_checkpoint_sha256", type=str, default=None)
    parser.add_argument("--da3_process_res", type=int, default=504)
    parser.add_argument(
        "--da3_process_res_method",
        choices=("upper_bound_resize",),
        default="upper_bound_resize",
    )
    parser.add_argument(
        "--da3_confidence_mask_mode",
        choices=("finite_positive_non_padding",),
        default="finite_positive_non_padding",
    )
    parser = StereoVAE.add_model_specific_args(parser)
    parser = StereoDataModule.add_data_specific_args(parser)
    return parser


def validate_runtime_args(args):
    if args.sequence_length != 4:
        raise ValueError("StereoVAE training requires sequence_length=4")
    if not 0 <= args.single_frame_source_index < args.sequence_length:
        raise ValueError("--single_frame_source_index must be in [0, 3]")
    if args.resolution != 256:
        raise ValueError("the frozen pilot recipe requires resolution=256")
    if args.four_mode_mixed_training:
        if args.batch_size != 24 or args.grad_accumulates != 1:
            raise ValueError("four-mode training is frozen to BS24 and GA1")
        if args.mode_updates_per_epoch < 4 or args.mode_updates_per_epoch % 4:
            raise ValueError("mode_updates_per_epoch must be a positive multiple of 4")
        if args.mode_schedule_start_update < 0:
            raise ValueError("mode_schedule_start_update must be non-negative")
        if (
            args.resume_from_checkpoint is None
            and args.mode_schedule_start_update != 0
        ):
            raise ValueError("mode_schedule_start_update requires a checkpoint")
        remaining_updates = args.max_steps - args.mode_schedule_start_update
        if remaining_updates < 1:
            raise ValueError("resume checkpoint has already reached max_steps")
        if args.mode_updates_per_epoch < remaining_updates:
            raise ValueError(
                "mode_updates_per_epoch must cover all remaining updates so the "
                "stateless resume schedule stays in one data epoch"
            )
        required_mono = {
            "mono_train_manifest": args.mono_train_manifest,
            "mono_val_manifest": args.mono_val_manifest,
            "mono_cache_root": args.mono_cache_root,
            "da3_repo": args.da3_repo,
            "da3_source_sha": args.da3_source_sha,
            "da3_checkpoint": args.da3_checkpoint,
            "da3_checkpoint_sha256": args.da3_checkpoint_sha256,
        }
        missing = [name for name, value in required_mono.items() if not value]
        if missing:
            raise ValueError("four-mode training requires " + ", ".join(missing))
        if args.stereo_data_backend != "lerobot_online":
            raise ValueError("four-mode online-teacher smoke requires lerobot_online")
        if not args.online_gt_enabled:
            raise ValueError("four-mode online-teacher smoke requires online GT")
        if args.da3_process_res != 504:
            raise ValueError("DA3-BASE smoke process resolution is frozen to 504")
        if args.da3_confidence_mask_mode != "finite_positive_non_padding":
            raise ValueError("formal DA3 confidence threshold is not frozen")
        for name, value, length in (
            ("da3_source_sha", args.da3_source_sha, 40),
            ("da3_checkpoint_sha256", args.da3_checkpoint_sha256, 64),
        ):
            if len(value) != length:
                raise ValueError(f"{name} must be a full hexadecimal digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"{name} must be hexadecimal") from error
    geometry_values = {
        "stereo_disparity_min_px": args.stereo_disparity_min_px,
        "stereo_disparity_max_px": args.stereo_disparity_max_px,
        "stereo_lr_error_abs_threshold_px": (
            args.stereo_lr_error_abs_threshold_px
        ),
        "stereo_lr_error_relative_threshold": (
            args.stereo_lr_error_relative_threshold
        ),
    }
    missing_geometry = [
        name for name, value in geometry_values.items() if value is None
    ]
    if missing_geometry:
        raise ValueError(
            "stereo supervision requires " + ", ".join(missing_geometry)
        )
    if not 0 <= args.stereo_disparity_min_px < args.stereo_disparity_max_px:
        raise ValueError("invalid disparity supervision range")
    if args.stereo_lr_error_abs_threshold_px < 0:
        raise ValueError("absolute LR threshold must be non-negative")
    if args.stereo_lr_error_relative_threshold < 0:
        raise ValueError("relative LR threshold must be non-negative")
    if args.stereo_data_backend == "manifest_v3":
        if args.stereo_train_manifest is None:
            raise ValueError("--stereo_train_manifest is required")
        if args.stereo_rgb_root is None or args.stereo_gt_root is None:
            raise ValueError("--stereo_rgb_root and --stereo_gt_root are required")
        if args.online_gt_enabled:
            raise ValueError("Manifest-v3 training cannot enable online GT")
    elif args.stereo_data_backend == "lerobot_online":
        required = {
            "lerobot_episode_manifest": args.lerobot_episode_manifest,
            "lerobot_dataset_root": args.lerobot_dataset_root,
            "lerobot_rectification_audit_sha256": (
                args.lerobot_rectification_audit_sha256
            ),
            "foundation_stereo_checkpoint_sha256": (
                args.foundation_stereo_checkpoint_sha256
            ),
        }
        if args.foundation_stereo_backend == "pytorch":
            required.update(
                {
                    "foundation_stereo_repo": args.foundation_stereo_repo,
                    "foundation_stereo_checkpoint": (
                        args.foundation_stereo_checkpoint
                    ),
                }
            )
        else:
            required.update(
                {
                    "foundation_stereo_engine": args.foundation_stereo_engine,
                    "foundation_stereo_engine_sha256": (
                        args.foundation_stereo_engine_sha256
                    ),
                    "foundation_stereo_engine_manifest": (
                        args.foundation_stereo_engine_manifest
                    ),
                    "foundation_stereo_engine_manifest_sha256": (
                        args.foundation_stereo_engine_manifest_sha256
                    ),
                }
            )
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "LeRobot online training requires " + ", ".join(missing)
            )
        if not args.online_gt_enabled:
            raise ValueError("LeRobot online training requires --online_gt_enabled=1")
        if len(args.lerobot_rectification_audit_sha256) != 64:
            raise ValueError("a full rectification audit SHA256 is required")
        if len(args.foundation_stereo_checkpoint_sha256) != 64:
            raise ValueError("a full FoundationStereo checkpoint SHA256 is required")
        if args.online_gt_cache_enabled and not args.online_gt_cache_root:
            raise ValueError("online GT cache requires --online_gt_cache_root")
        if args.foundation_stereo_pair_microbatch < 1:
            raise ValueError("FoundationStereo pair microbatch must be positive")
        engine_values = (
            args.foundation_stereo_engine,
            args.foundation_stereo_engine_sha256,
            args.foundation_stereo_engine_manifest,
            args.foundation_stereo_engine_manifest_sha256,
        )
        if args.foundation_stereo_backend == "pytorch":
            if any(value is not None for value in engine_values):
                raise ValueError(
                    "PyTorch FoundationStereo forbids TensorRT engine arguments"
                )
        else:
            if args.foundation_stereo_valid_iters != 32:
                raise ValueError("TensorRT FoundationStereo is frozen to 32 iterations")
            if args.foundation_stereo_pair_microbatch > 48:
                raise ValueError(
                    "TensorRT pair microbatch exceeds the frozen max batch 48"
                )
            validate_tensorrt_engine_assets(
                args.foundation_stereo_engine,
                args.foundation_stereo_engine_sha256,
                args.foundation_stereo_engine_manifest,
                args.foundation_stereo_engine_manifest_sha256,
                args.foundation_stereo_checkpoint_sha256,
            )
        if args.online_val_check_interval_steps < 1:
            raise ValueError("online validation interval must be positive")
        if args.lerobot_val_sample_limit != 512:
            raise ValueError("online validation sample count is frozen to 512")
        if args.lerobot_video_cache_capacity < 1:
            raise ValueError("LeRobot video cache capacity must be positive")
        if args.lerobot_maximum_timestamp_error_s <= 0:
            raise ValueError("LeRobot timestamp tolerance must be positive")
    else:
        raise ValueError(f"unsupported stereo data backend {args.stereo_data_backend}")
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if args.devices < 1:
        raise ValueError("--devices must be positive")
    if args.num_nodes < 1:
        raise ValueError("--num_nodes must be positive")
    if (
        args.four_mode_mixed_training
        and args.devices * args.num_nodes not in {1, 2, 8}
    ):
        raise ValueError(
            "the fixed 48-sample four-mode smoke supports 1, 2, or 8 DDP ranks"
        )
    if args.four_mode_mixed_training and args.mixed_stereo_sample_limit != 48:
        raise ValueError("the fixed four-mode smoke requires 48 stereo samples")
    if args.max_steps < 1:
        raise ValueError("--max_steps must be positive")
    if args.checkpoint_every_n_steps < 1:
        raise ValueError("--checkpoint_every_n_steps must be positive")
    if args.step_timing_warmup < 0:
        raise ValueError("--step_timing_warmup must be non-negative")
    if (
        args.step_timing_output is not None
        and args.step_timing_warmup >= args.max_steps
    ):
        raise ValueError("step timing warmup must leave at least one measured update")
    profile_schedule_steps = (
        args.torch_profile_wait
        + args.torch_profile_warmup
        + args.torch_profile_active
    )
    if args.torch_profile_output_dir is not None:
        if min(
            args.torch_profile_wait,
            args.torch_profile_warmup,
            args.torch_profile_active,
        ) < 0 or args.torch_profile_active < 1:
            raise ValueError("invalid torch profiler schedule")
        if profile_schedule_steps >= args.max_steps:
            raise ValueError(
                "torch profiler schedule must leave a post-profile update"
            )
    if args.train_epoch_repeats < 1:
        raise ValueError("--train_epoch_repeats must be positive")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_immutable_json(path, payload):
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"refusing to overwrite mismatched run metadata {path}")
        return serialized
    path.write_text(serialized, encoding="utf-8")
    return serialized


def write_online_gt_run_metadata(args):
    """Persist resolved backend provenance before an online-teacher run."""
    if not args.online_gt_enabled or int(os.environ.get("RANK", "0")) != 0:
        return
    output_root = Path(args.default_root_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = {key: _jsonable(value) for key, value in vars(args).items()}
    resolved_serialized = _write_immutable_json(
        output_root / "resolved_config.json", resolved
    )
    code_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    online_gt = {
        "backend": args.foundation_stereo_backend,
        "checkpoint_sha256": args.foundation_stereo_checkpoint_sha256,
        "valid_iters": args.foundation_stereo_valid_iters,
        "pair_microbatch": args.foundation_stereo_pair_microbatch,
        "bidirectional": True,
        "lr_consistency": True,
    }
    if args.foundation_stereo_backend == "pytorch":
        online_gt.update(
            {
                "repo": str(Path(args.foundation_stereo_repo).resolve()),
                "checkpoint": str(
                    Path(args.foundation_stereo_checkpoint).resolve()
                ),
            }
        )
    else:
        online_gt.update(
            {
                "engine": str(Path(args.foundation_stereo_engine).resolve()),
                "engine_sha256": args.foundation_stereo_engine_sha256,
                "engine_manifest": str(
                    Path(args.foundation_stereo_engine_manifest).resolve()
                ),
                "engine_manifest_sha256": (
                    args.foundation_stereo_engine_manifest_sha256
                ),
            }
        )
    run_manifest = {
        "schema": "stereo-vae-online-gt-run-v1",
        "code_sha": code_sha,
        "resolved_config_sha256": hashlib.sha256(
            resolved_serialized.encode("utf-8")
        ).hexdigest(),
        "online_gt": online_gt,
    }
    if args.four_mode_mixed_training:
        run_manifest["online_gt"]["da3"] = {
            "repo": str(Path(args.da3_repo).resolve()),
            "source_sha": args.da3_source_sha,
            "checkpoint": str(Path(args.da3_checkpoint).resolve()),
            "checkpoint_sha256": args.da3_checkpoint_sha256,
            "process_res": args.da3_process_res,
            "process_res_method": args.da3_process_res_method,
            "confidence_mask_mode": args.da3_confidence_mask_mode,
        }
    _write_immutable_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps({"online_gt_provenance": online_gt}, sort_keys=True))


def build_callbacks(args, has_validation):
    callbacks = []
    if args.online_gt_enabled:
        callbacks.append(OnlineFoundationGTCallback(args))
        if args.four_mode_mixed_training:
            callbacks.append(OnlineDepthAnything3GTCallback(args))
    callbacks.extend([
        ModelCheckpoint(
            every_n_train_steps=args.checkpoint_every_n_steps,
            save_top_k=-1,
            save_last=True,
            filename="{epoch}-{step}",
        )
    ])
    if not args.disable_media_logging:
        callbacks.extend(
            [
                ImageLogger(
                    batch_frequency=args.image_log_every_n_steps,
                    max_images=4,
                    clamp=True,
                ),
                VideoLogger(
                    batch_frequency=args.video_log_every_n_steps,
                    max_videos=4,
                    clamp=True,
                ),
            ]
        )
    if not args.disable_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if args.step_timing_output is not None:
        callbacks.append(
            StepTimingCallback(args.step_timing_output, args.step_timing_warmup)
        )
    if has_validation and (
        args.four_mode_mixed_training
        or args.stereo_data_backend == "manifest_v3"
    ):
        callbacks.append(
            ModelCheckpoint(
                monitor=(
                    "val/mixed/total_loss"
                    if args.four_mode_mixed_training
                    else "val/four/total_loss"
                ),
                every_n_epochs=1,
                save_top_k=3,
                mode="min",
                filename="best-{epoch}-{step}",
            )
        )
    return callbacks


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(
            args.resume_from_checkpoint, map_location="cpu", weights_only=False
        )
        counters = checkpoint.get("stereo_update_counters")
        if not isinstance(counters, dict):
            raise ValueError("resume checkpoint has no stereo update counters")
        args.mode_schedule_start_update = int(counters["generator_updates"])
    validate_runtime_args(args)
    if args.resume_from_checkpoint is not None and args.four_mode_mixed_training:
        if counters.get("mode_schedule_seed") != args.mode_schedule_seed:
            raise ValueError("resume checkpoint mode schedule seed mismatch")
        if counters.get("mode_updates") != mode_occurrences_before(
            args.mode_schedule_seed, args.mode_schedule_start_update
        ):
            raise ValueError("resume checkpoint counters disagree with next mode")
    write_online_gt_run_metadata(args)
    pl.seed_everything(args.seed)

    data = StereoDataModule(args)
    model = StereoVAE(args)
    has_validation = (
        args.stereo_data_backend == "lerobot_online"
        or args.stereo_val_manifest is not None
    )
    callbacks = build_callbacks(args, has_validation)

    profiler = None
    if args.torch_profile_output_dir is not None and int(
        os.environ.get("LOCAL_RANK", "0")
    ) == 0:
        set_profiling_enabled(True)
        trace_writer = TrainingTraceWriter(
            args.torch_profile_output_dir,
            args.torch_profile_active,
        )
        profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=args.torch_profile_wait,
                warmup=args.torch_profile_warmup,
                active=args.torch_profile_active,
                repeat=1,
            ),
            on_trace_ready=trace_writer,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        callbacks.append(TrainingProfilerStepCallback(profiler))

    logger = False
    if not args.disable_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=os.path.basename(os.path.abspath(args.default_root_dir)),
            save_dir=args.default_root_dir,
            config=vars(args),
        )

    precision = "32-true"
    if args.bf16:
        precision = "bf16-mixed"
    elif args.fp16:
        precision = "16-mixed"

    strategy = "auto"
    if args.devices * args.num_nodes > 1:
        strategy = DDPStrategy(
            static_graph=False,
            find_unused_parameters=True,
        )

    val_check_interval = 1.0
    check_val_every_n_epoch = 1
    if args.stereo_data_backend == "lerobot_online":
        val_check_interval = args.online_val_check_interval_steps
        check_val_every_n_epoch = None

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=precision,
        max_steps=-1 if args.gan_enabled else args.max_steps,
        max_epochs=-1,
        default_root_dir=args.default_root_dir,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=1.0 if has_validation else 0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        use_distributed_sampler=False,
    )
    with profiler if profiler is not None else nullcontext():
        trainer.fit(
            model,
            datamodule=data,
            ckpt_path=args.resume_from_checkpoint,
        )


if __name__ == "__main__":
    main()
