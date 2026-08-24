import argparse
import json
import os
import statistics
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.data import StereoDataModule
from stereo_tokenizer.modules.callbacks import ImageLogger, VideoLogger
from stereo_tokenizer.online_gt import OnlineFoundationGTCallback


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
        self.timings.append(
            {
                "step": generator_update,
                "temporal_mode": pl_module.last_temporal_mode,
                "interval_s": now - self.last_batch_end,
            }
        )
        self.last_batch_end = now
        self.last_generator_update = generator_update

    def on_train_end(self, trainer, pl_module):
        local_memory = torch.tensor(
            [
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
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
        payload = {
            "world_size": int(trainer.world_size),
            "per_device_batch_size": int(pl_module.args.batch_size),
            "warmup_updates": self.warmup_updates,
            "peak_memory_bytes_by_rank": [
                {
                    "rank": rank,
                    "allocated": int(memory[0].item()),
                    "reserved": int(memory[1].item()),
                }
                for rank, memory in enumerate(rank_memory)
            ],
            "timings": self.timings,
            "stable": _timing_summary(values),
            "stable_by_temporal_mode": stable_by_temporal_mode,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


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
    parser.add_argument("--online_gt_enabled", type=int, choices=(0, 1), default=0)
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
    parser.add_argument(
        "--online_gt_cache_enabled", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--online_gt_cache_root", type=str, default=None)
    parser.add_argument(
        "--online_val_check_interval_steps", type=int, default=500
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
            "foundation_stereo_repo": args.foundation_stereo_repo,
            "foundation_stereo_checkpoint": args.foundation_stereo_checkpoint,
            "foundation_stereo_checkpoint_sha256": (
                args.foundation_stereo_checkpoint_sha256
            ),
        }
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
    if args.train_epoch_repeats < 1:
        raise ValueError("--train_epoch_repeats must be positive")


def build_callbacks(args, has_validation):
    callbacks = []
    if args.online_gt_enabled:
        callbacks.append(OnlineFoundationGTCallback(args))
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
    if has_validation and args.stereo_data_backend == "manifest_v3":
        callbacks.append(
            ModelCheckpoint(
                monitor="val/four/total_loss",
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
    validate_runtime_args(args)
    pl.seed_everything(args.seed)

    data = StereoDataModule(args)
    model = StereoVAE(args)
    has_validation = (
        args.stereo_data_backend == "lerobot_online"
        or args.stereo_val_manifest is not None
    )
    callbacks = build_callbacks(args, has_validation)

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
    trainer.fit(
        model,
        datamodule=data,
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
