"""Compose Lightning callbacks for the configured training run."""

from __future__ import annotations

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from stereo_tokenizer.modules.callbacks import ImageLogger, VideoLogger
from stereo_tokenizer.online_gt import (
    OnlineDepthAnything3GTCallback,
    OnlineFoundationGTCallback,
)

from .profiling import DiscriminatorExpansionOptimizerCallback, StepTimingCallback


def build_callbacks(args):
    callbacks = []
    if getattr(args, "discriminator_expansion_checkpoint", None) is not None:
        callbacks.append(DiscriminatorExpansionOptimizerCallback())
    if args.step_timing_output is not None:
        callbacks.append(
            StepTimingCallback(args.step_timing_output, args.step_timing_warmup)
        )
    if args.online_gt_enabled:
        callbacks.append(OnlineFoundationGTCallback(args))
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
    callbacks.append(
        ModelCheckpoint(
            monitor="val/mixed/total_loss",
            every_n_epochs=1,
            save_top_k=3,
            mode="min",
            filename="best-{epoch}-{step}",
        )
    )
    return callbacks
