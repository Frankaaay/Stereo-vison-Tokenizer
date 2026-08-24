import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.data import StereoDataModule
from stereo_tokenizer.modules.callbacks import ImageLogger, VideoLogger


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="stereo-vae")
    parser.add_argument("--image_log_every_n_steps", type=int, default=750)
    parser.add_argument("--video_log_every_n_steps", type=int, default=1500)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
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
    if args.stereo_train_manifest is None:
        raise ValueError("--stereo_train_manifest is required")
    if args.stereo_rgb_root is None or args.stereo_gt_root is None:
        raise ValueError("--stereo_rgb_root and --stereo_gt_root are required")
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if args.devices < 1:
        raise ValueError("--devices must be positive")
    if args.num_nodes < 1:
        raise ValueError("--num_nodes must be positive")
    if args.max_steps < 1:
        raise ValueError("--max_steps must be positive")


def build_callbacks(args, has_validation):
    callbacks = [
        ModelCheckpoint(
            every_n_epochs=1,
            save_top_k=-1,
            save_last=True,
            filename="{epoch}-{step}",
        ),
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
    if not args.disable_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if has_validation:
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
    has_validation = args.stereo_val_manifest is not None
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

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp" if args.devices * args.num_nodes > 1 else "auto",
        precision=precision,
        max_steps=-1 if args.gan_enabled else args.max_steps,
        max_epochs=-1,
        default_root_dir=args.default_root_dir,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=1.0 if has_validation else 0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
    )
    trainer.fit(
        model,
        datamodule=data,
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
