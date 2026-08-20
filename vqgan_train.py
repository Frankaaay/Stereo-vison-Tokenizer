import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from OmniTokenizer import OmniTokenizer_VQGAN, VQGAN, VideoData
from OmniTokenizer.modules.callbacks import ImageLogger, VideoLogger


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=["omnitokenizer"], default="omnitokenizer")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="stereo-omnitokenizer")
    parser.add_argument("--image_log_every_n_steps", type=int, default=750)
    parser.add_argument("--video_log_every_n_steps", type=int, default=1500)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQGAN.add_model_specific_args(parser)
    parser = OmniTokenizer_VQGAN.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    return parser


def validate_runtime_args(args):
    if args.loader_type != "stereo_manifest":
        raise ValueError("Stereo OmniTokenizer training requires loader_type=stereo_manifest")
    if args.sequence_length != 4:
        raise ValueError("Stereo OmniTokenizer training requires sequence_length=4")
    if args.resolution != 256:
        raise ValueError("the frozen pilot recipe requires resolution=256")
    if args.stereo_train_manifest is None:
        raise ValueError("--stereo_train_manifest is required")
    if args.stereo_rgb_root is None or args.stereo_gt_root is None:
        raise ValueError("--stereo_rgb_root and --stereo_gt_root are required")
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")


def build_callbacks(args, has_validation):
    callbacks = [
        ModelCheckpoint(
            every_n_epochs=1,
            save_top_k=-1,
            save_last=True,
            filename="{epoch}-{step}",
        ),
        LearningRateMonitor(logging_interval="step"),
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
    if has_validation:
        callbacks.append(
            ModelCheckpoint(
                monitor="val/total_loss",
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

    data = VideoData(args)
    model = OmniTokenizer_VQGAN(args)
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

    trainer_overrides = {
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": 1,
        "limit_val_batches": 1.0 if has_validation else 0,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 1,
    }
    if args.bf16:
        trainer_overrides["precision"] = "bf16"
    elif args.fp16:
        trainer_overrides["precision"] = 16

    trainer = pl.Trainer.from_argparse_args(args, **trainer_overrides)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
