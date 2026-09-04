"""StereoVAE training CLI."""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.profiler import ProfilerActivity

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.data import StereoDataModule
from stereo_tokenizer.mode_sampling import MODE_IDS, mode_occurrences_before, parse_weight_spec
from stereo_tokenizer.profiling import set_profiling_enabled
from stereo_tokenizer.training.callbacks import build_callbacks
from stereo_tokenizer.training.checkpoints import (
    _load_continuation_checkpoint,
    _load_discriminator_expansion_checkpoint,
    _load_stage_transition_checkpoint,
)
from stereo_tokenizer.training.profiling import (
    TrainingProfilerStepCallback,
    TrainingTraceWriter,
)
from stereo_tokenizer.training.provenance import write_online_gt_run_metadata
from stereo_tokenizer.training.runtime import (
    _bind_node_manifest_contracts,
    _resolve_val_check_interval,
    build_parser,
    validate_runtime_args,
)


def main():
    parser = build_parser()
    args = parser.parse_args()
    checkpoint_args = (
        args.resume_from_checkpoint,
        args.continuation_checkpoint,
        args.stage_transition_checkpoint,
        args.discriminator_expansion_checkpoint,
    )
    if sum(value is not None for value in checkpoint_args) > 1:
        raise ValueError(
            "resume_from_checkpoint, continuation_checkpoint, "
            "stage_transition_checkpoint, and "
            "discriminator_expansion_checkpoint are mutually exclusive"
        )
    checkpoint_path = next(
        (value for value in checkpoint_args if value is not None), None
    )
    checkpoint = None
    counters = None
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        counters = checkpoint.get("stereo_update_counters")
        if not isinstance(counters, dict):
            raise ValueError("resume checkpoint has no stereo update counters")
        args.mode_schedule_start_update = int(counters["generator_updates"])
        if args.continuation_checkpoint is not None:
            digest = hashlib.sha256()
            with args.continuation_checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            args.continuation_checkpoint_sha256 = digest.hexdigest()
            args.continuation_source_generator_updates = int(
                counters["generator_updates"]
            )
            args.continuation_source_contract = {
                key: counters.get(key)
                for key in (
                    "node_manifest_contracts",
                    "per_device_batch_size",
                    "grad_accumulates",
                    "mode_batch_sizes",
                    "mode_grad_accumulates",
                    "mode_effective_global_batch_sizes",
                    "world_size_contract",
                )
            }
    _bind_node_manifest_contracts(args)
    validate_runtime_args(args)
    if checkpoint_path is not None:
        if counters.get("mode_schedule_seed") != args.mode_schedule_seed:
            raise ValueError("resume checkpoint mode schedule seed mismatch")
        if counters.get("mode_updates") != mode_occurrences_before(
            args.mode_schedule_seed,
            args.mode_schedule_start_update,
            parse_weight_spec(args.mode_update_weights, MODE_IDS),
        ):
            raise ValueError("resume checkpoint counters disagree with next mode")
    write_online_gt_run_metadata(args)
    pl.seed_everything(args.seed)

    data = StereoDataModule(args)
    model = StereoVAE(args)
    if args.continuation_checkpoint is not None:
        _load_continuation_checkpoint(
            model,
            checkpoint,
            args.continuation_checkpoint,
        )
    elif args.stage_transition_checkpoint is not None:
        _load_stage_transition_checkpoint(
            model,
            checkpoint,
            args.stage_transition_checkpoint,
        )
    elif args.discriminator_expansion_checkpoint is not None:
        _load_discriminator_expansion_checkpoint(
            model,
            checkpoint,
            args.discriminator_expansion_checkpoint,
        )
    checkpoint = None
    callbacks = build_callbacks(args)

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

    val_check_interval = _resolve_val_check_interval(args)
    check_val_every_n_epoch = None

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=precision,
        max_steps=(
            -1
            if args.gan_enabled
            or getattr(args, "continuation_checkpoint", None) is not None
            else args.max_steps
        ),
        max_epochs=-1,
        default_root_dir=args.default_root_dir,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        use_distributed_sampler=False,
        detect_anomaly=True,
    )
    with profiler if profiler is not None else nullcontext():
        trainer.fit(
            model,
            datamodule=data,
            ckpt_path=args.resume_from_checkpoint,
        )


if __name__ == "__main__":
    main()
