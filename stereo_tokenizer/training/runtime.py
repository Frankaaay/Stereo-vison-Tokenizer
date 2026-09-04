"""CLI construction and fail-closed training runtime validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.data import StereoDataModule
from stereo_tokenizer.mode_sampling import (
    MODE_IDS,
    mode_occurrences_before,
    parse_weight_spec,
    resolve_mode_int_spec,
)
from stereo_tokenizer.online_gt import validate_tensorrt_engine_assets


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
    parser.add_argument(
        "--distributed_mode", choices=("single", "ib"), default="single"
    )
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--continuation_checkpoint", type=Path, default=None)
    parser.add_argument("--stage_transition_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--discriminator_expansion_checkpoint", type=Path, default=None
    )
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
        choices=("las2_h", "pytorch", "tensorrt"),
        default="las2_h",
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
    parser.add_argument("--las2_h_repo", type=str, default=None)
    parser.add_argument("--las2_h_source_sha", type=str, default=None)
    parser.add_argument("--las2_h_checkpoint", type=str, default=None)
    parser.add_argument("--las2_h_checkpoint_sha256", type=str, default=None)
    parser.add_argument("--las2_h_valid_iters", type=int, default=4)
    parser.add_argument("--las2_h_max_disp", type=int, default=192)
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


def validate_distributed_runtime_args(args, environ=None):
    environ = os.environ if environ is None else environ
    expected_world_size = args.devices * args.num_nodes
    if args.distributed_mode == "single":
        if args.num_nodes != 1:
            raise ValueError("single distributed mode requires num_nodes=1")
    elif args.distributed_mode == "ib":
        if args.num_nodes != 2:
            raise ValueError("ib distributed mode requires num_nodes=2")
        missing = [
            name
            for name in ("NODE_RANK", "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE")
            if not environ.get(name)
        ]
        if missing:
            raise ValueError(
                "ib distributed mode requires environment variables "
                + ", ".join(missing)
            )
        try:
            node_rank = int(environ["NODE_RANK"])
            launcher_world_size = int(environ["WORLD_SIZE"])
            local_world_size = int(environ.get("LOCAL_WORLD_SIZE", args.devices))
        except ValueError as error:
            raise ValueError("distributed rank and world sizes must be integers") from error
        if not 0 <= node_rank < args.num_nodes:
            raise ValueError("NODE_RANK is outside the configured node range")
        if launcher_world_size != expected_world_size:
            raise ValueError(
                "torchrun WORLD_SIZE does not match devices times num_nodes"
            )
        if local_world_size != args.devices:
            raise ValueError("torchrun LOCAL_WORLD_SIZE does not match devices")
    else:
        raise ValueError(f"unsupported distributed mode {args.distributed_mode}")

    if environ.get("WORLD_SIZE"):
        try:
            launcher_world_size = int(environ["WORLD_SIZE"])
        except ValueError as error:
            raise ValueError("WORLD_SIZE must be an integer") from error
        if launcher_world_size != expected_world_size:
            raise ValueError(
                "launcher WORLD_SIZE does not match devices times num_nodes"
            )


def _validate_four_mode_batch_contract(args):
    if args.grad_accumulates != 1:
        raise ValueError(
            "four-mode training keeps global grad_accumulates=1 and uses "
            "mode_grad_accumulates"
        )
    if args.batch_size < 1:
        raise ValueError("four-mode batch size must be positive")
    world_size = args.devices * args.num_nodes
    if world_size < 1:
        raise ValueError("four-mode DDP world size must be positive")
    mode_batch_sizes = resolve_mode_int_spec(
        getattr(args, "mode_batch_sizes", None),
        fallback=int(args.batch_size),
    )
    mode_grad_accumulates = resolve_mode_int_spec(
        getattr(args, "mode_grad_accumulates", None),
        fallback=int(args.grad_accumulates),
    )
    if any(
        mode_grad_accumulates[mode_id] != 1
        for mode_id in MODE_IDS
        if mode_id.startswith("mono/")
    ):
        raise ValueError("mono modes currently require accumulation factor 1")
    effective_global_batches = {
        mode_id: mode_batch_sizes[mode_id]
        * mode_grad_accumulates[mode_id]
        * world_size
        for mode_id in MODE_IDS
    }
    if len(set(effective_global_batches.values())) != 1:
        raise ValueError(
            "four-mode effective global batch sizes must be equal: "
            f"{effective_global_batches}"
        )
    parse_weight_spec(args.mode_update_weights, MODE_IDS)
    parse_weight_spec(args.mono_dataset_weights, ("hy", "libero"))
    if args.num_nodes > 1 and not args.node_manifest_contracts:
        raise ValueError(
            "multi-node pretraining requires --node_manifest_contracts with "
            "the fixed node-rank to manifest mapping"
        )


def _resolve_val_check_interval(args):
    """Translate logical validation cadence to Lightning physical batches."""

    logical_interval = int(args.online_val_check_interval_steps)
    remaining_updates = int(args.max_steps) - int(args.mode_schedule_start_update)
    logical_interval = min(logical_interval, remaining_updates)
    mode_weights = parse_weight_spec(args.mode_update_weights, MODE_IDS)
    cycle_size = sum(mode_weights.values())
    schedule_start = int(args.mode_schedule_start_update)
    periodic_interval = logical_interval < remaining_updates
    if periodic_interval and (
        logical_interval % cycle_size != 0 or schedule_start % cycle_size != 0
    ):
        raise ValueError(
            "four-mode periodic validation requires a cycle-aligned schedule start "
            f"and whole mode-schedule cycles ({cycle_size} logical updates), or "
            "the interval must cover all remaining updates"
        )
    accumulation = resolve_mode_int_spec(
        args.mode_grad_accumulates,
        fallback=int(args.grad_accumulates),
    )
    before = mode_occurrences_before(
        int(args.mode_schedule_seed),
        schedule_start,
        mode_weights,
    )
    after = mode_occurrences_before(
        int(args.mode_schedule_seed),
        schedule_start + logical_interval,
        mode_weights,
    )
    return sum(
        (after[mode_id] - before[mode_id]) * accumulation[mode_id]
        for mode_id in MODE_IDS
    )


def _bind_node_manifest_contracts(args):
    """Bind each physical node rank to immutable manifest content hashes."""
    local = {}
    for dataset_id in ("hy", "libero", "umi"):
        manifest = getattr(args, f"{dataset_id}_manifest")
        if not manifest:
            raise ValueError(f"three-source training requires {dataset_id}_manifest")
        path = Path(manifest).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        local[dataset_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    node_rank = str(int(os.environ.get("NODE_RANK", "0")))
    if args.node_manifest_contracts:
        candidate = Path(args.node_manifest_contracts).expanduser()
        raw = (
            candidate.read_text(encoding="utf-8")
            if not args.node_manifest_contracts.lstrip().startswith("{")
            and candidate.is_file()
            else args.node_manifest_contracts
        )
        try:
            contracts = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("node_manifest_contracts must be valid JSON") from error
    else:
        contracts = {node_rank: local}
    expected_ranks = {str(rank) for rank in range(int(args.num_nodes))}
    if not isinstance(contracts, dict) or set(contracts) != expected_ranks:
        raise ValueError(
            f"node manifest contracts must contain exactly ranks {sorted(expected_ranks)}"
        )
    for rank, mapping in contracts.items():
        if not isinstance(mapping, dict) or set(mapping) != {"hy", "libero", "umi"}:
            raise ValueError(f"node {rank} must map hy, libero and umi manifests")
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
            for digest in mapping.values()
        ):
            raise ValueError(f"node {rank} manifest hashes must be full SHA256 digests")
    if contracts[node_rank] != local:
        raise ValueError(
            f"node rank {node_rank} manifest files disagree with NODE_MANIFEST_CONTRACTS"
        )
    args.node_manifest_contracts = json.dumps(
        contracts, sort_keys=True, separators=(",", ":")
    )


def validate_runtime_args(args):
    checkpoint_args = (
        getattr(args, "resume_from_checkpoint", None),
        getattr(args, "continuation_checkpoint", None),
        getattr(args, "stage_transition_checkpoint", None),
        getattr(args, "discriminator_expansion_checkpoint", None),
    )
    if sum(value is not None for value in checkpoint_args) > 1:
        raise ValueError(
            "resume_from_checkpoint, continuation_checkpoint, "
            "stage_transition_checkpoint, and "
            "discriminator_expansion_checkpoint are mutually exclusive"
        )
    if (
        getattr(args, "stage_transition_checkpoint", None) is not None
        and not args.gan_enabled
    ):
        raise ValueError("stage_transition_checkpoint requires GAN-enabled training")
    if getattr(args, "discriminator_expansion_checkpoint", None) is not None:
        if not args.gan_enabled:
            raise ValueError(
                "discriminator_expansion_checkpoint requires GAN-enabled training"
            )
        if args.image_gan_weight <= 0 or args.video_gan_weight <= 0:
            raise ValueError(
                "discriminator expansion requires positive image and video GAN weights"
            )
    if args.sequence_length != 4:
        raise ValueError("StereoVAE training requires sequence_length=4")
    if not 0 <= args.single_frame_source_index < args.sequence_length:
        raise ValueError("--single_frame_source_index must be in [0, 3]")
    if args.resolution != 256:
        raise ValueError("the frozen pilot recipe requires resolution=256")
    if args.single_frame_source_index != 0:
        raise ValueError(
            "four-mode training currently requires "
            "--single_frame_source_index=0"
        )
    _validate_four_mode_batch_contract(args)
    if args.mode_updates_per_epoch < 1:
        raise ValueError("mode_updates_per_epoch must be positive")
    if args.mode_schedule_start_update < 0:
        raise ValueError("mode_schedule_start_update must be non-negative")
    if (
        getattr(args, "resume_from_checkpoint", None) is None
        and getattr(args, "continuation_checkpoint", None) is None
        and getattr(args, "stage_transition_checkpoint", None) is None
        and getattr(args, "discriminator_expansion_checkpoint", None) is None
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
    required_sources = {
        "hy_manifest": args.hy_manifest,
        "hy_root_aliases": args.hy_root_aliases,
        "libero_manifest": args.libero_manifest,
        "libero_root_aliases": args.libero_root_aliases,
        "umi_manifest": args.umi_manifest,
        "umi_dataset_root": args.umi_dataset_root,
        "umi_rectification_audit_sha256": (
            args.umi_rectification_audit_sha256
        ),
        "da3_repo": args.da3_repo,
        "da3_source_sha": args.da3_source_sha,
        "da3_checkpoint": args.da3_checkpoint,
        "da3_checkpoint_sha256": args.da3_checkpoint_sha256,
    }
    missing = [name for name, value in required_sources.items() if not value]
    if missing:
        raise ValueError("three-source training requires " + ", ".join(missing))
    if not args.online_gt_enabled:
        raise ValueError("four-mode online-teacher smoke requires online GT")
    if args.da3_process_res != 504:
        raise ValueError("DA3-BASE smoke process resolution is frozen to 504")
    if args.da3_confidence_mask_mode != "finite_positive_non_padding":
        raise ValueError("formal DA3 confidence threshold is not frozen")
    for name, value, length in (
        ("da3_source_sha", args.da3_source_sha, 40),
        ("da3_checkpoint_sha256", args.da3_checkpoint_sha256, 64),
        (
            "umi_rectification_audit_sha256",
            args.umi_rectification_audit_sha256,
            64,
        ),
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
    teacher_sha256 = (
        args.las2_h_checkpoint_sha256
        if args.foundation_stereo_backend == "las2_h"
        else args.foundation_stereo_checkpoint_sha256
    )
    required = {"teacher_checkpoint_sha256": teacher_sha256}
    if args.foundation_stereo_backend == "las2_h":
        required.update(
            {
                "las2_h_repo": args.las2_h_repo,
                "las2_h_source_sha": args.las2_h_source_sha,
                "las2_h_checkpoint": args.las2_h_checkpoint,
                "las2_h_checkpoint_sha256": args.las2_h_checkpoint_sha256,
            }
        )
    elif args.foundation_stereo_backend == "pytorch":
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
            "online stereo teacher requires " + ", ".join(missing)
        )
    if not args.online_gt_enabled:
        raise ValueError("training requires --online_gt_enabled=1")
    if len(teacher_sha256) != 64:
        raise ValueError("a full online teacher checkpoint SHA256 is required")
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
    if args.foundation_stereo_backend == "las2_h":
        if len(args.las2_h_source_sha) != 40:
            raise ValueError("LAS2-H requires a full source Git SHA")
        try:
            int(args.las2_h_source_sha, 16)
        except ValueError as error:
            raise ValueError("LAS2-H source SHA must be hexadecimal") from error
        if args.las2_h_valid_iters < 1:
            raise ValueError("LAS2-H valid_iters must be positive")
        if args.las2_h_max_disp != 192:
            raise ValueError("LAS2-H max_disp is frozen to 192")
        if any(value is not None for value in engine_values):
            raise ValueError("LAS2-H forbids TensorRT engine arguments")
    elif args.foundation_stereo_backend == "pytorch":
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
    if args.lerobot_video_cache_capacity < 1:
        raise ValueError("LeRobot video cache capacity must be positive")
    if args.prefetch_factor < 1:
        raise ValueError("DataLoader prefetch factor must be positive")
    if args.lerobot_maximum_timestamp_error_s <= 0:
        raise ValueError("LeRobot timestamp tolerance must be positive")
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if args.devices < 1:
        raise ValueError("--devices must be positive")
    if args.num_nodes < 1:
        raise ValueError("--num_nodes must be positive")
    validate_distributed_runtime_args(args)
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
