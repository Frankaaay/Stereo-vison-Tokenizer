import argparse
import hashlib
import json
import os
from argparse import Namespace
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.utils import data as torch_data
from torch.utils.data import default_collate
from tqdm import tqdm

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.data import StereoDataModule, _load_root_aliases
from stereo_tokenizer.pretrain_data import HyLanceMonoDataset
from stereo_tokenizer.lerobot_data import fixed_episode_subset_indices
from stereo_tokenizer.mode_sampling import MODE_IDS
from stereo_tokenizer.modules.relative_depth import (
    relative_prediction_from_raw,
    relative_target_from_da3,
    relative_target_from_foundation_stereo,
)
from stereo_tokenizer.online_gt import (
    DepthAnything3OnlineTeacher,
    FoundationStereoOnlineTeacher,
    attach_da3_student_targets,
    sha256_file,
    stereo_supervision_valid_mask,
    validate_git_teacher_assets,
    validate_tensorrt_engine_assets,
)


STEREO_VIEW_NAMES = ("head", "lefthand", "righthand")
MONO_VIEW_NAMES = ("cam_high",)
CHECKPOINT_SEMANTIC_FIELDS = (
    "resolution",
    "image_channels",
    "norm_type",
    "embedding_dim",
    "latent_channels",
    "patch_size",
    "patch_embed",
    "enc_block",
    "dec_block",
    "twod_window_size",
    "defer_temporal_pool",
    "defer_spatial_pool",
    "spatial_pos",
    "spatial_depth",
    "temporal_depth",
    "causal_in_peg",
    "peg_backend",
    "causal_in_temporal_transformer",
    "dim_head",
    "heads",
    "attn_dropout",
    "ff_dropout",
    "ff_mult",
    "stereo_num_views",
    "stereo_num_frames",
    "stereo_search_radii",
    "stereo_search_direction",
    "relative_depth_epsilon",
    "stereo_disparity_min_px",
    "stereo_disparity_max_px",
    "stereo_lr_error_abs_threshold_px",
    "stereo_lr_error_relative_threshold",
)


def requested_eye_modes(args):
    if args.eval_eye_mode == "both":
        return ("mono", "stereo")
    return (args.eval_eye_mode,)


def requested_temporal_modes(args):
    if args.eval_temporal_mode == "both":
        return ("single_frame", "four_frame")
    return (args.eval_temporal_mode,)


def requested_mode_ids(args):
    eyes = set(requested_eye_modes(args))
    temporal_modes = set(requested_temporal_modes(args))
    mode_ids = tuple(
        mode_id
        for mode_id in MODE_IDS
        if mode_id.split("/", maxsplit=1)[0] in eyes
        and mode_id.split("/", maxsplit=1)[1] in temporal_modes
    )
    expected_count = len(eyes) * len(temporal_modes)
    if len(mode_ids) != expected_count or any(
        mode_id not in MODE_IDS for mode_id in mode_ids
    ):
        raise ValueError("requested evaluation modes do not match MODE_IDS")
    return mode_ids


def build_parser():
    parser = argparse.ArgumentParser()
    parser = StereoVAE.add_model_specific_args(parser)
    parser = StereoDataModule.add_data_specific_args(parser)
    parser.add_argument("--stereo_vae_ckpt", type=Path, required=True)
    parser.add_argument(
        "--eval_split", choices=["train", "val", "test"], default="val"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--foundation_stereo_backend",
        choices=("las2_h", "pytorch", "tensorrt"),
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
    parser.add_argument("--las2_h_repo", type=str, default=None)
    parser.add_argument("--las2_h_source_sha", type=str, default=None)
    parser.add_argument("--las2_h_checkpoint", type=str, default=None)
    parser.add_argument("--las2_h_checkpoint_sha256", type=str, default=None)
    parser.add_argument("--las2_h_valid_iters", type=int, default=4)
    parser.add_argument("--las2_h_max_disp", type=int, default=192)
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
    parser.add_argument("--visualization_dir", type=Path, default=None)
    parser.add_argument("--num_visualizations", type=int, default=0)
    parser.add_argument(
        "--eval_eye_mode",
        choices=["mono", "stereo", "both"],
        default="stereo",
    )
    parser.add_argument(
        "--eval_temporal_mode",
        choices=["single_frame", "four_frame", "both"],
        required=True,
    )
    return parser


def validate_args(args):
    requested_mode_ids(args)
    eye_modes = requested_eye_modes(args)
    if "stereo" in eye_modes:
        geometry = {
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
            name for name, value in geometry.items() if value is None
        ]
        if missing_geometry:
            raise ValueError(
                "stereo evaluation requires " + ", ".join(missing_geometry)
            )
        if not 0 <= args.stereo_disparity_min_px < args.stereo_disparity_max_px:
            raise ValueError("invalid disparity supervision range")
        if args.stereo_lr_error_abs_threshold_px < 0:
            raise ValueError("absolute LR threshold must be non-negative")
        if args.stereo_lr_error_relative_threshold < 0:
            raise ValueError("relative LR threshold must be non-negative")
    if "mono" in eye_modes:
        required = {
            "hy_manifest": args.hy_manifest,
            "hy_root_aliases": args.hy_root_aliases,
            "da3_repo": args.da3_repo,
            "da3_source_sha": args.da3_source_sha,
            "da3_checkpoint": args.da3_checkpoint,
            "da3_checkpoint_sha256": args.da3_checkpoint_sha256,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("mono DA3 evaluation requires " + ", ".join(missing))
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
        if args.da3_process_res != 504:
            raise ValueError("DA3-BASE evaluation process resolution is frozen to 504")
        if args.da3_confidence_mask_mode != "finite_positive_non_padding":
            raise ValueError("formal DA3 evaluation forbids confidence thresholding")
    if "stereo" in eye_modes:
        teacher_sha256 = (
            args.las2_h_checkpoint_sha256
            if args.foundation_stereo_backend == "las2_h"
            else args.foundation_stereo_checkpoint_sha256
        )
        required = {
            "lerobot_episode_manifest": args.lerobot_episode_manifest,
            "lerobot_dataset_root": args.lerobot_dataset_root,
            "lerobot_rectification_audit_sha256": (
                args.lerobot_rectification_audit_sha256
            ),
            "teacher_checkpoint_sha256": teacher_sha256,
        }
        if args.foundation_stereo_backend == "las2_h":
            required.update(
                {
                    "las2_h_repo": args.las2_h_repo,
                    "las2_h_source_sha": args.las2_h_source_sha,
                    "las2_h_checkpoint": args.las2_h_checkpoint,
                }
            )
        elif args.foundation_stereo_backend == "pytorch":
            required.update(
                {
                    "foundation_stereo_repo": args.foundation_stereo_repo,
                    "foundation_stereo_checkpoint": args.foundation_stereo_checkpoint,
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
                "LeRobot online evaluation requires " + ", ".join(missing)
            )
        if len(args.lerobot_rectification_audit_sha256) != 64:
            raise ValueError("a full rectification audit SHA256 is required")
        if len(teacher_sha256) != 64:
            raise ValueError("a full online teacher checkpoint SHA256 is required")
        if args.foundation_stereo_pair_microbatch < 1:
            raise ValueError("FoundationStereo pair microbatch must be positive")
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
        elif args.foundation_stereo_backend == "tensorrt":
            if args.foundation_stereo_valid_iters != 32:
                raise ValueError("TensorRT FoundationStereo is frozen to 32 iterations")
            if args.foundation_stereo_pair_microbatch > 48:
                raise ValueError("TensorRT pair microbatch exceeds frozen max batch 48")
            validate_tensorrt_engine_assets(
                args.foundation_stereo_engine,
                args.foundation_stereo_engine_sha256,
                args.foundation_stereo_engine_manifest,
                args.foundation_stereo_engine_manifest_sha256,
                args.foundation_stereo_checkpoint_sha256,
            )
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max_batches must be positive")
    if args.num_visualizations < 0:
        raise ValueError("--num_visualizations must be non-negative")
    if args.num_visualizations and args.visualization_dir is None:
        raise ValueError("visualizations require --visualization_dir")
    if args.visualization_dir is not None and args.visualization_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite visualization directory {args.visualization_dir}"
        )
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    if not 0 <= args.single_frame_source_index < args.stereo_num_frames:
        raise ValueError("--single_frame_source_index must be in [0, 3]")


def preflight_teacher_assets(args, eye_modes):
    """Validate requested teacher assets before model or CUDA initialization."""
    if "mono" in eye_modes:
        validate_git_teacher_assets(
            args.da3_repo,
            args.da3_source_sha,
            args.da3_checkpoint,
            args.da3_checkpoint_sha256,
            label="DA3",
            checkpoint_is_directory=True,
        )
    if "stereo" not in eye_modes:
        return
    backend = args.foundation_stereo_backend
    if backend == "las2_h":
        validate_git_teacher_assets(
            args.las2_h_repo,
            args.las2_h_source_sha,
            args.las2_h_checkpoint,
            args.las2_h_checkpoint_sha256,
            label="LAS2-H",
        )
    elif backend == "pytorch":
        repo = Path(args.foundation_stereo_repo).expanduser().resolve()
        checkpoint = Path(args.foundation_stereo_checkpoint).expanduser().resolve()
        if not repo.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError("FoundationStereo repo/checkpoint is missing")
        if sha256_file(checkpoint) != args.foundation_stereo_checkpoint_sha256:
            raise ValueError("FoundationStereo checkpoint SHA256 mismatch")


def dataset_provenance(args, eye_mode, dataset):
    if eye_mode == "mono":
        return {
            "manifest": str(dataset.manifest_path),
            "cache_root": str(dataset.cache_root),
            "sample_count": len(dataset),
            "video_contract": "[B,1,1,3,T,H,W]",
        }
    return {
        "manifest": str(dataset.manifest_path),
        "dataset_root": str(dataset.dataset_root),
        "split": args.eval_split,
        "sample_count": len(dataset),
        "rectification_audit_sha256": args.lerobot_rectification_audit_sha256,
        "video_contract": "[B,3,2,3,T,H,W]",
    }


def teacher_provenance(args, eye_mode):
    if eye_mode == "mono":
        return {
            "family": "depth_anything_3_base",
            "source_sha": args.da3_source_sha,
            "checkpoint_sha256": args.da3_checkpoint_sha256,
            "process_res": args.da3_process_res,
            "process_res_method": args.da3_process_res_method,
            "confidence_mask_mode": args.da3_confidence_mask_mode,
        }
    return {
        "family": "foundation_stereo",
        "backend": args.foundation_stereo_backend,
        "source_sha": (
            args.las2_h_source_sha
            if args.foundation_stereo_backend == "las2_h"
            else None
        ),
        "checkpoint_sha256": (
            args.las2_h_checkpoint_sha256
            if args.foundation_stereo_backend == "las2_h"
            else args.foundation_stereo_checkpoint_sha256
        ),
    }


def _checkpoint_model_args(checkpoint, checkpoint_path):
    hyperparameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError(f"{checkpoint_path}: missing checkpoint hyper_parameters")

    saved_args = hyperparameters.get("args", hyperparameters)
    if isinstance(saved_args, Namespace):
        return saved_args
    if isinstance(saved_args, Mapping):
        return Namespace(**saved_args)
    raise TypeError(
        f"{checkpoint_path}: checkpoint args must be a Namespace or mapping"
    )


def _comparable_config_value(value):
    return tuple(value) if isinstance(value, (list, tuple)) else value


def _validate_checkpoint_semantics(cli_args, checkpoint_args, checkpoint_path):
    mismatches = []
    for name in CHECKPOINT_SEMANTIC_FIELDS:
        if not hasattr(checkpoint_args, name):
            raise ValueError(f"{checkpoint_path}: checkpoint args missing {name}")
        if not hasattr(cli_args, name):
            raise ValueError(f"evaluation args missing {name}")
        cli_value = _comparable_config_value(getattr(cli_args, name))
        checkpoint_value = _comparable_config_value(getattr(checkpoint_args, name))
        if cli_value != checkpoint_value:
            mismatches.append(
                f"{name}: cli={cli_value!r}, checkpoint={checkpoint_value!r}"
            )
    if mismatches:
        raise ValueError(
            "evaluation configuration does not match checkpoint semantics: "
            + "; ".join(mismatches)
        )


def load_model(args, device):
    checkpoint = torch.load(
        args.stereo_vae_ckpt, map_location="cpu", weights_only=False
    )
    if "state_dict" not in checkpoint:
        raise ValueError(f"{args.stereo_vae_ckpt}: missing state_dict")
    checkpoint_args = _checkpoint_model_args(checkpoint, args.stereo_vae_ckpt)
    _validate_checkpoint_semantics(args, checkpoint_args, args.stereo_vae_ckpt)
    model = StereoVAE(checkpoint_args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model


def initialize_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not args.device.startswith("cuda"):
            raise ValueError("distributed evaluation requires a CUDA device")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    return device, rank, world_size


def _exact_lerobot_rank_indices(dataset, rank, world_size):
    by_shard = defaultdict(list)
    for span in dataset.episode_spans:
        by_shard[span.shard_id].append(span)
    assigned = []
    for position, shard_id in enumerate(sorted(by_shard)):
        if position % world_size == rank:
            assigned.extend(by_shard[shard_id])
    indices = []
    for span in assigned:
        indices.extend(range(span.first_sample, span.first_sample + span.sample_count))
    if not indices:
        raise ValueError(f"distributed evaluation rank {rank} received no samples")
    return indices


def _exact_mono_rank_indices(dataset, rank, world_size):
    indices = list(range(rank, len(dataset), world_size))
    if not indices:
        raise ValueError(f"distributed evaluation rank {rank} received no samples")
    return indices


def build_eval_dataset(args, eye_mode):
    if eye_mode == "mono":
        return HyLanceMonoDataset(
            args.hy_manifest,
            _load_root_aliases(args.hy_root_aliases, "--hy_root_aliases"),
            split=args.eval_split,
            single_frame_source_index=args.single_frame_source_index,
        )
    if eye_mode == "stereo":
        data_module = StereoDataModule(args, shuffle=False)
        dataset = data_module._dataset(False, split=args.eval_split)
        for record in dataset.records:
            for video in record["videos"].values():
                video_path = (
                    dataset.dataset_root / video["relative_path"]
                ).resolve()
                if not video_path.is_relative_to(dataset.dataset_root):
                    raise ValueError(f"video path escapes dataset root: {video_path}")
                if not video_path.is_file():
                    raise FileNotFoundError(video_path)
        return dataset
    raise ValueError(f"unsupported eye mode {eye_mode!r}")


def exact_eval_loader(args, dataset, eye_mode, rank, world_size):
    if eye_mode == "mono":
        indices = _exact_mono_rank_indices(dataset, rank, world_size)
    elif eye_mode == "stereo":
        indices = _exact_lerobot_rank_indices(dataset, rank, world_size)
    else:
        raise ValueError(f"unsupported eye mode {eye_mode!r}")
    subset = torch_data.Subset(dataset, indices)
    loader = torch_data.DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers) and args.num_workers > 0,
        collate_fn=default_collate,
        shuffle=False,
        drop_last=False,
    )
    return loader


def fixed_eval_case_indices(dataset, count, seed, eye_mode):
    if eye_mode == "stereo":
        return fixed_episode_subset_indices(dataset, count, seed=seed)
    if count < 1 or count > len(dataset):
        raise ValueError(
            f"fixed mono subset needs 1..{len(dataset)} samples, got {count}"
        )
    return sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{dataset.records[index]['sample_id']}".encode("utf-8")
        ).digest(),
    )[:count]


def build_online_teacher(args, eye_mode, device):
    if eye_mode == "mono":
        return DepthAnything3OnlineTeacher(
            args.da3_repo,
            args.da3_source_sha,
            args.da3_checkpoint,
            args.da3_checkpoint_sha256,
            device=device,
            process_res=args.da3_process_res,
            process_res_method=args.da3_process_res_method,
        )
    if eye_mode != "stereo":
        raise ValueError(f"unsupported eye mode {eye_mode!r}")
    backend = args.foundation_stereo_backend
    valid_iters = (
        args.las2_h_valid_iters
        if backend == "las2_h"
        else args.foundation_stereo_valid_iters
    )
    repo = args.las2_h_repo if backend == "las2_h" else args.foundation_stereo_repo
    checkpoint = (
        args.las2_h_checkpoint
        if backend == "las2_h"
        else args.foundation_stereo_checkpoint
    )
    checkpoint_sha256 = (
        args.las2_h_checkpoint_sha256
        if backend == "las2_h"
        else args.foundation_stereo_checkpoint_sha256
    )
    return FoundationStereoOnlineTeacher(
        repo,
        checkpoint,
        checkpoint_sha256,
        device=device,
        valid_iters=valid_iters,
        pair_microbatch=args.foundation_stereo_pair_microbatch,
        backend=backend,
        engine=args.foundation_stereo_engine,
        engine_sha256=args.foundation_stereo_engine_sha256,
        engine_manifest=args.foundation_stereo_engine_manifest,
        engine_manifest_sha256=args.foundation_stereo_engine_manifest_sha256,
        las2_h_repo=args.las2_h_repo,
        las2_h_source_sha=args.las2_h_source_sha,
        las2_h_checkpoint=args.las2_h_checkpoint,
        las2_h_checkpoint_sha256=args.las2_h_checkpoint_sha256,
        las2_h_valid_iters=args.las2_h_valid_iters,
        las2_h_max_disp=args.las2_h_max_disp,
    )


def attach_online_targets(args, eye_mode, teacher, batch):
    if eye_mode == "mono":
        native_depth, native_confidence = teacher.infer_processed(
            batch["da3_images"]
        )
        attach_da3_student_targets(
            batch,
            native_depth,
            native_confidence,
            process_res=args.da3_process_res,
            process_res_method=args.da3_process_res_method,
        )
        return
    if eye_mode != "stereo":
        raise ValueError(f"unsupported eye mode {eye_mode!r}")
    disparity, residual, base_valid = teacher.infer(batch["video"])
    batch["disparity"] = disparity
    batch["valid_mask"] = stereo_supervision_valid_mask(
        disparity,
        residual,
        base_valid,
        disparity_min_px=args.stereo_disparity_min_px,
        disparity_max_px=args.stereo_disparity_max_px,
        lr_error_abs_threshold_px=args.stereo_lr_error_abs_threshold_px,
        lr_error_relative_threshold=args.stereo_lr_error_relative_threshold,
    )

def batch_for_temporal_mode(batch, mode, source_index):
    if mode == "four_frame":
        return batch
    result = dict(batch)
    for key in (
        "video",
        "disparity",
        "da3_relative_depth",
        "da3_confidence",
        "valid_mask",
        "non_padding_mask",
    ):
        if key in batch:
            result[key] = batch[key][..., source_index : source_index + 1, :, :]
    if "da3_images" in batch:
        result["da3_images"] = batch["da3_images"][
            :, source_index : source_index + 1
        ]
    eye_modes = result.get("eye_mode", ())
    if isinstance(eye_modes, str):
        result["temporal_mode"] = "single_frame"
        result["mode_id"] = f"{eye_modes}/single_frame"
    else:
        eye_modes = list(eye_modes)
        result["temporal_mode"] = ["single_frame"] * len(eye_modes)
        result["mode_id"] = [f"{eye_mode}/single_frame" for eye_mode in eye_modes]
    return result


def reduce_accumulator(accumulator, world_size):
    if world_size == 1:
        return
    for value in accumulator.values():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)


def _rgb_image(tensor):
    array = (
        tensor.detach()
        .float()
        .clamp(-0.5, 0.5)
        .add(0.5)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _heatmap_image(tensor, valid, value_min, value_max):
    values = tensor.detach().float().cpu().numpy()
    mask = valid.detach().bool().cpu().numpy()
    scale = max(float(value_max) - float(value_min), 1e-6)
    normalized = np.clip((values - float(value_min)) / scale, 0.0, 1.0)
    anchors = np.asarray(
        [
            [48, 18, 59],
            [45, 100, 190],
            [35, 185, 155],
            [235, 215, 65],
            [180, 25, 35],
        ],
        dtype=np.float32,
    )
    position = normalized * (len(anchors) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (position - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    rgb[~mask] = 0
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _robust_range(tensor, valid, low=0.02, high=0.98):
    values = tensor.detach().float()[valid]
    if values.numel() == 0:
        raise ValueError("depth visualization has no valid pixels")
    value_min = float(torch.quantile(values, low).item())
    value_max = float(torch.quantile(values, high).item())
    if value_max <= value_min:
        value_max = value_min + 1e-6
    return value_min, value_max


def _batch_teacher_kind(batch):
    values = batch.get("teacher_kind", ())
    if isinstance(values, str):
        values = (values,)
    values = list(values)
    if not values or any(value != values[0] for value in values):
        raise ValueError("evaluation batch must contain one teacher kind")
    return values[0]


def _relative_target_from_batch(batch, relative_depth_epsilon):
    teacher_kind = _batch_teacher_kind(batch)
    if teacher_kind == "foundation_stereo":
        return relative_target_from_foundation_stereo(
            batch["disparity"],
            batch["valid_mask"],
            batch["fx"],
            batch["baseline_m"],
            epsilon=relative_depth_epsilon,
        )
    if teacher_kind == "da3":
        return relative_target_from_da3(
            batch["da3_relative_depth"],
            batch["valid_mask"],
            epsilon=relative_depth_epsilon,
        )
    raise ValueError(f"unsupported evaluation teacher kind {teacher_kind!r}")


def save_depth_case_visualization(
    output_path,
    batch,
    outputs,
    source_index,
    relative_depth_epsilon,
    view_names,
):
    four = outputs.get("four_frame")
    single = outputs.get("single_frame")
    if not outputs:
        raise ValueError("no temporal output to visualize")

    target = None
    four_relative = None
    if four is not None:
        target = _relative_target_from_batch(batch, relative_depth_epsilon)
        four_relative, _ = relative_prediction_from_raw(
            four.raw_relative_log_depth, batch["valid_mask"]
        )
    single_batch = None
    single_target = None
    single_relative = None
    if single is not None:
        single_batch = batch_for_temporal_mode(
            batch, "single_frame", source_index
        )
        single_target = _relative_target_from_batch(
            single_batch, relative_depth_epsilon
        )
        single_relative, _ = relative_prediction_from_raw(
            single.raw_relative_log_depth, single_batch["valid_mask"]
        )

    cell = 192
    label_width = 205
    header_height = 85
    columns = 4
    rows_per_view = 4 if four is not None and single is not None else (
        3 if four is not None else 1
    )
    rows = len(view_names) * rows_per_view
    canvas = Image.new(
        "RGB",
        (label_width + columns * cell, header_height + rows * cell),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"sample: {batch['sample_id'][0]}", fill="black")
    draw.text((12, 30), "relative log-depth; black = invalid", fill="black")
    headers = (
        ("t0", "t1", "t2", "t3")
        if four is not None
        else (f"GT t{source_index}", "single prediction", "single error", "valid mask")
    )
    for column, label in enumerate(headers):
        draw.text((label_width + column * cell + 8, 62), label, fill="black")

    for view_index, view_name in enumerate(view_names):
        first_row = view_index * rows_per_view
        if four is not None:
            valid = batch["valid_mask"][0, view_index, 0]
            gt = target.relative_log_depth[0, view_index, 0]
            pred_four = four_relative[0, view_index, 0]
            depth_min, depth_max = _robust_range(gt, valid)
            four_error = (pred_four - gt).abs()
            error_values = [four_error[valid]]
            if single is not None:
                single_valid = single_batch["valid_mask"][0, view_index, 0, 0]
                gt_single = single_target.relative_log_depth[0, view_index, 0, 0]
                pred_single = single_relative[0, view_index, 0, 0]
                single_error = (pred_single - gt_single).abs()
                error_values.append(single_error[single_valid])
            error_max = max(
                float(torch.quantile(torch.cat(error_values).float(), 0.98)),
                1e-3,
            )
            labels = (
                f"{view_name} GT log-ratio [{depth_min:.2f},{depth_max:.2f}]",
                f"{view_name} four prediction",
                f"{view_name} four abs log error [0,{error_max:.2f}]",
            )
            for offset, label in enumerate(labels):
                draw.text(
                    (8, header_height + (first_row + offset) * cell + 8),
                    label,
                    fill="black",
                )
            for frame_index in range(4):
                x = label_width + frame_index * cell
                images = (
                    _heatmap_image(gt[frame_index], valid[frame_index], depth_min, depth_max),
                    _heatmap_image(pred_four[frame_index], valid[frame_index], depth_min, depth_max),
                    _heatmap_image(four_error[frame_index], valid[frame_index], 0.0, error_max),
                )
                for offset, image in enumerate(images):
                    canvas.paste(
                        image.resize((cell, cell)),
                        (x, header_height + (first_row + offset) * cell),
                    )
            if single is None:
                continue
            single_row = first_row + 3
            draw.text(
                (8, header_height + single_row * cell + 8),
                f"{view_name} single: GT / pred / error / mask",
                fill="black",
            )
        else:
            single_valid = single_batch["valid_mask"][0, view_index, 0, 0]
            gt_single = single_target.relative_log_depth[0, view_index, 0, 0]
            pred_single = single_relative[0, view_index, 0, 0]
            depth_min, depth_max = _robust_range(gt_single, single_valid)
            single_error = (pred_single - gt_single).abs()
            error_max = max(
                float(torch.quantile(single_error[single_valid].float(), 0.98)),
                1e-3,
            )
            single_row = first_row
            draw.text(
                (8, header_height + single_row * cell + 8),
                f"{view_name} single t{source_index}",
                fill="black",
            )
        single_images = (
            _heatmap_image(gt_single, single_valid, depth_min, depth_max),
            _heatmap_image(pred_single, single_valid, depth_min, depth_max),
            _heatmap_image(single_error, single_valid, 0.0, error_max),
            Image.fromarray(
                single_valid.detach().byte().mul(255).cpu().numpy(), mode="L"
            ).convert("RGB"),
        )
        for column, image in enumerate(single_images):
            canvas.paste(
                image.resize((cell, cell)),
                (label_width + column * cell, header_height + single_row * cell),
            )
    canvas.save(output_path)


def save_case_visualization(
    output_path,
    sample_id,
    episode_id,
    video,
    outputs,
    view_names,
    source_index,
):
    if not outputs:
        raise ValueError("no temporal output to visualize")
    four = outputs.get("four_frame")
    single = outputs.get("single_frame")
    cell = 256
    label_width = 150
    header_height = 70
    columns = 5 if four is not None and single is not None else (
        4 if four is not None else 1
    )
    rows = len(view_names) * 2
    canvas = Image.new(
        "RGB",
        (label_width + columns * cell, header_height + rows * cell),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"sample: {sample_id}", fill="black")
    draw.text((12, 30), f"episode: {episode_id}", fill="black")
    if four is not None and single is not None:
        headers = ("t0", "t1", "t2", "t3", f"single t{source_index}")
    elif four is not None:
        headers = ("t0", "t1", "t2", "t3")
    else:
        headers = (f"single t{source_index}",)
    for column, label in enumerate(headers):
        draw.text((label_width + column * cell + 8, 48), label, fill="black")

    for view_index, view_name in enumerate(view_names):
        input_row = view_index * 2
        recon_row = input_row + 1
        draw.text((8, header_height + input_row * cell + 8), f"{view_name} input", fill="black")
        draw.text((8, header_height + recon_row * cell + 8), f"{view_name} recon", fill="black")
        if four is not None:
            for frame_index in range(4):
                source = _rgb_image(video[0, view_index, 0, :, frame_index])
                canvas.paste(
                    source,
                    (label_width + frame_index * cell, header_height + input_row * cell),
                )
                reconstruction = _rgb_image(
                    four.rgb[0, view_index, :, frame_index]
                )
                canvas.paste(
                    reconstruction,
                    (
                        label_width + frame_index * cell,
                        header_height + recon_row * cell,
                    ),
                )
        if single is not None and four is not None:
            source = _rgb_image(video[0, view_index, 0, :, source_index])
            canvas.paste(
                source,
                (label_width + 4 * cell, header_height + input_row * cell),
            )
            reconstruction = _rgb_image(single.rgb[0, view_index, :, 0])
            canvas.paste(
                reconstruction,
                (label_width + 4 * cell, header_height + recon_row * cell),
            )
        elif single is not None:
            source = _rgb_image(video[0, view_index, 0, :, source_index])
            reconstruction = _rgb_image(single.rgb[0, view_index, :, 0])
            canvas.paste(
                source,
                (label_width, header_height + input_row * cell),
            )
            canvas.paste(
                reconstruction,
                (label_width, header_height + recon_row * cell),
            )
    canvas.save(output_path)


def empty_accumulator(device, view_count):
    return {
        "sample_count": torch.zeros((), dtype=torch.long, device=device),
        "rgb_abs_sum": torch.zeros((), dtype=torch.float64, device=device),
        "rgb_count": torch.zeros((), dtype=torch.long, device=device),
        "relative_log_abs_sum": torch.zeros(
            view_count, dtype=torch.float64, device=device
        ),
        "valid_count": torch.zeros(view_count, dtype=torch.long, device=device),
        "relative_log_sq_sum": torch.zeros(
            view_count, dtype=torch.float64, device=device
        ),
    }


def update_metrics(accumulator, batch, output, relative_depth_epsilon):
    rgb_target = batch["video"][:, :, 0]
    expected_views = int(accumulator["valid_count"].numel())
    if int(rgb_target.shape[1]) != expected_views:
        raise ValueError(
            f"evaluation accumulator expects {expected_views} views, "
            f"batch contains {int(rgb_target.shape[1])}"
        )
    rgb_error = (output.rgb - rgb_target).abs()
    accumulator["sample_count"] += rgb_target.shape[0]
    accumulator["rgb_abs_sum"] += rgb_error.double().sum()
    accumulator["rgb_count"] += rgb_error.numel()

    valid = batch["valid_mask"]
    target = _relative_target_from_batch(batch, relative_depth_epsilon)
    prediction, _ = relative_prediction_from_raw(
        output.raw_relative_log_depth, valid
    )
    relative_error = prediction - target.relative_log_depth
    reduction_dims = (0, 2, 3, 4, 5)
    accumulator["relative_log_abs_sum"] += (
        relative_error.abs().double().masked_fill(~valid, 0).sum(dim=reduction_dims)
    )
    accumulator["valid_count"] += valid.sum(dim=reduction_dims)
    accumulator["relative_log_sq_sum"] += (
        relative_error.square()
        .double()
        .masked_fill(~valid, 0)
        .sum(dim=reduction_dims)
    )


def finalize_metrics(accumulator, view_names):
    if len(view_names) != int(accumulator["valid_count"].numel()):
        raise ValueError("view names disagree with evaluation accumulator")
    if accumulator["sample_count"].item() == 0:
        raise ValueError("evaluation loader produced no samples")
    if torch.any(accumulator["valid_count"] == 0):
        raise ValueError("at least one view has no valid relative-depth pixels")
    valid_count = accumulator["valid_count"].double()
    result = {
        "sample_count": int(accumulator["sample_count"].item()),
        "rgb_l1": float(
            (accumulator["rgb_abs_sum"] / accumulator["rgb_count"]).item()
        ),
        "valid_pixels": int(accumulator["valid_count"].sum().item()),
        "views": {},
    }
    for view_index, view_name in enumerate(view_names):
        result["views"][view_name] = {
            "valid_pixels": int(accumulator["valid_count"][view_index].item()),
            "relative_log_l1": float(
                (
                    accumulator["relative_log_abs_sum"][view_index]
                    / valid_count[view_index]
                ).item()
            ),
            "relative_log_rmse": float(
                torch.sqrt(
                    accumulator["relative_log_sq_sum"][view_index]
                    / valid_count[view_index]
                ).item()
            ),
        }
    return result


def evaluate_eye_mode(
    args,
    eye_mode,
    temporal_modes,
    dataset,
    teacher,
    model,
    device,
    rank,
    world_size,
):
    """Evaluate one native data/teacher contract across requested time modes."""
    loader = exact_eval_loader(args, dataset, eye_mode, rank, world_size)
    view_names = MONO_VIEW_NAMES if eye_mode == "mono" else STEREO_VIEW_NAMES
    accumulators = {
        temporal_mode: empty_accumulator(device, len(view_names))
        for temporal_mode in temporal_modes
    }
    with torch.inference_mode():
        progress = tqdm(loader, disable=rank != 0, desc=eye_mode)
        for batch_index, batch in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            attach_online_targets(args, eye_mode, teacher, tensor_batch)
            for temporal_mode in temporal_modes:
                mode_batch = batch_for_temporal_mode(
                    tensor_batch,
                    temporal_mode,
                    args.single_frame_source_index,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    output = model(
                        mode_batch["video"],
                        eye_mode=eye_mode,
                        temporal_mode=temporal_mode,
                        sample_posterior=False,
                    )
                update_metrics(
                    accumulators[temporal_mode],
                    mode_batch,
                    output,
                    args.relative_depth_epsilon,
                )
    for accumulator in accumulators.values():
        reduce_accumulator(accumulator, world_size)
    dataset_metadata = dataset_provenance(args, eye_mode, dataset)
    teacher_metadata = teacher_provenance(args, eye_mode)
    metrics_by_mode = {}
    for temporal_mode, accumulator in accumulators.items():
        mode_id = f"{eye_mode}/{temporal_mode}"
        if mode_id not in MODE_IDS:
            raise ValueError(f"unsupported mode ID {mode_id!r}")
        metrics = finalize_metrics(accumulator, view_names)
        metrics["dataset"] = dataset_metadata
        metrics["teacher"] = teacher_metadata
        metrics_by_mode[mode_id] = metrics
    if args.max_batches is None:
        for mode_id, metrics in metrics_by_mode.items():
            if metrics["sample_count"] != len(dataset):
                raise RuntimeError(
                    f"{mode_id} evaluated {metrics['sample_count']} samples, "
                    f"expected exact split size {len(dataset)}"
                )

    visualization_records = []
    if args.num_visualizations:
        eye_directory = args.visualization_dir / eye_mode
        if rank == 0:
            eye_directory.mkdir(parents=False, exist_ok=False)
        if world_size > 1:
            dist.barrier()
        case_indices = fixed_eval_case_indices(
            dataset,
            args.num_visualizations,
            seed=int(getattr(args, "seed", 1234)),
            eye_mode=eye_mode,
        )
        local_records = []
        with torch.inference_mode():
            for slot in range(rank, args.num_visualizations, world_size):
                batch = default_collate([dataset[case_indices[slot]]])
                tensor_batch = {
                    key: value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }
                attach_online_targets(args, eye_mode, teacher, tensor_batch)
                outputs = {}
                for temporal_mode in temporal_modes:
                    mode_batch = batch_for_temporal_mode(
                        tensor_batch,
                        temporal_mode,
                        args.single_frame_source_index,
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=args.bf16,
                    ):
                        outputs[temporal_mode] = model(
                            mode_batch["video"],
                            eye_mode=eye_mode,
                            temporal_mode=temporal_mode,
                            sample_posterior=False,
                        )
                filename = f"case-{slot:02d}.png"
                depth_filename = f"depth-case-{slot:02d}.png"
                save_case_visualization(
                    eye_directory / filename,
                    tensor_batch["sample_id"][0],
                    tensor_batch["episode_id"][0],
                    tensor_batch["video"],
                    outputs,
                    view_names,
                    args.single_frame_source_index,
                )
                save_depth_case_visualization(
                    eye_directory / depth_filename,
                    tensor_batch,
                    outputs,
                    args.single_frame_source_index,
                    args.relative_depth_epsilon,
                    view_names,
                )
                local_records.append(
                    {
                        "slot": slot,
                        "eye_mode": eye_mode,
                        "temporal_modes": list(temporal_modes),
                        "sample_id": tensor_batch["sample_id"][0],
                        "episode_id": tensor_batch["episode_id"][0],
                        "file": f"{eye_mode}/{filename}",
                        "depth_file": f"{eye_mode}/{depth_filename}",
                    }
                )
        if world_size > 1:
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(local_records, gathered, dst=0)
            if rank == 0:
                visualization_records = sorted(
                    (item for part in gathered for item in part),
                    key=lambda item: item["slot"],
                )
        else:
            visualization_records = local_records
        if rank == 0:
            (eye_directory / "cases.json").write_text(
                json.dumps(visualization_records, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return metrics_by_mode, visualization_records


def build_evaluation_result(
    args,
    mode_ids,
    metrics_by_mode,
    datasets_metadata,
    teachers_metadata,
    visualizations_by_eye,
    world_size,
):
    if tuple(metrics_by_mode) != tuple(mode_ids):
        raise RuntimeError(
            f"evaluation produced modes {tuple(metrics_by_mode)}, "
            f"expected {tuple(mode_ids)}"
        )
    return {
        "checkpoint": str(args.stereo_vae_ckpt.expanduser().resolve()),
        "posterior": "mean",
        "requested_modes": list(mode_ids),
        "source_frame_index": args.single_frame_source_index,
        "precision": "bf16" if args.bf16 else "fp32",
        "world_size": world_size,
        "modes": metrics_by_mode,
        "datasets": datasets_metadata,
        "teachers": teachers_metadata,
        "visualizations": visualizations_by_eye,
    }


def main():
    args = build_parser().parse_args()
    validate_args(args)
    eye_modes = requested_eye_modes(args)
    temporal_modes = requested_temporal_modes(args)
    mode_ids = requested_mode_ids(args)
    preflight_teacher_assets(args, eye_modes)
    datasets = {
        eye_mode: build_eval_dataset(args, eye_mode) for eye_mode in eye_modes
    }
    if args.num_visualizations:
        for eye_mode, dataset in datasets.items():
            if args.num_visualizations > len(dataset):
                raise ValueError(
                    f"{eye_mode} has only {len(dataset)} samples, cannot save "
                    f"{args.num_visualizations} visualizations"
                )

    device, rank, world_size = initialize_distributed(args)
    model = load_model(args, device)
    if args.num_visualizations:
        if rank == 0:
            args.visualization_dir.mkdir(parents=True, exist_ok=False)
        if world_size > 1:
            dist.barrier()

    metrics_by_mode = {}
    visualizations_by_eye = {}
    for eye_mode in eye_modes:
        teacher = build_online_teacher(args, eye_mode, device)
        eye_metrics, eye_visualizations = evaluate_eye_mode(
            args,
            eye_mode,
            temporal_modes,
            datasets[eye_mode],
            teacher,
            model,
            device,
            rank,
            world_size,
        )
        metrics_by_mode.update(eye_metrics)
        visualizations_by_eye[eye_mode] = eye_visualizations
        del teacher

    result = build_evaluation_result(
        args,
        mode_ids,
        metrics_by_mode,
        {
            eye_mode: dataset_provenance(args, eye_mode, datasets[eye_mode])
            for eye_mode in eye_modes
        },
        {
            eye_mode: teacher_provenance(args, eye_mode)
            for eye_mode in eye_modes
        },
        visualizations_by_eye,
        world_size,
    )
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
