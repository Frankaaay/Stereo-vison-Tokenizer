import argparse
import hashlib
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from stereo_tokenizer import StereoVAE
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
)


CHECKPOINT_SEMANTIC_FIELDS = (
    "resolution",
    "image_channels",
    "norm_type",
    "embedding_dim",
    "latent_channels",
    "patch_size",
    "enc_block",
    "dec_block",
    "twod_window_size",
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


def requested_single_frame_source_indices(args):
    configured = getattr(args, "single_frame_source_indices", None)
    if configured is None:
        return (int(getattr(args, "single_frame_source_index", 0)),)
    return tuple(int(index) for index in configured)


def evaluation_specs(args, eye_mode, temporal_modes=None):
    source_indices = requested_single_frame_source_indices(args)
    multiple_sources = len(source_indices) > 1
    specs = []
    if temporal_modes is None:
        temporal_modes = ("single_frame", "four_frame")
    for temporal_mode in temporal_modes:
        if temporal_mode == "single_frame":
            for source_index in source_indices:
                mode_id = f"{eye_mode}/single_frame"
                if multiple_sources:
                    mode_id += f"/source_{source_index}"
                specs.append((mode_id, temporal_mode, source_index))
        else:
            specs.append((f"{eye_mode}/four_frame", temporal_mode, None))
    return tuple(specs)


def build_parser():
    parser = argparse.ArgumentParser()
    parser = StereoVAE.add_model_specific_args(parser)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=int, choices=(0, 1), default=1)
    parser.add_argument("--persistent_workers", type=int, choices=(0, 1), default=1)
    parser.add_argument("--hy_root_aliases", type=str, default=None)
    parser.add_argument("--stereo_disparity_min_px", type=float, default=None)
    parser.add_argument("--stereo_disparity_max_px", type=float, default=None)
    parser.add_argument("--stereo_lr_error_abs_threshold_px", type=float, default=None)
    parser.add_argument("--stereo_lr_error_relative_threshold", type=float, default=None)
    parser.add_argument("--stereo_vae_ckpt", type=Path, required=True)
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
        "--single_frame_source_indices", type=int, nargs="+", default=None
    )
    parser.add_argument(
        "--eval_eye_mode",
        choices=["mono", "stereo"],
        default="stereo",
    )
    parser.add_argument(
        "--eval_temporal_mode",
        choices=["single_frame", "four_frame", "both"],
        required=True,
    )
    return parser


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
        source = batch["da3_images"]
        batch_size, views, time, channels, height, width = source.shape
        native_depth, native_confidence = teacher.infer_processed(
            source.reshape(batch_size * views, time, channels, height, width)
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
            :, :, source_index : source_index + 1
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
