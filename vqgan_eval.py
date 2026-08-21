import argparse
import json
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path

import torch
from tqdm import tqdm

from OmniTokenizer import OmniTokenizer_VQGAN, VQGAN, VideoData


VIEW_NAMES = ("head", "lefthand", "righthand")
CHECKPOINT_SEMANTIC_FIELDS = (
    "resolution",
    "image_channels",
    "norm_type",
    "embedding_dim",
    "codebook_dim",
    "patch_size",
    "patch_embed",
    "enc_block",
    "dec_block",
    "twod_window_size",
    "temporal_patch_size",
    "defer_temporal_pool",
    "defer_spatial_pool",
    "spatial_pos",
    "spatial_depth",
    "temporal_depth",
    "causal_in_temporal_transformer",
    "causal_in_peg",
    "dim_head",
    "heads",
    "attn_dropout",
    "ff_dropout",
    "ff_mult",
    "stereo_num_views",
    "stereo_num_frames",
    "stereo_search_radii",
    "stereo_search_direction",
    "stereo_disparity_scale",
    "stereo_disparity_bias",
    "stereo_disparity_epsilon",
    "stereo_mode",
    "stereo_disparity_min_px",
    "stereo_disparity_max_px",
    "stereo_lr_error_abs_threshold_px",
    "stereo_lr_error_relative_threshold",
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser = VQGAN.add_model_specific_args(parser)
    parser = OmniTokenizer_VQGAN.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    parser.add_argument("--vqgan_ckpt", type=Path, required=True)
    parser.add_argument("--eval_split", choices=["train", "val"], default="val")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser


def validate_args(args):
    if args.loader_type != "stereo_manifest":
        raise ValueError("Stereo evaluation requires loader_type=stereo_manifest")
    manifest = (
        args.stereo_train_manifest
        if args.eval_split == "train"
        else args.stereo_val_manifest
    )
    if manifest is None:
        raise ValueError(f"no Manifest v3 configured for split {args.eval_split}")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max_batches must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")


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
        args.vqgan_ckpt, map_location="cpu", weights_only=False
    )
    if "state_dict" not in checkpoint:
        raise ValueError(f"{args.vqgan_ckpt}: missing state_dict")
    checkpoint_args = _checkpoint_model_args(checkpoint, args.vqgan_ckpt)
    _validate_checkpoint_semantics(args, checkpoint_args, args.vqgan_ckpt)
    model = OmniTokenizer_VQGAN(checkpoint_args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model


def selected_loader(data, split):
    if split == "train":
        loaders = data.train_dataloader()
        if len(loaders) != 1:
            raise RuntimeError("Stereo evaluation expects one training loader")
        return loaders[0]
    loader = data.val_dataloader()
    if loader is None:
        raise RuntimeError("validation loader is unavailable")
    return loader


def empty_accumulator(device):
    return {
        "sample_count": torch.zeros((), dtype=torch.long, device=device),
        "rgb_abs_sum": torch.zeros((), dtype=torch.float64, device=device),
        "rgb_count": torch.zeros((), dtype=torch.long, device=device),
        "disp_abs_sum": torch.zeros(3, dtype=torch.float64, device=device),
        "valid_count": torch.zeros(3, dtype=torch.long, device=device),
        "depth_abs_rel_sum": torch.zeros(3, dtype=torch.float64, device=device),
        "depth_sq_sum": torch.zeros(3, dtype=torch.float64, device=device),
    }


def update_metrics(accumulator, batch, output):
    rgb_target = batch["video"][:, :, 0]
    rgb_error = (output.rgb - rgb_target).abs()
    accumulator["sample_count"] += rgb_target.shape[0]
    accumulator["rgb_abs_sum"] += rgb_error.double().sum()
    accumulator["rgb_count"] += rgb_error.numel()

    disparity_target = batch["disparity"]
    valid = batch["valid_mask"]
    disparity_error = (output.disparity - disparity_target).abs()
    reduction_dims = (0, 2, 3, 4, 5)
    accumulator["disp_abs_sum"] += (
        disparity_error.double().masked_fill(~valid, 0).sum(dim=reduction_dims)
    )
    accumulator["valid_count"] += valid.sum(dim=reduction_dims)

    calibration = (batch["fx"] * batch["baseline_m"]).reshape(
        batch["fx"].shape[0], 3, 1, 1, 1, 1
    )
    target_depth = calibration / disparity_target
    predicted_depth = calibration / output.disparity
    depth_error = predicted_depth - target_depth
    accumulator["depth_abs_rel_sum"] += (
        (depth_error.abs() / target_depth)
        .double()
        .masked_fill(~valid, 0)
        .sum(dim=reduction_dims)
    )
    accumulator["depth_sq_sum"] += (
        depth_error.square()
        .double()
        .masked_fill(~valid, 0)
        .sum(dim=reduction_dims)
    )


def finalize_metrics(accumulator):
    if accumulator["sample_count"].item() == 0:
        raise ValueError("evaluation loader produced no samples")
    if torch.any(accumulator["valid_count"] == 0):
        raise ValueError("at least one view has no valid disparity pixels")
    valid_count = accumulator["valid_count"].double()
    result = {
        "sample_count": int(accumulator["sample_count"].item()),
        "rgb_l1": float(
            (accumulator["rgb_abs_sum"] / accumulator["rgb_count"]).item()
        ),
        "views": {},
    }
    for view_index, view_name in enumerate(VIEW_NAMES):
        result["views"][view_name] = {
            "valid_pixels": int(accumulator["valid_count"][view_index].item()),
            "disparity_epe_px": float(
                (accumulator["disp_abs_sum"][view_index] / valid_count[view_index]).item()
            ),
            "depth_abs_rel": float(
                (
                    accumulator["depth_abs_rel_sum"][view_index]
                    / valid_count[view_index]
                ).item()
            ),
            "depth_rmse_m": float(
                torch.sqrt(
                    accumulator["depth_sq_sum"][view_index]
                    / valid_count[view_index]
                ).item()
            ),
        }
    return result


def main():
    args = build_parser().parse_args()
    validate_args(args)
    device = torch.device(args.device)
    model = load_model(args, device)
    loader = selected_loader(VideoData(args, shuffle=False), args.eval_split)
    accumulator = empty_accumulator(device)

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader)):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            output = model(
                tensor_batch["video"],
                sample_posterior=False,
            )
            update_metrics(accumulator, tensor_batch, output)

    result = {
        "checkpoint": str(args.vqgan_ckpt.expanduser().resolve()),
        "split": args.eval_split,
        "posterior": "mean",
        "metrics": finalize_metrics(accumulator),
    }
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
