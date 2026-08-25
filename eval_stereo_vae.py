import argparse
import json
import os
from argparse import Namespace
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.utils import data as torch_data
from torch.utils.data import default_collate
from tqdm import tqdm

from stereo_tokenizer import StereoVAE
from stereo_tokenizer.data import StereoDataModule
from stereo_tokenizer.lerobot_data import fixed_episode_subset_indices
from stereo_tokenizer.modules.stereo_geometry import disparity_to_depth
from stereo_tokenizer.online_gt import FoundationStereoOnlineTeacher


VIEW_NAMES = ("head", "lefthand", "righthand")
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
    "temporal_patch_size",
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
    "single_frame_source_index",
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
    parser.add_argument("--visualization_dir", type=Path, default=None)
    parser.add_argument("--num_visualizations", type=int, default=0)
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


def validate_args(args):
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
    missing_geometry = [name for name, value in geometry.items() if value is None]
    if missing_geometry:
        raise ValueError(
            "evaluation requires " + ", ".join(missing_geometry)
        )
    if not 0 <= args.stereo_disparity_min_px < args.stereo_disparity_max_px:
        raise ValueError("invalid disparity supervision range")
    if args.stereo_lr_error_abs_threshold_px < 0:
        raise ValueError("absolute LR threshold must be non-negative")
    if args.stereo_lr_error_relative_threshold < 0:
        raise ValueError("relative LR threshold must be non-negative")
    if args.stereo_data_backend == "lerobot_online":
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
                "LeRobot online evaluation requires " + ", ".join(missing)
            )
        if len(args.lerobot_rectification_audit_sha256) != 64:
            raise ValueError("a full rectification audit SHA256 is required")
        if len(args.foundation_stereo_checkpoint_sha256) != 64:
            raise ValueError("a full FoundationStereo checkpoint SHA256 is required")
        if args.foundation_stereo_pair_microbatch < 1:
            raise ValueError("FoundationStereo pair microbatch must be positive")
    else:
        if args.eval_split == "test":
            raise ValueError("Manifest-v3 evaluation has no separate test manifest")
        manifest = (
            args.stereo_train_manifest
            if args.eval_split == "train"
            else args.stereo_val_manifest
        )
        if manifest is None:
            raise ValueError(f"no Manifest v3 configured for split {args.eval_split}")
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


def selected_loader(data, split):
    if split == "train":
        return data.train_dataloader()
    loader = data.test_dataloader() if split == "test" else data.val_dataloader()
    if loader is None:
        raise RuntimeError(f"{split} loader is unavailable")
    return loader


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


def exact_eval_loader(data_module, args, rank, world_size):
    if args.stereo_data_backend != "lerobot_online":
        if world_size > 1:
            raise ValueError(
                "distributed exact evaluation is only supported for LeRobot online data"
            )
        return selected_loader(data_module, args.eval_split), None

    dataset = data_module._dataset(False, split=args.eval_split)
    indices = _exact_lerobot_rank_indices(dataset, rank, world_size)
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
    return loader, dataset


def build_online_teacher(args, device):
    if args.stereo_data_backend != "lerobot_online":
        return None
    return FoundationStereoOnlineTeacher(
        args.foundation_stereo_repo,
        args.foundation_stereo_checkpoint,
        args.foundation_stereo_checkpoint_sha256,
        device=device,
        valid_iters=args.foundation_stereo_valid_iters,
        pair_microbatch=args.foundation_stereo_pair_microbatch,
        backend="pytorch",
    )


def attach_online_targets(args, teacher, batch):
    if teacher is None:
        return
    disparity, residual, base_valid = teacher.infer(batch["video"])
    threshold = torch.maximum(
        residual.new_tensor(args.stereo_lr_error_abs_threshold_px),
        args.stereo_lr_error_relative_threshold * disparity,
    )
    batch["disparity"] = disparity
    batch["valid_mask"] = (
        base_valid
        & torch.isfinite(disparity)
        & torch.isfinite(residual)
        & (disparity >= args.stereo_disparity_min_px)
        & (disparity <= args.stereo_disparity_max_px)
        & (residual <= threshold)
    )


def requested_temporal_modes(args):
    if args.eval_temporal_mode == "both":
        return ("four_frame", "single_frame")
    return (args.eval_temporal_mode,)


def batch_for_temporal_mode(batch, mode, source_index):
    if mode == "four_frame":
        return batch
    result = dict(batch)
    for key in ("video", "disparity", "valid_mask"):
        result[key] = batch[key][..., source_index : source_index + 1, :, :]
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


def save_case_visualization(
    output_path,
    sample_id,
    episode_id,
    video,
    outputs,
):
    cell = 256
    label_width = 150
    header_height = 70
    columns = 5
    rows = len(VIEW_NAMES) * 2
    canvas = Image.new(
        "RGB",
        (label_width + columns * cell, header_height + rows * cell),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"sample: {sample_id}", fill="black")
    draw.text((12, 30), f"episode: {episode_id}", fill="black")
    headers = ("t0", "t1", "t2", "t3", "single t0")
    for column, label in enumerate(headers):
        draw.text((label_width + column * cell + 8, 48), label, fill="black")

    four = outputs.get("four_frame")
    single = outputs.get("single_frame")
    for view_index, view_name in enumerate(VIEW_NAMES):
        input_row = view_index * 2
        recon_row = input_row + 1
        draw.text((8, header_height + input_row * cell + 8), f"{view_name} input", fill="black")
        draw.text((8, header_height + recon_row * cell + 8), f"{view_name} recon", fill="black")
        for frame_index in range(4):
            source = _rgb_image(video[0, view_index, 0, :, frame_index])
            canvas.paste(
                source,
                (label_width + frame_index * cell, header_height + input_row * cell),
            )
            if four is not None:
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
        if single is not None:
            reconstruction = _rgb_image(single.rgb[0, view_index, :, 0])
            canvas.paste(
                reconstruction,
                (label_width + 4 * cell, header_height + recon_row * cell),
            )
    canvas.save(output_path)


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

    target_depth = disparity_to_depth(
        disparity_target,
        batch["fx"],
        batch["baseline_m"],
        valid_mask=valid,
    )
    predicted_depth = disparity_to_depth(
        output.disparity,
        batch["fx"],
        batch["baseline_m"],
        valid_mask=valid,
    )
    if not torch.equal(target_depth.valid_mask, valid):
        raise ValueError("valid target disparity failed metric-depth conversion")
    if not torch.equal(predicted_depth.valid_mask, valid):
        raise ValueError("predicted disparity is invalid on supervised pixels")

    depth_error = predicted_depth.depth - target_depth.depth
    safe_target_depth = torch.where(
        valid,
        target_depth.depth,
        torch.ones_like(target_depth.depth),
    )
    accumulator["depth_abs_rel_sum"] += (
        (depth_error.abs() / safe_target_depth)
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
    device, rank, world_size = initialize_distributed(args)
    model = load_model(args, device)
    data_module = StereoDataModule(args, shuffle=False)
    loader, lerobot_dataset = exact_eval_loader(
        data_module, args, rank, world_size
    )
    teacher = build_online_teacher(args, device)
    modes = requested_temporal_modes(args)
    accumulators = {mode: empty_accumulator(device) for mode in modes}

    with torch.inference_mode():
        progress = tqdm(loader, disable=rank != 0)
        for batch_index, batch in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            attach_online_targets(args, teacher, tensor_batch)
            for mode in modes:
                mode_batch = batch_for_temporal_mode(
                    tensor_batch, mode, args.single_frame_source_index
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    output = model(
                        mode_batch["video"],
                        eye_mode=args.eval_eye_mode,
                        temporal_mode=mode,
                        sample_posterior=False,
                    )
                update_metrics(accumulators[mode], mode_batch, output)

    for accumulator in accumulators.values():
        reduce_accumulator(accumulator, world_size)

    metrics_by_mode = {
        mode: finalize_metrics(accumulator)
        for mode, accumulator in accumulators.items()
    }
    if (
        args.stereo_data_backend == "lerobot_online"
        and args.max_batches is None
    ):
        expected = len(lerobot_dataset)
        for mode, metrics in metrics_by_mode.items():
            if metrics["sample_count"] != expected:
                raise RuntimeError(
                    f"{mode} evaluated {metrics['sample_count']} samples, "
                    f"expected exact split size {expected}"
                )

    visualization_records = []
    if args.num_visualizations:
        if lerobot_dataset is None:
            raise ValueError("case visualization currently requires LeRobot online data")
        if rank == 0:
            args.visualization_dir.mkdir(parents=True, exist_ok=False)
        if world_size > 1:
            dist.barrier()
        case_indices = fixed_episode_subset_indices(
            lerobot_dataset,
            args.num_visualizations,
            seed=int(getattr(args, "seed", 1234)),
        )
        local_records = []
        with torch.inference_mode():
            for slot in range(rank, args.num_visualizations, world_size):
                batch = default_collate([lerobot_dataset[case_indices[slot]]])
                tensor_batch = {
                    key: value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }
                attach_online_targets(args, teacher, tensor_batch)
                outputs = {}
                for mode in modes:
                    mode_batch = batch_for_temporal_mode(
                        tensor_batch, mode, args.single_frame_source_index
                    )
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=args.bf16,
                    ):
                        outputs[mode] = model(
                            mode_batch["video"],
                            eye_mode=args.eval_eye_mode,
                            temporal_mode=mode,
                            sample_posterior=False,
                        )
                filename = f"case-{slot:02d}.png"
                save_case_visualization(
                    args.visualization_dir / filename,
                    tensor_batch["sample_id"][0],
                    tensor_batch["episode_id"][0],
                    tensor_batch["video"],
                    outputs,
                )
                local_records.append(
                    {
                        "slot": slot,
                        "sample_id": tensor_batch["sample_id"][0],
                        "episode_id": tensor_batch["episode_id"][0],
                        "file": filename,
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

    result = {
        "checkpoint": str(args.stereo_vae_ckpt.expanduser().resolve()),
        "split": args.eval_split,
        "posterior": "mean",
        "eye_mode": args.eval_eye_mode,
        "temporal_mode": args.eval_temporal_mode,
        "source_frame_index": args.single_frame_source_index,
        "precision": "bf16" if args.bf16 else "fp32",
        "world_size": world_size,
        "modes": metrics_by_mode,
        "visualizations": visualization_records,
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.num_visualizations:
            (args.visualization_dir / "cases.json").write_text(
                json.dumps(visualization_records, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
