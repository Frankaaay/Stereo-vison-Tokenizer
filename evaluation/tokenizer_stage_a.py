"""CLI for the frozen Stereo Tokenizer Stage A1 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate
from tqdm import tqdm

from . import stage_a_runtime as runtime

from .stage_a_contract import sha256_file
from .stage_a_data import CanonicalStageADataset, build_canonical_selection
from .stage_a_metrics import (
    DYNAMIC_FLOW_MIN_PX,
    FLOW_FB_ABSOLUTE_THRESHOLD_PX2,
    FLOW_FB_RELATIVE_THRESHOLD,
    STATIC_FLOW_MAX_PX,
    StageA1MetricSuite,
)
from stereo_tokenizer.geometry import GeometryMapping


DA3_SOURCE_SHA = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_CHECKPOINT_SHA256 = (
    "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5"
)
LAS2_H_SOURCE_SHA = "8c97bd4c4da3712c2ac60003a23201dfdb5935f4"
LAS2_H_CHECKPOINT_SHA256 = (
    "758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4"
)
VGG16_CHECKPOINT_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
VGG16_CHECKPOINT_NAME = "vgg16-397923af.pth"


class _FrozenRAFT:
    """Strict local-checkpoint torchvision RAFT-Large inference wrapper."""

    def __init__(
        self,
        checkpoint: Path,
        expected_sha256: str,
        *,
        device: torch.device,
        microbatch: int,
    ) -> None:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.checkpoint = checkpoint.expanduser().resolve()
        self.sha256 = sha256_file(self.checkpoint)
        if self.sha256 != expected_sha256:
            raise ValueError(
                "RAFT checkpoint SHA mismatch: "
                f"requested={expected_sha256}, actual={self.sha256}"
            )
        if microbatch < 1:
            raise ValueError("RAFT microbatch must be positive")
        self.microbatch = int(microbatch)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("RAFT checkpoint must contain a state dictionary")
        self.model = raft_large(weights=None, progress=False)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"RAFT checkpoint structure mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        self.model.to(device).eval().requires_grad_(False)
        self.transforms = Raft_Large_Weights.C_T_SKHT_V2.transforms()

    @torch.inference_mode()
    def __call__(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape or first.ndim != 4 or first.shape[1] != 3:
            raise ValueError("RAFT inputs must be matching [N,3,H,W] tensors")
        if first.shape[-2] % 8 or first.shape[-1] % 8:
            raise ValueError("RAFT input height and width must be divisible by 8")
        outputs = []
        for start in range(0, first.shape[0], self.microbatch):
            end = min(first.shape[0], start + self.microbatch)
            first_batch, second_batch = self.transforms(
                first[start:end].float(), second[start:end].float()
            )
            predictions = self.model(first_batch, second_batch)
            if not isinstance(predictions, list) or not predictions:
                raise RuntimeError("RAFT did not return iterative flow predictions")
            outputs.append(predictions[-1].float())
        return torch.cat(outputs, dim=0)

    def provenance(self) -> dict[str, object]:
        return {
            "name": "torchvision.raft_large",
            "architecture": "RAFT-Large",
            "transform_contract": "Raft_Large_Weights.C_T_SKHT_V2.transforms",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.sha256,
            "microbatch": self.microbatch,
            "precision": "fp32",
            "flow_unit": "content-crop pixels",
            "static_flow_max_px": STATIC_FLOW_MAX_PX,
            "dynamic_flow_min_px": DYNAMIC_FLOW_MIN_PX,
            "forward_backward_relative_threshold": FLOW_FB_RELATIVE_THRESHOLD,
            "forward_backward_absolute_threshold_px2": (
                FLOW_FB_ABSOLUTE_THRESHOLD_PX2
            ),
        }


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), text=True, stderr=subprocess.STDOUT
    ).strip()


def _source_provenance() -> dict:
    diff = subprocess.check_output(("git", "diff", "--binary", "HEAD"))
    return {
        "cwd": str(Path.cwd()),
        "git_branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _metric_backbone_provenance() -> dict:
    torch_home = os.environ.get("TORCH_HOME")
    if not torch_home:
        raise ValueError("TORCH_HOME is required for the frozen LPIPS backbone")
    checkpoint = (
        Path(torch_home).expanduser().resolve() / "hub" / "checkpoints"
        / VGG16_CHECKPOINT_NAME
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen LPIPS backbone is missing: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != VGG16_CHECKPOINT_SHA256:
        raise ValueError(
            f"LPIPS VGG16 SHA mismatch: expected={VGG16_CHECKPOINT_SHA256}, "
            f"actual={actual}"
        )
    return {
        "name": "torchvision.vgg16.IMAGENET1K_V1",
        "role": "torchmetrics LPIPS VGG feature backbone",
        "path": str(checkpoint),
        "sha256": actual,
        "preprocessing": "torchmetrics LPIPS vgg normalize=False on RGB [-1,1]",
    }


def _environment_provenance() -> dict:
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "torchmetrics",
        "pytorch-lightning",
        "numpy",
        "av",
        "pylance",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "uv_lock_sha256": sha256_file(Path("uv.lock").resolve()),
        "metric_backbone": _metric_backbone_provenance(),
    }


def _checkpoint_provenance(path: Path, expected_sha256: str) -> dict:
    checkpoint_path = path.expanduser().resolve()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("checkpoint SHA256 must contain exactly 64 hexadecimal characters")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("checkpoint SHA256 must be hexadecimal") from error
    actual = sha256_file(checkpoint_path)
    if actual != expected_sha256:
        raise ValueError(
            f"checkpoint SHA mismatch: requested={expected_sha256}, actual={actual}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    counters = checkpoint.get("stereo_update_counters")
    if not isinstance(counters, dict):
        raise ValueError("checkpoint is missing stereo_update_counters")
    required_counters = (
        "generator_updates",
        "discriminator_updates",
        "batch_updates",
        "four_frame_updates",
        "single_frame_updates",
    )
    invalid_counters = {
        key: counters.get(key)
        for key in required_counters
        if not isinstance(counters.get(key), int) or counters[key] < 0
    }
    if invalid_counters:
        raise ValueError(f"checkpoint has invalid training counters: {invalid_counters}")
    if not isinstance(checkpoint.get("global_step"), int) or checkpoint["global_step"] < 0:
        raise ValueError("checkpoint has invalid global_step")
    if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] < 0:
        raise ValueError("checkpoint has invalid epoch")
    return {
        "path": str(checkpoint_path),
        "sha256": actual,
        "global_step": int(checkpoint["global_step"]),
        "epoch": int(checkpoint["epoch"]),
        "stereo_update_counters": _jsonable(counters),
    }


def _dataset_provenance(dataset) -> dict:
    result = dataset.provenance()
    result["selection_file_sha256"] = sha256_file(dataset.selection_path)
    if result.get("data_backend") == "hy_lance_manifest":
        return result
    config_hashes = {}
    for record in dataset.selection["records"]:
        path = str(record["canonical_config"])
        digest = str(record["canonical_config_sha256"])
        previous = config_hashes.setdefault(path, digest)
        if previous != digest:
            raise ValueError(f"conflicting canonical config hashes for {path}")
    result["canonical_config_sha256"] = dict(sorted(config_hashes.items()))
    return result


def _selection_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tokenizer_stage_a selection")
    parser.add_argument("--dataset-id", choices=("umi", "hy", "libero"), required=True)
    parser.add_argument("--identity-contract", type=Path, required=True)
    parser.add_argument("--canonical-config-root", type=Path)
    parser.add_argument("--canonical-loader-root", type=Path)
    parser.add_argument("--umi-publish-ledger", type=Path)
    parser.add_argument("--hy-manifest", type=Path)
    parser.add_argument("--hy-manifest-sha256")
    parser.add_argument("--hy-root-aliases")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    selection = build_canonical_selection(
        dataset_id=args.dataset_id,
        identity_contract_path=args.identity_contract,
        canonical_config_root=args.canonical_config_root,
        loader_root=args.canonical_loader_root,
        split=args.split,
        sample_count=args.sample_count,
        seed=args.seed,
        output=args.output,
        umi_publish_ledger=args.umi_publish_ledger,
        hy_manifest_path=args.hy_manifest,
        hy_manifest_sha256=args.hy_manifest_sha256,
        hy_root_aliases=args.hy_root_aliases,
    )
    print(
        json.dumps(
            {key: value for key, value in selection.items() if key != "records"},
            indent=2,
            sort_keys=True,
        )
    )


def _preflight_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tokenizer_stage_a preflight")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--canonical-loader-root", type=Path)
    parser.add_argument("--hy-root-aliases")
    parser.add_argument("--eye-mode", choices=("mono", "stereo"), required=True)
    parser.add_argument("--camera-key")
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv)
    dataset = CanonicalStageADataset(
        args.selection,
        loader_root=args.canonical_loader_root,
        eye_mode=args.eye_mode,
        camera_key=args.camera_key,
        hy_root_aliases=args.hy_root_aliases,
    )
    if args.samples < 1 or args.samples > len(dataset):
        raise ValueError("invalid preflight sample count")
    rows = []
    for index in range(args.samples):
        sample = dataset[index]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "video_shape": list(sample["video"].shape),
                "video_dtype": str(sample["video"].dtype),
                "video_min": float(sample["video"].min()),
                "video_max": float(sample["video"].max()),
                "valid_rgb_values": int(sample["rgb_valid_mask"].sum()),
                "source_frame_indices": sample["frame_index"].tolist(),
            }
        )
    print(
        json.dumps(
            {"dataset": _dataset_provenance(dataset), "samples": rows},
            indent=2,
            sort_keys=True,
        )
    )


def _run_parser() -> argparse.ArgumentParser:
    parser = runtime.build_parser()
    parser.prog = "tokenizer_stage_a run"
    parser.add_argument("--stage-a-dataset-id", choices=("umi", "hy", "libero"), required=True)
    parser.add_argument("--stage-a-selection", type=Path, required=True)
    parser.add_argument("--canonical-loader-root", type=Path, required=True)
    parser.add_argument("--stage-a-camera-key")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--raft-checkpoint", type=Path, required=True)
    parser.add_argument("--raft-checkpoint-sha256", required=True)
    parser.add_argument("--raft-microbatch", type=int, default=3)
    parser.add_argument("--rgb-only", action="store_true")
    runtime_required = {
        "stereo_vae_ckpt",
        "output_json",
        "eval_temporal_mode",
        "stage_a_dataset_id",
        "stage_a_selection",
    }
    for action in parser._actions:
        if action.required and action.dest not in runtime_required:
            action.required = False
    return parser


def _hydrate_checkpoint_semantics(args) -> None:
    checkpoint = torch.load(
        args.stereo_vae_ckpt, map_location="cpu", weights_only=False
    )
    checkpoint_args = runtime._checkpoint_model_args(
        checkpoint, args.stereo_vae_ckpt
    )
    for name in runtime.CHECKPOINT_SEMANTIC_FIELDS:
        setattr(args, name, getattr(checkpoint_args, name))
    if args.single_frame_source_index is None:
        args.single_frame_source_index = int(
            getattr(checkpoint_args, "single_frame_source_index")
        )


def _validate_run(args) -> None:
    if args.bf16:
        raise ValueError("Stage A quality metrics are frozen to FP32")
    if args.eval_eye_mode not in {"mono", "stereo"}:
        raise ValueError("one Stage A invocation evaluates one eye mode")
    if args.eval_temporal_mode != "both":
        raise ValueError("Stage A requires single_frame and four_frame together")
    source_indices = runtime.requested_single_frame_source_indices(args)
    if source_indices != (0, 1, 2, 3):
        raise ValueError("Stage A requires --single_frame_source_indices 0 1 2 3")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max_batches must be positive")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("sample percentiles currently require one H100 process")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage A run requires one allocated CUDA GPU")
    if args.eval_eye_mode == "mono" and not args.stage_a_camera_key:
        if args.stage_a_dataset_id != "hy":
            raise ValueError("non-Hy mono Stage A requires --stage-a-camera-key")
    if args.stage_a_dataset_id == "hy":
        if args.eval_eye_mode != "mono" or args.stage_a_camera_key:
            raise ValueError("Hy Stage A evaluates its three mono views together")
        if not args.hy_root_aliases:
            raise ValueError("Hy Stage A requires --hy-root-aliases")
    elif args.canonical_loader_root is None:
        raise ValueError("canonical Stage A requires --canonical-loader-root")
    if args.eval_eye_mode == "stereo" and args.stage_a_camera_key:
        raise ValueError("stereo Stage A does not accept --stage-a-camera-key")
    if args.num_visualizations < 0:
        raise ValueError("--num_visualizations must be non-negative")
    if args.num_visualizations and args.visualization_dir is None:
        raise ValueError("visualizations require --visualization_dir")
    if args.visualization_dir is not None and args.visualization_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite visualization directory {args.visualization_dir}"
        )
    if args.raft_microbatch < 1:
        raise ValueError("--raft-microbatch must be positive")
    if not args.raft_checkpoint.is_file():
        raise FileNotFoundError(args.raft_checkpoint)
    if len(args.raft_checkpoint_sha256) != 64:
        raise ValueError("--raft-checkpoint-sha256 must contain 64 characters")


def _mode_batch(batch: dict, temporal_mode: str, source_index: int | None):
    if temporal_mode == "four_frame":
        return batch
    result = runtime.batch_for_temporal_mode(batch, temporal_mode, source_index)
    result["rgb_valid_mask"] = batch["rgb_valid_mask"][
        ..., source_index : source_index + 1, :, :
    ]
    return result


def _attach_mono_reconstruction_teacher(args, teacher, batch, output) -> None:
    """Run DA3 on reconstructed RGB with the exact frozen geometry mapping."""

    prediction = output.rgb
    if (
        prediction.ndim != 6
        or not 1 <= prediction.shape[1] <= 3
        or prediction.shape[2] != 3
    ):
        raise ValueError("mono reconstruction must be [B,V,3,T,H,W] with V in [1,3]")
    batch_size, views, _, frames, height, width = prediction.shape
    geometry = GeometryMapping.from_collated(
        batch["geometry_mapping"], int(batch_size)
    )
    left, top, right, bottom = geometry.student_padding_ltrb
    flattened = prediction.permute(0, 1, 3, 2, 4, 5).reshape(
        batch_size * views * frames, 3, height, width
    )
    content = flattened[
        :,
        :,
        top : height - bottom,
        left : width - right,
    ]
    rectified = F.interpolate(
        content.float(),
        size=geometry.rectified_hw,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    rectified_u8 = (
        rectified.clamp(-0.5, 0.5)
        .add(0.5)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    processed = geometry.da3_preprocess(rectified_u8).reshape(
        batch_size * views, frames, 3, *geometry.da3_processed_hw
    )
    native_depth, native_confidence = teacher.infer_processed(processed)
    depth = geometry.map_da3_output_to_student(native_depth).reshape(
        batch_size, views, 1, frames, *geometry.student_output_hw
    )
    confidence = geometry.map_da3_output_to_student(native_confidence).reshape(
        batch_size, views, 1, frames, *geometry.student_output_hw
    )
    non_padding = batch["non_padding_mask"]
    valid = torch.isfinite(depth) & (depth > 0) & non_padding
    if not torch.isfinite(confidence.masked_select(non_padding)).all():
        raise ValueError("reconstruction DA3 confidence contains NaN/Inf")
    if torch.any(valid.sum(dim=(2, 3, 4, 5)) == 0):
        raise ValueError("reconstruction DA3 produced an empty valid mask")
    batch["reconstruction_da3_relative_depth"] = depth
    batch["reconstruction_da3_confidence"] = confidence
    batch["reconstruction_valid_mask"] = valid


def _fixed_visualization_indices(dataset, count: int) -> list[int]:
    ranked = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(
            (
                "1234:stage-a1-visualization:"
                f"{dataset.dataset_id}:"
                f"{dataset.selection['records'][index]['legacy_episode_id']}"
            ).encode("utf-8")
        ).digest(),
    )
    return ranked[:count]


def _stage_a_visualization_batch(batch):
    """Supply unit stereo calibration for teacher-relative visualizations."""

    if "disparity" not in batch:
        return batch
    has_fx = "fx" in batch
    has_baseline = "baseline_m" in batch
    if has_fx != has_baseline:
        raise ValueError("stereo visualization calibration must provide both fx and baseline_m")
    if has_fx:
        return batch
    disparity = batch["disparity"]
    if not isinstance(disparity, torch.Tensor) or disparity.ndim != 6:
        raise ValueError("stereo visualization disparity must use [B,V,1,T,H,W]")
    result = dict(batch)
    unit_calibration = torch.ones(
        disparity.shape[:2], device=disparity.device, dtype=torch.float32
    )
    result["fx"] = unit_calibration
    result["baseline_m"] = unit_calibration
    return result


def _save_visualizations(args, dataset, teacher, model, specs, device):
    if not args.num_visualizations:
        return []
    if teacher is None:
        raise ValueError("Stage A visualizations require the frozen teacher")
    if args.num_visualizations > len(dataset):
        raise ValueError("visualization count exceeds dataset size")
    args.visualization_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with torch.inference_mode():
        for slot, index in enumerate(
            _fixed_visualization_indices(dataset, args.num_visualizations)
        ):
            batch = default_collate([dataset[index]])
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            runtime.attach_online_targets(args, args.eval_eye_mode, teacher, batch)
            batch["valid_mask"] &= batch["rgb_valid_mask"]
            visualization_batch = _stage_a_visualization_batch(batch)
            outputs = {}
            for mode_id, temporal_mode, source_index in specs:
                mode_batch = _mode_batch(batch, temporal_mode, source_index)
                outputs[mode_id] = model(
                    mode_batch["video"],
                    eye_mode=args.eval_eye_mode,
                    temporal_mode=temporal_mode,
                    sample_posterior=False,
                )
            for source_index in (0, 1, 2, 3):
                single_key = (
                    f"{args.eval_eye_mode}/single_frame/source_{source_index}"
                )
                display = {
                    "single_frame": outputs[single_key],
                    "four_frame": outputs[f"{args.eval_eye_mode}/four_frame"],
                }
                rgb_name = f"case-{slot:02d}-source-{source_index}.png"
                depth_name = f"depth-case-{slot:02d}-source-{source_index}.png"
                runtime.save_case_visualization(
                    args.visualization_dir / rgb_name,
                    batch["sample_id"][0],
                    batch["episode_id"][0],
                    batch["video"],
                    display,
                    dataset.view_names,
                    source_index,
                )
                runtime.save_depth_case_visualization(
                    args.visualization_dir / depth_name,
                    visualization_batch,
                    display,
                    source_index,
                    args.relative_depth_epsilon,
                    dataset.view_names,
                )
                records.append(
                    {
                        "slot": slot,
                        "selection_index": index,
                        "sample_id": batch["sample_id"][0],
                        "source_frame_index": source_index,
                        "rgb_file": rgb_name,
                        "geometry_file": depth_name,
                    }
                )
    (args.visualization_dir / "cases.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records


def _run_command(argv: list[str]) -> None:
    args = _run_parser().parse_args(argv)
    _hydrate_checkpoint_semantics(args)
    _validate_run(args)
    environment = _environment_provenance()
    checkpoint = _checkpoint_provenance(
        args.stereo_vae_ckpt, args.checkpoint_sha256
    )
    dataset = CanonicalStageADataset(
        args.stage_a_selection,
        loader_root=args.canonical_loader_root,
        eye_mode=args.eval_eye_mode,
        camera_key=args.stage_a_camera_key,
        hy_root_aliases=args.hy_root_aliases,
    )
    if dataset.dataset_id != args.stage_a_dataset_id:
        raise ValueError("selection dataset ID disagrees with CLI")
    loader = DataLoader(
        Subset(dataset, list(range(len(dataset)))),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers) and args.num_workers > 0,
        collate_fn=default_collate,
        shuffle=False,
        drop_last=False,
    )
    device = torch.device("cuda")
    model = runtime.load_model(args, device)
    architecturally_trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    model.requires_grad_(False)
    if model.perceptual_model is None:
        raise RuntimeError("checkpoint LPIPS model is unavailable")
    flow_model = _FrozenRAFT(
        args.raft_checkpoint,
        args.raft_checkpoint_sha256,
        device=device,
        microbatch=args.raft_microbatch,
    )
    teacher = None
    if not args.rgb_only:
        runtime.preflight_teacher_assets(args, (args.eval_eye_mode,))
        teacher = runtime.build_online_teacher(args, args.eval_eye_mode, device)
    suite = StageA1MetricSuite(
        relative_depth_epsilon=args.relative_depth_epsilon
    )
    specs = runtime.evaluation_specs(args, args.eval_eye_mode)
    current_sample_ids = []
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(tqdm(loader, desc=args.eval_eye_mode)):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                current_sample_ids = [str(value) for value in batch["sample_id"]]
                tensor_batch = {
                    key: value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }
                if teacher is not None:
                    runtime.attach_online_targets(
                        args, args.eval_eye_mode, teacher, tensor_batch
                    )
                    tensor_batch["valid_mask"] &= tensor_batch["rgb_valid_mask"]
                for mode_id, temporal_mode, source_index in specs:
                    mode_batch = _mode_batch(
                        tensor_batch, temporal_mode, source_index
                    )
                    output = model(
                        mode_batch["video"],
                        eye_mode=args.eval_eye_mode,
                        temporal_mode=temporal_mode,
                        sample_posterior=False,
                    )
                    if teacher is not None and args.eval_eye_mode == "mono":
                        _attach_mono_reconstruction_teacher(
                            args, teacher, mode_batch, output
                        )
                    suite.update(
                        mode_id,
                        mode_batch,
                        output,
                        dataset.view_names,
                        model.perceptual_model,
                        flow_model,
                    )
    except Exception as error:
        failure_path = args.output_json.with_name(
            f"{args.output_json.stem}.failure.json"
        )
        if not failure_path.exists():
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(
                    {
                        "schema": "stereo-tokenizer-stage-a1-failure-v1",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "sample_ids": current_sample_ids,
                        "checkpoint": checkpoint,
                        "dataset": _dataset_provenance(dataset),
                        "provenance": _source_provenance(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    metrics = {
        mode_id: suite.finalize(mode_id, dataset.view_names)
        for mode_id, _, _ in specs
    }
    if args.max_batches is None:
        for mode_id, values in metrics.items():
            if values["sample_count"] != len(dataset):
                raise RuntimeError(
                    f"{mode_id}: evaluated {values['sample_count']}, expected {len(dataset)}"
                )
    visualizations = _save_visualizations(
        args, dataset, teacher, model, specs, device
    )
    result = {
        "schema": "stereo-tokenizer-stage-a1-result-v3",
        "status": "smoke" if args.max_batches is not None else "formal",
        "posterior": "mean",
        "quality_precision": "fp32",
        "checkpoint": checkpoint,
        "dataset": _dataset_provenance(dataset),
        "teacher": (
            None
            if teacher is None
            else runtime.teacher_provenance(args, args.eval_eye_mode)
        ),
        "flow_teacher": flow_model.provenance(),
        "requested_modes": [mode_id for mode_id, _, _ in specs],
        "single_frame_source_indices": [0, 1, 2, 3],
        "metrics": metrics,
        "tokenizer_parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "architecturally_trainable": architecturally_trainable,
            "runtime_requires_grad": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "evaluation_state": "eval_inference_mode_posterior_mean",
        },
        "visualizations": visualizations,
        "not_applicable": {
            "rfvd": "native four-frame clips are unsupported by the frozen I3D implementation",
            "fvmd": "not validated for native four-frame clips",
        },
        "pending_stage_a2": ["rfid"],
        "provenance": {
            **_source_provenance(),
            "environment": environment,
            "resolved_args": _jsonable(vars(args)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _percentile_summary(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("benchmark timings must be finite and non-empty")
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p90_ms": float(np.quantile(values, 0.90)),
    }


def _cuda_benchmark(function, *, warmup: int, iterations: int, repeats: int):
    all_times = []
    allocated = []
    reserved = []
    for _ in range(repeats):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            function()
            end.record()
        torch.cuda.synchronize()
        all_times.extend(start.elapsed_time(end) for start, end in zip(starts, ends))
        allocated.append(torch.cuda.max_memory_allocated())
        reserved.append(torch.cuda.max_memory_reserved())
    summary = _percentile_summary(all_times)
    summary["peak_allocated_bytes"] = max(allocated)
    summary["peak_reserved_bytes"] = max(reserved)
    return summary


def _benchmark_command(argv: list[str]) -> None:
    parser = _run_parser()
    parser.prog = "tokenizer_stage_a benchmark"
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--allow-nonformal-benchmark", action="store_true")
    args = parser.parse_args(argv)
    _hydrate_checkpoint_semantics(args)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage A benchmark requires one allocated CUDA GPU")
    if args.batch_size != 1 or not args.bf16:
        raise ValueError("formal benchmark requires --batch_size 1 --bf16")
    configured = (
        args.benchmark_warmup,
        args.benchmark_iterations,
        args.benchmark_repeats,
    )
    if not args.allow_nonformal_benchmark and configured != (20, 100, 3):
        raise ValueError("formal benchmark is frozen to warmup=20, iterations=100, repeats=3")
    if min(configured) < 1:
        raise ValueError("benchmark counts must be positive")
    environment = _environment_provenance()
    if args.eval_temporal_mode != "both":
        raise ValueError("benchmark requires both temporal modes")
    dataset = CanonicalStageADataset(
        args.stage_a_selection,
        loader_root=args.canonical_loader_root,
        eye_mode=args.eval_eye_mode,
        camera_key=args.stage_a_camera_key,
    )
    batch = default_collate([dataset[0]])
    device = torch.device("cuda")
    video = batch["video"].to(device)
    model = runtime.load_model(args, device)
    model.requires_grad_(False)
    modes = {}
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for temporal_mode, source_index in (("single_frame", 0), ("four_frame", None)):
            mode_batch = _mode_batch(
                {**batch, "video": video}, temporal_mode, source_index
            )
            mode_video = mode_batch["video"]
            encoded = model.encode(
                mode_video,
                eye_mode=args.eval_eye_mode,
                temporal_mode=temporal_mode,
                sample_posterior=False,
            )
            mode = {}
            mode["encode_including_posterior_mean"] = _cuda_benchmark(
                lambda: model.encode(
                    mode_video,
                    eye_mode=args.eval_eye_mode,
                    temporal_mode=temporal_mode,
                    sample_posterior=False,
                ),
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                repeats=args.benchmark_repeats,
            )
            mode["cached_posterior_mean"] = _cuda_benchmark(
                encoded.posterior.mode,
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                repeats=args.benchmark_repeats,
            )
            mode["decode"] = _cuda_benchmark(
                lambda: model.decode(
                    encoded.latent, temporal_mode=temporal_mode
                ),
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                repeats=args.benchmark_repeats,
            )
            mode["end_to_end"] = _cuda_benchmark(
                lambda: model(
                    mode_video,
                    eye_mode=args.eval_eye_mode,
                    temporal_mode=temporal_mode,
                    sample_posterior=False,
                ),
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                repeats=args.benchmark_repeats,
            )
            end_to_end_p50 = mode["end_to_end"]["p50_ms"]
            mode["throughput"] = {
                "samples_per_second": 1000.0 / end_to_end_p50,
                "frames_per_second": (
                    1000.0 * mode_video.shape[-3] / end_to_end_p50
                ),
            }
            mode["input_shape"] = list(mode_video.shape)
            mode["input_dtype"] = str(mode_video.dtype)
            mode["autocast_dtype"] = "torch.bfloat16"
            modes[temporal_mode] = mode
    result = {
        "schema": "stereo-tokenizer-stage-a1-benchmark-v1",
        "status": "smoke" if args.allow_nonformal_benchmark else "formal",
        "checkpoint": _checkpoint_provenance(
            args.stereo_vae_ckpt, args.checkpoint_sha256
        ),
        "dataset": _dataset_provenance(dataset),
        "precision": "bf16",
        "batch_size": 1,
        "warmup": args.benchmark_warmup,
        "iterations": args.benchmark_iterations,
        "repeats": args.benchmark_repeats,
        "posterior": "mean",
        "timing_scope": "model_only_excludes_data_decode_and_teacher",
        "modes": modes,
        "provenance": {
            **_source_provenance(),
            "environment": environment,
            "resolved_args": _jsonable(vars(args)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _report_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tokenizer_stage_a report")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.artifact_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    quality_paths = sorted((root / "quality").glob("*.json"))
    benchmark_paths = sorted((root / "benchmark").glob("*.json"))
    if len(quality_paths) != 10 or len(benchmark_paths) != 2:
        raise ValueError("Stage A report requires exactly 10 quality and 2 benchmark JSON files")
    quality = [json.loads(path.read_text()) for path in quality_paths]
    benchmarks = [json.loads(path.read_text()) for path in benchmark_paths]

    def require_sha256(value, label):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{label} must be one SHA256 digest")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{label} must be hexadecimal") from error

    def require_metric_backbone(environment, label):
        backbone = environment.get("metric_backbone")
        if not isinstance(backbone, dict):
            raise ValueError(f"{label} metric backbone provenance is missing")
        if (
            backbone.get("name") != "torchvision.vgg16.IMAGENET1K_V1"
            or backbone.get("role") != "torchmetrics LPIPS VGG feature backbone"
            or backbone.get("preprocessing")
            != "torchmetrics LPIPS vgg normalize=False on RGB [-1,1]"
        ):
            raise ValueError(f"{label} metric backbone contract mismatch")
        require_sha256(backbone.get("sha256"), f"{label} VGG16 hash")
        if backbone["sha256"] != VGG16_CHECKPOINT_SHA256:
            raise ValueError(f"{label} VGG16 hash does not match the frozen contract")
        if not str(backbone.get("path", "")):
            raise ValueError(f"{label} VGG16 path is missing")
        return backbone

    def require_summary(container, name, expected_count, label):
        summary = container.get(name)
        if not isinstance(summary, dict):
            raise ValueError(f"{label} is missing metric {name}")
        if int(summary.get("count", -1)) != expected_count:
            raise ValueError(f"{label}/{name} sample count mismatch")
        for field in ("mean", "p50", "p90", "p99"):
            value = summary.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{label}/{name}/{field} must be finite")
        if not summary["p50"] <= summary["p90"] <= summary["p99"]:
            raise ValueError(f"{label}/{name} percentiles are not monotonic")
        return summary

    def require_nonempty_summary(container, name, maximum_count, label):
        summary = container.get(name)
        if not isinstance(summary, dict):
            raise ValueError(f"{label} is missing metric {name}")
        count = int(summary.get("count", 0))
        if not 1 <= count <= maximum_count:
            raise ValueError(f"{label}/{name} coverage is empty or oversized")
        for field in ("mean", "p50", "p90", "p99"):
            value = summary.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{label}/{name}/{field} must be finite")
        return summary

    def require_rgb_v2_metrics(container, expected_count, *, four_frame, label):
        required = (
            "raw_rgb_l1",
            "raw_rgb_mse",
            "clamped_rgb_l1",
            "clamped_rgb_mse",
            "clamped_psnr_db",
            "clamped_ssim",
            "clamped_lpips",
            "rgb_valid_ratio",
            "rgb_out_of_range_pixel_ratio",
            "rgb_overshoot_positive_p50",
            "rgb_overshoot_positive_p90",
            "rgb_overshoot_positive_p99",
            "rgb_overshoot_positive_max",
        )
        summaries = {
            name: require_summary(container, name, expected_count, label)
            for name in required
        }
        for field in ("mean", "p50", "p90", "p99"):
            if summaries["clamped_rgb_l1"][field] > summaries["raw_rgb_l1"][field] + 1e-8:
                raise ValueError(f"{label}: clamped L1 exceeds raw L1")
            if summaries["clamped_rgb_mse"][field] > summaries["raw_rgb_mse"][field] + 1e-8:
                raise ValueError(f"{label}: clamped MSE exceeds raw MSE")
            ratio = summaries["rgb_out_of_range_pixel_ratio"][field]
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{label}: out-of-range ratio is outside [0,1]")
            if not (
                summaries["rgb_overshoot_positive_p50"][field]
                <= summaries["rgb_overshoot_positive_p90"][field]
                <= summaries["rgb_overshoot_positive_p99"][field]
                <= summaries["rgb_overshoot_positive_max"][field]
            ):
                raise ValueError(f"{label}: overshoot summaries are not monotonic")
        legacy = {
            "rgb_l1", "rgb_mse", "psnr_db", "ssim", "lpips",
            "temporal_delta_l1", "temporal_delta_lpips",
        }
        if legacy.intersection(container) or any(
            name.startswith("temporal_delta_") for name in container
        ):
            raise ValueError(f"{label}: legacy or mixed-domain metrics are present")
        if four_frame:
            for name in (
                "clamped_temporal_delta_l1",
                "clamped_temporal_delta_lpips",
                "clamped_temporal_delta_l1_pair_01",
                "clamped_temporal_delta_lpips_pair_01",
                "clamped_temporal_delta_l1_pair_12",
                "clamped_temporal_delta_lpips_pair_12",
                "clamped_temporal_delta_l1_pair_23",
                "clamped_temporal_delta_lpips_pair_23",
            ):
                require_summary(container, name, expected_count, label)
            for name in (
                "optical_flow_valid_ratio",
                "static_flow_valid_ratio",
                "dynamic_flow_valid_ratio",
                "clamped_optical_flow_warp_l1",
            ):
                require_summary(container, name, expected_count, label)
            for name in (
                "clamped_static_flicker_l1",
                "clamped_motion_flow_epe_px",
            ):
                require_nonempty_summary(container, name, expected_count, label)
            for base in (
                "clamped_optical_flow_warp_l1",
                "clamped_static_flicker_l1",
                "clamped_motion_flow_epe_px",
            ):
                for pair in ("pair_01", "pair_12", "pair_23"):
                    require_nonempty_summary(
                        container, f"{base}_{pair}", expected_count, label
                    )

    status_path = root / "job-status.json"
    if not status_path.is_file():
        raise FileNotFoundError("A1 report requires job-status.json")
    job_status = json.loads(status_path.read_text())
    if job_status.get("schema") != "stereo-tokenizer-stage-a1-job-status-v1":
        raise ValueError("job status schema mismatch")
    expected_artifacts = {
        str(path.relative_to(root)) for path in (*quality_paths, *benchmark_paths)
    }
    jobs = job_status.get("jobs", [])
    actual_artifacts = {job.get("artifact") for job in jobs}
    if len(jobs) != len(expected_artifacts) or actual_artifacts != expected_artifacts:
        raise ValueError("job status does not cover every result artifact exactly once")
    for job in jobs:
        if job.get("state") != "COMPLETED" or int(job.get("exit_code", -1)) != 0:
            raise ValueError("one or more formal Stage A jobs did not complete successfully")
        if not str(job.get("job_id", "")):
            raise ValueError("job status is missing one Slurm job ID")
        artifact = root / job["artifact"]
        require_sha256(job.get("sha256"), f"{artifact} status hash")
        if sha256_file(artifact) != job["sha256"]:
            raise ValueError(f"result artifact hash mismatch: {artifact}")
        log_path = root / str(job.get("log", ""))
        if not log_path.is_file():
            raise FileNotFoundError(f"formal job log is missing: {log_path}")
        require_sha256(job.get("log_sha256"), f"{log_path} status hash")
        if sha256_file(log_path) != job["log_sha256"]:
            raise ValueError(f"formal job log hash mismatch: {log_path}")

    expected = {
        ("umi", "stereo", None),
        *(("umi", "mono", camera) for camera in (
            "observation.images.cam_head_left",
            "observation.images.cam_head_right",
            "observation.images.cam_left_wrist_left",
            "observation.images.cam_left_wrist_right",
            "observation.images.cam_right_wrist_left",
            "observation.images.cam_right_wrist_right",
        )),
        ("libero", "mono", "observation.images.cam_head_left"),
        ("libero", "mono", "observation.images.cam_left_wrist_left"),
        ("hy", "mono", None),
    }
    actual = set()
    visualization_slots = set()
    source_fingerprints = set()
    environment_fingerprints = set()
    checkpoint_fingerprint = None
    checkpoint_sha256 = None
    flow_fingerprint = None
    selection_rows = {}
    for result in quality:
        if result.get("schema") != "stereo-tokenizer-stage-a1-result-v3":
            raise ValueError("quality result schema mismatch")
        dataset = result["dataset"]
        key = (dataset["dataset_id"], dataset["eye_mode"], dataset["camera_key"])
        actual.add(key)
        expected_count = 1024 if dataset["dataset_id"] in {"umi", "hy"} else 256
        expected_modes = {
            *(f"{dataset['eye_mode']}/single_frame/source_{index}" for index in range(4)),
            f"{dataset['eye_mode']}/four_frame",
        }
        if result.get("status") != "formal" or dataset["sample_count"] != expected_count:
            raise ValueError(f"non-formal or wrong sample count for {key}")
        if result.get("posterior") != "mean" or result.get("quality_precision") != "fp32":
            raise ValueError(f"quality precision/posterior contract mismatch for {key}")
        if set(result.get("requested_modes", [])) != expected_modes:
            raise ValueError(f"mode coverage mismatch for {key}")
        if result.get("single_frame_source_indices") != [0, 1, 2, 3]:
            raise ValueError(f"single-frame source coverage mismatch for {key}")
        checkpoint = result["checkpoint"]
        require_sha256(checkpoint.get("sha256"), "checkpoint hash")
        counters = checkpoint.get("stereo_update_counters", {})
        if not isinstance(counters, dict) or not isinstance(
            counters.get("generator_updates"), int
        ):
            raise ValueError("checkpoint generator update counter is missing")
        current_checkpoint_fingerprint = json.dumps(checkpoint, sort_keys=True)
        if checkpoint_fingerprint is None:
            checkpoint_fingerprint = current_checkpoint_fingerprint
            checkpoint_sha256 = checkpoint["sha256"]
        elif current_checkpoint_fingerprint != checkpoint_fingerprint:
            raise ValueError("checkpoint provenance mismatch across quality results")
        require_sha256(dataset.get("selection_sha256"), f"{key} selection semantic hash")
        require_sha256(dataset.get("selection_file_sha256"), f"{key} selection file hash")
        selection_path = Path(dataset["selection_path"])
        if not selection_path.is_file() or sha256_file(selection_path) != dataset["selection_file_sha256"]:
            raise ValueError(f"selection file drift for {key}")
        selection_payload = json.loads(selection_path.read_text())
        if (
            int(selection_payload.get("sample_count", -1)) != expected_count
            or len(selection_payload.get("records", [])) != expected_count
        ):
            raise ValueError(f"selection sample count mismatch for {key}")
        decode_validation = selection_payload.get("decode_validation", {})
        if int(decode_validation.get("accepted_count", -1)) != expected_count:
            raise ValueError(f"selection decode audit mismatch for {key}")
        require_sha256(
            decode_validation.get("rejected_episode_ids_sha256"),
            f"{key} rejected episode IDs hash",
        )
        identity = dataset.get("identity_contract", {})
        require_sha256(identity.get("sha256"), f"{key} identity contract hash")
        require_sha256(identity.get("source_manifest_sha256"), f"{key} manifest hash")
        manifest_sha = identity["source_manifest_sha256"]
        if dataset["dataset_id"] == "hy":
            if (
                dataset.get("data_backend") != "hy_lance_manifest"
                or dataset.get("camera_key") is not None
                or dataset.get("excluded_source_groups", {}).get("groups")
                != ["table_014"]
                or not dataset.get("included_source_groups")
                or "table_014" in dataset["included_source_groups"]
            ):
                raise ValueError("Hy manifest backend/exclusion contract mismatch")
            hy_manifest = dataset.get("hy_manifest", {})
            require_sha256(hy_manifest.get("sha256"), "Hy production manifest hash")
            hy_manifest_path = Path(str(hy_manifest.get("path", "")))
            if (
                not hy_manifest_path.is_file()
                or sha256_file(hy_manifest_path) != hy_manifest["sha256"]
                or selection_payload.get("hy_manifest") != hy_manifest
                or selection_payload.get("excluded_source_groups")
                != dataset["excluded_source_groups"]
                or selection_payload.get("included_source_groups")
                != dataset["included_source_groups"]
            ):
                raise ValueError("Hy production manifest or exclusion provenance drifted")
            manifest_sha = hy_manifest["sha256"]
        else:
            config_hashes = dataset.get("canonical_config_sha256", {})
            if not config_hashes:
                raise ValueError(f"canonical config hashes are missing for {key}")
            for config_path, digest in config_hashes.items():
                require_sha256(digest, f"{config_path} config hash")
                if sha256_file(Path(config_path)) != digest:
                    raise ValueError(f"canonical config drift for {config_path}")
            loader_sha = dataset.get("canonical_loader", {}).get("git_sha")
            if loader_sha != "d51377ac450b0066bc0c8eb13939bcfae47275ff":
                raise ValueError("canonical loader SHA mismatch")
        selection_rows.setdefault(
            dataset["dataset_id"],
            {
                "semantic_sha256": dataset["selection_sha256"],
                "file_sha256": dataset["selection_file_sha256"],
                "manifest_sha256": manifest_sha,
                "sample_count": expected_count,
                "decode_checked": int(
                    decode_validation.get("checked_candidate_count", -1)
                ),
                "decode_rejected": int(
                    decode_validation.get("rejected_count", -1)
                ),
                "rejected_ids_sha256": decode_validation[
                    "rejected_episode_ids_sha256"
                ],
                "excluded": (
                    dataset.get("excluded_source_groups")
                    if dataset["dataset_id"] == "hy"
                    else None
                ),
            },
        )
        if selection_rows[dataset["dataset_id"]]["semantic_sha256"] != dataset["selection_sha256"]:
            raise ValueError("one dataset used multiple selections")

        teacher = result.get("teacher", {})
        if dataset["eye_mode"] == "mono":
            if teacher.get("source_sha") != DA3_SOURCE_SHA or teacher.get("checkpoint_sha256") != DA3_CHECKPOINT_SHA256:
                raise ValueError(f"DA3 provenance mismatch for {key}")
        elif (
            teacher.get("backend") != "las2_h"
            or teacher.get("source_sha") != LAS2_H_SOURCE_SHA
            or teacher.get("checkpoint_sha256") != LAS2_H_CHECKPOINT_SHA256
        ):
            raise ValueError("LAS2-H provenance mismatch")

        flow_teacher = result.get("flow_teacher", {})
        if (
            flow_teacher.get("name") != "torchvision.raft_large"
            or flow_teacher.get("architecture") != "RAFT-Large"
            or flow_teacher.get("precision") != "fp32"
            or flow_teacher.get("flow_unit") != "content-crop pixels"
            or float(flow_teacher.get("static_flow_max_px", -1))
            != STATIC_FLOW_MAX_PX
            or float(flow_teacher.get("dynamic_flow_min_px", -1))
            != DYNAMIC_FLOW_MIN_PX
            or float(flow_teacher.get("forward_backward_relative_threshold", -1))
            != FLOW_FB_RELATIVE_THRESHOLD
            or float(flow_teacher.get("forward_backward_absolute_threshold_px2", -1))
            != FLOW_FB_ABSOLUTE_THRESHOLD_PX2
        ):
            raise ValueError(f"RAFT flow contract mismatch for {key}")
        require_sha256(
            flow_teacher.get("checkpoint_sha256"), f"{key} RAFT checkpoint hash"
        )
        current_flow_fingerprint = json.dumps(flow_teacher, sort_keys=True)
        if flow_fingerprint is None:
            flow_fingerprint = current_flow_fingerprint
        elif current_flow_fingerprint != flow_fingerprint:
            raise ValueError("RAFT provenance mismatch across quality results")

        parameters = result.get("tokenizer_parameters", {})
        if (
            int(parameters.get("total", 0)) <= 0
            or int(parameters.get("architecturally_trainable", 0)) <= 0
            or int(parameters.get("runtime_requires_grad", -1)) != 0
            or parameters.get("evaluation_state") != "eval_inference_mode_posterior_mean"
        ):
            raise ValueError(f"Tokenizer freeze/parameter provenance mismatch for {key}")
        if set(result.get("metrics", {})) != expected_modes:
            raise ValueError(f"metric mode coverage mismatch for {key}")
        for mode_id, mode in result["metrics"].items():
            if mode["sample_count"] != expected_count:
                raise ValueError(f"metric sample count mismatch for {key}")
            if int(mode.get("valid_rgb_values", 0)) <= 0 or int(mode.get("valid_teacher_pixels", 0)) <= 0:
                raise ValueError(f"empty metric mask for {key}")
            health = mode["output_health"]
            if (
                health["nan_count"]
                or health["inf_count"]
                or health.get("invalid_sample_count")
                or health.get("invalid_sample_ids")
            ):
                raise ValueError(f"invalid output in {key}")
            expected_health = {
                "all_value_count",
                "all_raw_min",
                "all_raw_max",
                "valid_value_count",
                "valid_raw_min",
                "valid_raw_max",
                "valid_pixel_count",
                "out_of_range_pixel_count",
                "out_of_range_pixel_ratio",
            }
            if not expected_health.issubset(health):
                raise ValueError(f"RGB v2 output health is incomplete for {key}")
            if {"value_count", "raw_min", "raw_max", "abs_gt_one_count", "abs_gt_one_ratio"}.intersection(health):
                raise ValueError(f"legacy output health is present for {key}")
            for name in ("all_raw_min", "all_raw_max", "valid_raw_min", "valid_raw_max", "out_of_range_pixel_ratio"):
                if not np.isfinite(health[name]):
                    raise ValueError(f"non-finite output health field {name} for {key}")
            if (
                int(health["all_value_count"]) <= 0
                or int(health["valid_value_count"]) != int(mode["valid_rgb_values"])
                or int(health["valid_pixel_count"]) * 3 != int(health["valid_value_count"])
                or not 0 <= int(health["out_of_range_pixel_count"]) <= int(health["valid_pixel_count"])
                or not 0.0 <= float(health["out_of_range_pixel_ratio"]) <= 1.0
                or abs(
                    float(health["out_of_range_pixel_ratio"])
                    - int(health["out_of_range_pixel_count"]) / int(health["valid_pixel_count"])
                ) > 1e-12
                or float(health["all_raw_min"]) > float(health["all_raw_max"])
                or float(health["valid_raw_min"]) > float(health["valid_raw_max"])
            ):
                raise ValueError(f"RGB v2 output health contract mismatch for {key}")
            for view_name, view_metrics in mode.get("per_view", {}).items():
                require_rgb_v2_metrics(
                    view_metrics,
                    expected_count,
                    four_frame=mode_id.endswith("/four_frame"),
                    label=f"{key}/{mode_id}/{view_name}",
                )
            require_rgb_v2_metrics(
                mode.get("per_sample_macro", {}),
                expected_count,
                four_frame=mode_id.endswith("/four_frame"),
                label=f"{key}/{mode_id}/macro",
            )
            if mode_id.endswith("/four_frame"):
                geometry_prefix = (
                    "reconstruction_teacher"
                    if dataset["eye_mode"] == "mono"
                    else "depth_head_teacher"
                )
                for scope_name, scope_metrics in (
                    *tuple(mode.get("per_view", {}).items()),
                    ("macro", mode.get("per_sample_macro", {})),
                ):
                    require_summary(
                        scope_metrics,
                        f"{geometry_prefix}_temporal_geometry_valid_ratio",
                        expected_count,
                        f"{key}/{mode_id}/{scope_name}",
                    )
                    require_nonempty_summary(
                        scope_metrics,
                        f"{geometry_prefix}_temporal_geometry_warp_l1",
                        expected_count,
                        f"{key}/{mode_id}/{scope_name}",
                    )
            teacher_invalid = mode.get("teacher_invalid_samples", [])
            if int(mode.get("teacher_invalid_count", -1)) != len(teacher_invalid):
                raise ValueError(f"teacher-invalid count mismatch for {key}")
            identities = set()
            for entry in teacher_invalid:
                if (
                    not isinstance(entry, dict)
                    or not str(entry.get("sample_id", ""))
                    or not str(entry.get("view", ""))
                    or entry.get("reason") != "empty_teacher_mask"
                ):
                    raise ValueError(f"invalid teacher exclusion record for {key}")
                identity = (entry["sample_id"], entry["view"])
                if identity in identities:
                    raise ValueError(f"duplicate teacher exclusion record for {key}")
                identities.add(identity)

        provenance = result.get("provenance", {})
        for name in ("cwd", "git_branch", "git_commit", "git_diff_sha256", "git_status_porcelain"):
            if name not in provenance:
                raise ValueError(f"quality provenance is missing {name}")
        require_sha256(provenance["git_diff_sha256"], "source diff hash")
        if len(provenance["git_commit"]) != 40:
            raise ValueError("source commit must be one full Git SHA")
        source_fingerprints.add(tuple(provenance[name] for name in (
            "cwd", "git_branch", "git_commit", "git_diff_sha256", "git_status_porcelain"
        )))
        environment = provenance.get("environment", {})
        if not environment.get("python", "").startswith("3.12.") or "H100" not in str(environment.get("gpu_name", "")):
            raise ValueError("formal Stage A quality must use Python 3.12 on H100")
        require_sha256(environment.get("uv_lock_sha256"), "quality uv.lock hash")
        require_metric_backbone(environment, "quality")
        environment_fingerprints.add(json.dumps(environment, sort_keys=True))

        cases = result.get("visualizations", [])
        wants_cases = key in {
            ("umi", "stereo", None),
            ("libero", "mono", "observation.images.cam_head_left"),
        }
        if wants_cases:
            expected_cases = {(slot, source) for slot in range(8) for source in range(4)}
            actual_cases = {(case["slot"], case["source_frame_index"]) for case in cases}
            if len(cases) != 32 or actual_cases != expected_cases:
                raise ValueError(f"fixed visualization coverage mismatch for {key}")
            visual_dir = Path(provenance["resolved_args"]["visualization_dir"])
            case_index = visual_dir / "cases.json"
            if not case_index.is_file() or json.loads(case_index.read_text()) != cases:
                raise ValueError(f"visualization index mismatch for {key}")
            for case in cases:
                for field in ("rgb_file", "geometry_file"):
                    image = visual_dir / case[field]
                    if not image.is_file() or image.stat().st_size == 0:
                        raise FileNotFoundError(f"visualization file missing: {image}")
                visualization_slots.add((dataset["dataset_id"], case["slot"]))
        elif cases:
            raise ValueError(f"unexpected visualizations for {key}")
    if actual != expected:
        raise ValueError(f"quality coverage mismatch: missing={expected-actual}, extra={actual-expected}")
    if visualization_slots != {
        *(("umi", index) for index in range(8)),
        *(("libero", index) for index in range(8)),
    }:
        raise ValueError("A1 report requires 8 fixed UMI and 8 fixed LIBERO visualizations")

    benchmark_eyes = set()
    for result in benchmarks:
        if result.get("schema") != "stereo-tokenizer-stage-a1-benchmark-v1":
            raise ValueError("benchmark result schema mismatch")
        dataset = result.get("dataset", {})
        benchmark_eyes.add(dataset.get("eye_mode"))
        if dataset.get("dataset_id") != "umi":
            raise ValueError("efficiency benchmark must use representative UMI input")
        if result.get("status") != "formal" or (
            result.get("warmup"), result.get("iterations"), result.get("repeats")
        ) != (20, 100, 3):
            raise ValueError("benchmark contract mismatch")
        if (
            result.get("precision") != "bf16"
            or int(result.get("batch_size", -1)) != 1
            or result.get("posterior") != "mean"
            or result.get("timing_scope") != "model_only_excludes_data_decode_and_teacher"
            or set(result.get("modes", {})) != {"single_frame", "four_frame"}
            or result.get("checkpoint", {}).get("sha256") != checkpoint_sha256
        ):
            raise ValueError("benchmark precision/scope/checkpoint mismatch")
        provenance = result.get("provenance", {})
        require_sha256(provenance.get("git_diff_sha256"), "benchmark source diff hash")
        source_fingerprints.add(tuple(provenance.get(name) for name in (
            "cwd", "git_branch", "git_commit", "git_diff_sha256", "git_status_porcelain"
        )))
        environment = provenance.get("environment", {})
        if not environment.get("python", "").startswith("3.12.") or "H100" not in str(environment.get("gpu_name", "")):
            raise ValueError("formal benchmark must use Python 3.12 on H100")
        require_sha256(environment.get("uv_lock_sha256"), "benchmark uv.lock hash")
        require_metric_backbone(environment, "benchmark")
        environment_fingerprints.add(json.dumps(environment, sort_keys=True))
    if benchmark_eyes != {"mono", "stereo"}:
        raise ValueError("benchmark must cover UMI mono and stereo")
    if len(source_fingerprints) != 1 or len(environment_fingerprints) != 1:
        raise ValueError("formal jobs used inconsistent source or environments")

    source = quality[0]["provenance"]
    source_patch = root / "source.patch"
    if (
        not source_patch.is_file()
        or sha256_file(source_patch) != source["git_diff_sha256"]
    ):
        raise ValueError("source.patch does not match the recorded Git diff SHA256")
    environment = source["environment"]
    metric_backbone = require_metric_backbone(environment, "formal")
    metric_backbone_path = Path(metric_backbone["path"])
    if (
        not metric_backbone_path.is_file()
        or sha256_file(metric_backbone_path) != metric_backbone["sha256"]
    ):
        raise ValueError("frozen LPIPS VGG16 file is missing or has changed")
    checkpoint = quality[0]["checkpoint"]
    flow_teacher = quality[0]["flow_teacher"]
    flow_checkpoint_path = Path(flow_teacher["checkpoint"])
    if (
        not flow_checkpoint_path.is_file()
        or sha256_file(flow_checkpoint_path) != flow_teacher["checkpoint_sha256"]
    ):
        raise ValueError("frozen RAFT checkpoint is missing or has changed")
    parameters = quality[0]["tokenizer_parameters"]
    package_text = ", ".join(
        f"{name}={version}" for name, version in sorted(environment["packages"].items())
    )
    status_text = source["git_status_porcelain"].replace("\n", "; ")
    lines = [
        "# Stereo Tokenizer Stage A1 Baseline（Preliminary）",
        "",
        "> 状态：PRELIMINARY。v6 增加 HY（显式排除 Table014）、RAFT warp/static flicker/motion consistency 和 teacher-relative temporal geometry；rFID 暂不执行。",
        "",
        "> v4 失效原因：raw L1/MSE/PSNR/LPIPS 与 clamp 后 SSIM 不在同一图像域，且旧越界阈值 abs(output)>1 与合法域 [-0.5,0.5] 不一致。v4 artifact 仅保留作审计。",
        "",
        "## 实验合同与 provenance",
        "",
        f"- Artifact 根目录：`{root}`",
        f"- 实际 cwd：`{source['cwd']}`",
        f"- Git branch / commit：`{source['git_branch']}` / `{source['git_commit']}`",
        f"- 未提交代码 diff：`{source_patch}`；SHA256：`{source['git_diff_sha256']}`",
        f"- `git status --porcelain`：`{status_text}`",
        f"- Checkpoint：`{checkpoint['path']}`",
        f"- Checkpoint SHA256：`{checkpoint['sha256']}`；global_step={checkpoint['global_step']}；epoch={checkpoint['epoch']}",
        f"- 直接训练计数：`{json.dumps(checkpoint['stereo_update_counters'], sort_keys=True)}`",
        "- 质量：FP32；效率：BF16；posterior mean；Tokenizer `eval + inference_mode` 且运行时冻结。",
        f"- Python：`{environment['python'].split()[0]}`；GPU：`{environment['gpu_name']}`；CUDA：`{environment['torch_cuda']}`；cuDNN：`{environment['cudnn']}`",
        f"- `uv.lock` SHA256：`{environment['uv_lock_sha256']}`",
        f"- LPIPS VGG16：`{metric_backbone['path']}`；SHA256：`{metric_backbone['sha256']}`；预处理：`{metric_backbone['preprocessing']}`",
        f"- RAFT-Large：`{flow_teacher['checkpoint']}`；SHA256：`{flow_teacher['checkpoint_sha256']}`；FP32；static≤{flow_teacher['static_flow_max_px']} px；dynamic≥{flow_teacher['dynamic_flow_min_px']} px。",
        f"- 关键包：{package_text}",
        f"- Tokenizer 参数：total={parameters['total']:,}；架构可训练={parameters['architecturally_trainable']:,}；运行时 requires_grad={parameters['runtime_requires_grad']:,}",
        f"- DA3：source `{DA3_SOURCE_SHA}`；weights `{DA3_CHECKPOINT_SHA256}`",
        f"- LAS2-H：source `{LAS2_H_SOURCE_SHA}`；weights `{LAS2_H_CHECKPOINT_SHA256}`",
        "",
        "### 数据与哈希",
        "",
        "| Dataset | Windows/cell | Decode checked/rejected | Explicit exclusion | Selection semantic SHA256 | Selection file SHA256 | Manifest SHA256 |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for dataset_id, row in sorted(selection_rows.items()):
        lines.append(
            f"| {dataset_id} | {row['sample_count']} | "
            f"{row['decode_checked']}/{row['decode_rejected']} | "
            f"{('none' if row['excluded'] is None else ','.join(row['excluded']['groups']) + ':' + str(row['excluded']['episode_count']))} | "
            f"`{row['semantic_sha256']}` | `{row['file_sha256']}` | "
            f"`{row['manifest_sha256']}` |"
        )
    lines.extend([
        "",
        "### 覆盖矩阵",
        "",
        "| Dataset | Eye | Camera/view cell | Windows | Modes | Macro inclusion |",
        "| --- | --- | --- | ---: | --- | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        camera = dataset["camera_key"] or "3 canonical stereo pairs"
        lines.append(
            f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {camera} | "
            f"{dataset['sample_count']} | single source 0/1/2/3 + four-frame | yes |"
        )
    lines.extend([
        "",
        "## Clamp-domain RGB 图像质量（per camera/view）",
        "",
        "下表所有 L1/MSE/PSNR/SSIM/LPIPS 都使用 `prediction.clamp(-0.5, 0.5)`；PSNR data_range=1.0。",
        "",
        "| Dataset | Eye | Camera/view | Mode | L1 mean | P50 | P90 | P99 | MSE | PSNR | SSIM | LPIPS | RGB mask |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            for view_name, metric in mode["per_view"].items():
                lines.append(
                    f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | {mode_id} | "
                    f"{metric['clamped_rgb_l1']['mean']:.6f} | {metric['clamped_rgb_l1']['p50']:.6f} | "
                    f"{metric['clamped_rgb_l1']['p90']:.6f} | {metric['clamped_rgb_l1']['p99']:.6f} | "
                    f"{metric['clamped_rgb_mse']['mean']:.6f} | {metric['clamped_psnr_db']['mean']:.3f} | "
                    f"{metric['clamped_ssim']['mean']:.6f} | {metric['clamped_lpips']['mean']:.6f} | "
                    f"{metric['rgb_valid_ratio']['mean']:.6f} |"
                )
    lines.extend([
        "",
        "### Dataset/eye/mode 等权 macro（clamp-domain）",
        "",
        "| Dataset | Eye | Mode | RGB L1 | MSE | PSNR | SSIM | LPIPS |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    macro_cells = {}
    for result in quality:
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            key = (dataset["dataset_id"], dataset["eye_mode"], mode_id)
            macro_cells.setdefault(key, []).append(mode["per_sample_macro"])
    for key, cells in sorted(macro_cells.items()):
        means = {
            name: float(np.mean([cell[name]["mean"] for cell in cells]))
            for name in (
                "clamped_rgb_l1",
                "clamped_rgb_mse",
                "clamped_psnr_db",
                "clamped_ssim",
                "clamped_lpips",
            )
        }
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {means['clamped_rgb_l1']:.6f} | "
            f"{means['clamped_rgb_mse']:.6f} | {means['clamped_psnr_db']:.3f} | "
            f"{means['clamped_ssim']:.6f} | {means['clamped_lpips']:.6f} |"
        )
    lines.extend([
        "",
        "## Raw RGB 数值稳定性诊断",
        "",
        "Raw L1/MSE 只用于诊断 decoder 数值稳定性，不作为正式图像质量结论。",
        "",
        "| Dataset | Eye | Camera/view | Mode | Raw L1 mean/P50/P90/P99 | Raw MSE mean/P50/P90/P99 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            for view_name, metric in mode["per_view"].items():
                l1 = metric["raw_rgb_l1"]
                mse = metric["raw_rgb_mse"]
                lines.append(
                    f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | {mode_id} | "
                    f"{l1['mean']:.6f}/{l1['p50']:.6f}/{l1['p90']:.6f}/{l1['p99']:.6f} | "
                    f"{mse['mean']:.6f}/{mse['p50']:.6f}/{mse['p90']:.6f}/{mse['p99']:.6f} |"
                )
    lines.extend([
        "",
        "## RGB 越界与 overshoot 诊断",
        "",
        "越界像素定义为 valid 时空像素中任一 RGB channel 超出 [-0.5,0.5]。每个 sample/view 先对正 overshoot `max_channel(relu(abs(output)-0.5))` 的全部越界像素精确求 P50/P90/P99/max，再汇总样本分布；不是全局近似分位数。无越界样本的 positive count 和这些统计均为 0。",
        "",
        "| Dataset | Eye | Camera/view | Mode | OOR ratio mean/P50/P90/P99 | sample P50 mean/P50/P90/P99 | sample P90 mean/P50/P90/P99 | sample P99 mean/P50/P90/P99 | sample max mean/P50/P90/P99 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            for view_name, metric in mode["per_view"].items():
                def render_summary(name):
                    value = metric[name]
                    return (
                        f"{value['mean']:.8f}/{value['p50']:.8f}/"
                        f"{value['p90']:.8f}/{value['p99']:.8f}"
                    )
                lines.append(
                    f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | {mode_id} | "
                    f"{render_summary('rgb_out_of_range_pixel_ratio')} | "
                    f"{render_summary('rgb_overshoot_positive_p50')} | "
                    f"{render_summary('rgb_overshoot_positive_p90')} | "
                    f"{render_summary('rgb_overshoot_positive_p99')} | "
                    f"{render_summary('rgb_overshoot_positive_max')} |"
                )
    lines.extend([
        "",
        "## Four-frame 时间一致性（clamp-domain）",
        "",
        "先逐帧 clamp 到 [-0.5,0.5]，再计算相邻帧 temporal delta。",
        "",
        "| Dataset | Eye | Camera/view | Δ L1 | Δ LPIPS | Δ01 L1/LPIPS | Δ12 L1/LPIPS | Δ23 L1/LPIPS |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        mode = result["metrics"][f"{dataset['eye_mode']}/four_frame"]
        for view_name, metric in mode["per_view"].items():
            pairs = []
            for pair in ("pair_01", "pair_12", "pair_23"):
                pairs.append(
                    f"{metric['clamped_temporal_delta_l1_' + pair]['mean']:.6f}/"
                    f"{metric['clamped_temporal_delta_lpips_' + pair]['mean']:.6f}"
                )
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | "
                f"{metric['clamped_temporal_delta_l1']['mean']:.6f} | "
                f"{metric['clamped_temporal_delta_lpips']['mean']:.6f} | "
                f"{pairs[0]} | {pairs[1]} | {pairs[2]} |"
            )
    lines.extend([
        "",
        "### RAFT flow-aware 时间指标",
        "",
        "Warp L1 使用目标视频的 backward flow 对齐相邻帧；static flicker 仅统计目标 flow≤0.5 px；motion EPE 仅统计目标 flow≥1.0 px。0.5–1.0 px 灰区不进入后二者。",
        "",
        "| Dataset | Eye | Camera/view | Warp L1 | Static flicker L1 | Motion EPE px | Flow/static/dynamic coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        mode = result["metrics"][f"{dataset['eye_mode']}/four_frame"]
        for view_name, metric in mode["per_view"].items():
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | "
                f"{metric['clamped_optical_flow_warp_l1']['mean']:.6f} | "
                f"{metric['clamped_static_flicker_l1']['mean']:.6f} | "
                f"{metric['clamped_motion_flow_epe_px']['mean']:.6f} | "
                f"{metric['optical_flow_valid_ratio']['mean']:.4f}/"
                f"{metric['static_flow_valid_ratio']['mean']:.4f}/"
                f"{metric['dynamic_flow_valid_ratio']['mean']:.4f} |"
            )
    lines.extend([
        "",
        "## Teacher-relative 几何（非真实 GT accuracy）",
        "",
        "| Dataset | Eye | Camera/view | Mode | Metric kind | log-L1 | RMSE | SILog | Mask coverage | Valid samples |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        prefix = "reconstruction_teacher" if dataset["eye_mode"] == "mono" else "depth_head_teacher"
        for mode_id, mode in result["metrics"].items():
            for view_name, metric in mode["per_view"].items():
                l1 = metric.get(prefix + "_relative_log_l1")
                rmse = metric.get(prefix + "_relative_log_rmse")
                silog = metric.get(prefix + "_relative_log_silog")
                coverage = metric.get(prefix + "_valid_ratio")
                render = lambda value: "N/A" if value is None else f"{value['mean']:.6f}"
                lines.append(
                    f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | {mode_id} | {prefix} | "
                    f"{render(l1)} | {render(rmse)} | {render(silog)} | "
                    f"{render(coverage)} | {0 if l1 is None else l1['count']} |"
                )
    lines.extend([
        "",
        "### Teacher-relative temporal geometry consistency",
        "",
        "| Dataset | Eye | Camera/view | Flow-aligned log-geometry L1 | Valid coverage | Pairs 01/12/23 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        prefix = "reconstruction_teacher" if dataset["eye_mode"] == "mono" else "depth_head_teacher"
        mode = result["metrics"][f"{dataset['eye_mode']}/four_frame"]
        for view_name, metric in mode["per_view"].items():
            name = f"{prefix}_temporal_geometry_warp_l1"
            pair_values = "/".join(
                f"{metric[f'{name}_pair_{pair}']['mean']:.6f}"
                for pair in ("01", "12", "23")
            )
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | {view_name} | "
                f"{metric[name]['mean']:.6f} | "
                f"{metric[f'{prefix}_temporal_geometry_valid_ratio']['mean']:.4f} | "
                f"{pair_values} |"
            )
    lines.extend([
        "",
        "## Bottleneck 与效率",
        "",
        "| Eye | Mode | Encode P50/P90 ms | Posterior mean P50/P90 ms | Decode P50/P90 ms | E2E P50/P90 ms | samples/s | frames/s | Peak alloc/reserved GiB |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ])
    for result in sorted(benchmarks, key=lambda value: value["dataset"]["eye_mode"]):
        for mode_name, mode in result["modes"].items():
            encode = mode["encode_including_posterior_mean"]
            posterior = mode["cached_posterior_mean"]
            decode = mode["decode"]
            timing = mode["end_to_end"]
            lines.append(
                f"| {result['dataset']['eye_mode']} | {mode_name} | "
                f"{encode['p50_ms']:.3f}/{encode['p90_ms']:.3f} | "
                f"{posterior['p50_ms']:.3f}/{posterior['p90_ms']:.3f} | "
                f"{decode['p50_ms']:.3f}/{decode['p90_ms']:.3f} | "
                f"{timing['p50_ms']:.3f}/{timing['p90_ms']:.3f} | "
                f"{mode['throughput']['samples_per_second']:.3f} | "
                f"{mode['throughput']['frames_per_second']:.3f} | "
                f"{timing['peak_allocated_bytes'] / 2**30:.3f}/"
                f"{timing['peak_reserved_bytes'] / 2**30:.3f} |"
            )
    lines.extend([
        "",
        "### Latent ABI",
        "",
        "| Dataset | Eye | Camera | Mode | Input shape/dtype | Latent shape/dtype | C | Tokens/window | Tokens/input frame | Spatial × | Temporal × | View × |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            abi = mode["latent_abi"]
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | "
                f"{dataset['camera_key'] or 'three canonical pairs'} | {mode_id} | "
                f"`{abi['input_shape_without_batch']}` / `{abi['input_dtype']}` | "
                f"`{abi['latent_shape_without_batch']}` / `{abi['latent_dtype']}` | "
                f"{abi['latent_channels']} | {abi['tokens_per_window']} | "
                f"{abi['tokens_per_input_frame']:.3f} | {abi['spatial_compression_ratio']:.1f} | "
                f"{abi['temporal_compression_ratio']:.1f} | {abi['view_compression_ratio']:.1f} |"
            )
    lines.extend([
        "",
        "## 输出健康、失败与排除样本",
        "",
        "| Dataset | Eye | Camera | Mode | NaN | Inf | Invalid | Teacher-empty | All raw min/max | Valid raw min/max | OOR pixels/valid pixels | OOR ratio | Valid RGB values | Valid teacher pixels |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ])
    teacher_exclusions = []
    for result in sorted(quality, key=lambda value: (
        value["dataset"]["dataset_id"], value["dataset"]["eye_mode"], value["dataset"]["camera_key"] or ""
    )):
        dataset = result["dataset"]
        for mode_id, mode in result["metrics"].items():
            health = mode["output_health"]
            lines.append(
                f"| {dataset['dataset_id']} | {dataset['eye_mode']} | "
                f"{dataset['camera_key'] or 'three canonical pairs'} | {mode_id} | "
                f"{health['nan_count']} | {health['inf_count']} | "
                f"{health['invalid_sample_count']} | {mode['teacher_invalid_count']} | "
                f"{health['all_raw_min']:.6f}/{health['all_raw_max']:.6f} | "
                f"{health['valid_raw_min']:.6f}/{health['valid_raw_max']:.6f} | "
                f"{health['out_of_range_pixel_count']}/{health['valid_pixel_count']} | "
                f"{health['out_of_range_pixel_ratio']:.8f} | {mode['valid_rgb_values']} | "
                f"{mode['valid_teacher_pixels']} |"
            )
            teacher_exclusions.extend(
                {
                    "dataset": dataset["dataset_id"],
                    "eye": dataset["eye_mode"],
                    "camera": dataset["camera_key"],
                    "mode": mode_id,
                    **entry,
                }
                for entry in mode["teacher_invalid_samples"]
            )
    lines.extend([
        "",
        "Teacher-empty view/frame 不影响同一固定窗口的 RGB 指标；该 view 的 teacher-relative error 缺失，几何汇总的 valid-sample count 会相应减少。",
    ])
    for entry in sorted(teacher_exclusions, key=lambda value: (
        value["dataset"], value["eye"], value["camera"] or "",
        value["mode"], value["sample_id"], value["view"]
    )):
        lines.append(
            f"- teacher exclusion: dataset={entry['dataset']}, eye={entry['eye']}, "
            f"camera={entry['camera'] or 'three canonical pairs'}, mode={entry['mode']}, "
            f"sample={entry['sample_id']}, view={entry['view']}, reason={entry['reason']}"
        )
    lines.extend([
        "",
        "每个 selection 的 decode checked/rejected 与 rejected episode IDs SHA256 已记录在数据表；完整排除原因保存在 selection JSON。",
        "",
        "### 固定案例",
        "",
        "共 16 个固定窗口：UMI 8、LIBERO 8；每个窗口保存四个 source position 的原图/重建与几何图，`cases.json` 和每个 PNG 均已在报告生成时核验。",
        "",
        "## 几何口径",
        "",
        "- Mono：DA3 分别推理原图与重建图，报告 `reconstruction_teacher_relative_*`。",
        "- Stereo：decoder 不重建右眼，报告 `depth_head_teacher_relative_*`；不称为 stereo 重建精度。",
        "- 没有独立真实 depth/disparity GT，因此本报告不声称真实几何 accuracy。",
        "",
        "## 阻断与未完成项",
        "",
        "- 每个 selection 的 decode checked/rejected 与 rejected episode IDs SHA256 已记录；完整排除原因保存在 selection JSON。",
        "- HY：通过训练同款 manifest reader 读取；Table014 因当前资产缺失被显式排除，排除数量和 episode IDs hash 写入 selection。",
        "- rFID：按本轮范围暂不执行。",
        "- rFVD：N/A，现有冻结 I3D-FVD 实现不支持本项目原生 4 帧合同；扩帧/插帧会改变评测对象。",
        "- FVMD：N/A，尚无经验证适用于原生 4 帧的冻结实现。",
        "",
        "## 决策",
        "",
        "1. **值得继续，但仍需补 rFID 与 Table014。** 当前结果可作为扩展后的 preliminary Stage A，不能表述成完整最终标准。",
        "2. **最大风险：** 当前几何指标只有 teacher-relative 证据；若误写为真实 depth/disparity accuracy，会得到错误结论。",
        "3. **最缺的关键证据：** 独立真实几何 GT、rFID，以及 Table014 的可读资产。",
        "4. **下一步：** 固定同一 selection 对 baseline/candidate 重跑，并补齐 Table014 后保持其他合同不变复测 HY。",
        "5. **置信度：80%（中等）。** 对已报告数字和可复现合同置信度较高；因 rFID、Table014 与独立几何 GT 缺失，不给高置信度。",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "quality_results": 10, "benchmarks": 2}, indent=2))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("expected one of: selection, preflight, run, benchmark, report")
    command, argv = sys.argv[1], sys.argv[2:]
    commands = {
        "selection": _selection_command,
        "preflight": _preflight_command,
        "run": _run_command,
        "benchmark": _benchmark_command,
        "report": _report_command,
    }
    if command not in commands:
        raise SystemExit(f"unknown command {command!r}")
    commands[command](argv)


if __name__ == "__main__":
    main()
