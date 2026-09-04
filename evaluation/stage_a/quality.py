"""Stage A quality evaluation command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate
from tqdm import tqdm

from stereo_tokenizer.geometry import GeometryMapping

from . import runtime
from .common import (
    _FrozenRAFT,
    _checkpoint_provenance,
    _dataset_provenance,
    _environment_provenance,
    _jsonable,
    _source_provenance,
)
from .data import CanonicalStageADataset
from .metrics import StageA1MetricSuite


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
