"""CLI for the frozen Stereo Tokenizer Stage A1 evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, default_collate
from tqdm import tqdm

import eval_stereo_vae as legacy

from .stage_a_contract import sha256_file
from .stage_a_data import CanonicalStageADataset, build_canonical_selection
from .stage_a_metrics import StageA1MetricSuite


CHECKPOINT_SHA256 = (
    "a74c3b72b32dfd296157e3b6ad24d0521731517e79e75f22786bca37c47d822e"
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), text=True, stderr=subprocess.STDOUT
    ).strip()


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
    return {
        "python": sys.version,
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _checkpoint_provenance(path: Path, expected_sha256: str) -> dict:
    checkpoint_path = path.expanduser().resolve()
    actual = sha256_file(checkpoint_path)
    if expected_sha256 != CHECKPOINT_SHA256 or actual != expected_sha256:
        raise ValueError(
            f"checkpoint SHA mismatch: frozen={CHECKPOINT_SHA256}, "
            f"requested={expected_sha256}, actual={actual}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    counters = checkpoint.get("stereo_update_counters")
    if not isinstance(counters, dict):
        raise ValueError("checkpoint is missing stereo_update_counters")
    required = {
        "generator_updates": 162500,
        "discriminator_updates": 118500,
        "batch_updates": 162500,
        "four_frame_updates": 81250,
        "single_frame_updates": 81250,
    }
    mismatches = {
        key: (counters.get(key), expected)
        for key, expected in required.items()
        if int(counters.get(key, -1)) != expected
    }
    if int(checkpoint.get("global_step", -1)) != 125000:
        mismatches["global_step"] = (checkpoint.get("global_step"), 125000)
    if mismatches:
        raise ValueError(f"checkpoint training-counter mismatch: {mismatches}")
    return {
        "path": str(checkpoint_path),
        "sha256": actual,
        "global_step": int(checkpoint["global_step"]),
        "epoch": int(checkpoint["epoch"]),
        "stereo_update_counters": _jsonable(counters),
    }


def _selection_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tokenizer_stage_a selection")
    parser.add_argument("--dataset-id", choices=("umi", "hy", "libero"), required=True)
    parser.add_argument("--identity-contract", type=Path, required=True)
    parser.add_argument("--canonical-config-root", type=Path, required=True)
    parser.add_argument("--canonical-loader-root", type=Path, required=True)
    parser.add_argument("--umi-publish-ledger", type=Path)
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
    parser.add_argument("--canonical-loader-root", type=Path, required=True)
    parser.add_argument("--eye-mode", choices=("mono", "stereo"), required=True)
    parser.add_argument("--camera-key")
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv)
    dataset = CanonicalStageADataset(
        args.selection,
        loader_root=args.canonical_loader_root,
        eye_mode=args.eye_mode,
        camera_key=args.camera_key,
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
            {"dataset": dataset.provenance(), "samples": rows},
            indent=2,
            sort_keys=True,
        )
    )


def _run_parser() -> argparse.ArgumentParser:
    parser = legacy.build_parser()
    parser.prog = "tokenizer_stage_a run"
    parser.add_argument("--stage-a-dataset-id", choices=("umi", "hy", "libero"), required=True)
    parser.add_argument("--stage-a-selection", type=Path, required=True)
    parser.add_argument("--canonical-loader-root", type=Path, required=True)
    parser.add_argument("--stage-a-camera-key")
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--rgb-only", action="store_true")
    runtime_required = {
        "stereo_vae_ckpt",
        "output_json",
        "eval_temporal_mode",
        "stage_a_dataset_id",
        "stage_a_selection",
        "canonical_loader_root",
    }
    for action in parser._actions:
        if action.required and action.dest not in runtime_required:
            action.required = False
    return parser


def _hydrate_checkpoint_semantics(args) -> None:
    checkpoint = torch.load(
        args.stereo_vae_ckpt, map_location="cpu", weights_only=False
    )
    checkpoint_args = legacy._checkpoint_model_args(
        checkpoint, args.stereo_vae_ckpt
    )
    for name in legacy.CHECKPOINT_SEMANTIC_FIELDS:
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
    source_indices = legacy.requested_single_frame_source_indices(args)
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
        raise ValueError("mono Stage A requires --stage-a-camera-key")
    if args.eval_eye_mode == "stereo" and args.stage_a_camera_key:
        raise ValueError("stereo Stage A does not accept --stage-a-camera-key")
    if args.num_visualizations:
        raise NotImplementedError(
            "Stage A1 fixed visualization export is not implemented yet"
        )


def _mode_batch(batch: dict, temporal_mode: str, source_index: int | None):
    if temporal_mode == "four_frame":
        return batch
    result = legacy.batch_for_temporal_mode(batch, temporal_mode, source_index)
    result["rgb_valid_mask"] = batch["rgb_valid_mask"][
        ..., source_index : source_index + 1, :, :
    ]
    return result


def _run_command(argv: list[str]) -> None:
    args = _run_parser().parse_args(argv)
    _hydrate_checkpoint_semantics(args)
    _validate_run(args)
    checkpoint = _checkpoint_provenance(
        args.stereo_vae_ckpt, args.checkpoint_sha256
    )
    dataset = CanonicalStageADataset(
        args.stage_a_selection,
        loader_root=args.canonical_loader_root,
        eye_mode=args.eval_eye_mode,
        camera_key=args.stage_a_camera_key,
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
    model = legacy.load_model(args, device)
    if model.perceptual_model is None:
        raise RuntimeError("checkpoint LPIPS model is unavailable")
    teacher = None
    if not args.rgb_only:
        legacy.preflight_teacher_assets(args, (args.eval_eye_mode,))
        teacher = legacy.build_online_teacher(args, args.eval_eye_mode, device)
    suite = StageA1MetricSuite(
        relative_depth_epsilon=args.relative_depth_epsilon
    )
    specs = legacy.evaluation_specs(args, args.eval_eye_mode)
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc=args.eval_eye_mode)):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            if teacher is not None:
                legacy.attach_online_targets(
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
                suite.update(
                    mode_id,
                    mode_batch,
                    output,
                    dataset.view_names,
                    model.perceptual_model,
                )
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
    result = {
        "schema": "stereo-tokenizer-stage-a1-result-v1",
        "status": "smoke" if args.max_batches is not None else "formal",
        "posterior": "mean",
        "quality_precision": "fp32",
        "checkpoint": checkpoint,
        "dataset": dataset.provenance(),
        "teacher": (
            None
            if teacher is None
            else legacy.teacher_provenance(args, args.eval_eye_mode)
        ),
        "requested_modes": [mode_id for mode_id, _, _ in specs],
        "single_frame_source_indices": [0, 1, 2, 3],
        "metrics": metrics,
        "tokenizer_parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "evaluation_state": "frozen_inference_mode",
        },
        "not_applicable": {
            "rfvd": "native four-frame clips are unsupported by the frozen I3D implementation",
            "fvmd": "not validated for native four-frame clips",
        },
        "pending_stage_a2": ["rfid", "raft_warp", "static_flicker", "motion_consistency"],
        "provenance": {
            "cwd": str(Path.cwd()),
            "git_branch": _git("branch", "--show-current"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": _git("status", "--porcelain"),
            "environment": _environment_provenance(),
            "resolved_args": _jsonable(vars(args)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("expected one of: selection, preflight, run")
    command, argv = sys.argv[1], sys.argv[2:]
    commands = {
        "selection": _selection_command,
        "preflight": _preflight_command,
        "run": _run_command,
    }
    if command not in commands:
        raise SystemExit(f"unknown command {command!r}")
    commands[command](argv)


if __name__ == "__main__":
    main()
