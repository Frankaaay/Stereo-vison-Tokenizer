#!/usr/bin/env python3
"""Write a fail-closed manifest for a built FoundationStereo TensorRT engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import tensorrt as trt


SCHEMA = "foundation-stereo-tensorrt-engine-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_sha256(path, expected, label):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return path, actual


def command_output(arguments, cwd=None):
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def dtype_name(dtype):
    if dtype == trt.float16:
        return "float16"
    if dtype == trt.float32:
        return "float32"
    raise ValueError(f"unsupported TensorRT IO dtype {dtype}")


def binding(engine, name, expected_mode, expected_shape):
    mode = engine.get_tensor_mode(name)
    if mode != expected_mode:
        raise ValueError(f"TensorRT binding {name} mode mismatch")
    if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
        raise ValueError(f"TensorRT binding {name} is not device IO")
    if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
        raise ValueError(f"TensorRT binding {name} is not linear")
    shape = [int(value) for value in engine.get_tensor_shape(name)]
    if shape != expected_shape:
        raise ValueError(f"TensorRT binding {name} shape mismatch: {shape}")
    return {
        "name": name,
        "mode": (
            "input" if expected_mode == trt.TensorIOMode.INPUT else "output"
        ),
        "dtype": dtype_name(engine.get_tensor_dtype(name)),
        "shape": shape,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation-stereo-repo", type=Path, required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--left-binding", default="left")
    parser.add_argument("--right-binding", default="right")
    parser.add_argument("--output-binding", default="disp")
    parser.add_argument("--onnx-export-command-file", type=Path, required=True)
    parser.add_argument("--trtexec-build-command-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    repo = args.foundation_stereo_repo.expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    repo_sha = command_output(["git", "rev-parse", "HEAD"], cwd=repo)
    if repo_sha != args.expected_repo_sha:
        raise ValueError("FoundationStereo repository SHA mismatch")
    checkpoint, checkpoint_sha = checked_sha256(
        args.checkpoint,
        args.expected_checkpoint_sha256,
        "FoundationStereo checkpoint",
    )
    config, config_sha = checked_sha256(args.config, None, "config")
    onnx, onnx_sha = checked_sha256(args.onnx, None, "ONNX")
    engine_path, engine_sha = checked_sha256(args.engine, None, "engine")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("failed to deserialize TensorRT engine")
    if engine.num_optimization_profiles != 1:
        raise ValueError("TensorRT engine must contain exactly one profile")
    expected_names = {
        args.left_binding,
        args.right_binding,
        args.output_binding,
    }
    actual_names = {
        engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
    }
    if actual_names != expected_names:
        raise ValueError(
            f"TensorRT engine IO names {sorted(actual_names)} do not match requested"
        )
    expected_profile = (
        (1, 3, 256, 256),
        (48, 3, 256, 256),
        (48, 3, 256, 256),
    )
    for name in (args.left_binding, args.right_binding):
        actual_profile = tuple(
            tuple(int(value) for value in shape)
            for shape in engine.get_tensor_profile_shape(name, 0)
        )
        if actual_profile != expected_profile:
            raise ValueError(f"TensorRT binding {name} profile mismatch")

    onnx_command = args.onnx_export_command_file.read_text(
        encoding="utf-8"
    ).strip()
    trtexec_command = args.trtexec_build_command_file.read_text(
        encoding="utf-8"
    ).strip()
    if not onnx_command or not trtexec_command:
        raise ValueError("export and build command files must be non-empty")
    driver = command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "--id=0",
        ]
    ).splitlines()[0]
    payload = {
        "schema": SCHEMA,
        "artifacts": {
            "foundation_stereo_repo_sha": repo_sha,
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            "onnx_sha256": onnx_sha,
            "engine_sha256": engine_sha,
        },
        "paths": {
            "foundation_stereo_repo": str(repo),
            "checkpoint": str(checkpoint),
            "config": str(config),
            "onnx": str(onnx),
            "engine": str(engine_path),
        },
        "build": {
            "height": 256,
            "width": 256,
            "valid_iters": 32,
            "opset": 16,
            "precision": "fp16",
            "xformers_disabled": True,
            "input_layout": "NCHW",
            "input_range": [0.0, 255.0],
            "output_semantics": "left_positive_disparity_px",
            "batch_profile": {"min": 1, "opt": 48, "max": 48},
        },
        "bindings": {
            "left": binding(
                engine,
                args.left_binding,
                trt.TensorIOMode.INPUT,
                [-1, 3, 256, 256],
            ),
            "right": binding(
                engine,
                args.right_binding,
                trt.TensorIOMode.INPUT,
                [-1, 3, 256, 256],
            ),
            "disparity": binding(
                engine,
                args.output_binding,
                trt.TensorIOMode.OUTPUT,
                [-1, 1, 256, 256],
            ),
        },
        "environment": {
            "tensorrt": str(trt.__version__),
            "cuda": str(torch.version.cuda),
            "driver": driver,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "commands": {
            "onnx_export": onnx_command,
            "trtexec_build": trtexec_command,
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(output), "sha256": sha256_file(output)}))


if __name__ == "__main__":
    main()
