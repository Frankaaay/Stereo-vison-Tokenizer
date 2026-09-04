"""Shared immutable provenance and metric-backbone helpers for Stage A."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import torch

from .contract import sha256_file
from .metrics import (
    DYNAMIC_FLOW_MIN_PX,
    FLOW_FB_ABSOLUTE_THRESHOLD_PX2,
    FLOW_FB_RELATIVE_THRESHOLD,
    STATIC_FLOW_MAX_PX,
)


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
