"""Frozen bidirectional FoundationStereo teacher for online disparity targets."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from .profiling import profile_region


CACHE_SCHEMA = "stereo-online-foundation-gt-v1"


class _UnavailableOpen3D(types.ModuleType):
    """Import-only stub for FoundationStereo's unused point-cloud helpers."""

    def __getattr__(self, name):
        raise RuntimeError(
            "Open3D is unavailable in online FoundationStereo inference; "
            f"attempted to access open3d.{name}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FoundationStereoOnlineTeacher:
    """Non-checkpointed frozen FoundationStereo inference wrapper."""

    def __init__(
        self,
        repo,
        checkpoint,
        checkpoint_sha256,
        *,
        device,
        valid_iters,
        pair_microbatch,
    ):
        self.repo = Path(repo).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.checkpoint_sha256 = checkpoint_sha256
        self.device = torch.device(device)
        self.valid_iters = int(valid_iters)
        self.pair_microbatch = int(pair_microbatch)
        if self.valid_iters not in {12, 16, 32}:
            raise ValueError("online FoundationStereo iterations must be 12, 16, or 32")
        if self.pair_microbatch < 1:
            raise ValueError("FoundationStereo pair microbatch must be positive")
        if not self.repo.is_dir() or not self.checkpoint.is_file():
            raise FileNotFoundError("FoundationStereo repo/checkpoint is missing")
        if sha256_file(self.checkpoint) != checkpoint_sha256:
            raise ValueError("FoundationStereo checkpoint SHA256 mismatch")
        self.model, self.config = self._load_model()

    def _load_model(self):
        try:
            import timm
            from omegaconf import OmegaConf
        except ImportError as error:
            raise RuntimeError(
                "online FoundationStereo requires timm and omegaconf"
            ) from error

        original_open3d = sys.modules.get("open3d")
        sys.modules["open3d"] = _UnavailableOpen3D("open3d")
        sys.path.insert(0, str(self.repo))
        try:
            from core.foundation_stereo import FoundationStereo
        finally:
            if sys.path[0] == str(self.repo):
                sys.path.pop(0)
            if original_open3d is None:
                sys.modules.pop("open3d", None)
            else:
                sys.modules["open3d"] = original_open3d

        config = OmegaConf.load(self.checkpoint.parent / "cfg.yaml")
        if "vit_size" not in config:
            config["vit_size"] = "vitl"
        original_create_model = timm.create_model
        original_hub_load = torch.hub.load

        def offline_create_model(model_name, *args, **kwargs):
            if model_name != "edgenext_small":
                raise RuntimeError(f"unexpected timm model request: {model_name}")
            kwargs["pretrained"] = False
            return original_create_model(model_name, *args, **kwargs)

        def offline_hub_load(repo_or_dir, model_name, *args, **kwargs):
            if repo_or_dir != "facebookresearch/dinov2":
                raise RuntimeError(f"unexpected torch.hub request: {repo_or_dir}")
            kwargs["source"] = "local"
            kwargs["pretrained"] = False
            return original_hub_load(
                str(self.repo / "dinov2"), model_name, *args, **kwargs
            )

        timm.create_model = offline_create_model
        torch.hub.load = offline_hub_load
        try:
            model = FoundationStereo(config)
        finally:
            timm.create_model = original_create_model
            torch.hub.load = original_hub_load
        # The full FoundationStereo training payload contains NumPy metadata.
        # Its bytes are trusted only after the frozen SHA256 check in __init__.
        payload = torch.load(
            self.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(payload["model"], strict=True)
        model.requires_grad_(False)
        model.to(self.device)
        model.eval()
        return model, config

    def _infer_microbatch(self, left, right):
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            disparity_left = self.model.forward(
                left, right, iters=self.valid_iters, test_mode=True
            )
            disparity_right = self.model.forward(
                torch.flip(right, dims=[3]),
                torch.flip(left, dims=[3]),
                iters=self.valid_iters,
                test_mode=True,
            )
            disparity_right = torch.flip(disparity_right, dims=[3])
        return disparity_left.float(), disparity_right.float()

    def infer(self, video):
        """Infer bidirectional disparity for [B,V,E,C,T,H,W] RGB in [-.5,.5]."""
        if video.ndim != 7 or video.shape[1:5] != (3, 2, 3, 4):
            raise ValueError(
                "online teacher expects video [B,3,2,3,4,H,W]"
            )
        left = video[:, :, 0].permute(0, 1, 3, 2, 4, 5).contiguous()
        right = video[:, :, 1].permute(0, 1, 3, 2, 4, 5).contiguous()
        batch, views, frames, channels, height, width = left.shape
        left = (left.reshape(-1, channels, height, width) + 0.5) * 255.0
        right = (right.reshape(-1, channels, height, width) + 0.5) * 255.0
        left_outputs = []
        right_outputs = []
        for start in range(0, left.shape[0], self.pair_microbatch):
            stop = start + self.pair_microbatch
            disparity_left, disparity_right = self._infer_microbatch(
                left[start:stop], right[start:stop]
            )
            left_outputs.append(disparity_left)
            right_outputs.append(disparity_right)
        disparity_left = torch.cat(left_outputs)
        disparity_right = torch.cat(right_outputs)
        residual, base_valid = self.lr_consistency(
            disparity_left, disparity_right
        )
        shape = (batch, views, frames, 1, height, width)
        disparity_left = disparity_left.reshape(shape).permute(0, 1, 3, 2, 4, 5)
        residual = residual.reshape(shape).permute(0, 1, 3, 2, 4, 5)
        base_valid = base_valid.reshape(shape).permute(0, 1, 3, 2, 4, 5)
        return disparity_left, residual, base_valid

    @staticmethod
    def lr_consistency(disparity_left, disparity_right):
        if disparity_left.shape != disparity_right.shape:
            raise ValueError("left/right teacher disparity shapes differ")
        if disparity_left.ndim != 4 or disparity_left.shape[1] != 1:
            raise ValueError("teacher disparity must be [N,1,H,W]")
        _, _, height, width = disparity_left.shape
        x = torch.arange(
            width, device=disparity_left.device, dtype=disparity_left.dtype
        ).view(1, 1, width)
        y = torch.arange(
            height, device=disparity_left.device, dtype=disparity_left.dtype
        ).view(1, height, 1)
        x_right = x - disparity_left[:, 0]
        grid_x = 2.0 * x_right / max(width - 1, 1) - 1.0
        grid_y = 2.0 * y.expand_as(grid_x) / max(height - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        sampled_right = F.grid_sample(
            disparity_right,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        finite = torch.isfinite(disparity_left) & torch.isfinite(sampled_right)
        valid = (
            finite
            & (disparity_left > 0)
            & (sampled_right > 0)
            & (x_right[:, None] >= 0)
            & (x_right[:, None] <= width - 1)
        )
        content = torch.zeros(
            (1, 1, height, width),
            device=disparity_left.device,
            dtype=torch.bool,
        )
        content[:, :, 32:224] = True
        valid &= content
        residual = torch.full_like(disparity_left, float("nan"))
        difference = torch.abs(disparity_left - sampled_right)
        residual = torch.where(valid, difference, residual)
        return residual, valid


class OnlineFoundationGTCallback(Callback):
    """Generate disparity targets after device transfer and before each step."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.teacher = None
        self.cache_enabled = bool(args.online_gt_cache_enabled)
        self.cache_root = (
            Path(args.online_gt_cache_root).expanduser().resolve()
            if args.online_gt_cache_root
            else None
        )
        if self.cache_enabled and self.cache_root is None:
            raise ValueError("online GT cache requires --online_gt_cache_root")

    @property
    def state_key(self):
        return (
            f"{self.__class__.__qualname__}:"
            f"{self.args.foundation_stereo_checkpoint_sha256}:"
            f"{self.args.foundation_stereo_valid_iters}"
        )

    def state_dict(self):
        return {}

    def setup(self, trainer, pl_module, stage=None):
        if stage not in (None, "fit") or self.teacher is not None:
            return
        self.teacher = FoundationStereoOnlineTeacher(
            self.args.foundation_stereo_repo,
            self.args.foundation_stereo_checkpoint,
            self.args.foundation_stereo_checkpoint_sha256,
            device=trainer.strategy.root_device,
            valid_iters=self.args.foundation_stereo_valid_iters,
            pair_microbatch=self.args.foundation_stereo_pair_microbatch,
        )

    def teardown(self, trainer, pl_module, stage=None):
        self.teacher = None

    def _cache_path(self, sample_id):
        digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        return self.cache_root / digest[:2] / f"{digest}.npz"

    def _cache_metadata(self, sample_id, contract_sha256):
        return {
            "schema": CACHE_SCHEMA,
            "sample_id": sample_id,
            "contract_sha256": contract_sha256,
            "checkpoint_sha256": self.args.foundation_stereo_checkpoint_sha256,
            "valid_iters": self.args.foundation_stereo_valid_iters,
            "bidirectional": True,
            "lr_consistency": True,
        }

    def _read_cache(self, sample_id, contract_sha256):
        if not self.cache_enabled:
            return None
        path = self._cache_path(sample_id)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != {"disparity", "valid_mask", "metadata_json"}:
                raise ValueError(f"{path}: invalid online GT cache keys")
            metadata = json.loads(str(cache["metadata_json"]))
            if metadata != self._cache_metadata(sample_id, contract_sha256):
                raise ValueError(f"{path}: online GT cache metadata mismatch")
            if cache["disparity"].dtype != np.float32:
                raise ValueError(f"{path}: online GT cache disparity must be float32")
            if cache["valid_mask"].dtype != np.bool_:
                raise ValueError(f"{path}: online GT cache mask must be bool")
            disparity = cache["disparity"].astype(np.float32)
            valid = cache["valid_mask"].astype(np.bool_)
        if disparity.shape != (3, 1, 4, 256, 256) or valid.shape != disparity.shape:
            raise ValueError(f"{path}: invalid online GT cache shape")
        return disparity, valid

    def _write_cache(self, sample_id, contract_sha256, disparity, valid):
        path = self._cache_path(sample_id)
        if path.exists():
            cached = self._read_cache(sample_id, contract_sha256)
            if cached is None:
                raise RuntimeError(f"failed to validate existing cache {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".partial-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(temporary)
        metadata = np.array(
            json.dumps(
                self._cache_metadata(sample_id, contract_sha256),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    disparity=disparity.astype(np.float32),
                    valid_mask=valid.astype(np.bool_),
                    metadata_json=metadata,
                )
            try:
                os.link(temporary, path)
            except FileExistsError:
                cached = self._read_cache(sample_id, contract_sha256)
                if cached is None:
                    raise RuntimeError(f"failed to validate concurrent cache {path}")
            temporary.unlink()
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise

    def _generate(self, batch, pl_module):
        if self.teacher is None:
            raise RuntimeError("online FoundationStereo teacher is not initialized")
        if not self.cache_enabled:
            disparity, residual, base_valid = self.teacher.infer(batch["video"])
            threshold = torch.maximum(
                residual.new_tensor(self.args.stereo_lr_error_abs_threshold_px),
                self.args.stereo_lr_error_relative_threshold * disparity,
            )
            batch["disparity"] = disparity
            batch["valid_mask"] = (
                base_valid
                & torch.isfinite(disparity)
                & torch.isfinite(residual)
                & (disparity >= self.args.stereo_disparity_min_px)
                & (disparity <= self.args.stereo_disparity_max_px)
                & (residual <= threshold)
            )
            return int(batch["video"].shape[0])

        sample_ids = list(batch["sample_id"])
        contract_hashes = list(batch["contract_sha256"])
        results = [None] * len(sample_ids)
        missing = []
        for index, (sample_id, contract_hash) in enumerate(
            zip(sample_ids, contract_hashes)
        ):
            cached = self._read_cache(sample_id, contract_hash)
            if cached is None:
                missing.append(index)
            else:
                results[index] = cached

        if missing:
            missing_tensor = torch.tensor(
                missing, device=batch["video"].device, dtype=torch.long
            )
            video = batch["video"].index_select(0, missing_tensor)
            disparity, residual, base_valid = self.teacher.infer(video)
            threshold = torch.maximum(
                residual.new_tensor(self.args.stereo_lr_error_abs_threshold_px),
                self.args.stereo_lr_error_relative_threshold * disparity,
            )
            valid = (
                base_valid
                & torch.isfinite(disparity)
                & torch.isfinite(residual)
                & (disparity >= self.args.stereo_disparity_min_px)
                & (disparity <= self.args.stereo_disparity_max_px)
                & (residual <= threshold)
            )
            for output_index, batch_index in enumerate(missing):
                item = (
                    disparity[output_index].detach().cpu().numpy(),
                    valid[output_index].detach().cpu().numpy(),
                )
                results[batch_index] = item
                if self.cache_enabled:
                    self._write_cache(
                        sample_ids[batch_index],
                        contract_hashes[batch_index],
                        item[0],
                        item[1],
                    )

        disparity = torch.from_numpy(np.stack([item[0] for item in results])).to(
            device=batch["video"].device, dtype=torch.float32
        )
        valid = torch.from_numpy(np.stack([item[1] for item in results])).to(
            device=batch["video"].device, dtype=torch.bool
        )
        batch["disparity"] = disparity
        batch["valid_mask"] = valid
        return len(missing)

    def _prepare_batch(self, trainer, pl_module, batch, prefix):
        started = time.perf_counter()
        with profile_region("stereo/online_gt/foundation_teacher"):
            generated_count = self._generate(batch, pl_module)
        torch.cuda.synchronize(batch["video"].device)
        seconds = time.perf_counter() - started
        pl_module.log(
            f"{prefix}/online_gt_seconds",
            seconds,
            on_step=prefix == "train",
            on_epoch=prefix != "train",
            sync_dist=True,
        )
        pl_module.log(
            f"{prefix}/online_gt_generated_samples",
            float(generated_count),
            on_step=prefix == "train",
            on_epoch=prefix != "train",
            sync_dist=True,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._prepare_batch(trainer, pl_module, batch, "train")

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._prepare_batch(trainer, pl_module, batch, "val")
