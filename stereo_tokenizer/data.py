"""Manifest-v3 dataset and Lightning data module for StereoVAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils import data
from torch.utils.data import default_collate

from .profiling import profile_region


def _profiled_collate(batch):
    with profile_region("stereo/data/collate"):
        return default_collate(batch)


class StereoManifestDataset(data.Dataset):
    """Structured stereo samples backed by independent RGB and GT caches."""

    VIEWS = ("head", "lefthand", "righthand")
    RGB_SHAPE = (3, 2, 3, 4, 256, 256)
    GT_SHAPE = (4, 3, 256, 256)
    REQUIRED_GT_KEYS = {
        "disparity_left",
        "lr_error_px",
        "base_valid_mask",
        "fx",
        "baseline_m",
    }

    def __init__(
        self,
        manifest_path,
        rgb_root,
        gt_root,
        *,
        disparity_min_px,
        disparity_max_px,
        lr_error_abs_threshold_px,
        lr_error_relative_threshold,
    ):
        super().__init__()
        resolved_parameters = {
            "manifest_path": manifest_path,
            "rgb_root": rgb_root,
            "gt_root": gt_root,
            "disparity_min_px": disparity_min_px,
            "disparity_max_px": disparity_max_px,
            "lr_error_abs_threshold_px": lr_error_abs_threshold_px,
            "lr_error_relative_threshold": lr_error_relative_threshold,
        }
        missing = [
            name for name, value in resolved_parameters.items() if value is None
        ]
        if missing:
            raise ValueError(
                "Stereo Manifest loader requires explicit values for "
                + ", ".join(missing)
            )

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.rgb_root = Path(rgb_root).expanduser().resolve()
        self.gt_root = Path(gt_root).expanduser().resolve()
        self.disparity_min_px = float(disparity_min_px)
        self.disparity_max_px = float(disparity_max_px)
        self.lr_error_abs_threshold_px = float(lr_error_abs_threshold_px)
        self.lr_error_relative_threshold = float(lr_error_relative_threshold)

        if not 0 <= self.disparity_min_px < self.disparity_max_px:
            raise ValueError("invalid disparity supervision range")
        if self.lr_error_abs_threshold_px < 0:
            raise ValueError("absolute LR threshold must be non-negative")
        if self.lr_error_relative_threshold < 0:
            raise ValueError("relative LR threshold must be non-negative")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if not self.rgb_root.is_dir():
            raise FileNotFoundError(self.rgb_root)
        if not self.gt_root.is_dir():
            raise FileNotFoundError(self.gt_root)

        self.records = self._read_manifest(self.manifest_path)

    @staticmethod
    def _read_manifest(path: Path):
        records = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from error
                if record.get("manifest_version") != 3:
                    raise ValueError(
                        f"{path}:{line_number}: expected manifest_version=3"
                    )
                if record.get("rgb_cache_schema") != "stereo-rgb-cache-v1":
                    raise ValueError(
                        f"{path}:{line_number}: unsupported RGB cache schema"
                    )
                for key in ("sample_id", "rgb_relative_path", "gt_relative_path"):
                    if not record.get(key):
                        raise ValueError(f"{path}:{line_number}: missing {key}")
                records.append(record)
        if not records:
            raise ValueError(f"manifest is empty: {path}")
        return records

    @staticmethod
    def _resolve_cache_path(root: Path, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError(f"cache path must be relative: {relative}")
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"cache path escapes root: {relative}")
        return resolved

    @staticmethod
    def _load_npz(path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            with np.load(path, allow_pickle=False) as cache:
                return {key: cache[key] for key in cache.files}
        except (OSError, ValueError) as error:
            raise ValueError(f"failed to read cache {path}: {error}") from error

    @staticmethod
    def _content_mask(record):
        preprocessing = record.get("preprocessing", {})
        if preprocessing.get("output_size_hw") != [256, 256]:
            raise ValueError(
                f"{record['sample_id']}: expected output_size_hw=[256,256]"
            )
        padding = preprocessing.get("padding_ltrb")
        if padding is None or len(padding) != 4:
            raise ValueError(f"{record['sample_id']}: invalid padding_ltrb")
        left, top, right, bottom = (int(value) for value in padding)
        if min(left, top, right, bottom) < 0:
            raise ValueError(f"{record['sample_id']}: negative padding")
        if left + right >= 256 or top + bottom >= 256:
            raise ValueError(f"{record['sample_id']}: padding removes all content")
        mask = np.zeros((256, 256), dtype=np.bool_)
        mask[top : 256 - bottom, left : 256 - right] = True
        return mask

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with profile_region("stereo/data/getitem"):
            return self._getitem_impl(index)

    def _getitem_impl(self, index):
        record = self.records[index]
        rgb_path = self._resolve_cache_path(
            self.rgb_root, record["rgb_relative_path"]
        )
        gt_path = self._resolve_cache_path(
            self.gt_root, record["gt_relative_path"]
        )
        with profile_region("stereo/data/rgb_npz_read_decompress"):
            rgb_cache = self._load_npz(rgb_path)
        with profile_region("stereo/data/gt_npz_read_decompress"):
            gt_cache = self._load_npz(gt_path)

        with profile_region("stereo/data/numpy_processing_and_tensor_conversion"):
            if set(rgb_cache) != {"rgb"}:
                raise ValueError(f"{rgb_path}: expected only the rgb array")
            rgb = rgb_cache["rgb"]
            if rgb.shape != self.RGB_SHAPE or rgb.dtype != np.uint8:
                raise ValueError(
                    f"{rgb_path}: expected uint8 {self.RGB_SHAPE}, "
                    f"got {rgb.dtype} {rgb.shape}"
                )

            missing = self.REQUIRED_GT_KEYS - set(gt_cache)
            if missing:
                raise ValueError(f"{gt_path}: missing GT arrays {sorted(missing)}")
            disparity = gt_cache["disparity_left"].astype(np.float32)
            lr_error = gt_cache["lr_error_px"].astype(np.float32)
            base_valid = gt_cache["base_valid_mask"].astype(np.bool_)
            if (
                disparity.shape != self.GT_SHAPE
                or lr_error.shape != self.GT_SHAPE
                or base_valid.shape != self.GT_SHAPE
            ):
                raise ValueError(f"{gt_path}: unexpected dense GT shape")

            fx = gt_cache["fx"].astype(np.float32)
            baseline_m = gt_cache["baseline_m"].astype(np.float32)
            if fx.shape != (3,) or baseline_m.shape != (3,):
                raise ValueError(f"{gt_path}: expected fx/baseline_m shape [3]")
            if not np.isfinite(fx).all() or not np.isfinite(baseline_m).all():
                raise ValueError(f"{gt_path}: non-finite calibration")
            if (fx <= 0).any() or (baseline_m <= 0).any():
                raise ValueError(f"{gt_path}: non-positive calibration")

            content = self._content_mask(record)[None, None]
            lr_threshold = np.maximum(
                self.lr_error_abs_threshold_px,
                self.lr_error_relative_threshold * disparity,
            )
            valid = (
                content
                & base_valid
                & np.isfinite(disparity)
                & np.isfinite(lr_error)
                & (disparity >= self.disparity_min_px)
                & (disparity <= self.disparity_max_px)
                & (lr_error <= lr_threshold)
            )

            disparity = np.transpose(disparity, (1, 0, 2, 3))[:, None]
            valid = np.transpose(valid, (1, 0, 2, 3))[:, None]
            video = torch.from_numpy(rgb.copy()).float().div_(255.0).sub_(0.5)
            return {
                "video": video,
                "disparity": torch.from_numpy(disparity.copy()),
                "valid_mask": torch.from_numpy(valid.copy()),
                "fx": torch.from_numpy(fx.copy()),
                "baseline_m": torch.from_numpy(baseline_m.copy()),
                "sample_id": record["sample_id"],
                "episode_id": record.get("episode_id", ""),
            }


class StereoDataModule(pl.LightningDataModule):
    def __init__(self, args, shuffle: bool = True):
        super().__init__()
        self.args = args
        self.shuffle = shuffle

    def _dataset(self, train: bool):
        manifest = (
            self.args.stereo_train_manifest
            if train
            else self.args.stereo_val_manifest
        )
        if manifest is None:
            if train:
                raise ValueError("--stereo_train_manifest is required")
            return None
        return StereoManifestDataset(
            manifest,
            self.args.stereo_rgb_root,
            self.args.stereo_gt_root,
            disparity_min_px=self.args.stereo_disparity_min_px,
            disparity_max_px=self.args.stereo_disparity_max_px,
            lr_error_abs_threshold_px=(
                self.args.stereo_lr_error_abs_threshold_px
            ),
            lr_error_relative_threshold=(
                self.args.stereo_lr_error_relative_threshold
            ),
        )

    def _dataloader(self, train: bool):
        dataset = self._dataset(train)
        if dataset is None:
            return None
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=train and self.shuffle,
            )
        else:
            sampler = None
        return data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            collate_fn=_profiled_collate,
            sampler=sampler,
            shuffle=sampler is None and train and self.shuffle,
            drop_last=train,
        )

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--sequence_length", type=int, default=4)
        parser.add_argument("--resolution", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--image_channels", type=int, default=3)
        parser.add_argument("--stereo_train_manifest", type=str, default=None)
        parser.add_argument("--stereo_val_manifest", type=str, default=None)
        parser.add_argument("--stereo_rgb_root", type=str, default=None)
        parser.add_argument("--stereo_gt_root", type=str, default=None)
        parser.add_argument("--stereo_disparity_min_px", type=float, default=None)
        parser.add_argument("--stereo_disparity_max_px", type=float, default=None)
        parser.add_argument(
            "--stereo_lr_error_abs_threshold_px", type=float, default=None
        )
        parser.add_argument(
            "--stereo_lr_error_relative_threshold", type=float, default=None
        )
        return parser
