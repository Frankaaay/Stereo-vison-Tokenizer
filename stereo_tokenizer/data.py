"""Manifest-v3 dataset and Lightning data module for StereoVAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils import data
from torch.utils.data import default_collate

from .geometry import GeometryMapping
from .lerobot_data import (
    EpisodeSequentialDistributedSampler,
    LeRobotStereoDataset,
    fixed_episode_subset_indices,
)
from .profiling import profile_region
from .mode_sampling import MixedModeBatchSampler, MixedModeDataset


def _profiled_collate(batch):
    with profile_region("stereo/data/collate"):
        return default_collate(batch)


class ModeSubset(data.Dataset):
    """Subset adapter retaining the native get_mode_item contract."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        if not self.indices:
            raise ValueError("mode subset cannot be empty")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.get_mode_item(index, "four_frame")

    def get_mode_item(self, index, temporal_mode):
        return self.dataset.get_mode_item(self.indices[index], temporal_mode)


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
        single_frame_source_index=2,
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
        self.single_frame_source_index = int(single_frame_source_index)

        if not 0 <= self.disparity_min_px < self.disparity_max_px:
            raise ValueError("invalid disparity supervision range")
        if self.lr_error_abs_threshold_px < 0:
            raise ValueError("absolute LR threshold must be non-negative")
        if self.lr_error_relative_threshold < 0:
            raise ValueError("relative LR threshold must be non-negative")
        if not 0 <= self.single_frame_source_index < 4:
            raise ValueError("single-frame source index must be in [0,3]")
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
            return self.get_mode_item(index, "four_frame")

    def get_mode_item(self, index, temporal_mode):
        sample = self._getitem_impl(index)
        if temporal_mode == "four_frame":
            selected = dict(sample)
        elif temporal_mode == "single_frame":
            frame_index = self.single_frame_source_index
            selected = dict(sample)
            selected["video"] = sample["video"][..., frame_index : frame_index + 1, :, :]
            selected["disparity"] = sample["disparity"][
                ..., frame_index : frame_index + 1, :, :
            ]
            selected["valid_mask"] = sample["valid_mask"][
                ..., frame_index : frame_index + 1, :, :
            ]
        else:
            raise ValueError(f"unsupported temporal mode {temporal_mode!r}")
        selected.update(
            {
                "mode_id": f"stereo/{temporal_mode}",
                "eye_mode": "stereo",
                "temporal_mode": temporal_mode,
                "view_count": 3,
                "teacher_kind": "foundation_stereo",
            }
        )
        return selected

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


class HyMonoSmokeDataset(data.Dataset):
    """Immutable raw-RGB Hy cam_high smoke samples with runtime letterboxing."""

    SCHEMA = "hy-mono-cam-high-smoke-v1"

    def __init__(self, manifest_path, cache_root):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if not self.cache_root.is_dir():
            raise FileNotFoundError(self.cache_root)
        self.records = self._read_jsonl(self.manifest_path)
        if len(self.records) != 48:
            raise ValueError("Hy mono smoke manifest must contain exactly 48 samples")
        verified_cache_hashes = {}
        for line_number, record in enumerate(self.records, start=1):
            if record.get("schema") != self.SCHEMA:
                raise ValueError(
                    f"{self.manifest_path}:{line_number}: unsupported mono schema"
                )
            required = (
                "sample_id",
                "rgb_relative_path",
                "rgb_sha256",
                "source_contract_sha256",
                "table_inventory_sha256",
                "start_frame",
                "frame_indices",
                "timestamps_s",
            )
            missing = [key for key in required if not record.get(key)]
            if missing:
                raise ValueError(
                    f"{self.manifest_path}:{line_number}: missing {missing}"
                )
            source_hw = record.get("source_hw")
            if (
                not isinstance(source_hw, list)
                or len(source_hw) != 2
                or any(not isinstance(value, int) or value < 1 for value in source_hw)
            ):
                raise ValueError("Hy mono smoke source_hw must contain positive H,W")
            if record.get("frame_indices") != [
                record["start_frame"] + offset for offset in (0, 3, 6, 9)
            ]:
                raise ValueError("Hy mono smoke frame offsets must be [0,3,6,9]")
            cache_path = StereoManifestDataset._resolve_cache_path(
                self.cache_root, record["rgb_relative_path"]
            )
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            digest = verified_cache_hashes.get(cache_path)
            if digest is None:
                digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
                verified_cache_hashes[cache_path] = digest
            if digest != record["rgb_sha256"]:
                raise ValueError(f"{cache_path}: RGB cache SHA256 mismatch")

    @staticmethod
    def _read_jsonl(path):
        records = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not records:
            raise ValueError(f"manifest is empty: {path}")
        return records

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _read_first_rgb_frame(path, expected_shape):
        """Decode only frame zero from the compressed rgb.npy NPZ member."""
        with zipfile.ZipFile(path, mode="r") as archive:
            with archive.open("rgb.npy", mode="r") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_1_0(stream)
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_2_0(stream)
                    )
                else:
                    raise ValueError(f"{path}: unsupported rgb.npy version {version}")
                if shape != expected_shape:
                    raise ValueError(f"{path}: unexpected rgb.npy shape {shape}")
                if fortran_order or dtype != np.dtype(np.uint8):
                    raise ValueError(f"{path}: rgb.npy must be C-order uint8")
                frame_bytes = int(np.prod(shape[1:], dtype=np.int64))
                payload = stream.read(frame_bytes)
                if len(payload) != frame_bytes:
                    raise ValueError(f"{path}: truncated first RGB frame")
        return np.frombuffer(payload, dtype=np.uint8).reshape((1, *shape[1:])).copy()

    def get_mode_item(self, index, temporal_mode):
        record = self.records[index]
        path = StereoManifestDataset._resolve_cache_path(
            self.cache_root, record["rgb_relative_path"]
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != {
                "rgb",
                "frame_index",
                "timestamp_s",
                "metadata_json",
            }:
                raise ValueError(f"{path}: invalid Hy mono smoke NPZ keys")
            frame_index = cache["frame_index"]
            timestamp_s = cache["timestamp_s"]
            metadata = json.loads(str(cache["metadata_json"]))
        if frame_index.tolist() != record["frame_indices"]:
            raise ValueError(f"{path}: frame indices disagree with manifest")
        if not np.allclose(timestamp_s, record["timestamps_s"], rtol=0, atol=0):
            raise ValueError(f"{path}: timestamps disagree with manifest")
        if metadata.get("sample_id") != record["sample_id"]:
            raise ValueError(f"{path}: metadata sample ID mismatch")
        if metadata.get("source_contract_sha256") != record["source_contract_sha256"]:
            raise ValueError(f"{path}: metadata source contract mismatch")
        source_hw = tuple(int(value) for value in record["source_hw"])
        source_shape = (4, 3, *source_hw)
        if temporal_mode == "single_frame":
            selected_rgb = self._read_first_rgb_frame(path, source_shape)
            selected_frame_index = frame_index[0:1]
            selected_timestamp = timestamp_s[0:1]
        elif temporal_mode == "four_frame":
            with np.load(path, allow_pickle=False) as cache:
                selected_rgb = cache["rgb"]
            if (
                selected_rgb.dtype != np.uint8
                or selected_rgb.shape != source_shape
            ):
                raise ValueError(f"{path}: expected uint8 {source_shape}")
            selected_frame_index = frame_index
            selected_timestamp = timestamp_s
        else:
            raise ValueError(f"unsupported temporal mode {temporal_mode!r}")

        raw_rgb = torch.from_numpy(selected_rgb.copy())
        geometry = GeometryMapping.create(source_hw, source_hw=source_hw)
        letterboxed, non_padding = geometry.student_letterbox(raw_rgb)
        da3_images = geometry.da3_preprocess(raw_rgb)
        video = letterboxed.div(255.0).sub(0.5)
        video = video.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0)
        non_padding = non_padding.permute(1, 0, 2, 3).unsqueeze(0)
        return {
            "video": video,
            "da3_images": da3_images,
            "non_padding_mask": non_padding,
            "geometry_mapping": geometry.to_collatable_metadata(),
            "sample_id": record["sample_id"],
            "episode_id": record["episode_id"],
            "frame_index": torch.from_numpy(selected_frame_index.copy()),
            "timestamp_s": torch.from_numpy(selected_timestamp.copy()),
            "contract_sha256": record["source_contract_sha256"],
            "table_inventory_sha256": record["table_inventory_sha256"],
            "mode_id": f"mono/{temporal_mode}",
            "eye_mode": "mono",
            "temporal_mode": temporal_mode,
            "view_count": 1,
            "teacher_kind": "da3",
        }


class StereoDataModule(pl.LightningDataModule):
    def __init__(self, args, shuffle: bool = True):
        super().__init__()
        self.args = args
        self.shuffle = shuffle
        self._profile_preloaded_train_dataset = None

    def profile_preload_train_dataset(self) -> int:
        if self._profile_preloaded_train_dataset is not None:
            raise RuntimeError("training dataset is already preloaded")
        dataset = self._dataset(True)
        self._profile_preloaded_train_dataset = [
            dataset[index] for index in range(len(dataset))
        ]
        return len(self._profile_preloaded_train_dataset)

    def _stereo_dataset(self, train: bool, split: str | None = None):
        if train and self._profile_preloaded_train_dataset is not None:
            return self._profile_preloaded_train_dataset
        backend = getattr(self.args, "stereo_data_backend", "manifest_v3")
        if backend == "lerobot_online":
            if self.args.train_epoch_repeats != 1:
                raise ValueError(
                    "LeRobot online training requires train_epoch_repeats=1"
                )
            resolved_split = split or ("train" if train else "val")
            dataset = LeRobotStereoDataset(
                self.args.lerobot_episode_manifest,
                self.args.lerobot_dataset_root,
                split=resolved_split,
                expected_rectification_audit_sha256=(
                    self.args.lerobot_rectification_audit_sha256
                ),
                video_cache_capacity=self.args.lerobot_video_cache_capacity,
                maximum_timestamp_error_s=(
                    self.args.lerobot_maximum_timestamp_error_s
                ),
                single_frame_source_index=self.args.single_frame_source_index,
            )
            mixed = bool(getattr(self.args, "four_mode_mixed_training", False))
            if resolved_split == "val" or (mixed and resolved_split == "train"):
                limit = int(
                    self.args.mixed_stereo_sample_limit
                    if mixed
                    else self.args.lerobot_val_sample_limit
                )
                if limit < 1:
                    raise ValueError("LeRobot fixed sample limit must be positive")
                indices = fixed_episode_subset_indices(
                    dataset,
                    limit,
                    seed=int(getattr(self.args, "seed", 1234)),
                )
                dataset = ModeSubset(dataset, indices)
            return dataset
        if backend != "manifest_v3":
            raise ValueError(f"unsupported stereo data backend: {backend}")

        manifest = (
            self.args.stereo_train_manifest
            if train
            else self.args.stereo_val_manifest
        )
        if manifest is None:
            if train:
                raise ValueError("--stereo_train_manifest is required")
            return None
        dataset = StereoManifestDataset(
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
            single_frame_source_index=self.args.single_frame_source_index,
        )
        repeats = int(getattr(self.args, "train_epoch_repeats", 1))
        if train and repeats != 1:
            dataset = data.ConcatDataset([dataset] * repeats)
        return dataset

    def _mono_dataset(self, train: bool):
        manifest = (
            self.args.mono_train_manifest if train else self.args.mono_val_manifest
        )
        if manifest is None:
            return None
        dataset = HyMonoSmokeDataset(manifest, self.args.mono_cache_root)
        mixed = bool(getattr(self.args, "four_mode_mixed_training", False))
        if train and mixed:
            limit = int(self.args.mixed_mono_sample_limit)
            if limit < 1 or limit > len(dataset):
                raise ValueError(
                    "mixed mono sample limit must be in [1, dataset size]"
                )
            dataset = ModeSubset(dataset, range(limit))
        return dataset

    def _dataset(self, train: bool, split: str | None = None):
        if not bool(getattr(self.args, "four_mode_mixed_training", False)):
            return self._stereo_dataset(train, split=split)
        if split is not None:
            raise ValueError("four-mode smoke does not use named dataset splits")
        stereo_dataset = self._stereo_dataset(train)
        mono_dataset = self._mono_dataset(train)
        if stereo_dataset is None and mono_dataset is None and not train:
            return None
        if stereo_dataset is None or mono_dataset is None:
            raise ValueError("four-mode data requires both mono and stereo datasets")
        return MixedModeDataset(
            mono_dataset=mono_dataset,
            stereo_dataset=stereo_dataset,
        )

    def _dataloader(self, train: bool, split: str | None = None):
        dataset = self._dataset(train, split=split)
        if dataset is None:
            return None
        pin_memory = bool(getattr(self.args, "pin_memory", False))
        if hasattr(self.args, "profile_pin_memory"):
            pin_memory = bool(self.args.profile_pin_memory)
        persistent_workers = bool(
            getattr(self.args, "persistent_workers", False)
        )
        if persistent_workers and self.args.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if isinstance(dataset, MixedModeDataset):
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
            batch_sampler = MixedModeBatchSampler(
                dataset.source_lengths,
                batch_size=self.args.batch_size,
                seed=int(self.args.mode_schedule_seed),
                updates_per_epoch=(
                    int(self.args.mode_updates_per_epoch) if train else 4
                ),
                start_update=(
                    int(self.args.mode_schedule_start_update) if train else 0
                ),
                num_replicas=world_size,
                rank=rank,
            )
            return data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=_profiled_collate,
            )
        if isinstance(dataset, LeRobotStereoDataset):
            sampler = EpisodeSequentialDistributedSampler(
                dataset,
                shuffle=train and self.shuffle,
                seed=int(getattr(self.args, "seed", 1234)),
            )
        elif dist.is_initialized():
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
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
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
        if getattr(self.args, "stereo_data_backend", "manifest_v3") == "lerobot_online":
            return self._dataloader(False, split="test")
        return self.val_dataloader()

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--sequence_length", type=int, default=4)
        parser.add_argument("--resolution", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pin_memory", type=int, choices=(0, 1), default=1)
        parser.add_argument(
            "--persistent_workers", type=int, choices=(0, 1), default=1
        )
        parser.add_argument("--train_epoch_repeats", type=int, default=1)
        parser.add_argument("--image_channels", type=int, default=3)
        parser.add_argument(
            "--stereo_data_backend",
            choices=("manifest_v3", "lerobot_online"),
            default="manifest_v3",
        )
        parser.add_argument("--stereo_train_manifest", type=str, default=None)
        parser.add_argument("--stereo_val_manifest", type=str, default=None)
        parser.add_argument("--stereo_rgb_root", type=str, default=None)
        parser.add_argument("--stereo_gt_root", type=str, default=None)
        parser.add_argument(
            "--four_mode_mixed_training", type=int, choices=(0, 1), default=0
        )
        parser.add_argument("--mono_train_manifest", type=str, default=None)
        parser.add_argument("--mono_val_manifest", type=str, default=None)
        parser.add_argument("--mono_cache_root", type=str, default=None)
        parser.add_argument("--mode_schedule_seed", type=int, default=1234)
        parser.add_argument("--mode_updates_per_epoch", type=int, default=0)
        parser.add_argument("--mode_schedule_start_update", type=int, default=0)
        parser.add_argument("--stereo_disparity_min_px", type=float, default=None)
        parser.add_argument("--stereo_disparity_max_px", type=float, default=None)
        parser.add_argument(
            "--stereo_lr_error_abs_threshold_px", type=float, default=None
        )
        parser.add_argument(
            "--stereo_lr_error_relative_threshold", type=float, default=None
        )
        parser.add_argument("--lerobot_episode_manifest", type=str, default=None)
        parser.add_argument("--lerobot_dataset_root", type=str, default=None)
        parser.add_argument(
            "--lerobot_rectification_audit_sha256", type=str, default=None
        )
        parser.add_argument(
            "--lerobot_video_cache_capacity", type=int, default=12
        )
        parser.add_argument(
            "--lerobot_maximum_timestamp_error_s", type=float, default=0.05
        )
        parser.add_argument("--lerobot_val_sample_limit", type=int, default=512)
        parser.add_argument("--mixed_mono_sample_limit", type=int, default=48)
        parser.add_argument("--mixed_stereo_sample_limit", type=int, default=48)
        return parser
