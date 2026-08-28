"""StereoVAE data module for legacy evaluation and three-source pretraining."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils import data
from torch.utils.data import default_collate

from .lerobot_data import (
    EpisodeSequentialDistributedSampler,
    LeRobotStereoDataset,
    fixed_episode_subset_indices,
)
from .mode_sampling import (
    MODE_IDS,
    DatasetSource,
    MixedModeBatchSampler,
    MixedModeDataset,
    parse_weight_spec,
)
from .pretrain_data import HyLanceMonoDataset, LiberoMonoDataset
from .profiling import profile_region


def _profiled_collate(batch):
    with profile_region("stereo/data/collate"):
        return default_collate(batch)


def _load_root_aliases(value: str | None, argument: str) -> dict[str, str]:
    """Load an alias map from inline JSON or a node-local JSON file."""
    if not value:
        raise ValueError(f"{argument} is required")
    candidate = Path(value).expanduser()
    if not value.lstrip().startswith("{") and candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{argument} must be a JSON object or path to a JSON file"
            ) from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{argument} must contain a non-empty JSON object")
    aliases = {}
    for alias, root in payload.items():
        if not isinstance(alias, str) or not alias or not isinstance(root, str) or not root:
            raise ValueError(f"{argument} aliases and paths must be non-empty strings")
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        aliases[alias] = str(path)
    return aliases


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


class StereoDataModule(pl.LightningDataModule):
    def __init__(self, args, shuffle: bool = True):
        super().__init__()
        self.args = args
        self.shuffle = shuffle

    def _legacy_stereo_dataset(self, train: bool, split: str | None = None):
        """Preserve the existing LeRobot evaluation path outside pretraining."""
        if self.args.train_epoch_repeats != 1:
            raise ValueError("LeRobot online training requires train_epoch_repeats=1")
        resolved_split = split or ("train" if train else "val")
        dataset = LeRobotStereoDataset(
            self.args.lerobot_episode_manifest,
            self.args.lerobot_dataset_root,
            split=resolved_split,
            expected_rectification_audit_sha256=self.args.lerobot_rectification_audit_sha256,
            video_cache_capacity=self.args.lerobot_video_cache_capacity,
            maximum_timestamp_error_s=self.args.lerobot_maximum_timestamp_error_s,
            single_frame_source_index=self.args.single_frame_source_index,
        )
        if not train and split is None:
            indices = fixed_episode_subset_indices(
                dataset,
                int(self.args.lerobot_val_sample_limit),
                seed=int(getattr(self.args, "seed", 1234)),
            )
            dataset = ModeSubset(dataset, indices)
        return dataset

    def _pretrain_dataset(self, train: bool):
        split = "train" if train else "val"
        common = {
            "split": split,
            "single_frame_source_index": self.args.single_frame_source_index,
        }
        sources = {
            "hy": DatasetSource(
                "mono",
                HyLanceMonoDataset(
                    self.args.hy_manifest,
                    _load_root_aliases(self.args.hy_root_aliases, "--hy_root_aliases"),
                    **common,
                ),
            ),
            "libero": DatasetSource(
                "mono",
                LiberoMonoDataset(
                    self.args.libero_manifest,
                    _load_root_aliases(
                        self.args.libero_root_aliases, "--libero_root_aliases"
                    ),
                    video_cache_capacity=self.args.lerobot_video_cache_capacity,
                    maximum_timestamp_error_s=self.args.lerobot_maximum_timestamp_error_s,
                    **common,
                ),
            ),
            "umi": DatasetSource(
                "stereo",
                LeRobotStereoDataset(
                    self.args.umi_manifest,
                    self.args.umi_dataset_root,
                    expected_rectification_audit_sha256=(
                        self.args.umi_rectification_audit_sha256
                    ),
                    video_cache_capacity=self.args.lerobot_video_cache_capacity,
                    maximum_timestamp_error_s=(
                        self.args.lerobot_maximum_timestamp_error_s
                    ),
                    **common,
                ),
            ),
        }
        return MixedModeDataset(sources)

    def _dataset(self, train: bool, split: str | None = None):
        if not bool(getattr(self.args, "four_mode_mixed_training", False)):
            return self._legacy_stereo_dataset(train, split=split)
        if split is not None:
            raise ValueError("three-source pretraining does not expose a named test split")
        return self._pretrain_dataset(train)

    def _local_shard(self) -> tuple[int, int]:
        """Shard within one node; manifests already select the physical node."""
        configured_devices = int(getattr(self.args, "devices", 1))
        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", str(configured_devices))
        )
        if local_world_size < 1:
            raise ValueError("invalid LOCAL_RANK/LOCAL_WORLD_SIZE")
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        elif dist.is_available() and dist.is_initialized():
            local_rank = int(dist.get_rank()) % local_world_size
        else:
            local_rank = 0
        if not 0 <= local_rank < local_world_size:
            raise ValueError("invalid LOCAL_RANK/LOCAL_WORLD_SIZE")
        return local_world_size, local_rank

    def _dataloader(self, train: bool, split: str | None = None):
        dataset = self._dataset(train, split=split)
        pin_memory = bool(getattr(self.args, "pin_memory", False))
        persistent_workers = bool(getattr(self.args, "persistent_workers", False))
        if persistent_workers and self.args.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        loader_kwargs = {}
        if self.args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(
                getattr(self.args, "prefetch_factor", 2)
            )
        if isinstance(dataset, MixedModeDataset):
            local_world_size, local_rank = self._local_shard()
            mode_weights = parse_weight_spec(self.args.mode_update_weights, MODE_IDS)
            mono_weights = parse_weight_spec(
                self.args.mono_dataset_weights, ("hy", "libero")
            )
            batch_sampler = MixedModeBatchSampler(
                dataset.source_lengths,
                batch_size=self.args.batch_size,
                seed=int(self.args.mode_schedule_seed),
                updates_per_epoch=(
                    int(self.args.mode_updates_per_epoch)
                    if train
                    else int(self.args.validation_mode_updates)
                ),
                start_update=(int(self.args.mode_schedule_start_update) if train else 0),
                mode_weights=mode_weights,
                mono_dataset_weights=mono_weights,
                shard_num_replicas=local_world_size,
                shard_rank=local_rank,
            )
            return data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=_profiled_collate,
                **loader_kwargs,
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
            **loader_kwargs,
        )

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self._dataloader(False, split="test")

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--sequence_length", type=int, default=4)
        parser.add_argument("--resolution", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--prefetch_factor", type=int, default=2)
        parser.add_argument("--pin_memory", type=int, choices=(0, 1), default=1)
        parser.add_argument("--persistent_workers", type=int, choices=(0, 1), default=1)
        parser.add_argument("--train_epoch_repeats", type=int, default=1)
        parser.add_argument("--image_channels", type=int, default=3)
        parser.add_argument("--four_mode_mixed_training", type=int, choices=(0, 1), default=0)
        parser.add_argument("--hy_manifest", type=str, default=None)
        parser.add_argument("--hy_root_aliases", type=str, default=None)
        parser.add_argument("--libero_manifest", type=str, default=None)
        parser.add_argument("--libero_root_aliases", type=str, default=None)
        parser.add_argument("--umi_manifest", type=str, default=None)
        parser.add_argument("--umi_dataset_root", type=str, default=None)
        parser.add_argument("--umi_rectification_audit_sha256", type=str, default=None)
        parser.add_argument("--mode_schedule_seed", type=int, default=1234)
        parser.add_argument("--mode_update_weights", type=str, default="35:35:15:15")
        parser.add_argument("--mono_dataset_weights", type=str, default="9:1")
        parser.add_argument("--node_manifest_contracts", type=str, default=None)
        parser.add_argument("--mode_updates_per_epoch", type=int, default=0)
        parser.add_argument("--validation_mode_updates", type=int, default=20)
        parser.add_argument("--mode_schedule_start_update", type=int, default=0)
        parser.add_argument("--stereo_disparity_min_px", type=float, default=None)
        parser.add_argument("--stereo_disparity_max_px", type=float, default=None)
        parser.add_argument("--stereo_lr_error_abs_threshold_px", type=float, default=None)
        parser.add_argument("--stereo_lr_error_relative_threshold", type=float, default=None)
        parser.add_argument("--lerobot_episode_manifest", type=str, default=None)
        parser.add_argument("--lerobot_dataset_root", type=str, default=None)
        parser.add_argument("--lerobot_rectification_audit_sha256", type=str, default=None)
        parser.add_argument("--lerobot_video_cache_capacity", type=int, default=12)
        parser.add_argument("--lerobot_maximum_timestamp_error_s", type=float, default=0.05)
        parser.add_argument("--lerobot_val_sample_limit", type=int, default=512)
        return parser
