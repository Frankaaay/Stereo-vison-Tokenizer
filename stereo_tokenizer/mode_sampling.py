from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping

from torch.utils import data


MODE_IDS = (
    "mono/single_frame",
    "mono/four_frame",
    "stereo/single_frame",
    "stereo/four_frame",
)


def _stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def mode_order_for_cycle(seed: int, cycle_index: int) -> tuple[str, ...]:
    if cycle_index < 0:
        raise ValueError("cycle_index must be non-negative")
    modes = list(MODE_IDS)
    random.Random(_stable_seed(seed, "mode-cycle", cycle_index)).shuffle(modes)
    return tuple(modes)


def mode_for_update(seed: int, update_index: int) -> str:
    if update_index < 0:
        raise ValueError("update_index must be non-negative")
    cycle_index, offset = divmod(update_index, len(MODE_IDS))
    return mode_order_for_cycle(seed, cycle_index)[offset]


def mode_occurrences_before(seed: int, update_index: int) -> dict[str, int]:
    if update_index < 0:
        raise ValueError("update_index must be non-negative")
    cycles, remainder = divmod(update_index, len(MODE_IDS))
    counts = {mode_id: cycles for mode_id in MODE_IDS}
    for mode_id in mode_order_for_cycle(seed, cycles)[:remainder]:
        counts[mode_id] += 1
    return counts


class MixedModeDataset(data.Dataset):
    """Dispatch mode-tagged indices to mono or stereo native datasets."""

    def __init__(self, *, mono_dataset, stereo_dataset):
        self.datasets = {"mono": mono_dataset, "stereo": stereo_dataset}
        for eye_mode, dataset in self.datasets.items():
            if not hasattr(dataset, "get_mode_item"):
                raise TypeError(f"{eye_mode} dataset must implement get_mode_item")
            if len(dataset) < 1:
                raise ValueError(f"{eye_mode} dataset is empty")

    @property
    def source_lengths(self) -> dict[str, int]:
        return {key: len(value) for key, value in self.datasets.items()}

    def __len__(self):
        return sum(self.source_lengths.values())

    def __getitem__(self, request):
        if not isinstance(request, tuple) or len(request) != 2:
            raise TypeError("mixed-mode dataset index must be (mode_id, sample_index)")
        mode_id, sample_index = request
        if mode_id not in MODE_IDS:
            raise ValueError(f"unsupported mode_id {mode_id!r}")
        eye_mode, temporal_mode = mode_id.split("/", maxsplit=1)
        sample = dict(
            self.datasets[eye_mode].get_mode_item(sample_index, temporal_mode)
        )
        video = sample.get("video")
        expected_views = 1 if eye_mode == "mono" else 3
        expected_eyes = 1 if eye_mode == "mono" else 2
        expected_time = 1 if temporal_mode == "single_frame" else 4
        if video is None or video.ndim != 6:
            raise ValueError("native sample video must use [V,E,C,T,H,W]")
        if tuple(video.shape[:4]) != (
            expected_views,
            expected_eyes,
            3,
            expected_time,
        ):
            raise ValueError(
                f"{mode_id} sample has incompatible video shape {tuple(video.shape)}"
            )
        sample.update(
            {
                "mode_id": mode_id,
                "eye_mode": eye_mode,
                "temporal_mode": temporal_mode,
                "view_count": expected_views,
                "teacher_kind": (
                    "da3" if eye_mode == "mono" else "foundation_stereo"
                ),
            }
        )
        return sample


class MixedModeBatchSampler(data.Sampler):
    """Stateless BS-fixed four-mode sampler with deterministic DDP scheduling."""

    def __init__(
        self,
        source_lengths: Mapping[str, int],
        *,
        batch_size: int,
        seed: int,
        updates_per_epoch: int,
        start_update: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if set(source_lengths) != {"mono", "stereo"}:
            raise ValueError("source_lengths must contain mono and stereo")
        if batch_size < 1 or updates_per_epoch < 1 or start_update < 0:
            raise ValueError("batch size/update counts must be positive")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank")
        self.source_lengths = {key: int(value) for key, value in source_lengths.items()}
        if min(self.source_lengths.values()) < 1:
            raise ValueError("mono and stereo sources must be non-empty")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.updates_per_epoch = int(updates_per_epoch)
        self.start_update = int(start_update)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self._local_indices = {
            eye_mode: list(range(self.rank, length, self.num_replicas))
            for eye_mode, length in self.source_lengths.items()
        }
        if any(not indices for indices in self._local_indices.values()):
            raise ValueError("a DDP rank was assigned no mono or stereo samples")

    def _batch_indices(self, mode_id: str, occurrence: int) -> list[int]:
        eye_mode = mode_id.split("/", maxsplit=1)[0]
        local = self._local_indices[eye_mode]
        start = occurrence * self.batch_size
        output = []
        while len(output) < self.batch_size:
            data_epoch, offset = divmod(start, len(local))
            permutation = list(local)
            random.Random(
                _stable_seed(self.seed, mode_id, "data-epoch", data_epoch)
            ).shuffle(permutation)
            take = min(self.batch_size - len(output), len(local) - offset)
            output.extend(permutation[offset : offset + take])
            start += take
        return output

    def __iter__(self):
        occurrences = mode_occurrences_before(self.seed, self.start_update)
        stop = self.start_update + self.updates_per_epoch
        for update_index in range(self.start_update, stop):
            mode_id = mode_for_update(self.seed, update_index)
            occurrence = occurrences[mode_id]
            occurrences[mode_id] += 1
            yield [
                (mode_id, sample_index)
                for sample_index in self._batch_indices(mode_id, occurrence)
            ]

    def __len__(self):
        return self.updates_per_epoch
