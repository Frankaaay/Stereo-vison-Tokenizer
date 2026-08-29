from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

from torch.utils import data


MODE_IDS = (
    "mono/single_frame",
    "mono/four_frame",
    "stereo/single_frame",
    "stereo/four_frame",
)
DATASET_IDS = ("hy", "libero", "umi")
DEFAULT_MODE_WEIGHTS = {mode_id: 1 for mode_id in MODE_IDS}
DEFAULT_MONO_DATASET_WEIGHTS = {"hy": 9, "libero": 1}


def _stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _normalize_weights(
    weights: Mapping[str, int] | None,
    expected_keys: tuple[str, ...],
    default: Mapping[str, int],
) -> dict[str, int]:
    normalized = dict(default if weights is None else weights)
    if set(normalized) != set(expected_keys):
        raise ValueError(f"weights must contain exactly {expected_keys}")
    if any(
        isinstance(value, bool) or int(value) != value or int(value) <= 0
        for value in normalized.values()
    ):
        raise ValueError("schedule weights must be positive integers")
    divisor = math.gcd(*(int(value) for value in normalized.values()))
    return {key: int(normalized[key]) // divisor for key in expected_keys}


def _weighted_order(
    seed: int,
    cycle_index: int,
    weights: Mapping[str, int],
    *,
    label: str,
) -> tuple[str, ...]:
    if cycle_index < 0:
        raise ValueError("cycle_index must be non-negative")
    items = [key for key, count in weights.items() for _ in range(int(count))]
    random.Random(_stable_seed(seed, label, cycle_index)).shuffle(items)
    return tuple(items)


def mode_order_for_cycle(
    seed: int,
    cycle_index: int,
    mode_weights: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    weights = _normalize_weights(mode_weights, MODE_IDS, DEFAULT_MODE_WEIGHTS)
    return _weighted_order(seed, cycle_index, weights, label="mode-cycle")


def mode_for_update(
    seed: int,
    update_index: int,
    mode_weights: Mapping[str, int] | None = None,
) -> str:
    if update_index < 0:
        raise ValueError("update_index must be non-negative")
    weights = _normalize_weights(mode_weights, MODE_IDS, DEFAULT_MODE_WEIGHTS)
    cycle_size = sum(weights.values())
    cycle_index, offset = divmod(update_index, cycle_size)
    return _weighted_order(seed, cycle_index, weights, label="mode-cycle")[offset]


def mode_occurrences_before(
    seed: int,
    update_index: int,
    mode_weights: Mapping[str, int] | None = None,
) -> dict[str, int]:
    if update_index < 0:
        raise ValueError("update_index must be non-negative")
    weights = _normalize_weights(mode_weights, MODE_IDS, DEFAULT_MODE_WEIGHTS)
    cycle_size = sum(weights.values())
    cycles, remainder = divmod(update_index, cycle_size)
    counts = {mode_id: cycles * weights[mode_id] for mode_id in MODE_IDS}
    for mode_id in _weighted_order(seed, cycles, weights, label="mode-cycle")[:remainder]:
        counts[mode_id] += 1
    return counts


def dataset_for_mode_occurrence(
    seed: int,
    mode_id: str,
    occurrence: int,
    mono_dataset_weights: Mapping[str, int] | None = None,
) -> str:
    if mode_id not in MODE_IDS or occurrence < 0:
        raise ValueError("invalid mode occurrence")
    if mode_id.startswith("stereo/"):
        return "umi"
    weights = _normalize_weights(
        mono_dataset_weights,
        ("hy", "libero"),
        DEFAULT_MONO_DATASET_WEIGHTS,
    )
    cycle_size = sum(weights.values())
    cycle_index, offset = divmod(occurrence, cycle_size)
    return _weighted_order(
        seed,
        cycle_index,
        weights,
        label="mono-dataset-cycle",
    )[offset]


def parse_weight_spec(value: str, keys: tuple[str, ...]) -> dict[str, int]:
    parts = value.split(":")
    if len(parts) != len(keys):
        raise ValueError(f"expected {len(keys)} colon-separated weights")
    try:
        parsed = {key: int(part) for key, part in zip(keys, parts)}
    except ValueError as error:
        raise ValueError("weights must be integers") from error
    return _normalize_weights(parsed, keys, parsed)


def parse_positive_int_spec(value: str, keys: tuple[str, ...]) -> dict[str, int]:
    """Parse an ordered positive-integer contract without weight normalization."""

    parts = value.split(":")
    if len(parts) != len(keys):
        raise ValueError(f"expected {len(keys)} colon-separated positive integers")
    try:
        parsed = {key: int(part) for key, part in zip(keys, parts)}
    except ValueError as error:
        raise ValueError("contract values must be integers") from error
    if any(value < 1 for value in parsed.values()):
        raise ValueError("contract values must be positive integers")
    return parsed


def resolve_mode_int_spec(
    value: str | None,
    *,
    fallback: int,
    keys: tuple[str, ...] = MODE_IDS,
) -> dict[str, int]:
    if fallback < 1:
        raise ValueError("fallback contract value must be positive")
    if value is None:
        return {key: int(fallback) for key in keys}
    return parse_positive_int_spec(value, keys)


@dataclass(frozen=True)
class DatasetSource:
    eye_mode: str
    dataset: data.Dataset


class MixedModeDataset(data.Dataset):
    """Dispatch homogeneous mode/dataset requests to native source datasets."""

    def __init__(self, sources: Mapping[str, DatasetSource]):
        if set(sources) != set(DATASET_IDS):
            raise ValueError(f"sources must contain exactly {DATASET_IDS}")
        self.sources = dict(sources)
        expected_eye_modes = {"hy": "mono", "libero": "mono", "umi": "stereo"}
        for dataset_id, source in self.sources.items():
            if source.eye_mode != expected_eye_modes[dataset_id]:
                raise ValueError(f"{dataset_id} must use {expected_eye_modes[dataset_id]}")
            if not hasattr(source.dataset, "get_mode_item") or len(source.dataset) < 1:
                raise ValueError(
                    f"{dataset_id} must be non-empty and implement get_mode_item"
                )

    @property
    def source_lengths(self) -> dict[str, int]:
        return {key: len(value.dataset) for key, value in self.sources.items()}

    def __len__(self):
        return sum(self.source_lengths.values())

    def __getitem__(self, request):
        if not isinstance(request, tuple) or len(request) != 3:
            raise TypeError("mixed-mode index must be (mode_id, dataset_id, sample_index)")
        mode_id, dataset_id, sample_index = request
        if mode_id not in MODE_IDS or dataset_id not in self.sources:
            raise ValueError("unsupported mixed-mode request")
        eye_mode, temporal_mode = mode_id.split("/", maxsplit=1)
        source = self.sources[dataset_id]
        if source.eye_mode != eye_mode:
            raise ValueError(f"{dataset_id} cannot provide {eye_mode} samples")
        sample = dict(source.dataset.get_mode_item(sample_index, temporal_mode))
        video = sample.get("video")
        expected = (
            1 if eye_mode == "mono" else 3,
            1 if eye_mode == "mono" else 2,
            3,
            1 if temporal_mode == "single_frame" else 4,
        )
        if video is None or video.ndim != 6 or tuple(video.shape[:4]) != expected:
            raise ValueError(f"{mode_id}/{dataset_id} sample has incompatible video shape")
        sample.update(
            mode_id=mode_id,
            dataset_id=dataset_id,
            eye_mode=eye_mode,
            temporal_mode=temporal_mode,
            view_count=expected[0],
            teacher_kind="da3" if eye_mode == "mono" else "foundation_stereo",
        )
        return sample


class MixedModeBatchSampler(data.Sampler):
    """Stateless weighted sampler supporting node-local physical manifests."""

    def __init__(
        self,
        source_lengths: Mapping[str, int],
        *,
        batch_size: int,
        mode_batch_sizes: Mapping[str, int] | None = None,
        mode_accumulation_factors: Mapping[str, int] | None = None,
        seed: int,
        updates_per_epoch: int,
        start_update: int = 0,
        mode_weights: Mapping[str, int] | None = None,
        mono_dataset_weights: Mapping[str, int] | None = None,
        shard_num_replicas: int = 1,
        shard_rank: int = 0,
    ):
        if set(source_lengths) != set(DATASET_IDS):
            raise ValueError(f"source_lengths must contain exactly {DATASET_IDS}")
        if batch_size < 1 or updates_per_epoch < 1 or start_update < 0:
            raise ValueError("batch size/update counts must be positive")
        if shard_num_replicas < 1 or not 0 <= shard_rank < shard_num_replicas:
            raise ValueError("invalid node-local shard rank")
        self.source_lengths = {key: int(value) for key, value in source_lengths.items()}
        if min(self.source_lengths.values()) < 1:
            raise ValueError("all three data sources must be non-empty")
        self.batch_size = int(batch_size)
        self.mode_batch_sizes = self._mode_positive_ints(
            mode_batch_sizes,
            fallback=self.batch_size,
            label="mode batch sizes",
        )
        self.mode_accumulation_factors = self._mode_positive_ints(
            mode_accumulation_factors,
            fallback=1,
            label="mode accumulation factors",
        )
        if any(
            self.mode_accumulation_factors[mode_id] != 1
            for mode_id in MODE_IDS
            if mode_id.startswith("mono/")
        ):
            raise ValueError("mono modes currently require accumulation factor 1")
        self.seed = int(seed)
        self.updates_per_epoch = int(updates_per_epoch)
        self.start_update = int(start_update)
        self.mode_weights = _normalize_weights(
            mode_weights, MODE_IDS, DEFAULT_MODE_WEIGHTS
        )
        self.mono_dataset_weights = _normalize_weights(
            mono_dataset_weights,
            ("hy", "libero"),
            DEFAULT_MONO_DATASET_WEIGHTS,
        )
        self.shard_num_replicas = int(shard_num_replicas)
        self.shard_rank = int(shard_rank)
        self._local_indices = {
            dataset_id: list(range(self.shard_rank, length, self.shard_num_replicas))
            for dataset_id, length in self.source_lengths.items()
        }
        if any(not indices for indices in self._local_indices.values()):
            raise ValueError("a node-local rank was assigned no samples")

    @staticmethod
    def _mode_positive_ints(
        values: Mapping[str, int] | None,
        *,
        fallback: int,
        label: str,
    ) -> dict[str, int]:
        normalized = (
            {mode_id: int(fallback) for mode_id in MODE_IDS}
            if values is None
            else dict(values)
        )
        if set(normalized) != set(MODE_IDS):
            raise ValueError(f"{label} must contain exactly {MODE_IDS}")
        if any(type(value) is not int or value < 1 for value in normalized.values()):
            raise ValueError(f"{label} must be positive integers")
        return {mode_id: normalized[mode_id] for mode_id in MODE_IDS}

    def _batch_indices(
        self,
        mode_id: str,
        dataset_id: str,
        occurrence: int,
        batch_size: int,
    ) -> list[int]:
        local = self._local_indices[dataset_id]
        start = occurrence * batch_size
        output = []
        while len(output) < batch_size:
            data_epoch, offset = divmod(start, len(local))
            permutation = list(local)
            random.Random(
                _stable_seed(self.seed, mode_id, dataset_id, "data-epoch", data_epoch)
            ).shuffle(permutation)
            take = min(batch_size - len(output), len(local) - offset)
            output.extend(permutation[offset : offset + take])
            start += take
        return output

    def __iter__(self):
        mode_counts = mode_occurrences_before(
            self.seed, self.start_update, self.mode_weights
        )
        mono_occurrence = sum(
            count
            for mode_id, count in mode_counts.items()
            if mode_id.startswith("mono/")
        )
        dataset_counts = {dataset_id: 0 for dataset_id in DATASET_IDS}
        for occurrence in range(mono_occurrence):
            dataset_counts[
                dataset_for_mode_occurrence(
                    self.seed,
                    "mono/four_frame",
                    occurrence,
                    self.mono_dataset_weights,
                )
            ] += 1
        dataset_counts["umi"] = sum(
            count * self.mode_accumulation_factors[mode_id]
            for mode_id, count in mode_counts.items()
            if mode_id.startswith("stereo/")
        )
        stop = self.start_update + self.updates_per_epoch
        for update_index in range(self.start_update, stop):
            mode_id = mode_for_update(self.seed, update_index, self.mode_weights)
            mode_occurrence = mode_counts[mode_id]
            mode_counts[mode_id] += 1
            dataset_id = dataset_for_mode_occurrence(
                self.seed,
                mode_id,
                mono_occurrence if mode_id.startswith("mono/") else mode_occurrence,
                self.mono_dataset_weights,
            )
            if mode_id.startswith("mono/"):
                mono_occurrence += 1
            for _ in range(self.mode_accumulation_factors[mode_id]):
                dataset_occurrence = dataset_counts[dataset_id]
                dataset_counts[dataset_id] += 1
                yield [
                    (mode_id, dataset_id, sample_index)
                    for sample_index in self._batch_indices(
                        mode_id,
                        dataset_id,
                        dataset_occurrence,
                        self.mode_batch_sizes[mode_id],
                    )
                ]

    def __len__(self):
        before = mode_occurrences_before(
            self.seed, self.start_update, self.mode_weights
        )
        after = mode_occurrences_before(
            self.seed,
            self.start_update + self.updates_per_epoch,
            self.mode_weights,
        )
        return sum(
            (after[mode_id] - before[mode_id])
            * self.mode_accumulation_factors[mode_id]
            for mode_id in MODE_IDS
        )
