"""LeRobot-v3 episode manifest and online StereoVAE input pipeline."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils import data

from .profiling import profile_region


SCHEMA = "lerobot-stereo-episode-v1"
VIEWS = ("head", "lefthand", "righthand")
EYES = ("left", "right")
VIDEO_KEYS = {
    ("head", "left"): "observation.images.head_left",
    ("head", "right"): "observation.images.head_right",
    ("lefthand", "left"): "observation.images.left_wrist_left",
    ("lefthand", "right"): "observation.images.left_wrist_right",
    ("righthand", "left"): "observation.images.right_wrist_left",
    ("righthand", "right"): "observation.images.right_wrist_right",
}
FRAME_OFFSETS = (0, 3, 6, 9)
START_STRIDE = 12
FPS = 30
SOURCE_HW = (480, 640)
RESIZE_HW = (192, 256)
OUTPUT_HW = (256, 256)
PADDING_LTRB = (0, 32, 0, 32)
CANONICAL_STORED_HW = (256, 256)
CANONICAL_STORED_TRANSFORM = "source_640x480_scale_0.4_pad_y32"
CALIBRATION_CATALOG_SCHEMA = "lerobot-stereo-calibration-catalog-v1"
SUPPORTED_DISTORTION_MODELS = {"plumb_bob", "rational_polynomial"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_count(length: int) -> int:
    """Count four-frame samples using offsets [0,3,6,9] and stride 12."""
    if length < 1:
        raise ValueError("episode length must be positive")
    return max(0, (int(length) - 1 - FRAME_OFFSETS[-1]) // START_STRIDE + 1)


def _matrix(values, shape, name):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape == (shape[0] * shape[1],):
        matrix = matrix.reshape(shape)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return matrix


def validate_calibration(calibration: dict, episode_id: str) -> None:
    for view in VIEWS:
        pair = calibration.get(view)
        if not isinstance(pair, dict):
            raise ValueError(f"{episode_id}: missing calibration for {view}")
        for eye in EYES:
            camera = pair.get(eye)
            if not isinstance(camera, dict):
                raise ValueError(f"{episode_id}: missing {view}/{eye} calibration")
            _matrix(camera.get("K"), (3, 3), f"{view}/{eye} K")
            distortion = np.asarray(camera.get("D"), dtype=np.float64)
            if (
                distortion.ndim != 1
                or distortion.size not in {4, 5, 8, 12, 14}
                or not np.isfinite(distortion).all()
            ):
                raise ValueError(f"{episode_id}: invalid {view}/{eye} D")
            if camera.get("distortion_model") not in SUPPORTED_DISTORTION_MODELS:
                raise ValueError(
                    f"{episode_id}: unsupported {view}/{eye} distortion model"
                )
            if (int(camera.get("height", -1)), int(camera.get("width", -1))) != (
                SOURCE_HW
            ):
                raise ValueError(
                    f"{episode_id}: unexpected {view}/{eye} calibration size"
                )
            _matrix(camera.get("R"), (3, 3), f"{view}/{eye} R")
            _matrix(camera.get("P"), (3, 4), f"{view}/{eye} P")
        left_p = _matrix(pair["left"]["P"], (3, 4), f"{view}/left P")
        right_p = _matrix(pair["right"]["P"], (3, 4), f"{view}/right P")
        if left_p[0, 0] <= 0 or right_p[0, 0] <= 0:
            raise ValueError(f"{episode_id}: non-positive focal length for {view}")
        baseline = -right_p[0, 3] / right_p[0, 0]
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError(f"{episode_id}: invalid baseline for {view}")


@dataclass(frozen=True)
class EpisodeSpan:
    record_index: int
    first_sample: int
    sample_count: int
    shard_id: str


class _AVContainerCache:
    """Worker-local LRU of PyAV containers for timestamp-addressed MP4 reads."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("video cache capacity must be positive")
        self.capacity = capacity
        self._containers = OrderedDict()

    def _open(self, path: Path):
        try:
            import av
        except ImportError as error:
            raise RuntimeError(
                "LeRobot MP4 loading requires PyAV in the training runtime"
            ) from error
        container = av.open(str(path), mode="r")
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            container.close()
            raise ValueError(f"{path}: expected exactly one video stream")
        return container, streams[0]

    def get(self, path: Path):
        key = str(path)
        cached = self._containers.pop(key, None)
        if cached is None:
            cached = self._open(path)
        self._containers[key] = cached
        while len(self._containers) > self.capacity:
            _, (container, _) = self._containers.popitem(last=False)
            container.close()
        return cached

    def close(self):
        for container, _ in self._containers.values():
            container.close()
        self._containers.clear()


class LeRobotStereoDataset(data.Dataset):
    """Four-frame samples decoded online from an episode-level LeRobot index."""

    def __init__(
        self,
        manifest_path,
        dataset_root,
        *,
        split: str,
        expected_rectification_audit_sha256: str,
        video_cache_capacity: int = 12,
        maximum_timestamp_error_s: float = 0.05,
        single_frame_source_index: int = 2,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.expected_rectification_audit_sha256 = (
            expected_rectification_audit_sha256
        )
        self.video_cache_capacity = int(video_cache_capacity)
        self.maximum_timestamp_error_s = float(maximum_timestamp_error_s)
        self.single_frame_source_index = int(single_frame_source_index)
        self._video_cache = None
        self._rectification_maps = {}

        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(self.dataset_root)
        if len(expected_rectification_audit_sha256) != 64:
            raise ValueError("a full rectification audit SHA256 is required")
        if self.maximum_timestamp_error_s <= 0:
            raise ValueError("maximum timestamp error must be positive")
        if not 0 <= self.single_frame_source_index < len(FRAME_OFFSETS):
            raise ValueError("single-frame source index is out of range")

        self.records = self._read_manifest()
        self._validate_stored_image_assets()
        self.episode_spans = []
        self._ends = []
        sample_count = 0
        for record_index, record in enumerate(self.records):
            count = int(record["window_count"])
            if count != window_count(int(record["length"])):
                raise ValueError(
                    f"{record['episode_id']}: inconsistent window_count={count}"
                )
            self.episode_spans.append(
                EpisodeSpan(
                    record_index=record_index,
                    first_sample=sample_count,
                    sample_count=count,
                    shard_id=record["shard_id"],
                )
            )
            sample_count += count
            self._ends.append(sample_count)
        self.sample_count = sample_count
        if self.sample_count == 0:
            raise ValueError(f"split {split} contains no four-frame samples")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_video_cache"] = None
        state["_rectification_maps"] = {}
        return state

    def __del__(self):
        if self._video_cache is not None:
            self._video_cache.close()

    def _read_manifest(self):
        records = []
        calibration_catalogs = {}
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{self.manifest_path}:{line_number}: invalid JSON"
                    ) from error
                if record.get("schema") != SCHEMA:
                    raise ValueError(
                        f"{self.manifest_path}:{line_number}: unsupported schema"
                    )
                if record.get("split") != self.split:
                    continue
                if "calibration" not in record:
                    self._resolve_catalog_calibration(
                        record, line_number, calibration_catalogs
                    )
                self._validate_record(record, line_number)
                records.append(record)
        if not records:
            raise ValueError(f"manifest contains no {self.split} episodes")
        return records

    def _resolve_catalog_calibration(self, record, line_number, catalogs):
        required = {
            "calibration_bundle_sha256",
            "calibration_catalog_relative_path",
            "calibration_catalog_sha256",
        }
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"{self.manifest_path}:{line_number}: missing {sorted(missing)}"
            )
        relative = Path(record["calibration_catalog_relative_path"])
        if relative.is_absolute() or not relative.parts:
            raise ValueError("invalid calibration catalog path")
        path = (self.manifest_path.parent / relative).resolve()
        if not path.is_relative_to(self.manifest_path.parent) or not path.is_file():
            raise FileNotFoundError(path)
        expected_sha256 = record["calibration_catalog_sha256"]
        if len(expected_sha256) != 64:
            raise ValueError("invalid calibration catalog SHA256")
        if path not in catalogs:
            if sha256_file(path) != expected_sha256:
                raise ValueError(f"calibration catalog SHA256 mismatch: {path}")
            catalog = json.loads(path.read_text(encoding="utf-8"))
            if catalog.get("schema") != CALIBRATION_CATALOG_SCHEMA:
                raise ValueError(f"unsupported calibration catalog: {path}")
            bundles = catalog.get("calibration_bundles")
            if not isinstance(bundles, dict) or not bundles:
                raise ValueError(f"empty calibration catalog: {path}")
            for bundle_sha256, calibration in bundles.items():
                if len(bundle_sha256) != 64:
                    raise ValueError(f"invalid calibration bundle key: {path}")
                validate_calibration(calibration, bundle_sha256)
            catalogs[path] = (expected_sha256, bundles)
        catalog_sha256, bundles = catalogs[path]
        if catalog_sha256 != expected_sha256:
            raise ValueError(f"conflicting calibration catalog SHA256: {path}")
        bundle_sha256 = record["calibration_bundle_sha256"]
        if bundle_sha256 not in bundles:
            raise ValueError(f"missing calibration bundle {bundle_sha256}")
        record["calibration"] = bundles[bundle_sha256]

    def _validate_record(self, record, line_number):
        required = {
            "episode_id",
            "shard_id",
            "episode_index",
            "length",
            "window_count",
            "videos",
            "calibration",
            "rectification",
            "contract_sha256",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"{self.manifest_path}:{line_number}: missing {sorted(missing)}"
            )
        rectification = record["rectification"]
        if rectification.get("mode") not in {
            "verified_pre_rectified",
            "apply_calibration",
        }:
            raise ValueError(
                f"{record['episode_id']}: rectification is not verified"
            )
        if (
            rectification.get("audit_sha256")
            != self.expected_rectification_audit_sha256
        ):
            raise ValueError(
                f"{record['episode_id']}: rectification audit SHA mismatch"
            )
        validate_calibration(record["calibration"], record["episode_id"])
        stored_image = record.get("stored_image")
        if stored_image is not None:
            expected = {
                "encoded_size_hw": list(CANONICAL_STORED_HW),
                "transform": CANONICAL_STORED_TRANSFORM,
                "source_size_hw": list(SOURCE_HW),
                "resize_size_hw": list(RESIZE_HW),
                "padding_ltrb": list(PADDING_LTRB),
            }
            if not isinstance(stored_image, dict):
                raise ValueError(
                    f"{record['episode_id']}: stored_image must be an object"
                )
            for key, value in expected.items():
                if stored_image.get(key) != value:
                    raise ValueError(
                        f"{record['episode_id']}: stored_image {key} mismatch"
                    )
            mask_path = Path(stored_image.get("pixel_mask_relative_path", ""))
            mask_sha256 = stored_image.get("pixel_mask_sha256", "")
            if mask_path.is_absolute() or not mask_path.parts:
                raise ValueError(
                    f"{record['episode_id']}: invalid stored-image pixel mask path"
                )
            if len(mask_sha256) != 64:
                raise ValueError(
                    f"{record['episode_id']}: invalid stored-image pixel mask SHA256"
                )
        for key in VIDEO_KEYS.values():
            video = record["videos"].get(key)
            if not isinstance(video, dict):
                raise ValueError(f"{record['episode_id']}: missing video {key}")
            relative = Path(video.get("relative_path", ""))
            if relative.is_absolute() or not relative.parts:
                raise ValueError(f"{record['episode_id']}: invalid video path {key}")
            if "from_timestamp" not in video or "to_timestamp" not in video:
                raise ValueError(f"{record['episode_id']}: missing interval for {key}")

    def _validate_stored_image_assets(self):
        expected_assets = {}
        for record in self.records:
            stored_image = record.get("stored_image")
            if stored_image is None:
                continue
            relative = Path(stored_image["pixel_mask_relative_path"])
            expected_sha256 = stored_image["pixel_mask_sha256"]
            previous = expected_assets.setdefault(relative, expected_sha256)
            if previous != expected_sha256:
                raise ValueError(f"conflicting pixel mask SHA256 for {relative}")
        for relative, expected_sha256 in expected_assets.items():
            path = (self.dataset_root / relative).resolve()
            if not path.is_relative_to(self.dataset_root) or not path.is_file():
                raise FileNotFoundError(path)
            if sha256_file(path) != expected_sha256:
                raise ValueError(f"pixel mask SHA256 mismatch: {path}")

    def __len__(self):
        return self.sample_count

    def _sample_address(self, index):
        if index < 0:
            index += self.sample_count
        if not 0 <= index < self.sample_count:
            raise IndexError(index)
        episode_position = bisect.bisect_right(self._ends, index)
        span = self.episode_spans[episode_position]
        local_window = index - span.first_sample
        start_frame = local_window * START_STRIDE
        return self.records[span.record_index], start_frame

    def _decode_frames(self, path: Path, timestamps):
        if self._video_cache is None:
            self._video_cache = _AVContainerCache(self.video_cache_capacity)
        container, stream = self._video_cache.get(path)
        time_base = float(stream.time_base)
        container.seek(
            max(0, int(timestamps[0] / time_base)),
            stream=stream,
            any_frame=False,
            backward=True,
        )
        candidates = [[] for _ in timestamps]
        stop_time = timestamps[-1] + self.maximum_timestamp_error_s
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            frame_time = float(frame.time)
            if frame_time < timestamps[0] - 0.5:
                continue
            image = None
            for target_index, target in enumerate(timestamps):
                if abs(frame_time - target) <= self.maximum_timestamp_error_s:
                    if image is None:
                        image = frame.to_ndarray(format="rgb24")
                    candidates[target_index].append((abs(frame_time - target), image))
            if frame_time > stop_time:
                break
        frames = []
        for target, options in zip(timestamps, candidates):
            if not options:
                raise RuntimeError(f"{path}: no decoded frame near timestamp {target}")
            frames.append(min(options, key=lambda option: option[0])[1])
        return frames

    def _rectification_map(self, record, view, eye):
        camera = record["calibration"][view][eye]
        key_payload = json.dumps(camera, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()
        cached = self._rectification_maps.get(key)
        if cached is not None:
            return cached
        k = _matrix(camera["K"], (3, 3), f"{view}/{eye} K")
        d = np.asarray(camera["D"], dtype=np.float64)
        r = _matrix(camera["R"], (3, 3), f"{view}/{eye} R")
        p = _matrix(camera["P"], (3, 4), f"{view}/{eye} P")
        maps = cv2.initUndistortRectifyMap(
            k,
            d,
            r,
            p[:, :3],
            (SOURCE_HW[1], SOURCE_HW[0]),
            cv2.CV_32FC1,
        )
        self._rectification_maps[key] = maps
        return maps

    def _prepare_image(self, image, record, view, eye):
        if record.get("stored_image") is not None:
            if image.shape != (*CANONICAL_STORED_HW, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{record['episode_id']}: expected stored uint8 "
                    f"{CANONICAL_STORED_HW}x3 image"
                )
            return image
        if image.shape != (*SOURCE_HW, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"{record['episode_id']}: expected uint8 {SOURCE_HW}x3 image"
            )
        if record["rectification"]["mode"] == "apply_calibration":
            map_x, map_y = self._rectification_map(record, view, eye)
            image = cv2.remap(
                image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        resized = cv2.resize(
            image,
            (RESIZE_HW[1], RESIZE_HW[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        output = np.full((*OUTPUT_HW, 3), 128, dtype=np.uint8)
        output[32:224] = resized
        return output

    def _output_calibration(self, record):
        fx = []
        baseline = []
        for view in VIEWS:
            left_p = _matrix(
                record["calibration"][view]["left"]["P"],
                (3, 4),
                f"{view}/left P",
            )
            right_p = _matrix(
                record["calibration"][view]["right"]["P"],
                (3, 4),
                f"{view}/right P",
            )
            fx.append(left_p[0, 0] * (RESIZE_HW[1] / SOURCE_HW[1]))
            baseline.append(-right_p[0, 3] / right_p[0, 0])
        return np.asarray(fx, dtype=np.float32), np.asarray(
            baseline, dtype=np.float32
        )

    def __getitem__(self, index):
        return self.get_mode_item(index, "four_frame")

    def get_mode_item(self, index, temporal_mode):
        with profile_region("stereo/data/lerobot_getitem"):
            record, start_frame = self._sample_address(index)
            if temporal_mode == "four_frame":
                frame_offsets = FRAME_OFFSETS
            elif temporal_mode == "single_frame":
                frame_offsets = (FRAME_OFFSETS[self.single_frame_source_index],)
            else:
                raise ValueError(f"unsupported temporal mode {temporal_mode!r}")
            images = np.empty(
                (3, 2, 3, len(frame_offsets), 256, 256), dtype=np.uint8
            )
            for view_index, view in enumerate(VIEWS):
                for eye_index, eye in enumerate(EYES):
                    key = VIDEO_KEYS[(view, eye)]
                    video = record["videos"][key]
                    relative = Path(video["relative_path"])
                    path = (self.dataset_root / relative).resolve()
                    if not path.is_relative_to(self.dataset_root):
                        raise ValueError(f"video path escapes dataset root: {relative}")
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    from_timestamp = float(video["from_timestamp"])
                    timestamps = [
                        from_timestamp + (start_frame + offset) / FPS
                        for offset in frame_offsets
                    ]
                    decoded = self._decode_frames(path, timestamps)
                    for frame_index, image in enumerate(decoded):
                        prepared = self._prepare_image(
                            image, record, view, eye
                        )
                        images[view_index, eye_index, :, frame_index] = (
                            prepared.transpose(2, 0, 1)
                        )
            fx, baseline = self._output_calibration(record)
            video = torch.from_numpy(images).float().div_(255.0).sub_(0.5)
            return {
                "video": video,
                "fx": torch.from_numpy(fx),
                "baseline_m": torch.from_numpy(baseline),
                "sample_id": f"{record['episode_id']}:{start_frame:06d}",
                "episode_id": record["episode_id"],
                "shard_id": record["shard_id"],
                "start_frame": start_frame,
                "contract_sha256": record["contract_sha256"],
                "mode_id": f"stereo/{temporal_mode}",
                "eye_mode": "stereo",
                "temporal_mode": temporal_mode,
                "view_count": 3,
                "teacher_kind": "foundation_stereo",
            }


class EpisodeSequentialDistributedSampler(data.Sampler):
    """Shuffle shards/episodes while yielding each episode in time order."""

    def __init__(
        self,
        dataset: LeRobotStereoDataset,
        *,
        shuffle: bool,
        seed: int,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank")
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _assigned_spans(self):
        by_shard = defaultdict(list)
        for span in self.dataset.episode_spans:
            by_shard[span.shard_id].append(span)
        shard_ids = sorted(by_shard)
        generator = random.Random(self.seed + self.epoch)
        if self.shuffle:
            generator.shuffle(shard_ids)
            for shard_id in shard_ids:
                generator.shuffle(by_shard[shard_id])

        assigned = [[] for _ in range(self.num_replicas)]
        for position, shard_id in enumerate(shard_ids):
            assigned[position % self.num_replicas].extend(by_shard[shard_id])
        lengths = [sum(span.sample_count for span in spans) for spans in assigned]
        if min(lengths) == 0:
            raise ValueError("a distributed rank was assigned no samples")
        return assigned, max(lengths)

    def _rank_indices(self):
        assigned, target = self._assigned_spans()
        indices = []
        for span in assigned[self.rank]:
            indices.extend(
                range(span.first_sample, span.first_sample + span.sample_count)
            )
        padding = target - len(indices)
        if padding:
            repeats = (padding + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:padding])
        return indices

    def __iter__(self):
        return iter(self._rank_indices())

    def __len__(self):
        _, target = self._assigned_spans()
        return target


def fixed_episode_subset_indices(
    dataset: LeRobotStereoDataset, count: int, seed: int
) -> list[int]:
    """Choose a stable single window from each of ``count`` distinct episodes."""
    if count < 1:
        raise ValueError("fixed subset count must be positive")
    if len(dataset.episode_spans) < count:
        raise ValueError(
            f"fixed subset needs {count} episodes, split has "
            f"{len(dataset.episode_spans)}"
        )
    ordered = sorted(
        dataset.episode_spans,
        key=lambda span: hashlib.sha256(
            (
                f"{seed}:{span.shard_id}:"
                f"{dataset.records[span.record_index]['episode_id']}"
            ).encode("utf-8")
        ).digest(),
    )
    indices = []
    for span in ordered[:count]:
        episode_id = dataset.records[span.record_index]["episode_id"]
        local_window = int.from_bytes(
            hashlib.sha256(
                f"{seed}:validation-window:{episode_id}".encode("utf-8")
            ).digest()[:8],
            "big",
        ) % span.sample_count
        indices.append(span.first_sample + local_window)
    return indices
