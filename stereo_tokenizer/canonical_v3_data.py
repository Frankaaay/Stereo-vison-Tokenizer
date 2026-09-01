"""Read-only six-camera adapter for NGAD canonical LeRobot v3 datasets."""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils import data


SCHEMA = "canonical-v3-stereo-ablation-v1"
VIEWS = ("head", "lefthand", "righthand")
EYES = ("left", "right")
VIDEO_KEYS = {
    ("head", "left"): "observation.images.cam_head_left",
    ("head", "right"): "observation.images.cam_head_right",
    ("lefthand", "left"): "observation.images.cam_left_wrist_left",
    ("lefthand", "right"): "observation.images.cam_left_wrist_right",
    ("righthand", "left"): "observation.images.cam_right_wrist_left",
    ("righthand", "right"): "observation.images.cam_right_wrist_right",
}
FRAME_OFFSETS = (0, 3, 6, 9)
START_STRIDE = 12


def window_count(length):
    return max(0, (int(length) - 1 - FRAME_OFFSETS[-1]) // START_STRIDE + 1)


@dataclass(frozen=True)
class EpisodeSpan:
    record_index: int
    first_sample: int
    sample_count: int
    shard_id: str


class _AVContainerCache:
    def __init__(self, capacity):
        if int(capacity) < 1:
            raise ValueError("video cache capacity must be positive")
        self.capacity = int(capacity)
        self._containers = OrderedDict()

    def get(self, path):
        try:
            import av
        except ImportError as error:
            raise RuntimeError("canonical-v3 MP4 loading requires PyAV") from error
        key = str(path)
        cached = self._containers.pop(key, None)
        if cached is None:
            container = av.open(key, mode="r")
            streams = [stream for stream in container.streams if stream.type == "video"]
            if len(streams) != 1:
                container.close()
                raise ValueError(f"{path}: expected exactly one video stream")
            cached = (container, streams[0])
        self._containers[key] = cached
        while len(self._containers) > self.capacity:
            _, (container, _) = self._containers.popitem(last=False)
            container.close()
        return cached

    def close(self):
        for container, _ in self._containers.values():
            container.close()
        self._containers.clear()


class CanonicalV3StereoDataset(data.Dataset):
    """Decode synchronized RGB already published at the tokenizer resolution."""

    is_canonical_v3_ablation = True

    def __init__(
        self,
        manifest_path,
        dataset_root,
        *,
        split,
        rectification_audit_sha256,
        pixel_mask_path=None,
        video_cache_capacity=12,
        maximum_timestamp_error_s=0.05,
        single_frame_source_index=0,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = str(split)
        self.rectification_audit_sha256 = str(rectification_audit_sha256)
        self.maximum_timestamp_error_s = float(maximum_timestamp_error_s)
        self.video_cache_capacity = int(video_cache_capacity)
        self.single_frame_source_index = int(single_frame_source_index)
        self._video_cache = None
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split {self.split!r}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(self.dataset_root)
        if len(self.rectification_audit_sha256) != 64:
            raise ValueError("a full rectification audit SHA256 is required")
        if not 0 <= self.single_frame_source_index < len(FRAME_OFFSETS):
            raise ValueError("single-frame source index is out of range")
        if self.maximum_timestamp_error_s <= 0:
            raise ValueError("maximum timestamp error must be positive")

        if pixel_mask_path is None:
            pixel_mask_path = self.dataset_root / "image_pixel_mask_umi.npz"
        self.pixel_mask_path = Path(pixel_mask_path).expanduser().resolve()
        if not self.pixel_mask_path.is_file():
            raise FileNotFoundError(self.pixel_mask_path)
        with np.load(self.pixel_mask_path) as payload:
            pixel_mask = np.asarray(payload["mask"], dtype=bool)
        if pixel_mask.shape != (256, 256) or pixel_mask.mean() < 0.5:
            raise ValueError("canonical-v3 pixel mask must be a valid [256,256] mask")
        self.pixel_mask = pixel_mask

        self.records = []
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("schema") != SCHEMA:
                    raise ValueError(
                        f"{self.manifest_path}:{line_number}: unsupported schema"
                    )
                if record.get("split") != self.split:
                    continue
                self._validate_record(record, line_number)
                self.records.append(record)
        if not self.records:
            raise ValueError(f"manifest contains no {self.split} episodes")

        self.episode_spans = []
        self._ends = []
        total = 0
        for record_index, record in enumerate(self.records):
            count = int(record["window_count"])
            self.episode_spans.append(
                EpisodeSpan(
                    record_index=record_index,
                    first_sample=total,
                    sample_count=count,
                    shard_id=f"file-{int(record['file_serial']):03d}",
                )
            )
            total += count
            self._ends.append(total)
        self.sample_count = total
        if total == 0:
            raise ValueError(f"split {self.split} contains no valid windows")

    def _validate_record(self, record, line_number):
        required = {
            "episode_id",
            "episode_index",
            "file_serial",
            "length",
            "window_count",
            "videos",
            "contract_sha256",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"{self.manifest_path}:{line_number}: missing {sorted(missing)}"
            )
        if int(record["window_count"]) != window_count(record["length"]):
            raise ValueError(f"{record['episode_id']}: inconsistent window count")
        for key in VIDEO_KEYS.values():
            video = record["videos"].get(key)
            if not isinstance(video, dict):
                raise ValueError(f"{record['episode_id']}: missing {key}")
            relative = Path(video.get("relative_path", ""))
            if relative.is_absolute() or not relative.parts:
                raise ValueError(f"{record['episode_id']}: invalid video path")
            if float(video["from_timestamp"]) >= float(video["to_timestamp"]):
                raise ValueError(f"{record['episode_id']}: invalid video interval")

    def __len__(self):
        return self.sample_count

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_video_cache"] = None
        return state

    def __del__(self):
        if self._video_cache is not None:
            self._video_cache.close()

    def _sample_address(self, index):
        if index < 0:
            index += self.sample_count
        if not 0 <= index < self.sample_count:
            raise IndexError(index)
        position = bisect.bisect_right(self._ends, index)
        span = self.episode_spans[position]
        return (
            self.records[span.record_index],
            (index - span.first_sample) * START_STRIDE,
        )

    def _decode_frames(self, path, timestamps):
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
            for position, target in enumerate(timestamps):
                error = abs(frame_time - target)
                if error <= self.maximum_timestamp_error_s:
                    if image is None:
                        image = frame.to_ndarray(format="rgb24")
                    candidates[position].append((error, image))
            if frame_time > stop_time:
                break
        output = []
        for target, options in zip(timestamps, candidates):
            if not options:
                raise RuntimeError(f"{path}: no frame near {target:.6f}s")
            image = min(options, key=lambda item: item[0])[1]
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError(f"{path}: expected uint8 [256,256,3]")
            image = image.copy()
            image[~self.pixel_mask] = 128
            output.append(image)
        return output

    def __getitem__(self, index):
        record, start_frame = self._sample_address(index)
        images = np.empty((3, 2, 3, 4, 256, 256), dtype=np.uint8)
        for view_index, view in enumerate(VIEWS):
            for eye_index, eye in enumerate(EYES):
                video = record["videos"][VIDEO_KEYS[(view, eye)]]
                path = (self.dataset_root / video["relative_path"]).resolve()
                if not path.is_relative_to(self.dataset_root):
                    raise ValueError("video path escapes canonical dataset root")
                timestamps = [
                    float(video["from_timestamp"]) + (start_frame + offset) / 30.0
                    for offset in FRAME_OFFSETS
                ]
                for frame_index, image in enumerate(
                    self._decode_frames(path, timestamps)
                ):
                    images[view_index, eye_index, :, frame_index] = image.transpose(
                        2, 0, 1
                    )
        video = torch.from_numpy(images).float().div_(255.0).sub_(0.5)
        # Unit camera scale is used only by the legacy aggregate metric.  The
        # ablation report's primary metric independently centers every view.
        unit_scale = torch.ones(3, dtype=torch.float32)
        return {
            "video": video,
            "fx": unit_scale.clone(),
            "baseline_m": unit_scale,
            "sample_id": f"{record['episode_id']}:{start_frame:06d}",
            "episode_id": record["episode_id"],
            "shard_id": f"file-{int(record['file_serial']):03d}",
            "start_frame": start_frame,
            "contract_sha256": record["contract_sha256"],
            "mode_id": "stereo/four_frame",
            "eye_mode": "stereo",
            "temporal_mode": "four_frame",
            "view_count": 3,
            "teacher_kind": "foundation_stereo",
            "geometry_scale_mode": "per_view_scale_free",
        }


def _episode_order(dataset, seed, label):
    return sorted(
        dataset.episode_spans,
        key=lambda span: hashlib.sha256(
            (
                f"{seed}:{label}:"
                f"{dataset.records[span.record_index]['episode_id']}"
            ).encode()
        ).digest(),
    )


def fixed_episode_window_pairs(dataset, episode_count, windows_per_episode, seed):
    """Return (base, wrong-right) indices with episode-level derangement."""

    episode_count = int(episode_count)
    windows_per_episode = int(windows_per_episode)
    if episode_count < 2 or windows_per_episode < 1:
        raise ValueError("ablation subset needs >=2 episodes and >=1 window")
    ordered = _episode_order(dataset, seed, "episode")[:episode_count]
    if len(ordered) != episode_count:
        raise ValueError(
            f"requested {episode_count} episodes, split has {len(ordered)}"
        )
    partners = ordered[1:] + ordered[:1]
    pairs = []
    for span, partner in zip(ordered, partners):
        episode_id = dataset.records[span.record_index]["episode_id"]
        locals_ordered = sorted(
            range(span.sample_count),
            key=lambda local: hashlib.sha256(
                f"{seed}:window:{episode_id}:{local}".encode()
            ).digest(),
        )
        if len(locals_ordered) < windows_per_episode:
            raise ValueError(
                f"{episode_id} has only {len(locals_ordered)} windows"
            )
        for local in sorted(locals_ordered[:windows_per_episode]):
            base = span.first_sample + local
            wrong_local = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:wrong:{episode_id}:{local}".encode()
                ).digest()[:8],
                "big",
            ) % partner.sample_count
            pairs.append((base, partner.first_sample + wrong_local))
    return tuple(pairs)


class CanonicalV3AblationSubset(data.Dataset):
    def __init__(self, dataset, pairs):
        self.dataset = dataset
        self.pairs = tuple((int(left), int(right)) for left, right in pairs)
        if not self.pairs:
            raise ValueError("ablation subset cannot be empty")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        base_index, wrong_index = self.pairs[index]
        base = dict(self.dataset[base_index])
        wrong = self.dataset[wrong_index]
        if base["episode_id"] == wrong["episode_id"]:
            raise RuntimeError("WRONG_RIGHT partner must come from another episode")
        base["wrong_right_video"] = wrong["video"][:, 1]
        base["wrong_episode_id"] = wrong["episode_id"]
        return base
