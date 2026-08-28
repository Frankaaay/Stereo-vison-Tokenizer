"""Manifest-driven UMI, Hy and LIBERO inputs for StereoVAE pretraining."""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .geometry import GeometryMapping
from .lerobot_data import _AVContainerCache, _matrix, validate_calibration


HY_SCHEMA = "hy-cam-high-episode-v1"
LIBERO_SCHEMA = "libero-mono-episode-v1"
UMI_SCHEMA = "umi-raw-stereo-episode-v1"
UMI_VIEWS = ("head", "lefthand", "righthand")
UMI_EYES = ("left", "right")


@dataclass(frozen=True)
class EpisodeSpan:
    record_index: int
    variant: str
    first_sample: int
    sample_count: int


def _read_jsonl(path: Path, schema: str, split: str) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if record.get("schema") != schema:
                raise ValueError(f"{path}:{line_number}: expected schema {schema}")
            if record.get("split") == split:
                records.append(record)
    if not records:
        raise ValueError(f"{path} contains no {split} records")
    return records


def _resolve_alias_path(
    root_aliases: dict[str, str | Path], alias: str, relative_path: str = "."
) -> Path:
    if alias not in root_aliases:
        raise ValueError(f"manifest root alias {alias!r} is not configured on this node")
    root = Path(root_aliases[alias]).expanduser().resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"manifest path must be relative: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"manifest path escapes root alias {alias}: {relative}")
    return resolved


def _window_count(length: int, offsets: tuple[int, ...], stride: int) -> int:
    if length <= offsets[-1] or stride <= 0:
        return 0
    return (int(length) - 1 - offsets[-1]) // int(stride) + 1


class _ManifestWindowDataset(data.Dataset):
    offsets: tuple[int, ...]
    stride: int

    def _build_spans(self, variants) -> None:
        self.spans: list[EpisodeSpan] = []
        self._ends: list[int] = []
        total = 0
        for record_index, record in enumerate(self.records):
            expected = _window_count(int(record["length"]), self.offsets, self.stride)
            if int(record.get("window_count", -1)) != expected:
                raise ValueError(
                    f"{record.get('episode_id')}: window_count must be {expected}"
                )
            if expected == 0:
                continue
            for variant in variants(record):
                self.spans.append(EpisodeSpan(record_index, variant, total, expected))
                total += expected
                self._ends.append(total)
        if total == 0:
            raise ValueError("manifest split contains no complete windows")
        self.sample_count = total

    def __len__(self):
        return self.sample_count

    def _sample_address(self, index: int) -> tuple[dict[str, Any], str, int]:
        if index < 0:
            index += self.sample_count
        if not 0 <= index < self.sample_count:
            raise IndexError(index)
        position = bisect.bisect_right(self._ends, index)
        span = self.spans[position]
        local_window = index - span.first_sample
        return self.records[span.record_index], span.variant, local_window * self.stride

    def __getitem__(self, index):
        return self.get_mode_item(index, "four_frame")

    def _frame_offsets(self, temporal_mode: str) -> tuple[int, ...]:
        if temporal_mode == "four_frame":
            return self.offsets
        if temporal_mode == "single_frame":
            return (self.offsets[self.single_frame_source_index],)
        raise ValueError(f"unsupported temporal mode {temporal_mode!r}")


def _mono_sample(
    rgb: np.ndarray,
    *,
    sample_id: str,
    episode_id: str,
    dataset_id: str,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    contract_sha256: str,
    temporal_mode: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("mono RGB must be uint8 [T,3,H,W]")
    raw_rgb = torch.from_numpy(rgb.copy())
    source_hw = tuple(int(value) for value in rgb.shape[-2:])
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
        "sample_id": sample_id,
        "episode_id": episode_id,
        "dataset_id": dataset_id,
        "frame_index": torch.from_numpy(frame_indices.copy()),
        "timestamp_s": torch.from_numpy(timestamps.copy()),
        "contract_sha256": contract_sha256,
        "mode_id": f"mono/{temporal_mode}",
        "eye_mode": "mono",
        "temporal_mode": temporal_mode,
        "view_count": 1,
        "teacher_kind": "da3",
        **extra,
    }


class HyLanceMonoDataset(_ManifestWindowDataset):
    """Read only Hy cam_high from node-local Lance tables."""

    offsets = (0, 3, 6, 9)
    stride = 12
    camera_column = "observation_images_cam_high"

    def __init__(
        self,
        manifest_path,
        root_aliases: dict[str, str | Path],
        *,
        split: str,
        single_frame_source_index: int = 0,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root_aliases = dict(root_aliases)
        self.single_frame_source_index = int(single_frame_source_index)
        if not 0 <= self.single_frame_source_index < 4:
            raise ValueError("single-frame source index must be in [0,3]")
        self.records = _read_jsonl(self.manifest_path, HY_SCHEMA, split)
        for record in self.records:
            required = {
                "root_alias",
                "table_name",
                "episode_id",
                "episode_index",
                "length",
                "dataset_from_index",
                "window_count",
                "source_contract_sha256",
            }
            if not required.issubset(record):
                raise ValueError(f"incomplete Hy manifest record: {record.get('episode_id')}")
        self._build_spans(lambda record: ("cam_high",))
        self._lance_handles: dict[tuple[str, str], Any] = {}
        self._lance_pid = os.getpid()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_lance_handles"] = {}
        state["_lance_pid"] = os.getpid()
        return state

    def _dataset(self, record):
        pid = os.getpid()
        if pid != self._lance_pid:
            self._lance_handles = {}
            self._lance_pid = pid
        key = (record["root_alias"], record["table_name"])
        if key not in self._lance_handles:
            try:
                import lance
            except ImportError as error:
                raise ImportError("Hy online loading requires pylance") from error
            table_root = _resolve_alias_path(
                self.root_aliases, record["root_alias"], record["table_name"]
            )
            lance_path = table_root / f"{record['table_name']}.lance"
            if not lance_path.is_dir():
                raise FileNotFoundError(lance_path)
            self._lance_handles[key] = lance.dataset(str(lance_path))
        return self._lance_handles[key]

    @staticmethod
    def _decode_jpeg(payload: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(payload)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (240, 424, 3):
            raise ValueError(f"Hy cam_high must be [240,424,3], got {rgb.shape}")
        return rgb.transpose(2, 0, 1).copy()

    @staticmethod
    def _take_episode_frames(lance_dataset, episode_index, frame_indices, columns):
        requested = [int(value) for value in frame_indices]
        frame_filter = ", ".join(str(value) for value in requested)
        rows = lance_dataset.to_table(
            filter=(
                f"episode_index = {int(episode_index)} AND "
                f"frame_index IN ({frame_filter})"
            ),
            columns=columns,
        ).to_pylist()
        by_frame = {}
        for row in rows:
            frame_index = int(row["frame_index"])
            if frame_index in by_frame:
                raise ValueError("Hy Lance frame identity is not unique")
            by_frame[frame_index] = row
        if set(by_frame) != set(requested):
            raise ValueError("Hy Lance frame identity mismatch")
        return [by_frame[frame_index] for frame_index in requested]

    @staticmethod
    def _timestamps_match_frame_rate(timestamps, frame_indices, fps):
        expected = np.asarray(frame_indices, np.float64) / float(fps)
        return np.allclose(
            np.asarray(timestamps, np.float64),
            expected,
            rtol=np.finfo(np.float32).eps,
            atol=5e-6,
        )

    def get_mode_item(self, index, temporal_mode):
        record, _, start = self._sample_address(index)
        offsets = self._frame_offsets(temporal_mode)
        relative = np.asarray([start + offset for offset in offsets], np.int64)
        rows = self._take_episode_frames(
            self._dataset(record),
            record["episode_index"],
            relative,
            columns=[
                "episode_index",
                "frame_index",
                "timestamp",
                self.camera_column,
            ],
        )
        if [int(row["episode_index"]) for row in rows] != [
            int(record["episode_index"])
        ] * len(rows):
            raise ValueError("Hy Lance window crosses an episode boundary")
        if [int(row["frame_index"]) for row in rows] != relative.tolist():
            raise ValueError("Hy Lance frame identity mismatch")
        timestamps = np.asarray([float(row["timestamp"]) for row in rows], np.float64)
        if not self._timestamps_match_frame_rate(
            timestamps, relative, record.get("fps", 30.0)
        ):
            raise ValueError("Hy timestamps disagree with frame_index/fps")
        rgb = np.stack([self._decode_jpeg(row[self.camera_column]) for row in rows])
        return _mono_sample(
            rgb,
            sample_id=f"hy/{record['table_name']}/{record['episode_id']}/{start:06d}",
            episode_id=record["episode_id"],
            dataset_id="hy",
            frame_indices=relative,
            timestamps=timestamps,
            contract_sha256=record["source_contract_sha256"],
            temporal_mode=temporal_mode,
            extra={"table_name": record["table_name"], "camera_id": "cam_high"},
        )


class LiberoMonoDataset(_ManifestWindowDataset):
    """Decode both LIBERO monocular views from LeRobot v2.1 episode MP4s."""

    offsets = (0, 2, 4, 6)
    stride = 8
    camera_keys = (
        "observation.images.image",
        "observation.images.wrist_image",
    )

    def __init__(
        self,
        manifest_path,
        root_aliases: dict[str, str | Path],
        *,
        split: str,
        single_frame_source_index: int = 0,
        video_cache_capacity: int = 12,
        maximum_timestamp_error_s: float = 0.025,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root_aliases = dict(root_aliases)
        self.single_frame_source_index = int(single_frame_source_index)
        self.video_cache_capacity = int(video_cache_capacity)
        self.maximum_timestamp_error_s = float(maximum_timestamp_error_s)
        if not 0 <= self.single_frame_source_index < 4:
            raise ValueError("single-frame source index must be in [0,3]")
        if self.video_cache_capacity < 1 or self.maximum_timestamp_error_s <= 0:
            raise ValueError("invalid LIBERO video cache/timestamp tolerance")
        self.records = _read_jsonl(self.manifest_path, LIBERO_SCHEMA, split)
        required = {
            "root_alias",
            "suite",
            "episode_id",
            "episode_index",
            "length",
            "window_count",
            "video_path",
            "source_contract_sha256",
        }
        for record in self.records:
            if not required.issubset(record):
                raise ValueError(
                    f"incomplete LIBERO manifest record: {record.get('episode_id')}"
                )
        self._build_spans(lambda record: self.camera_keys)
        self._video_cache = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_video_cache"] = None
        return state

    def __del__(self):
        if self._video_cache is not None:
            self._video_cache.close()

    def _decode_frames(self, path: Path, frame_indices: np.ndarray, fps: float):
        if self._video_cache is None:
            self._video_cache = _AVContainerCache(self.video_cache_capacity)
        container, stream = self._video_cache.get(path)
        time_base = float(stream.time_base)
        targets = frame_indices.astype(np.float64) / fps
        container.seek(
            max(0, int((targets[0] - 0.5) / time_base)),
            stream=stream,
            any_frame=False,
            backward=True,
        )
        candidates = [[] for _ in targets]
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            frame_time = float(frame.time)
            for target_index, target in enumerate(targets):
                error = abs(frame_time - target)
                if error <= self.maximum_timestamp_error_s:
                    candidates[target_index].append((error, frame))
            if frame_time > targets[-1] + self.maximum_timestamp_error_s:
                break
        output = []
        actual = []
        for target, options in zip(targets, candidates):
            if not options:
                raise RuntimeError(f"{path}: no frame near {target:.6f}s")
            _, frame = min(options, key=lambda item: item[0])
            image = frame.to_ndarray(format="rgb24")
            if image.shape != (512, 512, 3):
                raise ValueError(f"{path}: expected [512,512,3]")
            output.append(image.transpose(2, 0, 1))
            actual.append(float(frame.time))
        return np.stack(output).astype(np.uint8), np.asarray(actual, np.float64)

    def get_mode_item(self, index, temporal_mode):
        record, camera_key, start = self._sample_address(index)
        offsets = self._frame_offsets(temporal_mode)
        frame_indices = np.asarray([start + offset for offset in offsets], np.int64)
        suite_root = _resolve_alias_path(
            self.root_aliases, record["root_alias"], record["suite"]
        )
        episode_index = int(record["episode_index"])
        chunk = episode_index // int(record.get("chunks_size", 1000))
        relative = str(record["video_path"]).format(
            episode_chunk=chunk,
            video_key=camera_key,
            episode_index=episode_index,
        )
        video_path = _resolve_alias_path(
            {"suite": suite_root}, "suite", relative
        )
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        fps = float(record.get("fps", 20.0))
        rgb, timestamps = self._decode_frames(video_path, frame_indices, fps)
        camera_id = "agentview" if camera_key.endswith(".image") else "wrist"
        return _mono_sample(
            rgb,
            sample_id=f"libero/{record['suite']}/{episode_index:06d}/{camera_id}/{start:06d}",
            episode_id=record["episode_id"],
            dataset_id="libero",
            frame_indices=frame_indices,
            timestamps=timestamps,
            contract_sha256=record["source_contract_sha256"],
            temporal_mode=temporal_mode,
            extra={"suite": record["suite"], "camera_id": camera_id},
        )


class UMIRawStereoDataset(_ManifestWindowDataset):
    """Decode six Foxglove CompressedVideo H.264 streams directly from raw MCAP."""

    offsets = (0, 3, 6, 9)
    stride = 12

    def __init__(
        self,
        manifest_path,
        root_aliases: dict[str, str | Path],
        *,
        split: str,
        single_frame_source_index: int = 0,
        episode_cache_capacity: int = 2,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root_aliases = dict(root_aliases)
        self.single_frame_source_index = int(single_frame_source_index)
        self.episode_cache_capacity = int(episode_cache_capacity)
        if not 0 <= self.single_frame_source_index < 4:
            raise ValueError("single-frame source index must be in [0,3]")
        if self.episode_cache_capacity < 1:
            raise ValueError("UMI episode cache capacity must be positive")
        self.records = _read_jsonl(self.manifest_path, UMI_SCHEMA, split)
        for record in self.records:
            required = {
                "root_alias",
                "mcap_relative_path",
                "episode_id",
                "length",
                "window_count",
                "source_contract_sha256",
                "calibration",
                "topics",
            }
            if not required.issubset(record):
                raise ValueError(
                    f"incomplete UMI manifest record: {record.get('episode_id')}"
                )
            validate_calibration(record["calibration"], record["episode_id"])
            if set(record.get("topics", {})) != {
                f"{view}/{eye}" for view in UMI_VIEWS for eye in UMI_EYES
            }:
                raise ValueError(f"{record['episode_id']}: incomplete UMI topic map")
        self._build_spans(lambda record: ("stereo",))
        self._episode_cache = OrderedDict()
        self._rectification_maps = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_episode_cache"] = OrderedDict()
        state["_rectification_maps"] = {}
        return state

    def _decode_episode(self, record):
        cache_key = record["episode_id"]
        cached = self._episode_cache.pop(cache_key, None)
        if cached is not None:
            self._episode_cache[cache_key] = cached
            return cached
        try:
            import av
            from mcap.reader import make_reader
            from mcap_protobuf.decoder import DecoderFactory
        except ImportError as error:
            raise ImportError(
                "UMI raw loading requires av, mcap and mcap-protobuf-support"
            ) from error
        path = _resolve_alias_path(
            self.root_aliases, record["root_alias"], record["mcap_relative_path"]
        )
        reverse_topics = {topic: key for key, topic in record["topics"].items()}
        codecs = {key: av.CodecContext.create("h264", "r") for key in reverse_topics.values()}
        frames = {key: [] for key in reverse_topics.values()}
        timestamps = {key: [] for key in reverse_topics.values()}
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, channel, message, decoded in reader.iter_decoded_messages(
                topics=list(reverse_topics)
            ):
                key = reverse_topics[channel.topic]
                if str(getattr(decoded, "format", "h264")).lower() not in {
                    "h264",
                    "avc",
                    "h264-annex-b",
                }:
                    raise ValueError(f"{path}: unsupported video format")
                for packet in codecs[key].parse(bytes(decoded.data)):
                    for frame in codecs[key].decode(packet):
                        frames[key].append(frame.to_ndarray(format="rgb24"))
                        timestamps[key].append(message.log_time / 1e9)
        for key, codec in codecs.items():
            for frame in codec.decode(None):
                frames[key].append(frame.to_ndarray(format="rgb24"))
                timestamps[key].append(timestamps[key][-1] if timestamps[key] else 0.0)
        lengths = {key: len(value) for key, value in frames.items()}
        if min(lengths.values()) < int(record["length"]):
            raise RuntimeError(f"{path}: decoded stream lengths {lengths}")
        result = (frames, timestamps)
        self._episode_cache[cache_key] = result
        while len(self._episode_cache) > self.episode_cache_capacity:
            self._episode_cache.popitem(last=False)
        return result

    def _rectification_map(self, record, view, eye):
        camera = record["calibration"][view][eye]
        key = hashlib.sha256(
            json.dumps(camera, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if key not in self._rectification_maps:
            self._rectification_maps[key] = cv2.initUndistortRectifyMap(
                _matrix(camera["K"], (3, 3), "K"),
                np.asarray(camera["D"], np.float64),
                _matrix(camera["R"], (3, 3), "R"),
                _matrix(camera["P"], (3, 4), "P")[:, :3],
                (640, 480),
                cv2.CV_32FC1,
            )
        return self._rectification_maps[key]

    def _prepare(self, record, view, eye, image):
        if image.shape != (480, 640, 3):
            raise ValueError(f"{record['episode_id']}: expected UMI [480,640,3]")
        map_x, map_y = self._rectification_map(record, view, eye)
        image = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        image = cv2.resize(image, (256, 192), interpolation=cv2.INTER_LINEAR)
        output = np.full((256, 256, 3), 128, np.uint8)
        output[32:224] = image
        return output

    def get_mode_item(self, index, temporal_mode):
        record, _, start = self._sample_address(index)
        offsets = self._frame_offsets(temporal_mode)
        indices = np.asarray([start + offset for offset in offsets], np.int64)
        decoded, decoded_timestamps = self._decode_episode(record)
        images = np.empty((3, 2, 3, len(indices), 256, 256), np.uint8)
        timestamps = np.empty((3, 2, len(indices)), np.float64)
        fx = []
        baseline = []
        for view_index, view in enumerate(UMI_VIEWS):
            for eye_index, eye in enumerate(UMI_EYES):
                key = f"{view}/{eye}"
                for time_index, frame_index in enumerate(indices):
                    image = self._prepare(
                        record, view, eye, decoded[key][int(frame_index)]
                    )
                    images[view_index, eye_index, :, time_index] = image.transpose(2, 0, 1)
                    timestamps[view_index, eye_index, time_index] = decoded_timestamps[key][
                        int(frame_index)
                    ]
            left_p = _matrix(record["calibration"][view]["left"]["P"], (3, 4), "P")
            right_p = _matrix(record["calibration"][view]["right"]["P"], (3, 4), "P")
            fx.append(left_p[0, 0] * 256.0 / 640.0)
            baseline.append(-right_p[0, 3] / right_p[0, 0])
        maximum_pair_skew = float(record.get("maximum_pair_skew_s", 1.0 / 60.0))
        if np.max(np.abs(timestamps[:, 0] - timestamps[:, 1])) > maximum_pair_skew:
            raise ValueError(f"{record['episode_id']}: stereo timestamp skew exceeded")
        video = torch.from_numpy(images.copy()).float().div_(255.0).sub_(0.5)
        return {
            "video": video,
            "fx": torch.tensor(fx, dtype=torch.float32),
            "baseline_m": torch.tensor(baseline, dtype=torch.float32),
            "sample_id": f"umi/{record['episode_id']}/{start:06d}",
            "episode_id": record["episode_id"],
            "dataset_id": "umi",
            "frame_index": torch.from_numpy(indices.copy()),
            "timestamp_s": torch.from_numpy(timestamps.copy()),
            "contract_sha256": record["source_contract_sha256"],
            "mode_id": f"stereo/{temporal_mode}",
            "eye_mode": "stereo",
            "temporal_mode": temporal_mode,
            "view_count": 3,
            "teacher_kind": "foundation_stereo",
        }
