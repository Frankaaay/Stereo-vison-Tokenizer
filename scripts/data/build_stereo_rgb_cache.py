#!/usr/bin/env python3
"""Build an independent six-stream RGB cache and finalize Manifest v3.

The source Manifest v2 and FoundationStereo GT are never modified. Cache mode
may be sharded by episode; finalize mode succeeds only after every referenced
RGB cache passes the frozen shape/dtype contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from collections import defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np


VIEWS = ("head", "lefthand", "righthand")
EYES = ("left", "right")
VIDEO_TOPIC = re.compile(
    r"/camera/coracam_(?P<view>head|lefthand|righthand)/"
    r"(?P<eye>left|right)_h264/video"
)
EXPECTED_SOURCE_HW = (480, 640)
EXPECTED_RESIZE_HW = (192, 256)
EXPECTED_OUTPUT_HW = (256, 256)
EXPECTED_PADDING_LTRB = (0, 32, 0, 32)
RGB_SHAPE = (3, 2, 3, 4, 256, 256)
RGB_SCHEMA = "stereo-rgb-cache-v1"


class FlatTable:
    """Minimal FlatBuffer reader for foxglove.CompressedVideo."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.position = struct.unpack_from("<I", payload, 0)[0]
        self.vtable = self.position - struct.unpack_from(
            "<i", payload, self.position
        )[0]

    def field_offset(self, index: int) -> int:
        entry = 4 + index * 2
        vtable_size = struct.unpack_from("<H", self.payload, self.vtable)[0]
        if entry >= vtable_size:
            return 0
        return struct.unpack_from("<H", self.payload, self.vtable + entry)[0]

    def byte_vector(self, index: int) -> bytes:
        offset = self.field_offset(index)
        if offset == 0:
            raise ValueError(f"FlatBuffer field {index} is absent")
        field = self.position + offset
        vector = field + struct.unpack_from("<I", self.payload, field)[0]
        size = struct.unpack_from("<I", self.payload, vector)[0]
        start = vector + 4
        return self.payload[start : start + size]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def validate_v2_record(record: dict) -> None:
    required = ("sample_id", "episode_id", "mcap_path", "frames", "gt_relative_path")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise ValueError(f"manifest record is missing {missing}")
    if len(record["frames"]) != 4:
        raise ValueError(f"{record['sample_id']}: expected exactly four frames")
    preprocessing = record.get("preprocessing", {})
    expected_preprocessing = {
        "source_size_hw": [480, 640],
        "resize_size_hw": [192, 256],
        "output_size_hw": [256, 256],
        "padding_ltrb": [0, 32, 0, 32],
        "scale_xy": [0.4, 0.4],
    }
    for key, expected in expected_preprocessing.items():
        if preprocessing.get(key) != expected:
            raise ValueError(
                f"{record['sample_id']}: {key}={preprocessing.get(key)!r}, "
                f"expected {expected!r}"
            )
    for frame in record["frames"]:
        selections = frame.get("selections", {})
        for view in VIEWS:
            for eye in EYES:
                key = f"{view}/{eye}"
                selection = selections.get(key)
                if selection is None or "source_frame_index" not in selection:
                    raise ValueError(f"{record['sample_id']}: missing {key}")


def rgb_relative_path(record: dict) -> PurePosixPath:
    gt_path = PurePosixPath(record["gt_relative_path"])
    if gt_path.is_absolute() or not gt_path.parts or gt_path.parts[0] != "gt":
        raise ValueError(
            f"{record['sample_id']}: expected gt/<episode>/<sample>.npz"
        )
    if gt_path.suffix != ".npz":
        raise ValueError(f"{record['sample_id']}: GT cache must be .npz")
    return PurePosixPath("rgb", *gt_path.parts[1:])


def read_h264_streams(mcap_path: Path) -> dict[str, list[bytes]]:
    from mcap.reader import make_reader

    streams: dict[str, list[bytes]] = defaultdict(list)
    with mcap_path.open("rb") as source:
        for _, channel, message in make_reader(source).iter_messages():
            match = VIDEO_TOPIC.fullmatch(channel.topic)
            if match is None:
                continue
            key = f"{match['view']}/{match['eye']}"
            streams[key].append(FlatTable(message.data).byte_vector(2))
    expected = {f"{view}/{eye}" for view in VIEWS for eye in EYES}
    missing = expected - set(streams)
    if missing:
        raise ValueError(f"{mcap_path}: missing H.264 streams {sorted(missing)}")
    return dict(streams)


def decode_selected_frames(
    h264_packets: list[bytes], selected_indices: set[int]
) -> dict[int, np.ndarray]:
    import av

    codec = av.CodecContext.create("h264", "r")
    selected = {}
    decoder_errors = []
    for message_index, payload in enumerate(h264_packets):
        packet = av.Packet(payload)
        packet.pts = message_index
        packet.dts = message_index
        packet.time_base = Fraction(1, 1_000_000)
        try:
            frames = codec.decode(packet)
        except Exception as exc:
            decoder_errors.append(
                f"message={message_index}:{type(exc).__name__}:{exc}"
            )
            continue
        for frame in frames:
            if frame.pts is None:
                decoder_errors.append(
                    f"message={message_index}:decoded_frame_has_no_pts"
                )
                continue
            source_index = int(frame.pts)
            if source_index not in selected_indices:
                continue
            image = frame.to_ndarray(format="rgb24")
            if image.shape != (*EXPECTED_SOURCE_HW, 3):
                raise ValueError(
                    f"decoded frame {source_index} has shape {image.shape}"
                )
            selected[source_index] = image
        if len(selected) == len(selected_indices):
            break
    if len(selected) != len(selected_indices):
        try:
            flushed_frames = codec.decode(None)
        except Exception as exc:
            decoder_errors.append(f"flush:{type(exc).__name__}:{exc}")
            flushed_frames = []
        for frame in flushed_frames:
            if frame.pts is None:
                decoder_errors.append("flush:decoded_frame_has_no_pts")
                continue
            source_index = int(frame.pts)
            if source_index not in selected_indices:
                continue
            image = frame.to_ndarray(format="rgb24")
            if image.shape != (*EXPECTED_SOURCE_HW, 3):
                raise ValueError(
                    f"decoded frame {source_index} has shape {image.shape}"
                )
            selected[source_index] = image
    missing = selected_indices - set(selected)
    if missing:
        raise IndexError(
            "decoder did not produce requested MCAP message indices "
            f"{sorted(missing)}; decoder_errors={decoder_errors[:10]}"
        )
    return selected


def preprocess_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    resized = cv2.resize(
        image,
        (EXPECTED_RESIZE_HW[1], EXPECTED_RESIZE_HW[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    output = np.full((*EXPECTED_OUTPUT_HW, 3), 128, dtype=np.uint8)
    output[32:224] = resized
    return output


def selected_indices(records: list[dict]) -> dict[str, set[int]]:
    indices = {f"{view}/{eye}": set() for view in VIEWS for eye in EYES}
    for record in records:
        for frame in record["frames"]:
            for key in indices:
                indices[key].add(
                    int(frame["selections"][key]["source_frame_index"])
                )
    return indices


def assemble_rgb(record: dict, decoded: dict[str, dict[int, np.ndarray]]):
    rgb = np.empty(RGB_SHAPE, dtype=np.uint8)
    for view_index, view in enumerate(VIEWS):
        for eye_index, eye in enumerate(EYES):
            key = f"{view}/{eye}"
            for frame_index, frame in enumerate(record["frames"]):
                source_index = int(
                    frame["selections"][key]["source_frame_index"]
                )
                processed = preprocess_rgb(decoded[key][source_index])
                rgb[view_index, eye_index, :, frame_index] = np.moveaxis(
                    processed, -1, 0
                )
    return rgb


def validate_rgb_cache(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as cache:
        if cache.files != ["rgb"]:
            raise ValueError(f"{path}: expected only the rgb array")
        rgb = cache["rgb"]
        if rgb.shape != RGB_SHAPE or rgb.dtype != np.uint8:
            raise ValueError(
                f"{path}: expected uint8 {RGB_SHAPE}, got {rgb.dtype} {rgb.shape}"
            )


def write_rgb_cache(path: Path, rgb: np.ndarray) -> None:
    if rgb.shape != RGB_SHAPE or rgb.dtype != np.uint8:
        raise ValueError("attempted to write an invalid RGB cache")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("wb") as stream:
        np.savez(stream, rgb=rgb)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cache_records(
    manifest_v2: Path,
    output_root: Path,
    *,
    shard_index: int,
    shard_count: int,
) -> dict:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    records = read_jsonl(manifest_v2)
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        validate_v2_record(record)
        by_episode[record["episode_id"]].append(record)

    episodes = sorted(by_episode)
    selected_episodes = episodes[shard_index::shard_count]
    written = 0
    reused = 0
    for episode_id in selected_episodes:
        episode_records = by_episode[episode_id]
        mcap_paths = {record["mcap_path"] for record in episode_records}
        if len(mcap_paths) != 1:
            raise ValueError(f"{episode_id}: records reference multiple MCAPs")
        streams = read_h264_streams(Path(next(iter(mcap_paths))))
        required = selected_indices(episode_records)
        decoded = {
            key: decode_selected_frames(streams[key], required[key])
            for key in sorted(required)
        }
        for record in episode_records:
            target = output_root / Path(rgb_relative_path(record).as_posix())
            if target.exists():
                validate_rgb_cache(target)
                reused += 1
                continue
            write_rgb_cache(target, assemble_rgb(record, decoded))
            written += 1
    return {
        "schema": RGB_SCHEMA,
        "source_manifest": str(manifest_v2),
        "source_manifest_sha256": sha256_file(manifest_v2),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "episode_count": len(selected_episodes),
        "written": written,
        "reused": reused,
    }


def finalize_manifest(
    manifest_v2: Path,
    output_root: Path,
    manifest_v3: Path,
) -> dict:
    records = read_jsonl(manifest_v2)
    source_sha256 = sha256_file(manifest_v2)
    serialized = []
    for record in records:
        validate_v2_record(record)
        relative = rgb_relative_path(record)
        validate_rgb_cache(output_root / Path(relative.as_posix()))
        updated = dict(record)
        updated["manifest_version"] = 3
        updated["rgb_cache_schema"] = RGB_SCHEMA
        updated["rgb_relative_path"] = relative.as_posix()
        updated["source_manifest_sha256"] = source_sha256
        serialized.append(json.dumps(updated, sort_keys=True, ensure_ascii=False))
    payload = "\n".join(serialized) + "\n"

    manifest_v3.parent.mkdir(parents=True, exist_ok=True)
    if manifest_v3.exists():
        existing = manifest_v3.read_text(encoding="utf-8")
        if existing != payload:
            raise FileExistsError(
                f"refusing to overwrite different Manifest v3: {manifest_v3}"
            )
        return {
            "manifest_v3": str(manifest_v3),
            "sample_count": len(records),
            "reused": True,
        }

    temporary = manifest_v3.with_suffix(
        manifest_v3.suffix + f".tmp-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_v3)
    return {
        "manifest_v3": str(manifest_v3),
        "manifest_v3_sha256": sha256_file(manifest_v3),
        "sample_count": len(records),
        "reused": False,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument("--manifest-v2", type=Path, required=True)
    cache_parser.add_argument("--output-root", type=Path, required=True)
    cache_parser.add_argument("--shard-index", type=int, default=0)
    cache_parser.add_argument("--shard-count", type=int, default=1)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--manifest-v2", type=Path, required=True)
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.add_argument("--manifest-v3", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_v2 = args.manifest_v2.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.command == "cache":
        result = cache_records(
            manifest_v2,
            output_root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    else:
        result = finalize_manifest(
            manifest_v2,
            output_root,
            args.manifest_v3.expanduser().resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
