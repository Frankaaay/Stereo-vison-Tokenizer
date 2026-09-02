"""Build node-local Hy, LIBERO, or UMI episode manifests.

This is a CPU-only inventory pass. It never decodes video or writes below a
dataset root; only the explicit output JSONL and its summary are created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


OFFSETS_AND_STRIDE = {
    "hy": ((0, 3, 6, 9), 12),
    "libero": ((0, 2, 4, 6), 8),
    "umi": ((0, 3, 6, 9), 12),
}

HY_CAMERA_COLUMNS = {
    "cam_high": "observation_images_cam_high",
    "cam_left_wrist": "observation_images_cam_left_wrist",
    "cam_right_wrist": "observation_images_cam_right_wrist",
}
HY_CANONICAL_CAMERA_COLUMNS = {
    "cam_high": "observation_images_cam_head",
    "cam_left_wrist": "observation_images_cam_left_wrist",
    "cam_right_wrist": "observation_images_cam_right_wrist",
}
HY_CANONICAL_MASK = Path(
    "dataset_configs/masks/image_pixel_mask_hy_embodied.npz"
)


def _window_count(length, dataset_id):
    offsets, stride = OFFSETS_AND_STRIDE[dataset_id]
    return max(0, (int(length) - 1 - offsets[-1]) // stride + 1)


def _split(identity):
    bucket = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % 100
    return "train" if bucket < 98 else "val"


def _digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hy_camera_contract(root, schema_names):
    schema_names = set(schema_names)
    if set(HY_CAMERA_COLUMNS.values()).issubset(schema_names):
        return dict(HY_CAMERA_COLUMNS), None
    if not set(HY_CANONICAL_CAMERA_COLUMNS.values()).issubset(schema_names):
        missing = set(HY_CANONICAL_CAMERA_COLUMNS.values()).difference(schema_names)
        raise ValueError(f"missing Hy mono camera columns {sorted(missing)}")
    mask_path = root / HY_CANONICAL_MASK
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    with np.load(mask_path) as payload:
        mask = payload["mask"]
    if mask.shape != (256, 256) or mask.dtype != np.bool_:
        raise ValueError("canonical Hy mask must be bool [256,256]")
    y, x = np.where(mask)
    bbox = [int(y.min()), int(x.min()), int(y.max()) + 1, int(x.max()) + 1]
    if bbox != [55, 0, 200, 256] or not mask[55:200, :].all() or int(mask.sum()) != 37120:
        raise ValueError("canonical Hy mask content rectangle mismatch")
    return dict(HY_CANONICAL_CAMERA_COLUMNS), {
        "encoded_size_hw": [256, 256],
        "source_size_hw": [240, 424],
        "transform": "source_240x424_letterbox_256",
        "content_bbox_yxyx": bbox,
        "pixel_mask_relative_path": HY_CANONICAL_MASK.as_posix(),
        "pixel_mask_sha256": _sha256_file(mask_path),
    }


def _aliases(values):
    output = {}
    for value in values:
        alias, separator, raw_path = value.partition("=")
        if not separator or not alias or not raw_path:
            raise ValueError("roots must use alias=/absolute/path")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        output[alias] = path
    return output


def build_hy(roots):
    try:
        import lance
        import pyarrow.dataset as ds
    except ImportError as error:
        raise ImportError("Hy manifest generation requires pylance and pyarrow") from error
    for alias, root in roots.items():
        for table_root in sorted(root.glob("table_*")):
            table_name = table_root.name
            lance_path = table_root / f"{table_name}.lance"
            meta_root = table_root / "meta" / "episodes"
            if not lance_path.is_dir() or not meta_root.is_dir():
                continue
            camera_columns, stored_image = _hy_camera_contract(
                root, lance.dataset(str(lance_path)).schema.names
            )
            metadata = ds.dataset(str(meta_root), format="parquet").to_table(
                columns=[
                    "episode_index",
                    "length",
                    "dataset_from_index",
                    "dataset_to_index",
                ]
            )
            for row in metadata.to_pylist():
                length = int(row["length"])
                if int(row["dataset_to_index"]) - int(row["dataset_from_index"]) != length:
                    raise ValueError(f"{table_name}: inconsistent episode row range")
                episode_id = f"{table_name}:{int(row['episode_index'])}"
                contract = {
                    "root_alias": alias,
                    "table_name": table_name,
                    "episode_index": int(row["episode_index"]),
                    "length": length,
                    "dataset_from_index": int(row["dataset_from_index"]),
                    "camera_columns": camera_columns,
                    "fps": 30.0,
                }
                if stored_image is not None:
                    contract["stored_image"] = stored_image
                yield {
                    "schema": "hy-mono-three-camera-episode-v2",
                    "split": _split(episode_id),
                    "episode_id": episode_id,
                    "window_count": _window_count(length, "hy"),
                    "source_contract_sha256": _digest(contract),
                    **contract,
                }


def build_libero(roots):
    for alias, root in roots.items():
        candidates = [root] if (root / "meta" / "info.json").is_file() else sorted(root.iterdir())
        for suite_root in candidates:
            info_path = suite_root / "meta" / "info.json"
            episodes_path = suite_root / "meta" / "episodes.jsonl"
            if not info_path.is_file() or not episodes_path.is_file():
                continue
            info = json.loads(info_path.read_text(encoding="utf-8"))
            fps = float(info["fps"])
            chunks_size = int(info.get("chunks_size", 1000))
            video_path = info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
            for line in episodes_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                episode = json.loads(line)
                episode_index = int(episode["episode_index"])
                length = int(episode["length"])
                episode_id = f"{suite_root.name}:{episode_index}"
                contract = {
                    "root_alias": alias,
                    "suite": "." if suite_root == root else suite_root.name,
                    "episode_index": episode_index,
                    "length": length,
                    "fps": fps,
                    "chunks_size": chunks_size,
                    "video_path": video_path,
                    "camera_keys": [
                        "observation.images.image",
                        "observation.images.wrist_image",
                    ],
                }
                yield {
                    "schema": "libero-mono-episode-v1",
                    "split": _split(episode_id),
                    "episode_id": episode_id,
                    "window_count": _window_count(length, "libero"),
                    "source_contract_sha256": _digest(contract),
                    **contract,
                }


def _umi_calibration(sidecar):
    output = {}
    for view, sidecar_view in (
        ("head", "head"),
        ("lefthand", "left_wrist"),
        ("righthand", "right_wrist"),
    ):
        output[view] = {}
        for eye in ("left", "right"):
            camera = json.loads(sidecar[f"camera_{sidecar_view}_{eye}"])
            output[view][eye] = {
                "K": camera["k"],
                "D": camera["d"],
                "R": camera["r"],
                "P": camera["p"],
                "width": int(camera["width"]),
                "height": int(camera["height"]),
                "distortion_model": camera["distortion_model"],
            }
    return output


def _umi_topic_key(topic):
    for view, tokens in (
        ("head", ("coracam_head", "/camera/head/")),
        ("lefthand", ("coracam_lefthand", "/camera/left_wrist/")),
        ("righthand", ("coracam_righthand", "/camera/right_wrist/")),
    ):
        for eye in ("left", "right"):
            if any(token in topic for token in tokens) and (
                f"{eye}_h264" in topic or f"/{eye}/video_encoded" in topic
            ):
                return f"{view}/{eye}"
    return None


def build_umi(roots):
    try:
        from mcap.reader import make_reader
    except ImportError as error:
        raise ImportError("UMI manifest generation requires mcap") from error
    required_topics = {
        f"{view}/{eye}"
        for view in ("head", "lefthand", "righthand")
        for eye in ("left", "right")
    }
    for alias, root in roots.items():
        for mcap_path in sorted(root.glob("*/*/episode.mcap")):
            sidecar_path = mcap_path.parent / f"{mcap_path.parent.name}.json"
            if not sidecar_path.is_file():
                continue
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.get("task.review.status") != "Accepted":
                continue
            frames = sidecar.get("frames")
            if isinstance(frames, str):
                frames = json.loads(frames)
            if isinstance(frames, dict) and frames.get("status") != "done":
                continue
            try:
                calibration = _umi_calibration(sidecar)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            counts = Counter()
            topics = {}
            with mcap_path.open("rb") as stream:
                for _, channel, _ in make_reader(stream).iter_messages():
                    key = _umi_topic_key(channel.topic)
                    if key is not None:
                        if key in topics and topics[key] != channel.topic:
                            raise ValueError(f"{mcap_path}: ambiguous {key} topics")
                        topics[key] = channel.topic
                        counts[key] += 1
            if set(topics) != required_topics:
                continue
            length = min(counts.values())
            episode_id = mcap_path.parent.name
            contract = {
                "root_alias": alias,
                "mcap_relative_path": mcap_path.relative_to(root).as_posix(),
                "length": length,
                "fps": 30.0,
                "topics": topics,
                "calibration": calibration,
                "maximum_pair_skew_s": 1.0 / 60.0,
            }
            yield {
                "schema": "umi-raw-stereo-episode-v1",
                "split": _split(episode_id),
                "episode_id": episode_id,
                "window_count": _window_count(length, "umi"),
                "source_contract_sha256": _digest(contract),
                **contract,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("hy", "libero", "umi"))
    parser.add_argument("--root", action="append", required=True, help="alias=/absolute/path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = _aliases(args.root)
    records = list({"hy": build_hy, "libero": build_libero, "umi": build_umi}[args.dataset](roots))
    if not records:
        raise RuntimeError("inventory produced no accepted episodes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "dataset": args.dataset,
        "records": len(records),
        "windows": sum(record["window_count"] for record in records),
        "split_records": dict(Counter(record["split"] for record in records)),
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
