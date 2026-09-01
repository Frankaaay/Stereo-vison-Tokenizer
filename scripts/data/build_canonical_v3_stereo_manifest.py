#!/usr/bin/env python3
"""Build a deterministic JSONL index for canonical-v3 stereo ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


SCHEMA = "canonical-v3-stereo-ablation-v1"
VIDEO_KEYS = {
    ("head", "left"): "observation.images.cam_head_left",
    ("head", "right"): "observation.images.cam_head_right",
    ("lefthand", "left"): "observation.images.cam_left_wrist_left",
    ("lefthand", "right"): "observation.images.cam_left_wrist_right",
    ("righthand", "left"): "observation.images.cam_right_wrist_left",
    ("righthand", "right"): "observation.images.cam_right_wrist_right",
}


def window_count(length):
    return max(0, (int(length) - 1 - 9) // 12 + 1)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def split_for_episode(episode_id, seed):
    bucket = int.from_bytes(
        hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()[:8], "big"
    ) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


def source_ids(root):
    database = root / ".ngad" / "publish.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "select source_id, episode_index, file_serial, frames "
            "from episodes order by episode_index"
        ).fetchall()
    return {
        int(episode_index): {
            "source_id": str(source_id),
            "file_serial": int(file_serial),
            "frames": int(frames),
        }
        for source_id, episode_index, file_serial, frames in rows
    }


def episode_rows(root):
    try:
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError(
            "manifest construction requires pyarrow; the evaluation runtime "
            "only consumes the generated JSONL"
        ) from error
    metadata_root = root / "meta" / "episodes"
    if not metadata_root.is_dir():
        raise FileNotFoundError(metadata_root)
    columns = ["episode_index", "length"]
    for key in VIDEO_KEYS.values():
        columns.extend(
            (
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
                f"videos/{key}/to_timestamp",
            )
        )
    table = ds.dataset(str(metadata_root), format="parquet").to_table(
        columns=columns
    )
    return sorted(table.to_pylist(), key=lambda row: int(row["episode_index"]))


def build_records(root, seed):
    info_path = root / "meta" / "info.json"
    provenance_path = root / "meta" / "provenance.json"
    pixel_mask_path = root / "image_pixel_mask_umi.npz"
    for path in (info_path, provenance_path, pixel_mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("fps", -1)) != 30:
        raise ValueError("canonical-v3 ablation requires 30 FPS")
    feature_keys = set(info.get("features", {}))
    missing_video_keys = set(VIDEO_KEYS.values()) - feature_keys
    if missing_video_keys:
        raise ValueError(f"canonical-v3 is missing {sorted(missing_video_keys)}")

    identity = source_ids(root)
    rows = episode_rows(root)
    if len(rows) != len(identity):
        raise ValueError("parquet and publish database episode counts disagree")
    contract = {
        "schema": SCHEMA,
        "dataset_info_sha256": sha256_file(info_path),
        "dataset_provenance_sha256": sha256_file(provenance_path),
        "pixel_mask_sha256": sha256_file(pixel_mask_path),
        "split_seed": int(seed),
        "split_ratios": [0.90, 0.05, 0.05],
        "fps": 30,
        "frame_offsets": [0, 3, 6, 9],
        "start_stride": 12,
        "input_shape": [3, 2, 3, 4, 256, 256],
        "padding_policy": "replace mask-false pixels with uint8 128",
        "camera_scale": "not published; report centers each sample/view",
    }
    contract_sha256 = canonical_digest(contract)
    records = []
    for row in rows:
        episode_index = int(row["episode_index"])
        source = identity.get(episode_index)
        if source is None:
            raise ValueError(f"episode {episode_index} has no source identity")
        length = int(row["length"])
        if source["frames"] != length:
            raise ValueError(f"episode {episode_index}: length mismatch")
        videos = {}
        intervals = []
        for key in VIDEO_KEYS.values():
            chunk = int(row[f"videos/{key}/chunk_index"])
            file_index = int(row[f"videos/{key}/file_index"])
            start = float(row[f"videos/{key}/from_timestamp"])
            stop = float(row[f"videos/{key}/to_timestamp"])
            relative = Path(
                "videos", key, f"chunk-{chunk:03d}", f"file-{file_index:03d}.mp4"
            )
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            if not 0 <= start < stop:
                raise ValueError(f"episode {episode_index}: invalid {key} interval")
            if start + (length - 1) / 30.0 > stop + 0.05:
                raise ValueError(f"episode {episode_index}: short {key} interval")
            intervals.append((start, stop))
            videos[key] = {
                "relative_path": relative.as_posix(),
                "from_timestamp": start,
                "to_timestamp": stop,
            }
        if max(value[0] for value in intervals) - min(
            value[0] for value in intervals
        ) > 1e-6:
            raise ValueError(f"episode {episode_index}: camera starts are not synced")
        if max(value[1] for value in intervals) - min(
            value[1] for value in intervals
        ) > 1e-6:
            raise ValueError(f"episode {episode_index}: camera stops are not synced")
        count = window_count(length)
        if count == 0:
            continue
        records.append(
            {
                "schema": SCHEMA,
                "episode_id": source["source_id"],
                "episode_index": episode_index,
                "file_serial": source["file_serial"],
                "length": length,
                "window_count": count,
                "split": split_for_episode(source["source_id"], seed),
                "videos": videos,
                "contract_sha256": contract_sha256,
            }
        )
    return records, contract, contract_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=1234)
    args = parser.parse_args()
    for path in (args.output_manifest, args.output_summary):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    root = args.dataset_root.expanduser().resolve()
    records, contract, contract_sha256 = build_records(root, args.split_seed)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    splits = Counter(record["split"] for record in records)
    summary = {
        "schema": SCHEMA,
        "dataset_root": str(root),
        "episode_count": len(records),
        "window_count": sum(record["window_count"] for record in records),
        "split_episode_counts": dict(sorted(splits.items())),
        "contract": contract,
        "contract_sha256": contract_sha256,
        "manifest_sha256": sha256_file(args.output_manifest),
    }
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
