#!/usr/bin/env python3
"""Build a deterministic episode-level LeRobot StereoVAE manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from stereo_tokenizer.lerobot_data import (  # noqa: E402
    EYES,
    FRAME_OFFSETS,
    FPS,
    OUTPUT_HW,
    PADDING_LTRB,
    RESIZE_HW,
    SCHEMA,
    SOURCE_HW,
    START_STRIDE,
    VIDEO_KEYS,
    VIEWS,
    sha256_file,
    validate_calibration,
    window_count,
)


AUDIT_SCHEMA = "lerobot-stereo-rectification-audit-v1"
SOURCE_CAMERA_KEYS = {
    ("head", "left"): "camera_head_left",
    ("head", "right"): "camera_head_right",
    ("lefthand", "left"): "camera_left_wrist_left",
    ("lefthand", "right"): "camera_left_wrist_right",
    ("righthand", "left"): "camera_right_wrist_left",
    ("righthand", "right"): "camera_right_wrist_right",
}


def _sha256_json(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_failures(dataset_root: Path, shard_id: str):
    path = dataset_root / f"{shard_id}.failures.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = set()
    for row in payload:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{path}: malformed failure row")
        failures.add(row[0])
    return failures


def _source_episode(
    old_mcap_path: str,
    source_manifest_prefix: str,
    source_root: Path,
) -> Path:
    if not old_mcap_path.startswith(source_manifest_prefix):
        raise ValueError(
            f"source path does not start with {source_manifest_prefix}: "
            f"{old_mcap_path}"
        )
    relative = old_mcap_path[len(source_manifest_prefix) :]
    corrected_mcap = (source_root / relative).resolve()
    if not corrected_mcap.is_relative_to(source_root):
        raise ValueError(f"source path escapes source root: {old_mcap_path}")
    return corrected_mcap.parent.parent


def _camera(payload, key):
    raw = payload.get(key)
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"missing camera calibration {key}")
    return {
        "K": raw["k"],
        "D": raw["d"],
        "R": raw["r"],
        "P": raw["p"],
        "width": int(raw["width"]),
        "height": int(raw["height"]),
        "distortion_model": raw["distortion_model"],
    }


def _calibration(source_json: Path, episode_id: str):
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    result = {}
    for view in VIEWS:
        result[view] = {
            eye: _camera(payload, SOURCE_CAMERA_KEYS[(view, eye)])
            for eye in EYES
        }
    validate_calibration(result, episode_id)
    return result


def _load_rectification_audit(path: Path, dataset_root: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != AUDIT_SCHEMA:
        raise ValueError(f"{path}: unsupported rectification audit schema")
    if payload.get("result") != "pass":
        raise ValueError(f"{path}: rectification audit did not pass")
    mode = payload.get("selected_mode")
    if mode not in {"verified_pre_rectified", "apply_calibration"}:
        raise ValueError(f"{path}: invalid selected rectification mode")
    if Path(payload.get("dataset_root", "")).resolve() != dataset_root:
        raise ValueError(f"{path}: dataset root does not match")
    if int(payload.get("representative_pair_count", 0)) < 1:
        raise ValueError(f"{path}: no representative stereo pairs were audited")
    return payload, sha256_file(path)


def _episode_rows(shard: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("manifest construction requires pyarrow") from error
    path = shard / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    return table.to_pylist()


def _video_record(
    dataset_root: Path, shard_id: str, row, video_key: str, episode_length: int
):
    chunk = int(row[f"videos/{video_key}/chunk_index"])
    file_index = int(row[f"videos/{video_key}/file_index"])
    relative = Path(
        shard_id,
        "videos",
        video_key,
        f"chunk-{chunk:03d}",
        f"file-{file_index:03d}.mp4",
    )
    path = dataset_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    start = float(row[f"videos/{video_key}/from_timestamp"])
    stop = float(row[f"videos/{video_key}/to_timestamp"])
    if not 0 <= start < stop:
        raise ValueError(f"invalid video interval in {shard_id}: {video_key}")
    last_frame_time = start + (episode_length - 1) / FPS
    if last_frame_time > stop + 0.05:
        raise ValueError(
            f"video interval is too short in {shard_id}: {video_key}"
        )
    return {
        "relative_path": relative.as_posix(),
        "from_timestamp": start,
        "to_timestamp": stop,
    }


def collect_records(args, rectification_mode, audit_sha256):
    dataset_root = args.dataset_root.resolve()
    source_root = args.source_root.resolve()
    records = []
    input_count = 0
    failure_count = 0
    shards = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("shard_")
    )
    for shard in shards:
        shard_id = shard.name
        manifest_number = int(shard_id.split("_")[1])
        source_manifest = dataset_root / "_manifests" / f"m_{manifest_number:04d}"
        source_paths = [
            line
            for line in source_manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
        failures = _read_failures(dataset_root, shard_id)
        successful_sources = [path for path in source_paths if path not in failures]
        episode_rows = _episode_rows(shard)
        input_count += len(source_paths)
        failure_count += len(failures)
        if len(successful_sources) != len(episode_rows):
            raise ValueError(
                f"{shard_id}: {len(successful_sources)} successful sources but "
                f"{len(episode_rows)} converted episodes"
            )

        for source_path, row in zip(successful_sources, episode_rows):
            source_episode = _source_episode(
                source_path,
                args.source_manifest_prefix,
                source_root,
            )
            episode_id = source_episode.name
            source_json = source_episode / f"{episode_id}.json"
            if not source_json.is_file():
                raise FileNotFoundError(source_json)
            if int(row["episode_index"]) < 0:
                raise ValueError(f"{shard_id}: negative episode index")
            length = int(row["length"])
            count = window_count(length)
            if count == 0:
                continue
            calibration = _calibration(source_json, episode_id)
            videos = {
                key: _video_record(dataset_root, shard_id, row, key, length)
                for key in VIDEO_KEYS.values()
            }
            raw_tasks = row.get("tasks", [])
            if isinstance(raw_tasks, str):
                tasks = [raw_tasks]
            elif isinstance(raw_tasks, (list, tuple)):
                tasks = [str(task) for task in raw_tasks]
            else:
                raise ValueError(f"{shard_id}/{episode_id}: invalid tasks field")
            records.append(
                {
                    "schema": SCHEMA,
                    "episode_id": episode_id,
                    "shard_id": shard_id,
                    "episode_index": int(row["episode_index"]),
                    "tasks": tasks,
                    "length": length,
                    "window_count": count,
                    "source_episode_json": str(source_json),
                    "source_episode_json_sha256": sha256_file(source_json),
                    "videos": videos,
                    "calibration": calibration,
                    "rectification": {
                        "mode": rectification_mode,
                        "audit_sha256": audit_sha256,
                    },
                }
            )
    episode_ids = [record["episode_id"] for record in records]
    duplicate_ids = [
        episode_id
        for episode_id, count in Counter(episode_ids).items()
        if count != 1
    ]
    if duplicate_ids:
        raise ValueError(
            "duplicate episode identities would violate split isolation: "
            + ", ".join(sorted(duplicate_ids)[:10])
        )
    return records, {
        "shard_count": len(shards),
        "source_input_count": input_count,
        "source_failure_count": failure_count,
    }


def assign_splits(records, seed):
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['episode_id']}".encode("utf-8")
        ).digest(),
    )
    train_count = int(len(ordered) * 0.90)
    val_count = int(len(ordered) * 0.05)
    for index, record in enumerate(ordered):
        if index < train_count:
            record["split"] = "train"
        elif index < train_count + val_count:
            record["split"] = "val"
        else:
            record["split"] = "test"


def write_outputs(args, records, inventory, audit_sha256):
    preprocessing = {
        "source_size_hw": list(SOURCE_HW),
        "resize_size_hw": list(RESIZE_HW),
        "output_size_hw": list(OUTPUT_HW),
        "padding_ltrb": list(PADDING_LTRB),
        "scale_xy": [0.4, 0.4],
        "fps": FPS,
        "sample_start_stride_frames": START_STRIDE,
        "frame_offsets": list(FRAME_OFFSETS),
    }
    contract = {
        "schema": SCHEMA,
        "dataset_root": str(args.dataset_root.resolve()),
        "source_root": str(args.source_root.resolve()),
        "source_manifest_prefix": args.source_manifest_prefix,
        "split_seed": args.split_seed,
        "split_ratios": [0.90, 0.05, 0.05],
        "shuffle_granularity": "shard_then_episode",
        "read_granularity": "episode_time_order",
        "model_input_granularity": "four_frame_sample",
        "preprocessing": preprocessing,
        "rectification_audit_sha256": audit_sha256,
    }
    contract_sha256 = _sha256_json(contract)
    for record in records:
        record["contract_sha256"] = contract_sha256

    manifest_path = args.output_manifest.resolve()
    summary_path = args.output_summary.resolve()
    for path in (manifest_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    manifest_text = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        for record in sorted(
            records,
            key=lambda record: (record["shard_id"], record["episode_index"]),
        )
    )
    manifest_temporary = manifest_path.with_name(
        manifest_path.name + f".tmp-{os.getpid()}"
    )
    summary_temporary = summary_path.with_name(
        summary_path.name + f".tmp-{os.getpid()}"
    )
    try:
        manifest_temporary.write_text(manifest_text, encoding="utf-8", newline="\n")
        manifest_sha256 = sha256_file(manifest_temporary)
        split_episodes = Counter(record["split"] for record in records)
        split_samples = Counter()
        for record in records:
            split_samples[record["split"]] += record["window_count"]
        summary = {
            "schema": SCHEMA,
            "contract": contract,
            "contract_sha256": contract_sha256,
            "manifest_sha256": manifest_sha256,
            "inventory": inventory,
            "episode_count": len(records),
            "sample_count": sum(record["window_count"] for record in records),
            "split_episode_counts": dict(split_episodes),
            "split_sample_counts": dict(split_samples),
        }
        summary_temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(manifest_temporary, manifest_path)
        os.replace(summary_temporary, summary_path)
    except BaseException:
        for path in (manifest_temporary, summary_temporary):
            if path.exists():
                path.unlink()
        raise
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest-prefix",
        default="/data/umi_vio_data_260714/",
    )
    parser.add_argument("--rectification-audit", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    if not dataset_root.is_dir() or not source_root.is_dir():
        raise FileNotFoundError("dataset and source roots must exist")
    args.dataset_root = dataset_root
    args.source_root = source_root
    audit, audit_sha256 = _load_rectification_audit(
        args.rectification_audit.expanduser().resolve(), dataset_root
    )
    records, inventory = collect_records(
        args, audit["selected_mode"], audit_sha256
    )
    assign_splits(records, args.split_seed)
    summary = write_outputs(args, records, inventory, audit_sha256)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
