#!/usr/bin/env python3
"""Build a deterministic StereoVAE manifest for canonical UMI LeRobot v3."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from stereo_tokenizer.lerobot_data import (  # noqa: E402
    CANONICAL_STORED_TRANSFORM,
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


CANONICAL_VIDEO_KEYS = {
    ("head", "left"): "observation.images.cam_head_left",
    ("head", "right"): "observation.images.cam_head_right",
    ("lefthand", "left"): "observation.images.cam_left_wrist_left",
    ("lefthand", "right"): "observation.images.cam_left_wrist_right",
    ("righthand", "left"): "observation.images.cam_right_wrist_left",
    ("righthand", "right"): "observation.images.cam_right_wrist_right",
}
CALIBRATION_VIEW_KEYS = {
    "head": "head",
    "lefthand": "left_wrist",
    "righthand": "right_wrist",
}
REQUIRED_INPUTS = (
    "episode_index_to_umi_source_calibration.jsonl.gz",
    "calibration_bundles.jsonl.gz",
    "rectification_calibration_audit.json",
    "raw_640x480_to_canonical_256x256_transform.json",
    "provenance.json",
)


def _sha256_json(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def _normalize_calibration(source_calibration, bundle_sha256):
    if not isinstance(source_calibration, dict):
        raise ValueError(f"{bundle_sha256}: calibration must be an object")
    missing = set(CALIBRATION_VIEW_KEYS.values()).difference(source_calibration)
    if missing:
        raise ValueError(
            f"{bundle_sha256}: missing calibration views {sorted(missing)}"
        )
    calibration = {
        manifest_view: source_calibration[source_view]
        for manifest_view, source_view in CALIBRATION_VIEW_KEYS.items()
    }
    validate_calibration(calibration, bundle_sha256)
    return calibration


def _load_inputs(input_root: Path):
    for name in REQUIRED_INPUTS:
        if not (input_root / name).is_file():
            raise FileNotFoundError(input_root / name)

    mappings = {}
    mapping_path = input_root / REQUIRED_INPUTS[0]
    for row in _read_jsonl_gzip(mapping_path):
        episode_index = int(row["episode_index"])
        if episode_index in mappings:
            raise ValueError(f"duplicate episode mapping {episode_index}")
        mappings[episode_index] = row

    bundles = {}
    bundle_path = input_root / REQUIRED_INPUTS[1]
    for row in _read_jsonl_gzip(bundle_path):
        bundle_sha256 = row["calibration_bundle_sha256"]
        if bundle_sha256 in bundles:
            raise ValueError(f"duplicate calibration bundle {bundle_sha256}")
        calibration = _normalize_calibration(row["calibration"], bundle_sha256)
        bundles[bundle_sha256] = calibration

    transform_path = input_root / REQUIRED_INPUTS[3]
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    spatial = transform.get("spatial_transform", {})
    if (
        transform.get("schema_version")
        != "ngad_umi_raw_to_canonical_image_transform_v1"
        or spatial.get("source_height") != SOURCE_HW[0]
        or spatial.get("source_width") != SOURCE_HW[1]
        or spatial.get("resized_height") != RESIZE_HW[0]
        or spatial.get("resized_width") != RESIZE_HW[1]
        or spatial.get("padding_pixels")
        != {
            "left": PADDING_LTRB[0],
            "right": PADDING_LTRB[2],
            "top": PADDING_LTRB[1],
            "bottom": PADDING_LTRB[3],
            "value_rgb": [0, 0, 0],
        }
    ):
        raise ValueError("unsupported canonical UMI image transform")

    provenance_path = input_root / REQUIRED_INPUTS[4]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    mask = provenance.get("image_pixel_mask", {})
    if (
        mask.get("shape") != list(OUTPUT_HW)
        or mask.get("verified_true_region") != "mask[32:224,0:256]"
        or int(mask.get("true_pixel_count", -1)) != 49152
        or len(mask.get("sha256", "")) != 64
    ):
        raise ValueError("unsupported canonical UMI pixel mask")

    hashes = {
        name: sha256_file(input_root / name)
        for name in REQUIRED_INPUTS
    }
    return mappings, bundles, transform, provenance, hashes


def _canonical_table_root(dataset_root: Path) -> Path:
    if (dataset_root / "meta" / "info.json").is_file():
        return dataset_root
    candidates = [
        path
        for path in sorted(dataset_root.glob("table_*"))
        if (path / "meta" / "info.json").is_file()
        and (path / "meta" / "episodes").is_dir()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "canonical UMI dataset root must contain exactly one published table"
        )
    return candidates[0]


def _episode_rows(dataset_root: Path):
    try:
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError("canonical manifest construction requires pyarrow") from error
    columns = ["episode_index", "length", "tasks"]
    for source_key in CANONICAL_VIDEO_KEYS.values():
        columns.extend(
            [
                f"videos/{source_key}/chunk_index",
                f"videos/{source_key}/file_index",
                f"videos/{source_key}/from_timestamp",
                f"videos/{source_key}/to_timestamp",
            ]
        )
    table_root = _canonical_table_root(dataset_root)
    table = ds.dataset(str(table_root / "meta" / "episodes"), format="parquet").to_table(
        columns=columns
    )
    return table.to_pylist()


def _video_record(
    dataset_root: Path, table_root: Path, row, source_key: str, length: int
):
    chunk = int(row[f"videos/{source_key}/chunk_index"])
    file_index = int(row[f"videos/{source_key}/file_index"])
    table_relative = Path(
        "videos", source_key, f"chunk-{chunk:03d}", f"file-{file_index:03d}.mp4"
    )
    relative = table_root.relative_to(dataset_root) / table_relative
    path = dataset_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    start = float(row[f"videos/{source_key}/from_timestamp"])
    stop = float(row[f"videos/{source_key}/to_timestamp"])
    if not 0 <= start < stop:
        raise ValueError(f"invalid video interval for {source_key}")
    if start + (length - 1) / FPS > stop + 0.05:
        raise ValueError(f"video interval too short for {source_key}")
    return {
        "relative_path": relative.as_posix(),
        "from_timestamp": start,
        "to_timestamp": stop,
    }


def collect_records(dataset_root: Path, mappings, bundles, audit_sha256: str, mask):
    records = []
    table_root = _canonical_table_root(dataset_root)
    seen_episode_indices = set()
    seen_episode_ids = set()
    for row in _episode_rows(dataset_root):
        episode_index = int(row["episode_index"])
        if episode_index in seen_episode_indices:
            raise ValueError(f"duplicate canonical episode index {episode_index}")
        seen_episode_indices.add(episode_index)
        if episode_index not in mappings:
            raise ValueError(f"missing source mapping for episode {episode_index}")
        mapping = mappings[episode_index]
        episode_id = str(mapping["episode_uuid"])
        if episode_id in seen_episode_ids:
            raise ValueError(f"duplicate canonical episode UUID {episode_id}")
        seen_episode_ids.add(episode_id)
        calibration_sha256 = mapping["calibration_bundle_sha256"]
        if calibration_sha256 not in bundles:
            raise ValueError(f"missing calibration bundle {calibration_sha256}")
        length = int(row["length"])
        count = window_count(length)
        if count == 0:
            continue
        videos = {}
        shard_coordinates = set()
        for pair, manifest_key in VIDEO_KEYS.items():
            source_key = CANONICAL_VIDEO_KEYS[pair]
            videos[manifest_key] = _video_record(
                dataset_root, table_root, row, source_key, length
            )
            shard_coordinates.add(
                (
                    int(row[f"videos/{source_key}/chunk_index"]),
                    int(row[f"videos/{source_key}/file_index"]),
                )
            )
        if len(shard_coordinates) != 1:
            raise ValueError(f"episode {episode_index}: six videos do not share a shard")
        chunk, file_index = shard_coordinates.pop()
        tasks = row.get("tasks", [])
        if isinstance(tasks, str):
            tasks = [tasks]
        elif isinstance(tasks, (list, tuple)):
            tasks = [str(task) for task in tasks]
        else:
            raise ValueError(f"episode {episode_index}: invalid tasks")
        records.append(
            {
                "schema": SCHEMA,
                "episode_id": episode_id,
                "shard_id": f"canonical_{chunk:03d}_{file_index:03d}",
                "episode_index": episode_index,
                "tasks": tasks,
                "length": length,
                "window_count": count,
                "source_episode_json": mapping["source_sidecar"],
                "source_episode_json_sha256": mapping["sidecar_sha256"],
                "calibration_bundle_sha256": calibration_sha256,
                "videos": videos,
                "calibration": bundles[calibration_sha256],
                "rectification": {
                    "mode": "verified_pre_rectified",
                    "status": "data_side_confirmed_by_user",
                    "source_audit_result": "metadata_supported_and_data_side_confirmed",
                    "audit_sha256": audit_sha256,
                },
                "stored_image": {
                    "encoded_size_hw": list(OUTPUT_HW),
                    "transform": CANONICAL_STORED_TRANSFORM,
                    "source_size_hw": list(SOURCE_HW),
                    "resize_size_hw": list(RESIZE_HW),
                    "padding_ltrb": list(PADDING_LTRB),
                    "pixel_mask_relative_path": Path(mask["path"]).name,
                    "pixel_mask_sha256": mask["sha256"],
                },
            }
        )
    if set(mappings) != seen_episode_indices:
        missing = sorted(set(mappings) - seen_episode_indices)
        raise ValueError(f"mapping contains absent canonical episodes: {missing[:10]}")
    return records


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


def write_outputs(args, records, input_hashes, provenance):
    audit_sha256 = input_hashes["rectification_calibration_audit.json"]
    contract = {
        "schema": SCHEMA,
        "dataset_root": str(args.dataset_root),
        "input_root": str(args.input_root),
        "input_sha256": input_hashes,
        "split_seed": args.split_seed,
        "split_ratios": [0.90, 0.05, 0.05],
        "shuffle_granularity": "canonical_video_file_then_episode",
        "read_granularity": "episode_time_order",
        "model_input_granularity": "four_frame_sample",
        "preprocessing": {
            "source_size_hw": list(SOURCE_HW),
            "stored_size_hw": list(OUTPUT_HW),
            "resize_size_hw": list(RESIZE_HW),
            "padding_ltrb": list(PADDING_LTRB),
            "stored_transform": CANONICAL_STORED_TRANSFORM,
            "fps": FPS,
            "sample_start_stride_frames": START_STRIDE,
            "frame_offsets": list(FRAME_OFFSETS),
        },
        "rectification": {
            "mode": "verified_pre_rectified",
            "status": "data_side_confirmed_by_user",
            "audit_sha256": audit_sha256,
            "note": args.rectification_confirmation,
        },
        "image_pixel_mask": provenance["image_pixel_mask"],
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
    manifest_temporary = manifest_path.with_name(
        manifest_path.name + f".tmp-{os.getpid()}"
    )
    summary_temporary = summary_path.with_name(
        summary_path.name + f".tmp-{os.getpid()}"
    )
    try:
        with manifest_temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in sorted(
                records, key=lambda item: (item["shard_id"], item["episode_index"])
            ):
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
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
            "episode_count": len(records),
            "sample_count": sum(record["window_count"] for record in records),
            "calibration_bundle_count": len(
                {record["calibration_bundle_sha256"] for record in records}
            ),
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
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--rectification-confirmation", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.input_root = args.input_root.expanduser().resolve()
    if not args.dataset_root.is_dir() or not args.input_root.is_dir():
        raise FileNotFoundError("dataset and calibration input roots must exist")
    mappings, bundles, _, provenance, input_hashes = _load_inputs(args.input_root)
    expected_count = int(provenance["audit_episode_count"])
    table_root = _canonical_table_root(args.dataset_root)
    dataset_info = json.loads(
        (table_root / "meta" / "info.json").read_text(encoding="utf-8")
    )
    if int(dataset_info["total_episodes"]) != expected_count:
        raise ValueError("dataset and audit episode counts disagree")
    if len(mappings) != expected_count:
        raise ValueError("mapping and audit episode counts disagree")
    records = collect_records(
        args.dataset_root,
        mappings,
        bundles,
        input_hashes["rectification_calibration_audit.json"],
        provenance["image_pixel_mask"],
    )
    assign_splits(records, args.split_seed)
    summary = write_outputs(args, records, input_hashes, provenance)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
