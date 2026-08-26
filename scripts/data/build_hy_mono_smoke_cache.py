#!/usr/bin/env python3
"""Build an immutable node-local Hy cam_high RGB smoke cache from Lance."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


SCHEMA = "hy-mono-cam-high-smoke-v1"
DATASET_ID = "hy_embodied_0_5_vla"
CAMERA_KEY = "observation.images.cam_high"
LANCE_CAMERA_COLUMN = "observation_images_cam_high"
SOURCE_HW = (240, 424)
OFFSETS = (0, 3, 6, 9)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _publish_immutable(path: Path, payload: bytes) -> None:
    """Atomically publish bytes while accepting only an identical existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary artifact exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"refusing to overwrite mismatched artifact: {path}")
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError(
                        f"concurrent mismatched artifact appeared: {path}"
                    )
    finally:
        temporary.unlink(missing_ok=True)


def _publish_npz(path: Path, arrays: dict[str, np.ndarray]) -> tuple[str, int]:
    archive = io.BytesIO()
    with zipfile.ZipFile(
        archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as output:
        for name in sorted(arrays):
            array_payload = io.BytesIO()
            np.lib.format.write_array(
                array_payload, np.asanyarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            output.writestr(info, array_payload.getvalue(), compresslevel=6)
    payload = archive.getvalue()
    _publish_immutable(path, payload)
    return _sha256_bytes(payload), len(payload)


def _present_tables(root: Path) -> list[str]:
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("table_")
    )


def _episode_metadata_paths(table_root: Path) -> list[Path]:
    paths = sorted((table_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"episode metadata missing under {table_root}")
    return paths


def _build_inventory(root: Path, tables_json_sha256: str):
    import lance
    import pyarrow.parquet as pq

    entries = []
    episode_rows = []
    schemas = set()
    for table_name in _present_tables(root):
        table_root = root / table_name
        lance_path = table_root / f"{table_name}.lance"
        info_path = table_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        dataset = lance.dataset(str(lance_path))
        schema = tuple((field.name, str(field.type)) for field in dataset.schema)
        schemas.add(schema)
        metadata_paths = _episode_metadata_paths(table_root)
        metadata = pq.read_table(metadata_paths)
        required = {
            "episode_index",
            "length",
            "dataset_from_index",
            "dataset_to_index",
        }
        if not required.issubset(metadata.column_names):
            raise ValueError(f"{table_name}: incompatible episode metadata schema")
        for row in metadata.select(sorted(required)).to_pylist():
            episode_rows.append(
                {
                    "table_name": table_name,
                    "episode_index": int(row["episode_index"]),
                    "length": int(row["length"]),
                    "dataset_from_index": int(row["dataset_from_index"]),
                    "dataset_to_index": int(row["dataset_to_index"]),
                }
            )
        entries.append(
            {
                "table_name": table_name,
                "resolved_table_root": str(table_root.resolve(strict=True)),
                "lance_version": int(dataset.version),
                "row_count": int(dataset.count_rows()),
                "info_json_sha256": _sha256_file(info_path),
                "episode_metadata": [
                    {
                        "relative_path": str(path.relative_to(root)),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                    for path in metadata_paths
                ],
            }
        )
    if len(schemas) != 1:
        raise ValueError("present tables do not share one Lance schema")
    schema = list(schemas)[0]
    required_lance = {
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        LANCE_CAMERA_COLUMN,
    }
    if not required_lance.issubset(name for name, _ in schema):
        raise ValueError("Lance schema is missing required Hy smoke columns")
    inventory = {
        "tables_json_sha256": tables_json_sha256,
        "present_tables": entries,
        "lance_schema": [{"name": name, "type": kind} for name, kind in schema],
    }
    return inventory, episode_rows


def _decode_rgb(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "JPEG":
            raise ValueError(f"expected JPEG payload, got {image.format!r}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (*SOURCE_HW, 3):
        raise ValueError(f"expected RGB {SOURCE_HW + (3,)}, got {rgb.shape}")
    return np.transpose(rgb, (2, 0, 1)).copy()


def _validated_window(dataset, episode, start_frame: int, fps: float):
    row_indices = [
        episode["dataset_from_index"] + start_frame + offset for offset in OFFSETS
    ]
    table = dataset.take(
        row_indices,
        columns=[
            "episode_index",
            "frame_index",
            "index",
            "timestamp",
            LANCE_CAMERA_COLUMN,
        ],
    )
    rows = table.to_pylist()
    expected_frames = [start_frame + offset for offset in OFFSETS]
    episode_indices = [int(row["episode_index"]) for row in rows]
    frame_indices = [int(row["frame_index"]) for row in rows]
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], np.float64)
    if episode_indices != [episode["episode_index"]] * len(OFFSETS):
        raise ValueError("window crosses an episode boundary")
    if frame_indices != expected_frames:
        raise ValueError(f"frame indices {frame_indices} != {expected_frames}")
    if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps are non-finite or not strictly increasing")
    expected_timestamps = np.asarray(frame_indices, np.float64) / fps
    tolerance_s = max(5e-6, 1e-4 / fps)
    if not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=tolerance_s):
        raise ValueError("timestamps are inconsistent with frame_index / FPS")
    rgb = np.stack([_decode_rgb(row[LANCE_CAMERA_COLUMN]) for row in rows])
    return rgb, np.asarray(frame_indices, np.int64), timestamps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.sample_count < 1 or args.fps <= 0:
        raise ValueError("sample count and FPS must be positive")

    import lance

    root = args.root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    if output_root == root or root in output_root.parents:
        raise ValueError("output root must not be inside the source overlay")
    tables_json_path = root / "tables.json"
    tables_json = json.loads(tables_json_path.read_text(encoding="utf-8"))
    if tables_json.get("format") != "lerobot-lancedb-v3.0":
        raise ValueError("unexpected Hy table format")
    tables_json_sha256 = _sha256_file(tables_json_path)
    inventory, episodes = _build_inventory(root, tables_json_sha256)
    inventory_sha256 = _sha256_bytes(_canonical_json(inventory))
    source_contract = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "source_format": "lerobot-lancedb-v3.0",
        "camera_key": CAMERA_KEY,
        "lance_camera_column": LANCE_CAMERA_COLUMN,
        "fps": args.fps,
        "source_hw": list(SOURCE_HW),
        "frame_offsets": list(OFFSETS),
        "tables_json_sha256": tables_json_sha256,
        "table_inventory_sha256": inventory_sha256,
    }
    source_contract_sha256 = _sha256_bytes(_canonical_json(source_contract))

    eligible = [episode for episode in episodes if episode["length"] > OFFSETS[-1]]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    datasets = {}
    records = []
    skip_reasons = Counter()
    for episode in eligible:
        if len(records) == args.sample_count:
            break
        maximum_start = episode["length"] - 1 - OFFSETS[-1]
        start_frame = rng.randint(0, maximum_start)
        table_name = episode["table_name"]
        try:
            dataset = datasets.get(table_name)
            if dataset is None:
                dataset = lance.dataset(
                    str(root / table_name / f"{table_name}.lance")
                )
                datasets[table_name] = dataset
            rgb, frame_indices, timestamps = _validated_window(
                dataset, episode, start_frame, args.fps
            )
        except Exception as error:
            skip_reasons[f"{type(error).__name__}: {error}"] += 1
            continue

        episode_id = f"episode_{episode['episode_index']}"
        sample_id = (
            f"hy/{table_name}/{episode_id}/cam_high/frame_{start_frame}"
        )
        sample_hash = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        relative_path = Path("rgb") / f"{sample_hash}.npz"
        metadata = {
            "schema": SCHEMA,
            "sample_id": sample_id,
            "table_name": table_name,
            "episode_index": episode["episode_index"],
            "camera_key": CAMERA_KEY,
            "source_contract_sha256": source_contract_sha256,
            "table_inventory_sha256": inventory_sha256,
        }
        rgb_sha256, rgb_size_bytes = _publish_npz(
            output_root / relative_path,
            {
                "rgb": rgb,
                "frame_index": frame_indices,
                "timestamp_s": timestamps,
                "metadata_json": np.asarray(
                    json.dumps(metadata, separators=(",", ":"), sort_keys=True)
                ),
            },
        )
        records.append(
            {
                "schema": SCHEMA,
                "sample_id": sample_id,
                "dataset_id": DATASET_ID,
                "table_name": table_name,
                "episode_id": episode_id,
                "episode_index": episode["episode_index"],
                "camera_key": CAMERA_KEY,
                "start_frame": start_frame,
                "frame_indices": frame_indices.tolist(),
                "timestamps_s": timestamps.tolist(),
                "timestamp_source": "source_row",
                "fps": args.fps,
                "source_hw": list(SOURCE_HW),
                "split": "smoke",
                "rgb_relative_path": relative_path.as_posix(),
                "rgb_sha256": rgb_sha256,
                "rgb_size_bytes": rgb_size_bytes,
                "source_contract_sha256": source_contract_sha256,
                "table_inventory_sha256": inventory_sha256,
                "preprocess": {
                    "storage": "raw_rgb_uint8",
                    "letterboxed": False,
                },
            }
        )
    if len(records) != args.sample_count:
        raise RuntimeError(
            f"selected only {len(records)}/{args.sample_count} valid windows; "
            f"skip reasons: {dict(skip_reasons)}"
        )

    manifest_payload = b"".join(
        _canonical_json(record) + b"\n" for record in records
    )
    manifest_sha256 = _sha256_bytes(manifest_payload)
    summary = {
        "schema": SCHEMA,
        "seed": args.seed,
        "sample_count": len(records),
        "selected_episode_count": len(
            {(record["table_name"], record["episode_index"]) for record in records}
        ),
        "present_tables": [item["table_name"] for item in inventory["present_tables"]],
        "tables_json_sha256": tables_json_sha256,
        "table_inventory": inventory,
        "table_inventory_sha256": inventory_sha256,
        "source_contract": source_contract,
        "source_contract_sha256": source_contract_sha256,
        "manifest_sha256": manifest_sha256,
        "rgb_total_bytes": sum(record["rgb_size_bytes"] for record in records),
        "skip_reasons": dict(skip_reasons),
    }
    _publish_immutable(output_root / "manifest.jsonl", manifest_payload)
    _publish_immutable(
        output_root / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
