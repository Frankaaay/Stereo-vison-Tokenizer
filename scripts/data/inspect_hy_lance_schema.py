#!/usr/bin/env python3
"""Read-only Hy Lance schema probe without printing encoded image payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitized(value):
    if isinstance(value, bytes):
        return {
            "python_type": "bytes",
            "byte_length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, dict):
        return {key: _sanitized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitized(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    import lance

    root = args.root.expanduser().resolve(strict=True)
    tables_path = root / "tables.json"
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    present_tables = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("table_")
    )
    if args.table not in present_tables:
        raise ValueError(f"table {args.table!r} is not present on this node")
    lance_path = root / args.table / f"{args.table}.lance"
    dataset = lance.dataset(str(lance_path))
    first = dataset.take([0])
    row = {
        name: _sanitized(first.column(name)[0].as_py())
        for name in first.column_names
    }
    episode_index = int(row["episode_index"])
    episode_rows = dataset.to_table(
        columns=["episode_index", "frame_index", "index", "timestamp"],
        filter=f"episode_index = {episode_index}",
    ).to_pylist()
    if not episode_rows:
        raise RuntimeError(f"episode {episode_index} unexpectedly has no rows")
    frame_indices = [int(item["frame_index"]) for item in episode_rows]
    timestamps = [float(item["timestamp"]) for item in episode_rows]
    frame_strictly_increasing = all(
        right > left for left, right in zip(frame_indices, frame_indices[1:])
    )
    timestamp_strictly_increasing = all(
        right > left for left, right in zip(timestamps, timestamps[1:])
    )
    timestamp_residuals = [
        timestamp - frame_index / args.fps
        for frame_index, timestamp in zip(frame_indices, timestamps)
    ]
    finite_residuals = [value for value in timestamp_residuals if math.isfinite(value)]
    if len(finite_residuals) != len(timestamp_residuals):
        raise RuntimeError("episode contains a non-finite timestamp residual")

    episode_metadata_paths = sorted(
        (root / args.table / "meta" / "episodes").glob("chunk-*/file-*.parquet")
    )
    if not episode_metadata_paths:
        raise FileNotFoundError("episode metadata parquet is missing")
    import pyarrow.parquet as pq

    episode_metadata = pq.read_table(episode_metadata_paths)
    payload = {
        "root": str(root),
        "tables_json_sha256": _sha256(tables_path),
        "dataset_name": tables.get("dataset_name"),
        "format": tables.get("format"),
        "declared_fps": tables.get("tables", [{}])[0].get("fps"),
        "present_tables": present_tables,
        "probe_table": args.table,
        "lance_version": int(dataset.version),
        "row_count": int(dataset.count_rows()),
        "schema": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": bool(field.nullable),
            }
            for field in dataset.schema
        ],
        "first_row_sanitized": row,
        "probe_episode": {
            "episode_index": episode_index,
            "row_count": len(episode_rows),
            "first_frame_index": frame_indices[0],
            "last_frame_index": frame_indices[-1],
            "frame_index_strictly_increasing": frame_strictly_increasing,
            "timestamp_strictly_increasing": timestamp_strictly_increasing,
            "timestamp_minus_frame_over_fps_min_s": min(finite_residuals),
            "timestamp_minus_frame_over_fps_max_s": max(finite_residuals),
        },
        "episode_metadata": {
            "files": [
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in episode_metadata_paths
            ],
            "schema": [
                {"name": field.name, "type": str(field.type)}
                for field in episode_metadata.schema
            ],
            "row_count": int(episode_metadata.num_rows),
            "first_rows": _sanitized(episode_metadata.slice(0, 3).to_pylist()),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
