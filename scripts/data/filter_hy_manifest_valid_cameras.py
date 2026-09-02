"""Filter a Hy manifest to episodes with valid three-camera JPEG anchors."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CAMERA_IDS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FRAME_OFFSETS = (0, 3, 6, 9)
FRAME_STRIDE = 12


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema") != "hy-mono-three-camera-episode-v2":
                raise ValueError(f"{path}:{line_number}: invalid Hy schema")
            if set(record.get("camera_columns", {})) != set(CAMERA_IDS):
                raise ValueError(f"{path}:{line_number}: invalid camera contract")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _root_aliases(values: list[str]) -> dict[str, Path]:
    aliases = {}
    for value in values:
        alias, separator, raw_path = value.partition("=")
        if not separator or not alias or not raw_path:
            raise ValueError("roots must use alias=/absolute/path")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        aliases[alias] = path
    return aliases


def anchor_frame_indices(window_count: int) -> tuple[int, ...]:
    """Return the first, middle, and last four-frame training windows."""
    if window_count < 1:
        return ()
    starts = {0, ((int(window_count) - 1) // 2) * FRAME_STRIDE}
    starts.add((int(window_count) - 1) * FRAME_STRIDE)
    return tuple(sorted({start + offset for start in starts for offset in FRAME_OFFSETS}))


def jpeg_error(payload) -> str | None:
    if not isinstance(payload, bytes):
        return f"payload_type={type(payload).__name__}"
    if len(payload) < 4:
        return f"payload_length={len(payload)}"
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
            if image.format != "JPEG":
                return f"image_format={image.format}"
            if image.size != (424, 240):
                return f"image_size={image.size}"
    except Exception as error:  # PIL exposes several format-specific exceptions.
        return f"{type(error).__name__}: {error}"
    return None


def index_rows_by_identity(rows: list[dict]) -> dict[tuple[int, int], dict]:
    indexed = {}
    for row in rows:
        key = (int(row["episode_index"]), int(row["frame_index"]))
        if key in indexed:
            raise ValueError(f"duplicate Lance row identity {key}")
        indexed[key] = row
    return indexed


def anchor_filter(records: list[dict]) -> str:
    clauses = []
    for record in records:
        frames = ", ".join(
            str(value) for value in anchor_frame_indices(int(record["window_count"]))
        )
        clauses.append(
            f"(episode_index = {int(record['episode_index'])} AND "
            f"frame_index IN ({frames}))"
        )
    if not clauses:
        raise ValueError("cannot build an empty anchor filter")
    return " OR ".join(clauses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--episode_batch_size", type=int, default=128)
    args = parser.parse_args()
    if args.episode_batch_size < 1:
        raise ValueError("episode_batch_size must be positive")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    summary_path = output_path.with_suffix(".summary.json")
    rejected_path = output_path.with_suffix(".rejected.jsonl")
    for target in (output_path, summary_path, rejected_path):
        if target.exists():
            raise FileExistsError(target)

    try:
        import lance
    except ImportError as error:
        raise ImportError("Hy camera validation requires pylance") from error

    roots = _root_aliases(args.root)
    records = _read_jsonl(input_path)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["root_alias"], record["table_name"])].append(record)

    rejected: dict[str, dict] = {}
    invalid_by_camera = Counter()
    invalid_by_table = Counter()
    invalid_by_reason = Counter()
    checked_payloads = 0
    for (root_alias, table_name), table_records in sorted(grouped.items()):
        if root_alias not in roots:
            raise ValueError(f"missing root alias {root_alias}")
        lance_path = roots[root_alias] / table_name / f"{table_name}.lance"
        dataset = lance.dataset(str(lance_path))
        camera_columns = table_records[0]["camera_columns"]
        columns = ["episode_index", "frame_index", *camera_columns.values()]
        for batch_start in range(0, len(table_records), args.episode_batch_size):
            batch = table_records[batch_start : batch_start + args.episode_batch_size]
            requests = []
            for record in batch:
                for frame_index in anchor_frame_indices(int(record["window_count"])):
                    requests.append((record, frame_index))
            rows = dataset.to_table(
                filter=anchor_filter(batch), columns=columns
            ).to_pylist()
            if len(rows) != len(requests):
                raise ValueError(f"{table_name}: Lance anchor row count mismatch")
            indexed_rows = index_rows_by_identity(rows)
            requested_identities = {
                (int(record["episode_index"]), frame_index)
                for record, frame_index in requests
            }
            if set(indexed_rows) != requested_identities:
                raise ValueError(f"{table_name}: Lance anchor identity mismatch")
            for record, frame_index in requests:
                row = indexed_rows[(int(record["episode_index"]), frame_index)]
                for camera_id, column in camera_columns.items():
                    checked_payloads += 1
                    error = jpeg_error(row[column])
                    if error is None:
                        continue
                    entry = rejected.setdefault(
                        record["episode_id"],
                        {
                            "episode_id": record["episode_id"],
                            "root_alias": root_alias,
                            "table_name": table_name,
                            "episode_index": int(record["episode_index"]),
                            "failures": [],
                        },
                    )
                    entry["failures"].append(
                        {"camera_id": camera_id, "frame_index": frame_index, "error": error}
                    )
                    invalid_by_camera[camera_id] += 1
                    invalid_by_reason[error] += 1
        table_rejected = sum(record["episode_id"] in rejected for record in table_records)
        invalid_by_table[table_name] = table_rejected
        print(
            json.dumps(
                {
                    "table": table_name,
                    "records": len(table_records),
                    "rejected": table_rejected,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    accepted = [record for record in records if record["episode_id"] not in rejected]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in accepted),
        encoding="utf-8",
        newline="\n",
    )
    rejected_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in rejected.values()),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "schema": "hy-three-camera-anchor-validation-v1",
        "input_manifest": str(input_path),
        "input_manifest_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_manifest": str(output_path),
        "output_manifest_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "anchor_policy": "first_middle_last_four_frame_windows",
        "checked_payloads": checked_payloads,
        "input_records": len(records),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "accepted_windows_per_camera": sum(record["window_count"] for record in accepted),
        "accepted_windows_three_camera": 3
        * sum(record["window_count"] for record in accepted),
        "accepted_records_by_split": dict(Counter(record["split"] for record in accepted)),
        "accepted_windows_per_camera_by_split": {
            split: sum(record["window_count"] for record in accepted if record["split"] == split)
            for split in sorted({record["split"] for record in accepted})
        },
        "rejected_by_table": dict(invalid_by_table),
        "failure_count_by_camera": dict(invalid_by_camera),
        "failure_count_by_reason": dict(invalid_by_reason),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
