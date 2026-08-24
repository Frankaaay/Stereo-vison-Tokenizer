#!/usr/bin/env python3
"""Build a deterministic 512-sample candidate set for teacher comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from stereo_tokenizer.lerobot_data import LeRobotStereoDataset  # noqa: E402


REQUIRED_VISUAL_TAGS = (
    "near_object",
    "far_object",
    "low_texture",
    "reflective",
    "occlusion",
    "motion_blur",
    "multi_task_scene",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_train_records(path: Path):
    records = []
    sample_offset = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "train":
                continue
            record = dict(record)
            record["first_dataset_index"] = sample_offset
            sample_offset += int(record["window_count"])
            records.append(record)
    if not records:
        raise ValueError("episode manifest contains no train records")
    return records


def select(records, count, seed):
    by_task = defaultdict(list)
    for record in records:
        task_key = json.dumps(
            record.get("tasks", []),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        by_task[task_key].append(record)
    tasks = sorted(
        by_task,
        key=lambda task: hashlib.sha256(
            f"{seed}:task:{task}".encode("utf-8")
        ).digest(),
    )
    selected = []
    task_positions = defaultdict(int)
    while len(selected) < count:
        progress = False
        for task in tasks:
            candidates = sorted(
                by_task[task],
                key=lambda record: hashlib.sha256(
                    f"{seed}:{task}:{record['shard_id']}:{record['episode_id']}".encode()
                ).digest(),
            )
            position = task_positions[task]
            if position >= len(candidates):
                continue
            record = candidates[position]
            task_positions[task] += 1
            local_window = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:window:{record['episode_id']}".encode()
                ).digest()[:8],
                "big",
            ) % int(record["window_count"])
            start_frame = local_window * 12
            selected.append(
                {
                    "dataset_index": int(record["first_dataset_index"]) + local_window,
                    "sample_id": f"{record['episode_id']}:{start_frame:06d}",
                    "episode_id": record["episode_id"],
                    "shard_id": record["shard_id"],
                    "tasks": record.get("tasks", []),
                    "start_frame": start_frame,
                    "visual_tags": [],
                }
            )
            progress = True
            if len(selected) == count:
                break
        if not progress:
            raise ValueError(f"only {len(selected)} unique candidate episodes available")
    return selected


def _sample_tile(sample):
    video = ((sample["video"].numpy() + 0.5) * 255.0).clip(0, 255).astype(
        np.uint8
    )
    rows = []
    for view in range(3):
        frames = []
        for frame in range(4):
            image = video[view, 0, :, frame].transpose(1, 2, 0)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frames.append(cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA))
        rows.append(np.concatenate(frames, axis=1))
    tile = np.concatenate(rows, axis=0)
    tile = cv2.copyMakeBorder(
        tile, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    cv2.putText(
        tile,
        sample["sample_id"],
        (4, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def render_contact_sheets(dataset, samples, visual_root: Path):
    visual_root.mkdir(parents=True)
    tiles = []
    for entry in samples:
        sample = dataset[int(entry["dataset_index"])]
        if sample["sample_id"] != entry["sample_id"]:
            raise ValueError("selection changed before visualization")
        tiles.append(_sample_tile(sample))
    blank = np.zeros_like(tiles[0])
    for sheet_index, start in enumerate(range(0, len(tiles), 16)):
        page = tiles[start : start + 16]
        page.extend([blank] * (16 - len(page)))
        rows = [np.concatenate(page[row : row + 4], axis=1) for row in range(0, 16, 4)]
        sheet = np.concatenate(rows, axis=0)
        path = visual_root / f"sheet_{sheet_index:03d}.png"
        if not cv2.imwrite(str(path), sheet):
            raise RuntimeError(f"failed to write {path}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rectification-audit-sha256", required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.count != 512:
        raise ValueError("teacher comparison candidate count is frozen to 512")
    manifest = args.episode_manifest.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    visual_root = args.visual_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if visual_root.exists():
        raise FileExistsError(f"refusing to overwrite {visual_root}")
    samples = select(_read_train_records(manifest), args.count, args.seed)
    dataset = LeRobotStereoDataset(
        manifest,
        dataset_root,
        split="train",
        expected_rectification_audit_sha256=(
            args.rectification_audit_sha256
        ),
    )
    render_contact_sheets(dataset, samples, visual_root)
    payload = {
        "schema": "lerobot-teacher-selection-v1",
        "episode_manifest": str(manifest),
        "episode_manifest_sha256": _sha256_file(manifest),
        "visual_root": str(visual_root),
        "seed": args.seed,
        "sample_count": len(samples),
        "selection_granularity": "one_sample_per_episode",
        "all_samples_cover_views": ["head", "lefthand", "righthand"],
        "required_visual_tags": list(REQUIRED_VISUAL_TAGS),
        "coverage_counts": {tag: 0 for tag in REQUIRED_VISUAL_TAGS},
        "review_status": "pending",
        "review_note": (
            "Inspect fixed samples, add visual_tags, update coverage_counts, and set "
            "review_status=approved before running the comparison."
        ),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
