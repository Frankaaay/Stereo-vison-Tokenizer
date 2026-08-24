#!/usr/bin/env python3
"""Audit whether LeRobot stereo MP4 streams are already rectified."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import build_lerobot_stereo_manifest as manifest_builder


SCHEMA = "lerobot-stereo-rectification-audit-v1"
VIEWS = ("head", "lefthand", "righthand")
VIDEO_NAMES = {
    ("head", "left"): "observation.images.head_left",
    ("head", "right"): "observation.images.head_right",
    ("lefthand", "left"): "observation.images.left_wrist_left",
    ("lefthand", "right"): "observation.images.left_wrist_right",
    ("righthand", "left"): "observation.images.right_wrist_left",
    ("righthand", "right"): "observation.images.right_wrist_right",
}


def _epipolar_visual(raw_left, raw_right, rectified_left, rectified_right):
    rows = []
    for label, left, right in (
        ("raw", raw_left, raw_right),
        ("calibration-remap", rectified_left, rectified_right),
    ):
        pair = np.concatenate([left, right], axis=1)
        pair = cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)
        for y in range(40, pair.shape[0], 40):
            cv2.line(pair, (0, y), (pair.shape[1] - 1, y), (0, 255, 0), 1)
        cv2.putText(
            pair,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(pair)
    return np.concatenate(rows, axis=0)


def _decode_nearest(path: Path, timestamp: float, tolerance_s: float):
    try:
        import av
    except ImportError as error:
        raise RuntimeError("rectification audit requires PyAV") from error
    with av.open(str(path), mode="r") as container:
        stream = next(stream for stream in container.streams if stream.type == "video")
        container.seek(
            max(0, int(timestamp / float(stream.time_base))),
            stream=stream,
            any_frame=False,
            backward=True,
        )
        candidates = []
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            frame_time = float(frame.time)
            if abs(frame_time - timestamp) <= tolerance_s:
                candidates.append(
                    (
                        abs(frame_time - timestamp),
                        frame.to_ndarray(format="rgb24"),
                    )
                )
            if frame_time > timestamp + tolerance_s:
                break
    if not candidates:
        raise RuntimeError(f"{path}: no frame near {timestamp}")
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _rectify(image, camera):
    k = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(camera["D"], dtype=np.float64)
    r = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    p = np.asarray(camera["P"], dtype=np.float64).reshape(3, 4)
    maps = cv2.initUndistortRectifyMap(
        k,
        d,
        r,
        p[:, :3],
        (image.shape[1], image.shape[0]),
        cv2.CV_32FC1,
    )
    return cv2.remap(
        image,
        maps[0],
        maps[1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _vertical_residuals(left, right):
    detector = cv2.ORB_create(nfeatures=4000, fastThreshold=10)
    left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    left_keypoints, left_descriptors = detector.detectAndCompute(left_gray, None)
    right_keypoints, right_descriptors = detector.detectAndCompute(right_gray, None)
    if left_descriptors is None or right_descriptors is None:
        return np.empty(0, dtype=np.float32)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        left_descriptors, right_descriptors, k=2
    )
    residuals = []
    for pair in matches:
        if len(pair) != 2 or pair[0].distance >= 0.75 * pair[1].distance:
            continue
        left_point = left_keypoints[pair[0].queryIdx].pt
        right_point = right_keypoints[pair[0].trainIdx].pt
        disparity = left_point[0] - right_point[0]
        if disparity <= -2 or disparity >= left.shape[1]:
            continue
        residuals.append(abs(left_point[1] - right_point[1]))
    return np.asarray(residuals, dtype=np.float32)


def _metrics(values):
    if not values:
        return {"match_count": 0, "p50_px": None, "p95_px": None, "p99_px": None}
    array = np.concatenate(values)
    if not len(array):
        return {"match_count": 0, "p50_px": None, "p95_px": None, "p99_px": None}
    return {
        "match_count": int(array.size),
        "p50_px": float(np.percentile(array, 50)),
        "p95_px": float(np.percentile(array, 95)),
        "p99_px": float(np.percentile(array, 99)),
    }


def _candidate_episodes(args):
    shards = sorted(
        path
        for path in args.dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("shard_")
    )
    scored = sorted(
        shards,
        key=lambda path: hashlib.sha256(
            f"{args.seed}:{path.name}".encode("utf-8")
        ).digest(),
    )
    selected = scored[: min(args.episode_count, len(scored))]
    for shard in selected:
        shard_number = int(shard.name.split("_")[1])
        source_manifest = (
            args.dataset_root / "_manifests" / f"m_{shard_number:04d}"
        )
        source_paths = [
            line
            for line in source_manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
        failures = manifest_builder._read_failures(args.dataset_root, shard.name)
        sources = [path for path in source_paths if path not in failures]
        rows = manifest_builder._episode_rows(shard)
        if len(sources) != len(rows):
            raise ValueError(f"{shard.name}: source/episode count mismatch")
        position = int.from_bytes(
            hashlib.sha256(f"{args.seed}:{shard.name}:episode".encode()).digest()[:4],
            "big",
        ) % len(rows)
        yield shard, sources[position], rows[position]


def run_audit(args):
    raw_values = defaultdict(list)
    remapped_values = defaultdict(list)
    failures = []
    representative_pairs = 0
    candidate_episode_count = 0
    visual_counts = defaultdict(int)
    for shard, source_path, row in _candidate_episodes(args):
        candidate_episode_count += 1
        source_episode = manifest_builder._source_episode(
            source_path,
            args.source_manifest_prefix,
            args.source_root,
        )
        episode_id = source_episode.name
        source_json = source_episode / f"{episode_id}.json"
        calibration = manifest_builder._calibration(source_json, episode_id)
        for view in VIEWS:
            try:
                images = {}
                for eye in ("left", "right"):
                    video_name = VIDEO_NAMES[(view, eye)]
                    relative = Path(
                        shard.name,
                        "videos",
                        video_name,
                        f"chunk-{int(row[f'videos/{video_name}/chunk_index']):03d}",
                        f"file-{int(row[f'videos/{video_name}/file_index']):03d}.mp4",
                    )
                    interval_start = float(
                        row[f"videos/{video_name}/from_timestamp"]
                    )
                    interval_stop = float(row[f"videos/{video_name}/to_timestamp"])
                    timestamp = interval_start + (interval_stop - interval_start) * 0.5
                    images[eye] = _decode_nearest(
                        args.dataset_root / relative,
                        timestamp,
                        args.timestamp_tolerance_s,
                    )
                raw = _vertical_residuals(images["left"], images["right"])
                rectified_left = _rectify(
                    images["left"], calibration[view]["left"]
                )
                rectified_right = _rectify(
                    images["right"], calibration[view]["right"]
                )
                remapped = _vertical_residuals(rectified_left, rectified_right)
                if raw.size < args.minimum_matches_per_pair:
                    raise RuntimeError(f"only {raw.size} raw ORB matches")
                if remapped.size < args.minimum_matches_per_pair:
                    raise RuntimeError(f"only {remapped.size} remapped ORB matches")
                raw_values[view].append(raw)
                remapped_values[view].append(remapped)
                representative_pairs += 1
                if visual_counts[view] < args.visuals_per_view:
                    visual = _epipolar_visual(
                        images["left"],
                        images["right"],
                        rectified_left,
                        rectified_right,
                    )
                    path = args.visual_root / (
                        f"{view}_{visual_counts[view]:03d}_"
                        f"{shard.name}_{episode_id}.png"
                    )
                    if not cv2.imwrite(str(path), visual):
                        raise RuntimeError(f"failed to write {path}")
                    visual_counts[view] += 1
            except Exception as error:
                failures.append(
                    {
                        "episode_id": episode_id,
                        "shard_id": shard.name,
                        "view": view,
                        "error": str(error),
                    }
                )

    if candidate_episode_count == 0:
        raise ValueError("rectification audit found no candidate episodes")
    raw_metrics = {view: _metrics(raw_values[view]) for view in VIEWS}
    remapped_metrics = {view: _metrics(remapped_values[view]) for view in VIEWS}
    minimum_successful_pairs = math.ceil(
        candidate_episode_count * args.minimum_successful_pair_fraction
    )
    visual_pass = all(
        visual_counts[view] == args.visuals_per_view for view in VIEWS
    )
    raw_pass = all(
        len(raw_values[view]) >= minimum_successful_pairs
        and metrics["match_count"] >= args.minimum_total_matches_per_view
        and metrics["p95_px"] <= args.maximum_p95_vertical_error_px
        for view, metrics in raw_metrics.items()
    ) and visual_pass
    remapped_pass = all(
        len(remapped_values[view]) >= minimum_successful_pairs
        and metrics["match_count"] >= args.minimum_total_matches_per_view
        and metrics["p95_px"] <= args.maximum_p95_vertical_error_px
        for view, metrics in remapped_metrics.items()
    ) and visual_pass
    if raw_pass:
        result = "pass"
        selected_mode = "verified_pre_rectified"
    elif remapped_pass:
        result = "pass"
        selected_mode = "apply_calibration"
    else:
        result = "fail"
        selected_mode = None
    return {
        "schema": SCHEMA,
        "dataset_root": str(args.dataset_root),
        "source_root": str(args.source_root),
        "seed": args.seed,
        "requested_episode_count": args.episode_count,
        "candidate_episode_count": candidate_episode_count,
        "representative_pair_count": representative_pairs,
        "successful_pair_count_by_view": {
            view: len(raw_values[view]) for view in VIEWS
        },
        "failed_pair_count": len(failures),
        "visual_root": str(args.visual_root),
        "visual_count_by_view": {
            view: visual_counts[view] for view in VIEWS
        },
        "failures": failures,
        "thresholds": {
            "minimum_matches_per_pair": args.minimum_matches_per_pair,
            "minimum_total_matches_per_view": args.minimum_total_matches_per_view,
            "minimum_successful_pair_fraction": (
                args.minimum_successful_pair_fraction
            ),
            "minimum_successful_pairs_per_view": minimum_successful_pairs,
            "maximum_p95_vertical_error_px": args.maximum_p95_vertical_error_px,
        },
        "raw_video_metrics": raw_metrics,
        "apply_calibration_metrics": remapped_metrics,
        "result": result,
        "selected_mode": selected_mode,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest-prefix", default="/data/umi_vio_data_260714/"
    )
    parser.add_argument("--episode-count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timestamp-tolerance-s", type=float, default=0.05)
    parser.add_argument("--minimum-matches-per-pair", type=int, default=20)
    parser.add_argument("--minimum-total-matches-per-view", type=int, default=1000)
    parser.add_argument(
        "--minimum-successful-pair-fraction", type=float, default=0.9
    )
    parser.add_argument("--maximum-p95-vertical-error-px", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--visuals-per-view", type=int, default=8)
    return parser


def main():
    args = build_parser().parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.visual_root = args.visual_root.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.visual_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.visual_root}")
    if args.episode_count < 1:
        raise ValueError("episode count must be positive")
    if not args.dataset_root.is_dir() or not args.source_root.is_dir():
        raise FileNotFoundError("dataset and source roots must exist")
    if not 0 < args.minimum_successful_pair_fraction <= 1:
        raise ValueError("successful pair fraction must be in (0,1]")
    if args.minimum_matches_per_pair < 1 or args.minimum_total_matches_per_view < 1:
        raise ValueError("rectification match thresholds must be positive")
    if args.maximum_p95_vertical_error_px <= 0:
        raise ValueError("rectification vertical-error threshold must be positive")
    if args.visuals_per_view < 1:
        raise ValueError("rectification visual count must be positive")
    args.visual_root.mkdir(parents=True)
    payload = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, args.output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    if payload["result"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
