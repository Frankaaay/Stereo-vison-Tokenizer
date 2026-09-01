#!/usr/bin/env python3
"""Fail-closed sync/read/epipolar audit for canonical-v3 stereo RGB."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from stereo_tokenizer.canonical_v3_data import (
    CanonicalV3StereoDataset,
    VIDEO_KEYS,
    VIEWS,
)


def vertical_residuals(left, right):
    detector = cv2.ORB_create(nfeatures=4000, fastThreshold=10)
    left_keypoints, left_descriptors = detector.detectAndCompute(
        cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), None
    )
    right_keypoints, right_descriptors = detector.detectAndCompute(
        cv2.cvtColor(right, cv2.COLOR_RGB2GRAY), None
    )
    if left_descriptors is None or right_descriptors is None:
        return np.empty(0, dtype=np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(
        left_descriptors, right_descriptors, k=2
    )
    reverse = matcher.knnMatch(right_descriptors, left_descriptors, k=2)
    reverse_best = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    }
    residuals = []
    for pair in forward:
        if len(pair) != 2 or pair[0].distance >= 0.75 * pair[1].distance:
            continue
        if reverse_best.get(pair[0].trainIdx) != pair[0].queryIdx:
            continue
        left_point = left_keypoints[pair[0].queryIdx].pt
        right_point = right_keypoints[pair[0].trainIdx].pt
        disparity = left_point[0] - right_point[0]
        if -2 < disparity < left.shape[1]:
            residuals.append(abs(left_point[1] - right_point[1]))
    return np.asarray(residuals, dtype=np.float32)


def metrics(values):
    if not values:
        return {"match_count": 0, "median_px": None, "p95_px": None}
    array = np.concatenate(values)
    if not len(array):
        return {"match_count": 0, "median_px": None, "p95_px": None}
    return {
        "match_count": int(array.size),
        "median_px": float(np.median(array)),
        "p95_px": float(np.percentile(array, 95)),
    }


def selected_records(dataset, count, seed):
    ordered = sorted(
        dataset.records,
        key=lambda record: hashlib.sha256(
            f"{seed}:audit:{record['episode_id']}".encode()
        ).digest(),
    )
    if len(ordered) < count:
        raise ValueError(f"audit requested {count} episodes, found {len(ordered)}")
    return ordered[:count]


def decode_pair(dataset, record, view):
    images = {}
    for eye in ("left", "right"):
        video = record["videos"][VIDEO_KEYS[(view, eye)]]
        path = (dataset.dataset_root / video["relative_path"]).resolve()
        timestamp = (
            float(video["from_timestamp"]) + float(video["to_timestamp"])
        ) * 0.5
        images[eye] = dataset._decode_frames(path, [timestamp])[0]
    return images["left"], images["right"]


def visual_pair(left, right, label):
    pair = cv2.cvtColor(np.concatenate((left, right), axis=1), cv2.COLOR_RGB2BGR)
    for y in range(32, pair.shape[0], 32):
        cv2.line(pair, (0, y), (pair.shape[1] - 1, y), (0, 255, 0), 1)
    cv2.putText(
        pair,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return pair


def render_html(result, image_paths):
    status = str(result["result"]).upper()
    color = "#047857" if result["result"] == "pass" else "#b91c1c"
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(view)}</td>"
        f"<td>{values['match_count']}</td>"
        f"<td>{values['median_px'] if values['median_px'] is not None else 'N/A'}</td>"
        f"<td>{values['p95_px'] if values['p95_px'] is not None else 'N/A'}</td>"
        "</tr>"
        for view, values in result["metrics_by_view"].items()
    )
    images = []
    for path in image_paths:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        images.append(
            '<figure><img src="data:image/jpeg;base64,'
            f'{encoded}" alt="{html.escape(path.name)}"><figcaption>'
            f"{html.escape(path.name)}</figcaption></figure>"
        )
    raw = html.escape(json.dumps(result, indent=2, sort_keys=True))
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Stereo data gate</title>
<style>body{{max-width:1100px;margin:32px auto;padding:0 20px;font:14px/1.55 system-ui;color:#0f172a}}
.status{{display:inline-block;background:{color};color:white;padding:6px 12px;border-radius:999px;font-weight:700}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dbe2ea;text-align:right}}
th:first-child,td:first-child{{text-align:left}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}}
figure{{margin:0;border:1px solid #dbe2ea;padding:8px}}img{{width:100%}}pre{{overflow:auto;background:#0b1220;color:#dbeafe;padding:16px}}</style>
</head><body><span class="status">{status}</span><h1>H100 双目数据门禁</h1>
<p>读取率 {_format_ratio(result['valid_read_ratio'])}；有效匹配对比例 {_format_ratio(result['valid_match_pair_ratio'])}；
同步失败 {result['sync_failure_count']}。</p><table><thead><tr><th>View</th><th>Matches</th><th>Median vertical px</th><th>P95 vertical px</th></tr></thead>
<tbody>{rows}</tbody></table><h2>抽样双目对</h2><div class="grid">{''.join(images)}</div>
<h2>完整门禁 JSON</h2><pre>{raw}</pre></body></html>"""


def _format_ratio(value):
    return f"{float(value) * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--episode-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--minimum-read-ratio", type=float, default=0.99)
    parser.add_argument("--maximum-median-vertical-px", type=float, default=1.0)
    parser.add_argument("--maximum-p95-vertical-px", type=float, default=2.0)
    parser.add_argument("--minimum-matches-per-pair", type=int, default=20)
    parser.add_argument("--visuals-per-view", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    dataset = CanonicalV3StereoDataset(
        args.manifest,
        args.dataset_root,
        split=args.split,
        rectification_audit_sha256="0" * 64,
        video_cache_capacity=12,
    )
    residuals = defaultdict(list)
    read_failures = []
    match_failures = []
    visual_counts = defaultdict(int)
    records = selected_records(dataset, args.episode_count, args.seed)
    sync_failures = 0
    successful_reads = 0
    successful_matches = 0
    for record in records:
        starts = [
            float(video["from_timestamp"]) for video in record["videos"].values()
        ]
        stops = [
            float(video["to_timestamp"]) for video in record["videos"].values()
        ]
        if max(starts) - min(starts) > 1e-6 or max(stops) - min(stops) > 1e-6:
            sync_failures += 1
        for view in VIEWS:
            try:
                left, right = decode_pair(dataset, record, view)
                successful_reads += 1
                values = vertical_residuals(left, right)
                if len(values) < args.minimum_matches_per_pair:
                    match_failures.append(
                        {
                            "episode_id": record["episode_id"],
                            "view": view,
                            "error": f"only {len(values)} reciprocal ORB matches",
                        }
                    )
                    continue
                successful_matches += 1
                residuals[view].append(values)
                if visual_counts[view] < args.visuals_per_view:
                    output = args.output_dir / (
                        f"{view}-{visual_counts[view]:02d}-"
                        f"{record['episode_id']}.jpg"
                    )
                    if not cv2.imwrite(
                        str(output),
                        visual_pair(left, right, f"{view}/{record['episode_id']}"),
                    ):
                        raise RuntimeError(f"failed to write {output}")
                    visual_counts[view] += 1
            except Exception as error:
                read_failures.append(
                    {
                        "episode_id": record["episode_id"],
                        "view": view,
                        "error": str(error),
                    }
                )
    requested_pairs = args.episode_count * len(VIEWS)
    read_ratio = successful_reads / requested_pairs
    match_ratio = successful_matches / requested_pairs
    by_view = {view: metrics(residuals[view]) for view in VIEWS}
    passed = (
        sync_failures == 0
        and read_ratio >= args.minimum_read_ratio
        and match_ratio >= args.minimum_read_ratio
        and all(
            value["median_px"] is not None
            and value["median_px"] <= args.maximum_median_vertical_px
            and value["p95_px"] <= args.maximum_p95_vertical_px
            for value in by_view.values()
        )
    )
    result = {
        "schema": "canonical-v3-stereo-data-gate-v1",
        "result": "pass" if passed else "fail",
        "dataset_root": str(args.dataset_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "seed": args.seed,
        "episode_count": args.episode_count,
        "requested_pair_count": requested_pairs,
        "successful_read_count": successful_reads,
        "valid_read_ratio": read_ratio,
        "successful_match_pair_count": successful_matches,
        "valid_match_pair_ratio": match_ratio,
        "sync_failure_count": sync_failures,
        "metrics_by_view": by_view,
        "read_failures": read_failures,
        "match_failures": match_failures,
        "thresholds": {
            "minimum_read_ratio": args.minimum_read_ratio,
            "maximum_median_vertical_px": args.maximum_median_vertical_px,
            "maximum_p95_vertical_px": args.maximum_p95_vertical_px,
            "minimum_matches_per_pair": args.minimum_matches_per_pair,
        },
    }
    report = args.output_dir / "audit.json"
    report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "index.html").write_text(
        render_html(result, sorted(args.output_dir.glob("*.jpg"))),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
