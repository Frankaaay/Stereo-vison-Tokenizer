#!/usr/bin/env python3
"""Decode one single-frame and one four-frame sample from pretrain manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stereo_tokenizer.lerobot_data import LeRobotStereoDataset
from stereo_tokenizer.pretrain_data import HyLanceMonoDataset, LiberoMonoDataset


def _sample_summary(dataset):
    output = {"sample_count": len(dataset), "modes": {}}
    for mode in ("single_frame", "four_frame"):
        sample = dataset.get_mode_item(0, mode)
        output["modes"][mode] = {
            "sample_id": sample["sample_id"],
            "video_shape": list(sample["video"].shape),
            "video_dtype": str(sample["video"].dtype),
            "video_finite": bool(sample["video"].isfinite().all()),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--umi-manifest", type=Path)
    parser.add_argument("--umi-root", type=Path)
    parser.add_argument("--umi-audit-sha256")
    parser.add_argument("--libero-manifest", type=Path)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--hy-manifest", type=Path)
    parser.add_argument("--hy-root", type=Path)
    args = parser.parse_args()

    output = {}
    if args.umi_manifest is not None:
        if args.umi_root is None or args.umi_audit_sha256 is None:
            parser.error("UMI smoke requires --umi-root and --umi-audit-sha256")
        output["umi"] = _sample_summary(
            LeRobotStereoDataset(
                args.umi_manifest,
                args.umi_root,
                split="train",
                expected_rectification_audit_sha256=args.umi_audit_sha256,
            )
        )
    if args.libero_manifest is not None:
        if args.libero_root is None:
            parser.error("LIBERO smoke requires --libero-root")
        output["libero"] = _sample_summary(
            LiberoMonoDataset(
                args.libero_manifest,
                {"libero_primary": args.libero_root},
                split="train",
            )
        )
    if args.hy_manifest is not None:
        if args.hy_root is None:
            parser.error("Hy smoke requires --hy-root")
        output["hy"] = _sample_summary(
            HyLanceMonoDataset(
                args.hy_manifest,
                {"hy_primary": args.hy_root},
                split="train",
            )
        )
    if not output:
        parser.error("select at least one dataset manifest")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
