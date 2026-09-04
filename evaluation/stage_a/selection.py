"""Stage A deterministic selection and decode-preflight commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import _dataset_provenance
from .data import CanonicalStageADataset, build_canonical_selection


def _selection_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tokenizer_stage_a selection")
    parser.add_argument("--dataset-id", choices=("umi", "hy", "libero"), required=True)
    parser.add_argument("--identity-contract", type=Path, required=True)
    parser.add_argument("--canonical-config-root", type=Path)
    parser.add_argument("--canonical-loader-root", type=Path)
    parser.add_argument("--umi-publish-ledger", type=Path)
    parser.add_argument("--hy-manifest", type=Path)
    parser.add_argument("--hy-manifest-sha256")
    parser.add_argument("--hy-root-aliases")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    selection = build_canonical_selection(
        dataset_id=args.dataset_id,
        identity_contract_path=args.identity_contract,
        canonical_config_root=args.canonical_config_root,
        loader_root=args.canonical_loader_root,
        split=args.split,
        sample_count=args.sample_count,
        seed=args.seed,
        output=args.output,
        umi_publish_ledger=args.umi_publish_ledger,
        hy_manifest_path=args.hy_manifest,
        hy_manifest_sha256=args.hy_manifest_sha256,
        hy_root_aliases=args.hy_root_aliases,
    )
    print(
        json.dumps(
            {key: value for key, value in selection.items() if key != "records"},
            indent=2,
            sort_keys=True,
        )
    )


def _preflight_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tokenizer_stage_a preflight")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--canonical-loader-root", type=Path)
    parser.add_argument("--hy-root-aliases")
    parser.add_argument("--eye-mode", choices=("mono", "stereo"), required=True)
    parser.add_argument("--camera-key")
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv)
    dataset = CanonicalStageADataset(
        args.selection,
        loader_root=args.canonical_loader_root,
        eye_mode=args.eye_mode,
        camera_key=args.camera_key,
        hy_root_aliases=args.hy_root_aliases,
    )
    if args.samples < 1 or args.samples > len(dataset):
        raise ValueError("invalid preflight sample count")
    rows = []
    for index in range(args.samples):
        sample = dataset[index]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "video_shape": list(sample["video"].shape),
                "video_dtype": str(sample["video"].dtype),
                "video_min": float(sample["video"].min()),
                "video_max": float(sample["video"].max()),
                "valid_rgb_values": int(sample["rgb_valid_mask"].sum()),
                "source_frame_indices": sample["frame_index"].tolist(),
            }
        )
    print(
        json.dumps(
            {"dataset": _dataset_provenance(dataset), "samples": rows},
            indent=2,
            sort_keys=True,
        )
    )
