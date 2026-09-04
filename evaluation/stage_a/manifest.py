"""Freeze immutable episode-level splits from current H100 canonical data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .contract import (
    CANONICAL_SPLIT_SCHEMA,
    canonical_sha256,
    hash_order,
    sha256_file,
)
from .data import (
    CANONICAL_LOADER_SHA,
    _build_four_frame_dataset,
)


SPLIT_RATIOS = {"train": 0.90, "val": 0.05, "test": 0.05}
CONFIG_SCHEMA_VERSION = "ngad_canonical_dataloader_v2"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), text=True, stderr=subprocess.STDOUT
    ).strip()


def _apportion_counts(total: int) -> dict[str, int]:
    if total < len(SPLIT_RATIOS):
        raise ValueError("canonical split requires at least three episodes")
    raw = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    priority = {"test": 0, "val": 1, "train": 2}
    order = sorted(
        SPLIT_RATIOS,
        key=lambda name: (-(raw[name] - counts[name]), priority[name]),
    )
    for name in order[:remainder]:
        counts[name] += 1
    if sum(counts.values()) != total or any(value < 1 for value in counts.values()):
        raise RuntimeError("invalid split apportionment")
    return counts


def assign_splits(
    records: list[dict[str, Any]], *, dataset_id: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Assign exact 90/5/5 counts by a stable hash rank of episode identity."""

    identities = [str(record["episode_id"]) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{dataset_id}: duplicate canonical episode identities")
    counts = _apportion_counts(len(records))
    ranked = sorted(
        records,
        key=lambda record: hash_order(
            seed, "canonical-split", dataset_id, record["episode_id"]
        ),
    )
    boundaries = (
        ("train", counts["train"]),
        ("val", counts["val"]),
        ("test", counts["test"]),
    )
    position = 0
    assigned = []
    for split, count in boundaries:
        for record in ranked[position : position + count]:
            assigned.append({**record, "split": split})
        position += count
    if position != len(records):
        raise RuntimeError("split assignment did not consume every episode")
    assigned.sort(key=lambda record: str(record["episode_id"]))
    return assigned, counts


def _resolved_config_payload(source: Path) -> tuple[dict[str, Any], str]:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: YAML root must be a mapping")
    if set(payload) == {"dataset"}:
        payload = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "dataset": payload["dataset"],
        }
        repair = "added_missing_ngad_canonical_dataloader_v2_schema_version"
    elif set(payload) == {"schema_version", "dataset"}:
        if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"{source}: unsupported schema version")
        repair = "none"
    else:
        raise ValueError(f"{source}: unsupported YAML root fields")
    return payload, repair


def _write_config_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    payload, repair = _resolved_config_payload(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "resolved_path": str(destination),
        "resolved_sha256": sha256_file(destination),
        "repair": repair,
    }


def _umi_index_to_source_id(path: Path) -> tuple[dict[int, str], dict[str, str]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT source_id, episode_index FROM episodes"
        ).fetchall()
    finally:
        connection.close()
    mapping = {int(index): str(source_id) for source_id, index in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"{path}: duplicate canonical episode indices")
    return mapping, {"path": str(path), "sha256": sha256_file(path)}


def _catalog_records(
    *,
    dataset_id: str,
    source_configs: list[Path],
    stage_config_dir: Path,
    final_config_dir: Path,
    loader_root: Path,
    umi_publish_ledger: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str] | None]:
    umi_mapping = None
    ledger_provenance = None
    if dataset_id == "umi":
        if umi_publish_ledger is None or not umi_publish_ledger.is_file():
            raise FileNotFoundError("UMI manifest freeze requires publish.sqlite3")
        umi_mapping, ledger_provenance = _umi_index_to_source_id(
            umi_publish_ledger
        )

    records = []
    configs = []
    for source in source_configs:
        stage_config = stage_config_dir / source.name
        final_config = final_config_dir / source.name
        provenance = _write_config_snapshot(source, stage_config)
        provenance["resolved_path"] = str(final_config)
        configs.append(provenance)
        dataset = _build_four_frame_dataset(stage_config, loader_root)
        episodes = getattr(dataset, "_episodes", None)
        root_meta = getattr(dataset, "_root_meta", None)
        if not isinstance(episodes, list) or not isinstance(root_meta, list):
            raise RuntimeError("pinned canonical loader catalog ABI is unavailable")
        if len(root_meta) != 1:
            raise ValueError(f"{source}: manifest freeze requires one dataset root")
        source_fps = float(root_meta[0]["source_fps"])
        group = source.stem.removeprefix("hy_")
        for episode in episodes:
            canonical_index = int(episode["episode_index"])
            if dataset_id == "umi":
                episode_id = umi_mapping.get(canonical_index)
                if episode_id is None:
                    raise ValueError(
                        f"UMI ledger lacks canonical episode {canonical_index}"
                    )
            elif dataset_id == "hy":
                episode_id = f"{group}:{canonical_index:06d}"
            elif dataset_id == "libero":
                episode_id = f"libero:{canonical_index:06d}"
            else:
                raise ValueError(f"unsupported canonical dataset {dataset_id!r}")
            records.append(
                {
                    "episode_id": episode_id,
                    "canonical_config": str(final_config),
                    "canonical_config_sha256": provenance["resolved_sha256"],
                    "canonical_episode_index": canonical_index,
                    "canonical_group": group,
                    "canonical_rgb_target_length": int(
                        episode["rgb_target_length"]
                    ),
                    "source_fps": source_fps,
                }
            )
    if umi_mapping is not None and len(records) != len(umi_mapping):
        raise ValueError(
            f"UMI loader/ledger count mismatch: {len(records)} != {len(umi_mapping)}"
        )
    return records, configs, ledger_provenance


def freeze_manifest(
    *,
    dataset_id: str,
    canonical_config_root: Path,
    canonical_loader_root: Path,
    output_dir: Path,
    seed: int,
    umi_publish_ledger: Path | None,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if _git("status", "--porcelain"):
        raise ValueError("manifest freeze requires a clean Git worktree")
    source_root = canonical_config_root.expanduser().resolve()
    source_configs = sorted(source_root.glob("*.yaml"))
    if not source_configs:
        raise FileNotFoundError(f"no YAML configs under {source_root}")
    loader_root = canonical_loader_root.expanduser().resolve()
    stage = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    try:
        final_config_dir = output / "configs"
        records, configs, ledger = _catalog_records(
            dataset_id=dataset_id,
            source_configs=source_configs,
            stage_config_dir=stage / "configs",
            final_config_dir=final_config_dir,
            loader_root=loader_root,
            umi_publish_ledger=umi_publish_ledger,
        )
        records, split_counts = assign_splits(
            records, dataset_id=dataset_id, seed=seed
        )
        manifest = {
            "schema": CANONICAL_SPLIT_SCHEMA,
            "dataset_id": dataset_id,
            "split_policy": {
                "name": "sha256_global_episode_rank_exact_90_5_5_v1",
                "seed": int(seed),
                "ratios": SPLIT_RATIOS,
                "counts": split_counts,
                "unit": "episode",
            },
            "canonical_loader": {
                "path": str(loader_root),
                "git_sha": CANONICAL_LOADER_SHA,
            },
            "source_config_root": str(source_root),
            "configs": configs,
            "umi_publish_ledger": ledger,
            "generation": {
                "cwd": str(Path.cwd()),
                "git_branch": _git("branch", "--show-current"),
                "git_commit": _git("rev-parse", "HEAD"),
                "git_status_porcelain": "",
            },
            "records": records,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        filename = f"{dataset_id}-canonical-90-5-5-seed{seed}.json"
        manifest_path = stage / filename
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        file_sha = sha256_file(manifest_path)
        (stage / f"{filename}.sha256").write_text(
            f"{file_sha}  {filename}\n", encoding="utf-8"
        )
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "manifest_path": str(output / filename),
        "manifest_file_sha256": file_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "episode_count": len(records),
        "split_counts": split_counts,
        "config_count": len(configs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", choices=("umi", "hy", "libero"), required=True)
    parser.add_argument("--canonical-config-root", type=Path, required=True)
    parser.add_argument("--canonical-loader-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--umi-publish-ledger", type=Path)
    args = parser.parse_args()
    result = freeze_manifest(
        dataset_id=args.dataset_id,
        canonical_config_root=args.canonical_config_root,
        canonical_loader_root=args.canonical_loader_root,
        output_dir=args.output_dir,
        seed=args.seed,
        umi_publish_ledger=args.umi_publish_ledger,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
