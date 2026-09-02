"""Immutable identity and sample-selection contracts for Stage A."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


IDENTITY_SCHEMA = "stereo-tokenizer-checkpoint-split-identities-v1"
CANONICAL_SPLIT_SCHEMA = "stereo-tokenizer-canonical-split-manifest-v1"
SELECTION_SCHEMA = "stereo-tokenizer-stage-a-selection-v1"
IDENTITY_CONTRACT_SHA256 = {
    "hy": "fc0075580bbb5a353a9ae151ad8a604a5665b78bca0bf98f92eafcb6f0a17caf",
    "libero": "283be628c3449cad895d618238742b2dd0a21b32947e9a5dc979639591b1e715",
    "umi": "f3b7f85c32573edbb75e750cd3986fc6d68875243bbbb95f8a1d3cfa0c236a12",
}
SOURCE_MANIFEST_SHA256 = {
    "hy": "b25efc945ccd7e7afd2f1a76393ea19adde8fa072e1e9a2ca6348e0e5c1a45f9",
    "libero": "0299354a7225e979f6b9ff5fb3e26a975c811d2d41af44e042a6eade3f24bbf4",
    "umi": "5e8f58c769549372af070a6132ad826bd7172aaeabcebebff84426e66bc2120f",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_order(seed: int, *parts: object) -> bytes:
    value = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).digest()


def read_identity_contract(
    path: str | Path, *, dataset_id: str
) -> dict[str, Any]:
    """Read only the split identities extracted from checkpoint provenance."""

    if dataset_id not in IDENTITY_CONTRACT_SHA256:
        raise ValueError(f"unsupported Stage A dataset {dataset_id!r}")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    actual = sha256_file(source)
    schema = payload.get("schema")
    if schema == IDENTITY_SCHEMA:
        expected = IDENTITY_CONTRACT_SHA256[dataset_id]
        if actual != expected:
            raise ValueError(
                f"{source}: identity SHA256 mismatch, expected {expected}, got {actual}"
            )
        expected_header = {
            "schema": IDENTITY_SCHEMA,
            "dataset_id": dataset_id,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256[dataset_id],
        }
    elif schema == CANONICAL_SPLIT_SCHEMA:
        digest = payload.pop("manifest_sha256", None)
        if digest != canonical_sha256(payload):
            raise ValueError(f"{source}: canonical split manifest SHA256 mismatch")
        payload["manifest_sha256"] = digest
        payload["source_manifest_sha256"] = digest
        expected_header = {
            "schema": CANONICAL_SPLIT_SCHEMA,
            "dataset_id": dataset_id,
        }
    else:
        raise ValueError(f"{source}: unsupported identity schema {schema!r}")
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected_header.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{source}: identity contract mismatch {mismatches}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{source}: identity records must be non-empty")
    episode_ids = [str(record.get("episode_id", "")) for record in records]
    if any(not value for value in episode_ids) or len(set(episode_ids)) != len(
        episode_ids
    ):
        raise ValueError(f"{source}: empty or duplicate episode identities")
    invalid_splits = {
        record.get("split") for record in records
    } - {"train", "val", "test"}
    if invalid_splits:
        raise ValueError(f"{source}: invalid split labels {sorted(map(str, invalid_splits))}")
    if schema == CANONICAL_SPLIT_SCHEMA:
        required = {
            "episode_id",
            "split",
            "canonical_config",
            "canonical_config_sha256",
            "canonical_episode_index",
            "canonical_rgb_target_length",
            "source_fps",
        }
        if any(not required.issubset(record) for record in records):
            raise ValueError(f"{source}: canonical episode record fields are incomplete")
        policy = payload.get("split_policy", {})
        if policy.get("name") != "sha256_global_episode_rank_exact_90_5_5_v1":
            raise ValueError(f"{source}: unsupported canonical split policy")
        actual_counts = dict(Counter(record["split"] for record in records))
        if actual_counts != policy.get("counts"):
            raise ValueError(f"{source}: canonical split counts mismatch")
        loader = payload.get("canonical_loader", {})
        if loader.get("git_sha") != "d51377ac450b0066bc0c8eb13939bcfae47275ff":
            raise ValueError(f"{source}: canonical loader SHA mismatch")
    payload["identity_contract_path"] = str(source)
    payload["identity_contract_sha256"] = actual
    return payload


def select_episode_windows(
    candidates: Iterable[dict[str, Any]],
    *,
    dataset_id: str,
    split: str,
    sample_count: int,
    seed: int,
    identity_contract: dict[str, Any],
) -> dict[str, Any]:
    """Choose one deterministic non-overlapping window from distinct episodes."""

    if split not in {"val", "test"}:
        raise ValueError("formal Stage A selection is restricted to val/test")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    usable = []
    for candidate in candidates:
        windows = int(candidate["window_count"])
        if windows < 1:
            continue
        episode_id = str(candidate["legacy_episode_id"])
        window_index = int.from_bytes(
            hash_order(seed, "window", dataset_id, episode_id)[:8], "big"
        ) % windows
        row = dict(candidate)
        row.pop("window_count")
        row["legacy_window_index"] = window_index
        row["anchor_rgb_index"] = window_index * 4
        row["expected_frame_offsets"] = [0, 1, 2, 3]
        source_fps = float(row["source_fps"])
        if source_fps <= 0:
            raise ValueError(f"{dataset_id}: source_fps must be positive")
        row["expected_source_frame_indices"] = [
            round((row["anchor_rgb_index"] + offset) * source_fps / 10.0)
            for offset in (0, 1, 2, 3)
        ]
        usable.append(row)
    usable.sort(
        key=lambda row: hash_order(
            seed, "episode", dataset_id, row["legacy_episode_id"]
        )
    )
    if len(usable) < sample_count:
        raise ValueError(
            f"{dataset_id}/{split} has {len(usable)} mapped complete episodes, "
            f"needs {sample_count}"
        )
    selected = usable[:sample_count]
    for selection_index, row in enumerate(selected):
        row["selection_index"] = selection_index
    contract = {
        "schema": SELECTION_SCHEMA,
        "dataset_id": dataset_id,
        "split": split,
        "seed": int(seed),
        "sample_count": int(sample_count),
        "selection_policy": (
            "sha256_distinct_episode_then_one_nonoverlapping_window"
        ),
        "semantic_rgb_rate_hz": 10,
        "identity_contract": {
            "path": identity_contract["identity_contract_path"],
            "sha256": identity_contract["identity_contract_sha256"],
            "source_manifest_sha256": identity_contract[
                "source_manifest_sha256"
            ],
        },
        "records": selected,
    }
    contract["selection_sha256"] = canonical_sha256(contract)
    return contract


def write_selection(path: str | Path, contract: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_selection(path: str | Path) -> dict[str, Any]:
    selection = Path(path).expanduser().resolve()
    payload = json.loads(selection.read_text(encoding="utf-8"))
    digest = payload.pop("selection_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"{selection}: selection SHA256 mismatch")
    payload["selection_sha256"] = digest
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError(f"{selection}: unsupported selection schema")
    records = payload.get("records", [])
    if len(records) != int(payload.get("sample_count", -1)):
        raise ValueError(f"{selection}: record count mismatch")
    identities = [
        (
            row["canonical_config"],
            int(row["canonical_episode_index"]),
            int(row["anchor_rgb_index"]),
        )
        for row in records
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{selection}: duplicate canonical windows")
    payload["selection_path"] = str(selection)
    return payload
