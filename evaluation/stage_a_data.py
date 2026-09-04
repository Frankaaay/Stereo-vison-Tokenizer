"""Adapter from the pinned NGAD canonical loader to StereoVAE Stage A."""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from stereo_tokenizer.geometry import GeometryMapping
from stereo_tokenizer.pretrain_data import (
    HY_MONO_CAMERA_IDS,
    HY_SCHEMA,
    HyLanceMonoDataset,
    _mono_sample,
    _resolve_alias_path,
)

from .stage_a_contract import (
    CANONICAL_SPLIT_SCHEMA,
    canonical_sha256,
    read_identity_contract,
    read_selection,
    select_episode_windows,
    sha256_file,
    write_selection,
)


CANONICAL_LOADER_SHA = "d51377ac450b0066bc0c8eb13939bcfae47275ff"
CAMERA_KEYS = (
    "observation.images.cam_head_left",
    "observation.images.cam_head_right",
    "observation.images.cam_left_wrist_left",
    "observation.images.cam_left_wrist_right",
    "observation.images.cam_right_wrist_left",
    "observation.images.cam_right_wrist_right",
)
STEREO_VIEW_NAMES = ("head", "left_wrist", "right_wrist")
HY_EXCLUDED_TABLES = ("table_014",)


def _normalise_hy_table_name(value: Any) -> str:
    match = re.fullmatch(r"table[_-]?(\d+)", str(value).strip().lower())
    if match is None:
        raise ValueError(f"invalid Hy table name {value!r}")
    return f"table_{int(match.group(1)):03d}"


def parse_root_aliases(value: str | dict[str, str] | None) -> dict[str, Path]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("Hy root aliases must be valid JSON") from error
    if not isinstance(value, dict) or not value:
        raise ValueError("Hy evaluation requires non-empty root aliases")
    output = {}
    for alias, raw_path in value.items():
        if not isinstance(alias, str) or not alias or not isinstance(raw_path, str):
            raise ValueError("Hy root aliases must map strings to absolute paths")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        output[alias] = path
    return output


def _read_hy_manifest_matches(
    manifest_path: str | Path,
    expected_sha256: str,
    identities: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Hy manifest SHA mismatch: requested={expected_sha256}, actual={actual}"
        )
    normalized_identities = {
        (_normalise_hy_table_name(table), int(episode_index))
        for table, episode_index in identities
    }
    matched = {}
    open_text = gzip.open if path.suffix == ".gz" else open
    with open_text(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if record.get("schema") != HY_SCHEMA:
                raise ValueError(f"{path}:{line_number}: invalid Hy schema")
            key = (
                _normalise_hy_table_name(record.get("table_name")),
                int(record.get("episode_index", -1)),
            )
            if key not in normalized_identities:
                continue
            if key in matched:
                raise ValueError(f"Hy manifest contains duplicate identity {key}")
            record = dict(record)
            record["table_name"] = key[0]
            matched[key] = record
    return matched, {"path": str(path), "sha256": actual}


def _hy_lance_handle(
    record: dict[str, Any],
    root_aliases: dict[str, Path],
    handles: dict[tuple[str, str], Any],
):
    key = (str(record["root_alias"]), str(record["table_name"]))
    if key not in handles:
        try:
            import lance
        except ImportError as error:
            raise ImportError("Hy Stage A loading requires pylance") from error
        table_root = _resolve_alias_path(root_aliases, key[0], key[1])
        lance_path = table_root / f"{key[1]}.lance"
        if not lance_path.is_dir():
            raise FileNotFoundError(lance_path)
        handles[key] = lance.dataset(str(lance_path))
    return handles[key]


def _decode_hy_selection_record(
    selection_record: dict[str, Any],
    root_aliases: dict[str, Path],
    handles: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    record = selection_record["hy_manifest_record"]
    camera_columns = record.get("camera_columns")
    if not isinstance(camera_columns, dict) or set(camera_columns) != set(
        HY_MONO_CAMERA_IDS
    ):
        raise ValueError("selected Hy record has an invalid camera contract")
    frame_indices = np.asarray(
        selection_record["expected_source_frame_indices"], dtype=np.int64
    )
    columns = [camera_columns[camera] for camera in HY_MONO_CAMERA_IDS]
    rows = HyLanceMonoDataset._take_episode_frames(
        _hy_lance_handle(record, root_aliases, handles),
        int(record["episode_index"]),
        frame_indices,
        ["episode_index", "frame_index", "timestamp", *columns],
    )
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], np.float64)
    if not HyLanceMonoDataset._timestamps_match_frame_rate(
        timestamps, frame_indices, float(record.get("fps", 30.0))
    ):
        raise ValueError("selected Hy timestamps disagree with frame identity")
    stored_image = record.get("stored_image")
    expected_hw = (256, 256) if stored_image is not None else (240, 424)
    rgb = np.stack(
        [
            np.stack(
                [HyLanceMonoDataset._decode_jpeg(row[column], expected_hw) for row in rows]
            )
            for column in columns
        ]
    )
    source_hw_override = None
    if stored_image is not None:
        expected = {
            "encoded_size_hw": [256, 256],
            "source_size_hw": [240, 424],
            "transform": "source_240x424_letterbox_256",
            "content_bbox_yxyx": [55, 0, 200, 256],
        }
        if any(stored_image.get(key) != value for key, value in expected.items()):
            raise ValueError("selected Hy stored-image contract changed")
        mask_relative = Path(stored_image.get("pixel_mask_relative_path", ""))
        mask_path = _resolve_alias_path(
            root_aliases, record["root_alias"], str(mask_relative)
        )
        if (
            not mask_path.is_file()
            or sha256_file(mask_path) != stored_image.get("pixel_mask_sha256")
        ):
            raise ValueError("selected Hy pixel-mask asset changed")
        top, left, bottom, right = stored_image["content_bbox_yxyx"]
        rgb = rgb[:, :, :, top:bottom, left:right]
        source_hw_override = tuple(stored_image["source_size_hw"])
    sample = _mono_sample(
        rgb,
        sample_id=(
            f"hy/{record['table_name']}/{record['episode_id']}/"
            f"{int(frame_indices[0]):06d}"
        ),
        view_sample_ids=tuple(
            f"hy/{record['table_name']}/{record['episode_id']}/{camera}/"
            f"{int(frame_indices[0]):06d}"
            for camera in HY_MONO_CAMERA_IDS
        ),
        camera_ids=HY_MONO_CAMERA_IDS,
        episode_id=str(record["episode_id"]),
        dataset_id="hy",
        frame_indices=frame_indices,
        timestamps=timestamps,
        contract_sha256=str(record["source_contract_sha256"]),
        temporal_mode="four_frame",
        extra={"table_name": str(record["table_name"])},
        source_hw_override=source_hw_override,
    )
    sample["rgb_valid_mask"] = sample["non_padding_mask"]
    return sample


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def load_canonical_api(loader_root: str | Path):
    """Import only the exact reviewed canonical loader revision."""

    root = Path(loader_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    actual_sha = _git(root, "rev-parse", "HEAD")
    if actual_sha != CANONICAL_LOADER_SHA:
        raise ValueError(
            f"canonical loader SHA mismatch: expected {CANONICAL_LOADER_SHA}, "
            f"got {actual_sha}"
        )
    if _git(root, "status", "--porcelain"):
        raise ValueError(f"canonical loader worktree is dirty: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ngad_canonical_dataloader import NGADCanonicalDataset
    from ngad_canonical_dataloader.config import load_dataset_config

    return root, NGADCanonicalDataset, load_dataset_config


def _build_four_frame_dataset(config_path: Path, loader_root: str | Path):
    root, dataset_class, load_config = load_canonical_api(loader_root)
    config = load_config(config_path)
    kwargs = config.to_dataset_kwargs()
    if len(kwargs["dataset_dirs"]) != 1:
        raise ValueError("Stage A requires one physical canonical table per YAML")
    # Preserve the published root/mapping and semantic 10 Hz rate.  Evaluation
    # needs only four adjacent semantic frames and no state/action normalization.
    if float(kwargs["rgb_rate_hz"]) != 10.0:
        raise ValueError(f"{config_path}: Stage A requires rgb_rate_hz=10")
    kwargs.update(
        normalization_stats_path=None,
        frame_ranges=((0, 3),),
        max_samples=None,
        validation_split=0.0,
        split="train",
    )
    dataset = dataset_class(**kwargs)
    dataset.stage_a_resolved_config = kwargs
    dataset.stage_a_source_config = str(config_path)
    dataset.stage_a_loader_root = str(root)
    return dataset


def _episode_addresses(dataset) -> dict[int, tuple[int, dict[str, Any]]]:
    """Index a private loader ABI that is frozen by CANONICAL_LOADER_SHA."""

    episodes = getattr(dataset, "_episodes", None)
    ends = getattr(dataset, "_episode_window_ends", None)
    if not isinstance(episodes, list) or not isinstance(ends, list):
        raise RuntimeError("pinned canonical loader episode ABI is unavailable")
    if len(episodes) != len(ends):
        raise RuntimeError("canonical episode/end arrays disagree")
    output = {}
    previous = 0
    for episode, end in zip(episodes, ends):
        index = int(episode["episode_index"])
        if index in output:
            raise ValueError(f"duplicate canonical episode index {index}")
        output[index] = (previous, episode)
        previous = int(end)
    return output


def _config_for_identity(
    dataset_id: str, record: dict[str, Any], config_root: Path
) -> Path:
    if "canonical_config" in record:
        config = Path(record["canonical_config"]).expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
        return config
    if dataset_id == "umi":
        filename = "umi_table_000.yaml"
    elif dataset_id == "hy":
        table = str(record["table_name"])
        if not table.startswith("table_"):
            raise ValueError(f"invalid Hy table identity {table!r}")
        filename = f"hy_{table}.yaml"
    else:
        raise NotImplementedError(
            "LIBERO legacy suite-local episode IDs do not yet have an audited "
            "crosswalk to the new canonical global episode indices"
        )
    config = (config_root / filename).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    return config


def _umi_ledger(path: str | Path) -> tuple[dict[str, int], dict[str, str]]:
    ledger = Path(path).expanduser().resolve()
    if not ledger.is_file():
        raise FileNotFoundError(ledger)
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT source_id, episode_index FROM episodes"
        ).fetchall()
    finally:
        connection.close()
    mapping = {str(source_id): int(index) for source_id, index in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"{ledger}: duplicate source IDs")
    return mapping, {"path": str(ledger), "sha256": sha256_file(ledger)}


def build_canonical_selection(
    *,
    dataset_id: str,
    identity_contract_path: str | Path,
    canonical_config_root: str | Path | None,
    loader_root: str | Path | None,
    split: str,
    sample_count: int,
    seed: int,
    output: str | Path,
    umi_publish_ledger: str | Path | None = None,
    hy_manifest_path: str | Path | None = None,
    hy_manifest_sha256: str | None = None,
    hy_root_aliases: str | dict[str, str] | None = None,
) -> dict[str, Any]:
    """Map frozen checkpoint identities onto current canonical table windows."""

    identity = read_identity_contract(
        identity_contract_path, dataset_id=dataset_id
    )
    split_records = [
        record for record in identity["records"] if record.get("split") == split
    ]
    if not split_records:
        raise ValueError(f"identity contract has no {dataset_id}/{split} episodes")
    if dataset_id == "hy":
        if hy_manifest_path is None or hy_manifest_sha256 is None:
            raise ValueError("Hy selection requires its production manifest and SHA256")
        excluded = [
            record
            for record in split_records
            if _normalise_hy_table_name(record.get("table_name"))
            in HY_EXCLUDED_TABLES
        ]
        included = [
            record
            for record in split_records
            if _normalise_hy_table_name(record.get("table_name"))
            not in HY_EXCLUDED_TABLES
        ]
        requested = {
            (
                _normalise_hy_table_name(record["table_name"]),
                int(record["episode_index"]),
            )
            for record in included
        }
        manifest_records, manifest_provenance = _read_hy_manifest_matches(
            hy_manifest_path,
            hy_manifest_sha256,
            requested,
        )
        candidates = []
        missing = []
        for identity_record in included:
            key = (
                _normalise_hy_table_name(identity_record["table_name"]),
                int(identity_record["episode_index"]),
            )
            source = manifest_records.get(key)
            if source is None:
                missing.append(str(identity_record["episode_id"]))
                continue
            candidates.append(
                {
                    "legacy_episode_id": str(identity_record["episode_id"]),
                    "legacy_group": key[0],
                    "canonical_episode_index": key[1],
                    "canonical_rgb_target_length": int(source["length"]),
                    "source_fps": float(source.get("fps", 30.0)),
                    "window_count": int(source["window_count"]),
                    "hy_manifest_record": source,
                }
            )
        if missing:
            raise ValueError(
                "Hy post-exclusion identity join is incomplete: "
                f"missing={len(missing)}, sha256={canonical_sha256(sorted(missing))}"
            )
        aliases = parse_root_aliases(hy_root_aliases)
        handles: dict[tuple[str, str], Any] = {}

        def validate_hy_decode(row: dict[str, Any]) -> None:
            sample = _decode_hy_selection_record(row, aliases, handles)
            if tuple(sample["video"].shape) != (3, 1, 3, 4, 256, 256):
                raise ValueError("decoded Hy candidate has unexpected video shape")
            if not torch.isfinite(sample["video"]).all():
                raise ValueError("decoded Hy candidate contains NaN/Inf")

        selection = select_episode_windows(
            candidates,
            dataset_id=dataset_id,
            split=split,
            sample_count=sample_count,
            seed=seed,
            identity_contract=identity,
            candidate_validator=validate_hy_decode,
        )
        selection["data_backend"] = "hy_lance_manifest"
        selection["hy_manifest"] = manifest_provenance
        included_groups = sorted(
            {_normalise_hy_table_name(record["table_name"]) for record in included}
        )
        selected_groups = sorted(
            {str(record["legacy_group"]) for record in selection["records"]}
        )
        if selected_groups != included_groups:
            raise ValueError(
                "Hy selected windows do not cover every post-exclusion table: "
                f"expected={included_groups}, actual={selected_groups}"
            )
        selection["included_source_groups"] = included_groups
        selection["excluded_source_groups"] = {
            "groups": list(HY_EXCLUDED_TABLES),
            "episode_count": len(excluded),
            "episode_ids_sha256": canonical_sha256(
                sorted(str(record["episode_id"]) for record in excluded)
            ),
            "reason": "table_014 is explicitly unavailable for this evaluation",
        }
        selection["identity_mapping"] = {
            "identity_schema": identity["schema"],
            "identity_split_episode_count": len(split_records),
            "post_exclusion_episode_count": len(included),
            "mapped_complete_episode_count": len(candidates),
            "missing_episode_count": len(missing),
            "missing_episode_ids_sha256": canonical_sha256(sorted(missing)),
            "policy": "exact_identity_join_then_explicit_table_exclusion",
        }
        repo_root = Path.cwd().resolve()
        selection["generation"] = {
            "cwd": str(repo_root),
            "git_branch": _git(repo_root, "branch", "--show-current"),
            "git_commit": _git(repo_root, "rev-parse", "HEAD"),
            "git_status_porcelain": _git(repo_root, "status", "--porcelain"),
        }
        selection.pop("selection_sha256")
        selection["selection_sha256"] = canonical_sha256(selection)
        write_selection(output, selection)
        return selection
    if canonical_config_root is None or loader_root is None:
        raise ValueError("canonical selection requires config and loader roots")
    config_root = Path(canonical_config_root).expanduser().resolve()
    umi_mapping = None
    ledger_provenance = None
    direct_identity = all(
        "canonical_config" in record and "canonical_episode_index" in record
        for record in split_records
    )
    if dataset_id == "umi" and not direct_identity:
        if umi_publish_ledger is None:
            raise ValueError("legacy UMI mapping requires --umi-publish-ledger")
        umi_mapping, ledger_provenance = _umi_ledger(umi_publish_ledger)

    config_by_episode = {}
    missing = []
    for record in split_records:
        episode_id = str(record["episode_id"])
        try:
            config_by_episode[episode_id] = _config_for_identity(
                dataset_id, record, config_root
            )
            if direct_identity and sha256_file(config_by_episode[episode_id]) != record[
                "canonical_config_sha256"
            ]:
                raise ValueError(f"{episode_id}: canonical config drifted after manifest freeze")
        except FileNotFoundError:
            missing.append(episode_id)
    configs = set(config_by_episode.values())
    catalogs = {}
    for config in sorted(configs):
        dataset = _build_four_frame_dataset(config, loader_root)
        catalogs[str(config)] = (dataset, _episode_addresses(dataset))

    candidates = []
    for record in split_records:
        episode_id = str(record["episode_id"])
        config = config_by_episode.get(episode_id)
        if config is None:
            continue
        if direct_identity:
            canonical_index = int(record["canonical_episode_index"])
        elif dataset_id == "umi":
            canonical_index = umi_mapping.get(episode_id)
            if canonical_index is None:
                missing.append(episode_id)
                continue
        else:
            canonical_index = int(record["episode_index"])
        addresses = catalogs[str(config)][1]
        address = addresses.get(canonical_index)
        if address is None:
            missing.append(str(record["episode_id"]))
            continue
        _, episode = address
        target_length = int(episode["rgb_target_length"])
        root_meta = getattr(catalogs[str(config)][0], "_root_meta", None)
        if not isinstance(root_meta, list):
            raise RuntimeError("pinned canonical loader root metadata ABI is unavailable")
        source_fps = float(root_meta[int(episode["root_index"])]["source_fps"])
        if "source_fps" in record and source_fps != float(record["source_fps"]):
            raise ValueError(f"{episode_id}: source FPS drifted after manifest freeze")
        if "canonical_rgb_target_length" in record and target_length != int(
            record["canonical_rgb_target_length"]
        ):
            raise ValueError(
                f"{episode_id}: RGB target length drifted after manifest freeze"
            )
        window_count = max(0, (target_length - 4) // 4 + 1)
        candidates.append(
            {
                "legacy_episode_id": str(record["episode_id"]),
                "legacy_group": str(
                    record.get("table_name", record.get("suite", ""))
                ),
                "canonical_config": str(config),
                "canonical_config_sha256": sha256_file(config),
                "canonical_episode_index": int(canonical_index),
                "canonical_rgb_target_length": target_length,
                "source_fps": source_fps,
                "window_count": window_count,
            }
        )

    candidate_validator = None
    if identity["schema"] == CANONICAL_SPLIT_SCHEMA:
        def validate_decode(row: dict[str, Any]) -> None:
            config = row["canonical_config"]
            dataset, addresses = catalogs[config]
            start, _ = addresses[int(row["canonical_episode_index"])]
            source = dataset[start + int(row["anchor_rgb_index"])]
            if source["frame_offsets"].tolist() != row["expected_frame_offsets"]:
                raise ValueError("decoded frame offsets disagree with selection")
            if (
                source["source_frame_indices"].tolist()
                != row["expected_source_frame_indices"]
            ):
                raise ValueError("decoded source frames disagree with selection")
            if not bool(source["frame_valid"].all()):
                raise ValueError("decoded candidate contains invalid frames")
            if tuple(source["video"].shape) != (4, 6, 3, 256, 256):
                raise ValueError("decoded candidate has unexpected video shape")
            if not torch.isfinite(source["video"]).all():
                raise ValueError("decoded candidate video contains NaN or Inf")
            if tuple(source["image_pixel_mask"].shape) != (4, 6, 256, 256):
                raise ValueError("decoded candidate has unexpected pixel-mask shape")

        candidate_validator = validate_decode

    selection = select_episode_windows(
        candidates,
        dataset_id=dataset_id,
        split=split,
        sample_count=sample_count,
        seed=seed,
        identity_contract=identity,
        candidate_validator=candidate_validator,
    )
    selection["canonical_loader"] = {
        "path": str(Path(loader_root).expanduser().resolve()),
        "git_sha": CANONICAL_LOADER_SHA,
    }
    selection["umi_publish_ledger"] = ledger_provenance
    selection["identity_mapping"] = {
        "identity_schema": identity["schema"],
        "identity_split_episode_count": len(split_records),
        "mapped_complete_episode_count": len(candidates),
        "missing_episode_count": len(missing),
        "missing_episode_ids_sha256": canonical_sha256(sorted(missing)),
        "policy": "sample_only_from_exact_episode_identity_intersection",
    }
    repo_root = Path.cwd().resolve()
    selection["generation"] = {
        "cwd": str(repo_root),
        "git_branch": _git(repo_root, "branch", "--show-current"),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": _git(repo_root, "status", "--porcelain"),
    }
    selection.pop("selection_sha256")
    selection["selection_sha256"] = canonical_sha256(selection)
    write_selection(output, selection)
    return selection


class CanonicalStageADataset(Dataset):
    """Decode fixed selected windows through the official canonical loader."""

    fixed_selection = True

    def __init__(
        self,
        selection_path: str | Path,
        *,
        loader_root: str | Path | None,
        eye_mode: str,
        camera_key: str | None = None,
        hy_root_aliases: str | dict[str, str] | None = None,
    ) -> None:
        self.selection = read_selection(selection_path)
        self.selection_path = Path(selection_path).expanduser().resolve()
        self.dataset_id = str(self.selection["dataset_id"])
        self.eye_mode = str(eye_mode)
        if self.selection.get("data_backend") == "hy_lance_manifest":
            if self.dataset_id != "hy" or self.eye_mode != "mono":
                raise ValueError("Hy Lance Stage A selection requires hy/mono")
            if camera_key is not None:
                raise ValueError("Hy Lance Stage A evaluates all three mono views together")
            if self.selection.get("excluded_source_groups", {}).get("groups") != list(
                HY_EXCLUDED_TABLES
            ):
                raise ValueError("Hy excluded-table contract changed")
            selected_groups = sorted(
                {str(record["legacy_group"]) for record in self.selection["records"]}
            )
            if selected_groups != self.selection.get("included_source_groups"):
                raise ValueError("Hy selected-table coverage contract changed")
            manifest = self.selection.get("hy_manifest", {})
            if sha256_file(manifest.get("path", "")) != manifest.get("sha256"):
                raise ValueError("Hy production manifest drifted after selection creation")
            self.loader_root = None
            self.camera_key = None
            self.camera_index = None
            self.view_names = HY_MONO_CAMERA_IDS
            self._hy_root_aliases = parse_root_aliases(hy_root_aliases)
            self._hy_handles: dict[tuple[str, str], Any] = {}
            identity_meta = self.selection["identity_contract"]
            identity = read_identity_contract(
                identity_meta["path"], dataset_id=self.dataset_id
            )
            if (
                identity["identity_contract_sha256"] != identity_meta["sha256"]
                or identity["source_manifest_sha256"]
                != identity_meta["source_manifest_sha256"]
            ):
                raise ValueError("Hy identity contract drifted after selection creation")
            split_ids = {
                str(record["episode_id"])
                for record in identity["records"]
                if record.get("split") == self.selection["split"]
                and _normalise_hy_table_name(record.get("table_name"))
                not in HY_EXCLUDED_TABLES
            }
            selected_ids = {
                str(record["legacy_episode_id"])
                for record in self.selection["records"]
            }
            if not selected_ids.issubset(split_ids):
                raise ValueError("Hy selection escaped its frozen post-exclusion split")
            return
        if loader_root is None:
            raise ValueError("canonical Stage A dataset requires a loader root")
        actual_loader = Path(loader_root).expanduser().resolve()
        if not actual_loader.is_dir():
            raise FileNotFoundError(f"canonical loader is missing: {actual_loader}")
        expected_loader_sha = str(self.selection["canonical_loader"]["git_sha"])
        actual_loader_sha = subprocess.check_output(
            ("git", "-C", str(actual_loader), "rev-parse", "HEAD"),
            text=True,
        ).strip()
        loader_status = subprocess.check_output(
            ("git", "-C", str(actual_loader), "status", "--porcelain"),
            text=True,
        ).strip()
        if actual_loader_sha != expected_loader_sha or loader_status:
            raise ValueError(
                "canonical loader source mismatch: "
                f"expected_sha={expected_loader_sha}, actual_sha={actual_loader_sha}, "
                f"dirty={bool(loader_status)}"
            )
        self.loader_root = actual_loader
        identity_meta = self.selection["identity_contract"]
        identity = read_identity_contract(
            identity_meta["path"], dataset_id=self.dataset_id
        )
        if identity["identity_contract_sha256"] != identity_meta["sha256"]:
            raise ValueError("identity contract file drifted after selection creation")
        if (
            identity["source_manifest_sha256"]
            != identity_meta["source_manifest_sha256"]
        ):
            raise ValueError(
                "identity contract semantic digest drifted after selection creation"
            )
        split_ids = {
            str(record["episode_id"])
            for record in identity["records"]
            if record.get("split") == self.selection["split"]
        }
        selected_ids = {
            str(record["legacy_episode_id"])
            for record in self.selection["records"]
        }
        if not selected_ids.issubset(split_ids):
            raise ValueError("selection contains episodes outside the frozen split")
        ledger = self.selection.get("umi_publish_ledger")
        if ledger is not None and sha256_file(ledger["path"]) != ledger["sha256"]:
            raise ValueError("UMI publish ledger drifted after selection creation")
        if self.eye_mode not in {"mono", "stereo"}:
            raise ValueError("eye_mode must be mono or stereo")
        if self.eye_mode == "stereo":
            if self.dataset_id != "umi":
                raise ValueError("only canonical UMI has true stereo cameras")
            if camera_key is not None:
                raise ValueError("stereo mode does not accept one camera key")
            self.camera_key = None
            self.camera_index = None
            self.view_names = STEREO_VIEW_NAMES
        else:
            if camera_key not in CAMERA_KEYS:
                raise ValueError(f"mono mode requires one of {CAMERA_KEYS}")
            self.camera_key = str(camera_key)
            self.camera_index = CAMERA_KEYS.index(self.camera_key)
            self.view_names = (self.camera_key.removeprefix("observation.images."),)

        self._datasets = {}
        self._addresses = {}
        for config_text in sorted(
            {record["canonical_config"] for record in self.selection["records"]}
        ):
            config = Path(config_text).resolve()
            expected_sha = {
                record["canonical_config_sha256"]
                for record in self.selection["records"]
                if record["canonical_config"] == config_text
            }
            if len(expected_sha) != 1 or sha256_file(config) not in expected_sha:
                raise ValueError(f"canonical config drift: {config}")
            dataset = _build_four_frame_dataset(config, loader_root)
            self._datasets[config_text] = dataset
            self._addresses[config_text] = _episode_addresses(dataset)

        for record in self.selection["records"]:
            addresses = self._addresses[record["canonical_config"]]
            episode = addresses.get(int(record["canonical_episode_index"]))
            if episode is None:
                raise ValueError("selected canonical episode disappeared")
            if int(record["anchor_rgb_index"]) + 3 >= int(
                episode[1]["rgb_target_length"]
            ):
                raise ValueError("selected four-frame window crosses an episode")

    def __len__(self) -> int:
        return len(self.selection["records"])

    def _common(self, source: dict[str, Any], record: dict[str, Any]):
        offsets = source["frame_offsets"].tolist()
        indices = source["source_frame_indices"].tolist()
        if offsets != record["expected_frame_offsets"]:
            raise ValueError(f"frame offset drift: {offsets}")
        if indices != record["expected_source_frame_indices"]:
            raise ValueError(f"source frame drift: {indices}")
        if not bool(source["frame_valid"].all()):
            raise ValueError("selected Stage A window contains an invalid frame")
        return {
            "sample_id": (
                f"{self.dataset_id}:{record['legacy_episode_id']}:"
                f"{record['legacy_window_index']}"
            ),
            "episode_id": record["legacy_episode_id"],
            "dataset_id": self.dataset_id,
            "frame_index": source["source_frame_indices"].clone(),
            "timestamp_s": source["frame_timestamps"].clone(),
            "contract_sha256": self.selection["selection_sha256"],
            "temporal_mode": "four_frame",
            "mode_id": f"{self.eye_mode}/four_frame",
            "eye_mode": self.eye_mode,
            "data_info": source["data_info"],
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.selection["records"][index]
        if self.selection.get("data_backend") == "hy_lance_manifest":
            return _decode_hy_selection_record(
                record, self._hy_root_aliases, self._hy_handles
            )
        config = record["canonical_config"]
        start, _ = self._addresses[config][int(record["canonical_episode_index"])]
        source = self._datasets[config][start + int(record["anchor_rgb_index"])]
        common = self._common(source, record)
        video = source["video"]
        pixel_mask = source["image_pixel_mask"]
        camera_mask = source["camera_mask"]
        if tuple(video.shape) != (4, 6, 3, 256, 256):
            raise ValueError(f"unexpected canonical video shape {tuple(video.shape)}")
        if tuple(pixel_mask.shape) != (4, 6, 256, 256):
            raise ValueError("unexpected canonical pixel-mask shape")
        if self.eye_mode == "stereo":
            if not bool(camera_mask.all()):
                raise ValueError("stereo Stage A requires all six cameras")
            student = video.reshape(4, 3, 2, 3, 256, 256).permute(
                1, 2, 3, 0, 4, 5
            ) / 2.0
            masks = pixel_mask.reshape(4, 3, 2, 256, 256)
            rgb_mask = masks[:, :, 0].permute(1, 0, 2, 3).unsqueeze(1)
            return {
                **common,
                "video": student.contiguous(),
                "rgb_valid_mask": rgb_mask.contiguous(),
                "view_count": 3,
                "teacher_kind": "foundation_stereo",
            }

        camera = int(self.camera_index)
        if not bool(camera_mask[:, camera].all()):
            raise ValueError(f"selected mono camera is unavailable: {self.camera_key}")
        selected = video[:, camera]
        student = selected.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0) / 2.0
        rgb_mask = pixel_mask[:, camera].unsqueeze(0).unsqueeze(0)
        raw_rgb = ((selected + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        spatial_mask = pixel_mask[0, camera]
        if not torch.equal(
            pixel_mask[:, camera], spatial_mask.unsqueeze(0).expand(4, -1, -1)
        ):
            raise ValueError("mono pixel mask must be time-invariant")
        positions = torch.nonzero(spatial_mask, as_tuple=False)
        y0, x0 = positions.min(dim=0).values.tolist()
        y1, x1 = (positions.max(dim=0).values + 1).tolist()
        rectangle = torch.zeros_like(spatial_mask)
        rectangle[y0:y1, x0:x1] = True
        if not torch.equal(spatial_mask, rectangle):
            raise ValueError("mono pixel mask must describe one content rectangle")
        raw_content = raw_rgb[:, :, y0:y1, x0:x1]
        geometry = GeometryMapping.create(
            (y1 - y0, x1 - x0), source_hw=(y1 - y0, x1 - x0)
        )
        _, geometry_mask = geometry.student_letterbox(raw_content)
        expected_mask = geometry_mask.permute(1, 0, 2, 3).unsqueeze(0)
        if not torch.equal(expected_mask, rgb_mask):
            raise ValueError("published pixel mask disagrees with geometry mapping")
        return {
            **common,
            "video": student.contiguous(),
            "rgb_valid_mask": rgb_mask.contiguous(),
            "non_padding_mask": rgb_mask.contiguous(),
            "da3_images": geometry.da3_preprocess(raw_content).unsqueeze(0),
            "geometry_mapping": geometry.to_collatable_metadata(),
            "view_count": 1,
            "teacher_kind": "da3",
            "camera_key": self.camera_key,
        }

    def provenance(self) -> dict[str, Any]:
        if self.selection.get("data_backend") == "hy_lance_manifest":
            return {
                "dataset_id": self.dataset_id,
                "selection_path": str(self.selection_path),
                "selection_sha256": self.selection["selection_sha256"],
                "identity_contract": self.selection["identity_contract"],
                "sample_count": len(self),
                "eye_mode": self.eye_mode,
                "camera_key": None,
                "video_contract": "[B,3,1,3,T,H,W]",
                "data_backend": "hy_lance_manifest",
                "hy_manifest": self.selection["hy_manifest"],
                "excluded_source_groups": self.selection["excluded_source_groups"],
                "included_source_groups": self.selection["included_source_groups"],
                "root_aliases": {
                    alias: str(path) for alias, path in sorted(self._hy_root_aliases.items())
                },
            }
        return {
            "dataset_id": self.dataset_id,
            "selection_path": str(self.selection_path),
            "selection_sha256": self.selection["selection_sha256"],
            "identity_contract": self.selection["identity_contract"],
            "canonical_configs": sorted(self._datasets),
            "canonical_loader": self.selection["canonical_loader"],
            "canonical_loader_runtime_path": str(self.loader_root),
            "sample_count": len(self),
            "eye_mode": self.eye_mode,
            "camera_key": self.camera_key,
            "video_contract": (
                "[B,3,2,3,T,H,W]" if self.eye_mode == "stereo"
                else "[B,1,1,3,T,H,W]"
            ),
        }
