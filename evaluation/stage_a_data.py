"""Adapter from the pinned NGAD canonical loader to StereoVAE Stage A."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from stereo_tokenizer.geometry import GeometryMapping

from .stage_a_contract import (
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
    canonical_config_root: str | Path,
    loader_root: str | Path,
    split: str,
    sample_count: int,
    seed: int,
    output: str | Path,
    umi_publish_ledger: str | Path | None = None,
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
    selection = select_episode_windows(
        candidates,
        dataset_id=dataset_id,
        split=split,
        sample_count=sample_count,
        seed=seed,
        identity_contract=identity,
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
        loader_root: str | Path,
        eye_mode: str,
        camera_key: str | None = None,
    ) -> None:
        self.selection = read_selection(selection_path)
        self.selection_path = Path(selection_path).expanduser().resolve()
        self.dataset_id = str(self.selection["dataset_id"])
        expected_loader = Path(
            self.selection["canonical_loader"]["path"]
        ).expanduser().resolve()
        actual_loader = Path(loader_root).expanduser().resolve()
        if actual_loader != expected_loader:
            raise ValueError(
                f"canonical loader path mismatch: {actual_loader} != {expected_loader}"
            )
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
        self.eye_mode = str(eye_mode)
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
            "da3_images": geometry.da3_preprocess(raw_content),
            "geometry_mapping": geometry.to_collatable_metadata(),
            "view_count": 1,
            "teacher_kind": "da3",
            "camera_key": self.camera_key,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "selection_path": str(self.selection_path),
            "selection_sha256": self.selection["selection_sha256"],
            "identity_contract": self.selection["identity_contract"],
            "canonical_configs": sorted(self._datasets),
            "canonical_loader": self.selection["canonical_loader"],
            "sample_count": len(self),
            "eye_mode": self.eye_mode,
            "camera_key": self.camera_key,
            "video_contract": (
                "[B,3,2,3,T,H,W]" if self.eye_mode == "stereo"
                else "[B,1,1,3,T,H,W]"
            ),
        }
