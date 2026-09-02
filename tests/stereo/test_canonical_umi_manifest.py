import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

from stereo_tokenizer.lerobot_data import (
    CANONICAL_STORED_TRANSFORM,
    LeRobotStereoDataset,
    VIDEO_KEYS,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts" / "data" / "build_canonical_umi_stereo_manifest.py"
SPEC = importlib.util.spec_from_file_location("canonical_umi_manifest_builder", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def camera(fx=200.0, tx=0.0):
    return {
        "K": [fx, 0, 320, 0, fx, 240, 0, 0, 1],
        "D": [0.0] * 8,
        "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "P": [fx, 0, 320, tx, 0, fx, 240, 0, 0, 0, 1, 0],
        "width": 640,
        "height": 480,
        "distortion_model": "rational_polynomial",
    }


def calibration():
    return {
        source_view: {"left": camera(), "right": camera(tx=-11.0)}
        for source_view in ("head", "left_wrist", "right_wrist")
    }


class CanonicalUMIManifestTest(unittest.TestCase):
    def test_outer_dataset_root_resolves_one_published_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table_root = root / "table_000"
            (table_root / "meta" / "episodes").mkdir(parents=True)
            (table_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
            self.assertEqual(BUILDER._canonical_table_root(root), table_root)

    def test_delivery_calibration_view_names_are_normalized(self):
        source = calibration()
        normalized = BUILDER._normalize_calibration(source, "a" * 64)
        self.assertEqual(set(normalized), {"head", "lefthand", "righthand"})
        self.assertIs(normalized["lefthand"], source["left_wrist"])

    def test_builder_maps_flat_canonical_videos_and_reader_skips_second_resize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for source_key in BUILDER.CANONICAL_VIDEO_KEYS.values():
                path = root / "videos" / source_key / "chunk-000" / "file-007.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            mask_path = root / "image_pixel_mask_umi.npz"
            np.savez(mask_path, mask=np.ones((256, 256), dtype=bool))
            mask_sha256 = sha256_file(mask_path)
            row = {"episode_index": 3, "length": 22, "tasks": ["pick"]}
            for source_key in BUILDER.CANONICAL_VIDEO_KEYS.values():
                row[f"videos/{source_key}/chunk_index"] = 0
                row[f"videos/{source_key}/file_index"] = 7
                row[f"videos/{source_key}/from_timestamp"] = 1.0
                row[f"videos/{source_key}/to_timestamp"] = 2.0
            mappings = {
                3: {
                    "episode_uuid": "episode-uuid",
                    "source_sidecar": "/raw/episode-uuid.json",
                    "sidecar_sha256": "a" * 64,
                    "calibration_bundle_sha256": "b" * 64,
                }
            }
            bundles = {
                "b" * 64: {
                    manifest_view: calibration()[source_view]
                    for manifest_view, source_view in BUILDER.CALIBRATION_VIEW_KEYS.items()
                }
            }
            mask = {"path": str(mask_path), "sha256": mask_sha256}
            with mock.patch.object(BUILDER, "_episode_rows", return_value=[row]):
                records = BUILDER.collect_records(
                    root, mappings, bundles, "c" * 64, mask
                )
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["shard_id"], "canonical_000_007")
            self.assertEqual(
                record["videos"][VIDEO_KEYS[("head", "left")]]["relative_path"],
                "videos/observation.images.cam_head_left/chunk-000/file-007.mp4",
            )
            self.assertEqual(
                record["stored_image"]["transform"], CANONICAL_STORED_TRANSFORM
            )

            record["split"] = "train"
            record["contract_sha256"] = "d" * 64
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = LeRobotStereoDataset(
                manifest,
                root,
                split="train",
                expected_rectification_audit_sha256="c" * 64,
            )
            image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
            prepared = dataset._prepare_image(image, record, "head", "left")
            self.assertIs(prepared, image)
            fx, baseline = dataset._output_calibration(record)
            np.testing.assert_allclose(fx, [80.0, 80.0, 80.0])
            np.testing.assert_allclose(baseline, [0.055, 0.055, 0.055])

    def test_split_assignment_remains_exact_and_deterministic(self):
        records = [
            {"episode_id": f"episode-{index}", "shard_id": "canonical_000_000"}
            for index in range(100)
        ]
        first = [dict(record) for record in records]
        second = [dict(record) for record in records]
        BUILDER.assign_splits(first, 1234)
        BUILDER.assign_splits(second, 1234)
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(record["split"] for record in first),
            Counter({"train": 90, "val": 5, "test": 5}),
        )


if __name__ == "__main__":
    unittest.main()
