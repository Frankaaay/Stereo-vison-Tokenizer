import json
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.data.build_pretrain_manifest import (
    HY_CANONICAL_CAMERA_COLUMNS,
    _hy_camera_contract,
)
from stereo_tokenizer.data import _load_root_aliases
from stereo_tokenizer.pretrain_data import HY_SCHEMA, HyLanceMonoDataset, _mono_sample


class HyThreeCameraManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _manifest(self):
        path = self.root / "hy.jsonl"
        records = []
        for split in ("train", "val"):
            records.append(
                {
                    "schema": HY_SCHEMA,
                    "split": split,
                    "root_alias": "hy_primary",
                    "table_name": "table_001",
                    "episode_id": f"{split}-1",
                    "episode_index": 1,
                    "length": 24,
                    "dataset_from_index": 100,
                    "camera_columns": {
                        "cam_high": "observation_images_cam_high",
                        "cam_left_wrist": "observation_images_cam_left_wrist",
                        "cam_right_wrist": "observation_images_cam_right_wrist",
                    },
                    "window_count": 2,
                    "source_contract_sha256": "a" * 64,
                }
            )
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_hy_manifest_expands_three_cameras_equally(self):
        dataset = HyLanceMonoDataset(
            self._manifest(), {"hy_primary": self.root}, split="train"
        )
        self.assertEqual(
            [span.variant for span in dataset.spans],
            ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        )
        self.assertEqual([span.sample_count for span in dataset.spans], [2, 2, 2])
        self.assertEqual(len(dataset), 6)

    def test_canonical_hy_contract_crops_padding_before_geometry(self):
        mask_path = (
            self.root
            / "dataset_configs"
            / "masks"
            / "image_pixel_mask_hy_embodied.npz"
        )
        mask_path.parent.mkdir(parents=True)
        mask = np.zeros((256, 256), dtype=bool)
        mask[55:200, :] = True
        np.savez(mask_path, mask=mask)
        camera_columns, stored_image = _hy_camera_contract(
            self.root, HY_CANONICAL_CAMERA_COLUMNS.values()
        )
        self.assertEqual(camera_columns, HY_CANONICAL_CAMERA_COLUMNS)
        self.assertEqual(stored_image["content_bbox_yxyx"], [55, 0, 200, 256])

        canonical = np.zeros((256, 256, 3), dtype=np.uint8)
        canonical[55:200] = 127
        encoded = io.BytesIO()
        Image.fromarray(canonical).save(encoded, format="JPEG")
        decoded = HyLanceMonoDataset._decode_jpeg(encoded.getvalue(), (256, 256))
        cropped = decoded[:, 55:200, :][None]
        sample = _mono_sample(
            cropped,
            sample_id="hy/sample",
            episode_id="episode",
            dataset_id="hy",
            frame_indices=np.asarray([0], dtype=np.int64),
            timestamps=np.asarray([0.0], dtype=np.float64),
            contract_sha256="a" * 64,
            temporal_mode="single_frame",
            extra={},
            source_hw_override=(240, 424),
        )
        self.assertEqual(sample["geometry_mapping"]["source_hw"].tolist(), [240, 424])
        self.assertEqual(sample["geometry_mapping"]["rectified_hw"].tolist(), [145, 256])
        self.assertEqual(int(sample["non_padding_mask"].sum()), 37120)

    def test_root_alias_json_is_node_local(self):
        mapping = _load_root_aliases(
            json.dumps({"hy_primary": str(self.root)}), "--hy_root_aliases"
        )
        self.assertEqual(mapping, {"hy_primary": str(self.root.resolve())})

    def test_lance_rows_are_selected_by_episode_and_frame_identity(self):
        class Table:
            def __init__(self, rows):
                self.rows = rows

            def to_pylist(self):
                return self.rows

        class LanceDataset:
            def __init__(self):
                self.filter = None
                self.rows = [
                    {"episode_index": 7, "frame_index": 9},
                    {"episode_index": 7, "frame_index": 3},
                ]

            def to_table(self, *, filter, columns):
                self.filter = filter
                self.columns = columns
                return Table(self.rows)

        lance_dataset = LanceDataset()
        rows = HyLanceMonoDataset._take_episode_frames(
            lance_dataset,
            episode_index=7,
            frame_indices=[3, 9],
            columns=["episode_index", "frame_index"],
        )
        self.assertEqual([row["frame_index"] for row in rows], [3, 9])
        self.assertEqual(
            lance_dataset.filter,
            "episode_index = 7 AND frame_index IN (3, 9)",
        )
        lance_dataset.rows = [{"episode_index": 7, "frame_index": 3}]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            HyLanceMonoDataset._take_episode_frames(
                lance_dataset,
                episode_index=7,
                frame_indices=[3, 9],
                columns=["episode_index", "frame_index"],
            )

    def test_float32_timestamp_rounding_is_accepted_but_drift_is_rejected(self):
        rounded = np.asarray([133.6], dtype=np.float32).astype(np.float64)
        self.assertTrue(
            HyLanceMonoDataset._timestamps_match_frame_rate(
                rounded, np.asarray([4008]), 30.0
            )
        )
        self.assertFalse(
            HyLanceMonoDataset._timestamps_match_frame_rate(
                np.asarray([133.61]), np.asarray([4008]), 30.0
            )
        )


if __name__ == "__main__":
    unittest.main()
