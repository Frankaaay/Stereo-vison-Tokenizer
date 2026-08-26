import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stereo_tokenizer.data import HyMonoSmokeDataset


class HyMonoSmokeDatasetTest(unittest.TestCase):
    def _dataset(self, root: Path):
        rgb_root = root / "rgb"
        rgb_root.mkdir()
        cache_path = rgb_root / "sample.npz"
        frame_index = np.asarray([10, 13, 16, 19], np.int64)
        timestamp_s = frame_index.astype(np.float64) / 30.0
        rgb = np.zeros((4, 3, 240, 424), np.uint8)
        for frame in range(4):
            rgb[frame] = frame * 30
        with cache_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                rgb=rgb,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                metadata_json=np.asarray(
                    json.dumps(
                        {
                            "sample_id": "hy/table_000/episode_0/cam_high/frame_10",
                            "source_contract_sha256": "a" * 64,
                        }
                    )
                ),
            )
        cache_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        records = []
        for index in range(48):
            records.append(
                {
                    "schema": "hy-mono-cam-high-smoke-v1",
                    "sample_id": f"hy/table_000/episode_{index}/cam_high/frame_10",
                    "episode_id": f"episode_{index}",
                    "start_frame": 10,
                    "frame_indices": frame_index.tolist(),
                    "timestamps_s": timestamp_s.tolist(),
                    "source_hw": [240, 424],
                    "rgb_relative_path": "rgb/sample.npz",
                    "rgb_sha256": cache_sha,
                    "source_contract_sha256": "a" * 64,
                    "table_inventory_sha256": "b" * 64,
                }
            )
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return HyMonoSmokeDataset(manifest, root)

    def test_single_and_four_modes_keep_true_mono_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            single = dataset.get_mode_item(0, "single_frame")
            four = dataset.get_mode_item(0, "four_frame")

        self.assertEqual(single["video"].shape, (1, 1, 3, 1, 256, 256))
        self.assertEqual(four["video"].shape, (1, 1, 3, 4, 256, 256))
        self.assertEqual(single["da3_images"].shape, (1, 3, 280, 504))
        self.assertEqual(four["da3_images"].shape, (4, 3, 280, 504))
        self.assertEqual(single["non_padding_mask"].shape, (1, 1, 1, 256, 256))
        self.assertEqual(four["non_padding_mask"].shape, (1, 1, 4, 256, 256))
        geometry = single["geometry_mapping"]
        self.assertEqual(geometry["student_padding_ltrb"].tolist(), [0, 55, 0, 56])
        self.assertEqual(geometry["student_resized_hw"].tolist(), [145, 256])
        self.assertEqual(geometry["da3_processed_hw"].tolist(), [280, 504])
        self.assertTrue(single["non_padding_mask"][..., 55:200, :].all())
        self.assertFalse(single["non_padding_mask"][..., :55, :].any())
        self.assertEqual(single["mode_id"], "mono/single_frame")
        self.assertEqual(four["mode_id"], "mono/four_frame")
        self.assertTrue(torch.equal(single["frame_index"], torch.tensor([10])))


if __name__ == "__main__":
    unittest.main()
