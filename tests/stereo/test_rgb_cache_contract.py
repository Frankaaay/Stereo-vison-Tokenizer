import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "data" / "build_stereo_rgb_cache.py"
SPEC = importlib.util.spec_from_file_location("build_stereo_rgb_cache", SCRIPT)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class StereoRGBCacheContractTest(unittest.TestCase):
    def _record(self):
        selections = {
            f"{view}/{eye}": {"source_frame_index": frame_index}
            for view in CACHE.VIEWS
            for eye in CACHE.EYES
            for frame_index in (0,)
        }
        return {
            "sample_id": "episode:0000",
            "episode_id": "episode",
            "mcap_path": "/data/example.mcap",
            "gt_relative_path": "gt/episode/0000.npz",
            "frames": [
                {"selections": selections}
                for _ in range(4)
            ],
            "preprocessing": {
                "source_size_hw": [480, 640],
                "resize_size_hw": [192, 256],
                "output_size_hw": [256, 256],
                "padding_ltrb": [0, 32, 0, 32],
                "scale_xy": [0.4, 0.4],
            },
        }

    def test_v3_is_independent_and_references_valid_rgb_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_v2 = root / "pilot_manifest.jsonl"
            manifest_v2.write_text(
                json.dumps(self._record()) + "\n", encoding="utf-8"
            )
            output_root = root / "rgb_cache"
            cache_path = output_root / "rgb" / "episode" / "0000.npz"
            CACHE.write_rgb_cache(
                cache_path,
                np.zeros(CACHE.RGB_SHAPE, dtype=np.uint8),
            )
            manifest_v3 = root / "pilot_manifest_v3.jsonl"
            result = CACHE.finalize_manifest(
                manifest_v2, output_root, manifest_v3
            )

            record_v3 = json.loads(manifest_v3.read_text(encoding="utf-8"))
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(record_v3["manifest_version"], 3)
            self.assertEqual(record_v3["rgb_relative_path"], "rgb/episode/0000.npz")
            self.assertNotIn("rgb_relative_path", self._record())

    def test_existing_different_v3_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_v2 = root / "pilot_manifest.jsonl"
            manifest_v2.write_text(
                json.dumps(self._record()) + "\n", encoding="utf-8"
            )
            output_root = root / "rgb_cache"
            CACHE.write_rgb_cache(
                output_root / "rgb" / "episode" / "0000.npz",
                np.zeros(CACHE.RGB_SHAPE, dtype=np.uint8),
            )
            manifest_v3 = root / "pilot_manifest_v3.jsonl"
            manifest_v3.write_text("different\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                CACHE.finalize_manifest(manifest_v2, output_root, manifest_v3)


if __name__ == "__main__":
    unittest.main()
