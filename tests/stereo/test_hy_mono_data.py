import json
import tempfile
import unittest
from pathlib import Path

from stereo_tokenizer.data import _load_root_aliases
from stereo_tokenizer.pretrain_data import HY_SCHEMA, HyLanceMonoDataset


class HyCamHighManifestTest(unittest.TestCase):
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
                    "window_count": 2,
                    "source_contract_sha256": "a" * 64,
                }
            )
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_hy_manifest_expands_only_cam_high_windows(self):
        dataset = HyLanceMonoDataset(
            self._manifest(), {"hy_primary": self.root}, split="train"
        )
        self.assertEqual(dataset.camera_column, "observation_images_cam_high")
        self.assertEqual([span.variant for span in dataset.spans], ["cam_high"])
        self.assertEqual(len(dataset), 2)

    def test_root_alias_json_is_node_local(self):
        mapping = _load_root_aliases(
            json.dumps({"hy_primary": str(self.root)}), "--hy_root_aliases"
        )
        self.assertEqual(mapping, {"hy_primary": str(self.root.resolve())})


if __name__ == "__main__":
    unittest.main()
