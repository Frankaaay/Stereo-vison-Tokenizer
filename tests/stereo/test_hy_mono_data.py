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


if __name__ == "__main__":
    unittest.main()
