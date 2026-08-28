import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts.data.build_pretrain_manifest import build_umi


class UMIManifestTest(unittest.TestCase):
    def test_missing_calibration_is_rejected_before_mcap_scan(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode_root = root / "A_001" / "episode-1"
            episode_root.mkdir(parents=True)
            (episode_root / "episode.mcap").write_bytes(b"")
            sidecar = {
                "task.review.status": "Accepted",
                "frames": {"status": "done"},
            }
            (episode_root / "episode-1.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )

            reader_module = types.ModuleType("mcap.reader")
            reader_module.make_reader = mock.Mock(
                side_effect=AssertionError("invalid episode must not open MCAP")
            )
            mcap_module = types.ModuleType("mcap")
            mcap_module.reader = reader_module
            with mock.patch.dict(
                sys.modules,
                {"mcap": mcap_module, "mcap.reader": reader_module},
            ):
                self.assertEqual(list(build_umi({"umi": root})), [])
            reader_module.make_reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
