import io
import unittest

from PIL import Image

from scripts.data.filter_hy_manifest_valid_cameras import (
    anchor_frame_indices,
    index_rows_by_identity,
    jpeg_error,
)


class HyCameraManifestFilterTest(unittest.TestCase):
    def test_anchor_frames_cover_first_middle_and_last_windows(self):
        self.assertEqual(anchor_frame_indices(1), (0, 3, 6, 9))
        self.assertEqual(
            anchor_frame_indices(4),
            (0, 3, 6, 9, 12, 15, 18, 21, 36, 39, 42, 45),
        )

    def test_jpeg_validation_rejects_placeholder_and_accepts_expected_image(self):
        self.assertEqual(jpeg_error(b"\x00"), "payload_length=1")
        stream = io.BytesIO()
        Image.new("RGB", (424, 240)).save(stream, format="JPEG")
        self.assertIsNone(jpeg_error(stream.getvalue()))

    def test_lance_take_rows_are_indexed_without_assuming_return_order(self):
        rows = [
            {"episode_index": 2, "frame_index": 3},
            {"episode_index": 1, "frame_index": 0},
        ]
        indexed = index_rows_by_identity(rows)
        self.assertIs(indexed[(1, 0)], rows[1])
        self.assertIs(indexed[(2, 3)], rows[0])


if __name__ == "__main__":
    unittest.main()
