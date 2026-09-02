import io
import unittest

from PIL import Image

from scripts.data.filter_hy_manifest_valid_cameras import (
    anchor_frame_indices,
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


if __name__ == "__main__":
    unittest.main()
