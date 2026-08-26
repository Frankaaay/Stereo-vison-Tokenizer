import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import default_collate

from stereo_tokenizer.data import HyMonoSmokeDataset
from stereo_tokenizer.geometry import GeometryMapping
from stereo_tokenizer.online_gt import OnlineDepthAnything3GTCallback


class GeometryMappingTest(unittest.TestCase):
    def test_rectified_480x640_student_geometry(self):
        mapping = GeometryMapping.create((480, 640))
        self.assertEqual(mapping.student_resized_hw, (192, 256))
        self.assertEqual(mapping.student_padding_ltrb, (0, 32, 0, 32))

    def test_hy_240x424_student_geometry(self):
        mapping = GeometryMapping.create((240, 424))
        self.assertEqual(mapping.student_resized_hw, (145, 256))
        self.assertEqual(mapping.student_padding_ltrb, (0, 55, 0, 56))

    def test_da3_input_has_no_student_padding(self):
        mapping = GeometryMapping.create((240, 424))
        image = torch.full((1, 3, 240, 424), 255, dtype=torch.uint8)
        processed = mapping.da3_preprocess(image)
        self.assertEqual(processed.shape, (1, 3, 280, 504))
        self.assertTrue(torch.all(processed > 0))

    def test_da3_one_and_four_frame_outputs_map_to_student(self):
        for frames in (1, 4):
            mapping = GeometryMapping.create((480, 640))
            native = torch.ones(2, frames, *mapping.da3_processed_hw)
            mapped = mapping.map_da3_output_to_student(native).unsqueeze(1)
            self.assertEqual(mapped.shape, (2, 1, 1, frames, 256, 256))

    def test_padding_is_excluded_from_depth_validity(self):
        mapping = GeometryMapping.create((480, 640))
        native = torch.ones(1, 1, *mapping.da3_processed_hw)
        depth = mapping.map_da3_output_to_student(native).unsqueeze(1)
        _, non_padding = mapping.student_letterbox(
            torch.ones(1, 3, 480, 640, dtype=torch.uint8)
        )
        non_padding = non_padding.permute(1, 0, 2, 3).unsqueeze(0)
        valid = torch.isfinite(depth) & (depth > 0) & non_padding
        self.assertFalse(valid[..., :32, :].any())
        self.assertTrue(valid[..., 32:224, :].all())
        self.assertFalse(valid[..., 224:, :].any())

    def test_coordinate_ramp_aligns_da3_and_student_pixels(self):
        mapping = GeometryMapping.create((480, 640))
        ramp = torch.arange(640, dtype=torch.float32).div(639).mul(255).round()
        image = ramp.to(torch.uint8).view(1, 1, 1, 640).expand(1, 3, 480, 640)
        student, mask = mapping.student_letterbox(image)
        da3 = mapping.da3_preprocess(image)
        da3_red = da3[:, 0].mul(0.229).add(0.485)
        mapped = mapping.map_da3_output_to_student(da3_red.unsqueeze(0))[0, 0]
        expected = student[:, 0].div(255.0)
        content = mask[:, 0]
        self.assertLess(
            torch.abs(mapped.masked_select(content) - expected.masked_select(content))
            .mean()
            .item(),
            0.005,
        )

    def test_student_and_da3_fork_from_same_rectified_source(self):
        mapping = GeometryMapping.create((480, 640), source_hw=(720, 1280))
        self.assertEqual(mapping.source_hw, (720, 1280))
        self.assertEqual(mapping.rectified_hw, (480, 640))
        rectified = torch.zeros(1, 3, 480, 640, dtype=torch.uint8)
        mapping.student_letterbox(rectified)
        mapping.da3_preprocess(rectified)
        with self.assertRaisesRegex(ValueError, "rectified RGB tensor disagrees"):
            mapping.da3_preprocess(torch.zeros(1, 3, 720, 1280, dtype=torch.uint8))

    def test_tensor_metadata_disagreement_fails_closed(self):
        mapping = GeometryMapping.create((480, 640))
        collated = default_collate(
            [mapping.to_collatable_metadata(), mapping.to_collatable_metadata()]
        )
        restored = GeometryMapping.from_collated(collated, 2)
        self.assertEqual(restored, mapping)
        with self.assertRaisesRegex(ValueError, "DA3 output tensor disagrees"):
            mapping.map_da3_output_to_student(torch.ones(2, 1, 280, 504))
        collated["da3_processed_hw"][1, 0] += 14
        with self.assertRaisesRegex(ValueError, "must contain one"):
            GeometryMapping.from_collated(collated, 2)

    def test_da3_cache_geometry_and_provenance_mismatch_is_strict(self):
        callback = object.__new__(OnlineDepthAnything3GTCallback)
        callback.args = SimpleNamespace(
            da3_source_sha="a" * 40,
            da3_checkpoint_sha256="b" * 64,
            da3_process_res=504,
            da3_process_res_method="upper_bound_resize",
        )
        callback.cache_enabled = True
        callback.cache_namespace = "test"
        mapping = GeometryMapping.create((240, 424))
        with tempfile.TemporaryDirectory() as directory:
            callback.cache_root = Path(directory)
            path = callback._cache_path("sample", "single_frame")
            path.parent.mkdir(parents=True)
            metadata = callback._cache_metadata(
                "sample", "c" * 64, "single_frame", mapping
            )
            with path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    depth=np.ones((1, *mapping.da3_processed_hw), np.float32),
                    confidence=np.ones((1, *mapping.da3_processed_hw), np.float32),
                    metadata_json=np.asarray(json.dumps(metadata)),
                )
            self.assertIsNotNone(
                callback._read_cache(
                    "sample", "c" * 64, "single_frame", mapping
                )
            )
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                callback._read_cache(
                    "sample",
                    "d" * 64,
                    "single_frame",
                    mapping,
                )
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                callback._read_cache(
                    "sample",
                    "c" * 64,
                    "single_frame",
                    GeometryMapping.create((480, 640)),
                )

    def test_dataset_worker_has_no_da3_model_or_forward(self):
        data_source = inspect.getsource(HyMonoSmokeDataset)
        geometry_source = inspect.getsource(GeometryMapping.da3_preprocess)
        self.assertNotIn("DepthAnything3", data_source)
        self.assertNotIn("depth_anything_3", geometry_source)
        self.assertNotIn(".forward(", geometry_source)


if __name__ == "__main__":
    unittest.main()
