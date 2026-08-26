import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import default_collate

from eval_stereo_vae import (
    _exact_mono_rank_indices,
    batch_for_temporal_mode,
    empty_accumulator,
    finalize_metrics,
    update_metrics,
)
from stereo_tokenizer.data import HyMonoDataset, HyMonoSmokeDataset
from stereo_tokenizer.geometry import GeometryMapping
from stereo_tokenizer.online_gt import (
    DepthAnything3OnlineTeacher,
    OnlineDepthAnything3GTCallback,
    attach_da3_student_targets,
)


class GeometryMappingTest(unittest.TestCase):
    def test_da3_teacher_passes_empty_export_feature_layers(self):
        calls = []

        class FakeModel:
            def __call__(self, image, *, export_feat_layers):
                calls.append(export_feat_layers)
                shape = (image.shape[0], image.shape[1], *image.shape[-2:])
                return SimpleNamespace(
                    depth=torch.ones(shape),
                    depth_conf=torch.ones(shape),
                )

        teacher = object.__new__(DepthAnything3OnlineTeacher)
        teacher.device = torch.device("cpu")
        teacher.model = FakeModel()
        depth, confidence = teacher.infer_processed(torch.zeros(1, 1, 3, 28, 28))
        self.assertEqual(calls, [[]])
        self.assertEqual(depth.shape, (1, 1, 28, 28))
        self.assertEqual(confidence.shape, (1, 1, 28, 28))

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
            single_frame_source_index=0,
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
            self.assertEqual(metadata["single_frame_source_index"], 0)
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
            base_args = SimpleNamespace(**vars(callback.args))
            base_args.online_gt_cache_enabled = 0
            base_args.online_gt_cache_root = None
            changed_args = SimpleNamespace(**vars(base_args))
            changed_args.single_frame_source_index = 2
            base_callback = OnlineDepthAnything3GTCallback(base_args)
            changed_callback = OnlineDepthAnything3GTCallback(changed_args)
            self.assertNotEqual(
                base_callback.cache_namespace,
                changed_callback.cache_namespace,
            )

    def test_dataset_worker_has_no_da3_model_or_forward(self):
        data_source = inspect.getsource(HyMonoDataset)
        geometry_source = inspect.getsource(GeometryMapping.da3_preprocess)
        self.assertNotIn("DepthAnything3", data_source)
        self.assertNotIn("depth_anything_3", geometry_source)
        self.assertNotIn(".forward(", geometry_source)

    def test_formal_mono_contract_maps_da3_and_computes_one_view_metrics(self):
        mapping = GeometryMapping.create((240, 424))
        raw = torch.zeros(4, 3, 240, 424, dtype=torch.uint8)
        student, non_padding = mapping.student_letterbox(raw)
        batch = {
            "video": student.div(255.0)
            .sub(0.5)
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0),
            "da3_images": mapping.da3_preprocess(raw).unsqueeze(0),
            "non_padding_mask": non_padding.permute(1, 0, 2, 3)
            .unsqueeze(0)
            .unsqueeze(0),
            "geometry_mapping": default_collate(
                [mapping.to_collatable_metadata()]
            ),
            "eye_mode": ["mono"],
            "temporal_mode": ["four_frame"],
            "teacher_kind": ["da3"],
        }
        native_shape = (1, 4, *mapping.da3_processed_hw)
        attach_da3_student_targets(
            batch,
            torch.ones(native_shape),
            torch.ones(native_shape),
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        self.assertEqual(batch["video"].shape, (1, 1, 1, 3, 4, 256, 256))
        self.assertEqual(
            batch["da3_relative_depth"].shape,
            (1, 1, 1, 4, 256, 256),
        )
        single = batch_for_temporal_mode(batch, "single_frame", 2)
        self.assertEqual(single["video"].shape, (1, 1, 1, 3, 1, 256, 256))
        self.assertEqual(single["da3_relative_depth"].shape[3], 1)
        self.assertEqual(single["mode_id"], ["mono/single_frame"])
        output = SimpleNamespace(
            rgb=batch["video"][:, :, 0].clone(),
            raw_relative_log_depth=torch.zeros_like(batch["da3_relative_depth"]),
        )
        accumulator = empty_accumulator(torch.device("cpu"), 1)
        update_metrics(accumulator, batch, output, 1e-6)
        metrics = finalize_metrics(accumulator, ("cam_high",))
        self.assertEqual(metrics["sample_count"], 1)
        self.assertEqual(metrics["rgb_l1"], 0.0)
        self.assertEqual(metrics["views"]["cam_high"]["relative_log_l1"], 0.0)

    def test_mono_ddp_indices_are_exact_and_non_overlapping(self):
        dataset = list(range(7))
        rank_indices = [
            _exact_mono_rank_indices(dataset, rank, 3) for rank in range(3)
        ]
        self.assertEqual(
            sorted(index for indices in rank_indices for index in indices),
            list(range(7)),
        )
        self.assertEqual(
            sum(len(indices) for indices in rank_indices),
            len({index for indices in rank_indices for index in indices}),
        )


if __name__ == "__main__":
    unittest.main()
