import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import eval_stereo_vae as evaluation
from stereo_tokenizer.mode_sampling import MODE_IDS


class FourModeEvaluationTest(unittest.TestCase):
    def test_fixed_mono_cases_use_distinct_episode_spans_without_sample_ids(self):
        records = [
            {
                "root_alias": "hy_primary",
                "table_name": "table_000",
                "episode_id": f"table_000:{episode}",
            }
            for episode in range(3)
        ]
        spans = [
            SimpleNamespace(
                record_index=index,
                variant="cam_high",
                first_sample=index * 10,
                sample_count=10,
            )
            for index in range(3)
        ]
        dataset = SimpleNamespace(records=records, spans=spans)

        indices = evaluation.fixed_eval_case_indices(
            dataset, count=2, seed=1234, eye_mode="mono"
        )

        self.assertEqual(
            indices,
            evaluation.fixed_eval_case_indices(
                dataset, count=2, seed=1234, eye_mode="mono"
            ),
        )
        self.assertEqual(len(indices), 2)
        self.assertEqual(len({index // 10 for index in indices}), 2)
        self.assertTrue(all(0 <= index < 30 for index in indices))

    def test_mono_provenance_uses_hy_manifest_and_root_aliases(self):
        class Dataset:
            manifest_path = Path("hy-manifest.jsonl")
            root_aliases = {
                "hy_rest": Path("hy-rest"),
                "hy_primary": Path("hy-primary"),
            }

            def __len__(self):
                return 17

        provenance = evaluation.dataset_provenance(
            SimpleNamespace(), "mono", Dataset()
        )

        self.assertEqual(provenance["manifest"], "hy-manifest.jsonl")
        self.assertEqual(
            provenance["root_aliases"],
            {
                "hy_primary": str(Path("hy-primary").resolve()),
                "hy_rest": str(Path("hy-rest").resolve()),
            },
        )
        self.assertEqual(provenance["sample_count"], 17)
        self.assertNotIn("cache_root", provenance)

    def test_libero_mono_dataset_uses_its_manifest_and_decode_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                mono_dataset="libero",
                eval_split="test",
                single_frame_source_index=0,
                libero_manifest="libero.jsonl",
                libero_root_aliases=f'{{"libero": "{directory}"}}',
                lerobot_video_cache_capacity=7,
                lerobot_maximum_timestamp_error_s=0.05,
            )
            with mock.patch.object(
                evaluation, "LiberoMonoDataset", return_value="libero-dataset"
            ) as constructor:
                dataset = evaluation.build_eval_dataset(args, "mono")

        self.assertEqual(dataset, "libero-dataset")
        constructor.assert_called_once_with(
            "libero.jsonl",
            {"libero": str(Path(directory).resolve())},
            video_cache_capacity=7,
            maximum_timestamp_error_s=0.05,
            split="test",
            single_frame_source_index=0,
        )

    def test_fixed_libero_cases_use_suite_identity(self):
        dataset = SimpleNamespace(
            records=[
                {
                    "root_alias": "libero",
                    "suite": "libero_10",
                    "episode_id": f"episode-{index}",
                }
                for index in range(2)
            ],
            spans=[
                SimpleNamespace(
                    record_index=index,
                    variant="observation.images.image",
                    first_sample=index * 5,
                    sample_count=5,
                )
                for index in range(2)
            ],
        )

        indices = evaluation.fixed_eval_case_indices(
            dataset, count=2, seed=1234, eye_mode="mono"
        )

        self.assertEqual(len(indices), 2)
        self.assertEqual(len({index // 5 for index in indices}), 2)

    def test_all_nine_argument_combinations_expand_from_mode_ids(self):
        expected = {
            (eye, temporal): tuple(
                mode_id
                for mode_id in MODE_IDS
                if (eye == "both" or mode_id.startswith(f"{eye}/"))
                and (temporal == "both" or mode_id.endswith(f"/{temporal}"))
            )
            for eye in ("mono", "stereo", "both")
            for temporal in ("single_frame", "four_frame", "both")
        }
        for combination, mode_ids in expected.items():
            with self.subTest(combination=combination):
                args = Namespace(
                    eval_eye_mode=combination[0],
                    eval_temporal_mode=combination[1],
                )
                self.assertEqual(evaluation.requested_mode_ids(args), mode_ids)
        self.assertEqual(
            evaluation.requested_mode_ids(
                Namespace(eval_eye_mode="both", eval_temporal_mode="both")
            ),
            MODE_IDS,
        )

    def test_multiple_single_frame_sources_expand_without_changing_four_frame(self):
        args = Namespace(
            eval_eye_mode="both",
            eval_temporal_mode="both",
            single_frame_source_index=0,
            single_frame_source_indices=[0, 1, 2, 3],
        )

        self.assertEqual(
            evaluation.requested_mode_ids(args),
            (
                "mono/single_frame/source_0",
                "mono/single_frame/source_1",
                "mono/single_frame/source_2",
                "mono/single_frame/source_3",
                "mono/four_frame",
                "stereo/single_frame/source_0",
                "stereo/single_frame/source_1",
                "stereo/single_frame/source_2",
                "stereo/single_frame/source_3",
                "stereo/four_frame",
            ),
        )

    @staticmethod
    def _depth_batch():
        shape = (1, 1, 1, 4, 8, 8)
        return {
            "sample_id": ["sample-0"],
            "teacher_kind": ["da3"],
            "eye_mode": ["mono"],
            "temporal_mode": ["four_frame"],
            "mode_id": ["mono/four_frame"],
            "video": torch.zeros(1, 1, 1, 3, 4, 8, 8),
            "da3_relative_depth": torch.ones(shape),
            "valid_mask": torch.ones(shape, dtype=torch.bool),
        }

    @staticmethod
    def _output(frames):
        return SimpleNamespace(
            raw_relative_log_depth=torch.zeros(1, 1, 1, frames, 8, 8),
            rgb=torch.zeros(1, 1, 3, frames, 8, 8),
        )

    def test_depth_visualization_accepts_each_temporal_subset(self):
        batch = self._depth_batch()
        cases = (
            {"single_frame": self._output(1)},
            {"four_frame": self._output(4)},
            {
                "single_frame": self._output(1),
                "four_frame": self._output(4),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, outputs in enumerate(cases):
                path = Path(directory) / f"depth-{index}.png"
                evaluation.save_depth_case_visualization(
                    path, batch, outputs, 0, 1e-6, ("cam_high",)
                )
                self.assertTrue(path.is_file())

    def test_two_eye_sessions_call_teacher_once_and_all_four_model_modes(self):
        args = Namespace(
            batch_size=1,
            num_workers=0,
            pin_memory=0,
            persistent_workers=0,
            max_batches=None,
            bf16=False,
            relative_depth_epsilon=1e-6,
            single_frame_source_index=0,
            num_visualizations=0,
            eval_split="val",
            lerobot_rectification_audit_sha256="a" * 64,
            foundation_stereo_backend="pytorch",
            foundation_stereo_checkpoint_sha256="b" * 64,
            las2_h_source_sha=None,
            las2_h_checkpoint_sha256=None,
            da3_source_sha="c" * 40,
            da3_checkpoint_sha256="d" * 64,
            da3_process_res=504,
            da3_process_res_method="upper_bound_resize",
            da3_confidence_mask_mode="finite_positive_non_padding",
            mono_dataset="hy",
        )
        calls = {"teachers": [], "models": []}
        batch = {"video": torch.zeros(1, 1, 1, 3, 4, 2, 2)}

        class Dataset:
            manifest_path = Path("manifest.jsonl")
            root_aliases = {"hy_primary": Path("hy-primary")}
            dataset_root = Path("dataset")

            def __len__(self):
                return 1

        def attach(_args, eye_mode, _teacher, _batch):
            calls["teachers"].append(eye_mode)

        def model(_video, *, eye_mode, temporal_mode, sample_posterior):
            calls["models"].append(f"{eye_mode}/{temporal_mode}")
            return object()

        with mock.patch.object(evaluation, "exact_eval_loader", return_value=[batch]), \
             mock.patch.object(evaluation, "attach_online_targets", side_effect=attach), \
             mock.patch.object(evaluation, "empty_accumulator", return_value={}), \
             mock.patch.object(evaluation, "update_metrics"), \
             mock.patch.object(evaluation, "reduce_accumulator"), \
             mock.patch.object(
                 evaluation,
                 "finalize_metrics",
                 return_value={"sample_count": 1},
             ):
            metrics = {}
            for eye_mode in ("mono", "stereo"):
                eye_metrics, _ = evaluation.evaluate_eye_mode(
                    args,
                    eye_mode,
                    ("single_frame", "four_frame"),
                    Dataset(),
                    object(),
                    model,
                    torch.device("cpu"),
                    0,
                    1,
                )
                metrics.update(eye_metrics)
        self.assertEqual(calls["teachers"], ["mono", "stereo"])
        self.assertEqual(tuple(calls["models"]), MODE_IDS)
        self.assertEqual(tuple(metrics), MODE_IDS)
        self.assertEqual(set(metrics), set(MODE_IDS))

    def test_teacher_selection_matches_eye_mode(self):
        args = SimpleNamespace(
            da3_repo="repo",
            da3_source_sha="a" * 40,
            da3_checkpoint="checkpoint",
            da3_checkpoint_sha256="b" * 64,
            da3_process_res=504,
            da3_process_res_method="upper_bound_resize",
            foundation_stereo_backend="pytorch",
            foundation_stereo_valid_iters=32,
            foundation_stereo_pair_microbatch=1,
            foundation_stereo_repo="stereo-repo",
            foundation_stereo_checkpoint="stereo-checkpoint",
            foundation_stereo_checkpoint_sha256="c" * 64,
            foundation_stereo_engine=None,
            foundation_stereo_engine_sha256=None,
            foundation_stereo_engine_manifest=None,
            foundation_stereo_engine_manifest_sha256=None,
            las2_h_repo=None,
            las2_h_source_sha=None,
            las2_h_checkpoint=None,
            las2_h_checkpoint_sha256=None,
            las2_h_valid_iters=4,
            las2_h_max_disp=192,
        )
        with mock.patch.object(
            evaluation, "DepthAnything3OnlineTeacher", return_value="da3"
        ) as da3, mock.patch.object(
            evaluation, "FoundationStereoOnlineTeacher", return_value="foundation"
        ) as foundation:
            self.assertEqual(
                evaluation.build_online_teacher(args, "mono", "cpu"), "da3"
            )
            self.assertEqual(
                evaluation.build_online_teacher(args, "stereo", "cpu"),
                "foundation",
            )
        da3.assert_called_once()
        foundation.assert_called_once()

    def test_metrics_json_contract_contains_all_four_mode_keys(self):
        args = SimpleNamespace(
            stereo_vae_ckpt=Path("model.ckpt"),
            single_frame_source_index=0,
            bf16=True,
        )
        metrics = {mode_id: {"sample_count": 1} for mode_id in MODE_IDS}
        result = evaluation.build_evaluation_result(
            args,
            MODE_IDS,
            metrics,
            {"mono": {}, "stereo": {}},
            {"mono": {}, "stereo": {}},
            {"mono": [], "stereo": []},
            1,
        )
        self.assertEqual(result["requested_modes"], list(MODE_IDS))
        self.assertEqual(tuple(result["modes"]), MODE_IDS)
        self.assertEqual(set(result["datasets"]), {"mono", "stereo"})
        self.assertEqual(set(result["teachers"]), {"mono", "stereo"})

    def test_visualization_records_use_independent_eye_directories(self):
        args = SimpleNamespace(
            batch_size=1,
            num_workers=0,
            pin_memory=0,
            persistent_workers=0,
            max_batches=0,
            bf16=False,
            relative_depth_epsilon=1e-6,
            single_frame_source_index=0,
            num_visualizations=1,
            eval_split="val",
            lerobot_rectification_audit_sha256="a" * 64,
            foundation_stereo_backend="pytorch",
            foundation_stereo_checkpoint_sha256="b" * 64,
            las2_h_source_sha=None,
            las2_h_checkpoint_sha256=None,
            da3_source_sha="c" * 40,
            da3_checkpoint_sha256="d" * 64,
            da3_process_res=504,
            da3_process_res_method="upper_bound_resize",
            da3_confidence_mask_mode="finite_positive_non_padding",
            seed=1,
            mono_dataset="hy",
        )
        sample = {
            "video": torch.zeros(1, 1, 3, 4, 2, 2),
            "sample_id": "sample",
            "episode_id": "episode",
        }

        class Dataset:
            manifest_path = Path("manifest.jsonl")
            root_aliases = {"hy_primary": Path("hy-primary")}
            dataset_root = Path("dataset")

            def __len__(self):
                return 1

            def __getitem__(self, _index):
                return sample

        def touch(path, *_args, **_kwargs):
            Path(path).touch()

        with tempfile.TemporaryDirectory() as parent:
            args.visualization_dir = Path(parent) / "visualizations"
            args.visualization_dir.mkdir()
            with mock.patch.object(evaluation, "exact_eval_loader", return_value=[]), \
                 mock.patch.object(evaluation, "fixed_eval_case_indices", return_value=[0]), \
                 mock.patch.object(evaluation, "attach_online_targets"), \
                 mock.patch.object(evaluation, "save_case_visualization", side_effect=touch), \
                 mock.patch.object(evaluation, "save_depth_case_visualization", side_effect=touch), \
                 mock.patch.object(evaluation, "empty_accumulator", return_value={}), \
                 mock.patch.object(evaluation, "reduce_accumulator"), \
                 mock.patch.object(evaluation, "finalize_metrics", return_value={"sample_count": 1}):
                records = {}
                for eye_mode in ("mono", "stereo"):
                    _, eye_records = evaluation.evaluate_eye_mode(
                        args,
                        eye_mode,
                        ("single_frame",),
                        Dataset(),
                        object(),
                        lambda *_args, **_kwargs: self._output(1),
                        torch.device("cpu"),
                        0,
                        1,
                    )
                    records[eye_mode] = eye_records
            self.assertEqual(records["mono"][0]["file"], "mono/case-00.png")
            self.assertEqual(records["stereo"][0]["file"], "stereo/case-00.png")
            self.assertTrue((args.visualization_dir / "mono" / "cases.json").is_file())
            self.assertTrue((args.visualization_dir / "stereo" / "cases.json").is_file())

    def test_missing_teacher_assets_fail_in_preflight(self):
        args = SimpleNamespace(
            da3_repo="missing-repo",
            da3_source_sha="a" * 40,
            da3_checkpoint="missing-checkpoint",
            da3_checkpoint_sha256="b" * 64,
        )
        with self.assertRaises(FileNotFoundError):
            evaluation.preflight_teacher_assets(args, ("mono",))

    def test_four_mode_training_rejects_nonzero_source_index(self):
        from train_stereo_vae import validate_runtime_args

        args = SimpleNamespace(
            sequence_length=4,
            single_frame_source_index=1,
            resolution=256,
            four_mode_mixed_training=True,
        )
        with self.assertRaisesRegex(ValueError, "single_frame_source_index=0"):
            validate_runtime_args(args)


if __name__ == "__main__":
    unittest.main()
