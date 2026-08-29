import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class StereoEntrypointSourceTest(unittest.TestCase):
    def test_training_entry_has_no_legacy_inflation_and_supports_explicit_resume(self):
        source = (ROOT / "train_stereo_vae.py").read_text(encoding="utf-8")
        self.assertNotIn("inflate_gen", source)
        self.assertNotIn("inflate_dis", source)
        self.assertNotIn("os.listdir", source)
        self.assertIn("StereoVAE(args)", source)
        self.assertIn("StereoDataModule(args)", source)
        self.assertIn("limit_val_batches=1.0", source)
        self.assertIn("check_val_every_n_epoch = None", source)
        self.assertIn("check_val_every_n_epoch=check_val_every_n_epoch", source)
        self.assertIn("max_steps=-1 if args.gan_enabled else args.max_steps", source)
        self.assertIn("max_epochs=-1", source)
        self.assertIn("--resume_from_checkpoint", source)
        self.assertIn("ckpt_path=args.resume_from_checkpoint", source)

    def test_evaluation_is_deterministic_and_strict(self):
        source = (ROOT / "eval_stereo_vae.py").read_text(encoding="utf-8")
        self.assertIn("sample_posterior=False", source)
        self.assertIn("strict=True", source)
        self.assertIn("_checkpoint_model_args(checkpoint", source)
        self.assertIn("StereoVAE(checkpoint_args)", source)
        self.assertIn("--stereo_vae_ckpt", source)
        self.assertIn("_validate_checkpoint_semantics", source)
        self.assertIn("--eval_temporal_mode", source)
        self.assertNotIn("--eval_single_frame_index", source)
        self.assertIn("args.single_frame_source_index", source)
        self.assertNotIn(".codebook", source)
        self.assertIn("relative_log_l1", source)
        self.assertIn("relative_log_rmse", source)
        self.assertIn("relative_target_from_foundation_stereo(", source)
        self.assertNotIn("metric_depth", source)
        self.assertIn('choices=["train", "val", "test"]', source)
        self.assertIn('choices=["single_frame", "four_frame", "both"]', source)
        self.assertIn('choices=["mono", "stereo", "both"]', source)
        self.assertIn("from stereo_tokenizer.mode_sampling import MODE_IDS", source)
        self.assertIn("FoundationStereoOnlineTeacher", source)
        self.assertIn("DepthAnything3OnlineTeacher", source)
        self.assertIn("HyLanceMonoDataset", source)
        self.assertIn("args.hy_manifest", source)
        self.assertIn("relative_target_from_da3(", source)
        self.assertIn("_exact_mono_rank_indices", source)
        self.assertIn('choices=("las2_h", "pytorch", "tensorrt")', source)
        self.assertNotIn("stereo_data_backend", source)
        self.assertNotIn("stereo_train_manifest", source)
        self.assertNotIn("stereo_val_manifest", source)
        self.assertIn("_exact_lerobot_rank_indices", source)
        self.assertIn("dist.all_reduce", source)
        self.assertIn("metrics[\"sample_count\"] != len(dataset)", source)
        self.assertIn("save_case_visualization", source)
        self.assertIn("save_depth_case_visualization", source)
        self.assertIn('depth_filename = f"depth-case-{slot:02d}.png"', source)
        self.assertIn('"depth_file": f"{eye_mode}/{depth_filename}"', source)
        self.assertIn("def evaluate_eye_mode(", source)
        main_source = source[source.index("def main():") :]
        self.assertLess(
            main_source.index("preflight_teacher_assets(args, eye_modes)"),
            main_source.index("initialize_distributed(args)"),
        )
        self.assertLess(
            main_source.index("build_eval_dataset(args, eye_mode)"),
            main_source.index("initialize_distributed(args)"),
        )
        self.assertNotIn(
            "visualizations require --eval_temporal_mode=both", source
        )
        self.assertIn("fixed_episode_subset_indices", source)
        self.assertIn("fixed_eval_case_indices", source)

    def test_recipe_requires_unfrozen_experiment_parameters(self):
        source = (
            ROOT / "scripts" / "stereo" / "train_stereo_vae.sh"
        ).read_text(encoding="utf-8")
        for name in (
            "PER_DEVICE_BATCH_SIZE",
            "GRAD_ACCUMULATES",
            "RGB_WEIGHT",
            "RELATIVE_DEPTH_WEIGHT",
            "RELATIVE_GRADIENT_WEIGHT",
            "KL_WEIGHT",
            "KL_WARMUP_STEPS",
            "SINGLE_FRAME_SOURCE_INDEX",
        ):
            self.assertIn(f"${{{name}:?", source)
        self.assertIn('TRAIN_LAUNCHER=(python3)', source)
        self.assertIn('"${TRAIN_LAUNCHER[@]}" train_stereo_vae.py', source)
        self.assertIn("--latent_channels 48", source)
        self.assertIn(
            '--single_frame_source_index "${SINGLE_FRAME_SOURCE_INDEX}"',
            source,
        )
        self.assertIn("${LAS2_H_SOURCE_SHA:?", source)
        self.assertIn('--las2_h_source_sha "${LAS2_H_SOURCE_SHA}"', source)
        self.assertNotIn("single_frame_loss_weight", source)
        self.assertIn('GAN_ENABLED="${GAN_ENABLED:-0}"', source)
        self.assertIn("GAN_ARGS+=(--gan_enabled)", source)
        self.assertNotIn("--use_vae", source)
        self.assertNotIn("--fp16", source)

    def test_profile_regions_are_opt_in_and_cover_requested_components(self):
        helper = (ROOT / "stereo_tokenizer" / "profiling.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_ENABLED = False", helper)
        model = (ROOT / "stereo_tokenizer" / "model.py").read_text(
            encoding="utf-8"
        )
        data = (ROOT / "stereo_tokenizer" / "data.py").read_text(
            encoding="utf-8"
        )
        losses = (
            ROOT / "stereo_tokenizer" / "modules" / "stereo_losses.py"
        ).read_text(encoding="utf-8")
        for region in (
            "stereo/encoder/spatial_transformer",
            "stereo/encoder/stereo_fusion",
            "stereo/decoder/spatial_transformer",
            "stereo/loss/lpips_vgg",
            "stereo/update/backward",
            "stereo/update/gradient_clipping",
            "stereo/update/adam_step",
            "stereo/logging/train_metrics",
            "stereo/transfer/cpu_to_gpu",
        ):
            self.assertIn(region, model)

        self.assertIn("stereo/data/collate", data)
        self.assertNotIn("rgb_npz_read_decompress", data)
        self.assertNotIn("gt_npz_read_decompress", data)
        for region in (
            "stereo/loss/rgb",
            "stereo/loss/relative_depth",
            "stereo/loss/relative_gradient",
            "stereo/loss/kl",
        ):
            self.assertIn(region, losses)

    def test_peg_backends_are_explicit_and_training_uses_t1_conv2d(self):
        attention = (
            ROOT / "stereo_tokenizer" / "modules" / "attention.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'self._backend = "conv3d_contiguous"', attention
        )
        self.assertIn('"conv2d_t1_slice"', attention)
        launcher = (
            ROOT / "scripts" / "stereo" / "train_stereo_vae.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--peg_backend conv2d_t1_slice", launcher)
        self.assertIn("--pin_memory 1", launcher)
        self.assertIn("--persistent_workers 1", launcher)
        self.assertIn('--prefetch_factor "${PREFETCH_FACTOR:-2}"', launcher)
        self.assertIn(
            '--lerobot_video_cache_capacity "${LEROBOT_VIDEO_CACHE_CAPACITY:-12}"',
            launcher,
        )
        self.assertIn(
            '--train_epoch_repeats "${TRAIN_EPOCH_REPEATS:-1}"', launcher
        )
        self.assertIn(
            '--checkpoint_every_n_steps "${CHECKPOINT_EVERY_N_STEPS:-500}"',
            launcher,
        )
        self.assertIn("DISABLE_MEDIA_LOGGING:-0", launcher)
        train = (ROOT / "train_stereo_vae.py").read_text(encoding="utf-8")
        self.assertIn("peak_memory_bytes_by_rank", train)
        self.assertIn('"mode_id": pl_module.last_mode_id', train)
        self.assertIn("peak_memory_bytes_by_rank_and_mode", train)
        self.assertIn("DDPStrategy(", train)
        self.assertIn("static_graph=False", train)
        self.assertIn("find_unused_parameters=True", train)
        self.assertIn("--torch_profile_output_dir", train)
        self.assertIn("set_profiling_enabled(True)", train)
        self.assertIn("TrainingProfilerStepCallback", train)
        self.assertIn("ProfilerActivity.CPU", train)
        self.assertIn("ProfilerActivity.CUDA", train)
        self.assertIn('os.environ.get("LOCAL_RANK", "0")', train)
        self.assertIn("profile_memory=True", train)
        self.assertIn("record_shapes=True", train)
        self.assertNotIn('torch.__version__ >= "2.1.0"', attention)
        self.assertIn(
            "dropout_p=self.p_dropout if self.training else 0.0",
            attention,
        )
        self.assertIn("sdpa_mask = sdpa_mask.masked_fill(", attention)
        self.assertIn("TORCH_PROFILE_OUTPUT_DIR", launcher)
        callbacks = (
            ROOT / "stereo_tokenizer" / "modules" / "callbacks.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "self.check_frequency(pl_module.global_step, split)", callbacks
        )
        self.assertNotIn("self.check_frequency(batch_idx)", callbacks)
        model = (ROOT / "stereo_tokenizer" / "model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('on_epoch=not prefix.startswith("train/")', model)
        self.assertNotIn("--causal_in_temporal_transformer", launcher)

    def test_train_launcher_online_gt_is_explicit_and_cache_defaults_off(self) -> None:
        launcher = (ROOT / "scripts/stereo/train_stereo_vae.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("STEREO_DATA_BACKEND", launcher)
        self.assertIn(
            '--lerobot_episode_manifest "${LEROBOT_EPISODE_MANIFEST}"',
            launcher,
        )
        self.assertIn("--online_gt_enabled 1", launcher)
        self.assertIn(
            'ONLINE_GT_CACHE_ENABLED="${ONLINE_GT_CACHE_ENABLED:-0}"',
            launcher,
        )
        self.assertIn("--foundation_stereo_valid_iters", launcher)
        self.assertIn("--lerobot_rectification_audit_sha256", launcher)
        train = (ROOT / "train_stereo_vae.py").read_text(encoding="utf-8")
        self.assertIn("use_distributed_sampler=False", train)
        self.assertNotIn("max_time=", train)

    def test_train_launcher_has_fail_closed_optional_ib_mode(self) -> None:
        launcher = (ROOT / "scripts/stereo/train_stereo_vae.sh").read_text(
            encoding="utf-8"
        )
        train = (ROOT / "train_stereo_vae.py").read_text(encoding="utf-8")
        probe = (ROOT / "scripts/stereo/check_ib_collective.py").read_text(
            encoding="utf-8"
        )
        for token in (
            'DISTRIBUTED_MODE="${DISTRIBUTED_MODE:-single}"',
            'NUM_NODES="${NUM_NODES:-1}"',
            "WORLD_SIZE=$((NUM_NODES * GPU_COUNT))",
            "--nproc_per_node",
            "--node_rank",
            "--master_addr",
            "--master_port",
            '--num_nodes "${NUM_NODES}"',
            '--distributed_mode "${DISTRIBUTED_MODE}"',
            "NCCL_IB_DISABLE=0",
            "NCCL_SOCKET_IFNAME",
            "NCCL_IB_HCA",
            'grep -q "NET/IB"',
        ):
            self.assertIn(token, launcher)
        self.assertIn('choices=("single", "ib")', train)
        self.assertIn("validate_distributed_runtime_args(args)", train)
        self.assertIn("_validate_four_mode_batch_contract(args)", train)
        self.assertIn('backend="nccl"', probe)
        self.assertIn("dist.all_reduce", probe)
        self.assertIn("dist.all_gather_object", probe)

    def test_three_source_training_wires_manifests_da3_and_checkpoint(self) -> None:
        launcher = (ROOT / "scripts/stereo/train_stereo_vae.sh").read_text(
            encoding="utf-8"
        )
        train = (ROOT / "train_stereo_vae.py").read_text(encoding="utf-8")
        model = (ROOT / "stereo_tokenizer/model.py").read_text(encoding="utf-8")
        online_gt = (ROOT / "stereo_tokenizer/online_gt.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "FOUR_MODE_MIXED_TRAINING",
            "HY_MANIFEST",
            "LIBERO_MANIFEST",
            "UMI_MANIFEST",
            "UMI_DATASET_ROOT",
            "UMI_RECTIFICATION_AUDIT_SHA256",
            "MODE_UPDATES_PER_EPOCH",
            "DA3_CHECKPOINT_SHA256",
            "NODE_MANIFEST_CONTRACTS",
            "MODE_UPDATE_WEIGHTS",
            "MODE_BATCH_SIZES",
            "MODE_GRAD_ACCUMULATES",
            "MONO_DATASET_WEIGHTS",
        ):
            self.assertIn(token, launcher)
        self.assertNotIn("UMI_ROOT_ALIASES", launcher)
        self.assertNotIn("UMI_EPISODE_CACHE_CAPACITY", launcher)
        self.assertNotIn("four-mode training is frozen to per-device BS24", launcher)
        self.assertIn('"val/mixed/total_loss"', train)
        self.assertIn("OnlineDepthAnything3GTCallback", train)
        self.assertIn("mode_occurrences_before", train)
        self.assertIn("mode_for_update", model)
        self.assertNotIn("teacher_rgb_raw", model)
        self.assertIn("da3_images", online_gt)
        self.assertIn("DepthAnything3OnlineTeacher", online_gt)
        self.assertIn("finite_positive_non_padding", launcher)

    def test_tensorrt_backend_is_explicit_and_frozen_to_32_iterations(self):
        launcher = (ROOT / "scripts/stereo/train_stereo_vae.sh").read_text(
            encoding="utf-8"
        )
        train = (ROOT / "train_stereo_vae.py").read_text(encoding="utf-8")
        online_gt = (ROOT / "stereo_tokenizer/online_gt.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "FOUNDATION_STEREO_BACKEND",
            "FOUNDATION_STEREO_ENGINE_SHA256",
            "FOUNDATION_STEREO_ENGINE_MANIFEST_SHA256",
            'FOUNDATION_STEREO_VALID_ITERS}" != "32"',
        ):
            self.assertIn(token, launcher)
        for argument in (
            "--foundation_stereo_backend",
            "--foundation_stereo_engine",
            "--foundation_stereo_engine_sha256",
            "--foundation_stereo_engine_manifest",
            "--foundation_stereo_engine_manifest_sha256",
        ):
            self.assertIn(argument, train)
        self.assertIn("execute_async_v3", online_gt)
        self.assertIn("torch.cuda.current_stream", online_gt)
        self.assertIn("set_tensor_address", online_gt)

    def test_tensorrt_comparison_does_not_change_iteration_ablation(self):
        comparison = (
            ROOT / "scripts/stereo/compare_foundation_backends.py"
        ).read_text(encoding="utf-8")
        original = (
            ROOT / "scripts/stereo/compare_online_foundation_teacher.py"
        ).read_text(encoding="utf-8")
        self.assertIn('choices=("equivalence", "tensorrt_benchmark")', comparison)
        self.assertIn("equivalence comparison is limited to 32-64 samples", comparison)
        self.assertIn("backend pilot is frozen to the approved 408 selection", comparison)
        self.assertIn('valid_iters=32', comparison)
        self.assertIn("args.valid_iters != [32, 16, 12]", original)

    def test_tensorrt_manifest_writer_is_fail_closed(self):
        source = (
            ROOT / "scripts/stereo/write_foundation_tensorrt_manifest.py"
        ).read_text(encoding="utf-8")
        self.assertIn("refusing to overwrite", source)
        self.assertIn("expected_profile", source)
        self.assertIn('"valid_iters": 32', source)
        self.assertIn('"precision": "fp16"', source)
        self.assertIn('"xformers_disabled": True', source)

    def test_online_teacher_comparison_freezes_requested_order(self) -> None:
        source = (
            ROOT / "scripts/stereo/compare_online_foundation_teacher.py"
        ).read_text(encoding="utf-8")
        self.assertIn("args.valid_iters != [32, 16, 12]", source)
        self.assertIn("valid_mask_iou_with_32", source)
        self.assertIn("automatic_numeric_pass", source)
        self.assertIn("if not args.allow_pending_visual_review:", source)
        self.assertIn('"visual_sample_ids": sorted(visual_sample_ids)', source)

if __name__ == "__main__":
    unittest.main()
