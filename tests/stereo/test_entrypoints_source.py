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
        self.assertIn("limit_val_batches=1.0 if has_validation else 0", source)
        self.assertIn("check_val_every_n_epoch = 1", source)
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
        self.assertIn("FoundationStereoOnlineTeacher", source)
        self.assertIn("_exact_lerobot_rank_indices", source)
        self.assertIn("dist.all_reduce", source)
        self.assertIn("metrics[\"sample_count\"] != expected", source)
        self.assertIn("save_case_visualization", source)
        self.assertIn("save_depth_case_visualization", source)
        self.assertIn('depth_filename = f"depth-case-{slot:02d}.png"', source)
        self.assertIn('"depth_file": depth_filename', source)
        self.assertIn("fixed_episode_subset_indices", source)

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
        self.assertIn("python3 train_stereo_vae.py", source)
        self.assertIn("--latent_channels 48", source)
        self.assertIn(
            '--single_frame_source_index "${SINGLE_FRAME_SOURCE_INDEX}"',
            source,
        )
        self.assertNotIn("single_frame_loss_weight", source)
        self.assertNotIn("--gan_enabled", source)
        self.assertNotIn("--use_vae", source)

    def test_step_profiler_preserves_the_accepted_training_contract(self):
        source = (ROOT / "profile_stereo_step.py").read_text(encoding="utf-8")
        self.assertIn("set_profiling_enabled(True)", source)
        self.assertIn("max_steps=args.profile_updates", source)
        self.assertIn("if args.max_steps != 5000", source)
        self.assertIn("if args.batch_size != 8", source)
        self.assertIn("selected8 profiling freezes num_workers=0", source)
        self.assertIn("ProfilerActivity.CPU", source)
        self.assertIn("ProfilerActivity.CUDA", source)
        self.assertIn("record_shapes=True", source)
        self.assertIn("profile_memory=True", source)
        self.assertIn("with_stack=False", source)
        self.assertNotIn("torch.compile", source)
        self.assertNotIn("fused=True", source)

    def test_step_profiler_recipe_keeps_one_gpu_batch_eight_and_bf16(self):
        source = (
            ROOT / "scripts" / "stereo" / "profile_stereo_step.sh"
        ).read_text(encoding="utf-8")
        for argument in (
            "--devices 1",
            "--batch_size 8",
            '--num_workers "${PROFILE_NUM_WORKERS}"',
            "--bf16",
            "--max_steps 5000",
            "--profile_updates 40",
            "--profile_wait 15",
            "--profile_warmup 5",
            "--profile_active 10",
            "--perceptual_weight 1.0",
        ):
            self.assertIn(argument, source)
        self.assertIn("TORCH_HOME:-/home/frank/.cache/torch", source)
        self.assertIn("vgg16-397923af.pth", source)
        self.assertNotIn("--gan_enabled", source)
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

        for region in (
            "stereo/data/rgb_npz_read_decompress",
            "stereo/data/gt_npz_read_decompress",
            "stereo/data/numpy_processing_and_tensor_conversion",
            "stereo/data/collate",
        ):
            self.assertIn(region, data)
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
        profile = (ROOT / "profile_stereo_step.py").read_text(encoding="utf-8")
        self.assertIn(
            'self._backend = "conv3d_contiguous"', attention
        )
        self.assertIn('"conv3d_channels_last_3d"', attention)
        self.assertIn('"conv2d_t1_slice"', attention)
        self.assertIn("expected at least one spatial PEG module", profile)
        self.assertIn("wall_windows_by_temporal_mode", profile)
        launcher = (
            ROOT / "scripts" / "stereo" / "train_stereo_vae.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--peg_backend conv2d_t1_slice", launcher)
        self.assertIn("--pin_memory 1", launcher)
        self.assertIn("--persistent_workers 1", launcher)
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
        self.assertIn(
            'STEREO_DATA_BACKEND="${STEREO_DATA_BACKEND:-manifest_v3}"',
            launcher,
        )
        self.assertIn(
            'ONLINE_GT_CACHE_ENABLED="${ONLINE_GT_CACHE_ENABLED:-0}"',
            launcher,
        )
        self.assertIn("--foundation_stereo_valid_iters", launcher)
        self.assertIn("--lerobot_rectification_audit_sha256", launcher)
        train = (ROOT / "train_stereo_vae.py").read_text(encoding="utf-8")
        self.assertIn("use_distributed_sampler=False", train)
        self.assertNotIn("max_time=", train)

    def test_four_mode_smoke_wires_da3_and_average_checkpoint(self) -> None:
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
            "MONO_SMOKE_MANIFEST",
            "MODE_UPDATES_PER_EPOCH",
            "DA3_CHECKPOINT_SHA256",
            "--mixed_stereo_sample_limit 48",
        ):
            self.assertIn(token, launcher)
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

    def test_followup_profile_switches_are_explicit_and_disabled_by_default(self):
        profile = (ROOT / "profile_stereo_step.py").read_text(encoding="utf-8")
        launcher = (
            ROOT / "scripts" / "stereo" / "profile_stereo_step.sh"
        ).read_text(encoding="utf-8")
        data = (ROOT / "stereo_tokenizer" / "data.py").read_text(
            encoding="utf-8"
        )
        model = (ROOT / "stereo_tokenizer" / "model.py").read_text(
            encoding="utf-8"
        )
        for argument in (
            "--profile_preload_data",
            "--profile_pin_memory",
            "--profile_lpips_gt_cache",
        ):
            self.assertIn(argument, profile)
        for value in (
            "PROFILE_PRELOAD_DATA:-0",
            "PROFILE_PIN_MEMORY:-0",
            "PROFILE_LPIPS_GT_CACHE:-0",
        ):
            self.assertIn(value, launcher)
        self.assertIn("profile_preload_train_dataset", data)
        self.assertIn("LPIPS GT cache sample order changed", model)

    def test_full_dataset_profile_is_fail_closed(self):
        profile = (ROOT / "profile_stereo_step.py").read_text(encoding="utf-8")
        launcher = (
            ROOT / "scripts" / "stereo" / "profile_stereo_step.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('choices=("selected8", "full3407")', profile)
        self.assertIn("full3407 profiling freezes num_workers=8", profile)
        self.assertIn("full3407 profiling forbids data preload", profile)
        self.assertIn("expected exactly {expected_samples} samples", profile)
        self.assertIn("PROFILE_DATASET_MODE:-selected8", launcher)
        self.assertIn("PROFILE_NUM_WORKERS:-0", launcher)


if __name__ == "__main__":
    unittest.main()
