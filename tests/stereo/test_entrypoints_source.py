import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class StereoEntrypointSourceTest(unittest.TestCase):
    def test_training_entry_has_no_legacy_inflation_or_auto_resume(self):
        source = (ROOT / "train_stereo_vae.py").read_text(encoding="utf-8")
        self.assertNotIn("inflate_gen", source)
        self.assertNotIn("inflate_dis", source)
        self.assertNotIn("os.listdir", source)
        self.assertIn("StereoVAE(args)", source)
        self.assertIn("StereoDataModule(args)", source)
        self.assertIn("limit_val_batches=1.0 if has_validation else 0", source)
        self.assertIn("check_val_every_n_epoch=1", source)
        self.assertIn("max_steps=-1 if args.gan_enabled else args.max_steps", source)
        self.assertIn("max_epochs=-1", source)

    def test_evaluation_is_deterministic_and_strict(self):
        source = (ROOT / "eval_stereo_vae.py").read_text(encoding="utf-8")
        self.assertIn("sample_posterior=False", source)
        self.assertIn("strict=True", source)
        self.assertIn("_checkpoint_model_args(checkpoint", source)
        self.assertIn("StereoVAE(checkpoint_args)", source)
        self.assertIn("--stereo_vae_ckpt", source)
        self.assertIn("_validate_checkpoint_semantics", source)
        self.assertNotIn(".codebook", source)
        self.assertIn("depth_abs_rel", source)
        self.assertIn("disparity_to_depth(", source)
        self.assertNotIn("calibration / disparity_target", source)

    def test_recipe_requires_unfrozen_experiment_parameters(self):
        source = (
            ROOT / "scripts" / "stereo" / "train_stereo_vae.sh"
        ).read_text(encoding="utf-8")
        for name in (
            "PER_DEVICE_BATCH_SIZE",
            "GRAD_ACCUMULATES",
            "RGB_WEIGHT",
            "DISPARITY_WEIGHT",
            "GRADIENT_WEIGHT",
            "KL_WEIGHT",
            "KL_WARMUP_STEPS",
        ):
            self.assertIn(f"${{{name}:?", source)
        self.assertIn("python3 train_stereo_vae.py", source)
        self.assertIn("--latent_channels 48", source)
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
            "stereo/loss/disparity",
            "stereo/loss/disparity_gradient",
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
        self.assertIn("expected 14 PEG modules", profile)
        launcher = (
            ROOT / "scripts" / "stereo" / "train_stereo_vae.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--peg_backend conv2d_t1_slice", launcher)
        self.assertIn("--pin_memory 1", launcher)
        self.assertIn("--persistent_workers 1", launcher)
        self.assertIn(
            '--checkpoint_every_n_steps "${CHECKPOINT_EVERY_N_STEPS:-100}"',
            launcher,
        )
        train = (ROOT / "train_stereo_vae.py").read_text(encoding="utf-8")
        self.assertIn("DDPStrategy(", train)
        self.assertIn("static_graph=not args.gan_enabled", train)
        self.assertIn("find_unused_parameters=args.gan_enabled", train)
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
        self.assertIn('on_epoch=prefix != "train"', model)

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
