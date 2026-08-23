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
        self.assertIn("check_val_every_n_epoch=1", source)
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


if __name__ == "__main__":
    unittest.main()
