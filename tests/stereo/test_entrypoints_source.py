import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class StereoEntrypointSourceTest(unittest.TestCase):
    def test_training_entry_has_no_legacy_inflation_or_auto_resume(self):
        source = (ROOT / "vqgan_train.py").read_text(encoding="utf-8")
        self.assertNotIn("inflate_gen", source)
        self.assertNotIn("inflate_dis", source)
        self.assertNotIn("os.listdir", source)
        self.assertIn("limit_val_batches=1.0 if has_validation else 0", source)
        self.assertIn("check_val_every_n_epoch=1", source)

    def test_evaluation_is_deterministic_and_strict(self):
        source = (ROOT / "vqgan_eval.py").read_text(encoding="utf-8")
        self.assertIn("sample_posterior=False", source)
        self.assertIn("strict=True", source)
        self.assertIn("_checkpoint_model_args(checkpoint", source)
        self.assertIn("OmniTokenizer_VQGAN(checkpoint_args)", source)
        self.assertIn("_validate_checkpoint_semantics", source)
        self.assertNotIn(".codebook", source)
        self.assertIn("depth_abs_rel", source)

    def test_recipe_requires_unfrozen_experiment_parameters(self):
        source = (ROOT / "scripts" / "recons" / "train.sh").read_text(
            encoding="utf-8"
        )
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
        self.assertNotIn("--gan_enabled", source)
        self.assertNotIn("--use_vae", source)


if __name__ == "__main__":
    unittest.main()
