import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "stereo_tokenizer"
MODEL_SOURCE = PACKAGE / "model.py"
DATA_SOURCE = PACKAGE / "data.py"
LPIPS_SOURCE = PACKAGE / "modules" / "lpips.py"
CALLBACKS_SOURCE = PACKAGE / "modules" / "callbacks.py"
TRAIN_SOURCE = ROOT / "train_stereo_vae.py"
TRAIN_LAUNCHER_SOURCE = ROOT / "scripts" / "stereo" / "train_stereo_vae.sh"
ACTIVE_SOURCES = tuple(PACKAGE.rglob("*.py")) + (
    TRAIN_SOURCE,
    ROOT / "eval_stereo_vae.py",
)


class SourceBoundaryTest(unittest.TestCase):
    def test_stereo_sources_parse(self) -> None:
        for path in ACTIVE_SOURCES:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_public_package_exports_only_stereo_vae(self) -> None:
        tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "__all__"
        }
        self.assertEqual(
            assignments["__all__"],
            ["StereoVAE", "StereoEncodeOutput", "EyeMode", "TemporalMode"],
        )

    def test_legacy_packages_and_entrypoints_are_removed(self) -> None:
        for directory in (ROOT / "OmniTokenizer", ROOT / "Diffusion"):
            files = [
                path
                for path in directory.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ]
            self.assertEqual(files, [], str(directory))
        for path in (
            ROOT / "vqgan_train.py",
            ROOT / "vqgan_eval.py",
            ROOT / "transformer_train.py",
            ROOT / "transformer_eval.py",
        ):
            self.assertFalse(path.exists(), str(path))

    def test_active_sources_have_no_legacy_vq_api(self) -> None:
        combined_source = "\n".join(
            path.read_text(encoding="utf-8") for path in ACTIVE_SOURCES
        )
        for token in (
            "VQGAN",
            "OmniTokenizer",
            "codebook_dim",
            "pre_vq_conv",
            "post_vq_conv",
            "load_vqgan",
        ):
            self.assertNotIn(token, combined_source)

    def test_tokenizer_does_not_own_dit_patchify(self) -> None:
        combined_source = "\n".join(
            path.read_text(encoding="utf-8") for path in ACTIVE_SOURCES
        )
        for token in ("d_DiT", "unpatchify", "dit_patch", "DiTPatch"):
            self.assertNotIn(token, combined_source)

    def test_model_declares_stereo_model_names(self) -> None:
        tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
        classes = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
        expected = {
            "StereoVAE",
            "StereoEncoder",
            "StereoDecoder",
            "StereoEncodeOutput",
            "StereoDecodeOutput",
            "StereoVAEOutput",
        }
        self.assertEqual({name for name in classes if not name.startswith("_")}, expected)

    def test_data_module_classes_match_supported_sources(self) -> None:
        tree = ast.parse(DATA_SOURCE.read_text(encoding="utf-8"))
        classes = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
        self.assertEqual(
            classes,
            {
                "ModeSubset",
                "StereoDataModule",
            },
        )
        source = DATA_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("StereoManifestDataset", source)
        self.assertNotIn("manifest_v3", source)

    def test_lpips_pretrained_name_uses_value_comparison(self) -> None:
        tree = ast.parse(LPIPS_SOURCE.read_text(encoding="utf-8"))
        lpips = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LPIPS"
        )
        from_pretrained = next(
            node
            for node in lpips.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "from_pretrained"
        )
        comparisons = [
            node for node in ast.walk(from_pretrained) if isinstance(node, ast.Compare)
        ]
        self.assertTrue(
            any(
                isinstance(node.ops[0], ast.NotEq)
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "vgg_lpips"
                for node in comparisons
            )
        )

    def test_training_entrypoint_uses_lightning_2_trainer_api(self) -> None:
        source = TRAIN_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("Trainer.add_argparse_args", source)
        self.assertNotIn("Trainer.from_argparse_args", source)
        self.assertIn('parser.add_argument("--devices"', source)
        self.assertIn('precision = "bf16-mixed"', source)

        launcher = TRAIN_LAUNCHER_SOURCE.read_text(encoding="utf-8")
        self.assertIn('--devices "${GPU_COUNT}"', launcher)
        self.assertNotIn('--gpus "${GPU_COUNT}"', launcher)

    def test_update_based_timm_schedulers_use_independent_counters(self) -> None:
        source = MODEL_SOURCE.read_text(encoding="utf-8")
        self.assertIn("schedulers[0].step_update(self.generator_updates)", source)
        self.assertIn(
            "schedulers[1].step_update(self.discriminator_updates)", source
        )
        self.assertNotIn("step_update(self.global_step)", source)

    def test_callbacks_use_lightning_2_hooks_and_trainer_output_root(self) -> None:
        tree = ast.parse(CALLBACKS_SOURCE.read_text(encoding="utf-8"))
        imports = {
            node.module for node in tree.body if isinstance(node, ast.ImportFrom)
        }
        self.assertIn("pytorch_lightning.utilities", imports)
        self.assertNotIn("pytorch_lightning.utilities.distributed", imports)

        callback_classes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in {"ImageLogger", "VideoLogger"}
        }
        self.assertEqual(set(callback_classes), {"ImageLogger", "VideoLogger"})
        for callback in callback_classes.values():
            methods = {
                node.name: node
                for node in callback.body
                if isinstance(node, ast.FunctionDef)
            }
            train_hook = methods["on_train_batch_end"]
            self.assertEqual(
                [argument.arg for argument in train_hook.args.args],
                ["self", "trainer", "pl_module", "outputs", "batch", "batch_idx"],
            )
            self.assertTrue(
                any(
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "trainer"
                    and node.attr == "default_root_dir"
                    for node in ast.walk(callback)
                )
            )

    def test_encoder_owns_fusion_and_not_decoder_heads(self) -> None:
        tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
        encoder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "StereoEncoder"
        )
        constructor = next(
            node
            for node in encoder.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self_attributes = {
            node.attr
            for node in ast.walk(constructor)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }
        self.assertIn("stereo_fusion", self_attributes)
        self.assertIn("stereo_temporal_projection", self_attributes)
        self.assertIn("single_frame_projection", self_attributes)
        self.assertNotIn("stereo_rgb_head", self_attributes)
        self.assertNotIn("stereo_disparity_head", self_attributes)

    def test_tokenizer_does_not_own_memory_roles_or_combined_mode_loss(self) -> None:
        source = MODEL_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("LatentRole", source)
        self.assertNotIn("latent_role", source)
        self.assertNotIn("single_frame_loss_weight", source)
        self.assertNotIn("combined_loss", source)


if __name__ == "__main__":
    unittest.main()
