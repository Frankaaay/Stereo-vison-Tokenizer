import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
STEREO_SOURCE = ROOT / "OmniTokenizer" / "stereo"
DATA_SOURCE = ROOT / "OmniTokenizer" / "data.py"
LPIPS_SOURCE = ROOT / "OmniTokenizer" / "modules" / "lpips.py"
CALLBACKS_SOURCE = ROOT / "OmniTokenizer" / "modules" / "callbacks.py"
TRAIN_SOURCE = ROOT / "vqgan_train.py"
TRAIN_LAUNCHER_SOURCE = ROOT / "scripts" / "recons" / "train.sh"
TOKENIZER_SOURCES = (
    ROOT / "OmniTokenizer" / "omnitokenizer.py",
    ROOT / "OmniTokenizer" / "modules" / "stereo_fusion.py",
    ROOT / "OmniTokenizer" / "modules" / "stereo_geometry.py",
    ROOT / "OmniTokenizer" / "modules" / "stereo_losses.py",
)


class SourceBoundaryTest(unittest.TestCase):
    def test_main_stereo_sources_parse(self) -> None:
        for path in TOKENIZER_SOURCES:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_sidecar_implementation_is_removed(self) -> None:
        self.assertEqual(list(STEREO_SOURCE.glob("*.py")), [])

    def test_tokenizer_does_not_own_dit_patchify(self) -> None:
        combined_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in TOKENIZER_SOURCES
        )
        forbidden = ("d_DiT", "unpatchify", "dit_patch", "DiTPatch")
        for token in forbidden:
            self.assertNotIn(token, combined_source)

    def test_main_tokenizer_has_no_legacy_image_forward(self) -> None:
        source = TOKENIZER_SOURCES[0].read_text(encoding="utf-8")
        self.assertNotIn("def forward(self, video, is_image", source)
        self.assertNotIn("def forward(self, tokens, is_image", source)
        self.assertNotIn("self.codebook", source)

    def test_legacy_imagenet_dependency_is_not_imported_at_module_load(self) -> None:
        tree = ast.parse(DATA_SOURCE.read_text(encoding="utf-8"))
        top_level_modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertNotIn(
            "imagenet_stubs.imagenet_2012_labels", top_level_modules
        )

        image_dataset = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ImageDataset"
        )
        constructor = next(
            node
            for node in image_dataset.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "imagenet_stubs.imagenet_2012_labels"
                for node in ast.walk(constructor)
            )
        )

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
        self.assertFalse(
            any(
                isinstance(node.ops[0], (ast.Is, ast.IsNot))
                and isinstance(node.comparators[0], ast.Constant)
                and isinstance(node.comparators[0].value, str)
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

    def test_update_based_timm_schedulers_use_step_update(self) -> None:
        source = TOKENIZER_SOURCES[0].read_text(encoding="utf-8")
        self.assertIn("schedulers[0].step_update(self.generator_updates)", source)
        self.assertIn(
            "schedulers[1].step_update(self.discriminator_updates)", source
        )
        self.assertNotIn("step_update(self.global_step)", source)
        self.assertNotIn("schedulers[0].step(self.global_step)", source)
        self.assertNotIn("schedulers[1].step(self.global_step)", source)

    def test_callbacks_use_lightning_2_hooks_and_trainer_output_root(self) -> None:
        tree = ast.parse(CALLBACKS_SOURCE.read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
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
            self.assertFalse(
                any(
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "pl_module"
                    and node.value.attr == "logger"
                    and node.attr == "save_dir"
                    for node in ast.walk(callback)
                )
            )

    def test_encoder_owns_fusion_and_not_decoder_heads(self) -> None:
        tree = ast.parse(TOKENIZER_SOURCES[0].read_text(encoding="utf-8"))
        encoder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "OmniTokenizer_Encoder"
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
        loaded_names = {
            node.id for node in ast.walk(constructor) if isinstance(node, ast.Name)
        }
        self.assertIn("stereo_fusion", self_attributes)
        self.assertIn("stereo_temporal_projection", self_attributes)
        self.assertFalse(
            any(name.startswith("stereo_disparity_") for name in loaded_names)
        )
        self.assertNotIn("stereo_rgb_head", self_attributes)
        self.assertNotIn("stereo_disparity_head", self_attributes)


if __name__ == "__main__":
    unittest.main()
