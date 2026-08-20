import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
STEREO_SOURCE = ROOT / "OmniTokenizer" / "stereo"
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


if __name__ == "__main__":
    unittest.main()
