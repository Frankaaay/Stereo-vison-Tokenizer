import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
STEREO_SOURCE = ROOT / "OmniTokenizer" / "stereo"


class SourceBoundaryTest(unittest.TestCase):
    def test_stereo_sources_parse(self) -> None:
        for path in sorted(STEREO_SOURCE.glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_tokenizer_does_not_own_dit_patchify(self) -> None:
        combined_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(STEREO_SOURCE.glob("*.py"))
        )
        forbidden = ("d_DiT", "unpatchify", "dit_patch", "DiTPatch")
        for token in forbidden:
            self.assertNotIn(token, combined_source)


if __name__ == "__main__":
    unittest.main()
