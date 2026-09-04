import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

from stereo_tokenizer.online_gt import FoundationStereoOnlineTeacher, LAS2HOnlineTeacher


def test_las2h_backend_is_selectable_before_asset_validation():
    args = types.SimpleNamespace()
    with pytest.raises(FileNotFoundError) as error:
        FoundationStereoOnlineTeacher(
            "/missing/las2h-repo",
            "/missing/LAS2_H.pth",
            "0" * 64,
            device="cpu",
            valid_iters=4,
            pair_microbatch=48,
            backend="las2_h",
            las2_h_source_sha="a" * 40,
            las2_h_max_disp=192,
        )
    assert "LAS2-H" in str(error.value)


def test_las2h_pair_output_reshapes_to_three_views_and_four_frames():
    pair_output = torch.ones(12, 1, 8, 10)
    output = LAS2HOnlineTeacher.reshape_pair_output(
        pair_output, batch=1, views=3, frames=4
    )
    assert output.shape == (1, 3, 1, 4, 8, 10)
    assert output.dtype == torch.float32
    assert output.is_contiguous()


def test_las2h_pair_output_rejects_wrong_pair_count():
    with pytest.raises(ValueError, match="12 stereo pairs"):
        LAS2HOnlineTeacher.reshape_pair_output(
            torch.ones(11, 1, 8, 10), batch=1, views=3, frames=4
        )


def test_las2h_rejects_unpinned_source_before_loading_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        repo = root / "repo"
        repo.mkdir()
        checkpoint = root / "LAS2_H.pth"
        checkpoint.write_bytes(b"checkpoint")
        with mock.patch(
            "stereo_tokenizer.online_gt._validate_clean_git_source",
            side_effect=ValueError("LAS2-H source SHA mismatch"),
        ):
            with pytest.raises(ValueError, match="source SHA mismatch"):
                LAS2HOnlineTeacher(
                    repo,
                    checkpoint,
                    "b" * 64,
                    source_sha="a" * 40,
                    device="cuda:0",
                )
