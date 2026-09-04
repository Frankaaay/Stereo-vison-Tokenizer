"""Shared input-mode contracts for StereoVAE components."""

from typing import Literal


EyeMode = Literal["mono", "stereo"]
TemporalMode = Literal["single_frame", "four_frame"]


def temporal_mode_num_frames(temporal_mode: TemporalMode) -> int:
    if temporal_mode == "single_frame":
        return 1
    if temporal_mode == "four_frame":
        return 4
    raise ValueError(f"unsupported temporal mode {temporal_mode!r}")


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret
