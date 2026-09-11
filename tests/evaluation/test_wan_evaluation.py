from types import SimpleNamespace

import pytest
import torch

from evaluation.wan.adapter import WanReconstructor
from evaluation.wan.selection import all_hy_windows


class RecordingVAE:
    def __init__(self):
        self.clips = []

    def encode(self, clip, scale):
        self.clips.append(clip.clone())
        return clip

    def decode(self, latent, scale):
        return latent


def test_target_view_order_values_and_four_frame_contract():
    adapter = WanReconstructor.__new__(WanReconstructor)
    model = RecordingVAE()
    adapter.vae = SimpleNamespace(model=model, scale=None)
    adapter.latent_shapes = {}
    target = torch.linspace(-0.5, 0.5, 2 * 3 * 3 * 4 * 16 * 16).reshape(2, 3, 3, 4, 16, 16)
    result = adapter(target)
    assert torch.equal(result.rgb, target)
    assert result.latent.shape[:4] == (2, 3, 3, 5)
    assert len(model.clips) == 6
    assert all(clip.shape == (1, 3, 5, 16, 16) for clip in model.clips)
    for clip, original in zip(model.clips, target.flatten(0, 1)):
        assert torch.equal(clip[0, :, :4], original * 2)
        assert torch.equal(clip[:, :, 4], clip[:, :, 3])
    one = target[:, :, :, :1]
    assert torch.equal(adapter(one).rgb, one)
    assert model.clips[-1].shape[2] == 1


def test_invalid_inputs_fail_before_backend_execution():
    adapter = WanReconstructor.__new__(WanReconstructor)
    for target in [torch.zeros(1, 1, 3, 2, 16, 16), torch.full((1, 1, 3, 1, 16, 16), float('nan')), torch.ones(1, 1, 3, 1, 16, 16)]:
        with pytest.raises(ValueError):
            adapter(target)


def test_all_windows_keep_test_identity_and_exclude_table14():
    identity = {"records": [
        {"episode_id": "table_012:1", "table_name": "table_012", "episode_index": 1, "split": "test"},
        {"episode_id": "table_014:2", "table_name": "table_014", "episode_index": 2, "split": "test"},
        {"episode_id": "table_012:3", "table_name": "table_012", "episode_index": 3, "split": "train"},
    ]}
    matched = {("table_012", 1): {"window_count": 2, "length": 24, "fps": 30}}
    rows = all_hy_windows(identity, matched)
    assert [r["expected_source_frame_indices"] for r in rows] == [[0, 3, 6, 9], [12, 15, 18, 21]]
    assert {r["legacy_episode_id"] for r in rows} == {"table_012:1"}
    matched[("table_012", 1)]["length"] = 20
    with pytest.raises(ValueError, match="exceeds episode"):
        all_hy_windows(identity, matched)
