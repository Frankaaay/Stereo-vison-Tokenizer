from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class StereoFusionOutput:
    """Stereo matching output and diagnostics used by the tokenizer."""

    features: torch.Tensor
    attention: torch.Tensor
    confidence: torch.Tensor
    valid_mask: torch.Tensor


class StereoFusion(nn.Module):
    """Shared horizontal cross-attention for rectified stereo features.

    Inputs use channels-last layout ``[B, V, T, H, W, D]``. Projection
    weights are shared across views, while ``search_radii`` defines the valid
    horizontal candidates independently for each view.
    """

    _DIRECTIONS = {"left": -1, "right": 1}

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        head_dim: int,
        search_radii: Sequence[int],
        search_direction: str,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or head_dim <= 0:
            raise ValueError("dim, heads, and head_dim must be positive")
        if heads * head_dim != dim:
            raise ValueError(
                "StereoFusion requires heads * head_dim == dim so its shared "
                "output projection preserves the feature width"
            )
        if not search_radii:
            raise ValueError("search_radii must contain one value per view")
        if any(radius < 0 for radius in search_radii):
            raise ValueError("search radii must be non-negative")
        if search_direction not in self._DIRECTIONS:
            raise ValueError(
                f"search_direction must be one of {tuple(self._DIRECTIONS)}, "
                f"got {search_direction!r}"
            )

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.search_direction = search_direction
        self.max_search_radius = max(search_radii)

        self.register_buffer(
            "search_radii",
            torch.as_tensor(tuple(search_radii), dtype=torch.long),
            persistent=True,
        )

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.offset_bias = nn.Parameter(
            torch.zeros(heads, self.max_search_radius + 1)
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.scale = head_dim**-0.5

    def _candidate_layout(
        self, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if width <= 0:
            raise ValueError(f"feature width must be positive, got {width}")
        if self.max_search_radius >= width:
            raise ValueError(
                f"maximum search radius {self.max_search_radius} must be "
                f"smaller than feature width {width}"
            )

        x = torch.arange(width, device=device)[:, None]
        offsets = torch.arange(
            self.max_search_radius + 1, device=device
        )[None, :]
        direction = self._DIRECTIONS[self.search_direction]
        candidate_x = x + direction * offsets
        boundary_valid = (candidate_x >= 0) & (candidate_x < width)
        candidate_x = candidate_x.clamp(0, width - 1)

        radius_valid = offsets[None, :, :] <= self.search_radii.to(device)[
            :, None, None
        ]
        valid_mask = radius_valid & boundary_valid[None, :, :]
        return candidate_x, valid_mask

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        fusion_scale_override: float | None = None,
    ) -> StereoFusionOutput:
        if left.shape != right.shape:
            raise ValueError(
                "left and right feature shapes must match, got "
                f"{tuple(left.shape)} and {tuple(right.shape)}"
            )
        if left.ndim != 6:
            raise ValueError(
                "StereoFusion expects [B,V,T,H,W,D], "
                f"got a {left.ndim}-D tensor"
            )

        _, views, _, _, width, dim = left.shape
        if views != self.search_radii.numel():
            raise ValueError(
                f"input has {views} views but search_radii has "
                f"{self.search_radii.numel()} entries"
            )
        if dim != self.dim:
            raise ValueError(f"expected feature dim {self.dim}, got {dim}")

        candidate_x, valid_mask = self._candidate_layout(width, left.device)
        right_candidates = right[..., candidate_x, :]

        query = self.to_q(left).reshape(
            *left.shape[:-1], self.heads, self.head_dim
        )
        key = self.to_k(right_candidates).reshape(
            *right_candidates.shape[:-1], self.heads, self.head_dim
        )
        value = self.to_v(right_candidates).reshape(
            *right_candidates.shape[:-1], self.heads, self.head_dim
        )

        scores = torch.einsum("...hd,...khd->...hk", query, key) * self.scale
        scores = scores + self.offset_bias
        expanded_mask = valid_mask[None, :, None, None, :, None, :]
        scores = scores.masked_fill(~expanded_mask, -torch.inf)

        attention = torch.softmax(scores, dim=-1)
        attention_for_value = self.attention_dropout(attention)
        matched = torch.einsum(
            "...hk,...khd->...hd", attention_for_value, value
        )
        matched = matched.reshape(*left.shape[:-1], self.dim)
        delta = self.to_out(matched)

        probability = attention.clamp_min(torch.finfo(attention.dtype).tiny)
        entropy = -(attention * probability.log()).sum(dim=-1)
        valid_count = valid_mask.sum(dim=-1)
        entropy_denominator = valid_count.clamp_min(2).log().to(entropy.dtype)
        head_confidence = 1.0 - entropy / entropy_denominator[
            None, :, None, None, :, None
        ]
        head_confidence = torch.where(
            (valid_count == 1)[None, :, None, None, :, None],
            torch.ones_like(head_confidence),
            head_confidence,
        )
        confidence = head_confidence.mean(dim=-1).clamp(0.0, 1.0)

        # Sharpness is a diagnostic gate, not a trainable shortcut for making
        # the attention distribution artificially peaky.
        fusion_scale = (
            1.0
            if fusion_scale_override is None
            else float(fusion_scale_override)
        )
        if not math.isfinite(fusion_scale):
            raise ValueError("fusion_scale_override must be finite")
        fused = (
            left
            + fusion_scale
            * self.alpha
            * confidence.detach()[..., None]
            * delta
        )
        return StereoFusionOutput(
            features=fused,
            attention=attention,
            confidence=confidence,
            valid_mask=valid_mask,
        )
