from dataclasses import dataclass

import torch


@dataclass
class DepthOutput:
    """Metric depth and the mask of pixels where conversion is valid."""

    depth: torch.Tensor
    valid_mask: torch.Tensor


def _expand_calibration(
    name: str, value: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    batch, views, _, time, _, _ = reference.shape
    if value.ndim == 2:
        expected = (batch, views)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        return value[:, :, None, None, None, None]
    if value.ndim == 3:
        expected = (batch, views, time)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        return value[:, :, None, :, None, None]
    if value.ndim == 6:
        try:
            return torch.broadcast_to(value, reference.shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name} shape {value.shape} is not broadcastable to "
                f"{reference.shape}"
            ) from error
    raise ValueError(
        f"{name} must use [B,V], [B,V,T], or a 6-D broadcastable layout"
    )


def disparity_to_depth(
    disparity: torch.Tensor,
    focal_length_x: torch.Tensor,
    baseline_meters: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    minimum_disparity: float = 1e-6,
) -> DepthOutput:
    """Convert pixel disparity to metric depth using ``D = fx * B / d``."""

    if disparity.ndim != 6 or disparity.shape[2] != 1:
        raise ValueError("disparity must use [B,V,1,T,H,W]")
    if minimum_disparity <= 0:
        raise ValueError("minimum_disparity must be positive")

    fx = _expand_calibration("focal_length_x", focal_length_x, disparity)
    baseline = _expand_calibration("baseline_meters", baseline_meters, disparity)
    if not torch.isfinite(fx).all() or torch.any(fx <= 0):
        raise ValueError("focal_length_x must be finite and positive")
    if not torch.isfinite(baseline).all() or torch.any(baseline <= 0):
        raise ValueError("baseline_meters must be finite and positive")

    derived_valid = torch.isfinite(disparity) & (disparity > minimum_disparity)
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        if valid_mask.shape != disparity.shape:
            raise ValueError("valid_mask shape must match disparity")
        derived_valid = derived_valid & valid_mask

    safe_disparity = torch.where(
        derived_valid, disparity, torch.ones_like(disparity)
    )
    depth = fx * baseline / safe_disparity
    depth = torch.where(derived_valid, depth, torch.zeros_like(depth))
    return DepthOutput(depth=depth, valid_mask=derived_valid)
