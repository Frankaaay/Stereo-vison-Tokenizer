from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RelativeDepthTarget:
    relative_log_depth: torch.Tensor
    valid_mask: torch.Tensor
    sample_center: torch.Tensor
    teacher_kind: str


def _validate_dense_map(name: str, value: torch.Tensor) -> None:
    if value.ndim != 6:
        raise ValueError(f"{name} must use [B,V,1,T,H,W], got {value.shape}")
    if value.shape[2] != 1:
        raise ValueError(f"{name} must contain one geometry channel")


def _validate_mask(reference: torch.Tensor, valid_mask: torch.Tensor) -> None:
    _validate_dense_map("valid_mask", valid_mask)
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if valid_mask.shape != reference.shape:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} must match {reference.shape}"
        )


def center_relative_log_depth(
    log_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    require_all_finite: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove one view-equal log-depth center from every sample.

    Each view is first reduced across ``[C,T,H,W]`` using its valid pixels.
    The resulting view centers are then averaged, so views remain equal even
    when they contain different numbers of valid pixels.
    """

    _validate_dense_map("log_depth", log_depth)
    _validate_mask(log_depth, valid_mask)
    if require_all_finite and not torch.isfinite(log_depth).all():
        raise ValueError("log-depth prediction contains NaN/Inf")
    if not torch.isfinite(log_depth.masked_select(valid_mask)).all():
        raise ValueError("valid log-depth values contain NaN/Inf")

    reduction_dims = (2, 3, 4, 5)
    valid_count = valid_mask.sum(dim=reduction_dims)
    if torch.any(valid_count == 0):
        empty = torch.nonzero(valid_count == 0, as_tuple=False)
        raise ValueError(
            "every sample/view must contain valid depth; empty [B,V] entries: "
            f"{empty.detach().cpu().tolist()}"
        )
    safe_log_depth = torch.where(
        valid_mask,
        log_depth.float(),
        torch.zeros((), device=log_depth.device, dtype=torch.float32),
    )
    view_center = safe_log_depth.sum(dim=reduction_dims) / valid_count.float()
    sample_center = view_center.mean(dim=1).reshape(-1, 1, 1, 1, 1, 1)
    relative = log_depth.float() - sample_center
    return relative, sample_center


def relative_target_from_da3(
    relative_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float,
) -> RelativeDepthTarget:
    _validate_dense_map("DA3 relative depth", relative_depth)
    _validate_mask(relative_depth, valid_mask)
    if epsilon <= 0:
        raise ValueError("relative-depth epsilon must be positive")
    valid_depth = relative_depth.masked_select(valid_mask)
    if not torch.isfinite(valid_depth).all():
        raise ValueError("valid DA3 relative depth contains NaN/Inf")
    if torch.any(valid_depth <= 0):
        raise ValueError("valid DA3 relative depth must be positive")
    safe_depth = torch.where(
        valid_mask,
        relative_depth.float().clamp_min(epsilon),
        torch.ones((), device=relative_depth.device, dtype=torch.float32),
    )
    teacher_log_depth = safe_depth.log()
    relative, center = center_relative_log_depth(
        teacher_log_depth,
        valid_mask,
        require_all_finite=False,
    )
    return RelativeDepthTarget(relative, valid_mask, center, "da3")


def relative_target_from_foundation_stereo(
    disparity: torch.Tensor,
    valid_mask: torch.Tensor,
    fx: torch.Tensor,
    baseline_m: torch.Tensor,
    *,
    epsilon: float,
) -> RelativeDepthTarget:
    _validate_dense_map("FoundationStereo disparity", disparity)
    _validate_mask(disparity, valid_mask)
    if epsilon <= 0:
        raise ValueError("disparity epsilon must be positive")
    expected_calibration_shape = disparity.shape[:2]
    if fx.shape != expected_calibration_shape:
        raise ValueError(
            f"fx must use [B,V]={expected_calibration_shape}, got {fx.shape}"
        )
    if baseline_m.shape != expected_calibration_shape:
        raise ValueError(
            "baseline_m must use [B,V]="
            f"{expected_calibration_shape}, got {baseline_m.shape}"
        )
    for name, calibration in (("fx", fx), ("baseline_m", baseline_m)):
        if not torch.isfinite(calibration).all():
            raise ValueError(f"{name} contains NaN/Inf")
        if torch.any(calibration <= 0):
            raise ValueError(f"{name} must be positive")
    valid_disparity = disparity.masked_select(valid_mask)
    if not torch.isfinite(valid_disparity).all():
        raise ValueError("valid FoundationStereo disparity contains NaN/Inf")
    if torch.any(valid_disparity <= epsilon):
        raise ValueError("valid FoundationStereo disparity must exceed epsilon")

    safe_disparity = torch.where(
        valid_mask,
        disparity.float().clamp_min(epsilon),
        torch.ones((), device=disparity.device, dtype=torch.float32),
    )
    log_scale = (fx.float() * baseline_m.float()).log().reshape(
        disparity.shape[0], disparity.shape[1], 1, 1, 1, 1
    )
    teacher_log_depth = log_scale - safe_disparity.log()
    relative, center = center_relative_log_depth(
        teacher_log_depth,
        valid_mask,
        require_all_finite=False,
    )
    return RelativeDepthTarget(relative, valid_mask, center, "foundation_stereo")


def relative_prediction_from_raw(
    raw_relative_log_depth: torch.Tensor,
    teacher_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return center_relative_log_depth(
        raw_relative_log_depth,
        teacher_valid_mask,
        require_all_finite=True,
    )
