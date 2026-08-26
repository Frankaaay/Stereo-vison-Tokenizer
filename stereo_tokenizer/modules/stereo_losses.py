from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..profiling import profile_region


class PosteriorWithKL(Protocol):
    def kl(self) -> torch.Tensor: ...


@dataclass
class MaskedViewLoss:
    loss: torch.Tensor
    per_view: torch.Tensor
    valid_count: torch.Tensor
    supervised_sample_count: torch.Tensor


@dataclass
class StereoLossBreakdown:
    total: torch.Tensor
    rgb: torch.Tensor
    relative_depth: torch.Tensor
    relative_gradient: torch.Tensor
    kl: torch.Tensor
    relative_depth_per_view: torch.Tensor
    relative_depth_valid_count: torch.Tensor
    relative_depth_supervised_sample_count: torch.Tensor
    gradient_per_view: torch.Tensor
    gradient_valid_count: torch.Tensor
    gradient_supervised_sample_count: torch.Tensor


def _validate_dense_map(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 6:
        raise ValueError(f"{name} must use [B,V,C,T,H,W], got {tensor.shape}")


def _validate_mask(reference: torch.Tensor, valid_mask: torch.Tensor) -> None:
    _validate_dense_map("valid_mask", valid_mask)
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean; confidence weighting is separate")
    if valid_mask.shape != reference.shape:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} must match {reference.shape}"
        )


def _average_available_sample_views(
    per_sample_view: torch.Tensor, supervised_view: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if per_sample_view.ndim != 2 or supervised_view.shape != per_sample_view.shape:
        raise ValueError("sample/view reductions must use matching [B,V] tensors")
    if supervised_view.dtype != torch.bool:
        raise TypeError("supervised_view must be boolean")
    supervised_per_sample = supervised_view.sum(dim=1)
    if torch.any(supervised_per_sample == 0):
        empty_samples = torch.nonzero(
            supervised_per_sample == 0, as_tuple=False
        ).flatten()
        raise ValueError(
            "every sample must contain supervision in at least one view; "
            "empty [B]: "
            f"{empty_samples.detach().cpu().tolist()}"
        )
    supervised_sample_count = supervised_view.sum(dim=0)
    masked = per_sample_view.masked_fill(~supervised_view, 0)
    per_sample = masked.sum(dim=1) / supervised_per_sample.to(
        per_sample_view.dtype
    )
    per_view = masked.sum(dim=0) / supervised_sample_count.clamp_min(1).to(
        per_sample_view.dtype
    )
    return per_sample.mean(), per_view, supervised_sample_count


def _reduce_per_view(
    elementwise_loss: torch.Tensor, valid_mask: torch.Tensor
) -> MaskedViewLoss:
    reduction_dims = (2, 3, 4, 5)
    per_sample_view_count = valid_mask.sum(dim=reduction_dims)
    valid_count = per_sample_view_count.sum(dim=0)
    supervised_view = per_sample_view_count > 0
    loss_sum = elementwise_loss.masked_fill(~valid_mask, 0).sum(
        dim=reduction_dims
    )
    per_sample_view = loss_sum / per_sample_view_count.clamp_min(1).to(
        elementwise_loss.dtype
    )
    loss, per_view, supervised_sample_count = _average_available_sample_views(
        per_sample_view, supervised_view
    )
    return MaskedViewLoss(
        loss=loss,
        per_view=per_view,
        valid_count=valid_count,
        supervised_sample_count=supervised_sample_count,
    )


def _safe_masked_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep invalid target values out of loss kernels and their backward pass."""

    return torch.where(valid_mask, target, prediction.detach())


def masked_smooth_l1_relative_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta: float,
) -> MaskedViewLoss:
    """SmoothL1 on centered relative log-depth with sample/view equality."""

    _validate_dense_map("prediction", prediction)
    _validate_dense_map("target", target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target relative log-depth shapes must match")
    _validate_mask(prediction, valid_mask)
    if beta <= 0:
        raise ValueError("SmoothL1 beta must be positive")
    if not torch.isfinite(prediction).all():
        raise ValueError("predicted relative log-depth contains NaN/Inf")
    if not torch.isfinite(target.masked_select(valid_mask)).all():
        raise ValueError("valid target relative log-depth contains NaN/Inf")

    safe_target = _safe_masked_target(prediction, target, valid_mask)
    elementwise = F.smooth_l1_loss(
        prediction, safe_target, reduction="none", beta=beta
    )
    return _reduce_per_view(elementwise, valid_mask)


def masked_relative_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta: float,
) -> MaskedViewLoss:
    """Spatial x/y SmoothL1 on relative log-depth, normalized per sample/view."""

    _validate_dense_map("prediction", prediction)
    _validate_dense_map("target", target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target relative log-depth shapes must match")
    _validate_mask(prediction, valid_mask)
    if beta <= 0:
        raise ValueError("SmoothL1 beta must be positive")
    if prediction.shape[-2] < 2 or prediction.shape[-1] < 2:
        raise ValueError("gradient loss requires H>=2 and W>=2")
    if not torch.isfinite(prediction).all():
        raise ValueError("predicted relative log-depth contains NaN/Inf")
    if not torch.isfinite(target.masked_select(valid_mask)).all():
        raise ValueError("valid target relative log-depth contains NaN/Inf")

    safe_target = _safe_masked_target(prediction, target, valid_mask)
    prediction_dx = prediction[..., 1:] - prediction[..., :-1]
    target_dx = safe_target[..., 1:] - safe_target[..., :-1]
    mask_dx = valid_mask[..., 1:] & valid_mask[..., :-1]

    prediction_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = safe_target[..., 1:, :] - safe_target[..., :-1, :]
    mask_dy = valid_mask[..., 1:, :] & valid_mask[..., :-1, :]

    reduction_dims = (2, 3, 4, 5)
    count_x = mask_dx.sum(dim=reduction_dims)
    count_y = mask_dy.sum(dim=reduction_dims)
    loss_x = F.smooth_l1_loss(
        prediction_dx, target_dx, reduction="none", beta=beta
    ).masked_fill(~mask_dx, 0).sum(dim=reduction_dims)
    loss_y = F.smooth_l1_loss(
        prediction_dy, target_dy, reduction="none", beta=beta
    ).masked_fill(~mask_dy, 0).sum(dim=reduction_dims)
    supervised_view = (count_x > 0) & (count_y > 0)
    per_sample_view = 0.5 * (
        loss_x / count_x.clamp_min(1).to(prediction.dtype)
        + loss_y / count_y.clamp_min(1).to(prediction.dtype)
    )
    loss, per_view, supervised_sample_count = _average_available_sample_views(
        per_sample_view, supervised_view
    )
    valid_count = (count_x + count_y).sum(dim=0)
    return MaskedViewLoss(
        loss=loss,
        per_view=per_view,
        valid_count=valid_count,
        supervised_sample_count=supervised_sample_count,
    )


def posterior_kl_loss(posterior: PosteriorWithKL) -> torch.Tensor:
    """Element-summed KL per ``[B,V]``, then averaged across samples/views."""

    per_sample_view = posterior.kl()
    if per_sample_view.ndim != 2:
        raise ValueError(
            f"posterior.kl() must return [B,V], got {per_sample_view.shape}"
        )
    if not torch.isfinite(per_sample_view).all():
        raise ValueError("posterior KL contains NaN/Inf")
    return per_sample_view.mean()


def rgb_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: Literal["l1", "l2"],
) -> torch.Tensor:
    _validate_dense_map("RGB prediction", prediction)
    _validate_dense_map("RGB target", target)
    if prediction.shape != target.shape:
        raise ValueError("RGB prediction and target shapes must match")
    if loss_type == "l1":
        return F.l1_loss(prediction, target)
    if loss_type == "l2":
        return F.mse_loss(prediction, target)
    raise ValueError(f"unsupported RGB reconstruction loss {loss_type!r}")


class StereoReconstructionKLLoss(nn.Module):
    """Core RGB/relative-depth/gradient/KL objective with explicit weights.

    LPIPS, GAN, and feature matching remain separate training-stage terms so
    their activation gates cannot be hidden in this deterministic core loss.
    """

    def __init__(
        self,
        *,
        rgb_weight: float,
        relative_depth_weight: float,
        relative_gradient_weight: float,
        kl_weight: float,
        smooth_l1_beta: float,
        rgb_loss_type: Literal["l1", "l2"],
    ) -> None:
        super().__init__()
        weights = (
            rgb_weight,
            relative_depth_weight,
            relative_gradient_weight,
            kl_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        self.rgb_weight = rgb_weight
        self.relative_depth_weight = relative_depth_weight
        self.relative_gradient_weight = relative_gradient_weight
        self.kl_weight = kl_weight
        self.smooth_l1_beta = smooth_l1_beta
        self.rgb_loss_type = rgb_loss_type

    def forward(
        self,
        *,
        rgb_prediction: torch.Tensor,
        rgb_target: torch.Tensor,
        relative_depth_prediction: torch.Tensor,
        relative_depth_target: torch.Tensor,
        valid_mask: torch.Tensor,
        posterior: PosteriorWithKL,
        kl_weight_override: float | None = None,
    ) -> StereoLossBreakdown:
        with profile_region("stereo/loss/rgb"):
            rgb = rgb_reconstruction_loss(
                rgb_prediction, rgb_target, loss_type=self.rgb_loss_type
            )
        with profile_region("stereo/loss/relative_depth"):
            relative_depth = masked_smooth_l1_relative_depth_loss(
                relative_depth_prediction,
                relative_depth_target,
                valid_mask,
                beta=self.smooth_l1_beta,
            )
        with profile_region("stereo/loss/relative_gradient"):
            gradient = masked_relative_gradient_loss(
                relative_depth_prediction,
                relative_depth_target,
                valid_mask,
                beta=self.smooth_l1_beta,
            )
        with profile_region("stereo/loss/kl"):
            kl = posterior_kl_loss(posterior)
        kl_weight = (
            self.kl_weight
            if kl_weight_override is None
            else kl_weight_override
        )
        if kl_weight < 0:
            raise ValueError("kl_weight_override must be non-negative")
        total = (
            self.rgb_weight * rgb
            + self.relative_depth_weight * relative_depth.loss
            + self.relative_gradient_weight * gradient.loss
            + kl_weight * kl
        )
        return StereoLossBreakdown(
            total=total,
            rgb=rgb,
            relative_depth=relative_depth.loss,
            relative_gradient=gradient.loss,
            kl=kl,
            relative_depth_per_view=relative_depth.per_view,
            relative_depth_valid_count=relative_depth.valid_count,
            relative_depth_supervised_sample_count=(
                relative_depth.supervised_sample_count
            ),
            gradient_per_view=gradient.per_view,
            gradient_valid_count=gradient.valid_count,
            gradient_supervised_sample_count=gradient.supervised_sample_count,
        )
