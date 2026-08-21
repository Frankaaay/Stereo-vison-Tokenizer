from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class PosteriorWithKL(Protocol):
    def kl(self) -> torch.Tensor: ...


@dataclass
class MaskedViewLoss:
    loss: torch.Tensor
    per_view: torch.Tensor
    valid_count: torch.Tensor


@dataclass
class StereoLossBreakdown:
    total: torch.Tensor
    rgb: torch.Tensor
    disparity: torch.Tensor
    disparity_gradient: torch.Tensor
    kl: torch.Tensor
    disparity_per_view: torch.Tensor
    disparity_valid_count: torch.Tensor
    gradient_per_view: torch.Tensor
    gradient_valid_count: torch.Tensor


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


def _reduce_per_view(
    elementwise_loss: torch.Tensor, valid_mask: torch.Tensor
) -> MaskedViewLoss:
    reduction_dims = (0, 2, 3, 4, 5)
    valid_count = valid_mask.sum(dim=reduction_dims)
    if torch.any(valid_count == 0):
        empty_views = torch.nonzero(valid_count == 0, as_tuple=False).flatten()
        raise ValueError(
            "every view must contain valid supervision; empty view indices: "
            f"{empty_views.detach().cpu().tolist()}"
        )
    loss_sum = elementwise_loss.masked_fill(~valid_mask, 0).sum(
        dim=reduction_dims
    )
    per_view = loss_sum / valid_count.to(elementwise_loss.dtype)
    return MaskedViewLoss(
        loss=per_view.mean(), per_view=per_view, valid_count=valid_count
    )


def _safe_masked_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep invalid target values out of loss kernels and their backward pass."""

    return torch.where(valid_mask, target, prediction.detach())


def masked_smooth_l1_disparity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta: float,
) -> MaskedViewLoss:
    """SmoothL1 on normalized disparity, normalized per view before averaging."""

    _validate_dense_map("prediction", prediction)
    _validate_dense_map("target", target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target disparity shapes must match")
    _validate_mask(prediction, valid_mask)
    if beta <= 0:
        raise ValueError("SmoothL1 beta must be positive")
    if not torch.isfinite(prediction).all():
        raise ValueError("predicted normalized disparity contains NaN/Inf")
    if not torch.isfinite(target.masked_select(valid_mask)).all():
        raise ValueError("valid target disparity contains NaN/Inf")

    safe_target = _safe_masked_target(prediction, target, valid_mask)
    elementwise = F.smooth_l1_loss(
        prediction, safe_target, reduction="none", beta=beta
    )
    return _reduce_per_view(elementwise, valid_mask)


def masked_disparity_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    scale_px: float,
) -> MaskedViewLoss:
    """Pixel-disparity gradient L1 with an explicit target-derived scale."""

    _validate_dense_map("prediction", prediction)
    _validate_dense_map("target", target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target disparity shapes must match")
    _validate_mask(prediction, valid_mask)
    if scale_px <= 0:
        raise ValueError("gradient scale_px must be positive")
    if prediction.shape[-2] < 2 or prediction.shape[-1] < 2:
        raise ValueError("gradient loss requires H>=2 and W>=2")
    if not torch.isfinite(prediction).all():
        raise ValueError("predicted normalized disparity contains NaN/Inf")
    if not torch.isfinite(target.masked_select(valid_mask)).all():
        raise ValueError("valid target disparity contains NaN/Inf")

    safe_target = _safe_masked_target(prediction, target, valid_mask)
    prediction_dx = (prediction[..., 1:] - prediction[..., :-1]) / scale_px
    target_dx = (safe_target[..., 1:] - safe_target[..., :-1]) / scale_px
    mask_dx = valid_mask[..., 1:] & valid_mask[..., :-1]

    prediction_dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) / scale_px
    target_dy = (safe_target[..., 1:, :] - safe_target[..., :-1, :]) / scale_px
    mask_dy = valid_mask[..., 1:, :] & valid_mask[..., :-1, :]

    reduction_dims = (0, 2, 3, 4, 5)
    count_x = mask_dx.sum(dim=reduction_dims)
    count_y = mask_dy.sum(dim=reduction_dims)
    valid_count = count_x + count_y
    if torch.any(valid_count == 0):
        empty_views = torch.nonzero(valid_count == 0, as_tuple=False).flatten()
        raise ValueError(
            "every view must contain valid gradient pairs; empty view indices: "
            f"{empty_views.detach().cpu().tolist()}"
        )

    loss_x = (prediction_dx - target_dx).abs().masked_fill(~mask_dx, 0).sum(
        dim=reduction_dims
    )
    loss_y = (prediction_dy - target_dy).abs().masked_fill(~mask_dy, 0).sum(
        dim=reduction_dims
    )
    per_view = (loss_x + loss_y) / valid_count.to(prediction.dtype)
    return MaskedViewLoss(
        loss=per_view.mean(), per_view=per_view, valid_count=valid_count
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
    """Core RGB/disparity/gradient/KL objective with explicit weights.

    LPIPS, GAN, and feature matching remain separate training-stage terms so
    their activation gates cannot be hidden in this deterministic core loss.
    """

    def __init__(
        self,
        *,
        rgb_weight: float,
        disparity_weight: float,
        gradient_weight: float,
        kl_weight: float,
        smooth_l1_beta: float,
        rgb_loss_type: Literal["l1", "l2"],
    ) -> None:
        super().__init__()
        weights = (rgb_weight, disparity_weight, gradient_weight, kl_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        self.rgb_weight = rgb_weight
        self.disparity_weight = disparity_weight
        self.gradient_weight = gradient_weight
        self.kl_weight = kl_weight
        self.smooth_l1_beta = smooth_l1_beta
        self.rgb_loss_type = rgb_loss_type

    def forward(
        self,
        *,
        rgb_prediction: torch.Tensor,
        rgb_target: torch.Tensor,
        normalized_disparity_prediction: torch.Tensor,
        normalized_disparity_target: torch.Tensor,
        pixel_disparity_prediction: torch.Tensor,
        pixel_disparity_target: torch.Tensor,
        valid_mask: torch.Tensor,
        posterior: PosteriorWithKL,
        gradient_scale_px: float,
        kl_weight_override: float | None = None,
    ) -> StereoLossBreakdown:
        rgb = rgb_reconstruction_loss(
            rgb_prediction, rgb_target, loss_type=self.rgb_loss_type
        )
        disparity = masked_smooth_l1_disparity_loss(
            normalized_disparity_prediction,
            normalized_disparity_target,
            valid_mask,
            beta=self.smooth_l1_beta,
        )
        gradient = masked_disparity_gradient_loss(
            pixel_disparity_prediction,
            pixel_disparity_target,
            valid_mask,
            scale_px=gradient_scale_px,
        )
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
            + self.disparity_weight * disparity.loss
            + self.gradient_weight * gradient.loss
            + kl_weight * kl
        )
        return StereoLossBreakdown(
            total=total,
            rgb=rgb,
            disparity=disparity.loss,
            disparity_gradient=gradient.loss,
            kl=kl,
            disparity_per_view=disparity.per_view,
            disparity_valid_count=disparity.valid_count,
            gradient_per_view=gradient.per_view,
            gradient_valid_count=gradient.valid_count,
        )
