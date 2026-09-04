"""Mask-aware sample distributions for Stereo Tokenizer Stage A1."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from stereo_tokenizer.modules.relative_depth import (
    relative_prediction_from_raw,
    relative_target_from_da3,
    relative_target_from_foundation_stereo,
)


PERCENTILES = (0.50, 0.90, 0.99)
RGB_MIN = -0.5
RGB_MAX = 0.5
STATIC_FLOW_MAX_PX = 0.5
DYNAMIC_FLOW_MIN_PX = 1.0
FLOW_FB_RELATIVE_THRESHOLD = 0.01
FLOW_FB_ABSOLUTE_THRESHOLD_PX2 = 0.5


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("sample metrics must be non-empty and finite")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        **{
            f"p{int(quantile * 100):02d}": float(np.quantile(array, quantile))
            for quantile in PERCENTILES
        },
    }


def _content_bounds(mask: torch.Tensor) -> tuple[int, int, int, int]:
    if mask.dtype != torch.bool or mask.ndim != 3:
        raise TypeError("RGB mask must be bool [T,H,W]")
    if not torch.equal(mask, mask[0:1].expand_as(mask)):
        raise ValueError("Stage A SSIM/LPIPS requires a time-invariant pixel mask")
    spatial = mask[0]
    positions = torch.nonzero(spatial, as_tuple=False)
    if positions.numel() == 0:
        raise ValueError("RGB valid mask is empty")
    y0, x0 = positions.min(dim=0).values.tolist()
    y1, x1 = (positions.max(dim=0).values + 1).tolist()
    rectangle = torch.zeros_like(spatial)
    rectangle[y0:y1, x0:x1] = True
    if not torch.equal(spatial, rectangle):
        raise ValueError("Stage A SSIM/LPIPS requires a rectangular content mask")
    return int(y0), int(x0), int(y1), int(x1)


def _content_crop(
    target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop [C,T,H,W] tensors to one strict rectangular valid region."""

    y0, x0, y1, x1 = _content_bounds(mask)
    target_frames = target[:, :, y0:y1, x0:x1].permute(1, 0, 2, 3)
    prediction_frames = prediction[:, :, y0:y1, x0:x1].permute(1, 0, 2, 3)
    return target_frames, prediction_frames


def _warp_with_flow(
    source: torch.Tensor, flow_to_source: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``source`` at pixel coordinates displaced by ``flow_to_source``."""

    if source.ndim != 4 or flow_to_source.ndim != 4:
        raise ValueError("flow warp requires source [N,C,H,W] and flow [N,2,H,W]")
    if source.shape[0] != flow_to_source.shape[0] or source.shape[-2:] != flow_to_source.shape[-2:]:
        raise ValueError("flow warp source/flow shapes disagree")
    if flow_to_source.shape[1] != 2:
        raise ValueError("optical flow must have two channels")
    count, _, height, width = source.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=flow_to_source.dtype),
        torch.arange(width, device=source.device, dtype=flow_to_source.dtype),
        indexing="ij",
    )
    sample_x = x.unsqueeze(0) + flow_to_source[:, 0]
    sample_y = y.unsqueeze(0) + flow_to_source[:, 1]
    in_bounds = (
        (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
    )
    if width == 1 or height == 1:
        raise ValueError("flow warp requires spatial dimensions greater than one")
    grid = torch.stack(
        (
            sample_x.mul(2.0 / (width - 1)).sub(1.0),
            sample_y.mul(2.0 / (height - 1)).sub(1.0),
        ),
        dim=-1,
    )
    warped = torch.nn.functional.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    if tuple(warped.shape) != tuple(source.shape) or tuple(in_bounds.shape) != (
        count,
        height,
        width,
    ):
        raise RuntimeError("flow warp produced an invalid shape")
    return warped, in_bounds


def _flow_consistency_mask(
    flow: torch.Tensor, reverse_flow: torch.Tensor
) -> torch.Tensor:
    """Forward/backward consistency in the coordinate domain of ``flow``."""

    sampled_reverse, in_bounds = _warp_with_flow(reverse_flow, flow)
    residual_sq = (flow + sampled_reverse).square().sum(dim=1)
    scale_sq = flow.square().sum(dim=1) + sampled_reverse.square().sum(dim=1)
    threshold = (
        FLOW_FB_RELATIVE_THRESHOLD * scale_sq + FLOW_FB_ABSOLUTE_THRESHOLD_PX2
    )
    return in_bounds & torch.isfinite(residual_sq) & (residual_sq <= threshold)


def _flow_infer(flow_model: Any, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    flow = flow_model(first, second)
    if not isinstance(flow, torch.Tensor) or tuple(flow.shape) != (
        first.shape[0],
        2,
        first.shape[-2],
        first.shape[-1],
    ):
        raise ValueError("frozen flow model returned an invalid tensor")
    if not torch.isfinite(flow).all():
        raise ValueError("frozen flow model returned NaN/Inf")
    return flow.float()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float | None:
    expanded = mask.unsqueeze(1).expand_as(value)
    selected = value.masked_select(expanded)
    return None if selected.numel() == 0 else float(selected.mean().item())


def _temporal_flow_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    flow_model: Any,
) -> tuple[dict[str, float | int], dict[str, torch.Tensor]]:
    """Compute reference-flow temporal metrics on cropped [T,3,H,W] RGB."""

    if target.shape != prediction.shape or target.ndim != 4 or target.shape[0] < 2:
        raise ValueError("temporal flow metrics require matching [T,3,H,W] clips")
    return _temporal_flow_metrics_batch(
        target.unsqueeze(0), prediction.unsqueeze(0), flow_model
    )[0]


def _temporal_flow_metrics_batch(
    target: torch.Tensor,
    prediction: torch.Tensor,
    flow_model: Any,
) -> list[tuple[dict[str, float | int], dict[str, torch.Tensor]]]:
    """Vectorized flow inference for matching [N,T,3,H,W] content crops."""

    if target.shape != prediction.shape or target.ndim != 5 or target.shape[1] < 2:
        raise ValueError("batched temporal flow requires matching [N,T,3,H,W]")
    count, frames, channels, height, width = target.shape
    if channels != 3:
        raise ValueError("batched temporal flow requires RGB clips")
    pairs = frames - 1
    reference = target.add(0.5).clamp(0, 1)
    reconstruction = prediction.add(0.5).clamp(0, 1)

    def flatten_pairs(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            first.reshape(count * pairs, channels, height, width),
            second.reshape(count * pairs, channels, height, width),
        )

    reference_first, reference_second = flatten_pairs(
        reference[:, :-1], reference[:, 1:]
    )
    reconstruction_first, reconstruction_second = flatten_pairs(
        reconstruction[:, :-1], reconstruction[:, 1:]
    )
    forward = _flow_infer(flow_model, reference_first, reference_second)
    backward = _flow_infer(flow_model, reference_second, reference_first)
    reconstruction_forward = _flow_infer(
        flow_model, reconstruction_first, reconstruction_second
    )
    forward_valid = _flow_consistency_mask(forward, backward)
    backward_valid = _flow_consistency_mask(backward, forward)

    target_first, target_second = flatten_pairs(target[:, :-1], target[:, 1:])
    prediction_first, prediction_second = flatten_pairs(
        prediction[:, :-1], prediction[:, 1:]
    )
    warped_target, target_in_bounds = _warp_with_flow(target_first, backward)
    warped_prediction, prediction_in_bounds = _warp_with_flow(prediction_first, backward)
    backward_valid &= target_in_bounds & prediction_in_bounds
    warp_relation_error = (
        (prediction_second - warped_prediction) - (target_second - warped_target)
    ).abs()
    direct_relation_error = (
        (prediction_second - prediction_first) - (target_second - target_first)
    ).abs()
    backward_magnitude = backward.square().sum(dim=1).sqrt()
    forward_magnitude = forward.square().sum(dim=1).sqrt()
    static_mask = backward_valid & (backward_magnitude <= STATIC_FLOW_MAX_PX)
    dynamic_mask = forward_valid & (forward_magnitude >= DYNAMIC_FLOW_MIN_PX)
    flow_epe = (reconstruction_forward - forward).square().sum(dim=1).sqrt()

    reshaped = {
        "forward": forward.reshape(count, pairs, 2, height, width),
        "backward": backward.reshape(count, pairs, 2, height, width),
        "forward_valid": forward_valid.reshape(count, pairs, height, width),
        "backward_valid": backward_valid.reshape(count, pairs, height, width),
        "warp_relation_error": warp_relation_error.reshape(
            count, pairs, channels, height, width
        ),
        "direct_relation_error": direct_relation_error.reshape(
            count, pairs, channels, height, width
        ),
        "static_mask": static_mask.reshape(count, pairs, height, width),
        "dynamic_mask": dynamic_mask.reshape(count, pairs, height, width),
        "flow_epe": flow_epe.reshape(count, pairs, height, width),
    }
    results = []
    for sample_index in range(count):
        values: dict[str, float | int] = {
            "optical_flow_valid_ratio": float(
                reshaped["backward_valid"][sample_index].float().mean().item()
            ),
            "static_flow_valid_ratio": float(
                reshaped["static_mask"][sample_index].float().mean().item()
            ),
            "dynamic_flow_valid_ratio": float(
                reshaped["dynamic_mask"][sample_index].float().mean().item()
            ),
            "optical_flow_valid_pixels": int(
                reshaped["backward_valid"][sample_index].sum().item()
            ),
            "static_flow_valid_pixels": int(
                reshaped["static_mask"][sample_index].sum().item()
            ),
            "dynamic_flow_valid_pixels": int(
                reshaped["dynamic_mask"][sample_index].sum().item()
            ),
        }
        metric_masks = {
            "clamped_optical_flow_warp_l1": (
                reshaped["warp_relation_error"][sample_index],
                reshaped["backward_valid"][sample_index],
            ),
            "clamped_static_flicker_l1": (
                reshaped["direct_relation_error"][sample_index],
                reshaped["static_mask"][sample_index],
            ),
            "clamped_motion_flow_epe_px": (
                reshaped["flow_epe"][sample_index].unsqueeze(1),
                reshaped["dynamic_mask"][sample_index],
            ),
        }
        for name, (error, valid) in metric_masks.items():
            mean = _masked_mean(error, valid)
            if mean is not None:
                values[name] = mean
            for pair_index in range(pairs):
                pair_mean = _masked_mean(
                    error[pair_index : pair_index + 1],
                    valid[pair_index : pair_index + 1],
                )
                if pair_mean is not None:
                    values[f"{name}_pair_{pair_index}{pair_index + 1}"] = pair_mean
        results.append(
            (
                values,
                {
                    name: reshaped[name][sample_index]
                    for name in ("forward", "backward", "forward_valid", "backward_valid")
                },
            )
        )
    return results


def _temporal_geometry_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    target_valid: torch.Tensor,
    prediction_valid: torch.Tensor,
    flow_context: dict[str, torch.Tensor],
    metric_prefix: str,
) -> dict[str, float | int]:
    """Compare flow-aligned relative-log geometry changes across adjacent frames."""

    if target.shape != prediction.shape or target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("temporal geometry requires matching [T,1,H,W] tensors")
    if target_valid.shape != target.shape or prediction_valid.shape != target.shape:
        raise ValueError("temporal geometry masks must match geometry tensors")
    backward = flow_context["backward"]
    warped_target, in_bounds = _warp_with_flow(target[:-1], backward)
    warped_prediction, _ = _warp_with_flow(prediction[:-1], backward)
    warped_target_valid, _ = _warp_with_flow(target_valid[:-1].float(), backward)
    warped_prediction_valid, _ = _warp_with_flow(
        prediction_valid[:-1].float(), backward
    )
    valid = (
        flow_context["backward_valid"]
        & in_bounds
        & target_valid[1:, 0]
        & prediction_valid[1:, 0]
        & (warped_target_valid[:, 0] > 0.999)
        & (warped_prediction_valid[:, 0] > 0.999)
    )
    error = (
        (prediction[1:] - warped_prediction) - (target[1:] - warped_target)
    ).abs()
    name = f"{metric_prefix}_temporal_geometry_warp_l1"
    values: dict[str, float | int] = {
        f"{metric_prefix}_temporal_geometry_valid_ratio": float(
            valid.float().mean().item()
        ),
        f"{metric_prefix}_temporal_geometry_valid_pixels": int(valid.sum().item()),
    }
    mean = _masked_mean(error, valid)
    if mean is not None:
        values[name] = mean
    for pair_index in range(target.shape[0] - 1):
        pair_mean = _masked_mean(
            error[pair_index : pair_index + 1],
            valid[pair_index : pair_index + 1],
        )
        if pair_mean is not None:
            values[f"{name}_pair_{pair_index}{pair_index + 1}"] = pair_mean
    return values


def _lpips_mean(model, prediction: torch.Tensor, target: torch.Tensor) -> float:
    if model is None:
        raise RuntimeError("LPIPS weights are unavailable in the checkpoint")
    value = model(prediction.float() * 2.0, target.float() * 2.0)
    return float(value.float().mean().item())


def _ssim_mean(prediction: torch.Tensor, target: torch.Tensor) -> float:
    value = structural_similarity_index_measure(
        prediction.add(0.5).clamp(0, 1),
        target.add(0.5).clamp(0, 1),
        data_range=1.0,
        reduction="elementwise_mean",
    )
    return float(value.item())



def _rgb_overshoot_stats(
    prediction: torch.Tensor, mask: torch.Tensor
) -> dict[str, float | int]:
    """Return exact per-sample/view overshoot statistics on valid RGB pixels."""

    if prediction.ndim != 4 or prediction.shape[0] != 3:
        raise ValueError("RGB prediction must use [3,T,H,W]")
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(prediction.shape[1:]):
        raise ValueError("RGB overshoot mask must use bool [T,H,W]")
    valid_pixels = int(mask.sum().item())
    if valid_pixels == 0:
        raise ValueError("RGB overshoot mask is empty")
    pixel_overshoot = (prediction.abs() - RGB_MAX).clamp_min(0).amax(dim=0)
    valid_overshoot = pixel_overshoot.masked_select(mask)
    positive = valid_overshoot[valid_overshoot > 0]
    out_of_range_pixels = int(positive.numel())
    if out_of_range_pixels:
        quantiles = torch.quantile(
            positive.float(),
            torch.tensor((0.5, 0.9, 0.99), device=positive.device),
        )
        p50, p90, p99 = (float(value.item()) for value in quantiles)
        maximum = float(positive.max().item())
    else:
        p50 = p90 = p99 = maximum = 0.0
    return {
        "rgb_valid_pixels": valid_pixels,
        "rgb_out_of_range_pixels": out_of_range_pixels,
        "rgb_out_of_range_pixel_ratio": out_of_range_pixels / valid_pixels,
        "rgb_overshoot_positive_pixels": out_of_range_pixels,
        "rgb_overshoot_positive_p50": p50,
        "rgb_overshoot_positive_p90": p90,
        "rgb_overshoot_positive_p99": p99,
        "rgb_overshoot_positive_max": maximum,
    }

def _teacher_targets(batch: dict[str, Any], epsilon: float):
    valid = batch.get("valid_mask")
    if valid is None:
        return None, None, None
    if "da3_relative_depth" in batch:
        target = relative_target_from_da3(
            batch["da3_relative_depth"], valid, epsilon=epsilon
        )
        reconstruction_depth = batch.get("reconstruction_da3_relative_depth")
        reconstruction_valid = batch.get("reconstruction_valid_mask")
        if reconstruction_depth is None or reconstruction_valid is None:
            raise ValueError("mono teacher evaluation requires reconstruction DA3 output")
        reconstruction = relative_target_from_da3(
            reconstruction_depth,
            reconstruction_valid,
            epsilon=epsilon,
        )
        return target, reconstruction, "reconstruction_teacher"
    if "disparity" in batch:
        # Canonical-v3 currently publishes no stereo calibration.  Centering
        # inverse disparity with unit scale yields a clearly labelled
        # depth-head agreement metric, never physical depth accuracy.
        shape = batch["disparity"].shape[:2]
        unit = torch.ones(shape, device=valid.device, dtype=torch.float32)
        target = relative_target_from_foundation_stereo(
            batch["disparity"], valid, unit, unit, epsilon=epsilon
        )
        return target, None, "depth_head_teacher"
    raise ValueError("valid_mask is present without a recognized teacher target")


class StageA1MetricSuite:
    """Collect per-camera, per-mode P50/P90/P99 and output health."""

    def __init__(self, *, relative_depth_epsilon: float):
        self.relative_depth_epsilon = float(relative_depth_epsilon)
        self.samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.health: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "nan_count": 0,
                "inf_count": 0,
                "all_value_count": 0,
                "all_raw_min": float("inf"),
                "all_raw_max": float("-inf"),
                "valid_value_count": 0,
                "valid_raw_min": float("inf"),
                "valid_raw_max": float("-inf"),
                "valid_pixel_count": 0,
                "out_of_range_pixel_count": 0,
                "invalid_sample_count": 0,
                "invalid_sample_ids": [],
            }
        )
        self.abis: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
        self.teacher_invalid: dict[str, list[dict[str, str]]] = defaultdict(list)

    @torch.inference_mode()
    def update(
        self,
        mode_id: str,
        batch: dict[str, Any],
        output: Any,
        view_names: tuple[str, ...],
        perceptual_model: Any,
        flow_model: Any | None = None,
    ) -> None:
        target = batch["video"][:, :, 0].float()
        prediction = output.rgb.float()
        mask = batch.get("rgb_valid_mask")
        expected_mask = (
            target.shape[0], target.shape[1], 1, target.shape[3], *target.shape[-2:]
        )
        if target.shape != prediction.shape:
            raise ValueError(
                f"{mode_id}: target/prediction mismatch "
                f"{tuple(target.shape)} != {tuple(prediction.shape)}"
            )
        if mask is None or tuple(mask.shape) != expected_mask or mask.dtype != torch.bool:
            raise ValueError(f"{mode_id}: invalid RGB mask contract")
        if target.shape[1] != len(view_names):
            raise ValueError(f"{mode_id}: view-name contract mismatch")

        sample_ids = [str(value) for value in batch["sample_id"]]
        health = self.health[mode_id]
        health["nan_count"] += int(torch.isnan(prediction).sum().item())
        health["inf_count"] += int(torch.isinf(prediction).sum().item())
        health["all_value_count"] += prediction.numel()
        finite = torch.isfinite(prediction)
        finite_by_sample = finite.flatten(1).all(dim=1)
        invalid_indices = torch.nonzero(~finite_by_sample, as_tuple=False).flatten()
        health["invalid_sample_count"] += int(invalid_indices.numel())
        health["invalid_sample_ids"].extend(
            sample_ids[int(index)] for index in invalid_indices.tolist()
        )
        if finite.any():
            health["all_raw_min"] = min(
                health["all_raw_min"], float(prediction[finite].min().item())
            )
            health["all_raw_max"] = max(
                health["all_raw_max"], float(prediction[finite].max().item())
            )
        if not finite.all():
            raise ValueError(f"{mode_id}: reconstruction contains NaN or Inf")
        expanded_mask = mask.expand_as(target)
        valid_target = target.masked_select(expanded_mask)
        if (
            not torch.isfinite(valid_target).all()
            or torch.any(valid_target < RGB_MIN)
            or torch.any(valid_target > RGB_MAX)
        ):
            raise ValueError(
                f"{mode_id}: valid RGB target must be finite and normalized to "
                f"[{RGB_MIN},{RGB_MAX}]"
            )
        valid_prediction = prediction.masked_select(expanded_mask)
        health["valid_value_count"] += int(valid_prediction.numel())
        health["valid_raw_min"] = min(
            health["valid_raw_min"], float(valid_prediction.min().item())
        )
        health["valid_raw_max"] = max(
            health["valid_raw_max"], float(valid_prediction.max().item())
        )

        teacher_target, reconstruction_teacher, teacher_kind = _teacher_targets(
            batch, self.relative_depth_epsilon
        )
        teacher_prediction = None
        if teacher_target is not None and reconstruction_teacher is None:
            teacher_prediction, _ = relative_prediction_from_raw(
                output.raw_relative_log_depth, teacher_target.valid_mask
            )

        flow_cache: dict[
            tuple[int, int], tuple[dict[str, float | int], dict[str, torch.Tensor]]
        ] = {}
        if target.shape[3] > 1 and flow_model is not None:
            flow_groups: dict[
                tuple[int, int],
                list[tuple[tuple[int, int], torch.Tensor, torch.Tensor]],
            ] = defaultdict(list)
            for batch_index in range(target.shape[0]):
                for view_index in range(target.shape[1]):
                    target_crop, prediction_crop = _content_crop(
                        target[batch_index, view_index],
                        prediction[batch_index, view_index].clamp(RGB_MIN, RGB_MAX),
                        mask[batch_index, view_index, 0],
                    )
                    flow_groups[tuple(target_crop.shape[-2:])].append(
                        (
                            (batch_index, view_index),
                            target_crop,
                            prediction_crop,
                        )
                    )
            for group in flow_groups.values():
                group_results = _temporal_flow_metrics_batch(
                    torch.stack([entry[1] for entry in group]),
                    torch.stack([entry[2] for entry in group]),
                    flow_model,
                )
                for entry, result in zip(group, group_results, strict=True):
                    flow_cache[entry[0]] = result

        for batch_index, sample_id in enumerate(sample_ids):
            record = {"sample_id": str(sample_id), "views": {}}
            for view_index, view_name in enumerate(view_names):
                target_view = target[batch_index, view_index]
                prediction_view = prediction[batch_index, view_index]
                clamped_prediction_view = prediction_view.clamp(RGB_MIN, RGB_MAX)
                mask_view = mask[batch_index, view_index, 0]
                expanded = mask_view.unsqueeze(0).expand_as(target_view)
                raw_error = prediction_view - target_view
                clamped_error = clamped_prediction_view - target_view
                valid_count = int(expanded.sum().item())
                if valid_count == 0:
                    raise ValueError(f"{mode_id}/{view_name}: empty RGB mask")
                raw_mse = raw_error.square().masked_select(expanded).mean()
                clamped_mse = clamped_error.square().masked_select(expanded).mean()
                target_crop, clamped_prediction_crop = _content_crop(
                    target_view, clamped_prediction_view, mask_view
                )
                values = {
                    "raw_rgb_l1": float(
                        raw_error.abs().masked_select(expanded).mean().item()
                    ),
                    "raw_rgb_mse": float(raw_mse.item()),
                    "clamped_rgb_l1": float(
                        clamped_error.abs().masked_select(expanded).mean().item()
                    ),
                    "clamped_rgb_mse": float(clamped_mse.item()),
                    "clamped_psnr_db": float(
                        (
                            -10.0
                            * torch.log10(clamped_mse.clamp_min(1e-12))
                        ).item()
                    ),
                    "clamped_ssim": _ssim_mean(
                        clamped_prediction_crop, target_crop
                    ),
                    "clamped_lpips": _lpips_mean(
                        perceptual_model, clamped_prediction_crop, target_crop
                    ),
                    "rgb_valid_values": valid_count,
                    "rgb_valid_ratio": valid_count / expanded.numel(),
                    **_rgb_overshoot_stats(prediction_view, mask_view),
                }
                flow_context = None
                if target_view.shape[1] > 1 and flow_model is not None:
                    flow_values, flow_context = flow_cache[(batch_index, view_index)]
                    values.update(flow_values)
                if (
                    values["clamped_rgb_l1"] > values["raw_rgb_l1"] + 1e-8
                    or values["clamped_rgb_mse"] > values["raw_rgb_mse"] + 1e-8
                ):
                    raise RuntimeError("clamped RGB error exceeded raw RGB error")
                health["valid_pixel_count"] += values["rgb_valid_pixels"]
                health["out_of_range_pixel_count"] += values[
                    "rgb_out_of_range_pixels"
                ]
                if target_view.shape[1] > 1:
                    target_delta = target_view[:, 1:] - target_view[:, :-1]
                    prediction_delta = (
                        clamped_prediction_view[:, 1:]
                        - clamped_prediction_view[:, :-1]
                    )
                    delta_mask = mask_view[1:] & mask_view[:-1]
                    delta_expanded = delta_mask.unsqueeze(0).expand_as(target_delta)
                    values["clamped_temporal_delta_l1"] = float(
                        (prediction_delta - target_delta)
                        .abs()
                        .masked_select(delta_expanded)
                        .mean()
                        .item()
                    )
                    target_delta_crop = target_crop[1:] - target_crop[:-1]
                    prediction_delta_crop = (
                        clamped_prediction_crop[1:]
                        - clamped_prediction_crop[:-1]
                    )
                    values["clamped_temporal_delta_lpips"] = float(
                        perceptual_model(
                            prediction_delta_crop.float(), target_delta_crop.float()
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    for pair_index in range(target_delta.shape[1]):
                        pair_mask = delta_mask[pair_index].unsqueeze(0).expand_as(
                            target_delta[:, pair_index]
                        )
                        pair_error = (
                            prediction_delta[:, pair_index]
                            - target_delta[:, pair_index]
                        )
                        pair_name = f"pair_{pair_index}{pair_index + 1}"
                        values[f"clamped_temporal_delta_l1_{pair_name}"] = float(
                            pair_error.abs().masked_select(pair_mask).mean().item()
                        )
                        values[f"clamped_temporal_delta_lpips_{pair_name}"] = float(
                            perceptual_model(
                                prediction_delta_crop[pair_index : pair_index + 1].float(),
                                target_delta_crop[pair_index : pair_index + 1].float(),
                            )
                            .float()
                            .mean()
                            .item()
                        )
                if teacher_target is not None:
                    if reconstruction_teacher is None:
                        target_teacher_valid = teacher_target.valid_mask[
                            batch_index, view_index
                        ]
                        prediction_teacher_valid = target_teacher_valid
                        valid = target_teacher_valid
                        prediction_depth = teacher_prediction[batch_index, view_index]
                    else:
                        target_teacher_valid = teacher_target.valid_mask[
                            batch_index, view_index
                        ]
                        prediction_teacher_valid = reconstruction_teacher.valid_mask[
                            batch_index, view_index
                        ]
                        valid = target_teacher_valid & prediction_teacher_valid
                        prediction_depth = reconstruction_teacher.relative_log_depth[
                            batch_index, view_index
                        ]
                    difference = prediction_depth - teacher_target.relative_log_depth[
                        batch_index, view_index
                    ]
                    selected = difference.masked_select(valid)
                    if selected.numel() == 0:
                        values.update(
                            {
                                f"{teacher_kind}_valid_pixels": 0,
                                f"{teacher_kind}_valid_ratio": 0.0,
                            }
                        )
                        self.teacher_invalid[mode_id].append(
                            {
                                "sample_id": str(sample_id),
                                "view": str(view_name),
                                "reason": "empty_teacher_mask",
                            }
                        )
                    else:
                        values.update(
                            {
                                f"{teacher_kind}_relative_log_l1": float(
                                    selected.abs().mean().item()
                                ),
                                f"{teacher_kind}_relative_log_rmse": float(
                                    selected.square().mean().sqrt().item()
                                ),
                                f"{teacher_kind}_relative_log_silog": float(
                                    (
                                        selected.square().mean()
                                        - selected.mean().square()
                                    )
                                    .clamp_min(0)
                                    .sqrt()
                                    .item()
                                ),
                                f"{teacher_kind}_valid_pixels": int(selected.numel()),
                                f"{teacher_kind}_valid_ratio": (
                                    int(selected.numel()) / int(valid.numel())
                                ),
                            }
                        )
                    if flow_context is not None:
                        y0, x0, y1, x1 = _content_bounds(mask_view)
                        geometry_values = _temporal_geometry_metrics(
                            teacher_target.relative_log_depth[
                                batch_index, view_index, :, :, y0:y1, x0:x1
                            ].permute(1, 0, 2, 3),
                            prediction_depth[:, :, y0:y1, x0:x1].permute(1, 0, 2, 3),
                            target_teacher_valid[
                                :, :, y0:y1, x0:x1
                            ].permute(1, 0, 2, 3),
                            prediction_teacher_valid[
                                :, :, y0:y1, x0:x1
                            ].permute(1, 0, 2, 3),
                            flow_context,
                            teacher_kind,
                        )
                        values.update(geometry_values)
                record["views"][view_name] = values
            self.samples[mode_id].append(record)

        abi = (
            tuple(int(value) for value in batch["video"].shape[1:]),
            tuple(int(value) for value in output.latent.shape[1:]),
            str(batch["video"].dtype),
            str(output.latent.dtype),
        )
        self.abis[mode_id].add(abi)

    def finalize(self, mode_id: str, view_names: tuple[str, ...]) -> dict[str, Any]:
        records = self.samples.get(mode_id, [])
        if not records:
            raise ValueError(f"{mode_id}: no Stage A1 samples")
        metric_names = tuple(
            sorted(
                {
                    name
                    for record in records
                    for view in view_names
                    for name in record["views"][view]
                    if not name.endswith("_pixels")
                    and not name.endswith("_values")
                }
            )
        )
        per_view = {
            view: {
                name: _summary(
                    [
                        record["views"][view][name]
                        for record in records
                        if name in record["views"][view]
                    ]
                )
                for name in metric_names
                if any(name in record["views"][view] for record in records)
            }
            for view in view_names
        }
        per_sample_macro = {
            name: _summary(
                [
                    float(
                        np.mean(
                            [record["views"][view][name] for view in view_names]
                        )
                    )
                    for record in records
                    if all(name in record["views"][view] for view in view_names)
                ]
            )
            for name in metric_names
            if any(
                all(name in record["views"][view] for view in view_names)
                for record in records
            )
        }
        health = dict(self.health[mode_id])
        if health["valid_pixel_count"] <= 0:
            raise ValueError(f"{mode_id}: no valid RGB pixels")
        health["out_of_range_pixel_ratio"] = (
            health["out_of_range_pixel_count"] / health["valid_pixel_count"]
        )
        abi_values = self.abis[mode_id]
        if len(abi_values) != 1:
            raise ValueError(f"{mode_id}: latent ABI changed within one run")
        input_shape, latent_shape, input_dtype, latent_dtype = next(iter(abi_values))
        input_views, input_eyes, _, input_frames, input_h, input_w = input_shape
        latent_views, latent_channels, latent_frames, latent_h, latent_w = latent_shape
        tokens = latent_views * latent_frames * latent_h * latent_w
        return {
            "sample_count": len(records),
            "per_view": per_view,
            "per_sample_macro": per_sample_macro,
            "valid_rgb_values": int(
                sum(
                    record["views"][view]["rgb_valid_values"]
                    for record in records
                    for view in view_names
                )
            ),
            "valid_teacher_pixels": int(
                sum(
                    sum(
                        value
                        for name, value in record["views"][view].items()
                        if name.endswith("_valid_pixels")
                    )
                    for record in records
                    for view in view_names
                )
            ),
            "teacher_invalid_count": len(self.teacher_invalid[mode_id]),
            "teacher_invalid_samples": list(self.teacher_invalid[mode_id]),
            "output_health": health,
            "latent_abi": {
                "input_shape_without_batch": list(input_shape),
                "latent_shape_without_batch": list(latent_shape),
                "input_dtype": input_dtype,
                "latent_dtype": latent_dtype,
                "latent_channels": latent_channels,
                "tokens_per_window": tokens,
                "tokens_per_input_frame": tokens / input_frames,
                "spatial_compression_ratio": (input_h * input_w)
                / (latent_h * latent_w),
                "temporal_compression_ratio": input_frames / latent_frames,
                "view_compression_ratio": (input_views * input_eyes)
                / latent_views,
            },
        }
