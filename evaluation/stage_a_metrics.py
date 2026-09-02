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


def _content_crop(
    target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop [C,T,H,W] tensors to one strict rectangular valid region."""

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
    target_frames = target[:, :, y0:y1, x0:x1].permute(1, 0, 2, 3)
    prediction_frames = prediction[:, :, y0:y1, x0:x1].permute(1, 0, 2, 3)
    return target_frames, prediction_frames


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


def _teacher_target(batch: dict[str, Any], epsilon: float):
    valid = batch.get("valid_mask")
    if valid is None:
        return None
    if "da3_relative_depth" in batch:
        return relative_target_from_da3(
            batch["da3_relative_depth"], valid, epsilon=epsilon
        )
    if "disparity" in batch:
        # Canonical-v3 currently publishes no stereo calibration.  Centering
        # inverse disparity with unit scale yields a clearly labelled
        # LAS2-H-relative agreement metric, never physical depth accuracy.
        shape = batch["disparity"].shape[:2]
        unit = torch.ones(shape, device=valid.device, dtype=torch.float32)
        return relative_target_from_foundation_stereo(
            batch["disparity"], valid, unit, unit, epsilon=epsilon
        )
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
                "value_count": 0,
                "abs_gt_one_count": 0,
                "raw_min": float("inf"),
                "raw_max": float("-inf"),
                "invalid_sample_count": 0,
            }
        )
        self.abis: dict[str, set[tuple[Any, ...]]] = defaultdict(set)

    @torch.inference_mode()
    def update(
        self,
        mode_id: str,
        batch: dict[str, Any],
        output: Any,
        view_names: tuple[str, ...],
        perceptual_model: Any,
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

        health = self.health[mode_id]
        health["nan_count"] += int(torch.isnan(prediction).sum().item())
        health["inf_count"] += int(torch.isinf(prediction).sum().item())
        health["value_count"] += prediction.numel()
        health["abs_gt_one_count"] += int((prediction.abs() > 1).sum().item())
        finite = torch.isfinite(prediction)
        health["invalid_sample_count"] += int(
            (~finite.flatten(1).all(dim=1)).sum().item()
        )
        if finite.any():
            health["raw_min"] = min(
                health["raw_min"], float(prediction[finite].min().item())
            )
            health["raw_max"] = max(
                health["raw_max"], float(prediction[finite].max().item())
            )
        if not finite.all():
            raise ValueError(f"{mode_id}: reconstruction contains NaN or Inf")

        teacher_target = _teacher_target(batch, self.relative_depth_epsilon)
        teacher_prediction = None
        if teacher_target is not None:
            teacher_prediction, _ = relative_prediction_from_raw(
                output.raw_relative_log_depth, teacher_target.valid_mask
            )

        sample_ids = list(batch["sample_id"])
        for batch_index, sample_id in enumerate(sample_ids):
            record = {"sample_id": str(sample_id), "views": {}}
            for view_index, view_name in enumerate(view_names):
                target_view = target[batch_index, view_index]
                prediction_view = prediction[batch_index, view_index]
                mask_view = mask[batch_index, view_index, 0]
                expanded = mask_view.unsqueeze(0).expand_as(target_view)
                error = prediction_view - target_view
                valid_count = int(expanded.sum().item())
                if valid_count == 0:
                    raise ValueError(f"{mode_id}/{view_name}: empty RGB mask")
                mse = error.square().masked_select(expanded).mean()
                target_crop, prediction_crop = _content_crop(
                    target_view, prediction_view, mask_view
                )
                values = {
                    "rgb_l1": float(
                        error.abs().masked_select(expanded).mean().item()
                    ),
                    "rgb_mse": float(mse.item()),
                    "psnr_db": float(
                        (-10.0 * torch.log10(mse.clamp_min(1e-12))).item()
                    ),
                    "ssim": _ssim_mean(prediction_crop, target_crop),
                    "lpips": _lpips_mean(
                        perceptual_model, prediction_crop, target_crop
                    ),
                    "rgb_valid_values": valid_count,
                }
                if target_view.shape[1] > 1:
                    target_delta = target_view[:, 1:] - target_view[:, :-1]
                    prediction_delta = (
                        prediction_view[:, 1:] - prediction_view[:, :-1]
                    )
                    delta_mask = mask_view[1:] & mask_view[:-1]
                    delta_expanded = delta_mask.unsqueeze(0).expand_as(target_delta)
                    values["temporal_delta_l1"] = float(
                        (prediction_delta - target_delta)
                        .abs()
                        .masked_select(delta_expanded)
                        .mean()
                        .item()
                    )
                    target_delta_crop = target_crop[1:] - target_crop[:-1]
                    prediction_delta_crop = (
                        prediction_crop[1:] - prediction_crop[:-1]
                    )
                    values["temporal_delta_lpips"] = float(
                        perceptual_model(
                            prediction_delta_crop.float(), target_delta_crop.float()
                        )
                        .float()
                        .mean()
                        .item()
                    )
                if teacher_target is not None:
                    valid = teacher_target.valid_mask[batch_index, view_index]
                    difference = (
                        teacher_prediction[batch_index, view_index]
                        - teacher_target.relative_log_depth[batch_index, view_index]
                    )
                    selected = difference.masked_select(valid)
                    if selected.numel() == 0:
                        raise ValueError(f"{mode_id}/{view_name}: no teacher pixels")
                    values.update(
                        {
                            "teacher_relative_log_l1": float(
                                selected.abs().mean().item()
                            ),
                            "teacher_relative_log_rmse": float(
                                selected.square().mean().sqrt().item()
                            ),
                            "teacher_relative_log_silog": float(
                                (
                                    selected.square().mean()
                                    - selected.mean().square()
                                )
                                .clamp_min(0)
                                .sqrt()
                                .item()
                            ),
                            "teacher_valid_pixels": int(selected.numel()),
                        }
                    )
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
            name
            for name in records[0]["views"][view_names[0]]
            if not name.endswith("_pixels") and not name.endswith("_values")
        )
        per_view = {
            view: {
                name: _summary(
                    [record["views"][view][name] for record in records]
                )
                for name in metric_names
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
                ]
            )
            for name in metric_names
        }
        health = dict(self.health[mode_id])
        health["abs_gt_one_ratio"] = (
            health.pop("abs_gt_one_count") / health["value_count"]
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
                    record["views"][view].get("teacher_valid_pixels", 0)
                    for record in records
                    for view in view_names
                )
            ),
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
