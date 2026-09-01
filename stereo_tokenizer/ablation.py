"""Deterministic stereo-input ablations and paired engineering statistics."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


BASE_CONDITIONS = (
    "real_stereo",
    "copy_left",
    "fusion_off",
    "wrong_right",
    "shift_right",
    "time_reverse",
)


@dataclass(frozen=True)
class AblationCondition:
    name: str
    kind: str
    shift_px: int = 0
    fusion_scale_override: float | None = None


def expand_conditions(names, right_shifts):
    """Expand the requested CLI names into a deterministic execution order."""

    requested = [str(name).strip().lower() for name in names]
    unknown = sorted(set(requested) - set(BASE_CONDITIONS))
    if unknown:
        raise ValueError(f"unsupported ablation conditions: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("ablation conditions must be unique")
    shifts = tuple(int(value) for value in right_shifts)
    if 0 in shifts:
        raise ValueError("zero shift is REAL_STEREO, not a SHIFT_RIGHT condition")
    if len(set(shifts)) != len(shifts):
        raise ValueError("right shifts must be unique")

    ordered = []
    for name in requested:
        if name == "shift_right":
            ordered.extend(
                AblationCondition(
                    name=f"shift_right_{shift:+d}",
                    kind=name,
                    shift_px=shift,
                )
                for shift in sorted(shifts)
            )
        elif name == "fusion_off":
            ordered.append(
                AblationCondition(
                    name=name,
                    kind=name,
                    fusion_scale_override=0.0,
                )
            )
        else:
            ordered.append(AblationCondition(name=name, kind=name))
    if not ordered or ordered[0].name != "real_stereo":
        raise ValueError("REAL_STEREO must be requested first")
    return tuple(ordered)


def zero_fill_horizontal_shift(value: torch.Tensor, shift_px: int) -> torch.Tensor:
    """Shift image content horizontally without wrap-around."""

    shift_px = int(shift_px)
    width = value.shape[-1]
    if abs(shift_px) >= width:
        raise ValueError("absolute right-image shift must be smaller than width")
    if shift_px == 0:
        return value.clone()
    shifted = torch.zeros_like(value)
    if shift_px > 0:
        shifted[..., shift_px:] = value[..., : width - shift_px]
    else:
        amount = -shift_px
        shifted[..., : width - amount] = value[..., amount:]
    return shifted


def apply_student_condition(batch, condition: AblationCondition):
    """Return a shallow batch copy whose teacher tensors remain untouched."""

    result = dict(batch)
    video = batch["video"]
    if video.ndim != 7 or video.shape[2] != 2:
        raise ValueError("stereo ablation requires [B,V,2,C,T,H,W]")
    if condition.kind in {"real_stereo", "fusion_off"}:
        return result

    changed = video.clone()
    if condition.kind == "copy_left":
        changed[:, :, 1] = changed[:, :, 0]
    elif condition.kind == "wrong_right":
        wrong = batch.get("wrong_right_video")
        if wrong is None:
            raise ValueError("WRONG_RIGHT requires wrong_right_video in the batch")
        if wrong.shape != changed[:, :, 1].shape:
            raise ValueError("wrong right image shape disagrees with student video")
        changed[:, :, 1] = wrong
    elif condition.kind == "shift_right":
        changed[:, :, 1] = zero_fill_horizontal_shift(
            changed[:, :, 1], condition.shift_px
        )
    elif condition.kind == "time_reverse":
        if changed.shape[4] != 4:
            raise ValueError("TIME_REVERSE is only defined for four-frame inputs")
        changed[:, :, 1] = torch.flip(changed[:, :, 1], dims=(3,))
    else:
        raise ValueError(f"unsupported condition kind {condition.kind!r}")
    result["video"] = changed
    return result


def tensor_payload_sha256(sample_ids, tensors) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        encoded = str(sample_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def teacher_target_checksum(batch) -> str:
    required = ("disparity", "valid_mask")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"teacher checksum missing tensors: {missing}")
    tensors = {name: batch[name] for name in required}
    for name in ("fx", "baseline_m"):
        if name in batch:
            tensors[name] = batch[name]
    return tensor_payload_sha256(batch["sample_id"], tensors)


def _center_per_sample_view(value, valid):
    reduction_dims = (2, 3, 4, 5)
    count = valid.sum(dim=reduction_dims, keepdim=True)
    if torch.any(count == 0):
        raise ValueError("every sample/view needs valid teacher geometry")
    safe = value.float().masked_fill(~valid, 0)
    return value.float() - safe.sum(dim=reduction_dims, keepdim=True) / count


def scale_free_relative_pair(batch, output, epsilon):
    """Build per-view relative log depth without unavailable camera scale.

    Canonical-v3 publishes rectified RGB but no per-episode fx/baseline.  By
    centering each sample/view independently, the unknown positive camera
    scale cancels exactly while spatial and temporal geometry remain testable.
    """

    disparity = batch["disparity"].float()
    valid = batch["valid_mask"]
    safe = torch.where(
        valid,
        disparity.clamp_min(float(epsilon)),
        torch.ones((), device=disparity.device),
    )
    target = _center_per_sample_view(-safe.log(), valid)
    prediction = _center_per_sample_view(
        output.raw_relative_log_depth.float(), valid
    )
    return prediction, target, valid


def _masked_sample_view_mean(value, valid):
    dims = (2, 3, 4, 5)
    count = valid.sum(dim=dims).clamp_min(1)
    return value.masked_fill(~valid, 0).sum(dim=dims) / count


def _spearman(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2:
        return float("nan")
    first_rank = np.argsort(np.argsort(first)).astype(np.float64)
    second_rank = np.argsort(np.argsort(second)).astype(np.float64)
    if first_rank.std() == 0 or second_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _fusion_diagnostics(batch, output, patch_size):
    batch_size = batch["video"].shape[0]
    views = batch["video"].shape[1]
    names = (
        "fusion_confidence_mean",
        "fusion_attention_entropy",
        "fusion_boundary_offset_rate",
        "fusion_offset_teacher_mae",
        "fusion_offset_teacher_spearman",
    )
    result = {
        name: np.full((batch_size, views), np.nan, dtype=np.float64)
        for name in names
    }
    fusion = output.fusion
    if fusion is None:
        return result
    attention = fusion.attention.detach().float()
    confidence = fusion.confidence.detach().float()
    result["fusion_confidence_mean"] = (
        confidence.mean(dim=(2, 3, 4)).cpu().numpy()
    )
    probability = attention.clamp_min(torch.finfo(attention.dtype).tiny)
    entropy = -(attention * probability.log()).sum(dim=-1)
    valid_count = fusion.valid_mask.sum(dim=-1).clamp_min(1)
    normalized = entropy / valid_count.clamp_min(2).log().to(entropy.dtype)[
        None, :, None, None, :, None
    ]
    normalized = torch.where(
        (valid_count == 1)[None, :, None, None, :, None],
        torch.zeros_like(normalized),
        normalized,
    )
    result["fusion_attention_entropy"] = (
        normalized.mean(dim=(2, 3, 4, 5)).cpu().numpy()
    )
    last_valid = (valid_count - 1)[None, :, None, None, :, None]
    result["fusion_boundary_offset_rate"] = (
        (attention.argmax(dim=-1) == last_valid)
        .float()
        .mean(dim=(2, 3, 4, 5))
        .cpu()
        .numpy()
    )

    offsets = torch.arange(attention.shape[-1], device=attention.device)
    expected = (attention * offsets).sum(dim=-1).mean(dim=-1)
    disparity = batch["disparity"][:, :, 0].float()
    teacher_valid = batch["valid_mask"][:, :, 0].float()
    source_height, source_width = disparity.shape[-2:]
    grid_height, grid_width = expected.shape[-2:]
    flattened_disparity = disparity.reshape(-1, 1, source_height, source_width)
    flattened_valid = teacher_valid.reshape(-1, 1, source_height, source_width)
    teacher = torch.nn.functional.interpolate(
        flattened_disparity,
        size=(grid_height, grid_width),
        mode="area",
    ).reshape(*disparity.shape[:3], grid_height, grid_width)
    valid = (
        torch.nn.functional.interpolate(
            flattened_valid,
            size=(grid_height, grid_width),
            mode="area",
        ).reshape(*teacher_valid.shape[:3], grid_height, grid_width)
        >= 0.5
    )
    teacher = teacher / float(patch_size)
    for sample_index in range(batch_size):
        for view_index in range(views):
            mask = valid[sample_index, view_index]
            predicted_values = expected[sample_index, view_index][mask]
            teacher_values = teacher[sample_index, view_index][mask]
            if not len(predicted_values):
                continue
            result["fusion_offset_teacher_mae"][sample_index, view_index] = float(
                (predicted_values - teacher_values).abs().mean().item()
            )
            result["fusion_offset_teacher_spearman"][
                sample_index, view_index
            ] = _spearman(
                predicted_values.cpu().numpy(),
                teacher_values.cpu().numpy(),
            )
    return result


def paired_sample_records(
    batch,
    output,
    *,
    condition,
    mode_id,
    view_names,
    epsilon,
    teacher_checksum,
    real_latent=None,
    fusion_alpha=float("nan"),
    patch_size=16,
):
    prediction, target, valid = scale_free_relative_pair(batch, output, epsilon)
    error = prediction - target
    l1 = _masked_sample_view_mean(error.abs(), valid)
    rmse = _masked_sample_view_mean(error.square(), valid).sqrt()
    mean_error = _masked_sample_view_mean(error, valid)
    silog = (
        _masked_sample_view_mean(error.square(), valid) - mean_error.square()
    ).clamp_min(0).sqrt()
    rgb_target = batch["video"][:, :, 0].float()
    rgb_error = output.rgb.float() - rgb_target
    rgb_l1 = rgb_error.abs().mean(dim=(2, 3, 4, 5))
    rgb_mse = rgb_error.square().mean(dim=(2, 3, 4, 5))
    if rgb_target.shape[3] > 1:
        target_delta = rgb_target[:, :, :, 1:] - rgb_target[:, :, :, :-1]
        prediction_delta = (
            output.rgb.float()[:, :, :, 1:] - output.rgb.float()[:, :, :, :-1]
        )
        temporal_delta = (
            (prediction_delta - target_delta).abs().mean(dim=(2, 3, 4, 5))
        )
    else:
        temporal_delta = None
    fusion_diagnostics = _fusion_diagnostics(batch, output, patch_size)

    if real_latent is None:
        latent_l2 = torch.zeros(len(l1), device=l1.device)
        latent_cosine = torch.ones(len(l1), device=l1.device)
    else:
        current = output.latent.float().flatten(1)
        reference = real_latent.float().flatten(1)
        latent_l2 = (current - reference).square().mean(dim=1).sqrt()
        latent_cosine = torch.nn.functional.cosine_similarity(
            current, reference, dim=1
        )

    records = []
    for sample_index, sample_id in enumerate(batch["sample_id"]):
        for view_index, view_name in enumerate(view_names):
            records.append(
                {
                    "sample_id": str(sample_id),
                    "episode_id": str(batch["episode_id"][sample_index]),
                    "condition": condition.name,
                    "mode_id": mode_id,
                    "view": view_name,
                    "teacher_checksum": teacher_checksum,
                    "relative_log_l1": float(l1[sample_index, view_index].item()),
                    "relative_log_rmse": float(
                        rmse[sample_index, view_index].item()
                    ),
                    "relative_log_silog": float(
                        silog[sample_index, view_index].item()
                    ),
                    "rgb_l1": float(rgb_l1[sample_index, view_index].item()),
                    "rgb_psnr_db": float(
                        -10.0
                        * torch.log10(
                            rgb_mse[sample_index, view_index].clamp_min(1e-12)
                        ).item()
                    ),
                    "temporal_delta_l1": (
                        None
                        if temporal_delta is None
                        else float(
                            temporal_delta[sample_index, view_index].item()
                        )
                    ),
                    "latent_l2": float(latent_l2[sample_index].item()),
                    "latent_cosine": float(latent_cosine[sample_index].item()),
                    "fusion_alpha": float(fusion_alpha),
                    **{
                        name: _finite_or_none(
                            values[sample_index, view_index]
                        )
                        for name, values in fusion_diagnostics.items()
                    },
                }
            )
    return records


def summarize_records(records):
    groups = {}
    metric_names = (
        "relative_log_l1",
        "relative_log_rmse",
        "relative_log_silog",
        "rgb_l1",
        "rgb_psnr_db",
        "temporal_delta_l1",
        "latent_l2",
        "latent_cosine",
        "fusion_alpha",
        "fusion_confidence_mean",
        "fusion_attention_entropy",
        "fusion_boundary_offset_rate",
        "fusion_offset_teacher_mae",
        "fusion_offset_teacher_spearman",
    )
    for record in records:
        key = (record["condition"], record["mode_id"], record["view"])
        group = groups.setdefault(key, {name: [] for name in metric_names})
        for name in metric_names:
            value = record[name]
            if value is not None and np.isfinite(float(value)):
                group[name].append(float(value))
    return [
        {
            "condition": key[0],
            "mode_id": key[1],
            "view": key[2],
            "sample_count": len(values["relative_log_l1"]),
            **{
                name: (
                    float(np.mean(metric_values)) if metric_values else None
                )
                for name, metric_values in values.items()
            },
        }
        for key, values in sorted(groups.items())
    ]


def _bootstrap_seed(seed, *parts):
    payload = ":".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _paired_bootstrap_entry(records, condition, mode_id, view, iterations, seed):
    selected = [
        record
        for record in records
        if record["condition"] in {"real_stereo", condition}
        and (mode_id == "all" or record["mode_id"] == mode_id)
        and (view == "macro" or record["view"] == view)
    ]
    pairs = {}
    for record in selected:
        identity = (record["episode_id"], record["sample_id"], record["mode_id"], record["view"])
        pairs.setdefault(identity, {})[record["condition"]] = float(
            record["relative_log_l1"]
        )
    complete = [value for value in pairs.values() if len(value) == 2]
    if not complete:
        raise ValueError(f"no paired records for {condition}/{mode_id}/{view}")
    by_episode = {}
    for identity, values in pairs.items():
        if len(values) != 2:
            continue
        by_episode.setdefault(identity[0], []).append(
            (values["real_stereo"], values[condition])
        )
    episode_values = np.asarray(
        [
            (
                np.mean([pair[0] for pair in values]),
                np.mean([pair[1] for pair in values]),
            )
            for _, values in sorted(by_episode.items())
        ],
        dtype=np.float64,
    )
    real_mean = float(episode_values[:, 0].mean())
    condition_mean = float(episode_values[:, 1].mean())
    point = (condition_mean - real_mean) / max(condition_mean, 1e-12) * 100.0
    rng = np.random.default_rng(
        _bootstrap_seed(seed, condition, mode_id, view)
    )
    samples = rng.integers(
        0, len(episode_values), size=(int(iterations), len(episode_values))
    )
    sampled = episode_values[samples].mean(axis=1)
    gains = (
        (sampled[:, 1] - sampled[:, 0])
        / np.maximum(sampled[:, 1], 1e-12)
        * 100.0
    )
    return {
        "condition": condition,
        "mode_id": mode_id,
        "view": view,
        "episode_count": len(episode_values),
        "paired_value_count": len(complete),
        "real_l1": real_mean,
        "condition_l1": condition_mean,
        "stereo_gain_percent": point,
        "ci95_low": float(np.quantile(gains, 0.025)),
        "ci95_high": float(np.quantile(gains, 0.975)),
        "episode_win_rate": float(
            np.mean(episode_values[:, 0] < episode_values[:, 1])
        ),
    }


def paired_bootstrap(records, iterations=10_000, seed=1234):
    conditions = sorted(
        {record["condition"] for record in records} - {"real_stereo"}
    )
    modes = sorted({record["mode_id"] for record in records})
    views = sorted({record["view"] for record in records})
    entries = []
    for condition in conditions:
        entries.append(
            _paired_bootstrap_entry(
                records, condition, "all", "macro", iterations, seed
            )
        )
        for view in views:
            entries.append(
                _paired_bootstrap_entry(
                    records, condition, "all", view, iterations, seed
                )
            )
        for mode_id in modes:
            entries.append(
                _paired_bootstrap_entry(
                    records, condition, mode_id, "macro", iterations, seed
                )
            )
    return {
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster": "episode_id",
        "entries": entries,
    }


def decide_stereo_effect(bootstrap):
    entries = {
        (entry["condition"], entry["mode_id"], entry["view"]): entry
        for entry in bootstrap["entries"]
    }
    reasons = []
    for condition in ("copy_left", "fusion_off"):
        entry = entries.get((condition, "all", "macro"))
        if entry is None:
            reasons.append(f"missing {condition}")
        elif entry["stereo_gain_percent"] < 5.0 or entry["ci95_low"] <= 0.0:
            reasons.append(f"{condition} does not clear the 5%/CI gate")
    wrong = entries.get(("wrong_right", "all", "macro"))
    if wrong is None or wrong["ci95_low"] <= 0.0:
        reasons.append("wrong_right is not significantly worse")

    views = sorted(
        {
            key[2]
            for key in entries
            if key[0] in {"copy_left", "fusion_off"} and key[2] != "macro"
        }
    )
    view_gains = []
    for view in views:
        values = [
            entries[(condition, "all", view)]["stereo_gain_percent"]
            for condition in ("copy_left", "fusion_off")
            if (condition, "all", view) in entries
        ]
        if values:
            view_gains.append((view, min(values)))
    if sum(gain >= 5.0 for _, gain in view_gains) < 2:
        reasons.append("fewer than two views clear 5%")
    if any(gain < -2.0 for _, gain in view_gains):
        reasons.append("at least one view regresses by more than 2%")

    shift_entries = {
        int(condition.rsplit("_", 1)[1]): entry
        for (condition, mode, view), entry in entries.items()
        if condition.startswith("shift_right_")
        and mode == "all"
        and view == "macro"
    }
    if shift_entries:
        real_l1 = next(iter(shift_entries.values()))["real_l1"]
        if any(entry["condition_l1"] < real_l1 for entry in shift_entries.values()):
            reasons.append("zero shift is not the minimum")
        by_abs = {}
        for shift, entry in shift_entries.items():
            by_abs.setdefault(abs(shift), []).append(entry["condition_l1"])
        if 16 in by_abs and 32 in by_abs:
            if np.mean(by_abs[32]) < np.mean(by_abs[16]):
                reasons.append("|32px| is unexpectedly better than |16px|")
    else:
        reasons.append("missing shift response")
    return {
        "status": "pass" if not reasons else "inconclusive",
        "stereo_effective": not reasons,
        "reasons": reasons,
        "recommendation": (
            "continue_using_stereo_fusion"
            if not reasons
            else "run_paired_short_training_after_data_review"
        ),
        "thresholds": {
            "minimum_gain_percent": 5.0,
            "maximum_view_regression_percent": 2.0,
            "ci_rule": "95% lower bound > 0",
        },
    }


def write_paired_csv(path, records):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fields = list(records[0]) if records else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _shift_svg(bootstrap):
    points = []
    for entry in bootstrap["entries"]:
        condition = entry["condition"]
        if (
            condition.startswith("shift_right_")
            and entry["mode_id"] == "all"
            and entry["view"] == "macro"
        ):
            points.append(
                (
                    int(condition.rsplit("_", 1)[1]),
                    entry["condition_l1"],
                )
            )
    if not points:
        return "<p>无 shift 数据。</p>"
    real = next(
        entry["real_l1"]
        for entry in bootstrap["entries"]
        if entry["condition"].startswith("shift_right_")
        and entry["mode_id"] == "all"
        and entry["view"] == "macro"
    )
    points.append((0, real))
    points.sort()
    width, height, margin = 680, 250, 42
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = max((y_max - y_min) * 0.1, 1e-6)
    y_min, y_max = y_min - y_pad, y_max + y_pad
    coords = []
    for x_value, y_value in points:
        x = margin + (x_value - x_min) / max(x_max - x_min, 1) * (
            width - 2 * margin
        )
        y = height - margin - (y_value - y_min) / (y_max - y_min) * (
            height - 2 * margin
        )
        coords.append((x, y, x_value, y_value))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in coords)
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4"/>'
        f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle">'
        f'{x_value:+d}: {y_value:.4f}</text>'
        for x, y, x_value, y_value in coords
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="right shift response">'
        f'<polyline points="{polyline}" fill="none" stroke="#3b82f6" '
        'stroke-width="3"/>'
        f'<g fill="#0f172a" font-size="11">{circles}</g></svg>'
    )


def _visual_cards(paths):
    cards = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        cards.append(
            '<figure><img loading="lazy" src="data:'
            f'{mime};base64,{encoded}" alt="{html.escape(path.name)}">'
            f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
        )
    return "".join(cards) or "<p>本次运行未生成样例图。</p>"


def _format_metric(value, digits):
    if value is None or not np.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_html(metrics, bootstrap, decision, provenance, visual_paths=()):
    summary = metrics["summary"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['condition'])}</td>"
        f"<td>{html.escape(row['mode_id'])}</td>"
        f"<td>{html.escape(row['view'])}</td>"
        f"<td>{row['sample_count']}</td>"
        f"<td>{_format_metric(row['relative_log_l1'], 6)}</td>"
        f"<td>{_format_metric(row['relative_log_rmse'], 6)}</td>"
        f"<td>{_format_metric(row['relative_log_silog'], 6)}</td>"
        f"<td>{_format_metric(row['rgb_l1'], 6)}</td>"
        f"<td>{_format_metric(row['rgb_psnr_db'], 3)}</td>"
        f"<td>{_format_metric(row['temporal_delta_l1'], 6)}</td>"
        f"<td>{_format_metric(row['latent_l2'], 5)}</td>"
        f"<td>{_format_metric(row['latent_cosine'], 5)}</td>"
        f"<td>{_format_metric(row['fusion_alpha'], 5)}</td>"
        f"<td>{_format_metric(row['fusion_confidence_mean'], 4)}</td>"
        f"<td>{_format_metric(row['fusion_attention_entropy'], 4)}</td>"
        f"<td>{_format_metric(row['fusion_boundary_offset_rate'], 4)}</td>"
        f"<td>{_format_metric(row['fusion_offset_teacher_mae'], 4)}</td>"
        f"<td>{_format_metric(row['fusion_offset_teacher_spearman'], 4)}</td>"
        "</tr>"
        for row in summary
    )
    aggregate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(mode_id.split('/', 1)[0])}</td>"
        f"<td>{html.escape(mode_id.split('/', 1)[1])}</td>"
        f"<td>{_format_metric(values.get('rgb_l1'), 6)}</td>"
        f"<td>{_format_metric(values.get('rgb_psnr_db_frame_mean'), 3)}</td>"
        f"<td>{_format_metric(values.get('rgb_ssim_frame_mean'), 4)}</td>"
        f"<td>{_format_metric(values.get('rgb_lpips_frame_mean'), 4)}</td>"
        f"<td>{_format_metric(values.get('temporal_delta_l1'), 6)}</td>"
        f"<td>{_format_metric(values.get('temporal_delta_lpips_frame_mean'), 4)}</td>"
        "</tr>"
        for mode_id, values in metrics["evaluation"].get("modes", {}).items()
    )
    bootstrap_rows = "".join(
        "<tr>"
        f"<td>{html.escape(entry['condition'])}</td>"
        f"<td>{html.escape(entry['mode_id'])}</td>"
        f"<td>{html.escape(entry['view'])}</td>"
        f"<td>{entry['stereo_gain_percent']:.2f}%</td>"
        f"<td>[{entry['ci95_low']:.2f}, {entry['ci95_high']:.2f}]</td>"
        f"<td>{entry['episode_win_rate'] * 100:.1f}%</td>"
        "</tr>"
        for entry in bootstrap["entries"]
        if entry["view"] == "macro" or entry["mode_id"] == "all"
    )
    reason_text = (
        "全部工程门禁通过。"
        if not decision["reasons"]
        else "；".join(html.escape(reason) for reason in decision["reasons"])
    )
    status_class = "pass" if decision["status"] == "pass" else "warn"
    provenance_json = html.escape(
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False)
    )
    gpu_hours = (
        float(provenance.get("elapsed_seconds", 0.0))
        * int(provenance.get("world_size", 1))
        / 3600.0
    )
    resource_cards = "".join(
        f'<div class="card"><div class="muted">{label}</div><strong>{value}</strong></div>'
        for label, value in (
            ("GPU hours", _format_metric(gpu_hours, 3)),
            (
                "Student forwards/s",
                _format_metric(provenance.get("student_forwards_per_second"), 3),
            ),
            (
                "Peak CUDA GiB",
                _format_metric(
                    float(provenance.get("peak_cuda_memory_bytes", 0)) / 2**30,
                    2,
                ),
            ),
            ("Parameters", f"{int(provenance.get('parameter_count', 0)):,}"),
            ("Slurm job", html.escape(str(provenance.get("slurm_job_id") or "N/A"))),
            ("World size", int(provenance.get("world_size", 1))),
        )
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H100 Stereo Tokenizer Ablation</title>
<style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#dbe2ea;--bg:#f8fafc;--card:#fff;
--blue:#2563eb;--pass:#047857;--warn:#b45309}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,
-apple-system,"Segoe UI",sans-serif}}main{{max-width:1200px;margin:auto;padding:32px}}
h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:32px}}.muted{{color:var(--muted)}}
.hero,.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:20px;box-shadow:0 3px 12px #0f172a0a}}.status{{display:inline-block;
padding:5px 10px;border-radius:999px;color:white;font-weight:700}}
.status.pass{{background:var(--pass)}}.status.warn{{background:var(--warn)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:9px 10px;
border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),
td:nth-child(3){{text-align:left}}.scroll{{overflow:auto;border:1px solid var(--line);
border-radius:12px}}svg{{width:100%;background:white;border-radius:12px}}
.visuals{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}
figure{{margin:0;background:white;border:1px solid var(--line);padding:8px;
border-radius:10px}}figure img{{width:100%;display:block}}figcaption{{padding-top:6px;
color:var(--muted)}}pre{{overflow:auto;background:#0b1220;color:#dbeafe;padding:16px;
border-radius:10px}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
</style></head><body><main>
<section class="hero"><div class="status {status_class}">{html.escape(decision['status'].upper())}</div>
<h1>H100 双目 Tokenizer 消融汇报</h1>
<p>{reason_text}</p>
<p class="muted">主指标为每个样本、每个视角独立去尺度的 relative-log-depth。
canonical-v3 未发布 fx/baseline，因此本报告不声明 metric depth、EPE 或 D1。</p></section>
<h2>运行资源</h2><div class="grid">{resource_cards}</div>
<h2>配对统计</h2><div class="scroll"><table><thead><tr><th>Condition</th>
<th>Mode</th><th>View</th><th>StereoGain</th><th>95% CI</th><th>Episode win</th>
</tr></thead><tbody>{bootstrap_rows}</tbody></table></div>
<h2>Shift response</h2>{_shift_svg(bootstrap)}
<h2>RGB 与时序汇总</h2><div class="scroll"><table><thead><tr>
<th>Condition</th><th>Mode</th><th>RGB L1</th><th>PSNR</th><th>SSIM</th>
<th>LPIPS</th><th>Temporal delta L1</th><th>Temporal delta LPIPS</th>
</tr></thead><tbody>{aggregate_rows}</tbody></table></div>
<h2>完整指标</h2><div class="scroll"><table><thead><tr><th>Condition</th>
<th>Mode</th><th>View</th><th>N</th><th>Rel-log L1</th><th>RMSE</th>
<th>SILog</th><th>RGB L1</th><th>RGB PSNR</th><th>Temporal delta</th>
<th>Latent L2</th><th>Latent cosine</th><th>Alpha</th><th>Fusion confidence</th>
<th>Attention entropy</th><th>Boundary offset</th><th>Offset MAE</th><th>Offset Spearman</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h2>固定样例</h2><div class="visuals">{_visual_cards(visual_paths)}</div>
<h2>可复现信息</h2><pre><code>{provenance_json}</code></pre>
</main></body></html>"""


def write_ablation_report(
    report_dir,
    *,
    records,
    evaluation_result,
    provenance,
    visual_paths=(),
    bootstrap_iterations=10_000,
    seed=1234,
):
    report_dir = Path(report_dir)
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite report directory {report_dir}")
    report_dir.mkdir(parents=True)
    summary = summarize_records(records)
    bootstrap = paired_bootstrap(
        records, iterations=bootstrap_iterations, seed=seed
    )
    decision = decide_stereo_effect(bootstrap)
    metrics = {
        "schema": "stereo-input-ablation-metrics-v1",
        "metric_contract": "per_sample_per_view_scale_free_relative_log",
        "evaluation": evaluation_result,
        "summary": summary,
        "decision": decision,
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "config.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_paired_csv(report_dir / "paired_samples.csv", records)
    (report_dir / "index.html").write_text(
        render_html(metrics, bootstrap, decision, provenance, visual_paths),
        encoding="utf-8",
    )
    return decision
