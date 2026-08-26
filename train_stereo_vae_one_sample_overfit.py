"""Task-scoped RGB-only one-sample overfit diagnostic for StereoVAE.

This entrypoint intentionally bypasses the formal online-teacher trainer.  It is
kept separate so the diagnostic can be reverted without loosening production
data, teacher, batch-size, or four-mode scheduling contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from stereo_tokenizer.data import HyMonoDataset
from stereo_tokenizer.lerobot_data import LeRobotStereoDataset
from stereo_tokenizer.mode_sampling import MODE_IDS
from stereo_tokenizer.model import StereoVAE


SCHEMA = "stereo-vae-one-sample-rgb-overfit-v1"
DEFAULT_MILESTONES = (100, 500, 1000, 2000, 4000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*command: str) -> str:
    return subprocess.run(
        ["git", *command], check=True, capture_output=True, text=True
    ).stdout.strip()


def _model_args(max_steps: int) -> SimpleNamespace:
    """Return the exact architecture and disabled-loss diagnostic contract."""

    return SimpleNamespace(
        activation_in_disc="leaky_relu",
        apply_diffaug=False,
        apply_noise=False,
        attn_dropout=0.0,
        batch_size=1,
        causal_in_peg=True,
        causal_in_temporal_transformer=False,
        dec_block="tttt",
        defer_spatial_pool=False,
        defer_temporal_pool=False,
        devices=1,
        dim_head=64,
        disc_channels=64,
        disc_layers=3,
        disc_loss_type="hinge",
        discriminator_iter_start=max_steps + 1,
        embedding_dim=512,
        enc_block="ttww",
        ff_dropout=0.0,
        ff_mult=4.0,
        four_mode_mixed_training=False,
        gan_enabled=False,
        gan_feat_weight=0.0,
        grad_accumulates=1,
        grad_clip_val=1.0,
        grad_clip_val_disc=1.0,
        heads=8,
        image_channels=3,
        image_gan_weight=0.0,
        initialize_vit=True,
        kl_warmup_steps=0,
        kl_weight=0.0,
        latent_channels=48,
        max_steps=max_steps,
        mode_schedule_seed=1234,
        norm_type="group",
        num_nodes=1,
        patch_embed="linear",
        patch_size=16,
        peg_backend="conv2d_t1_slice",
        perceptual_weight=0.0,
        recon_loss_type="l1",
        relative_depth_epsilon=1e-6,
        relative_depth_weight=0.0,
        relative_gradient_weight=0.0,
        resolution=256,
        rgb_weight=1.0,
        sigmoid_in_disc=False,
        single_frame_source_index=0,
        smooth_l1_beta=1.0,
        spatial_depth=4,
        spatial_pos="rope",
        stereo_num_frames=4,
        stereo_num_views=3,
        stereo_search_direction="left",
        stereo_search_radii=[7, 7, 7],
        temporal_depth=4,
        twod_window_size=8,
        video_gan_weight=0.0,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _generator_parameters(model: StereoVAE):
    return (
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.posterior_projection.parameters())
        + list(model.latent_projection.parameters())
    )


def _validate_milestones(mode: str, max_steps: int, milestones: tuple[int, ...]):
    if not milestones or tuple(sorted(set(milestones))) != milestones:
        raise ValueError("milestones must be unique and strictly increasing")
    if milestones[-1] > (max_steps // 4 if mode == "joint" else max_steps):
        raise ValueError("a per-mode milestone exceeds the available updates")


def mode_for_step(mode: str, step: int) -> str:
    if step < 1:
        raise ValueError("step is one-based")
    if mode == "joint":
        return MODE_IDS[(step - 1) % len(MODE_IDS)]
    if mode not in MODE_IDS:
        raise ValueError(f"unsupported mode {mode!r}")
    return mode


def _load_samples(args) -> dict[str, dict]:
    mono = HyMonoDataset(
        args.mono_manifest,
        args.mono_cache_root,
        single_frame_source_index=0,
    )
    stereo = LeRobotStereoDataset(
        args.stereo_manifest,
        args.stereo_dataset_root,
        split="train",
        expected_rectification_audit_sha256=args.rectification_audit_sha256,
        video_cache_capacity=2,
        maximum_timestamp_error_s=0.05,
        single_frame_source_index=0,
    )
    samples = {}
    for mode_id in MODE_IDS:
        eye_mode, temporal_mode = mode_id.split("/", maxsplit=1)
        dataset = mono if eye_mode == "mono" else stereo
        index = args.mono_index if eye_mode == "mono" else args.stereo_index
        sample = dataset.get_mode_item(index, temporal_mode)
        video = sample["video"].contiguous().pin_memory()
        samples[mode_id] = {
            "video": video,
            "sample_id": sample["sample_id"],
            "episode_id": sample["episode_id"],
            "eye_mode": eye_mode,
            "temporal_mode": temporal_mode,
            "source_index": index,
        }
    return samples


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach().float().cpu().add(0.5).clamp(0, 1)
        .mul(255).round().byte().permute(1, 2, 0).numpy()
    )


def _save_comparison(
    path: Path, source: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor
) -> None:
    # source [V,E,C,T,H,W], target/prediction [V,C,T,H,W]
    views, eyes, _, frames, height, width = source.shape
    row_names = ["GT left", "reconstruction"]
    if eyes == 2:
        row_names.append("input right")
    label_width = 120
    header_height = 24
    canvas = Image.new(
        "RGB",
        (label_width + views * frames * width, header_height + len(row_names) * height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    for column in range(views * frames):
        view, frame = divmod(column, frames)
        draw.text(
            (label_width + column * width + 4, 4),
            f"view={view} frame={frame}",
            fill=(0, 0, 0),
        )
    for row, name in enumerate(row_names):
        draw.text((4, header_height + row * height + 4), name, fill=(0, 0, 0))
        for view in range(views):
            for frame in range(frames):
                if row == 0:
                    image = target[view, :, frame]
                elif row == 1:
                    image = prediction[view, :, frame]
                else:
                    image = source[view, 1, :, frame]
                canvas.paste(
                    Image.fromarray(_to_uint8(image)),
                    (label_width + (view * frames + frame) * width, header_height + row * height),
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def reconstruction_metrics(
    source: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor
) -> dict:
    error = prediction.float() - target.float()
    mse = error.square().mean()
    mae = error.abs().mean()
    result = {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "psnr_db": float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item()),
        "per_view_mae": [
            float(error[view].abs().mean().item()) for view in range(target.shape[0])
        ],
        "output_target_mapping": "reconstruction view v targets source left eye [v,0]",
    }
    if target.shape[2] > 1:
        result["gt_adjacent_frame_mae"] = [
            float((target[:, :, frame + 1] - target[:, :, frame]).abs().mean().item())
            for frame in range(target.shape[2] - 1)
        ]
        result["prediction_adjacent_frame_mae"] = [
            float(
                (prediction[:, :, frame + 1] - prediction[:, :, frame])
                .abs().mean().item()
            )
            for frame in range(prediction.shape[2] - 1)
        ]
    if source.shape[1] == 2:
        result["input_left_right_mae_per_view_frame"] = [
            [
                float(
                    (source[view, 0, :, frame] - source[view, 1, :, frame])
                    .abs().mean().item()
                )
                for frame in range(source.shape[3])
            ]
            for view in range(source.shape[0])
        ]
    return result


@torch.no_grad()
def _evaluate(model: StereoVAE, sample: dict, device: torch.device):
    model.eval()
    source = sample["video"]
    video = source.unsqueeze(0).to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            video,
            eye_mode=sample["eye_mode"],
            temporal_mode=sample["temporal_mode"],
            sample_posterior=False,
        )
    target = video[:, :, 0]
    return (
        source,
        target[0].detach().cpu(),
        output.rgb[0].detach().cpu(),
    )


def create_initial_checkpoint(args) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)
    model_args = _model_args(max_steps=16000)
    model = StereoVAE(model_args)
    payload = {
        "schema": SCHEMA,
        "seed": args.seed,
        "model_args": vars(model_args),
        "state_dict": model.state_dict(),
        "git_branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
    }
    torch.save(payload, output)
    print(json.dumps({"path": str(output), "sha256": _sha256(output)}), flush=True)


def train(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    max_steps = 16000 if args.mode == "joint" else 4000
    milestones = tuple(args.milestones)
    _validate_milestones(args.mode, max_steps, milestones)
    if len(args.initial_checkpoint_sha256) != 64:
        raise ValueError("initial checkpoint SHA256 must be a full digest")
    checkpoint_path = Path(args.initial_checkpoint).resolve()
    observed_sha = _sha256(checkpoint_path)
    if observed_sha != args.initial_checkpoint_sha256:
        raise ValueError("initial checkpoint SHA256 mismatch")

    _seed_everything(args.seed)
    model_args = _model_args(max_steps=max_steps)
    model = StereoVAE(model_args)
    initial = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if initial.get("schema") != SCHEMA or initial.get("seed") != args.seed:
        raise ValueError("initial checkpoint contract mismatch")
    model.load_state_dict(initial["state_dict"], strict=True)
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.Adam(_generator_parameters(model), lr=args.lr, betas=(0.5, 0.9))
    samples = _load_samples(args)
    selected_modes = MODE_IDS if args.mode == "joint" else (args.mode,)
    samples = {mode_id: samples[mode_id] for mode_id in selected_modes}

    provenance = {
        "schema": SCHEMA,
        "status": "running",
        "git_branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
        "mode": args.mode,
        "joint_mode_order": list(MODE_IDS) if args.mode == "joint" else None,
        "max_steps": max_steps,
        "per_mode_milestones": list(milestones),
        "joint_total_step_milestones": [value * 4 for value in milestones],
        "seed": args.seed,
        "batch_size": 1,
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": [0.5, 0.9]},
        "precision": "bf16 autocast",
        "sample_posterior_train": False,
        "sample_posterior_eval": False,
        "augmentation": "none; tensors decoded once and frozen",
        "losses": {
            "rgb_l1": 1.0,
            "lpips": 0.0,
            "kl": 0.0,
            "relative_depth": 0.0,
            "relative_gradient": 0.0,
            "image_gan": 0.0,
            "video_gan": 0.0,
            "feature_matching": 0.0,
        },
        "online_teacher": False,
        "initial_checkpoint": str(checkpoint_path),
        "initial_checkpoint_sha256": observed_sha,
        "model_args": vars(model_args),
        "mono_manifest": str(Path(args.mono_manifest).resolve()),
        "mono_manifest_sha256": _sha256(Path(args.mono_manifest).resolve()),
        "mono_cache_root": str(Path(args.mono_cache_root).resolve()),
        "mono_index": args.mono_index,
        "stereo_manifest": str(Path(args.stereo_manifest).resolve()),
        "stereo_manifest_sha256": _sha256(Path(args.stereo_manifest).resolve()),
        "stereo_dataset_root": str(Path(args.stereo_dataset_root).resolve()),
        "stereo_index": args.stereo_index,
        "rectification_audit_sha256": args.rectification_audit_sha256,
        "samples": {
            mode_id: {
                key: value
                for key, value in sample.items()
                if key != "video"
            }
            for mode_id, sample in samples.items()
        },
    }
    config_path = output / "resolved_config.json"
    config_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    mode_updates = {mode_id: 0 for mode_id in MODE_IDS}
    metrics_path = output / "metrics.jsonl"
    summary = {"status": "running"}
    try:
        with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_stream:
            for mode_id, sample in samples.items():
                source, target, prediction = _evaluate(model, sample, device)
                initial_metrics = reconstruction_metrics(source, target, prediction)
                initial_metrics.update({"kind": "milestone", "step": 0, "mode_id": mode_id, "mode_update": 0})
                metrics_stream.write(json.dumps(initial_metrics, sort_keys=True) + "\n")
                _save_comparison(output / "reconstructions" / mode_id.replace("/", "_") / "step_000000.png", source, target, prediction)

            for step in range(1, max_steps + 1):
                mode_id = mode_for_step(args.mode, step)
                sample = samples[mode_id]
                video = sample["video"].unsqueeze(0).to(device, non_blocking=True)
                target = video[:, :, 0]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model(
                        video,
                        eye_mode=sample["eye_mode"],
                        temporal_mode=sample["temporal_mode"],
                        sample_posterior=False,
                    ).rgb
                    loss = F.l1_loss(prediction.float(), target.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_generator_parameters(model), 1.0)
                optimizer.step()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                mode_updates[mode_id] += 1
                metrics_stream.write(json.dumps({
                    "kind": "train",
                    "step": step,
                    "mode_id": mode_id,
                    "mode_update": mode_updates[mode_id],
                    "rgb_l1": float(loss.detach().item()),
                    "step_time_s": elapsed,
                    "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }, sort_keys=True) + "\n")

                if args.mode == "joint":
                    milestone = mode_updates[MODE_IDS[-1]]
                    milestone_due = (
                        step % len(MODE_IDS) == 0
                        and milestone in milestones
                        and len(set(mode_updates.values())) == 1
                    )
                    modes_to_evaluate = MODE_IDS if milestone_due else ()
                else:
                    milestone = mode_updates[mode_id]
                    milestone_due = milestone in milestones
                    modes_to_evaluate = (mode_id,) if milestone_due else ()
                if milestone_due:
                    for eval_mode in modes_to_evaluate:
                        eval_sample = samples[eval_mode]
                        source, eval_target, eval_prediction = _evaluate(
                            model, eval_sample, device
                        )
                        result = reconstruction_metrics(
                            source, eval_target, eval_prediction
                        )
                        result.update(
                            {
                                "kind": "milestone",
                                "step": step,
                                "mode_id": eval_mode,
                                "mode_update": mode_updates[eval_mode],
                            }
                        )
                        metrics_stream.write(
                            json.dumps(result, sort_keys=True) + "\n"
                        )
                        stem = (
                            f"mode_{mode_updates[eval_mode]:06d}_"
                            f"total_{step:06d}"
                        )
                        _save_comparison(
                            output
                            / "reconstructions"
                            / eval_mode.replace("/", "_")
                            / f"{stem}.png",
                            source,
                            eval_target,
                            eval_prediction,
                        )
                    checkpoint_stem = f"mode_{milestone:06d}_total_{step:06d}"
                    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "schema": SCHEMA,
                            "git_sha": provenance["git_sha"],
                            "mode": args.mode,
                            "step": step,
                            "mode_updates": dict(mode_updates),
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "initial_checkpoint_sha256": observed_sha,
                        },
                        output / "checkpoints" / f"{checkpoint_stem}.pt",
                    )

        summary = {
            "status": "complete",
            "max_steps": max_steps,
            "mode_updates": mode_updates,
            "metrics": str(metrics_path),
        }
    finally:
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        provenance["status"] = summary["status"]
        config_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--output", required=True)
    initialize.add_argument("--seed", type=int, default=1234)

    run = subparsers.add_parser("train")
    run.add_argument("--mode", choices=(*MODE_IDS, "joint"), required=True)
    run.add_argument("--output_dir", required=True)
    run.add_argument("--initial_checkpoint", required=True)
    run.add_argument("--initial_checkpoint_sha256", required=True)
    run.add_argument("--mono_manifest", required=True)
    run.add_argument("--mono_cache_root", required=True)
    run.add_argument("--mono_index", type=int, required=True)
    run.add_argument("--stereo_manifest", required=True)
    run.add_argument("--stereo_dataset_root", required=True)
    run.add_argument("--stereo_index", type=int, required=True)
    run.add_argument("--rectification_audit_sha256", required=True)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--lr", type=float, default=1e-4)
    run.add_argument("--milestones", type=int, nargs="+", default=DEFAULT_MILESTONES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "initialize":
        create_initial_checkpoint(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
