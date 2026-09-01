from __future__ import annotations

import math
import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.models.layers import trunc_normal_
from .mode_sampling import (
    MODE_IDS,
    dataset_for_mode_occurrence,
    mode_for_update,
    mode_occurrences_before,
    parse_weight_spec,
    resolve_mode_int_spec,
)
from .modules import (
    LPIPS,
    StereoFusion,
    StereoFusionOutput,
    StereoLossBreakdown,
    StereoReconstructionKLLoss,
    relative_prediction_from_raw,
    relative_target_from_da3,
    relative_target_from_foundation_stereo,
)
from .modules.attention import PEG, Transformer
from .modules.discriminator import NLayerDiscriminator, NLayerDiscriminator3D
from .modules.vae import (
    DiagonalGaussianDistribution,
    StructuredDiagonalGaussianPosterior,
)
from .profiling import profile_region


EyeMode = Literal["mono", "stereo"]
TemporalMode = Literal["single_frame", "four_frame"]


def temporal_mode_num_frames(temporal_mode: TemporalMode) -> int:
    if temporal_mode == "single_frame":
        return 1
    if temporal_mode == "four_frame":
        return 4
    raise ValueError(f"unsupported temporal mode {temporal_mode!r}")


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


@dataclass
class StereoEncodeOutput:
    latent: torch.Tensor
    posterior: StructuredDiagonalGaussianPosterior
    fusion: Optional[StereoFusionOutput]
    eye_mode: EyeMode
    temporal_mode: TemporalMode
    source_num_frames: int


@dataclass
class StereoVAEOutput:
    rgb: torch.Tensor
    raw_relative_log_depth: torch.Tensor
    latent: torch.Tensor
    posterior: StructuredDiagonalGaussianPosterior
    fusion: Optional[StereoFusionOutput]
    eye_mode: EyeMode
    temporal_mode: TemporalMode
    source_num_frames: int


@dataclass
class _StereoCoreLossOutput:
    model: StereoVAEOutput
    loss: StereoLossBreakdown
    relative_log_depth_prediction: torch.Tensor
    relative_log_depth_target: torch.Tensor
    valid_mask: torch.Tensor
    effective_kl_weight: float


class StereoVAE(pl.LightningModule):
    """Continuous variational autoencoder for three synchronized stereo views."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._validate_configuration(args)

        self.latent_channels = args.latent_channels
        self.stereo_num_views = args.stereo_num_views
        self.stereo_num_frames = args.stereo_num_frames
        self.resolution = args.resolution
        self.patch_size = args.patch_size

        self.encoder = StereoEncoder(
            image_size=args.resolution,
            image_channel=args.image_channels,
            block=args.enc_block,
            window_size=args.twod_window_size,
            spatial_pos=args.spatial_pos,
            patch_embed=args.patch_embed,
            patch_size=args.patch_size,
            defer_temporal_pool=args.defer_temporal_pool,
            defer_spatial_pool=args.defer_spatial_pool,
            spatial_depth=args.spatial_depth,
            temporal_depth=args.temporal_depth,
            causal_in_peg=args.causal_in_peg,
            dim=args.embedding_dim,
            dim_head=args.dim_head,
            heads=args.heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            ff_mult=args.ff_mult,
            initialize=args.initialize_vit,
            stereo_num_views=args.stereo_num_views,
            stereo_num_frames=args.stereo_num_frames,
            stereo_search_radii=tuple(args.stereo_search_radii),
            stereo_search_direction=args.stereo_search_direction,
            # 保留原版配置接口；默认值为 False，四帧离线重建仍使用双向时间注意力。
            causal_in_temporal_transformer=args.causal_in_temporal_transformer,
        )
        self.decoder = StereoDecoder(
            image_size=args.resolution,
            image_channel=args.image_channels,
            block=args.dec_block,
            window_size=args.twod_window_size,
            spatial_pos=args.spatial_pos,
            patch_embed=args.patch_embed,
            patch_size=args.patch_size,
            defer_temporal_pool=args.defer_temporal_pool,
            defer_spatial_pool=args.defer_spatial_pool,
            spatial_depth=args.spatial_depth,
            temporal_depth=args.temporal_depth,
            causal_in_peg=args.causal_in_peg,
            dim=args.embedding_dim,
            dim_head=args.dim_head,
            heads=args.heads,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            ff_mult=args.ff_mult,
            gen_upscale=None,
            initialize=args.initialize_vit,
            stereo_num_views=args.stereo_num_views,
            stereo_num_frames=args.stereo_num_frames,
            # 与 Encoder 保持同一配置语义；默认 False 不启用 causal mask。
            causal_in_temporal_transformer=args.causal_in_temporal_transformer,
        )
        self.peg_backend = getattr(args, "peg_backend", "conv3d_contiguous")
        peg_count = 0
        for module in self.modules():
            if isinstance(module, PEG):
                module.set_backend(self.peg_backend)
                peg_count += 1
        if peg_count == 0:
            raise RuntimeError("StereoVAE contains no PEG modules")
        self.posterior_projection = nn.Sequential(
            Rearrange("b c t h w -> b t h w c"),
            nn.Linear(args.embedding_dim, 2 * args.latent_channels),
            Rearrange("b t h w c -> b c t h w"),
        )
        self.latent_projection = nn.Sequential(
            Rearrange("b c t h w -> b t h w c"),
            nn.Linear(args.latent_channels, args.embedding_dim),
            Rearrange("b t h w c -> b c t h w"),
        )

        self.core_objective = StereoReconstructionKLLoss(
            rgb_weight=args.rgb_weight,
            relative_depth_weight=args.relative_depth_weight,
            relative_gradient_weight=args.relative_gradient_weight,
            kl_weight=args.kl_weight,
            smooth_l1_beta=args.smooth_l1_beta,
            rgb_loss_type=args.recon_loss_type,
        )
        self.kl_weight = args.kl_weight
        self.kl_warmup_steps = args.kl_warmup_steps
        self.relative_depth_epsilon = args.relative_depth_epsilon
        self.perceptual_weight = args.perceptual_weight
        self.gan_enabled = args.gan_enabled
        self.image_gan_weight = args.image_gan_weight
        self.video_gan_weight = args.video_gan_weight
        self.gan_feat_weight = args.gan_feat_weight
        self.apply_diffaug = args.apply_diffaug

        if self.perceptual_weight > 0:
            self.perceptual_model = LPIPS().eval()
            self.perceptual_model.requires_grad_(False)
        else:
            self.perceptual_model = None

        self.image_discriminator = None
        self.video_discriminator = None
        if self.gan_enabled:
            discriminator_kwargs = dict(
                input_nc=args.image_channels,
                ndf=args.disc_channels,
                n_layers=args.disc_layers,
                norm_type=args.norm_type,
                use_sigmoid=args.sigmoid_in_disc,
                activation=args.activation_in_disc,
                apply_noise=args.apply_noise,
            )
            if self.image_gan_weight > 0:
                self.image_discriminator = NLayerDiscriminator(
                    **discriminator_kwargs
                )
            if self.video_gan_weight > 0:
                self.video_discriminator = NLayerDiscriminator3D(
                    **discriminator_kwargs
                )
            self.disc_loss = (
                vanilla_d_loss
                if args.disc_loss_type == "vanilla"
                else hinge_d_loss
            )
        else:
            self.disc_loss = None

        self.automatic_optimization = False
        self.grad_accumulates = args.grad_accumulates
        self.mode_batch_sizes = resolve_mode_int_spec(
            getattr(args, "mode_batch_sizes", None),
            fallback=int(args.batch_size),
        )
        self.mode_grad_accumulates = resolve_mode_int_spec(
            getattr(args, "mode_grad_accumulates", None),
            fallback=int(args.grad_accumulates),
        )
        self.grad_clip_val = args.grad_clip_val
        self.grad_clip_val_disc = args.grad_clip_val_disc
        self._micro_step = 0
        self._logical_mode_id = None
        self._logical_dataset_id = None
        self._logical_global_samples = 0
        self.generator_updates = 0
        self.discriminator_updates = 0
        self.four_frame_updates = 0
        self.single_frame_updates = 0
        self.mode_updates = {mode_id: 0 for mode_id in MODE_IDS}
        self.mode_samples = {mode_id: 0 for mode_id in MODE_IDS}
        self.batch_updates = 0
        self.last_temporal_mode: Optional[TemporalMode] = None
        self.last_mode_id: Optional[str] = None
        self.last_dataset_id: Optional[str] = None
        self.last_micro_step_index = 0
        self.last_accumulation_factor = 1
        self.last_microbatch_size = 0
        self.last_logical_global_samples = 0
        self._validation_mode_sums = None
        self._validation_mode_counts = None
        self.save_hyperparameters("args")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.perceptual_model is not None:
            self.perceptual_model.eval()
        return self

    @staticmethod
    def _validate_configuration(args) -> None:
        if args.image_channels != 3:
            raise ValueError("StereoVAE requires image_channels=3")
        if args.stereo_num_views != 3:
            raise ValueError("StereoVAE requires exactly three views")
        if args.stereo_num_frames != 4:
            raise ValueError("StereoVAE requires exactly four frames")
        if not 0 <= args.single_frame_source_index < args.stereo_num_frames:
            raise ValueError(
                "single_frame_source_index must select one of the four source frames"
            )
        if args.latent_channels != 48:
            raise ValueError("StereoVAE latent_channels must be 48")
        if args.patch_embed != "linear":
            raise ValueError("StereoVAE requires linear patch embedding")
        if args.defer_temporal_pool or args.defer_spatial_pool:
            raise ValueError("StereoVAE does not use deferred pooling")
        if len(args.enc_block) != args.spatial_depth:
            raise ValueError("enc_block length must equal spatial_depth")
        if len(args.dec_block) != args.spatial_depth:
            raise ValueError("dec_block length must equal spatial_depth")
        if len(args.stereo_search_radii) != args.stereo_num_views:
            raise ValueError("one stereo search radius is required per view")
        explicit_nonnegative = (
            "rgb_weight",
            "relative_depth_weight",
            "relative_gradient_weight",
            "kl_weight",
            "perceptual_weight",
            "image_gan_weight",
            "video_gan_weight",
            "gan_feat_weight",
        )
        for name in explicit_nonnegative:
            value = getattr(args, name)
            if value is None or value < 0:
                raise ValueError(f"{name} must be explicitly set and non-negative")
        if args.relative_depth_epsilon <= 0:
            raise ValueError("relative_depth_epsilon must be positive")
        if args.grad_accumulates <= 0:
            raise ValueError("grad_accumulates must be positive")
        if args.discriminator_iter_start < 0:
            raise ValueError("discriminator_iter_start must be non-negative")
        if args.gan_enabled and (
            args.image_gan_weight == 0 and args.video_gan_weight == 0
        ):
            raise ValueError("GAN is enabled but both GAN weights are zero")
        if args.gan_enabled and args.discriminator_iter_start >= args.max_steps:
            raise ValueError(
                "GAN activation must precede the final generator update"
            )

    @staticmethod
    def _read_checkpoint_counter(counters: Mapping, name: str) -> int:
        value = counters.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(
                f"checkpoint update counter {name} must be a non-negative integer"
            )
        return value

    def on_save_checkpoint(self, checkpoint) -> None:
        if self._micro_step != 0:
            raise RuntimeError("refusing to checkpoint an incomplete logical update")
        mode_weight_spec = getattr(self.args, "mode_update_weights", "1:1:1:1")
        mono_weight_spec = getattr(self.args, "mono_dataset_weights", "9:1")
        mode_weights = parse_weight_spec(mode_weight_spec, MODE_IDS)
        world_size_contract = int(
            getattr(self.args, "devices", 1) * getattr(self.args, "num_nodes", 1)
        )
        checkpoint["stereo_update_counters"] = {
            "generator_updates": self.generator_updates,
            "discriminator_updates": self.discriminator_updates,
            "four_frame_updates": self.four_frame_updates,
            "single_frame_updates": self.single_frame_updates,
            "mode_updates": dict(self.mode_updates),
            "mode_samples": dict(self.mode_samples),
            "batch_updates": self.batch_updates,
            "mode_contract": list(MODE_IDS),
            "mode_update_weights": mode_weights,
            "mono_dataset_weights": mono_weight_spec,
            "node_manifest_contracts": getattr(
                self.args, "node_manifest_contracts", None
            ),
            "per_device_batch_size": int(self.args.batch_size),
            "grad_accumulates": int(self.grad_accumulates),
            "mode_batch_sizes": dict(self.mode_batch_sizes),
            "mode_grad_accumulates": dict(self.mode_grad_accumulates),
            "mode_effective_global_batch_sizes": {
                mode_id: self.mode_batch_sizes[mode_id]
                * self.mode_grad_accumulates[mode_id]
                * world_size_contract
                for mode_id in MODE_IDS
            },
            "logical_update_contract_version": 1,
            "mode_schedule_seed": int(
                getattr(self.args, "mode_schedule_seed", 1234)
            ),
            "world_size_contract": world_size_contract,
        }

    def on_load_checkpoint(self, checkpoint) -> None:
        counters = checkpoint.get("stereo_update_counters")
        if counters is None:
            raise ValueError(
                "checkpoint has no deterministic temporal-mode update counters"
            )
        if not isinstance(counters, Mapping):
            raise TypeError("stereo_update_counters must be a mapping")

        generator_updates = self._read_checkpoint_counter(
            counters, "generator_updates"
        )
        discriminator_updates = self._read_checkpoint_counter(
            counters, "discriminator_updates"
        )
        four_frame_updates = self._read_checkpoint_counter(
            counters, "four_frame_updates"
        )
        single_frame_updates = self._read_checkpoint_counter(
            counters, "single_frame_updates"
        )
        batch_updates = self._read_checkpoint_counter(counters, "batch_updates")
        if counters.get("mode_contract") != list(MODE_IDS):
            raise ValueError("checkpoint four-mode contract mismatch")
        mode_weight_spec = getattr(self.args, "mode_update_weights", "1:1:1:1")
        mono_weight_spec = getattr(self.args, "mono_dataset_weights", "9:1")
        mode_weights = parse_weight_spec(mode_weight_spec, MODE_IDS)
        if counters.get("mode_update_weights") != mode_weights:
            raise ValueError("checkpoint mode update weights mismatch")
        if counters.get("mono_dataset_weights") != mono_weight_spec:
            raise ValueError("checkpoint mono dataset weights mismatch")
        if counters.get("node_manifest_contracts") != getattr(
            self.args, "node_manifest_contracts", None
        ):
            raise ValueError("checkpoint node-local manifest contract mismatch")
        if counters.get("per_device_batch_size") != int(self.args.batch_size):
            raise ValueError("checkpoint per-device batch size mismatch")
        if counters.get("grad_accumulates") != int(self.grad_accumulates):
            raise ValueError("checkpoint gradient accumulation mismatch")
        if counters.get("logical_update_contract_version") != 1:
            raise ValueError("checkpoint logical-update contract version mismatch")
        if counters.get("mode_batch_sizes") != self.mode_batch_sizes:
            raise ValueError("checkpoint per-mode batch size mismatch")
        if counters.get("mode_grad_accumulates") != self.mode_grad_accumulates:
            raise ValueError("checkpoint per-mode gradient accumulation mismatch")
        schedule_seed = int(getattr(self.args, "mode_schedule_seed", 1234))
        if counters.get("mode_schedule_seed") != schedule_seed:
            raise ValueError("checkpoint mode schedule seed mismatch")
        expected_world_size = int(
            getattr(self.args, "devices", 1)
            * getattr(self.args, "num_nodes", 1)
        )
        if counters.get("world_size_contract") != expected_world_size:
            raise ValueError("checkpoint DDP world-size contract mismatch")
        expected_effective_global_batches = {
            mode_id: self.mode_batch_sizes[mode_id]
            * self.mode_grad_accumulates[mode_id]
            * expected_world_size
            for mode_id in MODE_IDS
        }
        if (
            counters.get("mode_effective_global_batch_sizes")
            != expected_effective_global_batches
        ):
            raise ValueError("checkpoint effective global batch contract mismatch")
        mode_updates = counters.get("mode_updates")
        mode_samples = counters.get("mode_samples")
        if not isinstance(mode_updates, Mapping) or set(mode_updates) != set(MODE_IDS):
            raise ValueError("checkpoint mode update counters mismatch")
        if not isinstance(mode_samples, Mapping) or set(mode_samples) != set(MODE_IDS):
            raise ValueError("checkpoint mode sample counters mismatch")
        mode_updates = {
            mode_id: self._read_checkpoint_counter(mode_updates, mode_id)
            for mode_id in MODE_IDS
        }
        mode_samples = {
            mode_id: self._read_checkpoint_counter(mode_samples, mode_id)
            for mode_id in MODE_IDS
        }
        if discriminator_updates > generator_updates:
            raise ValueError("discriminator updates cannot exceed generator updates")
        if not self.gan_enabled and discriminator_updates != 0:
            raise ValueError("GAN-disabled checkpoint has discriminator updates")
        if bool(getattr(self.args, "four_mode_mixed_training", False)):
            expected_batch_updates = sum(
                mode_updates_count * self.mode_grad_accumulates[mode_id]
                for mode_id, mode_updates_count in mode_updates.items()
            )
            if batch_updates != expected_batch_updates:
                raise ValueError(
                    "batch updates disagree with per-mode accumulation contract"
                )
        elif batch_updates < generator_updates * self.grad_accumulates:
            raise ValueError("batch updates are inconsistent with generator updates")
        if generator_updates != sum(mode_updates.values()):
            raise ValueError("generator updates must equal the four mode counters")
        if generator_updates != four_frame_updates + single_frame_updates:
            raise ValueError(
                "generator updates must equal four-frame plus single-frame updates"
            )
        if four_frame_updates != (
            mode_updates["mono/four_frame"]
            + mode_updates["stereo/four_frame"]
        ):
            raise ValueError("four-frame counter disagrees with mode counters")
        if single_frame_updates != (
            mode_updates["mono/single_frame"]
            + mode_updates["stereo/single_frame"]
        ):
            raise ValueError("single-frame counter disagrees with mode counters")
        if bool(getattr(self.args, "four_mode_mixed_training", False)):
            expected_mode_updates = mode_occurrences_before(
                schedule_seed, generator_updates, mode_weights
            )
            if mode_updates != expected_mode_updates:
                raise ValueError(
                    "checkpoint mode counters disagree with seeded schedule"
                )
            expected_mode_samples = {
                mode_id: updates
                * self.mode_batch_sizes[mode_id]
                * self.mode_grad_accumulates[mode_id]
                * expected_world_size
                for mode_id, updates in mode_updates.items()
            }
            if mode_samples != expected_mode_samples:
                raise ValueError(
                    "checkpoint sample counters disagree with update/BS/DDP contract"
                )

        self.generator_updates = generator_updates
        self.discriminator_updates = discriminator_updates
        self.four_frame_updates = four_frame_updates
        self.single_frame_updates = single_frame_updates
        self.mode_updates = mode_updates
        self.mode_samples = mode_samples
        self.batch_updates = batch_updates
        self._micro_step = 0
        self._logical_mode_id = None
        self._logical_dataset_id = None
        self._logical_global_samples = 0

    @staticmethod
    def _flatten_view_videos(video: torch.Tensor) -> torch.Tensor:
        return rearrange(video, "b v c t h w -> (b v) c t h w")

    @staticmethod
    def _flatten_view_frames(video: torch.Tensor) -> torch.Tensor:
        return rearrange(video, "b v c t h w -> (b v t) c h w")

    @staticmethod
    def _unwrap_batch(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) != 1:
                raise ValueError("Stereo training accepts exactly one dataset batch")
            return batch[0]
        return batch

    @staticmethod
    def _set_requires_grad(module: nn.Module | None, enabled: bool) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    @staticmethod
    def _source_num_frames(temporal_mode: TemporalMode) -> int:
        return temporal_mode_num_frames(temporal_mode)

    @staticmethod
    def _uniform_batch_metadata(batch, key: str) -> str:
        value = batch.get(key)
        batch_size = int(batch["video"].shape[0])
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            raise ValueError(f"batch metadata {key!r} is missing or invalid")
        if len(values) != batch_size:
            raise ValueError(
                f"batch metadata {key!r} has {len(values)} entries for B={batch_size}"
            )
        if not values or any(item != values[0] for item in values):
            raise ValueError(f"batch mixes multiple values for {key!r}")
        return values[0]

    @staticmethod
    def _uniform_batch_integer(batch, key: str) -> int:
        value = batch.get(key)
        batch_size = int(batch["video"].shape[0])
        if isinstance(value, torch.Tensor):
            values = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, int):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            raise ValueError(f"batch metadata {key!r} is missing or invalid")
        if len(values) != batch_size:
            raise ValueError(
                f"batch metadata {key!r} has {len(values)} entries for B={batch_size}"
            )
        if not values or any(type(item) is not int for item in values):
            raise ValueError(f"batch metadata {key!r} must contain integers")
        if any(item != values[0] for item in values):
            raise ValueError(f"batch mixes multiple values for {key!r}")
        return values[0]

    @classmethod
    def _mode_from_batch(cls, batch) -> tuple[str, EyeMode, TemporalMode]:
        mode_id = cls._uniform_batch_metadata(batch, "mode_id")
        eye_mode = cls._uniform_batch_metadata(batch, "eye_mode")
        temporal_mode = cls._uniform_batch_metadata(batch, "temporal_mode")
        if mode_id not in MODE_IDS:
            raise ValueError(f"unsupported mode_id {mode_id!r}")
        if eye_mode not in ("mono", "stereo"):
            raise ValueError(f"unsupported eye_mode {eye_mode!r}")
        if temporal_mode not in ("single_frame", "four_frame"):
            raise ValueError(f"unsupported temporal_mode {temporal_mode!r}")
        if mode_id != f"{eye_mode}/{temporal_mode}":
            raise ValueError("mode_id disagrees with eye/temporal metadata")
        expected_view_count = 1 if eye_mode == "mono" else 3
        if cls._uniform_batch_integer(batch, "view_count") != expected_view_count:
            raise ValueError("view_count metadata disagrees with eye_mode")
        teacher_kind = cls._uniform_batch_metadata(batch, "teacher_kind")
        expected_teacher = "da3" if eye_mode == "mono" else "foundation_stereo"
        if teacher_kind != expected_teacher:
            raise ValueError("teacher_kind metadata disagrees with eye_mode")
        return mode_id, eye_mode, temporal_mode

    def _validate_video(
        self,
        video: torch.Tensor,
        *,
        eye_mode: EyeMode,
        temporal_mode: TemporalMode,
    ) -> None:
        if eye_mode not in ("mono", "stereo"):
            raise ValueError(f"unsupported eye mode {eye_mode!r}")
        expected_time = self._source_num_frames(temporal_mode)
        if video.ndim != 7:
            raise ValueError(
                "video must use [B,V,E,C,T,H,W], "
                f"got {tuple(video.shape)}"
            )
        _, views, eyes, channels, time, height, width = video.shape
        if eye_mode == "stereo" and (views, eyes) != (self.stereo_num_views, 2):
            raise ValueError("stereo eye mode requires V=3,E=2")
        if eye_mode == "mono" and (views, eyes) != (1, 1):
            raise ValueError("mono eye mode requires V=1,E=1")
        if channels != 3:
            raise ValueError(f"expected RGB input, got {channels} channels")
        if time != expected_time:
            raise ValueError(
                f"{temporal_mode} requires T={expected_time}, got T={time}"
            )
        if (height, width) != (self.resolution, self.resolution):
            raise ValueError(
                f"expected square resolution {self.resolution}, got {(height, width)}"
            )
        if not torch.is_floating_point(video):
            raise TypeError("video must be floating point and normalized to [-0.5,0.5]")
        if not torch.isfinite(video).all():
            raise ValueError("video contains NaN/Inf")

    def _prepare_temporal_batch(
        self,
        batch,
        *,
        temporal_mode: TemporalMode,
    ):
        batch = self._unwrap_batch(batch)
        self._source_num_frames(temporal_mode)
        video = batch["video"]
        expected_time = temporal_mode_num_frames(temporal_mode)
        if video.shape[-3] != expected_time:
            raise ValueError(
                f"{temporal_mode} batch must already be decoded with T={expected_time}"
            )
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        with profile_region("stereo/transfer/cpu_to_gpu"):
            return super().transfer_batch_to_device(
                batch, device, dataloader_idx
            )

    def encode(
        self,
        video: torch.Tensor,
        *,
        eye_mode: EyeMode,
        temporal_mode: TemporalMode,
        sample_posterior: Optional[bool] = None,
        fusion_scale_override: Optional[float] = None,
    ) -> StereoEncodeOutput:
        with profile_region("stereo/encoder/input_validation"):
            self._validate_video(
                video,
                eye_mode=eye_mode,
                temporal_mode=temporal_mode,
            )
        with profile_region("stereo/encoder/total"):
            encoded = self.encoder.forward_stereo(
                video,
                eye_mode=eye_mode,
                temporal_mode=temporal_mode,
                fusion_scale_override=fusion_scale_override,
            )
        with profile_region("stereo/encoder/posterior_projection"):
            parameters = self.posterior_projection(encoded.features)
        posterior = StructuredDiagonalGaussianPosterior(
            distribution=DiagonalGaussianDistribution(parameters),
            batch_size=encoded.batch_size,
            views=encoded.views,
        )
        should_sample = self.training if sample_posterior is None else sample_posterior
        with profile_region("stereo/encoder/posterior_sample"):
            latent = posterior.sample() if should_sample else posterior.mode()
        return StereoEncodeOutput(
            latent=latent,
            posterior=posterior,
            fusion=encoded.fusion,
            eye_mode=eye_mode,
            temporal_mode=temporal_mode,
            source_num_frames=self._source_num_frames(temporal_mode),
        )

    def decode(
        self,
        latent: torch.Tensor,
        *,
        temporal_mode: TemporalMode,
    ) -> StereoDecodeOutput:
        if latent.ndim != 6:
            raise ValueError(
                f"latent must use [B,V,C,1,H,W], got {tuple(latent.shape)}"
            )
        view_count = latent.shape[1]
        if view_count not in (1, self.stereo_num_views):
            raise ValueError("latent view count must be one or three")
        if latent.shape[2] != self.latent_channels or latent.shape[3] != 1:
            raise ValueError("latent must contain 48 channels and one temporal slot")
        flattened = rearrange(latent, "b v c t h w -> (b v) c t h w")
        with profile_region("stereo/decoder/latent_projection"):
            projected = self.latent_projection(flattened)
        with profile_region("stereo/decoder/total"):
            return self.decoder.forward_stereo(
                projected,
                temporal_mode=temporal_mode,
                view_count=view_count,
            )

    def forward(
        self,
        video: torch.Tensor,
        *,
        eye_mode: EyeMode,
        temporal_mode: TemporalMode,
        sample_posterior: Optional[bool] = None,
        fusion_scale_override: Optional[float] = None,
    ) -> StereoVAEOutput:
        encoded = self.encode(
            video,
            eye_mode=eye_mode,
            temporal_mode=temporal_mode,
            sample_posterior=sample_posterior,
            fusion_scale_override=fusion_scale_override,
        )
        decoded = self.decode(encoded.latent, temporal_mode=temporal_mode)
        return StereoVAEOutput(
            rgb=decoded.rgb,
            raw_relative_log_depth=decoded.raw_relative_log_depth,
            latent=encoded.latent,
            posterior=encoded.posterior,
            fusion=encoded.fusion,
            eye_mode=encoded.eye_mode,
            temporal_mode=encoded.temporal_mode,
            source_num_frames=encoded.source_num_frames,
        )

    def _effective_kl_weight(self) -> float:
        if self.kl_warmup_steps == 0:
            return self.kl_weight
        fraction = min(
            1.0,
            float(self.generator_updates) / self.kl_warmup_steps,
        )
        return self.kl_weight * fraction

    def compute_core_loss(
        self,
        batch,
        *,
        eye_mode: EyeMode,
        temporal_mode: TemporalMode,
        sample_posterior: Optional[bool] = None,
    ) -> _StereoCoreLossOutput:
        batch = self._unwrap_batch(batch)
        model_output = self(
            batch["video"],
            eye_mode=eye_mode,
            temporal_mode=temporal_mode,
            sample_posterior=sample_posterior,
        )
        rgb_target = batch["video"][:, :, 0]
        teacher_kind = self._uniform_batch_metadata(batch, "teacher_kind")
        valid_mask = batch["valid_mask"]
        if teacher_kind == "foundation_stereo":
            target = relative_target_from_foundation_stereo(
                batch["disparity"],
                valid_mask,
                batch["fx"],
                batch["baseline_m"],
                epsilon=self.relative_depth_epsilon,
            )
        elif teacher_kind == "da3":
            target = relative_target_from_da3(
                batch["da3_relative_depth"],
                valid_mask,
                epsilon=self.relative_depth_epsilon,
            )
        else:
            raise ValueError(f"unsupported teacher_kind {teacher_kind!r}")
        relative_prediction, _ = relative_prediction_from_raw(
            model_output.raw_relative_log_depth,
            target.valid_mask,
        )
        effective_kl_weight = self._effective_kl_weight()
        with profile_region("stereo/loss/core_total"):
            loss = self.core_objective(
                rgb_prediction=model_output.rgb,
                rgb_target=rgb_target,
                relative_depth_prediction=relative_prediction,
                relative_depth_target=target.relative_log_depth,
                valid_mask=target.valid_mask,
                posterior=model_output.posterior,
                kl_weight_override=effective_kl_weight,
            )
        return _StereoCoreLossOutput(
            model=model_output,
            loss=loss,
            relative_log_depth_prediction=relative_prediction,
            relative_log_depth_target=target.relative_log_depth,
            valid_mask=target.valid_mask,
            effective_kl_weight=effective_kl_weight,
        )

    def _perceptual_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.perceptual_model is None:
            return prediction.new_zeros(())
        prediction_frames = self._flatten_view_frames(prediction)
        target_frames = self._flatten_view_frames(target)
        with profile_region("stereo/loss/lpips_vgg"):
            return (
                self.perceptual_model(
                    prediction_frames * 2.0,
                    target_frames * 2.0,
                ).mean()
                * self.perceptual_weight
            )

    @staticmethod
    def _feature_matching_loss(fake_features, real_features):
        if len(fake_features) != len(real_features):
            raise RuntimeError("discriminator feature structures do not match")
        losses = [
            F.l1_loss(fake, real.detach())
            for fake, real in zip(fake_features[:-1], real_features[:-1])
        ]
        if not losses:
            return fake_features[0].new_zeros(())
        return torch.stack(losses).mean()

    def _gan_is_active(self) -> bool:
        return (
            self.gan_enabled
            and self.generator_updates >= self.args.discriminator_iter_start
        )

    def _generator_adversarial_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        temporal_mode: TemporalMode,
    ):
        zero = prediction.new_zeros(())
        if not self._gan_is_active():
            return zero, zero, zero, zero
        self._set_requires_grad(self.image_discriminator, False)
        self._set_requires_grad(self.video_discriminator, False)

        image_gan = zero
        video_gan = zero
        image_features = zero
        video_features = zero
        if self.image_gan_weight > 0:
            fake_images = self._flatten_view_frames(prediction)
            real_images = self._flatten_view_frames(target)
            fake_logits, fake_feature_list = self.image_discriminator(
                fake_images, False
            )
            with torch.no_grad():
                _, real_feature_list = self.image_discriminator(real_images, False)
            image_gan = -fake_logits.mean()
            image_features = self._feature_matching_loss(
                fake_feature_list, real_feature_list
            )
        if temporal_mode == "four_frame" and self.video_gan_weight > 0:
            fake_videos = self._flatten_view_videos(prediction)
            real_videos = self._flatten_view_videos(target)
            fake_logits, fake_feature_list = self.video_discriminator(
                fake_videos, False
            )
            with torch.no_grad():
                _, real_feature_list = self.video_discriminator(real_videos, False)
            video_gan = -fake_logits.mean()
            video_features = self._feature_matching_loss(
                fake_feature_list, real_feature_list
            )

        adversarial = (
            self.image_gan_weight * image_gan
            + self.video_gan_weight * video_gan
        )
        feature_matching = self.gan_feat_weight * (
            self.image_gan_weight * image_features
            + self.video_gan_weight * video_features
        )
        return adversarial, feature_matching, image_gan, video_gan

    def _discriminator_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        temporal_mode: TemporalMode,
    ):
        zero = prediction.new_zeros(())
        if not self._gan_is_active():
            return zero, zero, zero
        self._set_requires_grad(self.image_discriminator, True)
        self._set_requires_grad(self.video_discriminator, True)

        image_loss = zero
        video_loss = zero
        if self.image_gan_weight > 0:
            real_logits, _ = self.image_discriminator(
                self._flatten_view_frames(target).detach(),
                self.apply_diffaug,
            )
            fake_logits, _ = self.image_discriminator(
                self._flatten_view_frames(prediction).detach(),
                self.apply_diffaug,
            )
            image_loss = self.disc_loss(real_logits, fake_logits)
        if temporal_mode == "four_frame" and self.video_gan_weight > 0:
            real_logits, _ = self.video_discriminator(
                self._flatten_view_videos(target).detach(),
                self.apply_diffaug,
            )
            fake_logits, _ = self.video_discriminator(
                self._flatten_view_videos(prediction).detach(),
                self.apply_diffaug,
            )
            video_loss = self.disc_loss(real_logits, fake_logits)
        total = (
            self.image_gan_weight * image_loss
            + self.video_gan_weight * video_loss
        )
        return total, image_loss, video_loss

    @staticmethod
    def _as_sequence(value):
        return list(value) if isinstance(value, (list, tuple)) else [value]

    def _log_loss_breakdown(
        self,
        prefix: str,
        result: _StereoCoreLossOutput,
        *,
        total_loss: Optional[torch.Tensor] = None,
    ) -> None:
        view_count = result.model.raw_relative_log_depth.shape[1]
        batch_size = result.model.raw_relative_log_depth.shape[0]
        pixels_per_view = (
            result.model.raw_relative_log_depth.numel() // view_count
        )
        metrics = {
            f"{prefix}/total_loss": (
                result.loss.total if total_loss is None else total_loss
            ),
            f"{prefix}/rgb_loss": result.loss.rgb,
            f"{prefix}/relative_depth_loss": result.loss.relative_depth,
            f"{prefix}/relative_gradient_loss": result.loss.relative_gradient,
            f"{prefix}/kl_loss": result.loss.kl,
            f"{prefix}/kl_weight": result.model.rgb.new_tensor(
                result.effective_kl_weight
            ),
        }
        for view in range(view_count):
            metrics[f"{prefix}/relative_depth_loss_view_{view}"] = (
                result.loss.relative_depth_per_view[view]
            )
            metrics[f"{prefix}/valid_pixels_view_{view}"] = (
                result.loss.relative_depth_valid_count[view].float()
            )
            metrics[f"{prefix}/valid_ratio_view_{view}"] = (
                result.loss.relative_depth_valid_count[view].float()
                / pixels_per_view
            )
            supervised_samples = (
                result.loss.relative_depth_supervised_sample_count[view].float()
            )
            metrics[f"{prefix}/supervised_samples_view_{view}"] = (
                supervised_samples
            )
            metrics[f"{prefix}/empty_supervision_samples_view_{view}"] = (
                supervised_samples.new_tensor(float(batch_size))
                - supervised_samples
            )
            if result.model.fusion is not None:
                confidence = result.model.fusion.confidence[:, view]
                metrics[f"{prefix}/fusion_normalized_entropy_view_{view}"] = (
                    1.0 - confidence.mean()
                )
        self.log_dict(
            metrics,
            prog_bar=False,
            logger=True,
            on_step=prefix.startswith("train/"),
            on_epoch=not prefix.startswith("train/"),
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        with profile_region("stereo/step/training_step"):
            return self._profiled_training_step(batch)

    def _profiled_training_step(self, batch):
        source_batch = self._unwrap_batch(batch)
        mode_id, eye_mode, temporal_mode = self._mode_from_batch(source_batch)
        dataset_id = None
        if bool(getattr(self.args, "four_mode_mixed_training", False)):
            mode_weights = parse_weight_spec(
                getattr(self.args, "mode_update_weights", "1:1:1:1"), MODE_IDS
            )
            expected_mode = mode_for_update(
                int(self.args.mode_schedule_seed),
                self.generator_updates,
                mode_weights,
            )
            if mode_id != expected_mode:
                raise RuntimeError(
                    f"seeded mode schedule expected {expected_mode}, got {mode_id}"
                )
            dataset_id = self._uniform_batch_metadata(source_batch, "dataset_id")
            prior_modes = mode_occurrences_before(
                int(self.args.mode_schedule_seed), self.generator_updates, mode_weights
            )
            occurrence = (
                sum(
                    count
                    for prior_mode, count in prior_modes.items()
                    if prior_mode.startswith("mono/")
                )
                if mode_id.startswith("mono/")
                else prior_modes[mode_id]
            )
            expected_dataset = dataset_for_mode_occurrence(
                int(self.args.mode_schedule_seed),
                mode_id,
                occurrence,
                parse_weight_spec(
                    getattr(self.args, "mono_dataset_weights", "9:1"),
                    ("hy", "libero"),
                ),
            )
            if dataset_id != expected_dataset:
                raise RuntimeError(
                    f"seeded dataset schedule expected {expected_dataset}, got {dataset_id}"
                )
            self.last_dataset_id = dataset_id
            accumulation_factor = self.mode_grad_accumulates[mode_id]
        else:
            accumulation_factor = int(self.grad_accumulates)
        actual_batch_size = int(source_batch["video"].shape[0])
        if bool(getattr(self.args, "four_mode_mixed_training", False)) and (
            actual_batch_size != self.mode_batch_sizes[mode_id]
        ):
            raise RuntimeError(
                f"{mode_id} expected per-device batch "
                f"{self.mode_batch_sizes[mode_id]}, got {actual_batch_size}"
            )
        if self._micro_step == 0:
            self._logical_mode_id = mode_id
            self._logical_dataset_id = dataset_id
            self._logical_global_samples = 0
        elif (
            mode_id != self._logical_mode_id
            or dataset_id != self._logical_dataset_id
        ):
            raise RuntimeError("gradient accumulation window crossed mode or dataset")
        self.last_temporal_mode = temporal_mode
        self.last_mode_id = mode_id
        self.last_micro_step_index = self._micro_step + 1
        self.last_accumulation_factor = accumulation_factor
        batch = self._prepare_temporal_batch(
            source_batch,
            temporal_mode=temporal_mode,
        )
        result = self.compute_core_loss(
            batch,
            eye_mode=eye_mode,
            temporal_mode=temporal_mode,
            sample_posterior=True,
        )
        rgb_target = batch["video"][:, :, 0]
        perceptual = self._perceptual_loss(
            result.model.rgb,
            rgb_target,
        )
        gan_active = self._gan_is_active()
        adversarial, feature_matching, image_gan, video_gan = (
            self._generator_adversarial_loss(
                result.model.rgb,
                rgb_target,
                temporal_mode=temporal_mode,
            )
        )
        generator_loss = (
            result.loss.total + perceptual + adversarial + feature_matching
        )

        optimizers = self._as_sequence(self.optimizers())
        schedulers = self._as_sequence(self.lr_schedulers())
        generator_optimizer = optimizers[0]
        with profile_region("stereo/update/backward"):
            self.manual_backward(generator_loss / accumulation_factor)

        self.batch_updates += 1
        self._micro_step += 1
        if self._micro_step > accumulation_factor:
            raise RuntimeError("logical update received too many micro-batches")
        self.last_microbatch_size = actual_batch_size
        self._logical_global_samples += (
            self.last_microbatch_size * int(self.trainer.world_size)
        )
        should_step = self._micro_step == accumulation_factor
        if should_step:
            if self.grad_clip_val is not None:
                with profile_region("stereo/update/gradient_clipping"):
                    self.clip_gradients(
                        generator_optimizer,
                        gradient_clip_val=self.grad_clip_val,
                    )
            with profile_region("stereo/update/adam_step"):
                generator_optimizer.step()
            with profile_region("stereo/update/zero_grad"):
                generator_optimizer.zero_grad()
            if temporal_mode == "four_frame":
                self.four_frame_updates += 1
            else:
                self.single_frame_updates += 1
            self.mode_updates[mode_id] += 1
            self.mode_samples[mode_id] += self._logical_global_samples
            self.last_logical_global_samples = self._logical_global_samples
            self.generator_updates += 1
            with profile_region("stereo/update/scheduler"):
                schedulers[0].step_update(self.generator_updates)
            self._micro_step = 0
            self._logical_mode_id = None
            self._logical_dataset_id = None
            self._logical_global_samples = 0

        discriminator_total = result.model.rgb.new_zeros(())
        discriminator_image = result.model.rgb.new_zeros(())
        discriminator_video = result.model.rgb.new_zeros(())
        discriminator_has_path = self.image_gan_weight > 0 or (
            temporal_mode == "four_frame" and self.video_gan_weight > 0
        )
        if gan_active and discriminator_has_path:
            (
                discriminator_total,
                discriminator_image,
                discriminator_video,
            ) = self._discriminator_loss(
                result.model.rgb,
                rgb_target,
                temporal_mode=temporal_mode,
            )
            self.manual_backward(discriminator_total / accumulation_factor)
            if should_step:
                discriminator_optimizer = optimizers[1]
                if self.grad_clip_val_disc is not None:
                    self.clip_gradients(
                        discriminator_optimizer,
                        gradient_clip_val=self.grad_clip_val_disc,
                    )
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()
                self.discriminator_updates += 1
                schedulers[1].step_update(self.discriminator_updates)

        if should_step and self.generator_updates >= self.args.max_steps:
            self.trainer.should_stop = True

        prefix = f"train/{mode_id}"
        with profile_region("stereo/logging/loss_breakdown"):
            self._log_loss_breakdown(
                prefix,
                result,
                total_loss=generator_loss,
            )
        mode_metrics = {
            f"{prefix}/perceptual_loss": perceptual,
            f"{prefix}/adversarial_loss": adversarial,
            f"{prefix}/feature_matching_loss": feature_matching,
            f"{prefix}/g_image_loss": image_gan,
            f"{prefix}/discriminator_loss": discriminator_total,
            f"{prefix}/d_image_loss": discriminator_image,
            "train/generator_updates": result.model.rgb.new_tensor(
                float(self.generator_updates)
            ),
            "train/discriminator_updates": result.model.rgb.new_tensor(
                float(self.discriminator_updates)
            ),
            "train/batch_updates": result.model.rgb.new_tensor(
                float(self.batch_updates)
            ),
            "train/four_frame_updates": result.model.rgb.new_tensor(
                float(self.four_frame_updates)
            ),
            "train/single_frame_updates": result.model.rgb.new_tensor(
                float(self.single_frame_updates)
            ),
        }
        for tracked_mode in MODE_IDS:
            metric_name = tracked_mode.replace("/", "_")
            mode_metrics[f"train/{metric_name}_updates"] = (
                result.model.rgb.new_tensor(float(self.mode_updates[tracked_mode]))
            )
            mode_metrics[f"train/{metric_name}_samples"] = (
                result.model.rgb.new_tensor(float(self.mode_samples[tracked_mode]))
            )
        if temporal_mode == "four_frame":
            mode_metrics[f"{prefix}/g_video_loss"] = video_gan
            mode_metrics[f"{prefix}/d_video_loss"] = discriminator_video
        with profile_region("stereo/logging/train_metrics"):
            self.log_dict(
                mode_metrics,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        return {"loss": generator_loss.detach()}

    def on_validation_epoch_start(self):
        if bool(getattr(self.args, "four_mode_mixed_training", False)):
            self._validation_mode_sums = {
                mode_id: torch.zeros((), device=self.device, dtype=torch.float64)
                for mode_id in MODE_IDS
            }
            self._validation_mode_counts = {
                mode_id: torch.zeros((), device=self.device, dtype=torch.long)
                for mode_id in MODE_IDS
            }

    def validation_step(self, batch, batch_idx):
        source_batch = self._unwrap_batch(batch)
        mode_id, eye_mode, temporal_mode = self._mode_from_batch(source_batch)
        mode_batch = self._prepare_temporal_batch(
            source_batch,
            temporal_mode=temporal_mode,
        )
        result = self.compute_core_loss(
            mode_batch,
            eye_mode=eye_mode,
            temporal_mode=temporal_mode,
            sample_posterior=False,
        )
        rgb_target = mode_batch["video"][:, :, 0]
        perceptual = self._perceptual_loss(result.model.rgb, rgb_target)
        validation_total = result.loss.total + perceptual
        prefix = f"val/{mode_id}"
        self._log_loss_breakdown(
            prefix,
            result,
            total_loss=validation_total,
        )
        self.log(
            f"{prefix}/perceptual_loss",
            perceptual,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if self._validation_mode_sums is not None:
            batch_size = int(mode_batch["video"].shape[0])
            self._validation_mode_sums[mode_id] += (
                validation_total.detach().double() * batch_size
            )
            self._validation_mode_counts[mode_id] += batch_size

    def on_validation_epoch_end(self):
        if self._validation_mode_sums is None:
            return
        sums = torch.stack(
            [self._validation_mode_sums[mode_id] for mode_id in MODE_IDS]
        )
        counts = torch.stack(
            [self._validation_mode_counts[mode_id] for mode_id in MODE_IDS]
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        if torch.any(counts == 0):
            missing = [
                mode_id
                for mode_id, count in zip(MODE_IDS, counts.tolist())
                if count == 0
            ]
            raise RuntimeError(f"validation did not observe modes: {missing}")
        per_mode = sums / counts.double()
        self.log(
            "val/mixed/total_loss",
            per_mode.mean().float(),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self._validation_mode_sums = None
        self._validation_mode_counts = None

    def configure_optimizers(self):
        generator_parameters = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.posterior_projection.parameters())
            + list(self.latent_projection.parameters())
        )
        generator_optimizer = torch.optim.Adam(
            generator_parameters,
            lr=self.args.lr,
            betas=(0.5, 0.9),
        )
        generator_scheduler = CosineLRScheduler(
            generator_optimizer,
            lr_min=self.args.lr_min,
            t_initial=self.args.max_steps,
            warmup_lr_init=self.args.warmup_lr_init,
            warmup_t=self.args.warmup_steps,
            cycle_mul=1.0,
            cycle_limit=1,
            t_in_epochs=False,
        )
        optimizers = [generator_optimizer]
        schedulers = [
            {"scheduler": generator_scheduler, "interval": "step"}
        ]

        if self.gan_enabled:
            discriminator_steps = (
                self.args.max_steps - self.args.discriminator_iter_start
            )
            discriminator_parameters = []
            if self.image_discriminator is not None:
                discriminator_parameters.extend(
                    self.image_discriminator.parameters()
                )
            if self.video_discriminator is not None:
                discriminator_parameters.extend(
                    self.video_discriminator.parameters()
                )
            discriminator_optimizer = torch.optim.Adam(
                discriminator_parameters,
                lr=self.args.lr * self.args.dis_lr_multiplier,
                betas=(0.5, 0.9),
            )
            discriminator_scheduler = CosineLRScheduler(
                discriminator_optimizer,
                lr_min=(
                    self.args.lr_min * self.args.dis_lr_multiplier
                    if self.args.dis_minlr_multiplier
                    else self.args.lr_min
                ),
                t_initial=discriminator_steps,
                warmup_lr_init=self.args.warmup_lr_init,
                warmup_t=(
                    self.args.dis_warmup_steps
                    if self.args.dis_warmup_steps > 0
                    else self.args.warmup_steps
                ),
                cycle_mul=1.0,
                cycle_limit=1,
                t_in_epochs=False,
            )
            optimizers.append(discriminator_optimizer)
            schedulers.append(
                {"scheduler": discriminator_scheduler, "interval": "step"}
            )
        return optimizers, schedulers

    def on_train_start(self) -> None:
        if getattr(self.args, "stage_transition_checkpoint", None) is None:
            return
        schedulers = self._as_sequence(self.lr_schedulers())
        if not schedulers:
            raise RuntimeError("stage transition requires a generator scheduler")
        schedulers[0].step_update(self.generator_updates)
        if self.gan_enabled:
            if len(schedulers) != 2:
                raise RuntimeError(
                    "GAN stage transition requires generator and discriminator schedulers"
                )
            schedulers[1].step_update(self.discriminator_updates)

    def log_images(self, batch, **kwargs):
        source_batch = self._unwrap_batch(batch)
        mode_id, eye_mode, temporal_mode = self._mode_from_batch(source_batch)
        output = self(
            source_batch["video"],
            eye_mode=eye_mode,
            temporal_mode=temporal_mode,
            sample_posterior=False,
        )
        key = mode_id.replace("/", "_")
        return {
            f"{key}_inputs": self._flatten_view_frames(
                source_batch["video"][:, :, 0]
            ),
            f"{key}_reconstructions": self._flatten_view_frames(
                output.rgb
            ),
        }

    def log_videos(self, batch, **kwargs):
        batch = self._unwrap_batch(batch)
        _, eye_mode, temporal_mode = self._mode_from_batch(batch)
        output = self(
            batch["video"],
            eye_mode=eye_mode,
            temporal_mode=temporal_mode,
            sample_posterior=False,
        )
        return {
            "inputs": self._flatten_view_videos(batch["video"][:, :, 0]),
            "reconstructions": self._flatten_view_videos(output.rgb),
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--embedding_dim", type=int, default=512)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--disc_channels", type=int, default=64)
        parser.add_argument("--disc_layers", type=int, default=3)
        parser.add_argument("--discriminator_iter_start", type=int, default=50000)
        parser.add_argument(
            "--disc_loss_type",
            choices=["hinge", "vanilla"],
            default="hinge",
        )
        parser.add_argument("--image_gan_weight", type=float, required=True)
        parser.add_argument("--video_gan_weight", type=float, required=True)
        parser.add_argument("--gan_feat_weight", type=float, required=True)
        parser.add_argument("--perceptual_weight", type=float, required=True)
        parser.add_argument(
            "--norm_type",
            choices=["batch", "group"],
            default="group",
        )
        parser.add_argument("--lr_min", type=float, default=0.0)
        parser.add_argument("--warmup_steps", type=int, default=0)
        parser.add_argument("--warmup_lr_init", type=float, default=0.0)
        parser.add_argument("--grad_accumulates", type=int, default=1)
        parser.add_argument("--grad_clip_val", type=float, default=1.0)
        parser.add_argument("--grad_clip_val_disc", type=float, default=1.0)
        parser.add_argument("--kl_weight", type=float, required=True)
        parser.add_argument("--kl_warmup_steps", type=int, default=0)
        parser.add_argument("--initialize_vit", action="store_true")

        parser.add_argument("--sigmoid_in_disc", action="store_true")
        parser.add_argument(
            "--activation_in_disc",
            type=str,
            default="leaky_relu",
            choices=["leaky_relu", "tanh"],
        )
        parser.add_argument("--apply_noise", action="store_true")
        parser.add_argument("--apply_diffaug", action="store_true")
        parser.add_argument("--dis_warmup_steps", type=int, default=0)
        parser.add_argument("--dis_lr_multiplier", type=float, default=1.0)
        parser.add_argument("--dis_minlr_multiplier", action="store_true")

        parser.add_argument(
            "--recon_loss_type",
            type=str,
            default="l1",
            choices=["l1", "l2"],
        )
        parser.add_argument("--patch_size", type=int, default=16)
        parser.add_argument(
            "--patch_embed",
            type=str,
            default="linear",
            choices=["linear"],
        )
        parser.add_argument("--enc_block", type=str, default="tttt")
        parser.add_argument("--dec_block", type=str, default="tttt")
        parser.add_argument("--twod_window_size", type=int, default=4)
        parser.add_argument("--defer_temporal_pool", action="store_true")
        parser.add_argument("--defer_spatial_pool", action="store_true")
        parser.add_argument(
            "--spatial_pos",
            type=str,
            default="rel",
            choices=["rel", "rope"],
        )
        parser.add_argument("--spatial_depth", type=int, default=4)
        parser.add_argument("--temporal_depth", type=int, default=4)
        parser.add_argument("--causal_in_peg", action="store_true")
        # 保留原版配置接口；默认不启用 causal，视频四帧均已观测时使用双向注意力。
        parser.add_argument("--causal_in_temporal_transformer", action="store_true")
        parser.add_argument(
            "--peg_backend",
            choices=("conv3d_contiguous", "conv2d_t1_slice"),
            default="conv3d_contiguous",
        )
        parser.add_argument("--dim_head", type=int, default=64)
        parser.add_argument("--heads", type=int, default=8)
        parser.add_argument("--attn_dropout", type=float, default=0.0)
        parser.add_argument("--ff_dropout", type=float, default=0.0)
        parser.add_argument("--ff_mult", type=float, default=4.0)
        parser.add_argument("--latent_channels", type=int, default=48)

        parser.add_argument("--stereo_num_views", type=int, default=3)
        parser.add_argument("--stereo_num_frames", type=int, default=4)
        parser.add_argument(
            "--single_frame_source_index",
            type=int,
            required=True,
        )
        parser.add_argument(
            "--stereo_search_radii",
            nargs=3,
            type=int,
            required=True,
        )
        parser.add_argument(
            "--stereo_search_direction",
            choices=["left", "right"],
            default="left",
        )
        parser.add_argument("--rgb_weight", type=float, required=True)
        parser.add_argument("--relative_depth_weight", type=float, required=True)
        parser.add_argument("--relative_gradient_weight", type=float, required=True)
        parser.add_argument("--relative_depth_epsilon", type=float, default=1e-6)
        parser.add_argument("--smooth_l1_beta", type=float, default=1.0)
        parser.add_argument("--gan_enabled", action="store_true")
        return parser


@dataclass
class _StereoEncoderOutput:
    """Structured encoder result before the VAE posterior projection."""

    features: torch.Tensor
    fusion: Optional[StereoFusionOutput]
    batch_size: int
    views: int


class StereoEncoder(nn.Module):
    def __init__(self, image_size, patch_embed, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, dim=512,
                    causal_in_peg=True, causal_in_temporal_transformer=False,
                    dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., initialize=False,
                    stereo_num_views=None, stereo_num_frames=None, stereo_search_radii=None, stereo_search_direction="left"):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        image_height, image_width = self.image_size
        if image_height % patch_height or image_width % patch_width:
            raise ValueError("image dimensions must be divisible by patch size")
        if patch_embed != "linear":
            raise ValueError("Stereo Encoder requires linear patch embedding")
        if defer_temporal_pool or defer_spatial_pool:
            raise ValueError("Stereo Encoder does not use deferred pooling")
        if stereo_num_frames != 4:
            raise ValueError("Stereo Encoder requires exactly 4 frames")

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange(
                "b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(image_channel * patch_width * patch_height),
            nn.Linear(image_channel * patch_width * patch_height, dim),
            nn.LayerNorm(dim),
        )

        spatial_transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=causal_in_peg,
            ff_mult=ff_mult,
        )
        self.enc_spatial_transformer = Transformer(
            depth=spatial_depth,
            block=block,
            window_size=window_size,
            spatial_pos=spatial_pos,
            **spatial_transformer_kwargs,
        )

        # 四帧位置编码区分帧顺序；默认双向，保留原版参数以兼容旧配置。
        self.enc_temporal_position = nn.Parameter(torch.empty(1, stereo_num_frames, dim))
        self.enc_temporal_transformer = Transformer(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            depth=temporal_depth,
            block='t' * temporal_depth,
            causal=causal_in_temporal_transformer,
            peg=False,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult,
        )

        self.stereo_num_views = stereo_num_views
        self.stereo_num_frames = stereo_num_frames
        self.stereo_embedding_dim = dim
        if stereo_num_views != 3:
            raise ValueError("Stereo Encoder requires exactly 3 views")
        if image_channel != 3:
            raise ValueError("Stereo Encoder requires RGB input")
        if any(layer not in "tw" for layer in block):
            raise ValueError("Stereo Encoder supports only t/w spatial blocks")
        if stereo_search_radii is None:
            raise ValueError("stereo_search_radii must be explicitly configured")
        if len(stereo_search_radii) != stereo_num_views:
            raise ValueError("stereo_search_radii must contain one value per view")

        self.stereo_fusion = StereoFusion(
            dim=dim,
            heads=heads,
            head_dim=dim_head,
            search_radii=stereo_search_radii,
            search_direction=stereo_search_direction,
            attention_dropout=attn_dropout,
        )
        temporal_projection_width = stereo_num_frames * dim
        self.stereo_temporal_projection = nn.Sequential(
            nn.LayerNorm(temporal_projection_width),
            nn.Linear(temporal_projection_width, dim),
            nn.LayerNorm(dim),
        )
        self.single_frame_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        if initialize:
            self.apply(self._init_weights)
        trunc_normal_(self.enc_temporal_position, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]
    


    def _encode_stereo_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Run the original patch embedding and spatial Transformer per frame."""

        if frames.ndim != 4:
            raise ValueError(f"stereo frames must use [N,C,H,W], got {frames.shape}")
        count, _, height, width = frames.shape
        if (height, width) != self.image_size:
            raise ValueError(
                f"expected stereo frame size {self.image_size}, got {(height, width)}"
            )

        with profile_region("stereo/encoder/patch_embedding"):
            frame_tokens = self.to_patch_emb_first_frame(frames[:, :, None])
        grid_height, grid_width = frame_tokens.shape[2:4]
        tokens = rearrange(frame_tokens, "n 1 h w d -> n (h w) d")
        with profile_region("stereo/encoder/spatial_transformer"):
            tokens = self.enc_spatial_transformer(
                tokens,
                video_shape=(count, 1, grid_height, grid_width),
                is_spatial=True,
            )
        if tokens.shape[1] != grid_height * grid_width:
            raise RuntimeError(
                "structured Stereo Encoder must preserve the spatial token grid"
            )
        return rearrange(
            tokens,
            "n (h w) d -> n h w d",
            h=grid_height,
            w=grid_width,
        )

    def forward_stereo(
        self,
        video: torch.Tensor,
        *,
        eye_mode: EyeMode,
        temporal_mode: TemporalMode,
        fusion_scale_override: Optional[float] = None,
    ) -> _StereoEncoderOutput:
        """Encode one or four synchronized frames into one temporal latent slot.

        ``video`` uses ``[B,V,E,C,T,H,W]``. Spatial encoding is applied to
        every frame independently. StereoFusion is optional. A single frame
        uses its own D-to-D projection, while four fused frames exchange
        information before the final 4-to-1 sampler.
        """

        if eye_mode not in ("mono", "stereo"):
            raise ValueError(f"unsupported eye mode {eye_mode!r}")
        expected_time = temporal_mode_num_frames(temporal_mode)
        if video.ndim != 7:
            raise ValueError(
                "structured Stereo Encoder expects [B,V,E,C,T,H,W], "
                f"got {video.shape}"
            )

        batch, views, eyes, channels, time, height, width = video.shape
        if time != expected_time:
            raise ValueError(
                f"{temporal_mode} requires T={expected_time}, got T={time}"
            )
        if channels != 3:
            raise ValueError(f"expected RGB inputs, got {channels} channels")
        if eye_mode == "stereo" and (views, eyes) != (self.stereo_num_views, 2):
            raise ValueError("stereo mode requires V=3,E=2")
        if eye_mode == "mono" and (views, eyes) != (1, 1):
            raise ValueError("mono mode requires V=1,E=1")
        if (height, width) != self.image_size:
            raise ValueError(
                f"expected stereo frame size {self.image_size}, got {(height, width)}"
            )

        frame_batch = rearrange(
            video,
            "b v e c t h w -> (b v e t) c h w",
        )
        frame_features = self._encode_stereo_frames(frame_batch)
        grid_height, grid_width = frame_features.shape[1:3]
        frame_features = rearrange(
            frame_features,
            "(b v e t) h w d -> b v e t h w d",
            b=batch,
            v=views,
            e=eyes,
            t=time,
        )

        left = frame_features[:, :, 0]
        fusion_output: Optional[StereoFusionOutput]
        if eye_mode == "stereo":
            with profile_region("stereo/encoder/stereo_fusion"):
                fusion_output = self.stereo_fusion(
                    left,
                    frame_features[:, :, 1],
                    fusion_scale_override=fusion_scale_override,
                )
            fused = fusion_output.features
        else:
            fusion_output = None
            fused = left

        if temporal_mode == "single_frame":
            with profile_region("stereo/encoder/single_frame_projection"):
                projected = self.single_frame_projection(fused[:, :, 0])
        else:
            # 每个 View、每个空间位置各自形成长度为 4 的序列，不跨 View/空间混合。
            temporal_tokens = rearrange(
                fused,
                "b v t h w d -> (b v h w) t d",
            )
            temporal_tokens = temporal_tokens + self.enc_temporal_position
            with profile_region("stereo/encoder/temporal_transformer"):
                temporal_tokens = self.enc_temporal_transformer(
                    temporal_tokens,
                    video_shape=(
                        batch * views * grid_height * grid_width,
                        time,
                        1,
                        1,
                    ),
                    is_spatial=False,
                )
            temporal_features = rearrange(
                temporal_tokens,
                "(b v h w) t d -> b v h w (t d)",
                b=batch,
                v=views,
                h=grid_height,
                w=grid_width,
            )

            # Temporal Sampler 只负责在帧间注意力之后执行 4D -> D 压缩。
            with profile_region("stereo/encoder/four_frame_projection"):
                projected = self.stereo_temporal_projection(temporal_features)
        features = rearrange(projected, "b v h w d -> (b v) d 1 h w")
        return _StereoEncoderOutput(
            features=features,
            fusion=fusion_output,
            batch_size=batch,
            views=views,
        )




@dataclass
class StereoDecodeOutput:
    """RGB and raw relative log-depth decoded for each structured view."""

    rgb: torch.Tensor
    raw_relative_log_depth: torch.Tensor


class StereoDecoder(nn.Module):
    def __init__(self, image_size, patch_embed, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, dim=512,
                    causal_in_peg=True, causal_in_temporal_transformer=False,
                    dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., gen_upscale=None, initialize=False,
                    stereo_num_views=None, stereo_num_frames=None):
        super().__init__()
        if gen_upscale is not None:
            raise ValueError("Stereo Decoder does not use gen_upscale")
        if patch_embed != "linear":
            raise ValueError("Stereo Decoder requires linear pixel projection")
        if defer_temporal_pool or defer_spatial_pool:
            raise ValueError("Stereo Decoder does not use deferred pooling")
        if stereo_num_frames != 4:
            raise ValueError("Stereo Decoder requires exactly 4 frames")

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        spatial_transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=causal_in_peg,
            ff_mult=ff_mult,
        )
        self.dec_spatial_transformer = Transformer(
            depth=spatial_depth,
            block=block,
            window_size=window_size,
            spatial_pos=spatial_pos,
            **spatial_transformer_kwargs,
        )

        # 先将单 temporal slot 展开为四个帧级特征，再做双向时间建模。
        self.stereo_temporal_expansion = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, stereo_num_frames * dim),
        )
        self.single_frame_expansion = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.dec_temporal_position = nn.Parameter(torch.empty(1, stereo_num_frames, dim))
        self.dec_temporal_transformer = Transformer(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            depth=temporal_depth,
            block='t' * temporal_depth,
            causal=causal_in_temporal_transformer,
            peg=False,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult,
        )


        self.stereo_num_views = stereo_num_views
        self.stereo_num_frames = stereo_num_frames
        if stereo_num_views != 3:
            raise ValueError("Stereo Decoder requires exactly 3 views")
        if image_channel != 3:
            raise ValueError("Stereo Decoder requires RGB output")
        if any(layer not in "tw" for layer in block):
            raise ValueError("Stereo Decoder supports only t/w spatial blocks")
        patch_area = patch_height * patch_width
        # 四个帧级特征分别投影为一帧 patch，Head 不再一次生成四帧。
        self.stereo_rgb_head = nn.Linear(dim, image_channel * patch_area)
        self.relative_log_depth_head = nn.Linear(dim, patch_area, bias=False)

        if initialize:
            self.apply(self._init_weights)
        trunc_normal_(self.dec_temporal_position, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def patch_height_width(self):
        return (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

    def _decode_transformer_features(
        self,
        tokens: torch.Tensor,
        *,
        temporal_mode: TemporalMode,
    ) -> torch.Tensor:
        batch_views, time, height, width, dim = tokens.shape
        if time != 1:
            raise ValueError("temporal expansion expects exactly one latent slot")

        output_time = temporal_mode_num_frames(temporal_mode)
        if temporal_mode == "single_frame":
            with profile_region("stereo/decoder/single_frame_expansion"):
                expanded = self.single_frame_expansion(tokens[:, 0])[:, None]
            expanded = rearrange(expanded, "n t h w d -> (n h w) t d")
        else:
            # D -> 4D 后恢复四个帧级 feature，时间 Attention 在空间解码之前执行。
            with profile_region("stereo/decoder/four_frame_expansion"):
                expanded = self.stereo_temporal_expansion(tokens[:, 0])
            expanded = expanded.reshape(
                batch_views, height, width, self.stereo_num_frames, dim
            )
            expanded = rearrange(expanded, "n h w t d -> (n h w) t d")
            expanded = expanded + self.dec_temporal_position
            with profile_region("stereo/decoder/temporal_transformer"):
                expanded = self.dec_temporal_transformer(
                    expanded,
                    video_shape=(
                        batch_views * height * width,
                        self.stereo_num_frames,
                        1,
                        1,
                    ),
                    is_spatial=False,
                )

        # Spatial Decoder 把每帧视为独立样本，PEG 只能看到 T=1。
        frame_tokens = rearrange(
            expanded,
            "(n h w) t d -> (n t) (h w) d",
            n=batch_views,
            h=height,
            w=width,
        )
        with profile_region("stereo/decoder/spatial_transformer"):
            frame_tokens = self.dec_spatial_transformer(
                frame_tokens,
                video_shape=(batch_views * output_time, 1, height, width),
                is_spatial=True,
            )
        return rearrange(
            frame_tokens,
            "(n t) (h w) d -> n t h w d",
            n=batch_views,
            t=output_time,
            h=height,
            w=width,
        )


    def _unpatch_stereo(
        self,
        patches: torch.Tensor,
        *,
        output_channels: int,
        temporal_mode: TemporalMode,
    ) -> torch.Tensor:
        expected_time = temporal_mode_num_frames(temporal_mode)
        if patches.ndim != 5 or patches.shape[1] != expected_time:
            raise ValueError(
                f"structured decoder patches must use [B*V,{expected_time},H,W,D]"
            )
        patch_height, patch_width = self.patch_size
        return rearrange(
            patches,
            "n t h w (c p1 p2) -> n c t (h p1) (w p2)",
            c=output_channels,
            p1=patch_height,
            p2=patch_width,
        )

    def forward_stereo(
        self,
        tokens: torch.Tensor,
        *,
        temporal_mode: TemporalMode,
        view_count: int,
    ) -> StereoDecodeOutput:
        """Decode one latent slot into RGB and raw relative log-depth."""

        if tokens.ndim != 5:
            raise ValueError(
                f"structured decoder expects [B*V,D,1,H,W], got {tokens.shape}"
            )
        if tokens.shape[2] != 1:
            raise ValueError("structured Stereo Decoder requires exactly one slot")
        if view_count not in (1, self.stereo_num_views):
            raise ValueError("decoder view_count must be one or three")
        if tokens.shape[0] % view_count:
            raise ValueError("flattened decoder batch must be divisible by views")

        features = rearrange(tokens, "n d t h w -> n t h w d")
        features = self._decode_transformer_features(
            features,
            temporal_mode=temporal_mode,
        )
        expected_time = temporal_mode_num_frames(temporal_mode)
        if features.shape[1] != expected_time:
            raise RuntimeError(
                f"decoder must produce {expected_time} frame-level features"
            )

        with profile_region("stereo/decoder/rgb_head"):
            rgb = self._unpatch_stereo(
                self.stereo_rgb_head(features),
                output_channels=3,
                temporal_mode=temporal_mode,
            )
        with profile_region("stereo/decoder/relative_log_depth_head"):
            raw_relative_log_depth = self._unpatch_stereo(
                self.relative_log_depth_head(features),
                output_channels=1,
                temporal_mode=temporal_mode,
            )
        batch = tokens.shape[0] // view_count
        rgb = rearrange(
            rgb, "(b v) c t h w -> b v c t h w", b=batch, v=view_count
        )
        raw_relative_log_depth = rearrange(
            raw_relative_log_depth,
            "(b v) c t h w -> b v c t h w",
            b=batch,
            v=view_count,
        )
        return StereoDecodeOutput(
            rgb=rgb,
            raw_relative_log_depth=raw_relative_log_depth,
        )
