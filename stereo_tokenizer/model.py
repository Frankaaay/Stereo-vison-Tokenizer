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
from .modules import (
    LPIPS,
    StereoFusion,
    StereoFusionOutput,
    StereoLossBreakdown,
    StereoReconstructionKLLoss,
)
from .modules.attention import Transformer
from .modules.discriminator import NLayerDiscriminator, NLayerDiscriminator3D
from .modules.vae import (
    DiagonalGaussianDistribution,
    StructuredDiagonalGaussianPosterior,
)


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


@dataclass
class StereoVAEOutput:
    rgb: torch.Tensor
    disparity: torch.Tensor
    normalized_disparity: torch.Tensor
    latent: torch.Tensor
    posterior: StructuredDiagonalGaussianPosterior
    fusion: Optional[StereoFusionOutput]


@dataclass
class _StereoCoreLossOutput:
    model: StereoVAEOutput
    loss: StereoLossBreakdown
    normalized_disparity_target: torch.Tensor
    effective_kl_weight: float


class StereoVAE(pl.LightningModule):
    """Continuous variational autoencoder for three synchronized stereo views."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._validate_configuration(args)

        self.embedding_dim = args.embedding_dim
        self.latent_channels = args.latent_channels
        self.stereo_num_views = args.stereo_num_views
        self.stereo_num_frames = args.stereo_num_frames
        self.stereo_mode = args.stereo_mode
        self.resolution = args.resolution
        self.patch_size = args.patch_size

        self.encoder = StereoEncoder(
            image_size=args.resolution,
            image_channel=args.image_channels,
            norm_type=args.norm_type,
            block=args.enc_block,
            window_size=args.twod_window_size,
            spatial_pos=args.spatial_pos,
            patch_embed=args.patch_embed,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
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
            norm_type=args.norm_type,
            block=args.dec_block,
            window_size=args.twod_window_size,
            spatial_pos=args.spatial_pos,
            patch_embed=args.patch_embed,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
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
            stereo_disparity_scale=tuple(args.stereo_disparity_scale),
            stereo_disparity_bias=args.stereo_disparity_bias,
            stereo_disparity_epsilon=args.stereo_disparity_epsilon,
            # 与 Encoder 保持同一配置语义；默认 False 不启用 causal mask。
            causal_in_temporal_transformer=args.causal_in_temporal_transformer,
        )
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
            disparity_weight=args.disparity_weight,
            gradient_weight=args.gradient_weight,
            kl_weight=args.kl_weight,
            smooth_l1_beta=args.smooth_l1_beta,
            rgb_loss_type=args.recon_loss_type,
        )
        self.kl_weight = args.kl_weight
        self.kl_warmup_steps = args.kl_warmup_steps
        self.geometry_gradient_scale_px = args.geometry_gradient_scale_px
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
        self.grad_clip_val = args.grad_clip_val
        self.grad_clip_val_disc = args.grad_clip_val_disc
        self._micro_step = 0
        self.generator_updates = 0
        self.discriminator_updates = 0
        self.batch_updates = 0
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
        if len(args.stereo_disparity_scale) != args.stereo_num_views:
            raise ValueError("one disparity scale is required per view")
        explicit_nonnegative = (
            "rgb_weight",
            "disparity_weight",
            "gradient_weight",
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
        if args.geometry_gradient_scale_px <= 0:
            raise ValueError("geometry_gradient_scale_px must be positive")
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
        checkpoint["stereo_update_counters"] = {
            "generator_updates": self.generator_updates,
            "discriminator_updates": self.discriminator_updates,
            "batch_updates": self.batch_updates,
        }

    def on_load_checkpoint(self, checkpoint) -> None:
        counters = checkpoint.get("stereo_update_counters")
        if counters is None:
            if self.gan_enabled:
                raise ValueError(
                    "legacy GAN checkpoint has no independent optimizer counters"
                )
            global_step = checkpoint.get("global_step", 0)
            if type(global_step) is not int or global_step < 0:
                raise ValueError("checkpoint global_step must be a non-negative integer")
            self.generator_updates = global_step
            self.discriminator_updates = 0
            self.batch_updates = global_step * self.grad_accumulates
            self._micro_step = 0
            return
        if not isinstance(counters, Mapping):
            raise TypeError("stereo_update_counters must be a mapping")

        generator_updates = self._read_checkpoint_counter(
            counters, "generator_updates"
        )
        discriminator_updates = self._read_checkpoint_counter(
            counters, "discriminator_updates"
        )
        batch_updates = self._read_checkpoint_counter(counters, "batch_updates")
        if discriminator_updates > generator_updates:
            raise ValueError("discriminator updates cannot exceed generator updates")
        if not self.gan_enabled and discriminator_updates != 0:
            raise ValueError("GAN-disabled checkpoint has discriminator updates")
        if batch_updates < generator_updates * self.grad_accumulates:
            raise ValueError("batch updates are inconsistent with generator updates")

        self.generator_updates = generator_updates
        self.discriminator_updates = discriminator_updates
        self.batch_updates = batch_updates
        self._micro_step = 0

    @property
    def latent_shape(self):
        height = self.resolution // self.patch_size
        width = self.resolution // self.patch_size
        return (
            self.stereo_num_views,
            self.latent_channels,
            1,
            height,
            width,
        )

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

    def _validate_video(self, video: torch.Tensor) -> None:
        expected = (
            self.stereo_num_views,
            2,
            3,
            self.stereo_num_frames,
            self.resolution,
            self.resolution,
        )
        if video.ndim != 7 or tuple(video.shape[1:]) != expected:
            raise ValueError(
                "video must use [B,3,2,3,4,H,W] with configured square H/W; "
                f"expected (*,{expected}), got {tuple(video.shape)}"
            )
        if not torch.is_floating_point(video):
            raise TypeError("video must be floating point and normalized to [-0.5,0.5]")
        if not torch.isfinite(video).all():
            raise ValueError("video contains NaN/Inf")

    def encode(
        self,
        video: torch.Tensor,
        *,
        mode: Optional[Literal["mono", "stereo"]] = None,
        sample_posterior: Optional[bool] = None,
    ) -> StereoEncodeOutput:
        self._validate_video(video)
        resolved_mode = self.stereo_mode if mode is None else mode
        encoded = self.encoder.forward_stereo(video, mode=resolved_mode)
        parameters = self.posterior_projection(encoded.features)
        posterior = StructuredDiagonalGaussianPosterior(
            distribution=DiagonalGaussianDistribution(parameters),
            batch_size=encoded.batch_size,
            views=encoded.views,
        )
        should_sample = self.training if sample_posterior is None else sample_posterior
        latent = posterior.sample() if should_sample else posterior.mode()
        return StereoEncodeOutput(
            latent=latent,
            posterior=posterior,
            fusion=encoded.fusion,
        )

    def decode(self, latent: torch.Tensor) -> StereoDecodeOutput:
        if latent.ndim != 6:
            raise ValueError(
                f"latent must use [B,V,C,1,H,W], got {tuple(latent.shape)}"
            )
        if latent.shape[1] != self.stereo_num_views:
            raise ValueError("latent view count does not match configuration")
        if latent.shape[2] != self.latent_channels or latent.shape[3] != 1:
            raise ValueError("latent must contain 48 channels and one temporal slot")
        flattened = rearrange(latent, "b v c t h w -> (b v) c t h w")
        return self.decoder.forward_stereo(self.latent_projection(flattened))

    def forward(
        self,
        video: torch.Tensor,
        *,
        mode: Optional[Literal["mono", "stereo"]] = None,
        sample_posterior: Optional[bool] = None,
    ) -> StereoVAEOutput:
        encoded = self.encode(
            video,
            mode=mode,
            sample_posterior=sample_posterior,
        )
        decoded = self.decode(encoded.latent)
        return StereoVAEOutput(
            rgb=decoded.rgb,
            disparity=decoded.disparity,
            normalized_disparity=decoded.normalized_disparity,
            latent=encoded.latent,
            posterior=encoded.posterior,
            fusion=encoded.fusion,
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
        sample_posterior: Optional[bool] = None,
    ) -> _StereoCoreLossOutput:
        batch = self._unwrap_batch(batch)
        model_output = self(
            batch["video"],
            sample_posterior=sample_posterior,
        )
        rgb_target = batch["video"][:, :, 0]
        disparity_target = batch["disparity"]
        valid_mask = batch["valid_mask"]
        scale = self.decoder.stereo_disparity_scale.to(
            device=disparity_target.device,
            dtype=disparity_target.dtype,
        )
        normalized_target = disparity_target / scale
        effective_kl_weight = self._effective_kl_weight()
        loss = self.core_objective(
            rgb_prediction=model_output.rgb,
            rgb_target=rgb_target,
            normalized_disparity_prediction=model_output.normalized_disparity,
            normalized_disparity_target=normalized_target,
            pixel_disparity_prediction=model_output.disparity,
            pixel_disparity_target=disparity_target,
            valid_mask=valid_mask,
            posterior=model_output.posterior,
            gradient_scale_px=self.geometry_gradient_scale_px,
            kl_weight_override=effective_kl_weight,
        )
        return _StereoCoreLossOutput(
            model=model_output,
            loss=loss,
            normalized_disparity_target=normalized_target,
            effective_kl_weight=effective_kl_weight,
        )

    def _perceptual_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.perceptual_model is None:
            return prediction.new_zeros(())
        prediction_frames = self._flatten_view_frames(prediction)
        target_frames = self._flatten_view_frames(target)
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
        if self.video_gan_weight > 0:
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
        if self.video_gan_weight > 0:
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

    def _log_loss_breakdown(self, prefix: str, result: _StereoCoreLossOutput) -> None:
        pixels_per_view = result.model.disparity.numel() // self.stereo_num_views
        metrics = {
            f"{prefix}/loss": result.loss.total,
            f"{prefix}/rgb_loss": result.loss.rgb,
            f"{prefix}/disparity_loss": result.loss.disparity,
            f"{prefix}/gradient_loss": result.loss.disparity_gradient,
            f"{prefix}/kl_loss": result.loss.kl,
            f"{prefix}/kl_weight": result.model.rgb.new_tensor(
                result.effective_kl_weight
            ),
        }
        for view in range(self.stereo_num_views):
            metrics[f"{prefix}/disparity_loss_view_{view}"] = (
                result.loss.disparity_per_view[view]
            )
            metrics[f"{prefix}/valid_pixels_view_{view}"] = (
                result.loss.disparity_valid_count[view].float()
            )
            metrics[f"{prefix}/valid_ratio_view_{view}"] = (
                result.loss.disparity_valid_count[view].float()
                / pixels_per_view
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
            on_step=prefix == "train",
            on_epoch=True,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        batch = self._unwrap_batch(batch)
        result = self.compute_core_loss(batch, sample_posterior=True)
        rgb_target = batch["video"][:, :, 0]
        perceptual = self._perceptual_loss(result.model.rgb, rgb_target)
        gan_active = self._gan_is_active()
        adversarial, feature_matching, image_gan, video_gan = (
            self._generator_adversarial_loss(result.model.rgb, rgb_target)
        )
        generator_loss = (
            result.loss.total + perceptual + adversarial + feature_matching
        )

        optimizers = self._as_sequence(self.optimizers())
        schedulers = self._as_sequence(self.lr_schedulers())
        generator_optimizer = optimizers[0]
        self.manual_backward(generator_loss / self.grad_accumulates)

        self.batch_updates += 1
        self._micro_step += 1
        should_step = self._micro_step % self.grad_accumulates == 0
        if should_step:
            if self.grad_clip_val is not None:
                self.clip_gradients(
                    generator_optimizer,
                    gradient_clip_val=self.grad_clip_val,
                )
            generator_optimizer.step()
            generator_optimizer.zero_grad()
            self.generator_updates += 1
            schedulers[0].step_update(self.generator_updates)
            self._micro_step = 0

        discriminator_total = result.model.rgb.new_zeros(())
        discriminator_image = result.model.rgb.new_zeros(())
        discriminator_video = result.model.rgb.new_zeros(())
        if gan_active:
            (
                discriminator_total,
                discriminator_image,
                discriminator_video,
            ) = self._discriminator_loss(result.model.rgb, rgb_target)
            self.manual_backward(discriminator_total / self.grad_accumulates)
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

        self._log_loss_breakdown("train", result)
        self.log_dict(
            {
                "train/generator_loss": generator_loss,
                "train/perceptual_loss": perceptual,
                "train/adversarial_loss": adversarial,
                "train/feature_matching_loss": feature_matching,
                "train/g_image_loss": image_gan,
                "train/g_video_loss": video_gan,
                "train/discriminator_loss": discriminator_total,
                "train/d_image_loss": discriminator_image,
                "train/d_video_loss": discriminator_video,
                "train/generator_updates": result.model.rgb.new_tensor(
                    float(self.generator_updates)
                ),
                "train/discriminator_updates": result.model.rgb.new_tensor(
                    float(self.discriminator_updates)
                ),
                "train/batch_updates": result.model.rgb.new_tensor(
                    float(self.batch_updates)
                ),
            },
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return {"loss": generator_loss.detach()}

    def validation_step(self, batch, batch_idx):
        result = self.compute_core_loss(batch, sample_posterior=False)
        rgb_target = self._unwrap_batch(batch)["video"][:, :, 0]
        perceptual = self._perceptual_loss(result.model.rgb, rgb_target)
        self._log_loss_breakdown("val", result)
        self.log(
            "val/perceptual_loss",
            perceptual,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/total_loss",
            result.loss.total + perceptual,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

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

    def log_images(self, batch, **kwargs):
        batch = self._unwrap_batch(batch)
        output = self(batch["video"], sample_posterior=False)
        return {
            "inputs": self._flatten_view_frames(batch["video"][:, :, 0]),
            "reconstructions": self._flatten_view_frames(output.rgb),
        }

    def log_videos(self, batch, **kwargs):
        batch = self._unwrap_batch(batch)
        output = self(batch["video"], sample_posterior=False)
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
        parser.add_argument("--temporal_patch_size", type=int, default=4)
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
        parser.add_argument("--dim_head", type=int, default=64)
        parser.add_argument("--heads", type=int, default=8)
        parser.add_argument("--attn_dropout", type=float, default=0.0)
        parser.add_argument("--ff_dropout", type=float, default=0.0)
        parser.add_argument("--ff_mult", type=float, default=4.0)
        parser.add_argument("--latent_channels", type=int, default=48)

        parser.add_argument("--stereo_num_views", type=int, default=3)
        parser.add_argument("--stereo_num_frames", type=int, default=4)
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
        parser.add_argument(
            "--stereo_disparity_scale",
            nargs=3,
            type=float,
            required=True,
        )
        parser.add_argument(
            "--stereo_disparity_bias",
            type=float,
            required=True,
        )
        parser.add_argument(
            "--stereo_disparity_epsilon",
            type=float,
            default=1e-6,
        )
        parser.add_argument(
            "--stereo_mode",
            choices=["mono", "stereo"],
            default="stereo",
        )
        parser.add_argument("--rgb_weight", type=float, required=True)
        parser.add_argument("--disparity_weight", type=float, required=True)
        parser.add_argument("--gradient_weight", type=float, required=True)
        parser.add_argument(
            "--geometry_gradient_scale_px",
            type=float,
            required=True,
        )
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
    def __init__(self, image_size, patch_embed, norm_type, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, temporal_patch_size=2, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, dim=512,
                    causal_in_peg=True, causal_in_temporal_transformer=False,
                    dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., initialize=False,
                    stereo_num_views=None, stereo_num_frames=None, stereo_search_radii=None, stereo_search_direction="left"):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.block = block

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

        frame_tokens = self.to_patch_emb_first_frame(frames[:, :, None])
        grid_height, grid_width = frame_tokens.shape[2:4]
        tokens = rearrange(frame_tokens, "n 1 h w d -> n (h w) d")
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
        mode: Literal["mono", "stereo"],
    ) -> _StereoEncoderOutput:
        """Encode four synchronized frames into one temporal latent slot.

        ``video`` uses ``[B,V,E,C,T,H,W]``. Spatial encoding is applied to
        every frame independently. StereoFusion runs next, then the four fused
        frame features exchange information before the final 4-to-1 sampler.
        """

        if mode not in ("mono", "stereo"):
            raise ValueError(f"unsupported stereo encoder mode {mode!r}")
        if video.ndim != 7:
            raise ValueError(
                "structured Stereo Encoder expects [B,V,E,C,T,H,W], "
                f"got {video.shape}"
            )

        batch, views, eyes, channels, time, height, width = video.shape
        if views != self.stereo_num_views:
            raise ValueError(f"expected {self.stereo_num_views} views, got {views}")
        if time != self.stereo_num_frames:
            raise ValueError(f"expected T={self.stereo_num_frames}, got T={time}")
        if channels != 3:
            raise ValueError(f"expected RGB inputs, got {channels} channels")
        if mode == "stereo" and eyes != 2:
            raise ValueError("stereo mode requires exactly two eyes")
        if mode == "mono" and eyes not in (1, 2):
            raise ValueError("mono mode accepts one eye or ignores the second eye")
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
        if mode == "stereo":
            fusion_output = self.stereo_fusion(left, frame_features[:, :, 1])
            fused = fusion_output.features
        else:
            fusion_output = None
            fused = left

        # 每个 View、每个空间位置各自形成长度为 4 的序列，不跨 View/空间混合。
        temporal_tokens = rearrange(
            fused,
            "b v t h w d -> (b v h w) t d",
        )
        temporal_tokens = temporal_tokens + self.enc_temporal_position
        temporal_tokens = self.enc_temporal_transformer(
            temporal_tokens,
            video_shape=(batch * views * grid_height * grid_width, time, 1, 1),
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
    """Four-frame RGB and disparity decoded for each structured view."""

    rgb: torch.Tensor
    disparity: torch.Tensor
    normalized_disparity: torch.Tensor


class StereoDecoder(nn.Module):
    def __init__(self, image_size, patch_embed, norm_type, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, temporal_patch_size=2, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, dim=512,
                    causal_in_peg=True, causal_in_temporal_transformer=False,
                    dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., gen_upscale=None, initialize=False,
                    stereo_num_views=None, stereo_num_frames=None,
                    stereo_disparity_scale=None, stereo_disparity_bias=None,
                    stereo_disparity_epsilon=1e-6):
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
        self.block = block

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
        self.stereo_disparity_epsilon = stereo_disparity_epsilon
        if stereo_num_views != 3:
            raise ValueError("Stereo Decoder requires exactly 3 views")
        if image_channel != 3:
            raise ValueError("Stereo Decoder requires RGB output")
        if any(layer not in "tw" for layer in block):
            raise ValueError("Stereo Decoder supports only t/w spatial blocks")
        if stereo_disparity_scale is None:
            raise ValueError("stereo_disparity_scale must be explicitly configured")
        if len(stereo_disparity_scale) != stereo_num_views:
            raise ValueError("stereo_disparity_scale must contain one value per view")
        if any(scale <= 0 for scale in stereo_disparity_scale):
            raise ValueError("every stereo disparity scale must be positive")
        if stereo_disparity_bias is None or not math.isfinite(
            stereo_disparity_bias
        ):
            raise ValueError("stereo_disparity_bias must be finite")
        if stereo_disparity_epsilon <= 0:
            raise ValueError("stereo_disparity_epsilon must be positive")

        patch_area = patch_height * patch_width
        # 四个帧级特征分别投影为一帧 patch，Head 不再一次生成四帧。
        self.stereo_rgb_head = nn.Linear(dim, image_channel * patch_area)
        self.stereo_disparity_head = nn.Linear(dim, patch_area)
        self.register_buffer(
            "stereo_disparity_scale",
            torch.as_tensor(
                stereo_disparity_scale, dtype=torch.float32
            ).reshape(1, stereo_num_views, 1, 1, 1, 1),
            persistent=True,
        )

        if initialize:
            self.apply(self._init_weights)
        trunc_normal_(self.dec_temporal_position, std=.02)
        nn.init.constant_(self.stereo_disparity_head.bias, stereo_disparity_bias)

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

    def _decode_transformer_features(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_views, time, height, width, dim = tokens.shape
        if time != 1:
            raise ValueError("temporal expansion expects exactly one latent slot")

        # D -> 4D 后恢复四个帧级 feature，时间 Attention 在空间解码之前执行。
        expanded = self.stereo_temporal_expansion(tokens[:, 0])
        expanded = expanded.reshape(
            batch_views, height, width, self.stereo_num_frames, dim
        )
        expanded = rearrange(expanded, "n h w t d -> (n h w) t d")
        expanded = expanded + self.dec_temporal_position
        expanded = self.dec_temporal_transformer(
            expanded,
            video_shape=(batch_views * height * width, self.stereo_num_frames, 1, 1),
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
        frame_tokens = self.dec_spatial_transformer(
            frame_tokens,
            video_shape=(batch_views * self.stereo_num_frames, 1, height, width),
            is_spatial=True,
        )
        return rearrange(
            frame_tokens,
            "(n t) (h w) d -> n t h w d",
            n=batch_views,
            t=self.stereo_num_frames,
            h=height,
            w=width,
        )


    def _unpatch_stereo(
        self,
        patches: torch.Tensor,
        *,
        output_channels: int,
    ) -> torch.Tensor:
        if patches.ndim != 5 or patches.shape[1] != self.stereo_num_frames:
            raise ValueError(
                "structured decoder patches must use [B*V,4,H,W,D]"
            )
        patch_height, patch_width = self.patch_size
        return rearrange(
            patches,
            "n t h w (c p1 p2) -> n c t (h p1) (w p2)",
            c=output_channels,
            p1=patch_height,
            p2=patch_width,
        )

    def forward_stereo(self, tokens: torch.Tensor) -> StereoDecodeOutput:
        """Decode one latent slot into four RGB and disparity frames."""

        if tokens.ndim != 5:
            raise ValueError(
                f"structured decoder expects [B*V,D,1,H,W], got {tokens.shape}"
            )
        if tokens.shape[2] != 1:
            raise ValueError("structured Stereo Decoder requires exactly one slot")
        if tokens.shape[0] % self.stereo_num_views:
            raise ValueError("flattened decoder batch must be divisible by views")

        features = rearrange(tokens, "n d t h w -> n t h w d")
        features = self._decode_transformer_features(features)
        if features.shape[1] != self.stereo_num_frames:
            raise RuntimeError("decoder must produce four frame-level features")

        rgb = self._unpatch_stereo(
            self.stereo_rgb_head(features), output_channels=3
        )
        raw_disparity = self._unpatch_stereo(
            self.stereo_disparity_head(features), output_channels=1
        )
        batch = tokens.shape[0] // self.stereo_num_views
        rgb = rearrange(rgb, "(b v) c t h w -> b v c t h w", b=batch, v=self.stereo_num_views)
        raw_disparity = rearrange(
            raw_disparity,
            "(b v) c t h w -> b v c t h w",
            b=batch,
            v=self.stereo_num_views,
        )
        normalized_disparity = (
            F.softplus(raw_disparity) + self.stereo_disparity_epsilon
        )
        disparity = normalized_disparity * self.stereo_disparity_scale.to(
            device=normalized_disparity.device,
            dtype=normalized_disparity.dtype,
        )
        return StereoDecodeOutput(
            rgb=rgb,
            disparity=disparity,
            normalized_disparity=normalized_disparity,
        )
