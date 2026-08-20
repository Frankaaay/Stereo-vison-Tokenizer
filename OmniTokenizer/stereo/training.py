from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain
from typing import Literal, Optional, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import StereoLossBreakdown, StereoReconstructionKLLoss
from .model import StereoTokenizer, StereoTokenizerOutput


class StereoBatch(TypedDict):
    video: torch.Tensor
    disparity: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class StereoTrainingLossConfig:
    """All values must come from resolved config; none are guessed here."""

    mode: Literal["mono", "stereo"]
    rgb_weight: float
    disparity_weight: float
    gradient_weight: float
    kl_target_weight: float
    perceptual_weight: float
    kl_warmup_steps: int
    smooth_l1_beta: float
    geometry_gradient_scale_px: float
    rgb_loss_type: Literal["l1", "l2"]

    def validate(self) -> None:
        if self.mode not in ("mono", "stereo"):
            raise ValueError("training mode must be mono or stereo")
        weights = (
            self.rgb_weight,
            self.disparity_weight,
            self.gradient_weight,
            self.kl_target_weight,
            self.perceptual_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if self.kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps must be non-negative")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if self.geometry_gradient_scale_px <= 0:
            raise ValueError("geometry_gradient_scale_px must be positive")


@dataclass
class StereoTrainingStepOutput:
    model: StereoTokenizerOutput
    loss: StereoLossBreakdown
    normalized_disparity_target: torch.Tensor
    effective_kl_weight: float


@dataclass(frozen=True)
class StereoAdversarialConfig:
    """Explicit GAN gate and weights for a resolved training run."""

    enabled: bool
    start_step: int
    image_weight: float
    video_weight: float
    feature_matching_weight: float
    loss_type: Literal["hinge", "vanilla"] = "hinge"
    discriminator_channels: int = 64
    discriminator_layers: int = 3
    norm_type: str = "batch"
    activation: str = "leaky_relu"
    use_sigmoid: bool = False
    apply_blur: bool = False
    apply_noise: bool = False
    apply_diffaug: bool = False

    def validate(self) -> None:
        if self.start_step < 0:
            raise ValueError("GAN start_step must be non-negative")
        weights = (
            self.image_weight,
            self.video_weight,
            self.feature_matching_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("GAN weights must be non-negative")
        if self.enabled and self.image_weight == self.video_weight == 0:
            raise ValueError("enabled GAN requires an image or video weight")
        if self.loss_type not in ("hinge", "vanilla"):
            raise ValueError("GAN loss_type must be hinge or vanilla")
        if self.discriminator_channels <= 0 or self.discriminator_layers <= 0:
            raise ValueError("discriminator size must be positive")

    def active_at(self, global_step: int) -> bool:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        return self.enabled and global_step >= self.start_step


@dataclass(frozen=True)
class StereoOptimizerConfig:
    """Resolved Adam and warmup-cosine parameters; no run values are guessed."""

    autoencoder_lr: float
    discriminator_lr: float
    autoencoder_min_lr: float
    discriminator_min_lr: float
    autoencoder_warmup_start_lr: float
    discriminator_warmup_start_lr: float
    autoencoder_warmup_steps: int
    discriminator_warmup_steps: int
    total_steps: int

    def validate(self) -> None:
        learning_rates = (
            self.autoencoder_lr,
            self.discriminator_lr,
            self.autoencoder_min_lr,
            self.discriminator_min_lr,
            self.autoencoder_warmup_start_lr,
            self.discriminator_warmup_start_lr,
        )
        if any(rate < 0 for rate in learning_rates):
            raise ValueError("optimizer learning rates must be non-negative")
        if self.autoencoder_lr <= 0 or self.discriminator_lr <= 0:
            raise ValueError("base learning rates must be positive")
        if self.autoencoder_min_lr > self.autoencoder_lr:
            raise ValueError("autoencoder_min_lr cannot exceed autoencoder_lr")
        if self.discriminator_min_lr > self.discriminator_lr:
            raise ValueError("discriminator_min_lr cannot exceed discriminator_lr")
        if self.autoencoder_warmup_start_lr > self.autoencoder_lr:
            raise ValueError("autoencoder warmup start cannot exceed base LR")
        if self.discriminator_warmup_start_lr > self.discriminator_lr:
            raise ValueError("discriminator warmup start cannot exceed base LR")
        if self.autoencoder_warmup_steps < 0:
            raise ValueError("autoencoder_warmup_steps must be non-negative")
        if self.discriminator_warmup_steps < 0:
            raise ValueError("discriminator_warmup_steps must be non-negative")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.autoencoder_warmup_steps >= self.total_steps:
            raise ValueError("autoencoder warmup must be shorter than total_steps")
        if self.discriminator_warmup_steps >= self.total_steps:
            raise ValueError("discriminator warmup must be shorter than total_steps")


@dataclass(frozen=True)
class StereoValidationPolicy:
    """Validation is either disabled or runs once over a full split at epoch end."""

    enabled: bool
    manifest_path: Optional[str]
    split_name: Optional[str]

    def validate(self) -> None:
        if self.enabled and (not self.manifest_path or not self.split_name):
            raise ValueError(
                "enabled validation requires a manifest_path and split_name"
            )
        if not self.enabled and (
            self.manifest_path is not None or self.split_name is not None
        ):
            raise ValueError(
                "disabled validation must not carry a manifest or split name"
            )

    def should_run(self, *, epoch_end: bool) -> bool:
        self.validate()
        return self.enabled and epoch_end


@dataclass
class StereoGeneratorLossBreakdown:
    total: torch.Tensor
    core: StereoLossBreakdown
    perceptual: torch.Tensor
    image_adversarial: torch.Tensor
    video_adversarial: torch.Tensor
    feature_matching: torch.Tensor
    gan_active: bool


@dataclass
class StereoGeneratorStepOutput:
    core: StereoTrainingStepOutput
    loss: StereoGeneratorLossBreakdown


@dataclass
class StereoDiscriminatorLossBreakdown:
    total: torch.Tensor
    image: torch.Tensor
    video: torch.Tensor
    active: bool


@dataclass
class StereoValidationStepOutput:
    total: torch.Tensor
    core: StereoTrainingStepOutput
    perceptual: torch.Tensor


@dataclass
class StereoOptimizerBundle:
    autoencoder: torch.optim.Optimizer
    discriminator: Optional[torch.optim.Optimizer]
    autoencoder_scheduler: torch.optim.lr_scheduler.LRScheduler
    discriminator_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]


class StereoTokenizerTrainingCore(nn.Module):
    """Model plus deterministic RGB/disparity/gradient/KL training contract.

    LPIPS, GAN, feature matching, optimizer, and scheduler are explicit in
    ``StereoTokenizerTrainingStage``. Distributed runtime stays outside both.
    """

    def __init__(
        self,
        tokenizer: StereoTokenizer,
        loss_config: StereoTrainingLossConfig,
    ) -> None:
        super().__init__()
        loss_config.validate()
        self.tokenizer = tokenizer
        self.loss_config = loss_config
        self.objective = StereoReconstructionKLLoss(
            rgb_weight=loss_config.rgb_weight,
            disparity_weight=loss_config.disparity_weight,
            gradient_weight=loss_config.gradient_weight,
            kl_weight=loss_config.kl_target_weight,
            smooth_l1_beta=loss_config.smooth_l1_beta,
            rgb_loss_type=loss_config.rgb_loss_type,
        )

    def kl_weight_at(self, global_step: int) -> float:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        if self.loss_config.kl_warmup_steps == 0:
            return self.loss_config.kl_target_weight
        progress = min(global_step / self.loss_config.kl_warmup_steps, 1.0)
        return self.loss_config.kl_target_weight * progress

    def forward(
        self,
        batch: StereoBatch,
        *,
        global_step: int,
        sample_posterior: bool,
    ) -> StereoTrainingStepOutput:
        required = {"video", "disparity", "valid_mask"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"training batch is missing keys: {sorted(missing)}")

        video = batch["video"]
        disparity_target = batch["disparity"]
        valid_mask = batch["valid_mask"]
        model_output = self.tokenizer(
            video,
            mode=self.loss_config.mode,
            sample_posterior=sample_posterior,
        )

        rgb_target = video[:, :, 0]
        if disparity_target.shape != model_output.disparity.shape:
            raise ValueError(
                "pixel disparity target must match model disparity shape; "
                f"got {disparity_target.shape} and {model_output.disparity.shape}"
            )
        if valid_mask.shape != disparity_target.shape:
            raise ValueError("valid_mask must match pixel disparity target")

        scale = self.tokenizer.decoder.disparity_scale.to(
            device=disparity_target.device, dtype=disparity_target.dtype
        )
        normalized_disparity_target = disparity_target / scale
        effective_kl_weight = self.kl_weight_at(global_step)
        loss = self.objective(
            rgb_prediction=model_output.rgb,
            rgb_target=rgb_target,
            normalized_disparity_prediction=(
                model_output.normalized_disparity
            ),
            normalized_disparity_target=normalized_disparity_target,
            pixel_disparity_prediction=model_output.disparity,
            pixel_disparity_target=disparity_target,
            valid_mask=valid_mask,
            posterior=model_output.posterior,
            gradient_scale_px=self.loss_config.geometry_gradient_scale_px,
            kl_weight_override=effective_kl_weight,
        )
        return StereoTrainingStepOutput(
            model=model_output,
            loss=loss,
            normalized_disparity_target=normalized_disparity_target,
            effective_kl_weight=effective_kl_weight,
        )


def _flatten_rgb_frames(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 6:
        raise ValueError("RGB video must use [B,V,C,T,H,W]")
    batch, views, channels, time, height, width = video.shape
    return (
        video.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(batch * views * time, channels, height, width)
    )


def _flatten_view_clips(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 6:
        raise ValueError("RGB video must use [B,V,C,T,H,W]")
    batch, views, channels, time, height, width = video.shape
    return video.contiguous().reshape(
        batch * views, channels, time, height, width
    )


def _hinge_discriminator_loss(
    logits_real: torch.Tensor, logits_fake: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (
        F.relu(1.0 - logits_real).mean()
        + F.relu(1.0 + logits_fake).mean()
    )


def _vanilla_discriminator_loss(
    logits_real: torch.Tensor, logits_fake: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (
        F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean()
    )


def _feature_matching_loss(
    fake_features: list[torch.Tensor],
    real_features: list[torch.Tensor],
    *,
    discriminator_layers: int,
) -> torch.Tensor:
    if len(fake_features) != len(real_features) or len(fake_features) < 2:
        raise ValueError("discriminator feature lists must match and include logits")
    layer_weight = 4.0 / (discriminator_layers + 1)
    return sum(
        layer_weight * F.l1_loss(fake, real.detach())
        for fake, real in zip(fake_features[:-1], real_features[:-1])
    )


def _warmup_cosine_lambda(
    *,
    base_lr: float,
    min_lr: float,
    warmup_start_lr: float,
    warmup_steps: int,
    total_steps: int,
):
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            progress = step / warmup_steps
            learning_rate = warmup_start_lr + progress * (
                base_lr - warmup_start_lr
            )
        else:
            denominator = total_steps - warmup_steps
            progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
            learning_rate = min_lr + 0.5 * (base_lr - min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        return learning_rate / base_lr

    return multiplier


class StereoTokenizerTrainingStage(nn.Module):
    """Full RGB/VAE training objective with explicit perceptual and GAN gates."""

    def __init__(
        self,
        core: StereoTokenizerTrainingCore,
        adversarial_config: StereoAdversarialConfig,
        *,
        perceptual_model: Optional[nn.Module] = None,
        image_discriminator: Optional[nn.Module] = None,
        video_discriminator: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        adversarial_config.validate()
        self.core = core
        self.adversarial_config = adversarial_config

        if core.loss_config.perceptual_weight > 0:
            if perceptual_model is None:
                from ..modules import LPIPS

                perceptual_model = LPIPS()
            perceptual_model.requires_grad_(False)
            perceptual_model.eval()
        elif perceptual_model is not None:
            raise ValueError(
                "perceptual_model was provided while perceptual_weight is zero"
            )
        self.perceptual_model = perceptual_model

        if adversarial_config.enabled:
            if image_discriminator is None or video_discriminator is None:
                from ..base import NLayerDiscriminator, NLayerDiscriminator3D

                discriminator_kwargs = dict(
                    input_nc=core.tokenizer.config.image_channels,
                    ndf=adversarial_config.discriminator_channels,
                    n_layers=adversarial_config.discriminator_layers,
                    norm_type=adversarial_config.norm_type,
                    use_sigmoid=adversarial_config.use_sigmoid,
                    activation=adversarial_config.activation,
                    apply_blur=adversarial_config.apply_blur,
                    apply_noise=adversarial_config.apply_noise,
                )
                if image_discriminator is None:
                    image_discriminator = NLayerDiscriminator(
                        **discriminator_kwargs
                    )
                if video_discriminator is None:
                    video_discriminator = NLayerDiscriminator3D(
                        **discriminator_kwargs
                    )
        elif image_discriminator is not None or video_discriminator is not None:
            raise ValueError(
                "disabled GAN must not construct or receive discriminators"
            )
        self.image_discriminator = image_discriminator
        self.video_discriminator = video_discriminator

    def train(self, mode: bool = True) -> "StereoTokenizerTrainingStage":
        super().train(mode)
        if self.perceptual_model is not None:
            self.perceptual_model.eval()
        return self

    def _set_discriminator_requires_grad(self, requires_grad: bool) -> None:
        for discriminator in (
            self.image_discriminator,
            self.video_discriminator,
        ):
            if discriminator is not None:
                discriminator.requires_grad_(requires_grad)

    def _perceptual_loss(
        self, target_rgb: torch.Tensor, prediction_rgb: torch.Tensor
    ) -> torch.Tensor:
        if self.perceptual_model is None:
            return prediction_rgb.new_zeros(())
        target_frames = _flatten_rgb_frames(target_rgb) * 2.0
        prediction_frames = _flatten_rgb_frames(prediction_rgb) * 2.0
        return self.perceptual_model(target_frames, prediction_frames).mean()

    def generator_step(
        self,
        batch: StereoBatch,
        *,
        global_step: int,
        sample_posterior: bool = True,
    ) -> StereoGeneratorStepOutput:
        core_output = self.core(
            batch,
            global_step=global_step,
            sample_posterior=sample_posterior,
        )
        target_rgb = batch["video"][:, :, 0]
        prediction_rgb = core_output.model.rgb
        perceptual = self._perceptual_loss(target_rgb, prediction_rgb)
        zero = core_output.loss.total.new_zeros(())
        image_adversarial = zero
        video_adversarial = zero
        feature_matching = zero
        gan_active = self.adversarial_config.active_at(global_step)

        if gan_active:
            if self.image_discriminator is None or self.video_discriminator is None:
                raise RuntimeError("active GAN is missing a discriminator")
            real_frames = _flatten_rgb_frames(target_rgb)
            fake_frames = _flatten_rgb_frames(prediction_rgb)
            real_clips = _flatten_view_clips(target_rgb)
            fake_clips = _flatten_view_clips(prediction_rgb)
            self._set_discriminator_requires_grad(False)
            try:
                image_logits_fake, image_features_fake = self.image_discriminator(
                    fake_frames, False
                )
                video_logits_fake, video_features_fake = self.video_discriminator(
                    fake_clips, False
                )
                image_adversarial = -image_logits_fake.mean()
                video_adversarial = -video_logits_fake.mean()

                if self.adversarial_config.feature_matching_weight > 0:
                    with torch.no_grad():
                        _, image_features_real = self.image_discriminator(
                            real_frames, False
                        )
                        _, video_features_real = self.video_discriminator(
                            real_clips, False
                        )
                    image_feature_matching = _feature_matching_loss(
                        image_features_fake,
                        image_features_real,
                        discriminator_layers=(
                            self.adversarial_config.discriminator_layers
                        ),
                    )
                    video_feature_matching = _feature_matching_loss(
                        video_features_fake,
                        video_features_real,
                        discriminator_layers=(
                            self.adversarial_config.discriminator_layers
                        ),
                    )
                    feature_matching = (
                        image_feature_matching + video_feature_matching
                    )
            finally:
                self._set_discriminator_requires_grad(True)

        total = (
            core_output.loss.total
            + self.core.loss_config.perceptual_weight * perceptual
            + self.adversarial_config.image_weight * image_adversarial
            + self.adversarial_config.video_weight * video_adversarial
            + self.adversarial_config.feature_matching_weight
            * feature_matching
        )
        return StereoGeneratorStepOutput(
            core=core_output,
            loss=StereoGeneratorLossBreakdown(
                total=total,
                core=core_output.loss,
                perceptual=perceptual,
                image_adversarial=image_adversarial,
                video_adversarial=video_adversarial,
                feature_matching=feature_matching,
                gan_active=gan_active,
            ),
        )

    def discriminator_step(
        self,
        batch: StereoBatch,
        reconstructed_rgb: torch.Tensor,
        *,
        global_step: int,
    ) -> StereoDiscriminatorLossBreakdown:
        target_rgb = batch["video"][:, :, 0]
        zero = reconstructed_rgb.new_zeros(())
        active = self.adversarial_config.active_at(global_step)
        if not active:
            return StereoDiscriminatorLossBreakdown(
                total=zero, image=zero, video=zero, active=False
            )
        if self.image_discriminator is None or self.video_discriminator is None:
            raise RuntimeError("active GAN is missing a discriminator")
        if reconstructed_rgb.shape != target_rgb.shape:
            raise ValueError("reconstructed RGB must match the left RGB target")

        real_frames = _flatten_rgb_frames(target_rgb).detach()
        fake_frames = _flatten_rgb_frames(reconstructed_rgb).detach()
        real_clips = _flatten_view_clips(target_rgb).detach()
        fake_clips = _flatten_view_clips(reconstructed_rgb).detach()
        image_logits_real, _ = self.image_discriminator(
            real_frames, self.adversarial_config.apply_diffaug
        )
        image_logits_fake, _ = self.image_discriminator(
            fake_frames, self.adversarial_config.apply_diffaug
        )
        video_logits_real, _ = self.video_discriminator(
            real_clips, self.adversarial_config.apply_diffaug
        )
        video_logits_fake, _ = self.video_discriminator(
            fake_clips, self.adversarial_config.apply_diffaug
        )
        loss_function = (
            _hinge_discriminator_loss
            if self.adversarial_config.loss_type == "hinge"
            else _vanilla_discriminator_loss
        )
        image = loss_function(image_logits_real, image_logits_fake)
        video = loss_function(video_logits_real, video_logits_fake)
        total = (
            self.adversarial_config.image_weight * image
            + self.adversarial_config.video_weight * video
        )
        return StereoDiscriminatorLossBreakdown(
            total=total, image=image, video=video, active=True
        )

    @torch.no_grad()
    def validation_step(
        self,
        batch: StereoBatch,
        *,
        global_step: int,
    ) -> StereoValidationStepOutput:
        if self.training:
            raise RuntimeError("validation_step requires eval mode")
        core_output = self.core(
            batch,
            global_step=global_step,
            sample_posterior=False,
        )
        perceptual = self._perceptual_loss(
            batch["video"][:, :, 0], core_output.model.rgb
        )
        total = (
            core_output.loss.total
            + self.core.loss_config.perceptual_weight * perceptual
        )
        return StereoValidationStepOutput(
            total=total, core=core_output, perceptual=perceptual
        )

    def build_optimizers(
        self, config: StereoOptimizerConfig
    ) -> StereoOptimizerBundle:
        config.validate()
        autoencoder = torch.optim.Adam(
            self.core.tokenizer.parameters(),
            lr=config.autoencoder_lr,
            betas=(0.5, 0.9),
        )
        autoencoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
            autoencoder,
            _warmup_cosine_lambda(
                base_lr=config.autoencoder_lr,
                min_lr=config.autoencoder_min_lr,
                warmup_start_lr=config.autoencoder_warmup_start_lr,
                warmup_steps=config.autoencoder_warmup_steps,
                total_steps=config.total_steps,
            ),
        )

        discriminator: Optional[torch.optim.Optimizer] = None
        discriminator_scheduler: Optional[
            torch.optim.lr_scheduler.LRScheduler
        ] = None
        if self.adversarial_config.enabled:
            if self.image_discriminator is None or self.video_discriminator is None:
                raise RuntimeError("enabled GAN is missing a discriminator")
            discriminator = torch.optim.Adam(
                chain(
                    self.image_discriminator.parameters(),
                    self.video_discriminator.parameters(),
                ),
                lr=config.discriminator_lr,
                betas=(0.5, 0.9),
            )
            discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(
                discriminator,
                _warmup_cosine_lambda(
                    base_lr=config.discriminator_lr,
                    min_lr=config.discriminator_min_lr,
                    warmup_start_lr=config.discriminator_warmup_start_lr,
                    warmup_steps=config.discriminator_warmup_steps,
                    total_steps=config.total_steps,
                ),
            )
        return StereoOptimizerBundle(
            autoencoder=autoencoder,
            discriminator=discriminator,
            autoencoder_scheduler=autoencoder_scheduler,
            discriminator_scheduler=discriminator_scheduler,
        )
