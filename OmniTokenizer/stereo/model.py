import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.attention import Transformer
from .fusion import StereoFusion, StereoFusionOutput


@dataclass(frozen=True)
class StereoTokenizerConfig:
    """Explicit structural contract for the first StereoTokenizer version.

    Data-derived values have no defaults on purpose. A resolved run must state
    them rather than inheriting a guessed value from the model implementation.
    """

    search_radii: tuple[int, int, int]
    search_direction: Literal["left", "right"]
    disparity_scale: tuple[float, float, float]
    disparity_head_bias: float
    spatial_pos: Literal["rope"] = "rope"
    image_size: tuple[int, int] = (256, 256)
    patch_size: tuple[int, int] = (16, 16)
    num_views: int = 3
    num_frames: int = 4
    image_channels: int = 3
    embedding_dim: int = 512
    latent_channels: int = 48
    encoder_block: str = "ttww"
    decoder_block: str = "tttt"
    window_size: int = 8
    temporal_depth: int = 4
    attention_heads: int = 8
    head_dim: int = 64
    attention_dropout: float = 0.0
    feedforward_dropout: float = 0.0
    feedforward_multiplier: float = 4.0
    causal_temporal_attention: bool = True
    causal_peg: bool = True
    disparity_epsilon: float = 1e-6

    def validate(self) -> None:
        height, width = self.image_size
        patch_height, patch_width = self.patch_size
        if self.num_views != 3:
            raise ValueError("the first StereoTokenizer version requires 3 views")
        if self.num_frames != 4:
            raise ValueError("the first StereoTokenizer version requires T=4")
        if self.image_channels != 3:
            raise ValueError("StereoTokenizer requires RGB inputs")
        if self.latent_channels != 48:
            raise ValueError("the frozen StereoTokenizer latent width is 48")
        if self.spatial_pos != "rope":
            raise ValueError("the frozen StereoTokenizer spatial position is RoPE")
        if height % patch_height or width % patch_width:
            raise ValueError("image_size must be divisible by patch_size")
        if len(self.search_radii) != self.num_views:
            raise ValueError("search_radii must have one entry per view")
        if len(self.disparity_scale) != self.num_views:
            raise ValueError("disparity_scale must have one entry per view")
        if any(scale <= 0 for scale in self.disparity_scale):
            raise ValueError("every disparity scale must be positive")
        if not math.isfinite(self.disparity_head_bias):
            raise ValueError("disparity_head_bias must be finite")
        if self.attention_heads * self.head_dim != self.embedding_dim:
            raise ValueError(
                "attention_heads * head_dim must equal embedding_dim"
            )
        if not self.encoder_block or not self.decoder_block:
            raise ValueError("encoder_block and decoder_block cannot be empty")
        if any(block not in "tw" for block in self.encoder_block):
            raise ValueError("Stereo spatial encoder supports only t/w blocks")
        if any(block not in "tw" for block in self.decoder_block):
            raise ValueError("Stereo spatial decoder supports only t/w blocks")

        grid_height = height // patch_height
        grid_width = width // patch_width
        if max(self.search_radii) >= grid_width:
            raise ValueError("each search radius must be smaller than latent width")
        if "w" in self.encoder_block or "w" in self.decoder_block:
            if grid_height % self.window_size or grid_width % self.window_size:
                raise ValueError(
                    "window_size must divide both latent spatial dimensions"
                )


@dataclass
class DiagonalGaussianPosterior:
    mean: torch.Tensor
    logvar: torch.Tensor

    @classmethod
    def from_parameters(
        cls, parameters: torch.Tensor
    ) -> "DiagonalGaussianPosterior":
        if parameters.shape[2] % 2:
            raise ValueError("posterior parameter channels must be even")
        mean, logvar = parameters.chunk(2, dim=2)
        return cls(mean=mean, logvar=logvar.clamp(-30.0, 20.0))

    def sample(self) -> torch.Tensor:
        return self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(
            self.mean
        )

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """Return a per-sample, per-view KL sum with shape ``[B,V]``."""

        return 0.5 * (
            self.mean.square() + self.logvar.exp() - 1.0 - self.logvar
        ).sum(dim=(2, 3, 4, 5))


@dataclass
class StereoEncodeOutput:
    """Raw VAE latent output; no downstream scale/mean/std is applied."""

    latent: torch.Tensor
    posterior: DiagonalGaussianPosterior
    fusion: Optional[StereoFusionOutput]


@dataclass
class StereoTokenizerOutput:
    rgb: torch.Tensor
    disparity: torch.Tensor
    normalized_disparity: torch.Tensor
    latent: torch.Tensor
    posterior: DiagonalGaussianPosterior
    fusion: Optional[StereoFusionOutput]


class FrameSpatialEncoder(nn.Module):
    """Apply the shared spatial encoder to each view/eye/frame independently."""

    def __init__(self, config: StereoTokenizerConfig) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.image_channels = config.image_channels
        self.embedding_dim = config.embedding_dim

        patch_height, patch_width = self.patch_size
        patch_width_flat = self.image_channels * patch_height * patch_width
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_width_flat),
            nn.Linear(patch_width_flat, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )
        self.transformer = Transformer(
            dim=config.embedding_dim,
            depth=len(config.encoder_block),
            block=config.encoder_block,
            dim_head=config.head_dim,
            heads=config.attention_heads,
            ff_mult=config.feedforward_multiplier,
            peg=True,
            peg_causal=config.causal_peg,
            attn_dropout=config.attention_dropout,
            ff_dropout=config.feedforward_dropout,
            window_size=config.window_size,
            spatial_pos=config.spatial_pos,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"expected [N,C,H,W], got {tuple(images.shape)}")
        count, channels, height, width = images.shape
        if channels != self.image_channels:
            raise ValueError(f"expected {self.image_channels} channels, got {channels}")
        if (height, width) != self.image_size:
            raise ValueError(
                f"expected image size {self.image_size}, got {(height, width)}"
            )

        patch_height, patch_width = self.patch_size
        grid_height = height // patch_height
        grid_width = width // patch_width
        patches = images.unfold(2, patch_height, patch_height).unfold(
            3, patch_width, patch_width
        )
        patches = (
            patches.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .reshape(count, grid_height, grid_width, -1)
        )
        tokens = self.patch_embedding(patches).reshape(
            count, grid_height * grid_width, self.embedding_dim
        )
        tokens = self.transformer(
            tokens,
            video_shape=(count, 1, grid_height, grid_width),
            is_spatial=True,
        )
        return (
            tokens.reshape(count, grid_height, grid_width, self.embedding_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class TemporalTransformer(nn.Module):
    """The original OmniTokenizer temporal Transformer on latent slots."""

    def __init__(self, config: StereoTokenizerConfig) -> None:
        super().__init__()
        kwargs = dict(
            dim=config.embedding_dim,
            depth=config.temporal_depth,
            block="t" * config.temporal_depth,
            dim_head=config.head_dim,
            heads=config.attention_heads,
            ff_mult=config.feedforward_multiplier,
            peg=True,
            peg_causal=config.causal_peg,
            attn_dropout=config.attention_dropout,
            ff_dropout=config.feedforward_dropout,
            spatial_pos=config.spatial_pos,
        )
        if config.causal_temporal_attention:
            kwargs["causal"] = True
        self.transformer = Transformer(**kwargs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 6:
            raise ValueError("temporal features must use [B,V,T,H,W,D]")
        batch, views, time, height, width, dim = features.shape
        tokens = (
            features.permute(0, 1, 3, 4, 2, 5)
            .contiguous()
            .reshape(batch * views * height * width, time, dim)
        )
        tokens = self.transformer(
            tokens,
            video_shape=(batch * views, time, height, width),
            is_spatial=False,
        )
        return (
            tokens.reshape(batch, views, height, width, time, dim)
            .permute(0, 1, 4, 2, 3, 5)
            .contiguous()
        )


class StereoTemporalEncoder(nn.Module):
    """Reduce four fused frame features to one slot, then keep original T-attn."""

    def __init__(self, config: StereoTokenizerConfig) -> None:
        super().__init__()
        self.num_frames = config.num_frames
        self.embedding_dim = config.embedding_dim
        flattened_dim = self.num_frames * self.embedding_dim
        self.reducer = nn.Sequential(
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )
        self.temporal_transformer = TemporalTransformer(config)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 6:
            raise ValueError("expected fused features [B,V,T,H,W,D]")
        batch, views, time, height, width, dim = features.shape
        if time != self.num_frames or dim != self.embedding_dim:
            raise ValueError(
                f"expected T={self.num_frames}, D={self.embedding_dim}; "
                f"got T={time}, D={dim}"
            )
        reduced = (
            features.permute(0, 1, 3, 4, 2, 5)
            .contiguous()
            .reshape(batch, views, height, width, time * dim)
        )
        reduced = self.reducer(reduced).unsqueeze(2)
        reduced = self.temporal_transformer(reduced)
        return reduced.permute(0, 1, 5, 2, 3, 4).contiguous()


class StereoDecoder(nn.Module):
    def __init__(self, config: StereoTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.post_posterior = nn.Linear(
            config.latent_channels, config.embedding_dim
        )
        self.temporal_transformer = TemporalTransformer(config)
        self.spatial_transformer = Transformer(
            dim=config.embedding_dim,
            depth=len(config.decoder_block),
            block=config.decoder_block,
            dim_head=config.head_dim,
            heads=config.attention_heads,
            ff_mult=config.feedforward_multiplier,
            peg=True,
            peg_causal=config.causal_peg,
            attn_dropout=config.attention_dropout,
            ff_dropout=config.feedforward_dropout,
            window_size=config.window_size,
            spatial_pos=config.spatial_pos,
        )

        patch_height, patch_width = config.patch_size
        patch_area = patch_height * patch_width
        self.rgb_head = nn.Linear(
            config.embedding_dim,
            config.image_channels * config.num_frames * patch_area,
        )
        self.disparity_head = nn.Linear(
            config.embedding_dim, config.num_frames * patch_area
        )
        self.register_buffer(
            "disparity_scale",
            torch.as_tensor(config.disparity_scale, dtype=torch.float32).reshape(
                1, config.num_views, 1, 1, 1, 1
            ),
            persistent=True,
        )

    def initialize_disparity_bias(self) -> None:
        nn.init.constant_(
            self.disparity_head.bias, self.config.disparity_head_bias
        )

    def _unpatch(
        self, patches: torch.Tensor, output_channels: int
    ) -> torch.Tensor:
        batch, views, slots, grid_height, grid_width, _ = patches.shape
        if slots != 1:
            raise ValueError(
                f"the no-anchor decoder requires exactly one slot, got {slots}"
            )
        patch_height, patch_width = self.config.patch_size
        patches = patches.squeeze(2).reshape(
            batch,
            views,
            grid_height,
            grid_width,
            output_channels,
            self.config.num_frames,
            patch_height,
            patch_width,
        )
        return (
            patches.permute(0, 1, 4, 5, 2, 6, 3, 7)
            .contiguous()
            .reshape(
                batch,
                views,
                output_channels,
                self.config.num_frames,
                grid_height * patch_height,
                grid_width * patch_width,
            )
        )

    def forward(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent.ndim != 6:
            raise ValueError("latent must use [B,V,C,T,H,W]")
        batch, views, channels, slots, height, width = latent.shape
        if views != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} views, got {views}")
        if channels != self.config.latent_channels or slots != 1:
            raise ValueError(
                f"expected latent [B,V,{self.config.latent_channels},1,H,W], "
                f"got {tuple(latent.shape)}"
            )

        features = latent.permute(0, 1, 3, 4, 5, 2).contiguous()
        features = self.post_posterior(features)
        features = self.temporal_transformer(features)

        tokens = features.reshape(
            batch * views * slots, height * width, self.config.embedding_dim
        )
        tokens = self.spatial_transformer(
            tokens,
            video_shape=(batch * views, slots, height, width),
            is_spatial=True,
        )
        features = tokens.reshape(
            batch, views, slots, height, width, self.config.embedding_dim
        )

        rgb = self._unpatch(self.rgb_head(features), self.config.image_channels)
        raw_disparity = self._unpatch(self.disparity_head(features), 1)
        normalized_disparity = (
            F.softplus(raw_disparity) + self.config.disparity_epsilon
        )
        disparity = normalized_disparity * self.disparity_scale.to(
            normalized_disparity.dtype
        )
        return rgb, disparity, normalized_disparity


class StereoTokenizer(nn.Module):
    """First-version structured StereoTokenizer.

    The class accepts only ``[B,3,E,3,4,256,256]``-style structured clips.
    The legacy OmniTokenizer image-mode remains in the original class.
    """

    def __init__(self, config: StereoTokenizerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.spatial_encoder = FrameSpatialEncoder(config)
        self.stereo_fusion = StereoFusion(
            dim=config.embedding_dim,
            heads=config.attention_heads,
            head_dim=config.head_dim,
            search_radii=config.search_radii,
            search_direction=config.search_direction,
            attention_dropout=config.attention_dropout,
        )
        self.temporal_encoder = StereoTemporalEncoder(config)
        self.posterior_head = nn.Linear(
            config.embedding_dim, 2 * config.latent_channels
        )
        self.decoder = StereoDecoder(config)

        self.apply(self._initialize_module)
        self.decoder.initialize_disparity_bias()

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_video(self, video: torch.Tensor, mode: str) -> None:
        if video.ndim != 7:
            raise ValueError(
                "StereoTokenizer expects [B,V,E,C,T,H,W], "
                f"got {tuple(video.shape)}"
            )
        _, views, eyes, channels, time, height, width = video.shape
        if views != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} views, got {views}")
        if channels != self.config.image_channels:
            raise ValueError(
                f"expected {self.config.image_channels} RGB channels, got {channels}"
            )
        if time != self.config.num_frames:
            raise ValueError(f"expected T={self.config.num_frames}, got T={time}")
        if (height, width) != self.config.image_size:
            raise ValueError(
                f"expected image size {self.config.image_size}, got {(height, width)}"
            )
        if mode == "stereo" and eyes != 2:
            raise ValueError("stereo mode requires exactly two eyes")
        if mode == "mono" and eyes not in (1, 2):
            raise ValueError("mono mode accepts one eye or ignores the second eye")

    def encode(
        self,
        video: torch.Tensor,
        *,
        mode: Literal["mono", "stereo"],
        sample_posterior: Optional[bool] = None,
    ) -> StereoEncodeOutput:
        self._validate_video(video, mode)
        batch, views, eyes, channels, time, height, width = video.shape

        frame_batch = (
            video.permute(0, 1, 2, 4, 3, 5, 6)
            .contiguous()
            .reshape(batch * views * eyes * time, channels, height, width)
        )
        frame_features = self.spatial_encoder(frame_batch)
        grid_height, grid_width = frame_features.shape[-2:]
        frame_features = frame_features.reshape(
            batch,
            views,
            eyes,
            time,
            self.config.embedding_dim,
            grid_height,
            grid_width,
        )

        left = frame_features[:, :, 0].permute(0, 1, 2, 4, 5, 3).contiguous()
        fusion_output: Optional[StereoFusionOutput]
        if mode == "stereo":
            right = (
                frame_features[:, :, 1]
                .permute(0, 1, 2, 4, 5, 3)
                .contiguous()
            )
            fusion_output = self.stereo_fusion(left, right)
            fused_features = fusion_output.features
        else:
            fusion_output = None
            fused_features = left

        encoded = self.temporal_encoder(fused_features)
        posterior_parameters = self.posterior_head(
            encoded.permute(0, 1, 3, 4, 5, 2)
        ).permute(0, 1, 5, 2, 3, 4)
        posterior = DiagonalGaussianPosterior.from_parameters(
            posterior_parameters.contiguous()
        )

        # Training samples the posterior. Validation/inference uses its mode so
        # metrics and strict checkpoint roundtrips are deterministic.
        if sample_posterior is None:
            sample_posterior = self.training
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return StereoEncodeOutput(
            latent=latent, posterior=posterior, fusion=fusion_output
        )

    def decode(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decoder(latent)

    def forward(
        self,
        video: torch.Tensor,
        *,
        mode: Literal["mono", "stereo"],
        sample_posterior: Optional[bool] = None,
    ) -> StereoTokenizerOutput:
        encoded = self.encode(
            video, mode=mode, sample_posterior=sample_posterior
        )
        rgb, disparity, normalized_disparity = self.decode(encoded.latent)
        return StereoTokenizerOutput(
            rgb=rgb,
            disparity=disparity,
            normalized_disparity=normalized_disparity,
            latent=encoded.latent,
            posterior=encoded.posterior,
            fusion=encoded.fusion,
        )
