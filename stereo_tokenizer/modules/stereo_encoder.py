"""StereoVAE encoder architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

from stereo_tokenizer.contracts import (
    EyeMode,
    TemporalMode,
    pair,
    temporal_mode_num_frames,
)
from stereo_tokenizer.profiling import profile_region

from .attention import Transformer
from .stereo_fusion import StereoFusion, StereoFusionOutput


@dataclass
class _StereoEncoderOutput:
    """Structured encoder result before the VAE posterior projection."""

    features: torch.Tensor
    fusion: Optional[StereoFusionOutput]
    batch_size: int
    views: int


class StereoEncoder(nn.Module):
    def __init__(self, image_size, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16,
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
        if eye_mode == "mono" and (not 1 <= views <= self.stereo_num_views or eyes != 1):
            raise ValueError("mono mode requires V in [1,3],E=1")
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
                fusion_output = self.stereo_fusion(left, frame_features[:, :, 1])
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
            with torch.autocast(
                device_type=temporal_tokens.device.type,
                enabled=False,
            ):
                temporal_tokens = (
                    temporal_tokens.float() + self.enc_temporal_position
                )
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
