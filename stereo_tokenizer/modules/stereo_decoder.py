"""StereoVAE decoder architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

from stereo_tokenizer.contracts import TemporalMode, pair, temporal_mode_num_frames
from stereo_tokenizer.profiling import profile_region

from .attention import Transformer


@dataclass
class StereoDecodeOutput:
    """RGB and raw relative log-depth decoded for each structured view."""

    rgb: torch.Tensor
    raw_relative_log_depth: torch.Tensor


class StereoDecoder(nn.Module):
    def __init__(self, image_size, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16,
                    spatial_depth=4, temporal_depth=4, dim=512,
                    causal_in_peg=True, causal_in_temporal_transformer=False,
                    dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., gen_upscale=None, initialize=False,
                    stereo_num_views=None, stereo_num_frames=None):
        super().__init__()
        if gen_upscale is not None:
            raise ValueError("Stereo Decoder does not use gen_upscale")
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
            with torch.autocast(
                device_type=expanded.device.type,
                enabled=False,
            ):
                expanded = expanded.float() + self.dec_temporal_position
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
        if not 1 <= view_count <= self.stereo_num_views:
            raise ValueError("decoder view_count must be in [1,3]")
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
