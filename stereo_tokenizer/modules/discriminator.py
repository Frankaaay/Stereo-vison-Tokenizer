"""PatchGAN discriminators used by the optional StereoVAE GAN objective."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class _ApplyNoise(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial_shape = (inputs.shape[0], 1, *inputs.shape[2:])
        noise = torch.randn(spatial_shape, device=inputs.device, dtype=inputs.dtype)
        weight_shape = (1, -1, *([1] * (inputs.ndim - 2)))
        return inputs + self.weight.view(*weight_shape) * noise


def _normalization(channels: int, norm_type: str) -> nn.Module:
    if norm_type == "group":
        return nn.GroupNorm(32, channels, eps=1e-6, affine=True)
    if norm_type == "batch":
        return nn.SyncBatchNorm(channels)
    raise ValueError(f"unsupported discriminator norm_type: {norm_type}")


def _activation(name: str) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported discriminator activation: {name}")


def _rand_brightness(inputs: torch.Tensor) -> torch.Tensor:
    offset = torch.rand(
        inputs.shape[0], 1, 1, 1, device=inputs.device, dtype=inputs.dtype
    ) - 0.5
    return inputs + offset


def _rand_saturation(inputs: torch.Tensor) -> torch.Tensor:
    mean = inputs.mean(dim=1, keepdim=True)
    scale = torch.rand(
        inputs.shape[0], 1, 1, 1, device=inputs.device, dtype=inputs.dtype
    ) * 2
    return (inputs - mean) * scale + mean


def _rand_contrast(inputs: torch.Tensor) -> torch.Tensor:
    mean = inputs.mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rand(
        inputs.shape[0], 1, 1, 1, device=inputs.device, dtype=inputs.dtype
    ) + 0.5
    return (inputs - mean) * scale + mean


def _rand_translation(inputs: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    shift_x = int(inputs.shape[2] * ratio + 0.5)
    shift_y = int(inputs.shape[3] * ratio + 0.5)
    translation_x = torch.randint(
        -shift_x,
        shift_x + 1,
        size=(inputs.shape[0], 1, 1),
        device=inputs.device,
    )
    translation_y = torch.randint(
        -shift_y,
        shift_y + 1,
        size=(inputs.shape[0], 1, 1),
        device=inputs.device,
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(inputs.shape[0], device=inputs.device),
        torch.arange(inputs.shape[2], device=inputs.device),
        torch.arange(inputs.shape[3], device=inputs.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, inputs.shape[2] + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, inputs.shape[3] + 1)
    padded = F.pad(inputs, (1, 1, 1, 1))
    return (
        padded.permute(0, 2, 3, 1)[grid_batch, grid_x, grid_y]
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _rand_cutout(inputs: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    cutout_height = int(inputs.shape[2] * ratio + 0.5)
    cutout_width = int(inputs.shape[3] * ratio + 0.5)
    offset_x = torch.randint(
        0,
        inputs.shape[2] + (1 - cutout_height % 2),
        size=(inputs.shape[0], 1, 1),
        device=inputs.device,
    )
    offset_y = torch.randint(
        0,
        inputs.shape[3] + (1 - cutout_width % 2),
        size=(inputs.shape[0], 1, 1),
        device=inputs.device,
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(inputs.shape[0], device=inputs.device),
        torch.arange(cutout_height, device=inputs.device),
        torch.arange(cutout_width, device=inputs.device),
        indexing="ij",
    )
    grid_x = torch.clamp(
        grid_x + offset_x - cutout_height // 2,
        min=0,
        max=inputs.shape[2] - 1,
    )
    grid_y = torch.clamp(
        grid_y + offset_y - cutout_width // 2,
        min=0,
        max=inputs.shape[3] - 1,
    )
    mask = torch.ones(
        inputs.shape[0],
        inputs.shape[2],
        inputs.shape[3],
        device=inputs.device,
        dtype=inputs.dtype,
    )
    mask[grid_batch, grid_x, grid_y] = 0
    return inputs * mask.unsqueeze(1)


def _diff_augment(inputs: torch.Tensor) -> torch.Tensor:
    inputs = _rand_brightness(inputs)
    inputs = _rand_saturation(inputs)
    inputs = _rand_contrast(inputs)
    inputs = _rand_translation(inputs)
    return _rand_cutout(inputs)


class _PatchDiscriminator(nn.Module):
    convolution: Callable[..., nn.Module]

    def __init__(
        self,
        input_nc: int,
        *,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "batch",
        use_sigmoid: bool = False,
        getIntermFeat: bool = True,
        activation: str = "leaky_relu",
        apply_noise: bool = False,
    ):
        super().__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.noise = _ApplyNoise(input_nc) if apply_noise else nn.Identity()

        kernel_size = 4
        padding = 2
        blocks = [
            [
                self.convolution(
                    input_nc,
                    ndf,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                ),
                _activation(activation),
            ]
        ]
        channels = ndf
        for _ in range(1, n_layers):
            previous_channels = channels
            channels = min(channels * 2, 512)
            blocks.append(
                [
                    self.convolution(
                        previous_channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                    ),
                    _normalization(channels, norm_type),
                    _activation(activation),
                ]
            )

        previous_channels = channels
        channels = min(channels * 2, 512)
        blocks.append(
            [
                self.convolution(
                    previous_channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ),
                _normalization(channels, norm_type),
                _activation(activation),
            ]
        )
        blocks.append(
            [
                self.convolution(
                    channels,
                    1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            ]
        )
        if use_sigmoid:
            blocks.append([nn.Sigmoid()])

        # Match the existing intermediate-feature path: raw logits are the
        # final executed block. An optional sigmoid is retained only for the
        # non-intermediate path, as in the original PatchGAN implementation.
        self.block_count = n_layers + 2
        if getIntermFeat:
            for index, block in enumerate(blocks):
                setattr(self, f"model{index}", nn.Sequential(*block))
        else:
            self.model = nn.Sequential(
                *(layer for block in blocks for layer in block)
            )

    def _augment(self, inputs: torch.Tensor) -> torch.Tensor:
        return _diff_augment(inputs)

    def forward(self, inputs: torch.Tensor, apply_diffaug: bool = False):
        transformed = self.noise(inputs)
        if apply_diffaug:
            transformed = self._augment(transformed)
        if not self.getIntermFeat:
            return self.model(transformed), []

        features = []
        for index in range(self.block_count):
            transformed = getattr(self, f"model{index}")(transformed)
            features.append(transformed)
        return transformed, features


class NLayerDiscriminator(_PatchDiscriminator):
    convolution = nn.Conv2d


class NLayerDiscriminator3D(_PatchDiscriminator):
    convolution = nn.Conv3d

    def _augment(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        frames = rearrange(inputs, "b c t h w -> (b t) c h w")
        frames = _diff_augment(frames)
        return rearrange(frames, "(b t) c h w -> b c t h w", b=batch_size)
