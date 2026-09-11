"""Use the pinned official Wan VAE without importing the diffusion pipeline."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluation.stage_a.contract import sha256_file

SOURCE_SHA = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
WEIGHT_SHA = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"


class WanReconstructor:
    def __init__(self, source_root, checkpoint, device):
        root = Path(source_root).resolve()
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
        if head != SOURCE_SHA or dirty:
            raise ValueError("Wan source must be clean at the frozen commit")
        if sha256_file(checkpoint) != WEIGHT_SHA:
            raise ValueError("Wan checkpoint SHA256 mismatch")
        spec = importlib.util.spec_from_file_location("wan22_official_vae", root / "wan/modules/vae2_2.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.vae = module.Wan2_2_VAE(vae_pth=str(checkpoint), dtype=torch.float32, device=device)
        self.latent_shapes = {}

    @torch.inference_mode()
    def __call__(self, target):
        """Reconstruct target eyes [B,V,3,T,H,W] in the project [-.5,.5] domain."""
        if target.ndim != 6 or target.shape[2] != 3 or target.shape[3] not in (1, 4):
            raise ValueError("Wan evaluation expects [B,V,3,T,H,W], T=1 or 4")
        if not torch.isfinite(target).all() or target.abs().max() > 0.5:
            raise ValueError("Wan input must be finite RGB in [-0.5,0.5]")
        batch, views, _, frames, height, width = target.shape
        if height % 16 or width % 16:
            raise ValueError("Wan spatial dimensions must be divisible by 16")
        outputs = []
        latents = []
        for clip in target.flatten(0, 1):
            clip = clip.float().unsqueeze(0) * 2
            if frames == 4:
                clip = torch.cat((clip, clip[:, :, -1:]), dim=2)
            # Same official posterior mean/scale and decoder; explicit FP32.
            latent = self.vae.model.encode(clip, self.vae.scale)
            decoded = self.vae.model.decode(latent, self.vae.scale)
            if decoded.shape != clip.shape or not torch.isfinite(decoded).all():
                raise ValueError("Wan reconstruction shape or finite-value check failed")
            self.latent_shapes[str(frames)] = list(latent.shape[1:])
            latents.append(latent[0])
            # Official public decode clamps to [-1,1]. Scoring is on the original frames.
            outputs.append(decoded[0, :, :frames].clamp(-1, 1) / 2)
        return SimpleNamespace(
            rgb=torch.stack(outputs).reshape(batch, views, 3, frames, height, width),
            latent=torch.stack(latents).unflatten(0, (batch, views)),
        )
