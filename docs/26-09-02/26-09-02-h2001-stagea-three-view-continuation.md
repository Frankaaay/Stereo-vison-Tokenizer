# H200-1 Stage A three-view continuation

## Purpose

Continue the GAN-free Stage A model from the validated generator update 44,000
checkpoint, add the two Hy wrist cameras as equally sampled monocular inputs,
and train to generator update 200,000 on H200-1. This is a controlled
continuation rather than a strict Lightning resume because the Hy manifest and
logical batch contract change.

## Approved contract

- Branch: `hezhou-las2-h`.
- Source checkpoint:
  `/data/home/frank/experiments/stereo-three-source-stagea-bs192-h2001-20260829-v3/train/checkpoints/best-epoch=0-step=44000.ckpt`.
- Source counters: generator 44,000, discriminator 0.
- GAN remains disabled; image GAN, video GAN, and feature matching weights are 0.
- Hy cameras are equally sampled `cam_high`, `cam_left_wrist`, and
  `cam_right_wrist`; the episode split and accepted episode set remain frozen
  to the existing H200-1 formal 90/5/5 Hy manifest.
- Mono source weights remain Hy:LIBERO = 9:1.
- Logical mode weights remain `35:35:15:15`.
- Per-device mode batches are `48:48:48:24`; per-mode accumulation is
  `1:1:1:2`, so every logical update has effective global batch 384 on 8 GPUs.
- BF16, online DA3 and LAS2-H teachers, and disabled online GT cache are retained.
- Maximum generator update is 200,000. The continuation inherits update 44,000,
  creates a fresh optimizer, and aligns the scheduler directly to update 44,000;
  it does not count a new warmup phase.
- W&B logging is enabled with `WANDB_MODE=offline`.

## Counter and manifest semantics

The continuation checkpoint path loads the model state strictly while allowing
only the explicitly changed manifest and batch fields. Historical mode sample
counters remain unchanged. New samples and physical batches are accumulated
from a recorded transition baseline, so later checkpoints can strict-resume
without rewriting the Stage A history. The run manifest records the source
checkpoint SHA256, source counters, old data/batch contract, and the fresh
optimizer/scheduler policy.

The new Hy manifest is derived from the accepted records and exact split labels
of `hy_formal_90_5_5_v1.jsonl`. Each record carries all three Lance camera
columns, and the loader expands one equal-length span per camera. No image data
is copied or re-encoded, and the existing manifest is not overwritten.

## Preparation status

- Local base before implementation: `fa32f1a67bf47dc557bd2b3098a903d421bd9247`.
- Local Python compile and `git diff --check` pass.
- Windows cannot run the directed pytest suite because the current Python lacks
  pytest/Torch/Lightning; those gates must run in the pinned H200-1 runtime.
- H200-1 runtime validation, immutable manifest generation, final SHA/config
  capture, launch identity, log/output paths, and health result are pending.
