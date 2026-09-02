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
- Implemented and pushed commits:
  `4327f14333aba514d6863b405a6778b4ec28a755` and
  `dbcae41d11f3debe93c1470276056111c6a7b455`.
- Local Python compile and `git diff --check` passed. Windows lacked the runtime
  dependencies for the tensor tests.
- H200-1 directed runtime suite passed: 44 tests, 4 warnings, in 7.87 seconds;
  `bash -n scripts/stereo/train_stereo_vae.sh` also passed.
- The immutable Hy three-camera manifest is
  `/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260902-threeview-v1/manifests/hy_formal_90_5_5_threeview_v2.jsonl`,
  SHA256 `7ea00b7897811eaae47017531653b235de06b6e2f482e6e94836e3dbfea09c7b`.
  It preserves 57,913 accepted episodes and the exact split counts
  52,121/2,895/2,897. Effective three-camera train/val/test windows are
  12,097,197/661,815/670,113. Decode smoke passed for T=1 and T=4 on all three
  cameras.
- The source checkpoint SHA256 is
  `d22c11a7630bc36bb6168acc1452ddf3ba21c257418a08a12a76b8fe41a348b3`.
  A strict continuation preflight loaded the model and inherited generator
  update 44,000 and the historical mode sample counters; a fresh optimizer was
  created and the scheduler was aligned directly to update 44,000 at LR 1e-4.

## Launch record

- Attempt v1 stopped before any update with exit code 1. The first root-cause
  exception was Git safe-directory validation for the existing LAS2-H source
  checkout.
- Attempt v2 was interrupted during initialization after the user corrected the
  desired account/asset provenance. Its output was preserved and not reused.
- The active attempt is v3 in tmux session
  `stereo-stagea-threeview-bs384-h2001-v3` with output root
  `/data/home/frank/experiments/stereo-three-source-stagea-threeview-bs384-h2001-20260902-v3`.
  Launch script SHA256 is
  `18f9e6e81aeeeede7ee98a2825262cfb83948b64805398d44d0c6a092920de04`.
- The one-shot health check found the tmux session alive, eight distributed
  ranks initialized, no exit marker, and a resolved config with source update
  44,000, schedule start update 44,000, and target update 200,000. The run was
  still initializing teachers/models, so no training metric or W&B offline run
  directory existed yet.
- Immediately before launch, the H200 checkout was clean at
  `dbcae41d11f3debe93c1470276056111c6a7b455`. A live `git fetch` retry failed
  because H200 could not resolve `github.com`; launch proceeded on the already
  verified and previously synced SHA per the user's instruction to start
  immediately.
- Attempt v3 subsequently exited with code 134 before completing generator
  update 44,001. The first root-cause exception was not NCCL: rank 4 raised
  `PIL.UnidentifiedImageError` while fetching its first Hy
  `mono/four_frame` batch. Deterministic replay located sample index 9,881,756
  at `hy_rest/table_020`, episode `table_020:10932` (episode index 10,932),
  camera `cam_right_wrist`, window start 0, frame 0. The Lance payload was one
  byte `0x00`, so it was not a JPEG. The other ranks then waited at collective
  sequence 25 and only surfaced the secondary NCCL timeout after 30 minutes.
  Offline W&B run `offline-run-20260902_011444-eaejdrn8` was created, but no
  optimizer update or new checkpoint was produced. All eight GPUs were released.
