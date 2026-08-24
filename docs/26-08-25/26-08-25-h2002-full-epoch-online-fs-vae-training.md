# H200-2 Full-Epoch Online FoundationStereo + StereoVAE Training

## Purpose and status

- Status: **in progress** on 2026-08-25.
- Goal: train the merged alternating single/four-frame StereoVAE for one full
  H200-2-local train epoch, with online FoundationStereo targets generated
  serially before each VAE update.
- tmux: `stereo-full-epoch-h2002-260825`.
- Output:
  `/data/home/frank/experiments/stereo_merged_fs_vae_full_epoch_h2002_20260825_v1`.
- Log: `run.log` under the output directory.

## Frozen code and data

- Branch: `merged-fs-vae-single-four-profiling`.
- Exact training SHA:
  `e93a7aaf8b4dad7b3b54c03e7f4f4e56656fe1b8`.
- H200-2 worktree was clean at launch. No old temporal-incompatible checkpoint
  is loaded; initialization is random.
- Dataset root: `/data/shared/datasets/umi_lerobot_v3_260714`.
- H200-2-local manifest:
  `/data/home/frank/experiments/stereo_lerobot_cpu_20260824_approval1/h200_2_local_manifest_v1.jsonl`.
- Manifest SHA256:
  `96024f091bcf7aca844b4d4b99fad2eb6cb0f420aa693f1431340b79ac5fa53e`.
- Train/validation/test samples: 694,686 / 38,454 / 38,810.

## Resolved training contract

- One node, eight H200 GPUs, BF16, per-device BS24, global batch 192, GA=1.
- `max_steps=3619`, representing one complete train-manifest epoch.
- Temporal schedule is derived from completed generator updates:
  `four_frame -> single_frame -> four_frame -> single_frame`.
- Single-frame source index is 0.
- PEG backend is `conv2d_t1_slice`.
- DDP uses `static_graph=False` and `find_unused_parameters=True`.
- Seed is 1234.
- Adam with LR/min LR `1e-4/1e-4`, LR warmup 20 updates, KL warmup 100
  updates, and gradient clipping 1.0.
- RGB/disparity/gradient/KL/LPIPS weights are `1.0/1.0/0.1/1e-6/1.0`.
  All GAN losses are disabled.

## Online FoundationStereo contract

- Repo: `/data/home/frank/projects/FoundationStereo`, clean SHA `6e880681`.
- Checkpoint:
  `/data/home/frank/artifacts/foundation-stereo/23-51-11/model_best_bp2.pth`.
- Checkpoint SHA256:
  `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`.
- 32 iterations, pair microbatch 48, bidirectional inference and LR consistency.
- Disparity range 0.5--112 px, absolute LR threshold 1.0 px, relative threshold
  0.05.
- Online GT cache is disabled because H200-2 has only about 1.9 TiB free and
  the current cache representation does not have a validated capacity bound.

## Validation, logging, and checkpoints

- Validation every 500 optimizer updates, limited to 512 samples.
- Checkpoint every 500 updates, including `last.ckpt`.
- WandB is enabled in `WANDB_MODE=offline` because H200-2 has no configured
  online API key.
- Image/video logging every 750/1500 updates.
- Torch profiler is disabled. Lightweight complete-step timing remains enabled
  and will write `step_timings.json` at training end.

## Launch evidence and resource exception

- The initial launch command stopped before creating the output because its
  process guard matched its own command text. The retry changed only the guard
  to the non-self-matching `[t]rain_stereo_vae.py` form; no training or output
  was created by the first attempt.
- At the successful launch, unrelated LIBERO evaluation processes owned by
  `melody`/`maxliu` occupied about 14.3 GiB on GPUs 3, 4, and 6 and used roughly
  26--34% compute. The user explicitly authorized sharing those GPUs. No
  unrelated process was stopped or modified.
- The completed H200-1 BS24 acceptance used about 115.23 GB maximum allocated
  per rank. The shared H200-2 GPUs had about 128.87 GiB free at preflight;
  sharing therefore has limited headroom and may affect throughput or cause an
  OOM if the unrelated workloads grow.
- Startup health check: tmux is present, the parent plus eight DDP ranks exist,
  all ranks entered distributed initialization, and the log has no traceback or
  OOM. No optimizer update existed at the snapshot.

## ETA and remaining checks

- Initial estimate at launch: 17--22 hours for the training body, followed by
  about 15--30 minutes for final checkpoint and artifact validation. This uses
  the H200-1 BS24 non-trace median of about 16.4 seconds/update, widened for the
  explicitly shared H200-2 GPUs.
- Update the ETA after a later user-requested status snapshot provides real
  stable H200-2 steps. Do not poll continuously.
- Completion gates: finite losses, correct single/four counters, no DDP/OOM
  failure, validation records, checkpoint cadence, final `last.ckpt`, offline
  WandB media artifacts, and strict checkpoint load.
