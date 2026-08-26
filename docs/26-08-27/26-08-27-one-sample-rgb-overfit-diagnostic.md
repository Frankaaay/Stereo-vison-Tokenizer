# StereoVAE one-sample RGB-only overfit diagnostic

## Status

- Status: implementation and preflight in progress; the five GPU runs have not
  started yet.
- Date: 2026-08-27 (Asia/Shanghai).
- Host: `h200-1` (`frank`), planned GPU mapping 0 through 4.
- Branch: `hezhou-las2-h`.
- Baseline SHA: `80cba3661dc6de4d6967e1edf5a69823dc9d4e5e`.
- Diagnostic commit: pending. The commit will be explicitly labelled as a
  task-scoped diagnostic so it can be reverted as one unit after the experiment.

## Purpose

Test whether the current StereoVAE can memorize fixed RGB pixels independently
of online teacher latency/noise and all auxiliary objectives. The diagnostic is
implemented in the separate
`train_stereo_vae_one_sample_overfit.py` entrypoint; it does not relax the
formal trainer's frozen online-teacher, BS24, or 48-sample four-mode contracts.

## Frozen experiment contract

- Five independent single-process, single-GPU runs; no DDP.
- GPU 0: `mono/single_frame`, one fixed sample, 4,000 updates.
- GPU 1: `mono/four_frame`, the same underlying mono window, 4,000 updates.
- GPU 2: `stereo/single_frame`, one fixed sample, 4,000 updates.
- GPU 3: `stereo/four_frame`, the same underlying stereo window, 4,000 updates.
- GPU 4: joint, the same four mode-tagged samples in the exact repeating order
  `mono/single_frame`, `mono/four_frame`, `stereo/single_frame`,
  `stereo/four_frame`; 16,000 total updates and 4,000 updates per mode.
- BS=1, fixed seed 1234, Adam LR `1e-4`, betas `(0.5, 0.9)`, gradient clipping
  1.0, BF16 autocast, fixed LR, identical strict-loaded initialization.
- Training and evaluation use the posterior mean (`sample_posterior=False`) to
  isolate deterministic pixel memorization.
- RGB L1 weight 1.0. GAN, feature matching, LPIPS, KL, relative-depth, and
  relative-gradient weights are all zero. Online teacher inference is absent.
- Samples are decoded once and kept fixed. No shuffle, stochastic crop, or
  augmentation is used. Single-frame uses source index 0.
- Per-mode milestones: 100, 500, 1,000, 2,000, and 4,000. Joint total-step
  milestones: 400, 2,000, 4,000, 8,000, and 16,000.

Each run writes `resolved_config.json`, a per-step `metrics.jsonl`, milestone
checkpoints, GT/reconstruction comparison PNGs, and `summary.json`. Metrics
include RGB MAE/L1, MSE, PSNR, per-view MAE, GT/prediction adjacent-frame
differences for four-frame modes, input left/right differences and the explicit
left-target mapping for stereo modes, CUDA-synchronized step time, and peak
allocated/reserved memory.

## Runtime and assets

- Unified runtime symlink:
  `/data/home/frank/runtime/stereo-tokenizer-unified-v1` ->
  `/data/home/frank/runtime/da3-base-v1`.
- Task-private overlay:
  `/data/home/frank/runtime/stereo-tokenizer-profile-overlay-v1`.
- Overlay contains only `av==16.0.1`; neither the Hezhou environment nor the
  existing Frank unified environment is modified.
- Stereo manifest:
  `/data/home/frank/experiments/stereo_lerobot_cpu_20260824_approval1/h200_1_provisional_manifest_v1.jsonl`.
- Stereo dataset root: `/data/shared/datasets/umi_lerobot_v3_260714`.
- Mono manifest/cache root:
  `/data/shared/datasets/hy_mono_cam_high_smoke_v1`.

## Preflight evidence

- Remote worktree is clean at the baseline SHA and tracks
  `origin/hezhou-las2-h`.
- Unified Python is 3.12.3 with Torch 2.7.1+cu126 and Lightning 2.5.6; PyAV
  16.0.1 imports from the task-private overlay.
- GPUs 1 through 7 were empty at the initial check. GPU 0 had two root-owned
  TinyNav processes using about 2.45 GiB total. The user explicitly authorized
  stacking the diagnostic on GPU 0 without stopping or modifying those jobs.
- Local source compilation and `git diff --check` pass. The local Python lacks
  Torch, so the new runtime unit test must be executed on `h200-1` after the
  diagnostic commit is synchronized.

## Remaining launch gates

1. Select and record one mono window and one stereo window with visibly
   non-trivial four-frame motion; single/four modes must share their underlying
   window within each eye mode.
2. Commit and push the task-scoped diagnostic, fast-forward the clean H200-1
   clone, and run the focused server tests.
3. Create one shared initialization checkpoint, record its SHA256, and launch
   the five tmux processes with separate output/log paths.
4. Perform exactly one launch health check and report process/GPU mapping,
   first step, immediate errors, and throughput-based ETA.
