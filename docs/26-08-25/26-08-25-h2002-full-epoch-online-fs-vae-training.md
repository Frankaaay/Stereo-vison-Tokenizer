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

## User-requested stop and evaluation handoff

- At `2026-08-25T18:57:08+08:00`, the user requested that training stop so the
  model could be evaluated. `Ctrl-C` was sent to tmux session
  `stereo-full-epoch-h2002-260825`; Lightning reported `Detected
  KeyboardInterrupt, attempting graceful shutdown`.
- The final completed in-memory update shown by the log was 3007/3619 (83%).
  The post-stop check found zero `train_stereo_vae.py` processes and zero
  training memory allocated on all eight GPUs. The tmux session was retained
  for audit history; no unrelated process was stopped or modified.
- The evaluation checkpoint is the last complete periodic save at update 3000:
  `/data/home/frank/experiments/stereo_merged_fs_vae_full_epoch_h2002_20260825_v1/stereo-vae/pinjyeja/checkpoints/last.ckpt`.
  It is 729,458,227 bytes and matches the adjacent
  `epoch=0-step=3000.ckpt` size. Updates 3001--3007 were not checkpointed and
  must not be described as part of the evaluation model.
- Offline WandB run:
  `/data/home/frank/experiments/stereo_merged_fs_vae_full_epoch_h2002_20260825_v1/wandb/offline-run-20260825_010449-pinjyeja`.
  The extracted validation total loss changed from 0.705995 to 0.434044 for
  four-frame mode and from 0.693977 to 0.424743 for single-frame mode between
  updates 500 and 3000 (reductions of 38.52% and 38.80%). At update 3000,
  four-frame validation total loss was 0.009301 higher than single-frame
  validation total loss (2.19% relative to single-frame).
- The loss history is finite and continues decreasing through the six fixed
  validation points. The dominant final weighted validation contribution is
  LPIPS/perceptual loss; this observation is descriptive and is not yet an
  evaluation-quality conclusion.

## Original ETA and remaining checks

- Initial estimate at launch: 17--22 hours for the training body, followed by
  about 15--30 minutes for final checkpoint and artifact validation. This uses
  the H200-1 BS24 non-trace median of about 16.4 seconds/update, widened for the
  explicitly shared H200-2 GPUs.
- Update the ETA after a later user-requested status snapshot provides real
  stable H200-2 steps. Do not poll continuously.
- Completion gates: finite losses, correct single/four counters, no DDP/OOM
  failure, validation records, checkpoint cadence, final `last.ckpt`, offline
  WandB media artifacts, and strict checkpoint load.
- The one-epoch completion gate is intentionally not met because the user
  stopped at update 3007. Before reporting evaluation results, still verify a
  strict load of the update-3000 checkpoint and record the exact evaluation
  recipe/output; do not infer evaluation quality from training loss alone.

## Test-split evaluation implementation and launch plan

- The user requested a full test-split evaluation plus several reconstruction
  cases. The pre-existing `eval_stereo_vae.py` only accepted `train/val` and
  assumed cached disparity fields, so it could not evaluate the LeRobot online
  dataset or its separate 38,810-sample test split.
- The scoped evaluation extension adds `test`, constructs the same frozen
  PyTorch FoundationStereo teacher contract used by training, and generates the
  disparity/valid-mask target once per batch before evaluating both
  `four_frame` and `single_frame`. Posterior sampling remains disabled and the
  VAE checkpoint load remains `strict=True`.
- An eight-process torchrun evaluation assigns complete shards to ranks without
  padding. Metric accumulators are summed across ranks and a full run fails if
  either temporal mode's global sample count differs from the exact split size.
  This avoids the duplicate samples introduced by training's equal-length DDP
  sampler padding.
- Metrics remain RGB L1 plus per-view disparity EPE, depth AbsRel, depth RMSE,
  valid-pixel count, and total sample count. Evaluation precision is explicit;
  the planned run uses BF16 and per-rank BS24.
- Six visualization cases are selected deterministically from six distinct test
  episodes with seed 1234. Each PNG places the four left-eye input frames beside
  four-frame reconstructions and the frame-0 single-frame reconstruction for
  all three views. The output directory is fail-closed and will not overwrite
  an existing result.
- Planned fresh output root:
  `/data/home/frank/experiments/stereo_merged_fs_vae_test_eval_step3000_h2002_20260825_v1`.
  Launch is gated on local source tests, H200-2 strict-load/two-batch smoke, exact
  pushed SHA, clean server worktree, and a fresh output path. ETA will be based
  on the smoke throughput rather than the stopped training estimate.

### Evaluation deployment and smoke status

- Evaluation implementation commit:
  `d331025d2134f5ed480d4193b0e45ad1f32c2c97`. Local compile/source-boundary
  checks passed 27/27. H200-2 fast-forwarded cleanly to this SHA; the eval
  entrypoint-specific server tests passed 14/14. The broader server source test
  retains the known unrelated failure from 28 legacy
  `OmniTokenizer/__pycache__/*.pyc` files; they were not deleted.
- Smoke v1 stopped at argument parsing before model/data/GPU work because the
  launch command omitted nine parser-required loss-contract arguments. It is an
  operator launch omission rather than a model/evaluation failure. Its log is
  retained at
  `/data/home/frank/experiments/stereo_merged_fs_vae_test_eval_step3000_h2002_smoke_20260825_v1/run.log`.
- Smoke v2 uses the complete frozen loss contract, eight GPUs, BS24/rank, BF16,
  two batches/rank, both temporal modes, and six visualization cases. Output:
  `/data/home/frank/experiments/stereo_merged_fs_vae_test_eval_step3000_h2002_smoke_20260825_v2`.
  At the `2026-08-25T19:13:05+08:00` one-shot health snapshot, torchrun plus
  eight eval ranks were alive with no traceback/OOM, still in distributed model
  and FoundationStereo initialization; no first batch or metrics artifact yet.
  GPU memory was about 0.5--1.0 GiB for the new ranks, while the pre-existing
  small jobs on GPUs 0 and 6 remained untouched. The initial 1--2 minute smoke
  estimate was therefore too short; do not launch the full test run until this
  smoke exits successfully and its PNG/JSON artifacts are verified.

### Formal test evaluation launch

- Smoke v2 completed successfully at `2026-08-25T19:17:43+08:00`. It evaluated
  exactly 384 samples (two BS24 batches on each of eight ranks) for both modes,
  wrote finite metrics, exited without OOM/traceback, and produced six PNGs plus
  `cases.json`. Rank-0 progress measured 41 seconds for two batches, or about
  20.6 seconds/local batch after initialization.
- The full 38,810-sample test evaluation started at
  `2026-08-25T19:21:52+08:00` in tmux session
  `stereo-test-eval-h2002-260825`. Server code remains the tested evaluation SHA
  `d331025d2134f5ed480d4193b0e45ad1f32c2c97`; the later docs-only commit is not
  synced into the live run.
- Formal output:
  `/data/home/frank/experiments/stereo_merged_fs_vae_test_eval_step3000_h2002_20260825_v1`.
  The run uses eight GPUs, BS24/rank, BF16, PyTorch FoundationStereo at 32
  iterations/pair-microbatch 48, posterior mean, stereo eye mode, both temporal
  modes, frame-0 single source, exact non-padding shard assignment, and six
  deterministic test cases.
- Startup health check at `2026-08-25T19:22:07+08:00`: tmux and torchrun
  processes are present, the recorded code SHA matches, and the log has no
  immediate traceback/OOM. The process was still in distributed startup and had
  not allocated model GPU memory at this early snapshot.
- ETA estimated at launch: about 75--85 minutes for the evaluation body, based
  on 202 local batches/rank and the smoke's 20.6 seconds/batch plus observed
  initialization overhead. Allow another 2--5 minutes for six case PNGs,
  cross-rank aggregation, exact 38,810-sample count verification, `metrics.json`,
  and exit-marker checks. Do not poll continuously; refresh only on a later
  user-requested status snapshot.
