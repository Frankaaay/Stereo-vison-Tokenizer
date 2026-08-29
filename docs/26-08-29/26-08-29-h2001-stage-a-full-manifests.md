# H200-1 Stage A with formal manifests

## Purpose

Build formal episode-level 90/5/5 manifests for the H200-1 Hy and LIBERO
sources, retain the existing full UMI LeRobot 90/5/5 manifest, and launch the
authorized single-node eight-GPU Stage A run. The run is fresh and does not
strict-resume the earlier 40-update profiling checkpoint.

## Code and runtime

- Local and H200-1 branch: `hezhou-las2-h`
- Code SHA: `4f3d167793ccb3cfc8d99f88c5551c18ea90ee59`
- H200-1 repository: `/data/home/frank/projects/Stereo-vison-Tokenizer`
- Runtime: `/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260828/venv`
- Launch time: 2026-08-29 01:13 CST
- tmux: `stereo-stagea-h2001-20260829-v1`

No repository source was changed. Manifest construction was a CPU-only pass
whose outputs were written below `/data/home/frank/runtime`; the training run
writes below `/data/home/frank/experiments`.

## Formal manifests

Hy was checked table by table against the Lance `episode_index` and
`frame_index` identities. An accepted episode has exactly its declared number
of unique contiguous frame indices `[0, length)`. Thirty-five metadata episodes
from `table_014` were absent from the Lance identity inventory and were
excluded. LIBERO was checked for the existence of both declared camera videos.
The accepted records were deterministically ranked with seed 1234 and assigned
an exact episode-level 90/5/5 split.

- Hy: `/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260829/manifests/hy_formal_90_5_5_v1.jsonl`
  - SHA256: `b25efc945ccd7e7afd2f1a76393ea19adde8fa072e1e9a2ca6348e0e5c1a45f9`
  - accepted/rejected episodes: 57,913 / 35
  - train/val/test episodes: 52,121 / 2,895 / 2,897
  - train/val/test windows: 4,032,399 / 220,605 / 223,371
- LIBERO: `/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260829/manifests/libero_formal_90_5_5_v1.jsonl`
  - SHA256: `0299354a7225e979f6b9ff5fb3e26a975c811d2d41af44e042a6eade3f24bbf4`
  - accepted/rejected episodes: 1,712 / 0
  - train/val/test episodes: 1,540 / 85 / 87
  - train/val/test windows: 30,694 / 1,670 / 1,828
- UMI LeRobot: `/data/home/frank/experiments/stereo_lerobot_cpu_20260824_approval1/h200_1_provisional_manifest_v1.jsonl`
  - SHA256: `31457d9b1834953024d7e7ff59f5a21b74500d3ece4c19c755a14aff3dccaf6d`
  - train/val/test episodes: 56,363 / 3,131 / 3,132
  - train/val/test windows: 1,246,294 / 68,932 / 69,167
  - Its rectification status remains the previously accepted provisional user
    assumption; this run does not claim a new rectification audit.

## Training contract

- Output: `/data/home/frank/experiments/stereo-three-source-stagea-h2001-20260829-v1`
- Log: `run.log` under that output
- H200-1, 8 GPUs, BF16
- per-device batch 24, GA 1, global batch 192
- four logical-mode weights: 35:35:15:15
- mono source weights Hy:LIBERO = 9:1; Hy uses `cam_high` only
- real LAS2-H stereo teacher and DA3-BASE mono teacher
- GAN disabled; LPIPS weight 1.0
- workers 8, prefetch factor 2
- fresh run, maximum 100,000 generator updates
- checkpoint every 1,000 updates; online validation every 2,000 updates

## Launch status and ETA

The first startup snapshot found the tmux session and all eight Lightning
processes alive with no traceback. The snapshot occurred during distributed
model/teacher initialization, before the first training update and before
material GPU allocation, so it is not a throughput measurement.

Initial ETA, estimated at launch from the earlier same-contract 40-update v8
measurement, is approximately 20--24 hours for the 100,000-update training
body, plus checkpoint and final validation time. This is only a historical
estimate until a later user-requested status snapshot provides stable full-data
throughput. The user intends to stop the run manually on the morning of
2026-08-29 before starting Stage B.

## UMI decode audit after the Stage A failure

The run later failed after approximately update 3899 when rank 2 could not
decode `observation.images.left_wrist_right` near timestamp `116.335026` in
`shard_1803`. Other ranks subsequently timed out in a collective; the NCCL
watchdog was secondary to the DataLoader failure. Checkpoints through update
3000 remain, but no resume was started.

A targeted H200-1 check decoded the complete physical MP4 sequentially and
also repeated the training seek. Both paths found the nearest PTS at
`116.268359375`, an error of about 66.67 ms, which exceeds the frozen 50 ms
training tolerance. This proves a real timestamp-availability gap rather than
a seek-only false rejection.

The authorized full CPU audit started at 2026-08-29 10:23 CST:

- tmux: `umi-decode-audit-h2001-20260829-v1`
- output: `/data/home/frank/runtime/umi-lerobot-decode-audit-h2001-20260829-v1`
- workers: 24 of 192 logical CPUs
- audit script SHA256:
  `a64b5bcb3fe62b1f8d9243c885633e9e73cb14018b43e498edb244138740ad23`
- input manifest SHA256:
  `31457d9b1834953024d7e7ff59f5a21b74500d3ece4c19c755a14aff3dccaf6d`

Each physical MP4 is sequentially decoded once. Every declared four-frame
target for every episode is then checked against its actual decoded PTS for all
six streams. If any target in any stream exceeds 50 ms error, the complete
episode is conservatively rejected. Existing train/val/test assignments are
preserved. The final summary will report accepted and rejected episode/window
counts, source episode-hours, stride-covered hours, six-stream camera-hours,
failure reasons, and the filtered manifest SHA256.

The startup health check found the tmux and audit process alive without an
immediate exception. No shard had completed yet, so the initial 3--6 hour ETA
is based on the targeted full-file decode and 24-way concurrency rather than a
measured steady-state shard rate. Final aggregation should take only minutes
after the decode body.

## Mode-aware logical-update implementation

Local implementation started from clean branch `hezhou-las2-h` at base commit
`c5443e5bf8603ed66a952c724f8a1c5d0f559aaa`. The implementation is pending
user diff review and has not been pushed or synchronized to H200-1. No GPU
profile or training run was started, and the UMI CPU decode audit was not
interrupted.

The new four-mode batch contract is:

| mode | BS/GPU | micro-batches/logical update | effective global batch on 8 GPUs |
|---|---:|---:|---:|
| mono/single_frame | 48 | 1 | 384 |
| mono/four_frame | 48 | 1 | 384 |
| stereo/single_frame | 48 | 1 | 384 |
| stereo/four_frame | 24 | 2 | 384 |

The sampler keeps the seeded `35:35:15:15` weights in logical-update space and
emits the two `stereo/four_frame` physical micro-batches consecutively. The
model divides each loss by the mode accumulation factor and advances optimizer,
scheduler, generator/mode/sample counters only at the logical boundary. The
checkpoint contract now records per-mode BS, accumulation, effective global
batch, physical-batch counters, and a logical-update ABI version. Checkpointing
inside an incomplete accumulation window is rejected. Older Stage A
checkpoints are not strict-resume compatible with the new ABI; profiling must
use a fresh run or an explicitly authorized weights-only warm start. This
StereoVAE path has no EMA implementation, so no EMA behavior was added.

The launcher retains the scalar BS/GA path for legacy non-four-mode runs. In
four-mode training it keeps global `GRAD_ACCUMULATES=1`, accepts
`MODE_BATCH_SIZES` and `MODE_GRAD_ACCUMULATES`, and fail-closes unless every
mode has the configured effective global batch. Resolved config, run manifest,
checkpoint, and timing output all carry the per-mode contract. Validation uses
the per-mode batch sizes but does not duplicate batches for gradient
accumulation.

Local validation before user review:

- `python -m py_compile` passed for all modified Python source and tests.
- `python -m unittest tests.stereo.test_entrypoints_source
  tests.stereo.test_source_boundary`: 25 tests passed.
- Git Bash `bash -n scripts/stereo/train_stereo_vae.sh`: passed.
- Launcher fail-closed probes accepted `48:48:48:24` plus `1:1:1:2` through
  the batch-contract gate and rejected a `48:48:48:25` mismatch as effective
  global batch 400 versus configured 384. Both probes stopped before data/model
  access and did not launch Python.
- A dependency-free sampler smoke verified 23 physical batches for one
  20-logical-update `7:7:3:3` cycle and exact resume suffix preservation.
- `git diff --check`: passed.
- Tensor/runtime tests were not executed locally because the Windows Python
  environment has neither Torch nor PyTorch Lightning. They remain a required
  gate in the pinned H200 runtime after commit/push and exact-SHA sync.

## Mode-aware profiling checkpoint boundary fix

- Runtime: H200-1, branch `hezhou-las2-h`, commit `25dc4cfa3489f8766d92f13f449c65d50e808096`.
- Profiling output: `/data/home/frank/experiments/stereo-mode-aware-bs384-profile-h2001-v1`.
- Observed result: the run consumed 340 physical DataLoader batches, validated at
  `val/mixed/total_loss ~= 0.940`, then `save_last` failed with
  `refusing to checkpoint an incomplete logical update`.
- Root cause: `online_val_check_interval_steps` was passed directly to Lightning,
  which interprets it as physical batches. Under mode-aware GA, 340 logical updates
  expand to more than 340 physical batches, so validation and its checkpoint hook
  could run after the first micro-batch of a `stereo/four_frame` logical update.
- Fix: translate the logical validation cadence to the exact physical-batch count
  implied by the deterministic mode schedule and per-mode accumulation contract.
  Periodic validation must start on a schedule-cycle boundary and cover whole
  20-update cycles; a final-only interval may cover all remaining logical updates.
  The incomplete-checkpoint guard remains unchanged and fail-closed.
- Validation status: `py_compile` and `git diff --check` passed; the 25 local
  source-boundary tests passed. A read-only calculation in the pinned H200-1 runtime
  confirmed that this 340-logical-update schedule expands to exactly 391 physical
  batches (`len == iterated count == 391`). The committed H200 runtime gate and GPU
  rerun status are recorded below.

### H200-1 v2 rerun

- Code SHA: `aec9521b4149684c867347bc112e6a66c421d4a5`.
- H200 runtime gate: 57 targeted tests passed with 4 existing warnings.
- Launch time: 2026-08-29 12:06 CST.
- Status at the initial health check: running; all eight DDP ranks registered and
  no immediate traceback was present.
- tmux: `stereo-mode-aware-bs384-profile-h2001-v2`.
- Output: `/data/home/frank/experiments/stereo-mode-aware-bs384-profile-h2001-v2`.
- Log: `/data/home/frank/experiments/stereo-mode-aware-bs384-profile-h2001-v2/run.log`.
- Timing result target:
  `/data/home/frank/experiments/stereo-mode-aware-bs384-profile-h2001-v2/step_timings.json`.
- Initial ETA at 12:07 CST: training and validation should finish around 12:15-12:17;
  checkpoint and timing-artifact completion should follow by about 12:18. This is
  based on the v1 wall time of about 7m41s plus 51 previously truncated physical
  micro-batches, not yet on a stable v2 throughput sample.

### H200-1 v2 final result

- Exit code: 0; tmux and all training processes exited normally, and all GPUs
  returned to 0 MiB.
- Completed 340 logical updates as 391 physical micro-batches in about 7m54s,
  followed by validation with `val/mixed/total_loss ~= 0.943`.
- Checkpoints: `epoch=0-step=340.ckpt`, `last.ckpt`, and
  `best-epoch=0-step=340.ckpt` under the v2 output `train/checkpoints` directory.
- `step_timings.json` was generated successfully. After excluding 20 warmup logical
  updates per mode, the aggregate effective-global-sample throughput was about
  804.3 samples/s for mono/single, 298.0 for mono/four, 449.1 for stereo/single,
  and 112.4 for stereo/four. Across all 260 stable logical updates it was about
  325.0 samples/s.
- Conclusion: validation and checkpointing now occur on a complete logical-update
  boundary; the original incomplete-logical-update failure is resolved.

## Stage B Image-GAN capacity smoke

- Status: started on H200-1 at 2026-08-29 13:02 CST.
- Code SHA: `dcfdf3fb2264173bf7e48679e32a5a5f14099950`.
- tmux: `stereo-mode-aware-gan005-oom-smoke-h2001-v1`.
- Output: `/data/home/frank/experiments/stereo-mode-aware-gan005-oom-smoke-h2001-v1`.
- Contract: BF16, eight GPUs, mode batch sizes `48:48:48:24`, mode-aware GA
  `1:1:1:2`, effective global batch 384, real LAS2-H and DA3-BASE teachers,
  Image GAN weight 0.005 from update 0, Video GAN and feature matching disabled.
- Scope: 40 logical updates (two complete `7:7:3:3` mode cycles), validation,
  checkpoint, timing, and peak-memory collection.
- UMI input was changed from the provisional manifest to the decode-verified
  manifest with SHA256
  `5e8f58c769549372af070a6132ad826bd7172aaeabcebebff84426e66bc2120f`.
- Initial health check: all ranks were initializing without an immediate traceback.
  Initial ETA at 13:03 CST is about 3-6 minutes for the smoke body and validation;
  this is a conservative estimate because no stable GAN-on throughput existed yet.

### Image-GAN smoke result and formal run

- The smoke exited with code 0, produced validation/checkpoints/timing output, and
  returned all GPUs to 0 MiB. The largest observed per-rank allocation was about
  107.4 GiB in stereo/four; the largest reserved value was about 137.8 GiB in
  mono/four. This passed but leaves limited allocator headroom.
- The fresh formal run started at 2026-08-29 13:10 CST on code SHA
  `dcfdf3fb2264173bf7e48679e32a5a5f14099950`.
- tmux: `stereo-three-source-stageb-gan005-h2001-20260829-v1`.
- Output: `/data/home/frank/experiments/stereo-three-source-stageb-gan005-h2001-20260829-v1`.
- It uses the same BS/GA, teacher, verified-manifest, Image-GAN 0.005, and
  four-mode weighting contracts as the successful smoke; Video GAN and feature
  matching remain disabled. The target is 100,000 generator updates.
- Validation is every 2,000 generator updates. Lightning counts both generator and
  discriminator optimizer steps, so its checkpoint interval is configured as 2,000
  Lightning steps to obtain one checkpoint per 1,000 generator updates.
- Per-step timing callbacks are disabled for the formal run to avoid a CUDA
  synchronization on every physical batch.
- Initial health check: tmux and resolved config exist, rank 0 entered distributed
  initialization, and no immediate traceback was present.
- Initial ETA at 13:12 CST: roughly 35-50 hours for 100,000 generator updates,
  with final validation/checkpoint completion shortly afterward. This range is
  derived from the short GAN-on smoke and will be refreshed from a later stable
  formal-run snapshot; the expected completion window is approximately Sunday
  midnight through Monday afternoon CST.

### Correction: Stage A restarted without GAN

- The user corrected the phase contract: Stage A must not enable Image GAN. The
  incorrectly launched Stage B run was stopped at 2026-08-29 13:14 CST. Its tmux
  session was removed, its remaining DDP children were terminated by their exact
  output-path match, and all eight GPUs returned to 0 MiB. Its output directory was
  retained as an aborted-run record and was not reused.
- A new fresh Stage A run started at 13:15 CST:
  `stereo-three-source-stagea-bs384-h2001-20260829-v2`.
- Output: `/data/home/frank/experiments/stereo-three-source-stagea-bs384-h2001-20260829-v2`.
- Resolved config confirms `gan_enabled=false`, Image GAN 0, Video GAN 0, feature
  matching 0, no resume checkpoint, 100,000 generator updates, checkpoint every
  1,000 updates, validation every 2,000 updates, and the decode-verified UMI
  manifest. All four mode-aware BS/GA contracts remain unchanged.
- Initial health check found the tmux alive during model/DDP initialization with no
  immediate traceback. Based on the completed GAN-off profile, the initial body ETA
  is about 32-38 hours, plus periodic/final validation and checkpoint overhead; this
  will be refreshed from the first stable formal-run snapshot.
