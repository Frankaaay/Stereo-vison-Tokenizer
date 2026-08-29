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
