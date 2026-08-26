# Four-mode GAN-off step profiling

## Purpose

Measure the current `hezhou-las2-h` training graph without GAN and attribute
wall-clock time across the four mono/stereo x single/four-frame modes. This is
intended to test the claim that `full step - LAS2-H teacher` is VAE time; that
residual also contains data transfer, LPIPS, backward, DDP, optimizer, and
logging.

## Status

GAN-off profiling completed successfully. A strict GAN-on wall-time A/B is
being prepared on the same node/runtime/data contract.

- Branch / commit: `hezhou-las2-h` / `80cba3661dc6de4d6967e1edf5a69823dc9d4e5e`
- Node: H200-1, 8 x H200, BS24/device, global batch 192, GA1, BF16
- tmux: `stereo-fourmode-ganoff-profile-h2001-260827`
- Output: `/data/home/frank/experiments/stereo_four_mode_ganoff_profile_h2001_20260827_v1`
- Code: `/data/home/frank/projects/Stereo-vison-Tokenizer`
- Runtime: `/data/home/frank/runtime/stereo-tokenizer-unified-v1`
- Task-private overlay: `/data/home/frank/runtime/stereo-tokenizer-profile-overlay-v1`
  containing only `av==16.0.1`
- LAS2-H read-only source/checkpoint:
  `/data/home/hezhou/projects/LiteAnyStereo` at
  `8c97bd4c4da3712c2ac60003a23201dfdb5935f4`, and
  `/data/home/hezhou/artifacts/lite-any-stereo/checkpoints/LAS2_H.pth` with
  SHA256 `758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4`
- GAN, media logging, WandB, and online GT cache: disabled
- Loss weights RGB/relative-depth/relative-gradient/KL/LPIPS:
  `1.0/1.0/1.0/1e-6/1.0`
- LAS2-H: valid iters 4, max disparity 192, bidirectional plus LR consistency
- DA3: process resolution 504, upper-bound resize, finite-positive non-padding
- Current branch requires `single_frame_source_index=0`; the colleague report
  at `45fec2e` used source index 2 and GAN enabled, so it is not an exact runtime
  identity comparison.

Preflight passed with Python 3.12.3, Torch 2.7.1+cu126, Lightning 2.5.6,
PyAV 16.0.1 from the overlay, and `129 passed, 4 warnings` for
`tests/stereo`.

The driver runs one 28-update non-profiler baseline, then four 9-update focused
Kineto traces whose active update targets one mode each. The baseline discards
the first eight updates for steady-state full-step statistics. Focused traces
use wait/warmup/active `5/2/1` and are not used as unadjusted throughput.

Startup health check found the tmux, launcher, and eight ranks alive in DDP
initialization with no traceback or OOM. No stable step throughput existed yet.
Initial ETA at launch: 4--10 minutes for the training/trace body and 7--18
minutes including validation, trace export, and result aggregation.

## GAN-off results

The non-profiler run completed all 28 updates with exit code 0. After discarding
the first eight updates, each mode contributed five steady-state samples:

| Mode | Median (ms) | Mean (ms) | Throughput (sample-executions/s) |
| --- | ---: | ---: | ---: |
| mono/single | 157.96 | 158.60 | 1215.5 |
| mono/four | 530.04 | 530.10 | 362.2 |
| stereo/single | 353.65 | 353.83 | 542.9 |
| stereo/four | 1457.48 | 1459.39 | 131.7 |

The equal-mode mixed mean was 625.48 ms/step, or about 307.0 global sample
executions/s. Rank-0 peak allocated memory reached 114,991,282,688 bytes
(107.1 GiB), while peak reserved memory reached 126,533,763,072 bytes
(117.8 GiB). The focused stereo/four trace reserved more memory than the
non-profiler wall-time run and is not used as a throughput estimate.

Artifacts:

- Baseline timings: `baseline_seed1234/step_timings.json`
- Baseline config: `baseline_seed1234/resolved_config.json`
- Baseline manifest: `baseline_seed1234/run_manifest.json`
- Baseline log: `baseline_seed1234/run.log`
- Focused traces: `trace_*_seed*/torch_profile/`

## GAN-on A/B contract

The reversible launcher change makes GAN activation explicit and fail-closed.
The performance run uses `GAN_ENABLED=1`, image/video GAN weights `1.0/1.0`,
feature-matching weight `1.0`, and `DISCRIMINATOR_START=0`. Everything else
remains identical to the GAN-off baseline: fresh initialization, seed 1234,
8 GPUs, BS24/device, BF16, 28 updates, first eight warm-up updates, deterministic
1:1:1:1 mode order, no Kineto for wall time, no W&B/media, and no online cache.

The GAN-on result, memory safety decision, optional focused trace, comparison
with the colleague medians, and final artifact paths will be appended after the
run finishes.

## H200-2 strict A/B rerun

Status: **in progress**, launched at 2026-08-27 01:23:27 +08:00.

- Node: H200-2
- Branch / experiment commit: `hezhou-las2-h` /
  `bfab545de058e354376a6d1599a3d6b97eb2debe`
- Computational baseline: the main training paths are unchanged from
  `80cba3661dc6de4d6967e1edf5a69823dc9d4e5e`; the experiment commit only adds
  explicit opt-in GAN launcher arguments and its source-contract test.
- Runtime: `/data/home/frank/runtime/stereo-tokenizer-unified-v1`
- PyAV overlay: `/data/home/frank/runtime/stereo-tokenizer-wandb-overlay-v1`,
  providing `av==16.0.1`
- Output:
  `/data/home/frank/experiments/stereo_four_mode_gan_ab_h2002_20260827_v1`
- tmux: `stereo-fourmode-gan-ab-h2002-260827`
- Launcher log: `launcher.log`
- GAN-off run/log: `ganoff_seed1234/run.log`
- GAN-on run/log: `ganon_seed1234/run.log`

The H200-2 launcher first repeats the 28-update GAN-off baseline, then performs
the 28-update GAN-on run from a fresh initialization with weights `1/1/1` and
discriminator start update 0. This preserves a within-node strict A/B after the
requested move from H200-1. The H200-2 stereo manifest SHA256 matches H200-1,
but the node-local mono smoke manifest differs (`5f69331a...` versus
`b265c08f...`); therefore H200-1 results are a cross-node/data reference rather
than the primary strict control.

Startup health check found the tmux, launcher, and eight ranks alive in
initialization, with all eight GPUs opened, and no immediate traceback, OOM,
NCCL error, or NaN. No measured step throughput existed yet. Initial ETA at
01:24 +08:00, based on the completed H200-1 28-update baseline and the reported
roughly 0.2--2.0 second GAN-on step range: 4--10 minutes for both test bodies,
and 7--15 minutes including validation, output finalization, result aggregation,
and the memory-safety decision for focused tracing.
