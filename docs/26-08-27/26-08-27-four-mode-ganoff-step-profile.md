# Four-mode GAN-off step profiling

## Purpose

Measure the current `hezhou-las2-h` training graph without GAN and attribute
wall-clock time across the four mono/stereo x single/four-frame modes. This is
intended to test the claim that `full step - LAS2-H teacher` is VAE time; that
residual also contains data transfer, LPIPS, backward, DDP, optimizer, and
logging.

## Status

GAN-off profiling and the strict H200-1 GAN-on wall-time A/B both completed
successfully. The GAN-on run exited 0 after 28 updates; no OOM, NaN, DDP hang,
or traceback occurred.

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

The final result and trace decision are recorded below.

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

### H200-2 launch result

The launcher exited 1 before the first GAN-off update; GAN-on was not started.
No `step_timings.json` was produced, so this attempt contains no performance
measurements. The first root-cause exception on every rank was:

```text
ValueError: LAS2-H source repository is dirty
```

Read-only follow-up confirmed that the pinned source still had the required
HEAD `8c97bd4c4da3712c2ac60003a23201dfdb5935f4`, but its worktree contained the
untracked path `checkpoints`. The source is owned by `hezhou`, so it was left
untouched. At result inspection time, Hezhou-owned LAS2-H processes had also
occupied GPUs 0--3 with roughly 17--32 GiB each. This attempt is classified as
an environment/preflight failure, not an OOM, DDP, GAN, or model failure.

Failure artifacts:

- Launcher exit: `launcher_exit_code.txt` (`1`)
- GAN-off exit: `ganoff_seed1234/exit_code.txt` (`1`)
- Resolved config: `ganoff_seed1234/resolved_config.json`
- Run manifest: `ganoff_seed1234/run_manifest.json`
- Root-cause log: `ganoff_seed1234/run.log`

### H200-2 v2 relaunch

Status: **in progress**, launched at 2026-08-27 10:45:03 +08:00.

The user authorized a Frank-owned task-private LiteAnyStereo clone so the run
does not modify or depend on the dirty Hezhou worktree. The clone is:

- Path: `/data/home/frank/runtime/lite-any-stereo-8c97bd4-clean`
- Owner: `frank`
- Detached HEAD: `8c97bd4c4da3712c2ac60003a23201dfdb5935f4`
- Origin: `https://github.com/TomTomTommi/LiteAnyStereo.git`
- State before launch: clean

The v2 A/B output is
`/data/home/frank/experiments/stereo_four_mode_gan_ab_h2002_20260827_v2`,
and tmux is `stereo-fourmode-gan-ab-h2002-260827-v2`. The launcher and A/B
parameters are unchanged from v1 except that `LAS2_H_REPO` points to the clean
task-private clone and all artifacts use the new v2 root. Startup health check
found the tmux, launcher, and all eight ranks alive in GAN-off initialization;
the first ranks had loaded the teacher onto GPUs 0--1, and the logs contained no
immediate traceback, dirty-source rejection, OOM, NCCL error, or NaN.

Initial ETA at 10:46 +08:00 remains 4--10 minutes for both test bodies and
7--15 minutes including validation, result aggregation, and the focused-trace
memory-safety decision. No steady-state step existed at the health snapshot, so
this is still a history-based estimate rather than a measured-throughput ETA.

### H200-2 v2 result and fixed-data diagnosis

The v2 GAN-off process exited 137 before the first update and GAN-on did not
start. This was not an OOM: DataLoader workers showed that the reused H200-1
episode manifest referenced H200-2-missing video shards, while other referenced
videos had no decodable frame at their target timestamps. Representative errors
were missing files under shards `1572`, `1839`, and `1958`, and timestamp decode
failures under shards `0405`, `0654`, and `0750`. No `step_timings.json` was
produced.

H200-2 already had a locally built episode manifest at
`/data/home/frank/experiments/stereo_lerobot_cpu_20260824_approval1/h200_2_local_manifest_v1.jsonl`
with 48,529 episodes. The four-mode loader deterministically selects 48 distinct
train episodes/windows from it for seed 1234. Because the mono smoke also has 48
samples and every rank receives six source indices, one stereo-mode update and
one mono-mode update each cover the complete 48-sample source set (with local
BS24 repetition).

For v3, those actually selected 48 stereo episodes were materialized as:

- Manifest:
  `inputs/stereo_fixed_48_seed1234.jsonl`
- Summary/audit:
  `inputs/stereo_fixed_48_seed1234_summary.json`
- Manifest SHA256:
  `82ba78ceef0c3a86a7772d4f378cfc622ef8d1e55795475d881e1f6d6d865a2c`
- Contract: 48 distinct episodes, 48 selected windows, seed 1234,
  `single_frame_source_index=0`
- Validation: all 48 selected windows successfully decoded six-camera
  four-frame input with native shape `[3,2,3,4,256,256]`; this also covers the
  source-index-0 frame required by stereo/single.

### H200-2 v3 equal-count A/B

Status: **in progress**, launched at 2026-08-27 11:45:58 +08:00.

- Output:
  `/data/home/frank/experiments/stereo_four_mode_gan_ab_h2002_20260827_v3`
- tmux: `stereo-fourmode-gan-ab-h2002-260827-v3`
- Code: `hezhou-las2-h@bfab545de058e354376a6d1599a3d6b97eb2debe`
- LAS2-H: Frank-owned clean clone at exact `8c97bd4c...`
- Order: fresh 28-update GAN-off, then fresh 28-update GAN-on
- GAN-on weights: image/video/feature matching `1/1/1`, discriminator start 0
- Shared source counts: mono 48, stereo 48

Startup health check found the tmux, launcher, and eight ranks alive in GAN-off
initialization with no immediate missing-file, decode, dirty-source, OOM, NCCL,
NaN, or traceback error. No steady step existed yet. Initial ETA at 11:46 +08:00
is 4--10 minutes for both bodies and 7--15 minutes including result aggregation
and the focused-trace memory-safety decision, based on the H200-1 baseline and
the colleague GAN-on timing range.

The v3 GAN-off attempt then exited 1 before its first update because the
materialized manifest contained only the 48 selected `train` episodes, while
Lightning constructs the validation loader before entering the fit loop. Every
rank failed closed with `ValueError: manifest contains no val episodes`.
GAN-on did not start and no timing file was produced. The 48-sample decode audit
remains valid, but that manifest cannot be used directly by the current combined
train/validation entrypoint without adding a separately valid val split.

### H200-1 direct GAN-on A/B

Status: **completed successfully**, launched at 2026-08-27 12:07:38 +08:00.

Because the original H200-1 GAN-off baseline is already complete with the exact
node-local stereo/mono data, the H200-1 retry runs only one fresh 28-update
GAN-on body and compares it directly with that baseline.

- Output:
  `/data/home/frank/experiments/stereo_four_mode_ganon_profile_h2001_20260827_v1`
- tmux: `stereo-fourmode-ganon-h2001-260827-v1`
- Code: `hezhou-las2-h@bfab545de058e354376a6d1599a3d6b97eb2debe`
- Runtime/overlay: `stereo-tokenizer-unified-v1` plus
  `stereo-tokenizer-profile-overlay-v1` (`av==16.0.1`)
- Data/seed/order: identical to the completed H200-1 GAN-off baseline
- GAN-on: image/video/feature matching `1/1/1`, discriminator start 0
- Schedule: 28 updates, first 8 warm-up, final 20 giving 5 samples per mode

Pre-launch snapshot found all eight H200-1 GPUs at 0 MiB with no compute
process. The user explicitly authorized ignoring a possible root process on
GPU0; none was present at launch. Startup health check found the tmux, launcher,
and all eight ranks alive in initialization, with no immediate traceback, OOM,
NCCL error, NaN, dirty-source error, or data error. No steady step existed yet.
Initial ETA at 12:08 +08:00 is 3--7 minutes for the 28-update GAN-on body and
5--12 minutes including validation, result aggregation, comparison, and the
focused-trace memory-safety decision.

### H200-1 final GAN-on wall-time result

The run completed all 28 updates with exit code 0. Updates 1--8 were discarded
as warm-up; the remaining 20 updates provide five samples per mode. Throughput
uses the required `192 / step_seconds` definition.

| Mode | Median (ms) | Mean (ms) | P90 (ms) | Samples/s |
| --- | ---: | ---: | ---: | ---: |
| mono/single | 168.24 | 171.79 | 179.68 | 1141.26 |
| mono/four | 1079.82 | 1082.57 | 1090.90 | 177.81 |
| stereo/single | 382.79 | 383.40 | 385.11 | 501.58 |
| stereo/four | 3044.42 | 3049.36 | 3059.45 | 63.07 |

The equal-mode mean of the four medians is 1168.82 ms/step, corresponding to
164.27 sample-executions/s. The mean over all 20 steady updates is 1171.78
ms/step, corresponding to 163.85 sample-executions/s. Peak CUDA allocated
memory was 115,218,265,600 bytes (107.31 GiB), and peak reserved memory was
126,617,649,152 bytes (117.92 GiB).

### Strict A/B and colleague comparison

`GAN total delta` below is exactly `GAN-on median - GAN-off median`. It is the
complete wall-time cost of the enabled GAN path, including generator
adversarial/feature-matching work, discriminator work, extra backward/DDP, and
optimizer work. It must not be labelled VAE time.

| Mode | GAN-off (ms) | GAN-on (ms) | GAN total delta | Delta | Colleague GAN-on (ms) | Ours - colleague | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mono/single | 157.96 | 168.24 | +10.28 ms | +6.51% | 208.9 | -40.66 ms | -19.47% |
| mono/four | 530.04 | 1079.82 | +549.78 ms | +103.73% | 715.1 | +364.72 ms | +51.00% |
| stereo/single | 353.65 | 382.79 | +29.14 ms | +8.24% | 452.9 | -70.11 ms | -15.48% |
| stereo/four | 1457.48 | 3044.42 | +1586.93 ms | +108.88% | 1984.6 | +1059.82 ms | +53.40% |

The equal-mode mean of medians increased by 544.03 ms (+87.07%). Corresponding
mixed throughput fell from 307.31 to 164.27 sample-executions/s (-46.55%). The
result therefore does **not** reproduce the colleague's timing profile or the
reported stereo/four value near 1984.6 ms. Our single-frame modes are 15--19%
faster, while both four-frame modes are 51--53% slower. The mode-dependent
split points to the four-frame GAN path as the main discrepancy, but does not
by itself identify an individual kernel or loss as the cause.

This remains a controlled GAN-off/on A/B for the current experiment, but not an
identity reproduction of the colleague run. Known differences are:

- Colleague code `45fec2e341eb83ade04c9a1c88d824e49f3c9b5f`; current computational
  baseline `80cba3661dc6de4d6967e1edf5a69823dc9d4e5e`, with reversible launcher
  activation at `bfab545de058e354376a6d1599a3d6b97eb2debe`.
- Current node/runtime: H200-1, Python 3.12.3, Torch 2.7.1+cu126, Lightning
  2.5.6, and task-private PyAV 16.0.1 overlay. The colleague runtime/node
  identity was not recorded in the available report beyond 8 x H200.
- Current `single_frame_source_index=0`; colleague report used index 2.
- Current run is fresh initialization with `resume_from_checkpoint=null`;
  colleague fresh-init/resume/checkpoint state was not recorded.
- Current wall-time run has Kineto/focused profiling disabled; colleague
  profiler state was not recorded.

### Focused-trace decision and attribution limit

No GAN-on focused trace was launched. The wall-time run already reserved
117.92 GiB on a roughly 140.4-GiB H200, while the prior GAN-off stereo/four
focused trace had reached about 129.5 GB reserved. Profiler overhead on the
larger GAN graph was therefore not treated as safely bounded at BS24/GPU, and
the batch was not reduced because that would break the requested comparison.
The current trace regions also do not separately expose Image GAN, Video GAN,
feature matching, discriminator backward, and both optimizers.

Consequently, stereo/four can be stated precisely as 1457.48 ms GAN-off,
3044.42 ms GAN-on, and +1586.93 ms for the complete GAN path. It cannot be
honestly split further into teacher, VAE encoder/decoder, LPIPS, individual GAN
losses, backward, DDP, and optimizers from this wall-time A/B. In particular,
neither `full step - teacher` nor the full GAN delta is VAE time. A finer split
would require a separate, minimally instrumented profiling experiment and a
new memory-safety decision.

Final artifacts:

- GAN-off root:
  `/data/home/frank/experiments/stereo_four_mode_ganoff_profile_h2001_20260827_v1`
- GAN-off timings/config/log:
  `baseline_seed1234/step_timings.json`, `baseline_seed1234/resolved_config.json`,
  and `baseline_seed1234/run.log`
- GAN-on root:
  `/data/home/frank/experiments/stereo_four_mode_ganon_profile_h2001_20260827_v1`
- GAN-on timings/config/manifest/log:
  `ganon_seed1234/step_timings.json`, `ganon_seed1234/resolved_config.json`,
  `ganon_seed1234/run_manifest.json`, and `ganon_seed1234/run.log`
- GAN-on launcher exit and run exit:
  `launcher_exit_code.txt` and `ganon_seed1234/exit_code.txt`, both `0`

### Post-experiment cleanup

After the completed result was recorded, the reversible experiment-only
one-sample runner and its test were removed. The generic launcher was restored
to explicit GAN-off weights, and its source-contract test was restored. This
experiment document and all external experiment artifacts are retained.
