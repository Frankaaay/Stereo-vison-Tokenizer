# H200-2 four-mode module profile

## Purpose

Measure per-step DataLoader wait/transfer, online GT generation, and named
StereoVAE internal regions for all four mono/stereo x single/four-frame modes.
This is a bounded profiling run, not a training-quality or throughput run.

## Provenance and runtime

- Status: v1 failed during argument validation; v2 in progress
- v1 start: 2026-08-29 14:58:35 CST
- v2 start: 2026-08-29 15:00:00 CST
- Host: `h200-2` (`frank`, `lacy--214-30-239-42`)
- Branch: `hezhou-las2-h`
- Commit: `06aaf208873f1187ae08557d26bf6ef4ee20ea05`
- Runtime: `/data/home/frank/runtime/stereo-tokenizer-unified-v1`
- tmux: `stereo-four-mode-module-profile-bs24-h2002-v2`
- Output: `/data/home/frank/experiments/stereo-four-mode-module-profile-bs24-h2002-20260829-v2`
- Launcher: `launch.sh` under the output directory
- Log: `run.log` under the output directory
- Step timing target: `step_timings.json`
- Trace target: `torch_profile/`

The H200-2 clone was clean and fast-forwarded from `bfab545d` to the exact
local/origin SHA above. Thirty-eight targeted CPU contract tests passed before
launch. All eight H200 GPUs reported 0 MiB and no compute processes before
launch.

## Data contract

- Hy manifest: `/data/home/frank/runtime/stereo-tokenizer-pretrain-h2002-20260829-hy-odd-v1/manifests/hy_formal_90_5_5_v1.jsonl`
  - SHA256: `055699f4b1159a6ff55e77cc1379e052fb6292cd0525c82e61cb178198d56c86`
  - 42,826 accepted records; the conservative generator covered five odd
    tables and rejected 11,015 identity-mismatched or missing records.
- LIBERO manifest: `/data/home/frank/runtime/stereo-tokenizer-pretrain-h2002-20260829/manifests/libero_formal_90_5_5_v1.jsonl`
  - SHA256: `0299354a7225e979f6b9ff5fb3e26a975c811d2d41af44e042a6eade3f24bbf4`
- UMI manifest: `/data/home/frank/runtime/umi-lerobot-decode-audit-h2002-20260829-v1/umi_lerobot_decode_verified_v1.jsonl`
  - SHA256: `96024f091bcf7aca844b4d4b99fad2eb6cb0f420aa693f1431340b79ac5fa53e`
  - Full CPU decode audit accepted all 48,529 episodes.

## Profile contract

- Single node, eight H200 GPUs, BF16
- Per-device batch 24, gradient accumulation 1, global batch 192
- All four mode batch sizes 24 and accumulation factors 1
- Mode weights `35:35:15:15`, mono source weights Hy:LIBERO `9:1`
- 41 logical updates; the second complete 20-update mode cycle is measured and
  update 41 finalizes the profiler after its active window
- Step timing warmup: 20 updates
- Torch profiler schedule: wait 15, warmup 5, active 20
- GAN, online GT cache, WandB, and media logging disabled
- Real LAS2-H (`valid_iters=4`, pair microbatch 48) and DA3-BASE teachers
- Validation and checkpoint boundary: update 41

The v1 launcher used exactly 40 updates and exited before GPU initialization
with `torch profiler schedule must leave a post-profile update`. Its output was
retained. The v2 correction changes only maximum updates, validation, and
checkpoint boundaries from 40 to 41; wait/warmup/active and all data/model
contracts are unchanged. The v2 validation and checkpoint boundary is update
41.

The trace names separate DataLoader wait/CPU-to-GPU transfer, LAS2-H or DA3 GT,
encoder/decoder modules and heads, losses, backward, clipping, optimizer, and
scheduler. Rank 0 records the detailed CPU/CUDA trace; `step_timings.json`
provides the mode mapping for measured logical updates.

## Initial ETA

At launch, the initial estimate is 2--5 minutes for the 40-update training and
trace body, and 4--10 minutes for validation, checkpointing, trace export, and
final artifacts. This estimate is based on the prior H200-1 GAN-off four-mode
profile plus the larger 20-step active trace. It will be replaced by measured
timings after completion.

## Final result

The v2 run exited with code 0 and completed 41/41 logical updates. Detailed
profiling covered logical updates 21--40. Update 41 spent about 152.4 seconds
finalizing the trace and running validation/checkpoint hooks, so it is excluded
from all training-step comparisons. Validation reported
`val/mixed/total_loss ~= 1.190`; this fresh 41-update run is not a quality
evaluation.

Artifacts:

- `step_timings.json`: per-step mode, wall interval, DataLoader wait/transfer,
  throughput, and peak memory
- `torch_profile/rank0-trace-00.json.gz`: rank-0 CPU/CUDA trace
- `torch_profile/rank0-trace-00-regions.json`: aggregate named regions
- `module_timing_by_mode.json`: active-step mode-aware CUDA work attribution
- `analyze_trace.py`: deterministic trace-to-mode attribution helper
- three readable checkpoints under `train/checkpoints/`

The table reports medians over the active profile window. Data wait includes
main-process DataLoader wait plus CPU-to-GPU transfer. GT, encoder, decoder,
LPIPS, and backward are sums of CUDA kernel/memcpy/memset durations associated
with the named CPU region. Nested device regions must not be added to their
parent totals.

| mode | steps | wall ms | samples/s | data wait ms | GT ms | encoder ms | decoder ms | LPIPS ms | backward ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mono/single_frame | 7 | 177.13 | 1,083.9 | 7.11 | 58.59 | 2.18 | 2.30 | 16.74 | 27.05 |
| stereo/single_frame | 3 | 385.30 | 498.3 | 8.66 | 132.52 | 13.21 | 6.36 | 48.46 | 88.74 |
| mono/four_frame | 7 | 553.67 | 346.8 | 12.27 | 216.72 | 18.67 | 19.05 | 64.36 | 159.42 |
| stereo/four_frame | 3 | 1,778.27 | 108.0 | 68.04 | 518.90 | 81.74 | 55.32 | 193.91 | 515.04 |

Peak allocated memory on rank 0 was approximately 8.2 GiB for mono/single,
23.9 GiB for stereo/single, 33.2 GiB for mono/four, and 107.1 GiB for
stereo/four. Peak reserved memory reached about 132.0 GiB during profiling,
leaving limited allocator headroom on a 139.8-GiB H200.

Within the VAE, stereo/four encoder CUDA work was dominated by the spatial
transformer (~41.17 ms median), temporal transformer (~31.24 ms), and stereo
fusion (~6.66 ms). Its decoder was dominated by temporal (~31.27 ms) and
spatial (~23.15 ms) transformers. RGB/depth heads, core reconstruction losses,
gradient clipping, and Adam were individually small. Across all modes, online
GT, LPIPS, and backward were much larger than the lightweight output heads.

The absolute active-step wall times include torch-profiler overhead and should
be used for component attribution and same-run relative comparisons, not as
unprofiled production throughput. Mono/four had a DataLoader outlier at step
21 and a DA3 outlier at step 38, so medians are more representative than means.

## Mode-aware batch-scaling bottleneck A/B (in progress)

At `2026-08-29 15:45:07 CST`, a same-host H200-2 A/B was started in tmux
session `stereo-bs-scaling-bottleneck-h2002-v1`. The repo remained clean at
`06aaf208873f1187ae08557d26bf6ef4ee20ea05`; all eight GPUs were idle before
launch. The manifest, teacher, seed, loss, precision, worker, GAN-off, and
cache-off contracts are identical to the completed baseline above.

The experiments run serially and fail closed:

1. Unprofiled A uses mode batches `24:24:24:24`, mode GA `1:1:1:1`, and global
   batch 192. Updates 1--20 warm timing; updates 21--60 measure 40 logical
   updates, or 7,680 global samples.
2. Unprofiled B uses mode batches `48:48:48:24`, mode GA `1:1:1:2`, and global
   batch 384. Updates 1--20 warm timing; updates 21--40 measure 20 logical
   updates, also 7,680 global samples.
3. If both unprofiled arms succeed, the same B contract runs a detailed trace.
   The profiler schedule is wait 18, warmup 5, active 23 physical batches. The
   active logical interval is updates 21--40; each stereo/four update is the sum
   of its two GA microbatches. Update 41 provides the required post-profile
   update and validation/checkpoint boundary.

Output roots:

- `/data/home/frank/experiments/stereo-bs24-vs-modeaware384-throughput-h2002-20260829-v1`
- `/data/home/frank/experiments/stereo-modeaware384-module-profile-h2002-20260829-v1`

The first startup snapshot confirmed the tmux session and all eight distributed
workers. The log reached distributed initialization without an immediate
traceback; GPU memory was still at 4 MiB per device during initialization, so
no throughput sample existed yet. Initial ETA at launch is 8--18 minutes for
both unprofiled arms plus the profiled body, and 12--25 minutes including trace
export, validation, checkpoints, and comparison artifacts. This estimate uses
the prior 41-update profile, whose post-processing boundary took about 152
seconds; it will be replaced with measured elapsed times after completion.

### Unprofiled A/B result

Both equal-sample arms exited with code 0. A measured logical updates 21--60
and B measured 21--40; each interval contains exactly 7,680 global samples.
The deterministic comparison is stored in
`stereo-bs24-vs-modeaware384-throughput-h2002-20260829-v1/throughput_comparison.json`.

| mode | A median ms | B median ms | time ratio | A samples/s | B samples/s | throughput change |
|---|---:|---:|---:|---:|---:|---:|
| mono/single_frame | 191.67 | 240.64 | 1.255x | 1,016.5 | 1,595.7 | +57.0% |
| mono/four_frame | 709.74 | 1,015.61 | 1.431x | 270.5 | 378.1 | +39.8% |
| stereo/single_frame | 572.39 | 627.17 | 1.096x | 339.3 | 612.3 | +80.5% |
| stereo/four_frame | 1,565.40 | 3,719.76 | 2.376x | 122.7 | 103.2 | -15.8% |

Equal-sample aggregate throughput was 291.32 samples/s for A and 291.98
samples/s for B, a statistically negligible +0.22%. Thus the three physical
BS48 modes did obtain substantial batching gains, but stereo/four GA2 consumed
more than twice its A logical-step time and cancelled essentially all of those
gains. Its median summed DataLoader wait/transfer rose from 69.52 to 214.06 ms
(3.08x), while the other three modes' waits fell. This identifies
stereo/four GA2 as the mixed-run bottleneck, but the detailed trace is still
needed to split the remaining excess among GT, VAE/LPIPS, and backward.

Peak allocated memory approximately doubled for the three physical-BS48 modes:
mono/single 8.19 to 14.77 GiB, mono/four 33.29 to 65.14 GiB, and stereo/single
23.88 to 46.16 GiB. Stereo/four retained physical BS24 and stayed near 107.3
GiB; physical BS48 is therefore not a safe alternative for this mode.

### Detailed B profile correction (in progress)

The first B profile attempt exited before GPU initialization with code 1. Its
schedule incorrectly counted 23 physical microbatches, but
`TrainingProfilerStepCallback` advances only when `generator_updates`
increments, so profiler steps are logical updates. The first root-cause error
was `ValueError: torch profiler schedule must leave a post-profile update`.

The failed v1 directory was retained. At `2026-08-29 16:08:47 CST`, corrected
v2 was launched in tmux `stereo-modeaware384-profile-h2002-v2` with wait 15,
warmup 5, active 20, and max steps 41. This captures logical updates 21--40;
each stereo/four profiler step includes both GA microbatches. Output:
`/data/home/frank/experiments/stereo-modeaware384-module-profile-h2002-20260829-v2`.
The startup snapshot confirmed the rank-0 process and exact mode-aware batch
arguments; GPU initialization had not completed yet. Based on the baseline
profile, the refreshed ETA is 3--7 minutes for the profiled body and 6--12
minutes including trace export, validation, and checkpoints.

The corrected v2 subsequently exited with code 0. It completed all 41 logical
updates, exported a 63-MB rank-0 trace, produced step timing and aggregate
region files, wrote three checkpoints, and reported validation mixed total loss
of approximately 1.21. All eight GPUs were idle after completion. The detailed
mode attribution was started as a CPU-only post-processing step; the long A/B
launcher is queued immediately behind that parser so the parser cannot overlap
the measured GPU run.

## Long equal-sample throughput A/B (queued)

The long-run contract extends the short unprofiled comparison without changing
the SHA, manifests, teachers, seed, mode weights, losses, workers, precision,
or GAN/cache controls:

- A: 1,800 logical updates at mode BS `24:24:24:24`, GA `1:1:1:1`. Offline
  measurement uses updates 201--1800 (1,600 updates, 307,200 samples).
- B: 900 logical updates at mode BS `48:48:48:24`, GA `1:1:1:2`. Offline
  measurement uses updates 101--900 (800 updates, 307,200 samples).
- A warmup processes `200 * 192 = 38,400` samples; B warmup processes
  `100 * 384 = 38,400` samples.
- Both measured windows contain complete 20-update mode-schedule cycles and
  therefore exactly preserve the `35:35:15:15` mode ratio.
- `STEP_TIMING_WARMUP=0`; filtering is performed offline because the current
  callback's warmup value is per mode rather than global logical update.
- A and B run serially on all eight H200 GPUs and fail closed.

Output root:
`/data/home/frank/experiments/stereo-bs24-vs-modeaware384-long20m-h2002-20260829-v1`.
The expected measured bodies are approximately 20 minutes each, with an
initial total ETA of 42--50 minutes including two initializations and final
validation/checkpoint hooks.

The long run started at `2026-08-29 16:28:05 CST` in tmux
`stereo-bs-scaling-long20m-h2002-v1`. The startup snapshot confirmed A's exact
1,800-step BS24/GA1 command and all eight distributed workers. Each GPU had
allocated approximately 2.7 GiB while model/teacher initialization was still
finishing; no immediate exception was present. B remains queued behind A.

### Corrected B trace interpretation

The v2 trace confirms that the three physical-BS48 modes perform close to twice
the CUDA work of BS24 while processing twice the samples. Their profiled wall
ratios were 1.52x for mono/single, 1.89x for mono/four, and 1.72x for
stereo/single. Component ratios were generally 1.84--2.11x: the main exception
was the once-per-logical-update Adam step, which remained approximately 1.00x.

For stereo/four GA2, each individual BS24 microbatch had essentially unchanged
CUDA durations relative to A: GT, encoder, decoder, LPIPS, and backward were
all within roughly 0.6% of the A medians, while Adam still ran once. The
profiled logical wall interval nevertheless grew from 1,778.27 to 4,612.77 ms
(2.59x). This reinforces the unprofiled finding that GA2 adds no physical
batching benefit and that its super-linear penalty is primarily outside the
individual CUDA module kernels, consistent with the observed DataLoader/
transfer and host-side gaps.

The first version of the offline attribution helper stores a single value per
logical-step region name. With GA2, the second same-named region overwrites the
first, so the current B stereo/four component row represents one microbatch,
not the logical sum. The wall interval is unaffected. A corrected sum-over-
duplicate-regions pass is intentionally deferred until the long GPU run ends,
so parsing the 63-MB trace cannot perturb the long throughput measurement.

## Long A/B final result

Both long arms exited with code 0, with no traceback, exception, or CUDA OOM.
A completed 1,800/1,800 logical updates in about 21:20 including initialization
and final hooks; B completed 900/900 logical updates in about 18:55. The
predeclared offline windows contain exactly 307,200 samples per arm.

| mode | A aggregate samples/s | B aggregate samples/s | B/A change | A median step ms | B median step ms |
|---|---:|---:|---:|---:|---:|
| mono/single_frame | 579.69 | 876.21 | +51.15% | 333.06 | 437.70 |
| mono/four_frame | 266.80 | 307.65 | +15.31% | 742.01 | 1,245.88 |
| stereo/single_frame | 354.80 | 461.88 | +30.18% | 546.48 | 846.23 |
| stereo/four_frame | 116.85 | 115.40 | -1.24% | 1,684.30 | 3,303.88 |

Overall A processed its 307,200 measured samples in 1,112.72 seconds, or
276.08 samples/s. B processed the same number in 971.28 seconds, or 316.28
samples/s. Mode-aware global batch 384 therefore improved aggregate throughput
by **14.56%** and reduced equal-sample measured time by 141.44 seconds
(12.71%).

Twenty-logical-update block throughput also shows that this result is not a
single short-window fluctuation:

| arm | blocks | median | P10 | P90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| A BS24/GA1 | 80 | 275.87 | 267.30 | 283.88 | 254.07 | 359.06 |
| B mode-aware | 40 | 315.38 | 304.90 | 328.49 | 298.92 | 363.23 |

B's P10 exceeds A's P90 by about 7.4%, so the long-window distributions are
substantially separated. The earlier 40-vs-20-update short test sampled an
unusually favorable A interval and unfavorable B interval and reported only
+0.22%; it is not representative of sustained throughput.

The corrected GA2 trace attribution now sums duplicate regions from both
microbatches. For stereo/four, B/A CUDA-work ratios are GT 2.0005x, encoder
1.9977x, decoder 2.0015x, LPIPS 2.0004x, and backward 2.0044x; Adam remains
1.0005x because it runs once per logical update. Thus GA2 gives essentially no
compute batching benefit, but the long unprofiled result shows only a 1.24%
aggregate throughput loss for stereo/four, not the severe loss implied by the
short/profiled wall interval. The profiler's stereo/four 2.59x wall ratio is
dominated by trace/host-side overhead and must not be used as production
throughput.

Final comparison artifacts:

- `/data/home/frank/experiments/stereo-bs24-vs-modeaware384-long20m-h2002-20260829-v1/long_throughput_comparison.json`
- `/data/home/frank/experiments/stereo-modeaware384-module-profile-h2002-20260829-v2/module_timing_by_mode_ga_sum.json`
- `/data/home/frank/experiments/stereo-modeaware384-module-profile-h2002-20260829-v2/module_comparison_a24_b384_ga_sum.json`
