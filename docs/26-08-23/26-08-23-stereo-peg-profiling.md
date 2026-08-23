# StereoVAE PEG profiling and follow-up experiments

## Purpose and fixed contract

This record measures and optimizes the accepted eight-sample StereoVAE
overfit step without changing its model, loss, optimizer, logging, or data
semantics. All full-step comparisons use one H200, batch 8, BF16, the same
eight samples, the same seed, and 40 updates with 10 active Kineto steps.

- Branch: `frank-profiling`
- Initial profiling commit: `1c05b398b1c34bd38d064287010284603f225c0f`
- PEG experiment commit: `e0420dae062ee6a3f3bbd0240c6a36a1d1b560d6`
- Node/GPU: `h200-2`, physical GPU 0
- Manifest: `/data/home/frank/runtime/stereo-step-profile-input-v1/selected_8_manifest.jsonl`
- Manifest SHA256: `134287c322698fc06bb22f611664dc7f2f5d7a9b3066debdb0014a92a770c267`
- Runtime: Python 3.12.11, PyTorch 2.7.1+cu128, CUDA 12.8,
  PyTorch Lightning 2.5.6

The profiler keeps the original pageable DataLoader, CSV logging, LPIPS/VGG,
Adam, gradient clipping, and scheduler unless an experiment explicitly names
one of those factors as its single changed variable.

## Initial bottleneck

The original active-step median was 666.35 ms. Chrome-trace correlation
identified the dominant operation as the backward path of the 14 Transformer
PEG depthwise `Conv3d(512, 512, 3, groups=512)` modules. Their cumulative
backward kernel time was about 680.7 ms per step; this is cumulative kernel
accounting and must not be added to step wall time.

LPIPS/VGG was the largest forward module at about 63.6 CUDA ms per step.
DataLoader CPU work was about 229.9 ms, dominated by GT NPZ decompression and
NumPy/tensor conversion.

## PEG backend experiment

Three backends were tested at the two actual PEG shapes:

1. Original contiguous depthwise Conv3d.
2. The same Conv3d with `channels_last_3d` input and weight layout.
3. For the actual `T=1` path, a depthwise Conv2d using the only active temporal
   kernel slice while retaining the complete Conv3d parameter and optimizer
   state.

`channels_last_3d` had no measurable benefit: the B=192 path changed from
29.175 ms to 29.186 ms and the B=24 path from 7.167 ms to 7.200 ms.

The T=1 Conv2d path produced identical forward values in BF16 and FP32. All
inactive temporal weight slices had exactly zero gradient. FP32 input-gradient
relative error was at most about 1.9e-7; FP32 weight-gradient and Adam-moment
relative error was about 3.4e-4. BF16 weight-gradient relative error was 0.256%
for B=192 and 0.0062% for B=24. After one Adam step, BF16 weights were exactly
equal and FP32 weight relative error was about 6.1e-10. This is mathematical
equivalence with bounded kernel-reduction rounding, not a bitwise-identical
training trajectory.

| Backend | Active median | Relative throughput | Peak allocated |
| --- | ---: | ---: | ---: |
| Contiguous Conv3d | 660.26 ms | 1.000x | 32.85 GiB |
| T=1 Conv2d slice | 513.48 ms | 1.286x | 32.44 GiB |

The Conv2d path reduced the full step by 146.78 ms, or 22.2%. The backward CPU
envelope fell from 247.97 ms to 142.48 ms. LPIPS remained about 63.6 CUDA ms,
and GT NPZ decompression remained about 141 ms, supporting the PEG attribution.

Full-step outputs:

- Baseline: `/data/home/frank/runtime/stereo-peg-profile-v1/e0420da-h2002-gpu0-b8-40u-contiguous-v1`
- Conv2d: `/data/home/frank/runtime/stereo-peg-profile-v1/e0420da-h2002-gpu0-b8-40u-conv2d-v1`
- Microbenchmark: `/data/home/frank/runtime/stereo-peg-profile-v1/e0420da-h2002-gpu0-micro-v2/result.json`

Both full runs completed 40 updates without NaN, Inf, OOM, or a remaining GPU
process. The first two logged losses were identical. BF16 rounding differences
appeared from the third update and the short optimization trajectories then
diverged, as expected from the gradient comparison.

## Follow-up experiments

Status at creation: approved, implementation and measurements pending.

The next experiments keep the PEG Conv2d path as their common baseline and
change exactly one factor at a time:

1. Preload the fixed eight processed samples while retaining the same collate
   and batch tensors.
2. Enable DataLoader pinned memory and non-blocking H2D transfer.
3. Cache the fixed GT LPIPS normalized VGG features only after verifying loss,
   prediction gradient, scaling, normalization, dtype, and autocast behavior.

The apparent 60.6 ms `loss_breakdown` CPU region is not treated as logger cost:
the measured CSV logger and save work totals only about 0.48 ms per step. It is
currently interpreted as an asynchronous synchronization boundary.

## Incidents

One attempted baseline launch was rejected locally by PowerShell while parsing
remote `$()` and `<` syntax. The command did not reach H200-2, did not create an
output directory, and did not use a GPU. The simplified launcher then completed
with unchanged parameters.

## Current conclusion

PEG depthwise Conv3d backward was the original dominant bottleneck. The T=1
Conv2d slice is the only successful PEG candidate so far. It remains an
explicit profiling backend; ordinary training still defaults to Conv3d pending
a separate decision to promote it.
