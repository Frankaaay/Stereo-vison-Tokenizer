# StereoVAE PEG profiling and follow-up experiments

## Purpose and fixed contract

This record measures and optimizes the accepted eight-sample StereoVAE
overfit step without changing its model, loss, optimizer, logging, or data
semantics. All full-step comparisons use one H200, batch 8, BF16, the same
eight samples, the same seed, and 40 updates with 10 active Kineto steps.

- Branch: `frank-profiling`
- Initial profiling commit: `1c05b398b1c34bd38d064287010284603f225c0f`
- PEG experiment commit: `e0420dae062ee6a3f3bbd0240c6a36a1d1b560d6`
- Follow-up experiment commit: `09af7ff8e2e04cbc1f427df3aa651c699f8f7065`
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

Status: completed.

The next experiments keep the PEG Conv2d path as their common baseline and
change exactly one factor at a time:

1. Preload the fixed eight processed samples while retaining the same collate
   and batch tensors.
2. Enable DataLoader pinned memory and non-blocking H2D transfer.
3. Cache the fixed GT LPIPS normalized VGG features only after verifying loss,
   prediction gradient, scaling, normalization, dtype, and autocast behavior.

The new-SHA control and all three single-variable runs completed 40 updates.
Every logged metric at every one of the 40 steps was exactly equal to control.

| Run | Active median | Change from control | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| PEG Conv2d control | 509.37 ms | baseline | 29.70 GiB | 30.98 GiB |
| Preload eight samples | 288.93 ms | -43.3% | 29.69 GiB | 30.98 GiB |
| Pinned CPU memory | 501.77 ms | -1.49% | 29.69 GiB | 30.98 GiB |
| LPIPS GT cache, repeat | 482.82 ms | -5.21% | 29.50 GiB | 33.49 GiB |
| PEG Conv2d + preload + LPIPS cache | 263.14 ms | -48.3% | 29.50 GiB | 33.49 GiB |

Preloading reduced the DataLoader region from 223.86 ms to 7.06 ms and removed
the NPZ and NumPy work from the measured step. This is directly useful for the
fixed eight-sample acceptance run, but it is not evidence that the complete
3407-sample pilot should be held in memory.

Pinned memory reduced H2D CUDA time from 14.88 ms to 3.91 ms. DataLoader CPU
time increased from 223.86 ms to 238.50 ms because pinning itself has a cost,
leaving only a 7.59 ms end-to-end median improvement. Lightning already used
non-blocking GPU transfers; the experiment changed only DataLoader pinning.

The isolated 96-frame LPIPS check produced exactly equal loss and prediction
gradient. Baseline forward/backward took 118.17 ms and the cached form took
89.06 ms. The normalized GT feature cache is FP32 on GPU and occupies
3,070,230,528 bytes, or 2.859 GiB.

In the full step, caching reduced LPIPS CUDA time from 64.16 ms to 34.33 ms and
the backward CPU envelope from 142.47 ms to 114.63 ms. The first full cache run
contained several approximately 0.97 s DataLoader outliers and had an active
median of 481.25 ms. A repeat removed the unrelated CPU outliers and measured a
stable 482.82 ms median. Reserved GPU memory increased by about 2.51 GiB.

Follow-up outputs:

- Control: `/data/home/frank/runtime/stereo-followup-profile-v1/09af7ff-h2002-gpu0-b8-40u-control-v1`
- Preload: `/data/home/frank/runtime/stereo-followup-profile-v1/09af7ff-h2002-gpu0-b8-40u-preload-v1`
- Pinned: `/data/home/frank/runtime/stereo-followup-profile-v1/09af7ff-h2002-gpu0-b8-40u-pinned-v1`
- LPIPS cache repeat: `/data/home/frank/runtime/stereo-followup-profile-v1/09af7ff-h2002-gpu0-b8-40u-lpips-cache-v2`
- LPIPS equivalence microbenchmark: `/data/home/frank/runtime/stereo-followup-profile-v1/09af7ff-h2002-gpu0-lpips-micro-v2/result.json`

## Combined accepted candidates

The approved combination enabled the T=1 PEG Conv2d path, eight-sample
preloading, and the LPIPS GT feature cache. Pinned memory remained disabled so
that its CPU cost was not mixed into the result.

The combined run completed 40 updates with every logged metric at every step
exactly equal to the common control. Its active median was 263.14 ms per step,
compared with 509.37 ms for control: a 48.3% latency reduction and approximately
1.94x throughput. Compared with preload alone at 288.93 ms, LPIPS caching saved
another 25.79 ms, so the two follow-up improvements remained close to additive.

The combined DataLoader region was 7.14 ms and LPIPS was 34.34 CUDA ms. The
3,070,230,528-byte LPIPS cache remained the only persistent GPU-memory cost;
peak allocated and reserved memory were 29.50 GiB and 33.49 GiB. No NaN, Inf,
OOM, traceback, or remaining GPU process was observed.

Relative to the nearby original PEG Conv3d baseline of 660.26 ms, the combined
step is 60.1% shorter and provides approximately 2.51x throughput. This is a
cross-commit historical comparison; the 509.37 ms new-SHA control is the strict
comparison for the combined experiment.

Combined output:

- `/data/home/frank/runtime/stereo-followup-profile-v1/ed30c39-h2002-gpu0-b8-40u-combined-v1`

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
Conv2d slice is the only successful PEG candidate so far. For the fixed
eight-sample acceptance run, preloading is the largest measured follow-up
improvement. LPIPS GT caching is a smaller but real improvement with a 2.859
GiB persistent cache cost. The full-dataset experiment below supersedes the
earlier conclusion about pinned memory and records the production-launcher
decision.

## Full-dataset applicability audit

The approximately 500 ms control in this document intentionally used the
frozen eight-sample manifest and `num_workers=0`. It is not a measurement of
the full 3407-sample steady state. The formal launcher already defaults to
eight DataLoader workers, and a full epoch contains about 425 complete batches
instead of restarting the epoch after every update.

On H200-2, the full manifest contains 3407 samples. The RGB cache occupies
16.08 GB and is effectively uncompressed inside NPZ. The GT cache occupies
5.47 GB but expands to about 13.43 GB, with a sampled raw-to-disk ratio of
2.45x. The node has 192 CPU cores and approximately 1.9 TiB available RAM at
the audit point.

A read-only CPU DataLoader check consumed 35 batches per setting and measured
the last 30 batches without running a model or using a GPU:

| Workers | Mean batch wait | Median batch wait | Mean samples/s |
| ---: | ---: | ---: | ---: |
| 0 | 343.8 ms | 335.9 ms | 23.3 |
| 4 | 83.5 ms | 47.2 ms | 95.8 |
| 8 | 80.6 ms | 16.3 ms | 99.2 |
| 16 | 115.4 ms | 18.3 ms | 69.3 |

With four or eight workers, mean data production is substantially faster than
the approximately 288.9 ms compute-only step measured with PEG Conv2d and no
LPIPS cache. Data loading should therefore be mostly hidden by GPU work during
a long epoch. Sixteen workers produced worse mean throughput and larger tails,
so increasing worker count without measurement is not recommended.

The next strict measurement should use the full manifest, fixed sample order,
batch 8/BF16, PEG Conv2d, no preload, no LPIPS cache, and eight workers. A
second run should change only pinned memory. Further independent candidates are
batched prediction/target VGG execution for LPIPS, followed by `torch.compile`
only after the full-manifest input pipeline is measured. Re-encoding GT as
uncompressed or memory-mappable data should be considered only if DataLoader
wait remains on the end-to-end critical path; its decompression cost may
already be hidden by worker prefetch.

## Full-dataset F0/F1/F2 experiment

Status: completed on `h200-2`, physical GPU 0. All three runs used commit
`f22c6eeabf615aa4846f4baed786c36fb1399364`, the complete 3407-sample manifest,
batch 8, BF16, eight DataLoader workers, 40 optimizer updates, the same seed,
and no preload or LPIPS cache. F0 used the original Conv3d PEG and pageable
memory. F1 changed only PEG to the T=1 Conv2d slice. F2 changed only F1's
DataLoader memory to pinned memory with the existing non-blocking transfer.

| Run | Active median | Post-profile median | Main-process loader wait | H2D CUDA | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| F0 Conv3d, pageable | 452.06 ms | 440.90 ms | 1.32 ms | 25.55 ms | 30.11 GiB |
| F1 Conv2d, pageable | 301.81 ms | 293.37 ms | 1.28 ms | 24.92 ms | 29.69 GiB |
| F2 Conv2d, pinned | 264.45 ms | 255.99 ms | 0.05 ms | 3.33 ms | 29.70 GiB |

F1 reduced active-step latency by 33.2% relative to F0 and provided 1.50x
throughput. F2 reduced active-step latency by another 12.4% relative to F1.
Relative to F0, F2 reduced active latency by 41.5%; its post-profile steady
speed was about 256 ms/step, or 31.25 samples/s on one GPU. F1 and F2 produced
exactly equal values for every logged metric at all 40 updates. F0 and F1 had
the same first two generator losses and then developed bounded BF16
reduction-order differences, consistent with the PEG microbenchmark.

The full-data result changes the interpretation of pinning. With eight workers,
NPZ loading and collate are prefetched off the critical path, while pinning
reduces H2D from about 25 ms to 3.3 ms and removes nearly all main-process
loader wait. F2 is therefore the appropriate measured single-GPU baseline for
the planned distributed comparison. Preloading remains limited to fixed-small
acceptance runs and is not proposed for full training. The LPIPS cache also
remains excluded from full training because a per-sample cache does not scale
with dataset size and its persistent device-memory cost was measured only for
eight samples.

Outputs and result SHA256 values:

- F0: `/data/home/frank/runtime/stereo-full-profile-v1/f22c6ee-h2002-gpu0-b8-40u-f0-conv3d-w8-v1`, result `3b250c28ac13c8ec153729da2bc708854b81f1cb6ae0fd590138a5cce8c2db58`
- F1: `/data/home/frank/runtime/stereo-full-profile-v1/f22c6ee-h2002-gpu0-b8-40u-f1-conv2d-w8-v1`, result `0d53026516f28621d71769bed761670a135f018dadb2d0aaef4b4191d33b02e7`
- F2: `/data/home/frank/runtime/stereo-full-profile-v1/f22c6ee-h2002-gpu0-b8-40u-f2-conv2d-w8-pinned-v1`, result `1dc2b21a208879e67d51767686d07897bdc93b858565d054dca9d15557823fe1`

## Launcher promotion and verification

The formal `scripts/stereo/train_stereo_vae.sh` launcher now explicitly passes
`--peg_backend conv2d_t1_slice`. The model default remains Conv3d for backward
compatibility with direct API construction, while the training launcher opts
into the measured T=1 path. Conv2d fails closed if a future call presents
`T != 1`; that smoke check raised the expected `RuntimeError` on H200-2.

The production change is at commit
`1b278c1382889f5915172057258e60e0974647ca`. Twenty-one targeted H200
integration and entrypoint tests passed. The PEG microbenchmark again produced
identical forward values and zero gradients for inactive temporal slices. At
the large BF16 PEG shape, forward plus backward changed from 29.26 ms for
Conv3d to 0.96 ms for Conv2d. A broad discovery run passed 53 of 54 tests; the
only failure was the pre-existing source-boundary check seeing 28 legacy
`__pycache__` files. Those unrelated files were preserved.

The documented `/data/home/maxliu/runtime/ngadv1-hy/bin/python` is currently a
broken symlink on H200-2. Validation and profiling instead used Frank's
`omnitokenizer-e2` Python 3.12.11 environment, which provides PyTorch 2.7.1 and
PyTorch Lightning 2.5.6. This runtime drift must be resolved or explicitly
pinned before launching the eight-GPU overfit run.

## Planned 128-sample eight-GPU comparison

The H200-2 overfit manifest was revalidated at 128 records with SHA256
`3df1278276ef855c605b774af3ff34dcb13a23ca2c8481698698e0faea86700c`.
The requested distributed recipe uses one node, eight GPUs, batch 8 per GPU,
no accumulation, and therefore global batch 64. It has two optimizer updates
per 128-sample epoch, whereas the single-GPU batch-8 run has sixteen.

The speed comparison should use two paired, update-limited measurements with
the same manifest, model, BF16 mode, Conv2d PEG, eight workers per rank, pinned
memory, loss, optimizer, and logging: one GPU at batch 8 and eight GPUs at
batch 8 per GPU. Record median wall time per update, global samples/s, peak
memory, DDP communication, and throughput scaling. With measured times `t1`
and `t8`, scaling is `8 * t1 / t8` and efficiency is `t1 / t8`. No eight-GPU
speed is claimed before this measurement.

The requested comparison is valid for systems throughput but not for
optimization equivalence because global batch changes from 8 to 64. Any loss
or convergence comparison needs a separate single-GPU control with eight-step
gradient accumulation so both sides have global batch 64 and the same number
of sample exposures.

The current epoch-based checkpoint callback would save every two distributed
updates and contaminate both throughput and long overfit timing. Before the
eight-GPU run, the launcher needs an explicit checkpoint cadence suitable for
an update-based overfit and persistent DataLoader workers should be measured,
because a two-batch epoch otherwise restarts worker processes repeatedly.
Those are proposed experiment changes, not yet implemented or launched.
