# StereoVAE eight-H200 batch-scaling benchmark

## Purpose and fixed contract

This benchmark measures whether larger per-device batches improve the accepted
eight-GPU path and identifies its throughput knee. It is a throughput test, not
an optimization-quality comparison: changing the global batch changes training
semantics.

- Branch: `frank-profiling`
- Code commit: `8a269d149ab27aed88b247baeab0fdd6555e30da`
- Node: `h200-1`, eight H200 GPUs
- Manifest: 128 records from `overfit_128_v3.jsonl`
- Manifest SHA256:
  `3df1278276ef855c605b774af3ff34dcb13a23ca2c8481698698e0faea86700c`
- BF16, T=1 Conv2d PEG, pinned memory, eight persistent workers per rank
- Same model, loss, Adam, learning-rate schedule, and no gradient accumulation
- 30 updates per batch size; first 10 are warm-up and the final 20 are measured
- Media callbacks and WandB are explicitly disabled only for this throughput
  benchmark so the early power-of-two media schedule does not contaminate step
  16. Model, loss, backward, optimizer, and scheduler remain active.
- Checkpoint cadence is 500 updates. Each short run produced only `last.ckpt`.

The repeated-dataset factors were 15, 23, 30, 45, and 60 for per-device batch
8, 12, 16, 24, and 32 respectively. This keeps all 30 updates inside one loader
epoch and avoids worker restart overhead.

## Code and measurement changes

The training entry point and shell launcher now default to a checkpoint every
500 updates instead of every 100. `StepTimingCallback` records peak allocated
and reserved CUDA memory for every rank. The batch benchmark wrapper samples
per-GPU utilization, memory-controller utilization, memory use, power, and SM
clock once per second with `nvidia-smi`.

DCGM, Nsight Systems, and Nsight Compute were not available in `PATH` on the
node. Therefore the utilization result below is coarse `nvidia-smi` telemetry;
it does not directly measure SM Active, Tensor Core active cycles, achieved
BF16 FLOP/s, or exact HBM bandwidth.

For telemetry summaries, the active interval is the contiguous range from the
first to the last one-second sample where all ranks hold at least 20 GiB and
the median GPU utilization is at least 50%. Internal low-utilization samples
are retained. This removes initialization and final checkpoint tails without
selecting only high-utilization samples.

## Results

| Per-GPU batch | Global batch | Mean step | Samples/s | Gain vs BS8 | Peak allocated | Peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 64 | 268.36 ms | 238.49 | baseline | 29.91 GiB | 31.24 GiB |
| 12 | 96 | 385.04 ms | 249.33 | 4.55% | 44.33 GiB | 51.01 GiB |
| 16 | 128 | 498.42 ms | 256.81 | 7.68% | 58.72 GiB | 67.68 GiB |
| 24 | 192 | 729.31 ms | 263.26 | 10.39% | 87.53 GiB | 92.15 GiB |
| 32 | 256 | 963.95 ms | 265.57 | 11.36% | 116.33 GiB | 122.63 GiB |

Marginal throughput gains are 4.55% from BS8 to BS12, 3.00% from BS12 to
BS16, 2.51% from BS16 to BS24, and only 0.88% from BS24 to BS32. The throughput
knee is therefore BS16--BS24. BS32 is the measured maximum throughput, but it
is already on the plateau.

| Per-GPU batch | Mean GPU util | P10 / median / P90 | Samples at least 95% util | Mean / max power | Mean memory util |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 93.3% | 85 / 100 / 100% | 70.8% | 388.8 / 511.7 W | 34.0% |
| 12 | 92.4% | 80 / 100 / 100% | 73.9% | 415.0 / 537.7 W | 34.6% |
| 16 | 94.3% | 100 / 100 / 100% | 91.1% | 458.6 / 540.0 W | 47.6% |
| 24 | 92.8% | 82 / 100 / 100% | 78.5% | 430.1 / 565.6 W | 34.4% |
| 32 | 95.8% | 87 / 100 / 100% | 85.3% | 472.9 / 560.9 W | 41.4% |

All five runs completed without traceback, OOM, nonfinite failure, or NCCL
error, and all GPUs were released. At BS32, `nvidia-smi` reached 124.28 GiB
per GPU. An H200 exposes about 140.4 GiB here, leaving only about 16.1 GiB, so
BS32 does not have a comfortable production safety margin. BS24 leaves roughly
46.6 GiB and gains all but 0.88% of the maximum measured throughput.

## Conclusion and candidates

Increasing batch size helps, but it cannot provide a large additional speedup:
BS24 is only 10.39% faster than BS8, and BS32 adds less than one percent beyond
BS24. The workload has reached a batch-scaling throughput plateau with sustained
high coarse GPU utilization. This establishes workload saturation, but the
available counters do not prove that H200 peak BF16 FLOP/s is reached. Mean
power remains about 430--473 W versus the 700 W power limit, which is consistent
with a mixed-kernel workload rather than one dense GEMM continuously filling
the device.

For pure throughput, BS24 is the recommended safe benchmark setting. BS16 is
the conservative knee with substantially more memory headroom. Do not promote
either to the formal overfit recipe without jointly choosing the intended
global-batch and sample-exposure semantics.

The most direct next diagnostic is a kernel/module profile at BS16 or BS24 with
SM/Tensor/HBM counters when DCGM or Nsight is available. One visible candidate
is PyTorch's warning that FP32 matmul precision remains at its default; testing
`torch.set_float32_matmul_precision("high")` may enable TF32 for remaining FP32
matmuls, but it can change numerical behavior and is not enabled here. Existing
Conv2d PEG, pinned persistent loading, long loader epochs, and the new
500-update checkpoint cadence should remain enabled.

## Outputs

Remote outputs share this root:
`/data/home/frank/runtime/stereo-batch-scaling-v1/`

- `8a269d1-h2001-8gpu-b8-30u-v1`
- `8a269d1-h2001-8gpu-b12-30u-v1`
- `8a269d1-h2001-8gpu-b16-30u-v1`
- `8a269d1-h2001-8gpu-b24-30u-v1`
- `8a269d1-h2001-8gpu-b32-30u-v1`

Each directory contains `run.log`, `step_timings.json`,
`gpu_telemetry.csv`, and `checkpoints/last.ckpt`. The timing, telemetry, and log
files were also copied to the task-local visualization artifact directory under
`stereo-batch-scaling/`.
