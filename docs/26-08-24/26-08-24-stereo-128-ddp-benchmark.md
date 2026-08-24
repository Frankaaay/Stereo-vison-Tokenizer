# StereoVAE 128-sample single-node DDP benchmark

## Purpose and fixed inputs

This experiment measures the accepted F2 path on one and eight H200 GPUs, then
removes infrastructure overhead that is specific to the 128-sample overfit
subset. Model, loss, BF16 mode, Adam, batch size per device, and sample content
remain unchanged.

- Branch: `frank-profiling`
- Node: `h200-1`
- Manifest: `/data/shared/datasets/umi_raw_data0806_stereo_pilot_rgb_v2/overfit_128_v3.jsonl`
- Records: 128
- Manifest SHA256: `3df1278276ef855c605b774af3ff34dcb13a23ca2c8481698698e0faea86700c`
- Runtime: Python 3.12.11, PyTorch 2.7.1+cu128, PyTorch Lightning 2.5.6
- Per-device batch: 8
- One-GPU global batch: 8
- Eight-GPU global batch: 64
- Data path: T=1 Conv2d PEG, pinned memory, eight workers per rank unless
  explicitly stated otherwise

The user explicitly authorized creating the missing local `frank-profiling`
tracking branch in the existing H200-1 clone. A first `git switch --track`
attempt updated the index before rejecting the unregistered remote refspec. The
clone was clean immediately before that command. Creating the same authorized
branch at the already-fetched target SHA and completing the switch restored a
clean worktree without reset, deletion, or discarded user changes.

## Baseline and strict scaling result

At commit `bda069fd2b1368d0c7c333087edccad9d2b2ff02`, both sides ran 40 optimizer
updates on the same manifest. The one-GPU run used batch 8. The eight-GPU run
used batch 8 per rank and global batch 64.

| Measurement | One GPU | Eight GPUs |
| --- | ---: | ---: |
| Stable compute median | 254.05 ms | 270.50 ms |
| Samples per stable compute second | 31.49 | 236.60 |
| Throughput scaling | 1.00x | 7.51x |
| Scaling efficiency | 100% | 93.9% |

The one-GPU result reproduces the earlier full-data F2 value of about 256
ms/step. DDP adds about 16.45 ms to a normal update. This is the strict
same-commit scaling comparison.

The first eight-GPU attempt at the preceding commit failed before the second
iteration because default DDP rejected parameters that never participate in
the current graph. A read-only audit of the completed one-GPU checkpoint found
62 such parameters, primarily inactive `context_norm` and
`spatial_rel_pos_bias` parameters. None had Adam state. Commit `bda069f` uses
`DDPStrategy(static_graph=True)` for the fixed non-GAN graph, avoiding both the
failure and per-step unused-parameter traversal. GAN mode retains dynamic
unused-parameter detection because its graph can change with update count.

## Two-step epoch bottleneck

With 128 samples and global batch 64, the unmodified loader epoch contains only
two updates. Although normal DDP updates took about 265--270 ms, the boundary
between every pair of updates cost about 3.5--4.5 seconds. The initial end-to-end
eight-GPU mean was therefore 2.317 seconds/update, only 27.62 samples/s.

Three causes were tested:

1. Media callbacks used `batch_idx`, so batch zero of every epoch forced image
   and MP4 generation. Commit `8d0ffdda58dd00ea86a60f247d70568eafdf7d11`
   changes them to the documented global-step schedule and prevents duplicate
   logs at the same step.
2. Training metrics used `on_epoch=True` with distributed aggregation. Commit
   `da70b455edc6be6c92dbad29508467187975ce8d` keeps every-step training metrics
   and validation epoch metrics, but removes redundant training epoch
   aggregation.
3. Reducing workers from eight to one per rank did not remove the boundary and
   made normal data steps slower, so worker oversubscription was rejected as
   the root cause.

The first two changes removed real logging overhead but did not eliminate the
dominant loader-epoch boundary. The effective fix is to repeat the fixed
overfit dataset inside one DataLoader epoch. Commit
`6c11ddfdb9fde5bcacdb44f1081813624753a435` adds
`--train_epoch_repeats`, defaulting to one. It uses PyTorch `ConcatDataset`, so
the samples and schemas are unchanged. The 40-update benchmark used repeat 20:
128 samples x 20 / global batch 64 = 40 updates in one loader epoch.

## Final eight-GPU result

Output:
`/data/home/frank/runtime/stereo-128-ddp-benchmark-v1/6c11ddf-h2001-8gpu-b8-r20-40u-v1`

| Window | Step time | Global throughput |
| --- | ---: | ---: |
| Updates 6--40, including early media logs | 324.49 ms mean | 197.23 samples/s |
| Updates 33--40, after early media logs | 266.48 ms mean | 240.16 samples/s |
| Updates 33--40 median | 266.80 ms | 239.88 samples/s |

Using the measured one-GPU F2 baseline, the final steady window provides 7.63x
throughput scaling and 95.3% eight-GPU efficiency. Early media outputs remain
at global steps 1, 2, 4, 8, 16, and 32 as intended. After that warm-up region,
the next image log is step 750 and the next video log is step 1500.

The run completed 40 updates without traceback, nonfinite failure, OOM, or NCCL
error. `checkpoints/last.ckpt` is 730,736,627 bytes. All eight GPUs were released
after completion.

## Other outputs and incidents

- Same-SHA one GPU:
  `/data/home/frank/runtime/stereo-128-ddp-benchmark-v1/bda069f-h2001-1gpu-b8-40u-v1`
- Same-SHA eight GPU with two-step epochs:
  `/data/home/frank/runtime/stereo-128-ddp-benchmark-v1/bda069f-h2001-8gpu-b8-40u-v1`
- Global-step media logging check:
  `/data/home/frank/runtime/stereo-128-ddp-benchmark-v1/8d0ffdd-h2001-8gpu-b8-40u-v1`
- Train epoch-metric check:
  `/data/home/frank/runtime/stereo-128-ddp-benchmark-v1/da70b45-h2001-8gpu-b8-40u-v1`
- One-worker rejection test:
  `/data/home/frank/runtime/stereo-128-ddp-benchmark-v1/da70b45-h2001-8gpu-b8-w1-20u-v1`

Before H200-1 became the target, a one-GPU attempt on H200-2 collided with a
different user's job that started after the empty-GPU precheck. No foreign
process was killed. The first H200-1 eight-GPU launch exposed the DDP unused
parameter error described above; its processes exited and GPUs were verified
free before the static-graph retry.

## Proposed 128-sample overfit recipe

Keep the final measured path:

- eight H200 GPUs on one node
- batch 8 per GPU, global batch 64, no accumulation
- BF16, T=1 Conv2d PEG, pinned memory, eight workers per rank
- non-GAN static-graph DDP
- checkpoint every 500 optimizer updates
- global-step media logging and step-only training metrics
- one loader epoch covering the intended update budget

For a chosen `MAX_STEPS`, set
`TRAIN_EPOCH_REPEATS = ceil(MAX_STEPS * 64 / 128)`, which simplifies to
`ceil(MAX_STEPS / 2)` for this exact manifest and global batch. For example,
5000 updates would use 2500 repeats. This changes only the loader epoch boundary
and shuffle grouping; it does not change total update count or per-update global
batch. The long-run update budget and acceptance probes remain to be confirmed
before launch.
