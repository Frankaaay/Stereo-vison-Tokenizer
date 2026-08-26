# H200 dual-node InfiniBand training

## Status

- Date: 2026-08-27 (Asia/Shanghai).
- Status: implementation and validation in progress.
- Branch: `IB-test`.
- Baseline: `80cba3661dc6de4d6967e1edf5a69823dc9d4e5e`.
- Local worktree: `C:\Project\Stereo-vison-Tokenizer-IB-test`.
- Hosts: `h200-1` (`NODE_RANK=0`) and `h200-2` (`NODE_RANK=1`).

## Scope

Add an optional fail-closed `ib` distributed mode without changing the model,
losses, teachers, optimizer, or dataset formats. The default `single` mode keeps
normal one-node training. The approved validation scope for tonight is:

1. local static and focused unit tests;
2. two nodes with two GPUs per node NCCL/IB collective probe;
3. two nodes with two GPUs per node four-mode four-update smoke.

No two-node one-GPU probe, strict resume, 16-GPU canary, or full training is in
scope tonight.

## Implementation contract

- `scripts/stereo/train_stereo_vae.sh` selects `single` or `ib`, computes
  `WORLD_SIZE=NUM_NODES*GPU_COUNT`, validates the full global batch, and uses
  `torchrun` only for IB mode.
- IB mode requires exactly two nodes, explicit node rank/master address/port,
  `bond0` bootstrap, native `mlx5_0:1` through `mlx5_7:1`, and NCCL debug logs.
- A successful IB launcher exit requires an observed `NET/IB` marker; Socket
  fallback is not accepted.
- `train_stereo_vae.py` validates torchrun world sizes, accepts four-mode world
  sizes 1/2/4/8/16, and records distributed provenance in run metadata schema
  `stereo-vae-online-gt-run-v2`.
- `scripts/stereo/check_ib_collective.py` runs a deterministic CUDA/NCCL
  all-reduce and gathers rank/host/device evidence.

## Frozen four-rank smoke recipe

```text
NUM_NODES=2
GPU_COUNT=2
WORLD_SIZE=4
PER_DEVICE_BATCH_SIZE=24
GRAD_ACCUMULATES=1
GLOBAL_BATCH_SIZE=96
MAX_STEPS=4
MODE_UPDATES_PER_EPOCH=4
```

The fixed 48-sample mono/stereo sources repeat within rank-local BS24 batches at
world size four. The result therefore establishes execution, memory, sampler,
and transport stability only; it is not unique-data throughput or convergence
evidence.

## Results

### Local validation

- Git for Windows Bash syntax check passed for
  `scripts/stereo/train_stereo_vae.sh`.
- Python compilation passed for the training entrypoint, collective probe, and
  focused test modules.
- `tests.stereo.test_entrypoints_source` passed 12/12.
- The local base Python lacks `pytorch_lightning`, so runtime-dependent sampler
  and distributed argument tests are deferred to the existing H200 project
  runtime; this is a local dependency absence rather than a test assertion
  failure.
- `git diff --check` passed apart from Git's existing LF-to-CRLF checkout
  warnings.

Server results are pending. Record the pushed commit, server SHA/status,
runtime and asset hashes, selected GPUs, exact commands,
output/log/checkpoint paths, NCCL transport evidence, metrics, failures, and ETA
here after each authorized stage.
