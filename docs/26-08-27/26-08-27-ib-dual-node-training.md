# H200 dual-node InfiniBand training

## Status

- Date: 2026-08-27 (Asia/Shanghai).
- Status: optional IB implementation and the approved minimal link validation
  are complete. The four-mode training smoke was removed from tonight's scope
  by the user after H200 GPU contention was observed.
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

### Pushed implementation and H200 tests

- Implementation commit:
  `9f9196448ba5c20f1747c8b1d24e7b58e51426b8`
  (`feat: add optional dual-node IB training`).
- H200-2 project runtime: Torch `2.7.1+cu126`, CUDA `12.6`, Lightning `2.5.6`.
- Full H200-2 `tests/stereo` suite: `135 passed, 4 warnings in 6.26s`.
- The two formal server clones were being changed by concurrent tasks, so the
  user authorized isolated worktrees at
  `/data/home/frank/worktrees/Stereo-vison-Tokenizer-IB-test`. Both were clean
  at the implementation commit for the successful probe.

### Minimal two-node IB collective

The user narrowed the remote GPU scope to proving the IB link only. The final
probe used physical GPU 7 on each node, one process per node, one scalar
all-reduce, `bond0` out-of-band bootstrap, and only `mlx5_7:1` for NCCL data
transport. It did not run the model, load data or teachers, or write a
checkpoint.

```text
output root on each node:
/data/home/frank/experiments/stereo_ib_collective_2node_1gpu_20260827_v2
master: 214.30.239.40:29642
world size: 2
CUDA_VISIBLE_DEVICES: 7
NCCL_SOCKET_IFNAME: =bond0
NCCL_IB_HCA: mlx5_7:1
```

Both node launchers exited `0`. Rank 0 emitted:

```json
{"ranks":[{"all_reduce_sum":3.0,"hostname":"lacy--214-30-239-40","rank":0,"world_size":2},{"all_reduce_sum":3.0,"hostname":"lacy--214-30-239-42","rank":1,"world_size":2}],"status":"ok"}
```

The NCCL logs on both nodes report all of the required transport evidence:

- `NET/IB : Using [0]mlx5_7:1/IB [RO]`;
- `Using network IB`;
- `GPU Direct RDMA Enabled for HCA 0 'mlx5_7'`;
- bidirectional send/receive channels `via NET/IB/1/GDRDMA`.

No probe process remained afterward. Physical GPU 7 returned to 4 MiB on
H200-1 and 0 MiB on H200-2. The nodes have an approximately four-minute clock
offset and c10d printed hostname reverse-lookup warnings during rendezvous;
TCP rendezvous nevertheless connected and the collective completed correctly.

An earlier `v1` attempt is not IB evidence: H200-2's formal clone was switched
by a concurrent task after preflight, so its probe file disappeared before the
rank entered NCCL. The H200-1 process from that attempt was explicitly stopped;
no foreign process was signaled. Isolated worktrees removed that race for `v2`.

The result proves the selected two-node `mlx5_7` NCCL/GDRDMA path. It does not
yet prove multi-rail scaling, 2x2 or 2x8 rank behavior, model memory, four-mode
training, checkpointing, resume, or full-training throughput.

### Two-node 16-rank IB collective

After both clusters were reported idle, a fresh read-only snapshot confirmed
all 16 H200 GPUs at 0 MiB with no compute process, all `mlx5_0` through
`mlx5_7` ports `Active/LinkUp/InfiniBand`, clean identical worktrees at
`45891c52ebef5bb0b6ba714b49eb8fb7ccf2f68b`, an unused rendezvous port, and a
new output path. The authorized probe then used eight ranks per node:

```text
output root on each node:
/data/home/frank/experiments/stereo_ib_collective_2node_8gpu_20260827_v1
master: 214.30.239.40:29643
world size: 16
CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
NCCL_SOCKET_IFNAME: =bond0
NCCL_IB_HCA: mlx5_0:1,...,mlx5_7:1
```

Both launch sessions exited `0`. Rank 0 gathered all global ranks 0 through 15
with the expected host/local-rank mapping and every rank reported
`all_reduce_sum=136.0`, equal to `1+...+16`. Each node produced eight NCCL logs.
The logs on both nodes show:

- `Using network IB`;
- bidirectional cross-node channels `via NET/IB/.../GDRDMA`;
- `NET/IB : Using` with all eight physical devices `mlx5_0:1` through
  `mlx5_7:1`;
- GPU Direct RDMA enabled for every physical HCA.

The first transport summary appeared only after a long c10d initialization:
hostname reverse-lookup warning `err=-3` repeated for several minutes before
all ranks entered NCCL. The collective nevertheless completed within the
bounded run and returned success. This is a startup-latency issue to diagnose
before formal training, not an IB data-path failure.

No probe process remained and all 16 GPUs returned to 0 MiB. The command to
terminate the probe was issued only after the sessions had already exited; it
reported every exact probe PID as absent, so no process was signaled. A separate
H200-2 workload appeared after preflight and was left untouched.

This result proves 8+8 rank initialization, complete rank mapping, access to all
eight native HCA devices, and a correct NCCL/GDRDMA collective. Because the
payload was one scalar, it does not establish aggregate multi-rail bandwidth,
large-message scaling, model memory, training stability, checkpoint/resume, or
full-data throughput.
