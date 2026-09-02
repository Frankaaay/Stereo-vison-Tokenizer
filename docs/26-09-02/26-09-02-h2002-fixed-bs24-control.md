# H200-2 fixed-BS24 continuation control

## Purpose

Run a concurrent fixed-BS24 control for the active H200-1 mode-aware Stage A
continuation and compare actual processed-sample throughput. Both runs continue
from generator update 44,000 toward update 200,000 with GAN disabled and W&B
offline.

## Provenance and contracts

- H200-2 host/user: `h200-2` (`frank`, `lacy--214-30-239-42`).
- Branch: `hezhou-las2-h`; H200-2 launch SHA:
  `242c970c7790cdd0a08d7babe54accf992cd8c6f`.
- H200-2 worktree was clean before launch. Its training source is identical to
  the active H200-1 source; the later H200-2 SHA changes only manifest tooling
  and documentation.
- Source checkpoint:
  `/data/home/frank/artifacts/stereo-tokenizer/stagea-update44000-20260829/best-epoch=0-step=44000.ckpt`,
  SHA256 `d22c11a7630bc36bb6168acc1452ddf3ba21c257418a08a12a76b8fe41a348b3`.
- Resolved continuation fields confirm source generator update 44,000 and mode
  schedule start update 44,000. Maximum generator update is 200,000.
- Control mode batches are `24:24:24:24`; mode accumulation is `1:1:1:1`;
  effective global batch is 192 for every update on eight GPUs.
- Mode weights are `35:35:15:15`, mono Hy:LIBERO is `9:1`, LR is 1e-4,
  BF16 is enabled, all GAN weights are zero, online DA3/LAS2-H teachers are
  enabled, and online GT cache is disabled.
- Offline W&B run ID: `55ljq1oa`.

H200-2 uses its required node-local manifests. The validated three-camera Hy
manifest is
`/data/home/frank/runtime/stereo-tokenizer-pretrain-h2002-20260902-threeview-v1/manifests/hy_formal_90_5_5_threeview_validated_v3.jsonl`,
SHA256 `b3688640fae412daa0c4011129098bfb8a0a7e1247c0884835a5f2da9c886bcb`.
It contains 35,115 accepted episodes and 8,183,196 three-camera windows. LIBERO
SHA256 is `0299354a7225e979f6b9ff5fb3e26a975c811d2d41af44e042a6eade3f24bbf4`;
UMI SHA256 is `96024f091bcf7aca844b4d4b99fad2eb6cb0f420aa693f1431340b79ac5fa53e`.

## Launch and health

- Output root:
  `/data/home/frank/experiments/stereo-three-source-stagea-threeview-fixedbs24-h2002-20260902-v1`.
- tmux: `stereo-stagea-threeview-fixedbs24-h2002-v1`.
- Launch script SHA256:
  `3d76d9c83a5ab7946a81c69dbba92b984f0c71010e107044f52a583958843a24`.
- The first health gate passed generator update 44,001. All eight GPUs were at
  approximately 90--100% utilization with no traceback, invalid image, OOM, or
  exit marker.

## Initial direct throughput comparison

The comparison uses actual global samples, not raw Lightning step rate. For
the H200-1 mode-aware run, each completed logical update contributes 384
samples; a partially observed stereo/four GA2 update contributes 192 samples
per physical batch. For the H200-2 fixed-BS24 run, every physical batch and
logical update contributes 192 samples.

Two snapshots approximately five minutes apart produced:

| Run | Delta samples | Delta training-loop seconds | Samples/s |
|---|---:|---:|---:|
| H200-1 mode-aware BS48/24, GA1/2 | 100,992 | 405 | 249.36 |
| H200-2 fixed BS24, GA1 | 90,240 | 406 | 222.27 |

The observed mode-aware speedup is `12.19%`. This agrees in direction and is
reasonably close to the earlier strict H200-2 equal-sample result of `14.56%`,
but it is not a strict reproduction: the current arms run concurrently on
different nodes, use node-local Hy subsets, and use different pinned runtime
layouts. Both formal runs remained active at the second snapshot. The snapshot
progress inferred generator updates 52,445 on H200-1 and 44,576 on H200-2; the
H200-2 resolved config, rather than this inference, is the direct evidence for
the 44,000 continuation source.

## Longer-window result and control shutdown

A later pair of snapshots used an approximately 13.4-minute interval:

| Run | Delta samples | Delta training-loop seconds | Samples/s |
|---|---:|---:|---:|
| H200-1 mode-aware BS48/24, GA1/2 | 200,832 | 803 | 250.10 |
| H200-2 fixed BS24, GA1 | 177,600 | 801 | 221.72 |

The observed mode-aware speedup was `12.80%`. This remained in the same range
as the initial `12.19%` window and the historical strict equal-sample result of
`14.56%`. Per the experiment stop criterion, the H200-2 control received a
normal `Ctrl-C` through tmux after this snapshot.

Post-stop checks confirmed that the H200-2 tmux session no longer existed,
there were no matching training processes, and `nvidia-smi` reported no compute
processes. The wrapper recorded `exit_code.txt=1`, which reflects the requested
interrupt rather than a training fault; the log ended at physical batch 1,581
without a traceback. Offline W&B artifact `run-55ljq1oa.wandb` is nonempty.

The latest completed checkpoint is
`stereo-vae/55ljq1oa/checkpoints/last.ckpt` (729,458,299 bytes). Direct checkpoint
fields report Lightning `global_step=1000` and
`stereo_update_counters.generator_updates=45000`; its counter transition records
the source generator update as 44,000. Work completed after update 45,000 and
before the interrupt is represented in the log/W&B run but not in a newer
checkpoint. The H200-1 formal run remained active after the control shutdown.
