# H200-1 Stage B Image GAN

## Objective

Start Stage B from the Stage A generator at update 44,000 with Image GAN enabled,
while preserving the four-mode data schedule and avoiding an invalid Lightning
strict resume across a changed optimizer/model topology.

## Transition contract

- Source checkpoint:
  `/data/home/frank/experiments/stereo-three-source-stagea-bs192-h2001-20260829-v3/train/checkpoints/epoch=0-step=44000.ckpt`
- Preserve generator/model weights and deterministic generator, mode, sample, and
  batch counters from the source checkpoint.
- Require the source to contain no discriminator weights and zero discriminator
  updates.
- Initialize the Image discriminator, generator optimizer, discriminator optimizer,
  and both schedulers fresh. Align the generator scheduler to global update 44,000
  before the first Stage B update; start the discriminator scheduler at update 0.
- Continue the four-mode schedule from logical generator update 44,000 to 100,000.
- Stage B is an explicit stage transition, not a Lightning strict resume. The new
  `stage_transition_checkpoint` provenance field records the source checkpoint.

## Requested training configuration

- H200-1, one node, 8 GPUs, BF16
- BS 24/GPU, GA 1, global batch 192 for all four modes
- mode weights 35:35:15:15; Hy:LIBERO 9:1; Hy `cam_high` only
- verified UMI LeRobot manifest; workers 8; prefetch factor 2
- LPIPS 1.0; Image GAN 0.005; Video GAN 0; GAN feature matching 0
- Image discriminator active from generator update 44,000
- target generator update 100,000

## Launch and health

- Runtime commit: `02c2793091fa7e74f533328f9d41fda9d02bc07f`
- H200-1 directed gate: shell syntax passed; 46 runtime/integration tests passed.
- Output: `/data/home/frank/experiments/stereo-three-source-stageb-imagegan-bs192-h2001-20260830-v1`
- tmux: `stereo-stageb-imagegan-bs192-h2001-v1`
- Started at 2026-08-30 00:37 CST with master port 29668.
- The immutable resolved config records `mode_schedule_start_update=44000`, the
  Stage A transition checkpoint, no strict-resume checkpoint, BF16, 8 GPUs, BS24/GA1,
  Image GAN 0.005, Video GAN 0, and feature matching 0. The run manifest records the
  same code SHA and effective global batch 192 for every mode.
- After five minutes, the run had completed about 378 Stage B logical updates at
  approximately 1.24 updates/s. All GPUs remained at 100% utilization with about
  135.8-136.0 GiB used per GPU. Image generator/discriminator loss paths were active;
  there was no traceback, CUDA OOM, NCCL error, NaN report, or exit code.
- ETA estimated at 00:42 CST: about 12.5 hours for the remaining pure update body at
  the observed rate; 13-16 hours including periodic validation and checkpointing.
  Final checkpoint/counter verification should require another 10-20 minutes.
