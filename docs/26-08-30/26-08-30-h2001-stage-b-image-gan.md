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

## Status

Implementation and directed validation are in progress. The formal output directory,
runtime commit, tmux name, launch health, and ETA will be appended after launch.
