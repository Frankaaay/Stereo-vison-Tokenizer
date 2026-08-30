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

## Stage B update-100000 evaluation

- Stage B completed normally at 2026-08-30 13:16 CST with `exit_code=0`; the tmux
  session and training processes exited, all eight H200 GPUs returned to 0 MiB, and
  the strict error scan found no traceback, CUDA OOM, NCCL error, or runtime error.
- The selected readable checkpoint is
  `/data/home/frank/experiments/stereo-three-source-stageb-imagegan-bs192-h2001-20260830-v1/train/checkpoints/best-epoch=0-step=112000.ckpt`.
  Its directly loaded `stereo_update_counters` records 100,000 generator updates,
  56,000 discriminator updates, 50,000 single-frame updates, and 50,000 four-frame
  updates. The filename/global step is not being used as the generator counter.
- A same-contract comparison evaluation started at 2026-08-30 14:59 CST in tmux
  `stereo-stageb-step100000-eval-h2001-v1`, writing to
  `/data/home/frank/experiments/stereo-stageb-step100000-eval-h2001-20260830-v1`.
  It reuses the successful Stage A v3 train/test contract: strict checkpoint load,
  BF16 on one H200, real DA3-BASE/LAS2-H teachers, 20 BS24 batches for every
  mono/stereo x single/four mode on each split, and two deterministic Hy plus two
  deterministic UMI reconstruction cases per split.
- The one-shot startup check found the tmux and evaluation process alive during
  checkpoint/dependency initialization, with no exit marker or immediate error.
  CUDA context creation had not occurred at that snapshot. Based on the prior v3
  run, the evaluation-body ETA is 15:04-15:11 CST; metrics, all 16 reconstruction
  panels, exit status, and artifact validation are expected by 15:06-15:14 CST.
- The evaluation completed normally at 2026-08-30 15:01 CST with `exit_code=0`.
  The tmux and process exited, the strict error scan remained empty, train/test
  metrics JSON files are readable, and all 16 referenced reconstruction PNGs exist
  with nonzero size (eight for each split; RGB and depth for two mono and two stereo
  cases).
- Against the same Stage A update-44,000 test cases, Stage B test RGB L1 changed as
  follows (lower is better): mono single 0.01875 -> 0.02303 (+22.83%), mono four
  0.01878 -> 0.02323 (+23.68%), stereo single 0.03912 -> 0.04090 (+4.54%), and
  stereo four 0.04299 -> 0.04192 (-2.48%).
- Test relative-log depth L1 improved for mono four from 0.06644 to 0.05897
  (-11.24%) and for every stereo view by approximately 14.99-17.29%. Mono single
  regressed from 0.11155 to 0.11623 (+4.20%). These are deterministic quantitative
  comparisons; visual quality of the new panels has not yet been inspected.

## Stage C Video GAN transition implementation

- The user authorized implementing, testing, pushing, and launching Stage C from
  generator update 100,000 to 300,000 with LPIPS 1.0, Image GAN 0.005, Video GAN
  0.005, feature matching 0, uniform BS24/GPU and GA1, and unchanged four-mode/data
  contracts. Video GAN remains four-frame-only by model contract.
- A Lightning strict resume is invalid because the Stage B checkpoint contains an
  Image discriminator and a two-optimizer topology but no Video discriminator. The
  existing GAN-free stage-transition path is also intentionally invalid because it
  rejects every source discriminator weight.
- The new explicit `discriminator_expansion_checkpoint` path is fail-closed: the
  source must contain Image discriminator weights, no Video discriminator weights,
  positive discriminator updates, and exactly generator/discriminator optimizer
  states. Model loading must preserve every existing key and may miss only the new
  Video discriminator keys.
- At optimizer initialization, the generator optimizer state is restored exactly.
  The shared discriminator optimizer maps all existing Image discriminator Adam
  states to the unchanged leading parameter sequence and leaves only the appended
  Video discriminator parameters without state. New 300k schedulers are aligned to
  the checkpoint's direct generator/discriminator counters. No optimizer is silently
  reset.
- The selected Stage B source is
  `/data/home/frank/experiments/stereo-three-source-stageb-imagegan-bs192-h2001-20260830-v1/train/checkpoints/best-epoch=0-step=112000.ckpt`,
  directly verified as generator/discriminator updates 100,000/56,000 with 16 Image
  discriminator state keys, zero Video discriminator state keys, and two optimizer
  states. Implementation validation and H200 smoke are pending before formal launch.
