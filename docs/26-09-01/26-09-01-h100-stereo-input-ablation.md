# H100 stereo-input ablation

## Objective

This experiment tests whether the frozen Stage A StereoVAE uses geometrically
correct right-camera input.  It deliberately excludes GAN and rGAN so that RGB
realism cannot confound the stereo mechanism conclusion.  The implementation
lives on `exp/stereo-input-ablation-h100` and preserves the existing default
forward path and checkpoint ABI.

## Frozen assets and data contract

- Student checkpoint: Stage A update 44,000, copied from H200-1 and accepted
  only with SHA256
  `b86f938a584476ee8ce47bdd635432deed08994225dad68ff6283ebd0d27a213`.
- Teacher: LAS2-H source
  `8c97bd4c4da3712c2ac60003a23201dfdb5935f4`, valid iters 4, max disparity
  192, checkpoint SHA256
  `758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4`.
- H100 data:
  `/gpfs/jiuquyun/datasets/PRETRAIN_DATA/UMI-Collectsite-KS3-canonical-v3`.
- Episode split is deterministic SHA256 hashing with seed 1234 and ratios
  90/5/5. Four-frame windows preserve the training offsets `[0,3,6,9]` and
  stride 12.
- Published RGB is already 256x256. Pixels outside
  `image_pixel_mask_umi.npz` are replaced with uint8 128, matching the Stage A
  padding convention.

Canonical-v3 does not publish per-episode focal length or baseline.  The
primary geometry metric therefore centers log-depth independently for every
sample and camera view, which exactly removes the unknown positive camera
scale. Metric depth, disparity EPE, and D1 are not claimed in this report.

## Fail-closed data gate

The manifest builder checks all six video streams, exact episode interval
synchrony, episode lengths, 30 FPS, source identity, and file containment.  The
gate then samples 100 test episodes and all three stereo views. It requires:

- metadata interval synchronization for all six streams;
- successful frame decode ratio at least 99%;
- at least 20 reciprocal Lowe-ratio ORB matches for at least 99% of pairs;
- aggregate vertical residual median at most 1 px and P95 at most 2 px for
  every view.

A failed gate terminates model evaluation. The failure JSON and example stereo
pairs remain the reportable outcome.

## Frozen-checkpoint conditions

The evaluator runs the same deterministic windows and cached LAS2-H tensors for
`REAL_STEREO`, `COPY_LEFT`, `FUSION_OFF`, episode-deranged `WRONG_RIGHT`,
zero-filled horizontal shifts at -32/-16/+16/+32 pixels, and four-frame
`TIME_REVERSE`. The teacher is called before student intervention, cached per
sample, and protected by a target checksum. `fusion_scale_override=0` changes
no parameter and makes the fusion residual exactly zero.

Profiles are smoke 8x2, diagnostic 64x4, and main 128x8 episodes/windows. Each
profile evaluates Stage A source-frame index 0 and four-frame mode for all
three views. The paired report uses 10,000 episode-cluster bootstrap draws.

The engineering pass rule is the approved joint gate: REAL must beat both
COPY_LEFT and FUSION_OFF by at least 5% with a positive 95% CI lower bound,
beat WRONG_RIGHT significantly, clear 5% in at least two views with no view
worse than -2%, and exhibit the required shift response. Attention diagnostics
are explanatory only.

## H100 execution

Login nodes only build manifests, run the CPU audit, check `sinfo`/`squeue`, and
submit jobs. GPU execution is Slurm-only. Smoke uses one H100/debug; diagnostic
and main use eight H100/normal. The launcher refuses a dirty repository, a Git
SHA mismatch, an existing output directory, a failed data gate, or checkpoint
hash mismatch. Every run stores environment, Git, Slurm, GPU, configuration,
raw metrics, paired samples, bootstrap output, visualizations, and a single-file
offline HTML report.

If the zero-training main result is inconclusive after excluding a data fault,
the next commit will add `left_only` and launch the approved paired 2k/5k/10k
training arms from identical step-0 initialization. No one-sided retraining is
permitted.

## Status

Implementation prepared locally; H100 validation and run identifiers are
pending. This document will be updated with the exact committed SHA, Slurm job
IDs, measured throughput, outcome, and final artifact paths.
