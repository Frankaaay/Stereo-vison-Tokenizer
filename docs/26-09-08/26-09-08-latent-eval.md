# H200-2 latent 40k evaluation

- User authorized evaluation of Z24/Z48/Z96 and explicitly excluded the unreviewed
  `doc/Stereo Tokenizer Plan.md` from this task.
- All three final checkpoint generator counters are 40,000, exits 0, weights finite;
  per-mode sample counts match: 21,504,000 / 4,480,000 / 7,680,000 / 1,728,000.
- Use training-native Hy/LIBERO/UMI full test splits from the exact training manifests;
  do not reuse H100 canonical or H200-1 split contracts.
- Experiment-only script reuses metric sums from `cfa07e2^:eval_stereo_vae.py` and
  current strict checkpoint loading/teacher/temporal helpers. RGB metrics crop the
  non-padding rectangle; geometry uses the existing teacher-valid mask.
- Fixed FP32, TF32 disabled, posterior mean, batch 4, 8 exact disjoint rank shards.
  Evaluate source indices 0/1/2/3 and four-frame. All three models share each decoded
  batch and generated teacher target. Record rank sample/target hashes and checkpoint
  SHA256, full counts, RGB L1/PSNR/SSIM/LPIPS, temporal delta L1/LPIPS and per-view
  relative-log L1/RMSE/SILog. Geometry scores are teacher agreement, not metric GT.
- Scripts: `doc/frank/h2002-latent-eval-20260908.py` and matching `.sh`.
- Output: `/data/home/frank/experiments/stereo-latent-eval-h2002-20260908-v1`.
- First run one batch/rank for each dataset, then full LIBERO/UMI/Hy. Stop the serial
  launcher on any failure; no incomplete run is presented as full-test evidence.
- Prelaunch validation: local Python compile; remote smoke still pending.
