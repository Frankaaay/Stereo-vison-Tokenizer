# H200-2 multi-view mono forward equivalence probe

## Purpose

Compare the trained StereoVAE reconstruction path when Hy/LIBERO monocular
views are evaluated separately versus folded into one model invocation. Confirm
that stereo remains on its native three-view/two-eye path and that four frames
are processed jointly in one forward rather than as four independent calls.

This is a bounded two-case probe per dataset, not a full-split admission test.

## Provenance

- Date: 2026-09-02 CST
- Host/user: `h200-2` (`frank`, `lacy--214-30-239-42`)
- Branch: `hezhou-las2-h`
- Git SHA: `1bf8972225fe05d4cb61ad94252f6e19561532cc`
- Remote worktree: clean before and after the run
- Runtime: `/data/home/frank/runtime/stereo-tokenizer-unified-v1`
- Read-only dependency overlays:
  - `/data/home/frank/runtime/hy-lance-export-v1/lib/python3.12/site-packages`
  - `/data/home/frank/runtime/stereo-tokenizer-wandb-overlay-v1`
- Precision/posterior: BF16 autocast, deterministic posterior mode
- Seed: `1234`
- T1 source frame index: `0`
- Successful output:
  `/data/home/frank/experiments/stereo-multiview-forward-ab-h2002-20260902-v3`
- Result SHA256:
  `d19ce663241a9aadf178af1c374b834628ddbdb39b4c9e06a1982d66fe251d5b`

Checkpoint:

- Path:
  `/data/home/frank/artifacts/stereo-tokenizer/stagec-update162500-20260831/last.ckpt`
- SHA256:
  `a74c3b72b32dfd296157e3b6ad24d0521731517e79e75f22786bca37c47d822e`
- Direct counters: generator `162500`, discriminator `118500`, single-frame
  `81250`, four-frame `81250`

## Data and call contract

- Hy input: `[1,3,1,3,T,256,256]`; episodes `table_013:3130` and
  `table_021:2199`.
- LIBERO input: `[1,2,1,3,T,256,256]`; episodes
  `libero_10_no_noops_lerobot:320` and
  `libero_10_no_noops_lerobot:158`.
- UMI stereo input: `[1,3,2,3,T,256,256]`; episodes
  `a051b9dbcaabef85983fc4ab856bb506` and
  `e076f2daddf14643572bb9fb7ce592c1`.
- T1 and T4 both produce latent shape `[1,V,48,1,16,16]`.
- T4 always enters one model invocation containing all four frames.

The existing public model validation still requires mono `V=1,E=1`. The
experiment therefore accepts the requested structured `[B,V,1,C,T,H,W]`
input in a wrapper, folds views to `[B*V,1,1,C,T,H,W]`, invokes the unmodified
model exactly once, and restores `[B,V,...]` outputs. The separate baseline
invokes the same model once per view. This tests the current network's batching
semantics without changing production code or claiming that the public API
already accepts mono `V=2/3` directly.

The H200-2 documented Hy manifest is the legacy high-camera schema, while the
current data class requires the newer three-camera schema. No manifest was
generated. For this probe only, the legacy test episode index was adapted in
memory and the same Lance episode/window read the explicit existing columns
`observation_images_cam_high`, `observation_images_cam_left_wrist`, and
`observation_images_cam_right_wrist`. This is sufficient for the two selected
cases but is not a replacement for a validated H200-2 three-camera manifest.

## Results

Worst separate-versus-joined tensor differences over the two cases:

| Dataset | Mode | latent max / mean abs | RGB max / mean abs | Exact equal |
|---|---|---:|---:|---|
| Hy | T1 | 0.031250 / 0.002178 | 0.250 / 0.000480 | no |
| Hy | T4 | 0.039551 / 0.002082 | 4.625 / 0.000527 | no |
| LIBERO | T1 | 0.019531 / 0.002239 | 0.250 / 0.000478 | no |
| LIBERO | T4 | 0.035156 / 0.002386 | 0.500 / 0.000361 | no |
| UMI stereo | T1 | 0 / 0 | 0 / 0 | yes |
| UMI stereo | T4 | 0 / 0 | 0 / 0 | yes |

Joined/native reconstruction means over two cases:

| Dataset | Mode | RGB L1 | global PSNR dB | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| Hy | T1 | 0.046072 | 11.1637 | 0.841107 | 0.124453 |
| Hy | T4 | 0.050748 | 6.4435 | 0.827554 | 0.131383 |
| LIBERO | T1 | 0.022722 | 11.3949 | 0.928295 | 0.066789 |
| LIBERO | T4 | 0.030289 | 6.4806 | 0.903582 | 0.090817 |
| UMI stereo | T1 | 0.042193 | 10.8517 | 0.852070 | 0.122524 |
| UMI stereo | T4 | 0.049638 | 6.3163 | 0.814830 | 0.141846 |

Joined minus separate metric changes for mono:

| Dataset | Mode | RGB L1 | global PSNR dB | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| Hy | T1 | +0.00000654 | -0.00858 | +0.000121 | -0.0000636 |
| Hy | T4 | -0.00006121 | +0.01954 | +0.0000567 | -0.0000654 |
| LIBERO | T1 | -0.00000456 | -0.01834 | +0.0000820 | -0.0001048 |
| LIBERO | T4 | -0.00000084 | +0.00532 | -0.00000283 | -0.00000885 |

The visual panels show no obvious reconstruction-quality change at normal
display scale. Their `abs diff x64` rows expose small structured BF16
differences. The large Hy T4 maximum is a sparse raw-output outlier: its mean
absolute RGB difference remains `5.27e-4`, and aggregate quality metrics remain
close. The two-case evidence supports practical visual similarity, not strict
numerical equivalence.

## Execution health and failures

- v1 failed before GPU work because the external script directory omitted the
  repository from `PYTHONPATH`.
- v2 completed all Hy/LIBERO forwards and panels, then failed while recording a
  nonexistent UMI `frame_index` metadata key. UMI uses `start_frame`; model
  execution itself did not fail.
- v3 corrected only that experiment metadata field and exited `0`.
- v3 produced `results.json` plus 12 non-empty PNG panels.
- No model, dataset, checkpoint, or production source was modified.
- No process remained after completion; all eight GPUs reported `0 MiB`.

## Conclusion

The native stereo `V=3,E=2` route is unchanged and deterministic for both T1
and T4. Folding Hy/LIBERO views into one BF16 model invocation preserves
two-case reconstruction metrics and visual quality to a very small aggregate
difference, but it is not bitwise or numerically exact. A production API change
should therefore preserve one isolated sequence per view, add direct `V=2/3`
shape/output tests, define numerical tolerances, and run a larger fixed-split
comparison before being described as equivalent.
