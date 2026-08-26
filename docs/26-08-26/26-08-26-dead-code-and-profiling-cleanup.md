# Dead code and legacy profiling cleanup

## Purpose and scope

This cleanup removes code that is not reachable from the current structured
mono/stereo VAE training and evaluation paths. It is split into three commits:

1. remove confirmed unused imports, helpers, fields, and parameters;
2. remove obsolete compatibility APIs and unreachable Transformer branches;
3. remove the old standalone step, PEG, and LPIPS-cache profiling experiments
   together with their profiler-only data/model/test paths.

The target branch is `hezhou-las2-h`. The reviewed base commit is
`45fec2e341eb83ade04c9a1c88d824e49f3c9b5f`; the first two cleanup commits are
`9247a9b` and `b761c2a`. The third cleanup is the commit containing this record.

## Removed profiling artifacts

- `profile_stereo_step.py`
- `profile_peg_backends.py`
- `profile_lpips_gt_cache.py`
- `scripts/stereo/profile_stereo_step.sh`
- profiler-only dataset preloading and `profile_pin_memory` override
- profiler-only fixed-target LPIPS feature cache
- the experimental `conv3d_channels_last_3d` PEG backend
- source tests that existed only to freeze the deleted profiler recipes

Historical profiling result documents are retained because they record the
measurements behind the production `conv2d_t1_slice` decision.

## Preserved production profiling contract

The cleanup keeps `stereo_tokenizer/profiling.py`, all production
`profile_region(...)` calls, the opt-in profiler in `train_stereo_vae.py`, and
the formal launcher's `TORCH_PROFILE_OUTPUT_DIR` support. The production PEG
backends remain `conv3d_contiguous` and `conv2d_t1_slice`.

## Validation

- full-repository Python AST parsing;
- `git diff --check`;
- `python -m unittest tests.stereo.test_source_boundary`;
- repository-wide reference searches for every removed API and profiling
  artifact.

Tensor-dependent unit tests could not run in the local Windows Python because
that interpreter does not have `torch` or `pytest`. No training, evaluation,
preprocessing, remote operation, or GPU task was started.

## Follow-up one-off script cleanup

After commit `81ccce7`, two additional completed experiment utilities were
removed from the working tree:

- `scripts/data/inspect_hy_lance_schema.py`: the production Hy smoke-cache
  builder already performs the required Lance schema and inventory checks;
- `scripts/stereo/benchmark_batch_scaling.sh`: the completed batch-size sweep
  wrapper only added `nvidia-smi` telemetry around the production launcher.

FoundationStereo remains a supported online backend. Its PyTorch, TensorRT,
and LAS2-H implementations, backend comparison tools, frozen teacher selection,
TensorRT manifest writer, launcher branches, and tests are intentionally
unchanged.
