# Retire the Manifest v3 stereo dataset route

## Purpose and scope

- Date: 2026-08-26
- Local repository: `C:\Project\Stereo-vison-Tokenizer`
- Branch: `hezhou-las2-h`
- Starting HEAD: `81ccce71674728490c2dcabd5715588a1469283a`
- Status: local implementation and validation; not committed or pushed

This change retires `StereoManifestDataset` and the pre-generated stereo RGB/GT
Manifest v3 route. It does not delete or mutate any repository-external dataset,
RGB cache, GT cache, checkpoint, or training output.

## Resulting data and supervision routes

- Stereo data is loaded only through `LeRobotStereoDataset`.
- Stereo targets are generated online by `OnlineFoundationGTCallback` during
  training, using the selected `las2_h`, `pytorch`, or `tensorrt` backend.
- Evaluation also constructs `FoundationStereoOnlineTeacher` directly and
  supports the same three backends.
- The optional online stereo GT cache remains incremental: cache hits are read,
  misses run the teacher, and newly generated targets are written back.
- Mono data remains `HyMonoSmokeDataset`; mono targets remain online DA3 targets
  with the existing optional DA3 cache.
- Four-mode mono/stereo x single/four scheduling, samplers, single-frame source
  selection, target conversion, losses, and checkpoint callbacks are retained.

## Removed code and entrypoints

- Removed `StereoManifestDataset` from `stereo_tokenizer/data.py`.
- Removed the Manifest v3 data-backend selector and its manifest/RGB/GT CLI
  arguments from the DataModule, training validation, launcher, and evaluation.
- Removed `scripts/data/build_stereo_rgb_cache.py`.
- Removed `tests/stereo/test_rgb_cache_contract.py` and updated source-boundary
  tests to require the online-only route.
- Moved cache-relative path validation to module-level `_resolve_cache_path`, so
  the retained Hy mono loader no longer depends on the retired stereo class.
- Explicit LeRobot eval splits bypass the fixed training-validation subset, so
  exact train/val/test evaluation still checks the complete requested split.

The old standalone step profiler referenced by the handoff was already removed
in commit `81ccce7`; no additional profiler deletion was needed here.

## Cache provenance limitation

The retained online cache namespace records teacher backend/checkpoint/iteration
provenance, but its current key does not encode every target-semantic input. In
particular, `single_frame_source_index`, disparity range, and LR-consistency
thresholds are not all part of the cache key. Existing cache contents therefore
must only be reused with the exact configuration that created them. Expanding
the schema would invalidate the current namespace and is intentionally outside
this route-removal change.

No `cache_only` mode, lazy-teacher lifecycle, new public interface, or data
format was introduced.

## Validation

Completed local checks:

- Changed Python sources parse successfully with `ast.parse`.
- `C:\Program Files\Git\bin\bash.exe -n
  scripts/stereo/train_stereo_vae.sh` passes.
- `python -m unittest tests.stereo.test_source_boundary
  tests.stereo.test_entrypoints_source` passes: 24 tests.
- Active source, launcher, tests, and README contain no retired Manifest v3
  identifiers except negative source assertions.
- `git diff --check` passes.

The deeper `test_lerobot_online_contract` and `test_mixed_mode_data` modules
could not be imported by the current local Python because `torch` and
`pytorch_lightning` are absent. No dependency or environment mutation was made.

No training, evaluation, preprocessing, remote write, or GPU workload is part of
this cleanup.

## Coexisting local cleanup

The worktree already contained the separately authorized removal of
`scripts/data/inspect_hy_lance_schema.py` and
`scripts/stereo/benchmark_batch_scaling.sh`, plus its documentation update.
Those changes were preserved and are not attributed to this route retirement.
