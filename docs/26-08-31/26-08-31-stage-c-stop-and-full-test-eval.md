# Stage C stop and full test evaluation

## Purpose

Treat the user-requested Stage C interruption as the completion boundary for this
version, preserve its checkpoint inventory, and evaluate the complete formal test
splits for all three training sources: Hy-Embodied, LIBERO, and UMI.

## Training stop

- Location: `h200-1`
- Run: `stereo-three-source-stagec-videogan-bs192-h2001-20260830-v1`
- Tmux: `stereo-stagec-videogan-bs192-h2001-v1`
- Stop requested at: 2026-08-31 12:52 CST
- Stop completed at: 2026-08-31 12:53 CST
- Launcher exit marker: `137`; the log records a user `KeyboardInterrupt`, after
  which the distributed launcher forcefully reaped the remaining ranks.
- Post-stop evidence: no matching rank process, no compute process, and all eight
  GPUs at 0 MiB process memory.

The latest stable complete endpoint is `train/checkpoints/last.ckpt`. Direct
checkpoint inspection, rather than its filename or Lightning progress bar, gives:

- `generator_updates=162500`
- `discriminator_updates=118500`
- `global_step=125000`
- 16 Image discriminator keys and 16 Video discriminator keys
- two optimizer states

Checkpoint inventory at the stop boundary is 129 files: 125 periodic checkpoints,
three `best-*` checkpoints, and one `last.ckpt`. Periodic checkpoints are retained
without a top-k limit; validation-best retention remains top 3.

## Full-test evaluator support

The existing evaluator handled Hy as its sole mono source and UMI as stereo, so it
could not truthfully cover all three formal test datasets. Add the minimal
`--mono_dataset {hy,libero}` selector while preserving `hy` as the default. LIBERO
uses the existing `LiberoMonoDataset` decode and timestamp contract, the same DA3
teacher, and separate provenance. Fixed visualization selection accepts either the
Hy `table_name` identity or LIBERO `suite` identity.

Formal manifest inventory on `h200-1`:

- Hy test: 2,897 records and 223,371 windows
- LIBERO test: 87 records and 1,828 base windows
- UMI test: 3,132 records and 69,167 windows

## Validation and evaluation status

- Local static gate: `python -m py_compile` passed.
- Local tensor test gate is unavailable because the Windows Python environment has
  no Torch; the directed test must run in the pinned H200 runtime after the pushed
  commit is fast-forwarded.
- The first H200 directed test correctly rejected a fixture whose mocked LIBERO root
  did not exist. The fixture was changed to a real temporary directory; production
  fail-closed path validation is unchanged.
- Implementation commits: `b01344ba4ad34a3e29f66d4b6af89821d72b9e6d`
  and fixture-only follow-up `75051e71b193290814e5978af2dc7cab2eb5f711`.
  Both were pushed to `origin/hezhou-las2-h`; H200-1 is clean and fast-forwarded to
  the latter exact SHA.
- H200 directed tests: 18/18 passed in the pinned runtime.
- Full test evaluation started at 2026-08-31 12:59 CST in tmux
  `stereo-stagec-update162500-full-test-eval-v1`, writing to
  `/data/home/frank/experiments/stereo-stagec-update162500-full-test-eval-h2001-20260831-v1`.
  The launcher directly re-verifies generator update 162,500 and runs an 8-GPU,
  one-batch real LIBERO/DA3 smoke before the complete Hy, LIBERO, and UMI test splits.
  Each full dataset covers both temporal modes and saves two deterministic RGB/depth
  cases.
- The one-shot startup check found the smoke torchrun alive during dependency/model
  initialization, with no exit marker or immediate error. CUDA contexts had not yet
  appeared. Based on the prior single-GPU sampled evaluator and the now 8-way exact
  sharding, the initial full-body ETA is 15:00-17:00 CST; JSON/artifact validation
  and copying viewable cases are initially expected by 15:15-17:30 CST. This is a
  historical-throughput range and must be refreshed from the next user-requested
  live snapshot after the real full-split loop begins.
