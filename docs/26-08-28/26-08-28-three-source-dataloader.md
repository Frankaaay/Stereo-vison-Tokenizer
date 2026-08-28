# Three-source StereoVAE DataLoader implementation

## Scope and provenance

- Date: 2026-08-28
- Local worktree: `C:\Project\Stereo-vison-Tokenizer`
- Branch / starting HEAD: `hezhou-las2-h` / `5b345206babe63a58acdfcb38f375059885ed8cf`
- Implementation commit: `2a3f5b46d3a9200d3b6d09f5548ed84b3279ead6`.
- Merge integration: preserve the three-source loader while incorporating the independent `d03d966` prefetch and legacy LeRobot copy-elision changes. The legacy video-container default remains 12 because the measured 36-container working set applies to the old six-camera LeRobot workload, not Hy Lance, LIBERO mono, or UMI MCAP.
- This implementation extends the existing StereoVAE loader. It does not use or import the WAM DataLoader.

## Data contract

Three manifest schemas feed one existing model batch ABI:

- Hy: `hy-cam-high-episode-v1`; only `observation_images_cam_high`, never either wrist camera; Lance JPEG frames at offsets `[0,3,6,9]`, window stride 12.
- LIBERO: `libero-mono-episode-v1`; `observation.images.image` and `observation.images.wrist_image` are independent monocular samples; offsets `[0,2,4,6]`, stride 8.
- UMI: the existing `lerobot-stereo-episode-v1` manifest and `LeRobotStereoDataset`; six LeRobot v3 MP4 streams form three calibrated, pre-rectified stereo views with offsets `[0,3,6,9]`, stride 12, and the existing 256-square student geometry. Raw MCAP is no longer the training backend.

Hy and LIBERO manifests contain logical `root_alias` values and relative paths, and launch configuration maps each alias to a physical node-local absolute root. UMI uses the existing episode manifest plus an explicit node-local LeRobot root and rectification-audit SHA. The loader contains no table parity, H200-2 table inventory, or dataset-root constants.

The old frozen `HyMonoSmokeDataset`, its 48-sample NPZ cache contract, and its source-limit arguments were removed. The existing non-mixed LeRobot stereo evaluation path remains available.

## H200-1 / H200-2 behavior

For single-node operation, code is identical on both nodes. H200-1 supplies a Hy manifest enumerating its even tables and H200-2 supplies a Hy manifest enumerating its odd tables; each launch also supplies that node's root-alias JSON. Moving nodes therefore changes manifests/configuration, not Python code.

For dual-node operation, each node still opens only its own physical manifests. `LOCAL_RANK` shards within a node, while `NODE_RANK` selects the already-generated node-local manifest set. The stateless sampler emits the same number and type of updates on every rank and deterministically cycles shorter sources. `NODE_MANIFEST_CONTRACTS` must contain the SHA256 of Hy, LIBERO, and UMI manifests for both node ranks. Each node verifies its local files against that global mapping, and the normalized mapping is stored in checkpoints.

A checkpoint whose manifest mapping, mode weights, mono dataset weights, batch size, gradient accumulation, schedule seed, or world size differs fails strict resume. H200-2 to H200-1 with a different Hy manifest is therefore a new data stage: load model weights if desired, but reset the sampler/update continuation instead of claiming strict resume.

## Sampling defaults

- Mode update weights: mono single / mono four / stereo single / stereo four = `35:35:15:15` (normalized internally to `7:7:3:3`).
- Mono source weights: Hy / LIBERO = `9:1`.
- Stereo source: UMI only.
- Each update is homogeneous in mode and dataset, so DA3 and FoundationStereo callback contracts remain unchanged.

## Manifest generation and acceptance

`scripts/data/build_pretrain_manifest.py` performs a CPU-only inventory pass:

- Hy reads every discovered `table_*` episode metadata parquet and records Lance row ranges. It does not hard-code odd/even tables.
- LIBERO reads LeRobot `meta/info.json` and `meta/episodes.jsonl` and records the canonical video path template.
- UMI accepts only reviewed episodes with completed frame metadata, inventories all six compressed-video topics, takes the shortest stream length, and records all three stereo calibrations.
- All builders write a deterministic JSONL plus a summary containing record count, window count, split counts, and manifest SHA256.

Before launch, acceptance requires: non-empty train/val records, no missing required fields, exact recomputation of every `window_count`, all alias roots present, paths contained within roots, Hy frame identity/timestamps consistent with Lance rows, LIBERO frames within the timestamp tolerance, UMI six-stream decode length at least the manifest length, valid 3x3/3x4 calibration matrices, and left/right timestamp skew within the manifest threshold.

The removed UMI MCAP throughput benchmark is not a launch gate and no benchmark/remux stage is present in this implementation.

## Verification performed

- Python syntax compilation for all changed Python entrypoints and datasets: passed.
- Bash syntax check for `scripts/stereo/train_stereo_vae.sh`: passed.
- `git diff --check`: passed (only Git's Windows LF/CRLF notices).
- Pure-Python deterministic sampler harness: passed for two node-local ranks, equal update counts, and disjoint even/odd local indices.
- Full pytest was not runnable in the local Windows interpreter because it has neither `pytest` nor `torch`; no remote source copy or environment mutation was made to bypass the project's local-edit/pushed-SHA rule.

Merge verification additionally covers the explicit `prefetch_factor=2` launcher/parser wiring, positive-value validation, the legacy LeRobot NumPy copy elision, the retained cache default of 12, the combined dependency lock, and the three-source entrypoint assertions.
The 12 source-level entrypoint tests pass after correcting the evaluation assertion to check the inherited parser's runtime use (`args.hy_manifest`) rather than requiring the delegated option registration to be duplicated in `eval_stereo_vae.py`.

## Remaining launch gates

Before H200 execution, push the verified merge revision and fast-forward the clean server clone to that exact SHA. Then generate and inspect the node-local manifests, install/sync the newly declared `pylance`, `mcap`, and `mcap-protobuf-support` dependencies in the approved runtime, run CPU decode/collation tests against real records, and perform the normal clean-SHA/data/GPU/output preflight. No GPU availability claim is made here.

## H200-1 CPU execution status

- Run location: `h200-1`, branch `hezhou-las2-h`, starting SHA `72fdfdb9c51d045ac9c0f4989bc2b7e32aefa260`.
- Runtime: `/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260828/venv`, synchronized from `uv.lock`; Torch 2.7.1+cu126, PyArrow 23.0.0, Lance 10.0.0, MCAP 1.4.0, PyAV 16.0.1, and OpenCV 4.11.0.
- Output root: `/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260828`.
- Stage 2 LIBERO completed: 1,712 records, 34,192 windows, manifest SHA256 `7abd9129b3654dd69cd867d99e1434f2070718334e1909b953e7a4da4af126a2`.
- Stage 2 Hy completed against the H200-1 even-table roots: 57,948 records, 4,478,726 windows, manifest SHA256 `97913f5c98148046e024f1bbebd5eedae7825ddbb85ea62df4f829478947cc83`.
- Stage 2 UMI initially failed before publication on an accepted episode missing `camera_left_wrist_left`. Commit `bc9c326accd28f3d0cec7472434571feaaf4cc0c` rejects incomplete/malformed calibration before opening the MCAP; its local and H200-1 directed tests both pass.
- Corrected UMI inventory was launched in tmux `sttok-h2001-stage2-umi-v2-20260828`; log and timing files are `logs/stage2-umi-v2.log` and `logs/stage2-umi-v2.time` below the output root. The builder had no intermediate throughput output.
- The raw UMI inventory was subsequently stopped without publishing a manifest. Per the updated user decision, mixed-mode pretraining now reuses `/data/shared/datasets/umi_lerobot_v3_260714` and the existing H200-1 episode manifest instead of scanning raw MCAP.

### LeRobot UMI switch and CPU timing completion

- Commit `98e83370517456a880c287dc064b823d08ad8d18` switches the mixed-mode UMI source to the existing `LeRobotStereoDataset`. The formal CLI is now `umi_manifest + umi_dataset_root + umi_rectification_audit_sha256`; the raw-specific root-alias and MCAP episode-cache arguments are not used by the launcher.
- H200-1 passed Bash syntax validation and 30 targeted tests covering LeRobot, mixed scheduling, runtime contracts, and entrypoint wiring.
- Stage 4 fixed manifests contain 32 train episodes each. SHA256: Hy `d798e2dc516c8b5b990b5f9d523a60abc3f29a0d78f5626498003c2fb17591a7`, LIBERO `ab13df45e4151e6ab926c893e43d01024cb1c771f338a01a300fcb178335d937`, UMI `eacd4338ccaa75e6271f85d717e5c96d283bfa159b8cf264a20269d688bcd174`.
- Real decode/preprocess validation passed all six dataset/mode combinations. The available fixed-manifest window counts are Hy 2,628, LIBERO 2,078, and UMI 581.
- Stage 5 measured DataLoader-only batch wait with BS24, 8 workers, prefetch factor 2, pinning disabled, 5 warmup steps, and 40 measured steps per combination. Teachers, model forward/backward, optimizer, and GPU work were excluded.

| dataset/mode | mean step | p50 | p95 | mean samples/s |
|---|---:|---:|---:|---:|
| Hy single | 0.0200 s | 0.0063 s | 0.0916 s | 1198.9 |
| Hy four | 0.0701 s | 0.0238 s | 0.3559 s | 342.6 |
| LIBERO single | 0.0582 s | 0.0102 s | 0.3901 s | 412.6 |
| LIBERO four | 0.1528 s | 0.0329 s | 0.9506 s | 157.1 |
| UMI single | 0.1190 s | 0.0205 s | 0.5396 s | 201.7 |
| UMI four | 0.3804 s | 0.0442 s | 2.0018 s | 63.1 |

The immutable timing result is `/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260828/stage5-dataloader-timing-bs24-w8-40steps.json`, SHA256 `3f624c5e60181fa81285ca719eaadd575edf9488fe7e7ca03cd93c649f209f14`. The low p50 and much larger p95 values reflect prefetched batches followed by periodic worker refill waits; mean and p95 are the useful capacity signals, not the minimum.

## H200-1 real-teacher 40-update run

Status: **blocked after five fresh attempts**; the latest run reached 2/40 real
updates before exposing a Hy episode-boundary data-contract error.

- Code: clean `hezhou-las2-h@48de910de5d949d4b8286ea36591c75860a5002e` on H200-1.
- Output: `/data/home/frank/experiments/stereo-three-source-40u-h2001-v1`; training root is its `train/` child, console log is `run.log`, and the final per-update timing/memory artifact is `step_timings.json`.
- Contract: fresh single-node 8-GPU run, BF16, BS24/GPU, GA1, global batch 192, 40 generator updates, mode weights `35:35:15:15`, mono weights `9:1`, Hy `cam_high` only, existing LeRobot UMI, 8 workers, prefetch factor 2, and GAN/image GAN/video GAN/feature matching all disabled.
- Teachers: real LAS2-H at source SHA `8c97bd4c4da3712c2ac60003a23201dfdb5935f4`, checkpoint SHA256 `758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4`, valid iters 4/max disparity 192; real DA3-BASE at source SHA `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, checkpoint SHA256 `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`, process resolution 504.
- Inputs are the fixed smoke32 Hy/LIBERO/UMI manifests listed above. No raw UMI manifest generation, online GT cache, resume checkpoint, WandB, media logging, or profiler is enabled.
- Startup health check found tmux plus all eight ranks alive, BF16 AMP selected, and no immediate traceback, OOM, NCCL error, or NaN/Inf. The snapshot was still in DDP initialization, before the first update.
- Initial ETA at 23:57 +08:00 was 8--18 minutes for the 40-update training body and 12--25 minutes including validation and post-processing. The run did not reach an update, so this estimate was not validated.

### Startup failures and current blocker

All failed output directories are retained and were never reused.

- v1 (`stereo-three-source-40u-h2001-v1`) exited 137 during Lightning process-group cleanup. Multiple subprocesses entered the metadata writer because the pre-Trainer gate relied only on `RANK`; their distributed environment differed, so immutable `run_manifest.json` rejected the mismatch. This was not a GPU OOM.
- v2 fixed `MASTER_ADDR=127.0.0.1` and `MASTER_PORT=29641`, but still failed metadata comparison because the parent wrote `node_rank=null` while Lightning children supplied node rank zero.
- v3 additionally fixed `NODE_RANK=0`, passed metadata verification and registered all eight NCCL ranks, then exited 137 during cleanup because Lightning set child `LOCAL_RANK` without `LOCAL_WORLD_SIZE`; ranks 1--7 hit `invalid LOCAL_RANK/LOCAL_WORLD_SIZE`.
- v4 additionally fixed `LOCAL_WORLD_SIZE=8`, passed DDP and train-loader construction, then exited 1 before the first update because the fixed smoke32 manifests contain only `train` records while Lightning constructs validation loaders before training.
- v5 used the full manifests (Hy 56,797 train / 1,151 val; LIBERO 1,684 / 28; existing LeRobot UMI 56,363 / 3,131), passed all preceding gates, and completed two real GAN-off BF16 generator updates with LAS2-H/DA3 before failing while fetching the third update. Rank 4 worker 2 raised `ValueError: Hy Lance window crosses an episode boundary` from `HyLanceMonoDataset.get_mode_item`.

v5 output is `/data/home/frank/experiments/stereo-three-source-40u-h2001-v5`, console log is `run.log`, and `exit_code.txt` is `1`. No checkpoint or final `step_timings.json` was produced because training stopped at update 2/40. GPUs returned to 0 MiB after cleanup. No v6 was launched: the Hy manifest/reader episode-boundary disagreement is now the first unresolved data-contract blocker and requires a targeted diagnosis before another GPU attempt.

### Hy boundary diagnosis and local repair

The CPU schedule audit reproduced the first bad sample without decoding images:
rank 0, update index 2, `mono/four_frame`, manifest sample 1,422,200,
episode `table_014:7123`, relative frames `[2280,2283,2286,2289]`. Adding the
manifest `dataset_from_index=6691507` and using the result as Lance physical row
positions returned episode 7147 frames `[506,509,512,515]`. Querying the same
Lance table by the immutable identities `episode_index=7123` and requested
`frame_index` values returned the four exact target rows. The root cause is
therefore that Hy metadata's global dataset offsets are not Lance physical row
positions; the window itself does not cross an episode boundary.

The pending local repair changes Hy reads to predicate on episode/frame identity,
reject missing or duplicate identities, and restore requested temporal order. It
also makes pre-Trainer metadata writing reject nonzero Lightning `LOCAL_RANK`
when global `RANK` is absent, and lets single-node Lightning DDP infer the local
world size from configured devices when `LOCAL_WORLD_SIZE` is absent. Directed
regression tests cover all three failures. This repair has only static local
validation so far; it must be committed, pushed, fast-forwarded to H200-1, and
run through the targeted CPU tests before any v6 GPU launch.
