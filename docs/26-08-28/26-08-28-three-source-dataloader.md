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
