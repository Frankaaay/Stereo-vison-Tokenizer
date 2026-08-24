# LeRobot Online FoundationStereo Teacher Implementation

## Status

- Date: 2026-08-24
- Branch: `frank-profiling`
- Starting commit: `d63335a31ccf8c405a2a4956da59d70441fad652`
- Implementation location: local Windows worktree only
- Current status: code and contract preparation; no test, manifest generation,
  server synchronization, GPU experiment, training launch, output, or checkpoint
  has been executed.

## Accepted contract

- Dataset scope is the H200-1-visible LeRobot-v3 tree only. It must not be
  described as a deduplicated two-node dataset.
- Manifest and split granularity are episode-level.
- Shuffle granularity is shard then episode; samples inside an episode are read
  in time order.
- Model input granularity is one four-frame sample.
- At 30 Hz, sample starts use stride 12 and the four frame offsets are
  `[0, 3, 6, 9]`.
- The split is deterministic 90/5/5 with seed 1234. Test is not used for
  training decisions.
- Validation uses one deterministic window from each of 512 distinct validation
  episodes every 500 optimizer updates. The test split has a separate loader and
  is not used for training decisions.
- Online GT cache support is optional and defaults to disabled.
- FoundationStereo remains bidirectional and retains the LR-consistency mask.
  The formal iteration count is not frozen until the 32/16/12 comparison is
  run and reviewed.
- The user explicitly removed a code-level ten-hour graceful-stop change from
  this implementation scope. Existing `max_steps` and checkpoint behavior are
  preserved.

## Rectification and calibration boundary

FoundationStereo does not consume camera calibration as a network input, but
the adapter must read every episode's `K/D/R/P` and must not infer from the
presence of calibration alone that MP4 frames are already rectified.

The implementation supports exactly two manifest-bound modes:

- `verified_pre_rectified`: an epipolar audit has verified the encoded MP4 pair;
- `apply_calibration`: the adapter applies OpenCV rectification maps generated
  from the episode's `K/D/R/P` before resize and padding.

Manifest construction requires a passing rectification-audit artifact and
stores its SHA256. The Dataset refuses an unverified mode or an audit-hash
mismatch. Output focal length uses `P_left[0,0] * 0.4`; baseline remains
`-P_right[0,3] / P_right[0,0]`.

## Implemented code paths

- `stereo_tokenizer/lerobot_data.py`
  - episode manifest validation;
  - MP4 timestamp decoding;
  - optional calibration rectification;
  - `640x480 -> 256x192 -> 256x256` letterbox;
  - output calibration scaling;
  - shard/episode shuffle and episode-time-order distributed sampler.
- `scripts/data/audit_lerobot_stereo_rectification.py`
  - representative raw-versus-remapped epipolar residual audit;
  - raw/remapped epipolar-line review images;
  - fail-closed selection of `verified_pre_rectified` or `apply_calibration`.
- `scripts/data/build_lerobot_stereo_manifest.py`
  - source failure exclusion and source-prefix remapping;
  - calibration and video-interval binding;
  - duplicate episode rejection and normalized task lists;
  - deterministic episode split and contract/manifest hashes;
  - refusal to overwrite outputs.
- `stereo_tokenizer/online_gt.py`
  - strict offline ViT-L checkpoint loading;
  - one frozen bidirectional teacher per DDP rank;
  - 12/16/32 iteration modes;
  - LR consistency and the existing final validity thresholds;
  - optional non-overwriting write-through cache, disabled by default.
- `train_stereo_vae.py`, `stereo_tokenizer/data.py`, and
  `scripts/stereo/train_stereo_vae.sh`
  - explicit `manifest_v3` versus `lerobot_online` backend;
  - online-teacher configuration, representative fixed validation subset, and
    a separate untouched test loader;
  - Lightning distributed-sampler replacement disabled because the data module
    already assigns complete shards/episodes to ranks;
  - existing StereoVAE core loss remains unchanged.
- `scripts/data/build_lerobot_teacher_selection.py`
  - deterministic 512-episode candidate selection;
  - deterministic left-camera contact sheets covering all three views and four
    frames for manual tagging;
  - comparison is blocked until required visual coverage tags are reviewed and
    the selection artifact is explicitly marked approved.
- `scripts/stereo/compare_online_foundation_teacher.py`
  - fixed 32 -> 16 -> 12 comparison order on identical samples;
  - one untimed model warm-up per configuration;
  - decode and teacher timing, throughput, pair latency, and peak memory;
  - finite/valid ratios, LR residual, difference quantiles, mask IoU, temporal
    consistency, and fixed visualizations;
  - disparity differences measured on the 32-iteration valid region, plus mask
    IoU and per-view valid-ratio deltas;
  - numeric 16-versus-32 project gate plus mandatory valid-ratio and visual
    reviews.

## Planned validation and experiments

No item below has run yet. Each needs user approval before execution.

1. Local contract/unit tests and source-boundary regression tests. Static AST
   parsing and `git diff --check` have run; executable tests have not.
2. H200-1 read-only runtime/dependency preflight for PyAV, PyArrow, timm, OmegaConf,
   OpenCV, the local DINOv2 tree, and strict FoundationStereo checkpoint load.
3. Rectification audit on 96 deterministic episodes plus raw/remapped epipolar
   images. Before running, approve the current project thresholds: at least 90%
   successful pairs per view, at least 20 ORB matches per accepted pair, at
   least 1000 total matches per view, and vertical-residual P95 <= 1.5 px.
4. Episode manifest/summary generation and complete count/hash/split-isolation
   audit. This produces the H200-1-only 90/5/5 episode split.
5. Decoder/sample-contract smoke: timestamp tolerance, six stream mapping,
   `[0,3,6,9]` frames, stride 12, rectification mode, image letterbox, `fx`, and
   baseline.
6. 512-sample candidate generation, contact-sheet review, visual tagging, and
   selection freeze.
7. Eight-H200 32/16/12 teacher comparison with cache disabled, fixed sample and
   batch order, the specified quality gate, and manual valid-ratio/visual review.
8. One-GPU functional/memory smoke for online teacher plus StereoVAE, with cache
   disabled.
9. Eight-GPU BS/gradient-accumulation comparison at fixed global batch 192. The
   initial grid is BS4/GA6, BS8/GA3, and BS12/GA2; it does not assume the old
   offline-GT BS24 remains safe.
10. Eight-GPU short end-to-end run measuring MP4 decode, online teacher, VAE,
   backward, optimizer, memory, loss, and validation health.
11. Only after the code/data gates pass: local commit/push and H200-1
    fast-forward-only sync to an exact clean SHA.
12. Separately approved long training launch. No code-level ten-hour graceful
    stop will be added; the resolved launch must use the retained `max_steps`
    contract or another separately reviewed operational boundary.

## CPU-only execution status (2026-08-24)

- Code commit: `0ee70f14b452438dd2db7d51cd149aefae1976ac` on
  `frank-profiling`; pushed to `origin/frank-profiling` and fast-forwarded to a
  clean H200-1 clone. H200-2 was not touched.
- Local source regression: 11/11 passed with CUDA hidden. The local Python lacks
  Torch, so dynamic contract tests were moved to the H200 CPU runtime rather
  than installing local dependencies.
- H200-1 CPU regression: 18/18 passed with `CUDA_VISIBLE_DEVICES=-1`;
  `torch.cuda.device_count()==0` and `torch.cuda.is_initialized()==False`.
- H200-1 GPUs 0-7 are occupied by `wuhao98` LIBERO evaluation processes. H200-2
  GPUs 0-7 were idle in the latest read-only snapshot. No GPU process was
  modified and no FoundationStereo forward or training experiment was started.
- H200-1 dataset snapshot: node-local LeRobot root is 1.4 TiB with 1983 shard
  directories; `/data` has 7.4 TiB available.
- Unified CPU runtime: Python 3.12, Torch 2.7.1+cu128, Lightning 2.5.6, PyAV
  16.0.1, PyArrow 23.0.0, OpenCV 4.10.0, timm 1.0.28, OmegaConf 2.3.0.
- FoundationStereo checkpoint: 3.1 GiB, SHA256
  `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`.
  CPU payload inspection found 1360 model tensors, all on CPU, at
  `global_step=200000`, `epoch=40`.
- The unified Python 3.12 runtime on both H200 nodes was supplemented with
  `trimesh==5.0.0`, `joblib==1.5.2`, and the FoundationStereo-runtime-matched
  `open3d==0.19.0`. Imports of the first two were verified with CUDA hidden;
  both checks reported zero visible CUDA devices and an uninitialized CUDA
  runtime. `pip check` still reports the pre-existing unsupported `decord 0.6.0`
  package. CPU strict model construction now reaches the unconditional Open3D
  import in FoundationStereo `Utils.py`, then stops because Open3D's optional
  visualization dependency `plotly` is absent. A dry-run showed that completing
  Open3D's dependency closure would add about 30 packages including Dash,
  IPython/Jupyter components, and a scikit-learn binary wheel; this unrelated UI
  stack was not added to the unified training runtime. No model forward ran.
- Rectification audit completed in tmux `stereo-lerobot-rectify-260824`, output
  root `/data/home/frank/experiments/stereo_lerobot_cpu_20260824_approval1`.
  The schema-valid `rectification_audit.json` result is **fail** with no selected
  mode: 96 candidate episodes, 267 representative pairs, and 21 failed pair
  probes. Raw P50/P95 vertical residuals were head 1.728/20.902 px, left hand
  1.200/20.736 px, and right hand 1.200/40.320 px. Applying calibration was not
  better: 4.800/27.360, 2.986/24.883, and 2.986/40.248 px respectively.
- Visual inspection of the three fixed `*_000_*` audit panels shows that raw
  stereo content is already approximately aligned to horizontal epipolar lines;
  calibration remap introduces crop/black borders without a visible alignment
  gain. This is evidence that the encoded videos may already be rectified, but
  the current Lowe-ratio-only ORB residual distribution contains too many
  outliers to freeze `verified_pre_rectified`. A symmetric/RANSAC-inlier audit
  is required before selecting the manifest rectification mode.
- The user authorized a temporary `verified_pre_rectified` assumption while the
  data team confirms upstream processing. The manifest builder now requires the
  explicit `--allow-provisional-pre-rectified` flag for this path and preserves
  `status=provisional_user_assumption`, the failed source-audit result, and its
  SHA256 in every record and in the contract. Without the flag, the original
  fail-closed behavior is unchanged; the failed audit is never rewritten as a
  pass.
- Local validation for the provisional path: Python compilation passed,
  `git diff --check` passed, and the CPU-hidden entrypoint/source-boundary suite
  passed 23/23. Manifest generation, real-data decoder smoke, and 512-sample
  selection are next and have not yet started.
- GPU work on an idle H200-2 was subsequently authorized. FoundationStereo
  forward remains gated on the manifest/decoder/selection checks and a fresh
  GPU ownership snapshot immediately before launch.
