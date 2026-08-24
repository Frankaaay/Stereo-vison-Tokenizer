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
  passed 23/23. Commit `e36756a573aaf19a3b676e2776c8aece645d29e1`
  was pushed and both H200 clones were cleanly fast-forwarded to that exact SHA.
  H200-1 passed all 31 targeted tests. H200-2 passed the eight directly relevant
  online-contract tests; its combined 31-test run had one unrelated failure
  because 28 ignored legacy `OmniTokenizer/__pycache__/*.pyc` files remain. They
  were not deleted.
- H200-1-only provisional manifest generation ran from approximately 13:27 to
  13:37 +08:00 in tmux `stereo-lerobot-manifest-260824`. Inputs were 1983
  `shard_*` directories and 63,453 source-manifest rows. Outputs are
  `h200_1_provisional_manifest_v1.jsonl`,
  `h200_1_provisional_manifest_v1_summary.json`, and `manifest_build.log` under
  `/data/home/frank/experiments/stereo_lerobot_cpu_20260824_approval1`.
  It contains 62,626 unique episodes and 1,384,393 four-frame samples; split
  episode counts are 56,363/3,131/3,132 and sample counts are
  1,246,294/68,932/69,167 for train/val/test. Independent streaming validation
  reproduced all counts and SHA256
  `31457d9b1834953024d7e7ff59f5a21b74500d3ece4c19c755a14aff3dccaf6d`.
  Contract SHA256 is
  `6144b1b0c4690ad374491209f5f20c306184703986119e521177fd3642ed77f3`.
- The real six-stream decoder/data-contract smoke passed on two consecutive
  samples. It verified six MP4 streams per sample, offsets `[0,3,6,9]`, stride
  12, output shape `[3,2,3,4,256,256]`, letterbox padding, finite scaled `fx`,
  and positive unscaled baselines of 54.997-55.270 mm. CPU decode measured
  0.137 seconds per four-frame sample. Result:
  `decoder_contract_smoke_v1.json` in the same experiment root.
- The first fixed 512-sample selection/contact-sheet job ran from approximately
  13:36 to 13:53 +08:00 in tmux `stereo-teacher-selection-260824`; log is
  `teacher_selection.log`. It produced `teacher_selection_512_v1.json` and 32
  PNG sheets under `teacher_selection_visuals_v1`. Its review status remains
  pending because the visual coverage tags have not been inspected.
- The data team subsequently confirmed that the encoded stereo videos are
  already rectified. This supersedes the provisional uncertainty: production
  preprocessing must use `verified_pre_rectified` and must not apply OpenCV
  calibration remap. The failed ORB audit remains useful only as evidence that
  its Lowe-ratio-only P95 gate was not robust enough; it is no longer a data
  readiness blocker.
- Selection generation completed with 512 unique episodes and 32 contact
  sheets, but its review remains pending and all seven required visual coverage
  counts are zero. The eight-GPU comparison correctly refuses this state.
- Latest GPU readiness snapshot: H200-2 has eight idle H200 GPUs and no compute
  processes, while all H200-1 GPUs are occupied by `maxliu` processes. H200-2
  has only shards 0-1551; 68 of the current H200-1 selection's 512 samples come
  from shards 1552-1978 and are therefore unavailable there.
  Formal GPU comparison is blocked until the sampling scope is explicitly
  changed to an H200-2-resident subset (or the missing data is synchronized),
  the exact checkpoint is synchronized and hash-verified, and visual coverage
  review is approved.
- The user approved restarting selection against the H200-2-resident scope.
  The new frozen scope is H1-manifest train episodes in shards 0-1551, with the
  same seed, task balancing, and one-sample-per-episode rule; it must be labeled
  as an H200-2-resident subset. A `--maximum-shard-index` selection filter and
  its unit test were added locally; Python compilation, `git diff --check`, and
  23/23 source tests pass.
- Checkpoint rsync from the H200-1 node-local artifact to the identical H200-2
  path completed through the jump host: 3,298,527,334 bytes in approximately
  2 minutes 56 seconds. H200-2 local SHA256 is
  `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`,
  exactly matching the frozen reference. No GPU process has been started.
- GPU work on an idle H200-2 was subsequently authorized. FoundationStereo
  forward remains gated on the manifest/decoder/selection checks and a fresh
  GPU ownership snapshot immediately before launch.
- Resident-shard selection implementation commit
  `cc128f13721cf2e0436e16ea60e2f81186eaf4be` was pushed and both H200 clones
  were cleanly fast-forwarded to it. The directly relevant contract suite passed
  9/9 independently on H200-1 and H200-2.
- The replacement H200-2-resident 512-sample selection started at approximately
  2026-08-24 14:00 +08:00 in tmux
  `stereo-teacher-selection-h2-260824`, using `--maximum-shard-index 1551`.
  Output/log paths are `teacher_selection_512_h200_2_resident_v1.json`,
  `teacher_selection_visuals_h200_2_resident_v1`, and
  `teacher_selection_h200_2_resident.log` under the H200-1 experiment root.
  Startup health found the expected CPU process and no immediate traceback.
  ETA is 15-20 minutes based on the first 512-sample contact-sheet run's actual
  17-minute duration. GPU forward remains not started.
- GPU preflight found that H200-2 lacked the complete FoundationStereo repo,
  local DINOv2 tree, and checkpoint-adjacent `cfg.yaml`. The H200-1 source was
  verified clean at `master@6e8806816b533e4d13ddbb95ffa907b797060a62`;
  `cfg.yaml` SHA256 is
  `a9d9dd2137c30edc2236194f62df14d222dad5fd3287a33c7540b543bb93853f`.
  Rsync of these assets to H200-2 is in progress through the jump host.
- FoundationStereo imports Open3D unconditionally from `Utils.py`, even though
  online disparity inference does not use its point-cloud helpers. The project
  loader now installs an import-only Open3D stub while importing FoundationStereo
  and restores the original module immediately afterward. Any actual Open3D API
  access raises an explicit runtime error. This avoids adding the unrelated
  Dash/Jupyter/scikit-learn dependency closure to the training environment.
  Local Python compilation, `git diff --check`, and 23/23 source tests pass;
  server strict construction remains pending asset sync and code deployment.
- H200-2 strict CPU construction with CUDA hidden confirmed that the Open3D
  blocker is removed and local DINOv2 constructs successfully. It then stopped
  at PyTorch 2.6+'s `torch.load(weights_only=True)` default because the verified
  FoundationStereo training payload contains NumPy scalar metadata. The loader
  now explicitly uses `weights_only=False` only after the frozen checkpoint
  SHA256 has matched, and still applies `load_state_dict(strict=True)`.
- Compatibility commit `e2b0c2d621b35ec5e3515627ddd31c08c0dca360`
  was pushed and both H200 clones were cleanly fast-forwarded to it. H200-2
  strict CPU construction then passed completely: FoundationStereo, offline
  DINOv2, checkpoint payload, and `load_state_dict(strict=True)` all succeeded
  with zero visible CUDA devices and an uninitialized CUDA runtime. xFormers is
  absent, so DINOv2 reports its standard attention/MLP fallback.
- A single-GPU functional smoke on H200-2 GPU0 passed for one four-frame sample,
  bidirectional inference, 16 iterations, and pair microbatch 1. Forward-only
  time was 3.090 seconds/sample, finite disparity ratio 1.0, LR-valid ratio
  0.6684, and peak allocated/reserved memory 1.875/2.047 GB. Dataset decode was
  0.163 seconds. Result:
  `/data/home/frank/experiments/stereo_lerobot_gpu_20260824_h2/teacher_single_sample_smoke_v1.json`.
  This is a functionality gate, not the formal throughput result.
- The H200-2-resident selection completed with 512 unique episodes/indices,
  eligible shard range 7-1551, 44,006 eligible train episodes, and manifest SHA
  `31457d9b1834953024d7e7ff59f5a21b74500d3ece4c19c755a14aff3dccaf6d`.
  Selection SHA is
  `96e2461d98c3952d703e5680a94e74277f2033eb33aa7f63a70fccd801d9b0a0`.
- The full H200-2 512-sample/six-stream CPU decode audit started at
  approximately 2026-08-24 14:21 +08:00 in tmux
  `stereo-h2-decode-audit-260824`, with retry log
  `h200_2_resident_decode_audit_retry1.log`. The first launch did not decode any
  sample because the selection JSON had not landed; that failed log is retained.
  The explicit rsync retry transferred 155,300 bytes and the second launch is
  healthy at about 258% CPU, 1.28 GB RSS, zero GPU memory, and no immediate
  traceback. ETA is 15-25 minutes based on the two contact-sheet decode runs.
- That H1-manifest-on-H2 audit completed in 64.9 seconds and failed: 413/512
  samples decoded, 90 lacked a decodable frame near the requested timestamp,
  and nine referenced a missing MP4. Failures covered 87 shards from
  `shard_0045` through `shard_1544` and first surfaced on `head_left` because
  dataset decoding stops at the first stream error. This proves that shard-name
  overlap does not make the two node-local datasets interchangeable.
- Per user direction, H1 manifest reuse was abandoned. Direct H200-2-local
  manifest generation started at approximately 2026-08-24 14:34 +08:00 in tmux
  `stereo-h2-local-manifest-260824`, using H200-2's own 1552 shards, 49,564
  source rows, failures, Parquet metadata, videos, source calibration, and
  confirmed-pre-rectified mode. Outputs/log are `h200_2_local_manifest_v1.jsonl`,
  `h200_2_local_manifest_v1_summary.json`, and
  `h200_2_local_manifest_build.log` under the H200-2 experiment root. Startup
  health was clean at about 165% CPU, 1.13 GB RSS, and zero GPU memory. ETA is
  7-12 minutes, scaled from the H200-1 1,983-shard manifest's actual duration.
  H200-2-local selection will follow immediately; no redundant full decode audit
  will be run because contact-sheet generation itself decodes all 512 samples.
- The user then prioritized immediate quantitative speed measurement over
  waiting for the H200-2-local full manifest. The local-manifest tmux was stopped
  to avoid CPU/I/O contention; no GPU work was running. From the prior decode
  audit's 413 verified-complete samples, the first 408 were frozen so eight ranks
  receive exactly 51 samples each. Selection SHA256 is
  `e7165387e9c40583c9c00e3f27eb7d05d3ee96e0f820347dfee8614f16736306`.
  The comparison accepts an explicit `--allow-pending-visual-review` mode while
  preserving `review_status=pending_quantitative_only`; it does not fabricate
  visual coverage approval. This run is for throughput and quantitative
  32/16/12 comparison only.
- The approved 408-sample eight-GPU comparison started on H200-2 at
  approximately 2026-08-24 14:47 +08:00 from clean commit `a7214d7`, in tmux
  `fs-teacher-408-8gpu-260824`. It uses 51 verified-decodable samples per rank,
  bidirectional LR consistency, valid iterations 32/16/12, sample batch size 8,
  and pair microbatch 48. Output and log are
  `/data/home/frank/experiments/foundation_teacher_compare_h200_2_verified_408_20260824_v1`
  and the same path with `.log`. The initial `torchrun --standalone` attempt
  spawned no workers because local rendezvous hostname resolution stalled; its
  own tmux was stopped before CUDA initialization or output creation, then the
  same experiment was restarted with explicit single-node rendezvous at
  `127.0.0.1:29651`. Startup health passed: all eight workers exist, ranks map
  to GPU0-7, each GPU holds about 1.5 GiB during model initialization, and no
  traceback is present. Initial ETA at 14:48 +08:00 is 5-15 minutes for all
  three configurations, based on the 3.09-second unbatched single-sample smoke
  and the expected gain from eight GPUs plus pair batching; result aggregation
  and artifact validation should finish within roughly 2 additional minutes.
