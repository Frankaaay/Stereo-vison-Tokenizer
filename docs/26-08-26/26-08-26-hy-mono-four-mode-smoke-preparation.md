# Hy cam_high mono cache 与四模式 smoke 接线

## 状态与边界

- 日期：2026-08-26
- 本地分支：`merged-fs-vae-single-four-profiling`
- 基线 HEAD：`33cce26eb08756350a51aa6cf52102c39065497b`
- 本地状态：未提交；未 push；H200 源码未修改或同步。
- 已执行的远端写操作仅为用户授权的 H200-1/H200-2 node-local raw RGB smoke cache，
  以及 H200-2 Frank-owned task-private Lance export runtime。
- 未启动 GPU、DA3 teacher、FoundationStereo teacher、训练或评估。

## 2026-08-26 generic geometry 接管更新

本轮在同一未提交 worktree 上继续，分支仍为
`merged-fs-vae-single-four-profiling`，基线 HEAD 仍为
`33cce26eb08756350a51aa6cf52102c39065497b`。未 commit、未 push、未同步
H200 Git clone，也未启动任何 GPU forward 或训练测试。

新增共享 `stereo_tokenizer/geometry.py`，由一份 rectified source geometry
同时推导：

- Student aspect-preserving resize、letterbox、non-padding mask；
- DA3 `upper_bound_resize` 的 longest-side resize、nearest-multiple-of-14
  二次对齐与 ImageNet normalization；
- 严格、可 collate 的 geometry metadata；
- DA3 processed output 直接映射到 Student resized grid 后再 letterbox。

已核对冻结 DA3 source
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` 的
`InputProcessor`：放大使用 OpenCV cubic、缩小使用 area；两维都以 tie-up 规则
取最近 14 倍数。共享实现复现该 CPU preprocessing，不 import DA3 model，worker
只生成 `da3_images`。DataLoader mono item 现使用
`video / da3_images / non_padding_mask / geometry_mapping`；callback 只执行 teacher
forward、严格核对 homogeneous batch geometry，并把 processed-grid depth/confidence
映射到 Student grid。

已推导并写入定向测试的两个合同：

```text
rectified 480x640:
  Student 192x256, padding LTRB=[0,32,0,32]
  DA3 378x504

Hy 240x424:
  Student 145x256, padding LTRB=[0,55,0,56]
  DA3 280x504
```

DA3 native cache schema 更新为 `da3-processed-relative-depth-cache-v2`，metadata
包含完整 geometry mapping 和动态 tensor shape；contract 或 geometry 不一致时严格
失败，不会读取或覆盖旧格式 cache。centered target 仍不落盘。

本地验证：

- `python -m py_compile ...`：通过；
- `python tests/stereo/test_entrypoints_source.py`：15/15；
- `git diff --check`：通过，仅有既存 CRLF notices；
- 本地 Python 实时缺少 Torch 和 Lightning；提交并同步两台 H200 后，已在 H200-2
  Frank unified runtime 的 CPU 模式执行完整 `tests/stereo`，最终 115/115 通过。

## H200-2 Lance runtime、schema 与 immutable cache

只读 preflight 与交接无实质差异：H200-2 present tables 仍为
`table_001,003,005,007,009,011`，`tables.json` SHA256 仍为
`7ea2d524ecca41161348dddcdfe38ceade422a75b3de91d6a512ecfffbcf941f`；
目标 runtime 与 output root 在操作前均不存在。`/data` 当时剩余约 1.8 TiB，使用率
97%。

按已批准边界创建 Frank-owned task-private CPU runtime：

```text
/data/home/frank/runtime/hy-lance-export-v1
Python 3.12
pylance 8.0.0
pyarrow 23.0.0
numpy 2.5.2
Pillow 12.3.0
```

未修改 `/data/home/frank/runtime/stereo-tokenizer-unified-v1`。inspector/exporter 均将
当前本地脚本按 LF-normalized stdin 传给远端 Python，没有编辑 H200 source clone。

`table_001` schema audit：

- Lance row count：`10,788,879`；episode metadata row count：`11,594`；
- 字段为 `episode_index:int32`、`frame_index:int32`、`index:int64`、
  `timestamp:float`、`task_index:int32`、三个 image binary、
  `observation_state:fixed_size_list<float>[16]`、`action:fixed_size_list<float>[2]`；
- probe episode 0：1450 rows，frame/timestamp 严格递增，
  `timestamp - frame_index/30` 范围约
  `[-1.7801921e-6,+1.7801921e-6]` 秒；
- JPEG 字段只输出 byte length 和 SHA256，没有输出 payload。

H200-2 node-local immutable cache：

```text
output root: /data/shared/datasets/hy_mono_cam_high_smoke_v1
present tables: table_001,003,005,007,009,011
samples / distinct episodes: 48 / 48
NPZ / total files / manifest lines: 48 / 50 / 48
RGB payload bytes: 37986442
directory du -sb: 38036148
manifest SHA256: 5f69331a4afda18590fe67d6da41aee328314326f0c645fc5d0a4beec6d6b3db
summary SHA256: f3b620c3d24a0f7c1a09ff05e393114ae95558bd3fddbe7599478058481ddaa7
table inventory SHA256: 454c454bd5b7df4c264a6f69f900b7075fe1d097cc33480d012a3811379b8458
source contract SHA256: 46ed93bb21e1c4dd40f279a9de3e742f9b27502252c1297f208995664121b000
skip: 1 candidate with timestamp inconsistent with frame_index / FPS
owner: frank:ai-users
```

相同 exporter 以相同 seed/参数重跑 exit 0；manifest、inventory、source-contract、
sample/episode count、RGB bytes 和 skip summary 全部一致，证明 immutable rerun
幂等。

## 后续测试 readiness 与授权门

已完成准备：generic geometry 代码与十项定向测试、静态/source 检查、H200-2
task-private Lance runtime、schema audit、48-window raw RGB cache 与幂等验证。

已完成：十项 geometry/cache tests 与完整 CPU tensor suite；两台 H200 都已通过
fast-forward-only 同步到精确提交。尚缺：两 rank schedule/resume execution、BS24
四模式 GPU residency smoke、small-data overfit 与 strict resume。

## 双节点只读证据

数据根：
`/data/shared/datasets/hy_lance_overlay_20260720_12tables_v1`

H200-1：

- present tables：`table_000,002,004,006,008,010`
- `tables.json` SHA256：
  `ebb32ce296eb1e4f15eff14a5864f256315ce419f6ebb684d3128970fc2c077e`
- 可用 Lance 合同：将旧 Hy runtime 的 Python 3.12 site-packages 通过
  `PYTHONPATH` 交给 Frank 的统一 Python；`lance==8.0.0`、`pyarrow==23.0.0`。

H200-2：

- present tables：`table_001,003,005,007,009,011`
- `tables.json` SHA256：
  `7ea2d524ecca41161348dddcdfe38ceade422a75b3de91d6a512ecfffbcf941f`
- 旧 Hy runtime 的解释器链接指向不存在的
  `/data/cache/conda/envs/maxliu/fastwam/bin/python`；统一 Frank runtime 本身没有
  Lance/PyArrow。把旧 `site-packages` 注入统一 Python 会异常断开，不能作为可用
  ABI 合同。本轮已改用上文 Frank-owned task-private runtime 完成 row audit 与 cache
  导出；旧 runtime 仍不作为 ABI 合同。
- 发现其他用户环境中有 Lance，但没有借用该环境生成 Frank 的数据产物。

H200-1 的真实 Lance row schema：

```text
episode_index                         int32
frame_index                           int32
index                                 int64
timestamp                             float
task_index                            int32
observation_images_cam_high           binary
observation_images_cam_left_wrist     binary
observation_images_cam_right_wrist    binary
observation_state                     fixed_size_list<float>[16]
action                                fixed_size_list<float>[2]
```

JPEG bytes 只记录长度和 SHA，没有输出 payload。episode metadata schema 为
`episode_index,length,dataset_from_index,dataset_to_index`。探测的 episode 0 共
1566 帧，frame index 与 timestamp 均严格递增；timestamp 相对
`frame_index / 30` 的误差范围约为 `[-1.8e-6,+1.8e-6]` 秒。

固定 teacher 资产在两端重新计算的 SHA 一致：

- DA3 source：`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- DA3-BASE revision：`f4a6c9b3c95e41c82048423d3493a81ec3fa810e`
- `model.safetensors` SHA256：
  `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`
- FoundationStereo `model_best_bp2.pth` SHA256：
  `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`

## H200-1 mono cache

路径：`/data/shared/datasets/hy_mono_cam_high_smoke_v1`

- manifest：48 条；48 个不同 `(table,episode)`；seed 1234；无跳过。
- frame offsets：`[0,3,6,9]`；四帧同 episode；30 FPS timestamp 已核对。
- 48 个 NPZ 中的 JPEG 均解码为 RGB 240x424，再存为 raw uint8
  `[4,3,240,424]`；没有 letterbox 或 teacher target。
- 文件数：50（48 NPZ、manifest、summary）。
- NPZ 总字节数：37,823,902；目录 `du -sb`：37,873,540。
- manifest SHA256：
  `b265c08f5d60a3822ab43b8b025dfdee6a6c95d91384e149b61be08ac601ab6c`
- summary SHA256：
  `de8cdfc3456c0afc587a024a4d2fae005a07144f69c9d63acaec5ea4eb7c03ae`
- table inventory SHA256：
  `ca5551d2092a2a02fbffdb4c6515712bbf6cd3bc3bb5813b4a10b0713683b610`
- source contract SHA256：
  `e09d0c7474c472a0e70f0f0115587a52f9c7294683eddded505e69f29457c7d0`

相同命令重复执行后逐字节接受已有产物，manifest/SHA 不变；不一致内容会拒绝覆盖。

## 四模式实现合同

- `HyMonoSmokeDataset` 校验 manifest/cache SHA、帧号、timestamp、metadata；运行时将
  manifest 声明的 rectified HxW 交给共享 geometry API；Hy 240x424 自动得到
  145x256 与 top/bottom padding 55/56。
- `mono/single_frame` 从 `rgb.npy` 的压缩流只解出 frame 0；不会先完整解压四帧。
  worker 产出 DA3 `[B,1,3,H_da3,W_da3]` processed tensor。
- `mono/four_frame` 完整读取四帧；worker 产出
  `[B,4,3,H_da3,W_da3]`，teacher 一次 forward。
- mono student 输入严格为 `[B,1,1,3,T,256,256]`，encoder 跳过 StereoFusion。
- worker 侧复现 DA3 官方 `upper_bound_resize`、process resolution 504、14 倍数对齐
  与 ImageNet normalize；DA3 model 不在 worker import/forward。
- DA3 depth/confidence 从 processed grid 直接 resize 到 Student resized grid，再使用
  同一 padding；不再经由 hard-coded raw 240x424 中间网格。
- 第一轮 mask 固定为 `finite(depth) & depth>0 & non_padding`；confidence 只缓存和
  记录均值，不设阈值。DA3 native cache 保存 processed-grid depth/confidence、完整
  geometry 与 provenance，不保存 centered target。
- 固定 48 mono + 48 stereo 的 smoke 最多支持 2 个 DDP rank；两 rank 各得到不重叠
  的 24 samples。8 updates 正好是两个 seeded super-cycle，每 mode 两次。
- checkpoint 保存 seed/counters；resume 前校验 counters 与 seeded prefix，DataLoader
  从 `generator_updates` 对应的下一 mode 开始。
- validation 每种 mode 各算 `core + LPIPS`，best checkpoint 监控四种 mode sample
  mean 的等权平均 `val/mixed/total_loss`。

## 小数据过拟合实验设计（尚未启动）

实验走与正式训练相同的 dataset、online teacher、mixed sampler、loss、optimizer、
checkpoint/resume 路径，只把固定 source 缩小到 48+48。推荐分两步：

1. 接口 smoke：单节点 1 或 2 GPU，BS24、GA1、8 updates、每 mode 2 次，验证 shape、
   finite loss、mode 次序、分 rank sample、峰值显存和 checkpoint 下一 mode。
2. 小数据过拟合：仍用固定 48+48，`MODE_UPDATES_PER_EPOCH=MAX_STEPS` 且总 step 为 4
   的倍数；观察四种 mode 各自 RGB/relative-depth/gradient/KL/LPIPS 曲线，以及等权
   validation mean。随机性来自固定 subset 内的 seeded sample permutation，不在每次
   resume 时重新抽 subset。

GPU 启动前仍需实时复核 GPU/进程、node-local input、frozen assets 与 output 不存在，
并按用户已确认的下述 smoke 合同执行。

### 建议的精确 smoke / resume / overfit 合同（仅提案，未获启动授权）

共同冻结参数：H200-2 node-local LeRobot manifest
`/data/home/frank/experiments/stereo_lerobot_cpu_20260824_approval1/h200_2_local_manifest_v1.jsonl`
（启动前重验 SHA256
`96024f091bcf7aca844b4d4b99fad2eb6cb0f420aa693f1431340b79ac5fa53e`），dataset root
`/data/shared/datasets/umi_lerobot_v3_260714`，rectification audit SHA256
`41d2bfecaae85dd18f7cfd1a2a3a2177e8fd4aa8897be1cb411d85c3092a7d25`，H200-2 mono
manifest/cache 使用本节上方的新 cache。两个 teacher 使用本文件记录的 frozen source、
checkpoint 与 SHA。统一使用 BF16、BS24/device、GA1、ratio 1:1:1:1、seed 1234、
source index 0、FS PyTorch valid_iters 32、pair microbatch 48、DA3 process resolution
504、`finite_positive_non_padding`，以及 `lr=lr_min=1e-4`、warmup 20、KL warmup
100、RGB/relative-depth/relative-gradient/KL/LPIPS 权重
`1.0/1.0/0.1/1e-6/1.0`，GAN weights 维持 0。

1. 两 rank 接口与 checkpoint smoke：选择实时空闲且同构的 2 张 H200，
   `GPU_COUNT=2, GLOBAL_BATCH_SIZE=48, MAX_STEPS=4,
   MODE_UPDATES_PER_EPOCH=8, CHECKPOINT_EVERY_N_STEPS=4,
   ONLINE_VAL_CHECK_INTERVAL_STEPS=4, ONLINE_GT_CACHE_ENABLED=0`，output 提议为
   `/data/home/frank/experiments/stereo_hy_four_mode_smoke4_h2002_v1`。
2. strict resume：从第 1 阶段实际验证过的 `last.ckpt` 恢复，仍为 2 GPU/BS24/GA1，
   `MAX_STEPS=8, MODE_UPDATES_PER_EPOCH=8,
   CHECKPOINT_EVERY_N_STEPS=4, ONLINE_VAL_CHECK_INTERVAL_STEPS=4`，独立 output 提议为
   `/data/home/frank/experiments/stereo_hy_four_mode_resume8_h2002_v1`。结束时每 mode
   恰好 2 updates，检查 counters、下一 mode、finite loss、peak memory 与 checkpoint。
3. 小数据 overfit：仍固定 48 mono + 48 stereo、2 GPU/BS24/GA1，建议首个预算
   `MAX_STEPS=400, MODE_UPDATES_PER_EPOCH=400,
   CHECKPOINT_EVERY_N_STEPS=100, ONLINE_VAL_CHECK_INTERVAL_STEPS=100`；output 提议为
   `/data/home/frank/experiments/stereo_hy_four_mode_overfit400_h2002_v1`。为避免重复 teacher
   forward，提议 `ONLINE_GT_CACHE_ENABLED=1`，独立 cache root 为
   `/data/home/frank/experiments/stereo_hy_four_mode_online_gt_cache_h2002_v1`；其 immutable
   geometry/provenance mismatch 必须 fail closed。

上述 GPU ID 必须在 launch preflight 后才可填写；不得继承旧任务的共享 GPU 授权。
WandB 建议保持 logging enabled 且显式 `WANDB_MODE=offline`，也需用户确认本次沿用。

## 已执行验证

```text
python -m py_compile <本次 Python 文件>
python tests/stereo/test_entrypoints_source.py       # 15/15 passed
git diff --check                                    # passed; only CRLF notices
fixed48 two-rank sampler standalone assertion        # passed
partial compressed NPZ frame-0 decode assertion      # passed
H200-1 cache exporter identical rerun                 # passed
```

本地 Python 没有 torch、pytorch_lightning、pytest，因此 tensor/unit suite 尚未运行；
没有为了测试擅自安装本地依赖。H200-2 unified runtime 首次完整 suite 为
113 passed / 2 failed；失败来自旧 source-boundary test 未允许新增 mono dataset classes，
以及把 `__pycache__/*.pyc` 误判为 legacy source。定向修复后两台服务器同步到
`e4f29649f08aa4aab9a06cd7e1c3a36b657263b8`，复跑结果为
`115 passed, 4 warnings in 4.58s`；warnings 均为依赖 deprecation。

## 当前 blocker

数据准备、代码同步和 CPU/tensor blocker 均已解除。下一门是 H200-2 两 rank BS24
GPU smoke 的实时 launch preflight；GPU smoke 完成前不进入 400-step overfit。
