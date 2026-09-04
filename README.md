# StereoVAE：单目/双目四模式视觉编码器

本分支实现一个从零训练的结构化 VAE，训练模式为：

- `mono/single_frame`
- `mono/four_frame`
- `stereo/single_frame`
- `stereo/four_frame`

单目输入为 `[B,1,1,3,T,256,256]`，双目输入为
`[B,3,2,3,T,256,256]`，其中 `T=1` 或 `T=4`。编码器输出
`[B,V,48,1,16,16]` latent；解码器输出参考左目 RGB 和
`raw_relative_log_depth`。模型不直接输出 metric depth，也不加载本仓库之外的
VAE 预训练权重。

本 README 面向一台全新的 Linux GPU 服务器，覆盖：

1. 系统与 Python 环境；
2. LAS2-H 与 DA3-BASE 两套在线 GT teacher；
3. Hy、LIBERO、UMI 三源 manifest 与 UMI rectification 合同；
4. 四模式统一采样与逻辑更新调度；
5. 环境检查、单元测试、四模式 smoke、恢复训练和评估。

> 当前生产训练链路固定为 Hy/LIBERO/UMI 三数据源四模式采样。各模式必须具有相同
> effective global batch，且 mono 模式固定 `GA=1`；启动时会 fail closed 校验完整合同。

## 1. 当前执行链路

| 数据 | 输入 | 在线 GT | 训练目标 |
| --- | --- | --- | --- |
| UMI stereo | 三组同步且校正后的左右目 MP4 | LAS2-H，双向推理与 LR consistency | pixel disparity → centered relative log-depth |
| Hy/LIBERO mono | manifest 选定的相机帧 | DA3-BASE | native relative depth → centered relative log-depth |

两套 GT 都在训练 callback 中在线生成。`ONLINE_GT_CACHE_ENABLED=1` 只开启增量
teacher cache；它不是必须提前离线生成的数据集。缓存只能在 teacher、权重、预处理、
阈值和 source-frame 配置完全相同时复用。

主要入口：

- `scripts/stereo/train_stereo_vae.sh`：统一训练 launcher；
- `train_stereo_vae.py`：参数校验、teacher provenance、Lightning Trainer；
- `evaluation/tokenizer_stage_a.py`：统一 Stage A selection、preflight、质量、性能和报告入口；
- `evaluation/stage_a_runtime.py`：Stage A 共享 checkpoint、teacher 与可视化运行时；
- `scripts/data/audit_lerobot_stereo_rectification.py`：双目校正审计；
- `scripts/data/build_lerobot_stereo_manifest.py`：episode 级 90/5/5 manifest；
- `scripts/data/build_pretrain_manifest.py`：构建三源训练 manifest。

## 2. 已验证的软件/硬件基线

推荐使用与当前 H200 smoke 一致的组合：

- Linux x86_64；
- NVIDIA driver 能运行 CUDA 12.6 wheel；
- Python 3.12；
- PyTorch `2.7.1+cu126`；
- torchvision `0.22.1+cu126`；
- PyTorch Lightning `2.5.6`；
- NumPy `1.26.2`；
- OpenCV `4.11.0`；
- PyAV `16.0.1`；
- WandB `0.23.1`（可用 `DISABLE_WANDB=1` 完全关闭）。

四模式 `BS24` 的已测最重模式是 `stereo/four_frame`，单 rank 峰值 allocated 约
116 GB、reserved 约 128–133 GB，因此推荐 141 GB H200。显存更小的 GPU 不能直接沿用
这条冻结 recipe；本分支会拒绝把四模式 batch size 改成其他值。

先安装系统工具。以下命令以自带 Python 3.12 的 Ubuntu 24.04 为例；其他发行版请用
对应包管理器提供 Python 3.12，不要回退到未验证的 Python 版本：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg libgl1 libglib2.0-0 python3.12 python3.12-venv
nvidia-smi
uv --version
```

## 3. 目录规划与代码检出

数据、teacher、Python 环境和训练输出都放在仓库外：

```bash
export WORK_ROOT=/data/$USER/stereo-vae
export PROJECT_ROOT=$WORK_ROOT/Stereo-vison-Tokenizer
export RUNTIME_ROOT=$WORK_ROOT/runtime
export EXTERNAL_ROOT=$WORK_ROOT/external
export ASSET_ROOT=$WORK_ROOT/assets
export RUN_ROOT=$WORK_ROOT/runs

mkdir -p "$RUNTIME_ROOT" "$EXTERNAL_ROOT" "$ASSET_ROOT" "$RUN_ROOT"
git lfs install
git clone --branch hezhou-las2-h \
  https://github.com/Frankaaay/Stereo-vison-Tokenizer.git \
  "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

git status --short --branch
git rev-parse HEAD
```

运行前应记录实际 branch、完整 SHA 和 clean 状态。若要复现某次实验，应再
`git checkout <FULL_COMMIT_SHA>`，不要依赖分支名推断代码版本。

仓库内目录职责固定为：

- `doc/`：冻结的设计规格、数据合同和长期架构决策；
- `docs/YY-MM-DD/`：实际实验、调试、运行记录和阶段结论；
- `environments/`：与根训练环境隔离的辅助 uv project；
- 根目录 `pyproject.toml` 与 `uv.lock`：训练环境的直接依赖声明和完整锁定结果。

Python environment、数据、teacher、checkpoint、cache 和 run output 都放在仓库外；
仓库只保存可复现它们的声明、锁文件、代码和记录。

## 4. 创建训练环境

根目录 `pyproject.toml` 是训练环境的唯一直接依赖声明，`uv.lock` 固定完整传递依赖图，
包括 PyTorch `2.7.1+cu126`、torchvision `0.22.1+cu126`、PyTorch Lightning `2.5.6`
和 xFormers `0.0.31.post1`。不要手工编辑 `uv.lock`，也不要再从旧
`requirements.txt` 安装。

实际 venv 仍放在仓库外。`UV_PROJECT_ENVIRONMENT` 必须在每次 sync 前指向目标环境，
避免在仓库中生成或误用 `.venv`：

```bash
cd "$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/train"
uv lock --check
uv sync --frozen --python 3.12
source "$RUNTIME_ROOT/train/bin/activate"
```

训练 lock 只面向 Linux x86_64，并把 torch/torchvision 显式绑定到官方 CUDA 12.6
wheel index。uv 可以使用现有 Python 3.12，也可以在缺失时安装 managed Python；最终仍须
以版本打印和 CUDA preflight 验证实际解释器及 wheel。

### 4.1 安装 LAS2-H source

本分支直接从 LAS2 repository 导入 `core.models`，不把它安装进本仓库：

```bash
export LAS2_H_REPO=$EXTERNAL_ROOT/LiteAnyStereo
export LAS2_H_SOURCE_SHA=8c97bd4c4da3712c2ac60003a23201dfdb5935f4
git clone https://github.com/TomTomTommi/LiteAnyStereo.git "$LAS2_H_REPO"
git -C "$LAS2_H_REPO" checkout "$LAS2_H_SOURCE_SHA"
git -C "$LAS2_H_REPO" status --short
```

上述 source SHA 是 2026-08-26 核对的官方 `main`。LAS2-H 运行所需的 PyTorch、
OpenCV、timm、pandas 等依赖已在上一节安装；ONNX、TensorRT demo 依赖不是本分支
LAS2-H PyTorch teacher 的必需项。

### 4.2 安装 DA3 source

```bash
export DA3_REPO=$EXTERNAL_ROOT/depth-anything-3
export DA3_SOURCE_SHA=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4

git clone https://github.com/ByteDance-Seed/depth-anything-3.git "$DA3_REPO"
git -C "$DA3_REPO" checkout "$DA3_SOURCE_SHA"
git -C "$DA3_REPO" status --short

# DA3 的第三方依赖和 xformers 已由根 uv.lock 固定；这里只暴露冻结 source。
uv pip install \
  --python "$RUNTIME_ROOT/train/bin/python" \
  --no-deps \
  -e "$DA3_REPO"
```

根项目的 `uv sync` 是 exact sync，之后若再次 sync，会移除这个仓库外 editable package；
因此每次重建训练环境时都要在 sync 后重新执行上面的 `--no-deps -e`，然后完成 source SHA、
import 和版本检查。不得去掉 `--no-deps` 让 DA3 再次解析或替换 locked dependency。

安装 DA3 后必须重新核对 PyTorch 没有被依赖解析器替换：

```bash
uv pip check --python "$RUNTIME_ROOT/train/bin/python"

python - <<'PY'
import torch
import torchvision
import pytorch_lightning as pl
import av
import cv2
from depth_anything_3.api import DepthAnything3

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("lightning", pl.__version__)
print("pyav", av.__version__)
print("opencv", cv2.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.__version__.startswith("2.7.1")
assert torchvision.__version__.startswith("0.22.1")
assert pl.__version__ == "2.5.6"
assert av.__version__ == "16.0.1"
assert torch.cuda.is_available()
PY
```

## 5. 下载并校验两套 GT 权重

### 5.1 LAS2-H stereo disparity teacher

官方 source：<https://github.com/TomTomTommi/LiteAnyStereo>

官方权重：<https://huggingface.co/tomtomtommi/LiteAnyStereoV2>

```bash
export LAS2_H_CHECKPOINT_DIR=$ASSET_ROOT/las2-h
export LAS2_H_CHECKPOINT_REVISION=17788d91618646fa781a14462e2926a034b9f49d

hf download tomtomtommi/LiteAnyStereoV2 LAS2_H.pth \
  --revision "$LAS2_H_CHECKPOINT_REVISION" \
  --local-dir "$LAS2_H_CHECKPOINT_DIR"

export LAS2_H_CHECKPOINT=$LAS2_H_CHECKPOINT_DIR/LAS2_H.pth
export LAS2_H_CHECKPOINT_SHA256=$(sha256sum "$LAS2_H_CHECKPOINT" | cut -d ' ' -f 1)
test ${#LAS2_H_CHECKPOINT_SHA256} -eq 64
printf 'LAS2-H SHA256=%s\n' "$LAS2_H_CHECKPOINT_SHA256"
```

训练时固定：

- backend：`las2_h`；
- model：LAS2-H；
- `max_disp=192`；
- `valid_iters=4`；
- 左/右双向推理；
- LR consistency、disparity range 和 non-padding mask 共同形成 valid mask。

程序会在加载前重新计算 checkpoint SHA256；不匹配时直接失败。

### 5.2 DA3-BASE mono relative-depth teacher

官方 source：<https://github.com/ByteDance-Seed/depth-anything-3>

官方权重：<https://huggingface.co/depth-anything/DA3-BASE>

```bash
export DA3_CHECKPOINT=$ASSET_ROOT/DA3-BASE
export DA3_CHECKPOINT_REVISION=f4a6c9b3c95e41c82048423d3493a81ec3fa810e

hf download depth-anything/DA3-BASE \
  --revision "$DA3_CHECKPOINT_REVISION" \
  --local-dir "$DA3_CHECKPOINT"

export DA3_CHECKPOINT_SHA256=$(sha256sum "$DA3_CHECKPOINT/model.safetensors" | cut -d ' ' -f 1)
test ${#DA3_CHECKPOINT_SHA256} -eq 64
printf 'DA3-BASE model.safetensors SHA256=%s\n' "$DA3_CHECKPOINT_SHA256"
```

当前验证过的 `model.safetensors` SHA256 是：

```text
e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5
```

训练时固定 `process_res=504`、`upper_bound_resize` 和
`finite(depth) & depth>0 & non_padding` mask。DA3 confidence 只记录和缓存，不应用
未冻结的 confidence threshold。

## 6. 数据准备

### 6.1 仓库能够和不能够完成的步骤

本仓库不包含以下上游转换器：

- raw MCAP → LeRobot v3 shard；
- raw Hy 数据 → Lance overlay。

因此，新服务器必须先获得以下两份只读上游数据：

1. 含六路 stereo MP4、episode parquet、`_manifests` 的 LeRobot v3 根目录，以及
   对应的 source episode JSON/calibration 根目录；
2. 含 `tables.json` 和 `table_*/table_*.lance` 的 Hy Lance overlay。

从这两个上游产品开始，下面的 audit、manifest、mono cache 和在线 GT 全部可由本仓库
重建。

### 6.2 LeRobot stereo 预期结构

```text
$LEROBOT_DATASET_ROOT/
├── _manifests/m_0000
├── shard_0000/
│   ├── meta/episodes/chunk-000/file-000.parquet
│   └── videos/
│       ├── observation.images.head_left/...
│       ├── observation.images.head_right/...
│       ├── observation.images.left_wrist_left/...
│       ├── observation.images.left_wrist_right/...
│       ├── observation.images.right_wrist_left/...
│       └── observation.images.right_wrist_right/...
└── shard_0000.failures.json                 # 可选

$LEROBOT_SOURCE_ROOT/<episode>/.../<episode>.json
```

数据合同固定为 30 FPS、源分辨率 `480x640`、帧 offsets `[0,3,6,9]`、sample
start stride 12。Student preprocessing 为保持宽高比的
`480x640 → 192x256 → top/bottom pad 32 → 256x256`。

设置路径；`SOURCE_MANIFEST_PREFIX` 必须与 `_manifests/m_*` 中记录的旧 source 路径
前缀逐字一致：

```bash
export LEROBOT_DATASET_ROOT=/data/datasets/umi_lerobot_v3
export LEROBOT_SOURCE_ROOT=/data/datasets/umi_source_episodes
export SOURCE_MANIFEST_PREFIX=/data/umi_vio_data_260714/
export PREPROCESS_ROOT=$WORK_ROOT/preprocessed/lerobot
mkdir -p "$PREPROCESS_ROOT"
```

#### A. Rectification audit

audit 输出路径和 visual 目录必须尚不存在：

```bash
cd "$PROJECT_ROOT"
python scripts/data/audit_lerobot_stereo_rectification.py \
  --dataset-root "$LEROBOT_DATASET_ROOT" \
  --source-root "$LEROBOT_SOURCE_ROOT" \
  --source-manifest-prefix "$SOURCE_MANIFEST_PREFIX" \
  --episode-count 96 \
  --seed 1234 \
  --output "$PREPROCESS_ROOT/rectification_audit.json" \
  --visual-root "$PREPROCESS_ROOT/rectification_visuals"
```

成功条件是 JSON 中 `result="pass"`，并且 `selected_mode` 为
`verified_pre_rectified` 或 `apply_calibration`。必须人工查看每个视角的 epipolar
visual；不要用 `--allow-provisional-pre-rectified` 进入正式训练。

```bash
export LEROBOT_RECTIFICATION_AUDIT_SHA256=$(sha256sum "$PREPROCESS_ROOT/rectification_audit.json" | cut -d ' ' -f 1)
printf 'rectification audit SHA256=%s\n' "$LEROBOT_RECTIFICATION_AUDIT_SHA256"
```

#### B. Episode manifest 与 90/5/5 split

```bash
python scripts/data/build_lerobot_stereo_manifest.py \
  --dataset-root "$LEROBOT_DATASET_ROOT" \
  --source-root "$LEROBOT_SOURCE_ROOT" \
  --source-manifest-prefix "$SOURCE_MANIFEST_PREFIX" \
  --rectification-audit "$PREPROCESS_ROOT/rectification_audit.json" \
  --split-seed 1234 \
  --output-manifest "$PREPROCESS_ROOT/episode_manifest.jsonl" \
  --output-summary "$PREPROCESS_ROOT/episode_manifest_summary.json"

export LEROBOT_EPISODE_MANIFEST=$PREPROCESS_ROOT/episode_manifest.jsonl
sha256sum "$LEROBOT_EPISODE_MANIFEST" "$PREPROCESS_ROOT/episode_manifest_summary.json"
```

split 在 episode 层完成，不会让同一 episode 的窗口跨 train/val/test。manifest 会保存
视频路径、calibration、rectification audit SHA、预处理合同和 source JSON SHA。

### 6.3 三源 node-local manifests

训练直接读取 Hy Lance、LIBERO LeRobot v2.1 和 UMI LeRobot v3。Hy/LIBERO manifest
只保存逻辑 `root_alias` 与相对路径，每台 H200 用自己的 alias JSON 映射物理根；UMI
沿用现有 episode manifest、node-local dataset root 和 rectification audit SHA。Hy 只枚举
`observation_images_cam_high`，不使用腕部相机：

```bash
python scripts/data/build_pretrain_manifest.py hy \
  --root hy_primary=/data/shared/hy_embodied/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data \
  --root hy_rest=/data/shared/hy_embodied_rest/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data \
  --output "$WORK_ROOT/manifests/hy-h200-2.jsonl"

python scripts/data/build_pretrain_manifest.py libero \
  --root libero=/data/shared/offline/datasets/libero_mujoco3.3.2 \
  --output "$WORK_ROOT/manifests/libero-h200-2.jsonl"

python scripts/data/build_pretrain_manifest.py umi \
  --root umi=/data/shared/datasets/umi_raw_data_260714 \
  --output "$WORK_ROOT/manifests/umi-h200-2.jsonl"
```

H200-1 对其本地物理根重复执行同一命令即可；builder 不硬编码奇偶 table。每个 JSONL
旁边会生成 summary JSON，记录 episode/window/split 数和 manifest SHA256。

## 7. 运行前检查

### 7.1 代码与 Python

```bash
cd "$PROJECT_ROOT"
git status --short --branch
git rev-parse HEAD
uv lock --check
uv pip check --python "$RUNTIME_ROOT/train/bin/python"

python -m py_compile \
  train_stereo_vae.py evaluation/tokenizer_stage_a.py evaluation/stage_a_runtime.py \
  stereo_tokenizer/data.py stereo_tokenizer/online_gt.py
bash -n scripts/stereo/train_stereo_vae.sh
python -m pytest -q tests
```

### 7.2 数据与 teacher

```bash
test -f "$LEROBOT_EPISODE_MANIFEST"
test -d "$LEROBOT_DATASET_ROOT"
test -f "$HY_MANIFEST"
test -f "$LIBERO_MANIFEST"
test -f "$UMI_MANIFEST"
test -d "$LAS2_H_REPO"
test -f "$LAS2_H_CHECKPOINT"
test -d "$DA3_REPO"
test -f "$DA3_CHECKPOINT/model.safetensors"

test "$(git -C "$LAS2_H_REPO" rev-parse HEAD)" = "$LAS2_H_SOURCE_SHA"
test -z "$(git -C "$LAS2_H_REPO" status --porcelain)"
test "$(git -C "$DA3_REPO" rev-parse HEAD)" = "$DA3_SOURCE_SHA"
test -z "$(git -C "$DA3_REPO" status --porcelain)"
test "$(sha256sum "$LAS2_H_CHECKPOINT" | cut -d ' ' -f 1)" = "$LAS2_H_CHECKPOINT_SHA256"
test "$(sha256sum "$DA3_CHECKPOINT/model.safetensors" | cut -d ' ' -f 1)" = "$DA3_CHECKPOINT_SHA256"
```

### 7.3 GPU

```bash
nvidia-smi
```

启动前确认所选 GPU 没有其他进程和显存占用。不要自动 kill 不属于本次任务的进程。

## 8. 单机与双机 IB 模式

训练 launcher 默认保持单机语义：

```bash
export DISTRIBUTED_MODE=single
export NUM_NODES=1
```

单机模式不要求 `NODE_RANK`、`MASTER_ADDR` 或 `MASTER_PORT`。`GPU_COUNT` 仍表示本机
可见 GPU 数。双机 H200 使用显式、fail-closed 的 IB 模式；两端必须使用同一代码 SHA、
配置和端口，并分别设置 node rank：

```bash
export DISTRIBUTED_MODE=ib
export NUM_NODES=2
export GPU_COUNT=2
export NODE_RANK=0                 # h200-1；h200-2 使用 1
export MASTER_ADDR=214.30.239.40   # 启动前重新核对 h200-1 bond0
export MASTER_PORT=<UNIQUE_PORT>
export GLOBAL_BATCH_SIZE=96        # 2 nodes * 2 GPUs * BS24 * GA1
```

IB 模式使用 `bond0` 完成 rendezvous，并将 NCCL HCA限制为
`mlx5_0:1` 到 `mlx5_7:1`。每台节点在各自 node-local `OUTPUT_ROOT` 写入 NCCL 日志；
launcher 只有在日志出现真实 `NET/IB` transport 时才返回成功，不允许静默回退 Socket。

`GPU_COUNT` 是每节点卡数，因此完整 batch合同为：

| nodes x GPUs/node | world size | BS/device | GA | global batch |
| --- | ---: | ---: | ---: | ---: |
| 1 x 8 | 8 | 24 | 1 | 192 |
| 2 x 1 | 2 | 24 | 1 | 48 |
| 2 x 2 | 4 | 24 | 1 | 96 |
| 2 x 8 | 16 | 24 | 1 | 384 |

正式训练前先用 `scripts/stereo/check_ib_collective.py` 完成双机 collective gate。
关闭 IB 后仍可使用现有单机单卡或单机 DDP；不同 world size之间不得 strict resume。

## 9. 三数据源四模式最小运行配置

下面是已经验证过的 2 GPU H200 配置。输出目录必须使用一个从未存在过的新路径：

```bash
source "$RUNTIME_ROOT/train/bin/activate"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0,1
export OUTPUT_ROOT=$RUN_ROOT/four-mode-smoke-$(date +%Y%m%d-%H%M%S)
export GPU_COUNT=2
export GLOBAL_BATCH_SIZE=48
export PER_DEVICE_BATCH_SIZE=24
export GRAD_ACCUMULATES=1
export MAX_STEPS=4
export MODE_UPDATES_PER_EPOCH=20

export LEARNING_RATE=1e-4
export MIN_LEARNING_RATE=1e-4
export WARMUP_STEPS=20
export KL_WARMUP_STEPS=100
export RGB_WEIGHT=1.0
export RELATIVE_DEPTH_WEIGHT=1.0
export RELATIVE_GRADIENT_WEIGHT=0.1
export KL_WEIGHT=1e-6
export PERCEPTUAL_WEIGHT=1.0
export SINGLE_FRAME_SOURCE_INDEX=0

export MODE_UPDATE_WEIGHTS=35:35:15:15
export MONO_DATASET_WEIGHTS=9:1
export MODE_SCHEDULE_SEED=1234
export HY_MANIFEST=$WORK_ROOT/manifests/hy-h200-2.jsonl
export HY_ROOT_ALIASES='{"hy_primary":"/data/shared/hy_embodied/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data","hy_rest":"/data/shared/hy_embodied_rest/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data"}'
export LIBERO_MANIFEST=$WORK_ROOT/manifests/libero-h200-2.jsonl
export LIBERO_ROOT_ALIASES='{"libero":"/data/shared/offline/datasets/libero_mujoco3.3.2"}'
export UMI_MANIFEST=/data/home/frank/experiments/stereo_lerobot_cpu_20260824_approval1/h200_2_local_manifest_v1.jsonl
export UMI_DATASET_ROOT=/data/shared/datasets/umi_lerobot_v3_260714
export UMI_RECTIFICATION_AUDIT_SHA256=41d2bfecaae85dd18f7cfd1a2a3a2177e8fd4aa8897be1cb411d85c3092a7d25

export FOUNDATION_STEREO_BACKEND=las2_h
export LAS2_H_REPO
export LAS2_H_SOURCE_SHA
export LAS2_H_CHECKPOINT
export LAS2_H_CHECKPOINT_SHA256
export LAS2_H_VALID_ITERS=4
export LAS2_H_MAX_DISP=192
export FOUNDATION_STEREO_PAIR_MICROBATCH=48

export DA3_REPO
export DA3_SOURCE_SHA
export DA3_CHECKPOINT
export DA3_CHECKPOINT_SHA256

export ONLINE_GT_CACHE_ENABLED=0
export ONLINE_VAL_CHECK_INTERVAL_STEPS=4
export CHECKPOINT_EVERY_N_STEPS=4
export NUM_WORKERS=8
export DISABLE_WANDB=1
export DISABLE_MEDIA_LOGGING=1

bash scripts/stereo/train_stereo_vae.sh 2>&1 | tee "$OUTPUT_ROOT.console.log"
```

`PERCEPTUAL_WEIGHT=1.0` 会在首次构造模型时下载 torchvision VGG16 权重和项目内
LPIPS `vgg.pth`。离线服务器应提前在同版本环境中完成一次下载并复制对应缓存；若只验证
接口，可把 `PERCEPTUAL_WEIGHT=0`，但该结果不能与上面的冻结 loss 配置比较。

成功 smoke 至少应满足：

- shell exit code 为 0；
- 日志没有 traceback、CUDA OOM 或 NaN/Inf；
- `resolved_config.json` 与 `run_manifest.json` 存在；
- checkpoint 目录中存在可读取的 `last.ckpt`；
- `last.ckpt` 中四种 mode 各完成 1 update；
- validation 完成四种 mode，`val/mixed/total_loss` 有限。

8 GPU fresh smoke 使用 `GPU_COUNT=8`、`GLOBAL_BATCH_SIZE=192`，其他值不变。固定
48 条 source 会在每个 local batch 内重复，因此只能验证 8-rank DDP/显存/执行稳定性，
不能把它解释成 192 个唯一样本的吞吐。

## 10. 开启在线 GT cache

smoke 成功后，小数据 overfit 可使用新的仓库外 cache root：

```bash
export ONLINE_GT_CACHE_ENABLED=1
export ONLINE_GT_CACHE_ROOT=$RUN_ROOT/online-gt-cache-v1
```

LAS2-H 与 DA3 使用各自 provenance namespace。Stereo cache schema v4 会把 LAS2 source
SHA、checkpoint、`SINGLE_FRAME_SOURCE_INDEX`、disparity range 和 LR threshold 编入
namespace/metadata；DA3 cache schema v3 同样编码 source frame、source/checkpoint 和
preprocess。任一语义变化都会进入新 namespace，旧 stereo v3 / DA3 v2 cache 不会被复用。

## 11. 恢复训练

resume 必须保持原 checkpoint 的代码语义、world size、BS24、GA1、seed、数据、teacher
权重和四模式 schedule。使用新的 output root，并把 `MAX_STEPS` 设为新的总目标 step：

```bash
export RESUME_FROM_CHECKPOINT=/absolute/path/to/previous/checkpoints/last.ckpt
export OUTPUT_ROOT=$RUN_ROOT/four-mode-resume-$(date +%Y%m%d-%H%M%S)
export MAX_STEPS=8
export MODE_UPDATES_PER_EPOCH=8
export CHECKPOINT_EVERY_N_STEPS=4
export ONLINE_VAL_CHECK_INTERVAL_STEPS=4

bash scripts/stereo/train_stereo_vae.sh 2>&1 | tee "$OUTPUT_ROOT.console.log"
```

程序会严格读取 `stereo_update_counters`，校验 schedule seed、已完成的 mode prefix 和
下一 mode；缺少这些字段的旧 checkpoint 不允许推断式恢复。

## 12. 统一 Stage A 评估

旧的通用 evaluator 已移除。正式评估统一使用
`evaluation/tokenizer_stage_a.py` 的五个子命令：

- `selection`：冻结 UMI、Hy、LIBERO 的评估样本与 provenance；
- `preflight`：少量解码并核验 canonical batch ABI；
- `run`：一次模型调用编码完整视图，输出 RGB、teacher-relative 几何与时序指标；
- `benchmark`：报告 encode、posterior mean、decode 和端到端性能；
- `report`：核验完整 artifact 集并生成 Stage A scorecard。

每个子命令的当前参数以 `--help` 为准：

```bash
python -m evaluation.tokenizer_stage_a selection --help
python -m evaluation.tokenizer_stage_a preflight --help
python -m evaluation.tokenizer_stage_a run --help
python -m evaluation.tokenizer_stage_a benchmark --help
python -m evaluation.tokenizer_stage_a report --help
```

正式结果仍须同时满足：程序 exit code 0、冻结 selection 和 checkpoint SHA、精确
sample count、有限指标、完整 mode/source-position 覆盖，以及可读取的案例和报告产物。

## 14. 输出与 provenance

每次训练至少保留：

- `resolved_config.json`；
- `run_manifest.json`，包括代码 SHA、teacher backend、权重 SHA、LAS2 source SHA 和
  DA3 source SHA；
- Lightning checkpoint，尤其 `last.ckpt`；
- console log 与 WandB offline/online run（若启用）；
- 数据 manifest、summary、rectification audit 及各自 SHA256；
- 评估 `metrics.json` 和 deterministic case images。

数据集、teacher 权重、online GT cache、checkpoint、日志和 run output 都不得提交进
Git。更换服务器时复制这些仓库外资产，或按本 README 重新生成，并重新核对所有 SHA。

## 15. 常见故障

- `ModuleNotFoundError: av`：训练环境缺少 `av==16.0.1`；这会在 LeRobot MP4
  DataLoader worker 中失败。
- `ModuleNotFoundError: wandb`：安装 `wandb==0.23.1`，或显式
  `DISABLE_WANDB=1`；`WANDB_MODE=offline` 不能替代缺失的 package。
- `DA3 source SHA mismatch`：DA3 repo 不在冻结 commit，或 repo 有未提交修改。
- `LAS2-H source SHA mismatch`：LAS2 repo 不在 `LAS2_H_SOURCE_SHA`，或 repo 有未提交修改。
- `DA3 checkpoint SHA256 mismatch` / `LAS2-H checkpoint SHA256 mismatch`：传入的
  环境变量与实际文件不一致，重新计算 SHA，不要绕过检查。
- `LeRobot MP4 loading requires PyAV`：PyAV 未安装或当前 shell 没激活训练 venv。
- `rectification audit did not pass`：检查 source-root、calibration、视频同步和可视化；
  不要用 provisional flag 掩盖失败。
- 四模式报 BS/world-size 错误：当前合同只接受 BS24、GA1 和 1/2/8 ranks。
- CUDA OOM：已验证的 BS24 需要 H200 级大显存；不要把 teardown 的非零 exit code当作
  根因，应读取日志中的第一个真实 CUDA/算子异常。

## License

本项目使用仓库根目录 `LICENSE` 中的 MIT License。
