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
3. LeRobot 双目数据的 rectification audit 与 manifest；
4. Hy `cam_high` 单目 smoke cache；
5. 环境检查、单元测试、四模式 smoke、恢复训练和评估。

> 当前四模式数据链路固定为 **48 条 Hy mono + 48 条 LeRobot stereo**，只支持
> 1、2 或 8 个 DDP rank，且固定 `BS=24/GPU, GA=1`。它适用于接口 smoke、显存验证、
> checkpoint/resume 和小数据过拟合，不应当被描述为正式全量四模式训练。

## 1. 当前执行链路

| 数据 | 输入 | 在线 GT | 训练目标 |
| --- | --- | --- | --- |
| LeRobot stereo | 三组同步且校正后的左右目 MP4 | LAS2-H，双向推理与 LR consistency | pixel disparity → centered relative log-depth |
| Hy mono | `cam_high` 的 1/4 帧 RGB cache | DA3-BASE | native relative depth → centered relative log-depth |

两套 GT 都在训练 callback 中在线生成。`ONLINE_GT_CACHE_ENABLED=1` 只开启增量
teacher cache；它不是必须提前离线生成的数据集。缓存只能在 teacher、权重、预处理、
阈值和 source-frame 配置完全相同时复用。

主要入口：

- `scripts/stereo/train_stereo_vae.sh`：统一训练 launcher；
- `train_stereo_vae.py`：参数校验、teacher provenance、Lightning Trainer；
- `eval_stereo_vae.py`：严格 checkpoint 加载与 stereo 评估；
- `scripts/data/audit_lerobot_stereo_rectification.py`：双目校正审计；
- `scripts/data/build_lerobot_stereo_manifest.py`：episode 级 90/5/5 manifest；
- `scripts/data/build_hy_mono_smoke_cache.py`：Hy Lance → 48 条 immutable RGB cache。

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

训练直接读取 Hy Lance、LIBERO LeRobot v2.1 和 UMI raw MCAP。manifest 只保存逻辑
`root_alias` 与相对路径；每台 H200 用自己的 alias JSON 映射物理根。Hy 只枚举
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
  train_stereo_vae.py eval_stereo_vae.py \
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

export FOUR_MODE_MIXED_TRAINING=1
export MODE_UPDATE_WEIGHTS=35:35:15:15
export MONO_DATASET_WEIGHTS=9:1
export MODE_SCHEDULE_SEED=1234
export HY_MANIFEST=$WORK_ROOT/manifests/hy-h200-2.jsonl
export HY_ROOT_ALIASES='{"hy_primary":"/data/shared/hy_embodied/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data","hy_rest":"/data/shared/hy_embodied_rest/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data"}'
export LIBERO_MANIFEST=$WORK_ROOT/manifests/libero-h200-2.jsonl
export LIBERO_ROOT_ALIASES='{"libero":"/data/shared/offline/datasets/libero_mujoco3.3.2"}'
export UMI_MANIFEST=$WORK_ROOT/manifests/umi-h200-2.jsonl
export UMI_ROOT_ALIASES='{"umi":"/data/shared/datasets/umi_raw_data_260714"}'

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

## 12. Stereo-only LAS2-H 训练

若只训练 LeRobot stereo 的 single/four-frame 路径，不需要 Hy cache 或 DA3：

```bash
export FOUR_MODE_MIXED_TRAINING=0
export FOUNDATION_STEREO_BACKEND=las2_h
export GPU_COUNT=8
export PER_DEVICE_BATCH_SIZE=24
export GRAD_ACCUMULATES=1
export GLOBAL_BATCH_SIZE=192
export MAX_STEPS=<TOTAL_UPDATES>
export OUTPUT_ROOT=$RUN_ROOT/stereo-only-$(date +%Y%m%d-%H%M%S)

# 其余 LeRobot、LAS2-H、loss、LR、checkpoint 和 logging 变量沿用第 9 节。
bash scripts/stereo/train_stereo_vae.sh 2>&1 | tee "$OUTPUT_ROOT.console.log"
```

正式长任务启动前应单独冻结 `MAX_STEPS`、LR/warmup、验证频率、checkpoint cadence、
WandB 模式和 output path；不要把四步 smoke 参数直接当作正式训练计划。

## 13. 严格评估

评估只接受与 checkpoint architecture 完全一致的 CLI 参数（数据选择参数
`single_frame_source_index` 除外），并使用 posterior mean 与在线 teacher。以下示例在单
GPU 上对 val split 跑两个 batch，同时评估
single/four-frame：

```bash
export STEREO_VAE_CKPT=/absolute/path/to/checkpoints/last.ckpt
export EVAL_ROOT=$RUN_ROOT/eval-$(date +%Y%m%d-%H%M%S)
mkdir -p "$EVAL_ROOT"

python eval_stereo_vae.py \
  --stereo_vae_ckpt "$STEREO_VAE_CKPT" \
  --eval_split val \
  --eval_eye_mode stereo \
  --eval_temporal_mode both \
  --device cuda \
  --bf16 \
  --max_batches 2 \
  --output_json "$EVAL_ROOT/metrics.json" \
  --visualization_dir "$EVAL_ROOT/cases" \
  --num_visualizations 2 \
  --lerobot_episode_manifest "$LEROBOT_EPISODE_MANIFEST" \
  --lerobot_dataset_root "$LEROBOT_DATASET_ROOT" \
  --lerobot_rectification_audit_sha256 "$LEROBOT_RECTIFICATION_AUDIT_SHA256" \
  --foundation_stereo_backend las2_h \
  --las2_h_repo "$LAS2_H_REPO" \
  --las2_h_source_sha "$LAS2_H_SOURCE_SHA" \
  --las2_h_checkpoint "$LAS2_H_CHECKPOINT" \
  --las2_h_checkpoint_sha256 "$LAS2_H_CHECKPOINT_SHA256" \
  --las2_h_valid_iters 4 \
  --las2_h_max_disp 192 \
  --foundation_stereo_pair_microbatch 48 \
  --resolution 256 \
  --sequence_length 4 \
  --image_channels 3 \
  --patch_embed linear \
  --patch_size 16 \
  --spatial_depth 4 \
  --temporal_depth 4 \
  --embedding_dim 512 \
  --latent_channels 48 \
  --enc_block ttww \
  --dec_block tttt \
  --twod_window_size 8 \
  --spatial_pos rope \
  --causal_in_peg \
  --peg_backend conv2d_t1_slice \
  --dim_head 64 \
  --heads 8 \
  --initialize_vit \
  --stereo_num_views 3 \
  --stereo_num_frames 4 \
  --single_frame_source_index 0 \
  --stereo_search_radii 7 7 7 \
  --stereo_search_direction left \
  --stereo_disparity_min_px 0.5 \
  --stereo_disparity_max_px 112.0 \
  --stereo_lr_error_abs_threshold_px 1.0 \
  --stereo_lr_error_relative_threshold 0.05 \
  --rgb_weight 1.0 \
  --relative_depth_weight 1.0 \
  --relative_gradient_weight 0.1 \
  --relative_depth_epsilon 1e-6 \
  --kl_weight 1e-6 \
  --perceptual_weight 1.0 \
  --image_gan_weight 0 \
  --video_gan_weight 0 \
  --gan_feat_weight 0 \
  --recon_loss_type l1 \
  --smooth_l1_beta 1.0 \
  --batch_size 8 \
  --num_workers 4 \
  --pin_memory 1 \
  --persistent_workers 1
```

### 13.1 正式 mono + DA3 评估

Mono evaluation 直接读取 Hy cam_high Lance manifest；不会从 stereo batch 截取眼睛。
Dataset 输出严格为 `[B,1,1,3,T,256,256]`，DA3 接收未带 Student padding 的原始比例
预处理，输出映射回 Student geometry。Hy manifest 的唯一视觉字段是 cam_high：

```bash
export MONO_EVAL_ROOT=$RUN_ROOT/mono-eval-$(date +%Y%m%d-%H%M%S)
mkdir -p "$MONO_EVAL_ROOT"

python eval_stereo_vae.py \
  --stereo_vae_ckpt "$STEREO_VAE_CKPT" \
  --eval_eye_mode mono \
  --eval_temporal_mode both \
  --device cuda \
  --bf16 \
  --output_json "$MONO_EVAL_ROOT/metrics.json" \
  --visualization_dir "$MONO_EVAL_ROOT/cases" \
  --num_visualizations 2 \
  --hy_manifest "$HY_MANIFEST" \
  --hy_root_aliases "$HY_ROOT_ALIASES" \
  --da3_repo "$DA3_REPO" \
  --da3_source_sha "$DA3_SOURCE_SHA" \
  --da3_checkpoint "$DA3_CHECKPOINT" \
  --da3_checkpoint_sha256 "$DA3_CHECKPOINT_SHA256" \
  --da3_process_res 504 \
  --da3_process_res_method upper_bound_resize \
  --da3_confidence_mask_mode finite_positive_non_padding \
  --resolution 256 \
  --sequence_length 4 \
  --image_channels 3 \
  --patch_embed linear \
  --patch_size 16 \
  --spatial_depth 4 \
  --temporal_depth 4 \
  --embedding_dim 512 \
  --latent_channels 48 \
  --enc_block ttww \
  --dec_block tttt \
  --twod_window_size 8 \
  --spatial_pos rope \
  --causal_in_peg \
  --peg_backend conv2d_t1_slice \
  --dim_head 64 \
  --heads 8 \
  --initialize_vit \
  --stereo_num_views 3 \
  --stereo_num_frames 4 \
  --single_frame_source_index 0 \
  --stereo_search_radii 7 7 7 \
  --stereo_search_direction left \
  --stereo_disparity_min_px 0.5 \
  --stereo_disparity_max_px 112.0 \
  --stereo_lr_error_abs_threshold_px 1.0 \
  --stereo_lr_error_relative_threshold 0.05 \
  --rgb_weight 1.0 \
  --relative_depth_weight 1.0 \
  --relative_gradient_weight 0.1 \
  --relative_depth_epsilon 1e-6 \
  --kl_weight 1e-6 \
  --perceptual_weight 1.0 \
  --image_gan_weight 0 \
  --video_gan_weight 0 \
  --gan_feat_weight 0 \
  --recon_loss_type l1 \
  --smooth_l1_beta 1.0 \
  --batch_size 8 \
  --num_workers 4 \
  --pin_memory 1 \
  --persistent_workers 1
```

Mono `metrics.json` 对 `cam_high` 报告 RGB L1、有效 depth pixels、centered
relative-log-depth L1/RMSE，并记录 DA3 source/checkpoint/process provenance。可视化支持
`single_frame`、`four_frame` 或 `both`，只渲染实际请求的输出。

### 13.2 一次运行全部四种模式

`--eval_eye_mode` 接受 `mono|stereo|both`，`--eval_temporal_mode` 接受
`single_frame|four_frame|both`，二者按笛卡尔积展开，并复用训练侧 `MODE_IDS`。例如：

```bash
python eval_stereo_vae.py \
  ... \
  --eval_eye_mode both \
  --eval_temporal_mode both \
  --hy_manifest "$HY_MANIFEST" \
  --hy_root_aliases "$HY_ROOT_ALIASES" \
  --da3_repo "$DA3_REPO" \
  --da3_source_sha "$DA3_SOURCE_SHA" \
  --da3_checkpoint "$DA3_CHECKPOINT" \
  --da3_checkpoint_sha256 "$DA3_CHECKPOINT_SHA256" \
  --lerobot_episode_manifest "$LEROBOT_EPISODE_MANIFEST" \
  --lerobot_dataset_root "$LEROBOT_DATASET_ROOT" \
  --lerobot_rectification_audit_sha256 "$LEROBOT_RECTIFICATION_AUDIT_SHA256" \
  --foundation_stereo_backend las2_h \
  --las2_h_repo "$LAS2_H_REPO" \
  --las2_h_source_sha "$LAS2_H_SOURCE_SHA" \
  --las2_h_checkpoint "$LAS2_H_CHECKPOINT" \
  --las2_h_checkpoint_sha256 "$LAS2_H_CHECKPOINT_SHA256"
```

内部按 eye mode 建立两个独立 session：mono 使用正式 mono dataset + DA3，stereo 使用
LeRobot stereo dataset + FoundationStereo/LAS2-H。同一 eye session 内 teacher 每个 batch
只推理一次，single/four 共用 GT。`metrics.json` 使用完整 `mono/single_frame` 等 mode key，
并分别记录 `datasets.mono|stereo`、`teachers.mono|stereo` 以及每个 mode 的精确 provenance。
可视化分别写入 `visualizations/mono/` 和 `visualizations/stereo/`，不会把两个不同 sample
universe 画成伪配对。

`metrics.json`、程序 exit code 0、精确 sample count、有限指标和可视化图片共同构成
评估成功；仅看到进程启动不算完成。完整 split 评估时删除 `--max_batches`。

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
