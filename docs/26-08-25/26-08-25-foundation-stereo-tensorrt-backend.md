# FoundationStereo TensorRT 32-iter backend

## 状态

- 日期：2026-08-25
- 本地 workspace：`C:\Project\Stereo-vison-Tokenizer`
- 用户指定分支：`merged-fs-vae-single-four-profiling`
- 修改前 HEAD：`7a0eb190e585b661f03e15698de5b7910b272415`
- 计划测试节点：`h200-1`（最新用户指令覆盖交接文本中的 `h200-2`）
- 当前状态：本地实现和 source 验证完成；未连接 H200、未创建 TensorRT 资产、未使用 GPU

## 目的和冻结合同

本次只实现 FoundationStereo TensorRT FP16 backend，不引入 LAS2-H、INT8、数据变更或长训练。
冻结合同如下：

- checkpoint：`23-51-11/model_best_bp2.pth`
- checkpoint SHA256：
  `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`
- `valid_iters=32`
- 输入 `[N,3,256,256]`、NCHW、RGB 0–255
- 输出 `[N,1,256,256]` 左视角正 disparity，单位 pixel
- engine profile：min/opt/max batch `1/48/48`
- 每 sample 12 个 stereo pair，保留正向、翻转反向和现有 LR consistency
- disparity `[0.5,112.0]`
- LR threshold `max(1.0, 0.05*d)`
- pilot cache off，pair microbatch 48

## 实现

### TensorRT runner

`stereo_tokenizer/online_gt.py` 增加 `pytorch|tensorrt` backend 边界。TensorRT 路径：

- 惰性导入 TensorRT，未安装 TensorRT 时不影响 PyTorch import；
- 每个 callback/rank 独立反序列化 engine 并创建 execution context；
- 使用 TensorRT v10 name-based IO API；
- 输入和输出直接绑定 PyTorch CUDA tensor `data_ptr()`；
- 使用 `torch.cuda.current_stream()` 和 `execute_async_v3()`；
- runner 内无 `.cpu()`、`.numpy()` 或 NumPy staging；
- engine、manifest、checkpoint provenance 或 runtime 环境不匹配时 fail closed；
- engine IO 必须是 device、linear、动态 batch、固定 256×256；
- 同一个 execution context 顺序执行正向和翻转反向。

manifest schema 为 `foundation-stereo-tensorrt-engine-v1`，必须记录：

- FoundationStereo repo、checkpoint、cfg、ONNX、engine SHA；
- height/width、32 iterations、opset 16、FP16、XFormers disabled；
- batch profile 和每个 IO binding 的 name/mode/dtype/shape；
- TensorRT/CUDA/driver/GPU；
- 完整 ONNX export 和 `trtexec` build 命令。

### CLI、run provenance 和 cache

`train_stereo_vae.py` 增加：

- `--foundation_stereo_backend`
- `--foundation_stereo_engine`
- `--foundation_stereo_engine_sha256`
- `--foundation_stereo_engine_manifest`
- `--foundation_stereo_engine_manifest_sha256`

TensorRT 模式只接受 32 iterations 和 pair microbatch 不大于 48，并在训练对象创建前验证全部资产。
在线 teacher run 会在 output root 写入不可静默覆盖的 `resolved_config.json` 和
`run_manifest.json`，日志打印 backend provenance。callback state key 包含 backend、checkpoint、engine
和 manifest SHA。

cache schema 更新为 `stereo-online-foundation-gt-v2`，路径 namespace 同时包含 backend 和完整资产
provenance，PyTorch 与 TensorRT cache 不会碰撞。pilot 仍冻结 cache disabled。

### launcher 和专用入口

- `scripts/stereo/train_stereo_vae.sh` 根据 `FOUNDATION_STEREO_BACKEND` 条件要求 PyTorch 或
  TensorRT 资产；TensorRT launcher 显式拒绝非 32 iterations。
- `scripts/stereo/compare_online_foundation_teacher.py` 的 `32/16/12` 语义未修改。
- 新增 `scripts/stereo/compare_foundation_backends.py`：
  - `equivalence`：只允许 32–64 samples，比较 PyTorch 32 与 TensorRT 32；
  - `tensorrt_benchmark`：只运行批准的 408-sample TensorRT arm，不重跑完整 PyTorch arm；
  - 直接 engine smoke 覆盖 batch 1、36、48；
  - 记录 engine、双向 teacher、LR consistency、decode、端到端吞吐和显存；
  - 执行交接文本冻结的数值、mask 和性能 gate。
- 新增 `scripts/stereo/write_foundation_tensorrt_manifest.py`，从实际 engine 读取 IO/profile，生成
  不覆盖已有文件的 manifest。

## 本地验证

已完成：

```text
python -m py_compile stereo_tokenizer/online_gt.py train_stereo_vae.py \
  scripts/stereo/compare_foundation_backends.py \
  scripts/stereo/write_foundation_tensorrt_manifest.py \
  tests/stereo/test_foundation_tensorrt_backend.py

python -m unittest \
  tests.stereo.test_entrypoints_source \
  tests.stereo.test_source_boundary
```

结果：27 tests passed；`git diff --check` passed；所有修改 Python 文件无超过 99 字符的行。

当前 Windows Python 没有 `torch`，因此运行态测试无法在本机导入。首个错误为：

```text
ModuleNotFoundError: No module named 'torch'
```

这同样影响修改前已有的 `tests.stereo.test_lerobot_online_contract`。没有安装本地依赖或改变环境；
运行态 PyTorch/TensorRT/CUDA 测试列入 H200 preflight。

本机没有可用 WSL distribution，因此 `bash -n` 未执行；launcher source tests 已通过，H200 上必须先补
shell syntax check。

## h200-1 计划（尚未执行）

### Git 门禁

H200 正式 clone 记录为本地 `frank` 跟踪 `origin/frank`，而本次用户指定实现分支为
`merged-fs-vae-single-four-profiling`。在连接或修改服务器前需要用户明确确认如何让 `h200-1`
到达本次精确已推送 SHA；不得自行创建 local branch/worktree、切 detached SHA、改写 `frank`、rebase
或 reset。

### 只读 preflight

用户确认后，先单次合并查询：

- `h200-1` Git status/branch/HEAD/upstream/origin；
- FoundationStereo repo SHA；
- checkpoint/cfg hash；
- LeRobot manifest、408 selection 和 rectification audit hash；
- Python、TensorRT、CUDA、driver；
- GPU owner、PID 和 command line；
- 目标 asset/output 路径不存在。

任一 GPU 被占用、worktree dirty、SHA 无法按授权方式到达、资产 hash 不符或输出已存在，停止并报告。

### 计划资产和输出

- TensorRT asset：
  `/data/home/frank/artifacts/foundation-stereo/trt/23-51-11_256x256_iters32_fp16_v1`
- equivalence output：
  `/data/home/frank/experiments/foundation_stereo_trt_h2001_equivalence_20260825_v1`
- 408 benchmark output：
  `/data/home/frank/experiments/foundation_stereo_trt_h2001_benchmark_20260825_v1`
- 3-step smoke output：
  `/data/home/frank/experiments/stereo_merged_fs_vae_trt_smoke_h2001_20260825_v1`

全部路径必须在执行前确认不存在；不覆盖已有目录。

### 执行顺序和 gate

1. 补跑 source/unit tests 和 `bash -n`。
2. 用官方 `scripts/make_onnx.py` 导出 256×256、32-iter、opset 16、dynamic batch ONNX，并运行
   ONNX checker。
3. 用 FP16 和 `left/right` 的 1/48/48 profile 构建 H200 engine。
4. 生成 manifest 和 SHA 清单。
5. 单 GPU batch 1/36/48 smoke。
6. 32–64 sample PyTorch-vs-TensorRT equivalence。
7. gate 通过后，8 GPU 完成 408-sample TensorRT benchmark；复用已记录 PyTorch 性能基线。
8. gate 通过后最多 3 optimizer steps 集成 smoke。
9. 不启动长训练，报告后等待用户决定。

正确性 gate：finite ratio 1.0；差异 P50/P95/P99 不超过 0.02/0.10/0.50 px；valid-mask IoU
不低于 0.99；每视角 valid ratio 变化不超过 0.2 个百分点。性能 gate：teacher seconds/pair
不超过 0.02669 s，即相对现有 0.05338 s 至少 2×。

任务尚未启动，因此没有运行 ETA。取得用户确认并完成首次真实 smoke 后，根据实际 engine build 时间和
至少两个有效吞吐采样更新主体完成及后处理完成 ETA。
