# 文件架构重构记录

## 目的与边界

- 基线分支：`hezhou-las2-h`
- 基线 commit：`cfa07e2d5d970b7fbdfb20508ffc6a7e06bb8f96`
- 仅按既有职责拆分文件，保持训练 CLI、Stage A CLI、模型 forward、checkpoint
  state-dict key 和数据合同不变。
- 不拆分或修改 `stereo_tokenizer/online_gt.py` 内的教师 backend；不改变 GAN 行为和依赖。

## 新目录职责

### `evaluation/stage_a/`

- `selection.py`：冻结 selection 与解码 preflight。
- `quality.py`：Stage A 质量评估。
- `benchmark.py`：CUDA latency、memory 与 throughput benchmark。
- `report.py`：artifact 校验与 scorecard 生成。
- `contract.py`、`data.py`、`manifest.py`、`metrics.py`、`runtime.py`：对应的
  合同、数据、manifest、指标与共享执行支持。
- `evaluation/tokenizer_stage_a.py` 只保留稳定子命令分发入口。

### `stereo_tokenizer/training/`

- `runtime.py`：parser、分布式、batch、schedule 与运行参数校验。
- `checkpoints.py`：continuation、stage transition、discriminator expansion。
- `provenance.py`：immutable resolved config 与 run manifest。
- `profiling.py`：step timing 和 torch profiler callbacks。
- `callbacks.py`：训练 callback 组装。
- 根目录 `train_stereo_vae.py` 只负责 checkpoint 输入准备、对象组装和 Trainer 启动。

### 模型与测试

- `stereo_tokenizer/modules/stereo_encoder.py`：`StereoEncoder`。
- `stereo_tokenizer/modules/stereo_decoder.py`：`StereoDecoder`。
- `stereo_tokenizer/contracts.py`：共享 eye/temporal mode 类型与帧数合同。
- `tests/` 按 `data`、`evaluation`、`model`、`teachers`、`training` 镜像生产职责。

## 兼容性

- 训练仍使用 `python train_stereo_vae.py`。
- Stage A 仍使用 `python -m evaluation.tokenizer_stage_a <command>`。
- `from stereo_tokenizer import StereoVAE` 保持不变。
- Encoder/Decoder 在 `StereoVAE` 中仍注册为 `encoder`/`decoder`，参数属性名未改变，
  因此本次文件迁移不改变 state-dict key。

## 验证

- `python -m compileall -q train_stereo_vae.py stereo_tokenizer evaluation tests`：通过。
- `python -m unittest tests.test_source_boundary tests.training.test_entrypoints_source`：
  24 项通过。
- Git Bash 对 `train_stereo_vae.sh` 和 `run_h100_canonical_smoke.sh` 执行
  `bash -n`：通过。
- 对基线 commit 和拆分后模块进行 AST 对比：49 个迁移的函数/类函数体完全一致。
- `StereoVAE` 完整 class AST 与基线一致，Encoder/Decoder 仍注册在原属性名下。
- `online_gt.py`、`pyproject.toml`、`uv.lock` 与基线无差异，C 类范围未改动。
- `git diff --check`：通过（仅有 Windows LF/CRLF 提示）。
- 本机没有 PyTorch，未运行 tensor/CUDA 单元测试和 GPU smoke；这些仍是运行时门禁。
