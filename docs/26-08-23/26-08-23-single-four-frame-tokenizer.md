# Single/Four-frame Stereo Tokenizer 实施记录

## 状态

- 日期：2026-08-23。
- 开发位置：`C:\Project\Stereo-vison-Tokenizer-single-four-frame`。
- 分支：`frank-single-four-frame-tokenizer`。
- 基线：`frank@6665b830d8b8ac7c596787492e88d64971882370`，包含 PR #3 四帧双向 temporal attention。
- 当前状态：本地实现与无 Torch 静态测试已完成；尚未 commit、push、同步 H200、启动动态测试、训练或评估。

## 目的

在不复制单帧、不修改 Manifest/cache `T=4` 合同、不改变 core loss 数学公式的前提下，让同一个 StereoVAE 显式支持：

- `stereo + four_frame`：四帧双向 temporal attention 后 `4D→D`，编码为一个 latent，再解码四帧；
- `stereo + single_frame`：共享 Spatial Encoder 与 StereoFusion，跳过全部四帧 temporal 模块，经独立 `D→D` projection 编码为一个 latent，再经单帧 Decoder 重建一帧。

`mono|stereo` 与 `single_frame|four_frame` 是两组正交的编码属性。模型接口保留四种组合能力，但第一版正式训练固定 `eye_mode=stereo`，只在 temporal mode 间交替。

## 模型和接口

- `StereoEncodeOutput` 新增 `eye_mode`、`temporal_mode`、`source_num_frames`；不定义 `LatentRole`。
- `encode()`、`forward()` 显式要求 `eye_mode` 与 `temporal_mode`；`decode()` 显式要求 `temporal_mode`。
- `single_frame` 严格要求 `T=1`，`four_frame` 严格要求 `T=4`，不根据 shape 静默推断。
- Encoder 共享 patch embedding、Spatial Transformer、StereoFusion 与 posterior head；新增 `single_frame_projection`，four 路径保持 PR #3 position→bidirectional attention→sampler。
- Decoder 新增 `single_frame_expansion`；single 跳过 four-frame position/attention，两条路径共享 Spatial Decoder、RGB Head 与 disparity Head。
- raw latent ABI 均为 `[B,3,48,1,H',W']`。Tokenizer 只携带固有编码元数据；current/history/prediction、dense/sparse、timestamp/age、observed/predicted/verified、memory position 与 valid mask 均由下游 Memory 系统管理。

## 训练路由

模式只由已完成的 generator optimizer update 数决定：

```python
temporal_mode = (
    "four_frame"
    if generator_updates % 2 == 0
    else "single_frame"
)
```

- `generator_updates` 在同一 gradient accumulation window 内不变，因此所有 micro-batch 使用同一种模式。
- Dataset、Manifest 和 cache 仍输出 `T=4`。single batch 按必填 `--single_frame_source_index` 同步截取 video、disparity 和 valid mask；当前批准配置为 `0`。
- 每个 batch 只运行当前模式的一次 forward 和一次原有 `StereoReconstructionKLLoss`；没有联合 loss、mode 权重或随机路由。
- LPIPS 与 image GAN 可用于两种模式；video GAN 只允许 four，single 不调用 video discriminator。
- validation 不交替；每个固定 validation batch 分别运行 four 与 single，记录 `val/four/*`、`val/single/*`。
- best checkpoint 暂按 `val/four/total_loss` 排序。

## Counter 和 checkpoint

checkpoint 保存并恢复：

- `generator_updates`；
- `discriminator_updates`；
- `four_frame_updates`；
- `single_frame_updates`；
- `batch_updates`。

加载时 fail closed 校验：

```text
generator_updates == four_frame_updates + single_frame_updates
four_frame_updates == (generator_updates + 1) // 2
single_frame_updates == generator_updates // 2
```

缺少 temporal-mode counters 的旧 checkpoint 不进行推断式恢复。新增 single Encoder/Decoder 参数也意味着本结构应从头训练。

## 日志

训练 batch 只写实际运行模式，不为未运行模式写零 loss：

- `train/four/total_loss|rgb_loss|disparity_loss|gradient_loss|kl_loss`；
- `train/single/total_loss|rgb_loss|disparity_loss|gradient_loss|kl_loss`；
- `train/generator_updates`、`train/four_frame_updates`、`train/single_frame_updates`、`train/batch_updates`。

validation 每个 batch 同时写 `val/four/*` 与 `val/single/*`。

## 训练预算语义

代码没有自动修改 batch size、learning rate、max steps、warmup 或 scheduler。若原计划是 100k four-only updates，而 1:1 交替后仍保持 100k total updates，则 four/single 各约 50k。若未来要保留 100k four updates，可把总训练量提高到约 200k，并另行重新确认 warmup 与 scheduler horizon；这不是本次代码的自动行为。

## 修改文件

- `stereo_tokenizer/model.py`：模式合同、双 temporal branch、交替训练、GAN 路由、计数器、分模式日志与 validation。
- `stereo_tokenizer/__init__.py`：导出 `StereoEncodeOutput`、`EyeMode`、`TemporalMode`，不导出 Memory role。
- `train_stereo_vae.py`：校验 single source index，best checkpoint 改为 `val/four/total_loss`，并提供显式 `--resume_from_checkpoint` 入口以恢复 optimizer、scheduler 和模式计数器。
- `scripts/stereo/train_stereo_vae.sh`：要求并传入 `SINGLE_FRAME_SOURCE_INDEX`。
- `eval_stereo_vae.py`：显式选择 eye/temporal mode，single 评估同步截取输入与 GT，输出 JSON 记录模式。
- `tests/stereo/`：补充模式路由、T=1/T=4 shape、跳过四帧模块、梯度、交替、accumulation、DDP 确定性语义、checkpoint resume 和 loss T=1 测试。
- `doc/Stereo Tokenizer Plan.md`、`README.md`：同步架构、训练和 Memory 边界。

## 当前验证

- `python -m compileall -q stereo_tokenizer train_stereo_vae.py eval_stereo_vae.py tests`：通过。
- `python -m unittest tests.stereo.test_source_boundary tests.stereo.test_entrypoints_source`：通过，16 tests。
- `python -m unittest discover -s tests/stereo -p 'test_*.py'`：23 个已发现项中，18 个无 Torch 测试通过；5 个 tensor test module 在导入阶段统一因本机缺少 `torch` 报错，未进入测试逻辑。
- `git diff --check`：通过；仅报告 Windows 工作树未来 LF→CRLF 转换提示，无 whitespace error。
- 当前 Windows Python 缺少 `torch`，因此 tensor forward/backward、完整 `tests/stereo` 和 strict state-dict 动态测试尚未完成，不能记为通过。
- 未获授权连接 H200，本次未进行远端动态验证或任何 GPU 任务。

## 后续 Gate

用户先审查本地 diff。审查通过后，如需 commit/push、服务器同步、H200 动态单测或 smoke，必须在当前对话中另行明确授权，并按仓库 Git/H200 流程执行。
