# Single/Four StereoVAE 与 online FoundationStereo profiling 合并

## 目的与 Git 基线

- 日期：2026-08-24。
- 合并分支：`merged-fs-vae-single-four-profiling`。
- 模型基线：`origin/frank@9662035322ff98014c53eb4f55c27c3be704ad49`，已包含 PR #3 四帧双向 temporal attention 和 PR #4 single/four-frame 训练合同。
- 能力来源：`frank-profiling@7790abb00249a3f31d21a6cb9ddca3ad1edf493f`。
- 本次结果：当前 merge commit；未创建额外 worktree，未同步服务器，未启动 GPU 任务。

## 保留的模型与训练语义

- `four_frame` 严格使用 `T=4`，执行四帧 learned position、双向 temporal attention，再通过 `4D -> D` sampler 形成一个 latent slot。
- `single_frame` 严格使用 `T=1`，共享 Spatial Encoder 和 StereoFusion，跳过四帧 temporal 模块并使用独立 `D -> D` projection/expansion。
- 训练模式继续由 `generator_updates` 以 four/single 交替决定；gradient accumulation 窗口内模式不变。
- checkpoint 继续严格保存并恢复 generator/discriminator、four/single 和 batch counters；训练入口保留 `--resume_from_checkpoint`。
- 正式 launcher 不启用 causal temporal attention。

## 合入的 profiling 与 online FS 能力

- Spatial PEG 支持 `conv2d_t1_slice`，正式 launcher 启用该后端；four-frame temporal Transformer 本身 `peg=False`，Spatial Encoder/Decoder 的 PEG 输入仍为 `T=1`。
- DataLoader 启用 pinned memory 和 persistent workers；离线 Manifest 可显式使用 `train_epoch_repeats`，LeRobot online backend 固定 repeats=1。
- 保留 H2D、Encoder、StereoFusion、single/four temporal branch、Decoder、RGB/disparity/gradient/KL/LPIPS、backward、gradient clipping、Adam、scheduler 和 logging 的 opt-in profiling regions。
- step timing 现在记录每个 update 的 `temporal_mode`，并分别汇总 single/four 稳态时间。
- 加入 LeRobot episode 数据链路、rectification/manifest/teacher-selection 工具，以及冻结的双向 FoundationStereo online teacher。
- online 训练顺序保持为完整 `T=4` batch 先生成 disparity/valid mask，随后 VAE 同步选择 four 或截取 single frame，再执行 forward、loss、backward 和 optimizer。online cache 默认关闭。

## 不沿用的旧 profiling 假设

- 不使用 profiling 分支的旧 four-only temporal 编解码结构。
- 不向正式 launcher 传 `--causal_in_temporal_transformer`。
- single/four 交替会改变每个 update 的已使用参数集合，因此多卡策略使用 `static_graph=False` 和 `find_unused_parameters=True`，不沿用旧 four-only 的 static DDP graph。
- best checkpoint 继续监控 `val/four/total_loss`，不使用旧指标名 `val/total_loss`。
- 旧 profile 工具的精确 14-PEG 断言被移除；新工具要求至少一个 Spatial PEG，并分别记录 single/four timing。
- alternating profile 禁止共享固定 LPIPS GT feature cache，因为 single/four 的 target frame 数不同；该 cache 不进入正式训练。
- profiling 分支上旧模型的吞吐数字只作为历史证据，不能表示合并后 single/four 模型的速度。

## 本地验证

- `python -m py_compile`：合并后的 model、training entry、profiling entry、data、online teacher、LeRobot data 和相关测试文件通过。
- `python -m unittest tests.stereo.test_source_boundary tests.stereo.test_entrypoints_source`：24/24 通过。
- `git diff --check`：通过；仅有 Windows LF/CRLF 提示。
- Windows 当前 Python 缺少 Torch，因此 tensor forward/backward、PEG 动态路径和 online teacher 动态合同未在本地执行。

## 尚未执行的 H200 Gate

1. CPU/CUDA 动态测试：single/four shape、梯度、temporal attention 顺序、PEG T=1 fail-closed、online T=4 GT 与 single source index 对齐、checkpoint strict resume。
2. 单 GPU 四个 optimizer updates：`four -> single -> four -> single`，覆盖 online FS、VAE、finite losses、显存和 checkpoint。

## H200-2 BS24 在线链路测试（准备中）

- 用户已授权将第三分支 push 并在 H200-2 上进行 GPU 测试；目标为每卡
  BS24、八卡 global batch 192、BF16、online FoundationStereo 32 iterations、
  pair microbatch 48、cache off、GA=1。
- 目标分支已 push 为 `merged-fs-vae-single-four-profiling`，同步时精确 SHA 为
  `027c253f3114f95139905e7263a7dab1bff1c497`；H200-2 已切换到该分支并保持 clean。
- H200-2 同时存在 `melody` 的 NGADv1pp eval：GPU 1/3/6 各占约 14.3 GiB，
  GPU4 约 5.7 GiB，且部分卡有约 27--32% 计算利用率。用户明确允许共卡测试；
  因此本次 wall time 必须标记为共享 GPU 条件，不能视为独占卡吞吐基准。
- 主训练入口原先只输出完整 step timing，细粒度 regions 仅在旧 cached-GT
  单卡 profiler 中启用。为准确测量 online FS -> VAE 串行链路，新增默认关闭的
  rank0-only PyTorch profiler 参数；它不改变默认训练数学，运行时输出 data/H2D、
  FoundationStereo、encoder/single-four temporal/decoder、loss、backward、Adam、
  operator 和 Chrome trace。八个 rank 的完整 step timing 仍由原 callback 输出。
- 计划先运行定向 Torch/合同测试与四步交替 smoke，再运行 15 updates 的八卡
  BS24 profiling（5 wait、2 warmup、4 active，并保留 post-profile updates）。结果、
  首个异常、显存、吞吐和瓶颈将在本节完成后补记。
3. 八 GPU 动态 DDP smoke：确认 changing unused-parameter set 不报 reducer/NCCL 错误，各 rank mode counters 一致。
4. 合并后重新测量 BS 和 single/four 分模式吞吐；不得把旧模型结果直接升级为正式 recipe。

上述 H200 测试、服务器同步、commit push 和正式训练均需按授权边界单独执行。
