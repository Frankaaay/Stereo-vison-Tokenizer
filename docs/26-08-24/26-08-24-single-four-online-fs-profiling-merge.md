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

## H200 合并后验证与 BS32 profiling（进行中）

- 第三分支已 push 为 `merged-fs-vae-single-four-profiling`。H200-1 的本轮
  精确运行 SHA 为 `e93a7aaf8b4dad7b3b54c03e7f4f4e56656fe1b8`，同步后
  branch/upstream/HEAD 一致且 worktree clean。
- 主训练入口原先只输出完整 step timing，细粒度 regions 仅在旧 cached-GT
  单卡 profiler 中启用。为准确测量 online FS -> VAE 串行链路，新增默认关闭的
  rank0-only PyTorch profiler 参数；它不改变默认训练数学，运行时输出 data/H2D、
  FoundationStereo、encoder/single-four temporal/decoder、loss、backward、Adam、
  operator 和 Chrome trace。八个 rank 的完整 step timing 仍由原 callback 输出。
- 本地验证：24/24 source/entrypoint tests、compileall 和 diff-check 通过。H200-1
  使用 Python 3.12、Torch 2.7.1+cu128、Lightning 2.5.6，在 CUDA hidden 下运行
  53 个 encoder、decoder、loss、StereoVAE 和 entrypoint 定向测试，全部通过。
- H200-1 运行前八张 H200 均为零 compute process、零显存占用。FoundationStereo
  repo 为 clean `6e880681`；checkpoint SHA256 为
  `60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`。
  使用 H200-1 full manifest，共 1,384,393 个 four-frame samples。
- BS32 输出为
  `/data/home/frank/experiments/stereo_merged_fs_vae_bs32_profile_20260824_v1`。
  它在第一个 optimizer update（初始 `four_frame`）的 LPIPS convolution 失败；八个
  rank 均 OOM，每卡 PyTorch 已分配约 136.18 GiB、仅余约 0.5--0.7 GiB，却仍需
  申请 3.00 GiB。profiler 当时仍处于 wait 阶段，因此不是 active trace 的额外显存；
  没有完成任何 update，退出码为 1，所有 rank 已退出且显存恢复为零。
- 用户随后要求降到 BS24。新测试在 tmux `stereo-merged-bs24-profile-260824`
  进行中，输出
  `/data/home/frank/experiments/stereo_merged_fs_vae_bs24_profile_20260824_v1`。
  除每卡 BS24/global192 外，其余配置保持不变：8 GPU、GA=1、BF16、15 updates、
  single source index 0、online FS 32 iterations、pair microbatch 48、cache off；
  timing 丢弃前五步，rank0 profiler 为 wait 5、warmup 2、active 4。
- BS24 启动健康检查通过：tmux、主进程和八个 DDP rank 均存在，日志进入 BF16/DDP
  初始化，没有 traceback 或 OOM；检查时尚未产生第一个 update。初始 ETA 为
  7--12 分钟完成主体，另需约 2--3 分钟验证 timing、trace、显存、loss、checkpoint
  和 strict resume。
- 完成后必须核对：`four -> single` 交替与 counters、finite loss、动态 DDP、
  single/four 分模式 step time、各 named region、peak allocated/reserved、checkpoint
  写入和 strict resume。不得把旧模型的 BS32 吞吐直接当作本次结果。
