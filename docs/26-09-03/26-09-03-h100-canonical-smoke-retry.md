# H100 canonical 8 卡 smoke VGG 缓存重试

## 目的与运行合同

- 复用既有 H100 train runtime、三份 canonical manifest 与 teacher 资产，重试 8 卡四模式训练 smoke。
- 运行源码为 `hezhou-las2-h@d7d5ca61acc19432300ccd7ef67f11494c481493`，H100 clone clean 且与 origin 同步。
- 单节点 8×H100；四模式 batch 为 `24:24:24:12`，梯度累积为 `1:1:1:2`，每个逻辑 update 的全局有效 batch 均为 192；共运行 4 个逻辑 update。

## VGG 缓存与重试记录

- 持久化 VGG16 文件为 `/gpfs/jiuquyun/checkpoints/Frank/stereo-vae/runtime-assets/torch/hub/checkpoints/vgg16-397923af.pth`，大小 553,433,881 字节，SHA256 为 `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`。
- 前序 Job `2269` 在 update 0 前失败：LPIPS 从公网向节点 `/local` 下载 VGG16，所得文件不完整，哈希校验失败。
- 首次重提 Job `2481` 虽通过 `--export` 传入 `TORCH_HOME`，运行时仍继承为节点 `/local`；启动健康检查发现重复下载后于 36 秒主动取消，未进入训练 update。
- 第二次重提在 Slurm `--wrap` 内显式导出持久化 `TORCH_HOME`，`sbatch --test-only` 通过；正式 Job ID 为 `2483`，QOS `debug`，申请 8 GPU、64 CPU、512 GiB、1 小时。

## Hy timestamp 根因与修复

- 按 seed 1234 和 8 rank 精确还原 Job `2483` 的失败批次：两个 `stereo/four_frame` micro-batch 完成第 1 个逻辑 update，`mono/single_frame` LIBERO 完成第 2 个，随后在 `mono/four_frame` Hy 取数时失败。
- 对失败 Hy batch 的实际 Lance rows 做只读审计，部分 episode 使用 `timestamp ~= frame_index / fps`，另一些使用 `timestamp ~= (frame_index + 1) / fps`。失败样本均为后一种严格 1-based 帧时钟；相对时间间隔误差最大约 3.1 微秒，不是随机漂移或 episode/frame 错配。
- `HyLanceMonoDataset._timestamps_match_frame_rate` 现在仅接受上述 0-based 或 1-based 精确帧网格，继续拒绝非有限值、shape 不同、无效 FPS、半帧偏移和任意漂移。定向测试增加真实 1-based float32 样本及半帧反例。

## 当前状态

- Job `2483` 启动于节点 `xn01-gpu1-0057`。日志明确打印持久化 `TORCH_HOME`，未再次下载 VGG；LPIPS、bf16、CUDA、NCCL、8 卡可见性和模型初始化均已通过。
- 输出目录：`/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/h100-canonical-smoke4-d7d5ca6-v3`。
- Slurm 日志：`/gpfs/jiuquyun/home/Frank/logs/stereo-smoke8-d7d5ca6-v3-2483.out`。
- 最终状态为 `FAILED (1:0)`，elapsed 1 分 38 秒；训练进度显示到 `3/4`，在取后续 Hy batch 时由 `HyLanceMonoDataset.get_mode_item` 抛出 `ValueError: Hy timestamps disagree with frame_index/fps`。未生成 checkpoint 或指标文件，不能用进度条替代直接 checkpoint counter 证明已完成 3 个逻辑 update。
- 失败前进度条的最后累计速率为约 0.66 logical update/s；按每个逻辑 update 192 samples 折算约 127 samples/s。样本过少且混合不同模式，这不是稳态或分模式吞吐。
- Slurm accounting 记录 `gres/gpumem=2330M`、`gres/gpuutil=14`；训练未完整覆盖四模式且没有 Torch per-rank peak-memory 埋点，因此 2,330 MiB 只能作为本次失败短跑的 accounting 观测，不能作为 BS `24:24:24:12` 的可信峰值显存或容量结论。

## 训练数据规模

- Hy：211,381 条 train manifest records；16,368,717 个 train windows，每个 window 有 3 个 mono camera variant，因此 DataLoader 长度为 49,106,151 samples。全 manifest 为 215,577 records、16,703,900 windows。
- LIBERO：1,684 条 train records；33,564 个 train windows，每个 window 有 2 个 mono camera variant，因此 DataLoader 长度为 67,128 samples。全 manifest 为 1,712 records、34,192 windows。
- UMI：81,141 个 train episodes、1,494,802 个 train stereo samples；每个 sample 同时包含 head、left wrist、right wrist 三组双目。全 manifest 为 90,157 episodes、1,661,796 samples，另有 val 4,507 episodes/83,027 samples 和 test 4,509 episodes/83,967 samples。
- 三个 train DataLoader 的表面长度合计为 50,668,081 samples，但单个 Hy/LIBERO sample 是一路 mono，单个 UMI sample 是三组双目，不能按此总数直接比较像素或相机帧量。实际 sampler 使用四模式权重 `1:1:1:1`、mono 数据源权重 Hy:LIBERO=`1:1`，不会按原始数据量比例抽样。
