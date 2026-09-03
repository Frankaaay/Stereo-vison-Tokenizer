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

### timing 重跑与 launcher 门禁

- Hy 修复 SHA `ad717d57b9315a92a5c04a3a6488703947970833` 在 H100 锁定 runtime 通过 `tests.stereo.test_hy_mono_data` 的 5 个测试。
- timing Job `2535`：Slurm `COMPLETED (0:0)`，4 个逻辑 update、5 个 micro-batch、20 个 validation batch，生成三个 step-4 checkpoint 与 `step_timing.json`；但产物自证 `world_size=1`，日志也显示 `Starting with 1 processes`。该作业虽然申请 8 GPU，实际仅单进程运行，不能作为 8 卡显存或吞吐结论。
- 根因是 H100 Slurm allocation 使用一个 task，`DISTRIBUTED_MODE=single` launcher 又直接调用 `python3`；Lightning 采用 Slurm world size 1，没有自行拉起 8 个 rank。
- 单节点多卡 `single` launcher 改为 `torchrun --standalone --nnodes 1 --nproc_per_node GPU_COUNT`；单卡仍直接使用 `python3`，双节点 `ib` 路径不变。后续 smoke 必须同时满足 Job exit 0、`step_timing.json world_size=8`、checkpoint 直接 counters 和 8 个 rank 的显存记录。

## 训练数据规模

- Hy：211,381 条 train manifest records；16,368,717 个 train windows，每个 window 有 3 个 mono camera variant，因此 DataLoader 长度为 49,106,151 samples。全 manifest 为 215,577 records、16,703,900 windows。
- LIBERO：1,684 条 train records；33,564 个 train windows，每个 window 有 2 个 mono camera variant，因此 DataLoader 长度为 67,128 samples。全 manifest 为 1,712 records、34,192 windows。
- UMI：81,141 个 train episodes、1,494,802 个 train stereo samples；每个 sample 同时包含 head、left wrist、right wrist 三组双目。全 manifest 为 90,157 episodes、1,661,796 samples，另有 val 4,507 episodes/83,027 samples 和 test 4,509 episodes/83,967 samples。
- 三个 train DataLoader 的表面长度合计为 50,668,081 samples，但单个 Hy/LIBERO sample 是一路 mono，单个 UMI sample 是三组双目，不能按此总数直接比较像素或相机帧量。实际 sampler 使用四模式权重 `1:1:1:1`、mono 数据源权重 Hy:LIBERO=`1:1`，不会按原始数据量比例抽样。

## 真实 8 卡 smoke 结果

- launcher 修复 SHA 为 `be0c7a25dc2842f752df674566bc7d76727682af`；H100 锁定 runtime 通过 8 个 launcher/distributed 定向测试及 `bash -n`。
- Job `2538` 使用单节点 8×H100，最终 `COMPLETED (0:0)`，elapsed `00:04:57`。日志确认 `GLOBAL_RANK 0..7`、`LOCAL_RANK 0..7`、`Starting with 8 processes`，且没有 Traceback、RuntimeError、ValueError、OOM 或 kill。
- 输出目录为 `/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/h100-canonical-smoke4-be0c7a2-timing-v1`，Slurm 日志为 `/gpfs/jiuquyun/home/Frank/logs/stereo-smoke8-be0c7a2-2538.out`。
- `last.ckpt` 直接读取结果：`generator_updates=4`、`batch_updates=5`、`discriminator_updates=0`、`single_frame_updates=2`、`four_frame_updates=2`、`world_size_contract=8`。四个模式各完成 1 个 logical update、各消费 192 samples，合计 768 samples；这与 BS `24:24:24:12`、GA `1:1:1:2` 的有效全局 batch 192 一致。
- 四模式逻辑 update 速度：mono/single `0.395 s, 485.64 samples/s`；mono/four `0.597 s, 321.75 samples/s`；stereo/single `0.544 s, 353.12 samples/s`；stereo/four `3.599 s, 53.35 samples/s`。训练 update 区间合计 5.135 秒，对应聚合约 149.57 samples/s，不含进程启动、模型/数据初始化、validation 和 checkpoint。
- 8 个 rank 中的最大 CUDA allocated/reserved：mono/single `8.190/58.607 GiB`；mono/four `33.172/65.080 GiB`；stereo/single `23.886/65.080 GiB`；stereo/four `54.021/58.607 GiB`。全程最大 reserved 为 `65.080 GiB/GPU`，相对 80 GiB H100 名义容量余量约 `14.92 GiB`，未发生 OOM。reserved 会保留此前模式的 allocator cache，模式工作峰值应优先看 allocated。
- 每个模式仅测 1 个 logical update 且 warmup 为 0；结果足以证明四模式容量、DDP、数据和 checkpoint 闭环，但属于冷启动 smoke，不作为稳态吞吐 benchmark。stereo/four 的首个 micro-batch 明显包含冷启动成本，其两个 micro-batch 分别为 36.69 和 99.60 samples/s。
- 产物包括 `best-epoch=0-step=4.ckpt`、`epoch=0-step=4.ckpt`、`last.ckpt`、`resolved_config.json`、`run_manifest.json` 和 `step_timing.json`，均非空。

## 8 卡 BS24/12 与 BS30/15 稳态 A/B

- A/B 使用同一 clean SHA `5c7e638d5d4f446dc8140cd632b7b7780351e114`（相对训练代码 SHA `be0c7a25dc2842f752df674566bc7d76727682af` 只增加文档）、seed 1234、数据、teacher、持久化 VGG 和四模式 `1:1:1:1` 调度；分别在 `xn01-gpu1-0049` 与 `xn01-gpu1-0050` 并行运行。
- 基线 Job `2553`：BS `24:24:24:12`、GA `1:1:1:2`，`COMPLETED (0:0)`，640 logical updates/800 micro-batches/122,880 samples，纯训练 timing 1,050.52 秒（17.51 分钟），Slurm elapsed 22:23。
- 候选 Job `2554`：BS `30:30:30:15`、GA `1:1:1:2`，`COMPLETED (0:0)`，480 logical updates/600 micro-batches/115,200 samples，纯训练 timing 1,190.62 秒（19.84 分钟），Slurm elapsed 24:45。两者均确认 8 ranks，无 OOM、Traceback、RuntimeError、ValueError、残留作业或空产物。
- 每模式先丢弃 64 个 warmup update 后，基线稳定窗口为 384 updates/73,728 samples/621.84 秒，前后半段聚合吞吐为 116.88/120.30 samples/s；候选稳定窗口为 224 updates/53,760 samples/557.35 秒，前后半段为 97.54/95.40 samples/s，均已进入相对稳定区间。
- 等模式、等样本量窗口各为 53,760 samples：基线 119.47 samples/s，候选 96.46 samples/s，候选下降 19.27%。分模式变化为 mono/single `+3.19%`、mono/four `-17.00%`、stereo/single `+11.68%`、stereo/four `-32.47%`；最重模式的退化抵消了单帧收益。
- 8 ranks 最大 allocated/reserved（GiB）：基线 mono/single `8.194/74.889`、mono/four `33.296/76.451`、stereo/single `23.886/74.889`、stereo/four `54.437/74.889`；候选依次为 `9.859/73.607`、`41.209/76.975`、`29.453/73.607`、`67.582/73.607`。候选 stereo/four allocated 增加 13.145 GiB，且全局最大 reserved 达 76.975 GiB。
- 结论：`30:30:30:15 / 1:1:1:2` 可以完成长于 15 分钟的 8 卡训练，不会立即 OOM；但显存余量明显收窄，等样本吞吐下降 19.27%，stereo/four 单项下降 32.47%，不建议作为当前四模式默认值。保留 `24:24:24:12 / 1:1:1:2` 更合适。
- 输出目录：`/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/h100-ab-bs24-20260903-2553` 与 `/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/h100-ab-bs30-20260903-2554`；提交脚本和分析脚本保存在 `/gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/benchmarks-20260903`。
