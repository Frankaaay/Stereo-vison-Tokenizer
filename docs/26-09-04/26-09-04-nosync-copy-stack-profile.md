# GA2 no-sync 与 copy/cat 源码栈实验

## 目的与合同

- 验证 manual optimization 下 stereo/four GA2 第一 microbatch 的 DDP 同步是否值得消除，并用带 Python stack 的 operator trace 定位 `copy_`、`cat`、`contiguous`。
- 分支 `hezhou-las2-h`；实验 SHA `605a0bd176c7d4678bdee051d18a47cbe8c9ca2f`。
- H100 Slurm 单节点 8×H100；batch `24:24:24:12`、GA `1:1:1:2`、四模式有效 global batch 均为 192；在线 DA3/LAS2-H，cache/GAN/W&B/media 关闭。
- 吞吐按 sync A → no-sync A → no-sync B → sync B 在同一节点顺序运行；每 arm 121 logical updates，每 mode 跳过前 4 次，共 105 个稳定 updates。sync/no-sync 各合并 210 个稳定 updates。
- stack arm 为 no-sync、21 logical updates，profiler wait/warmup/active=`4/4/8`，开启 shape、memory、Python stack。

## 作业与产物

- 单元测试 Job `3286`：`COMPLETED 0:0`，24 passed。
- 8 卡矩阵 Job `3288`：`COMPLETED 0:0`，elapsed `00:44:47`；五个 arm 独立退出码均为 0，stack arm validation 20/20，`val/mixed/total_loss=1.360`。
- 汇总解析 Job `3302`：`COMPLETED 0:0`，elapsed `00:05:26`。
- native-parent 初版 Job `3304` 因临时解析器 O(N²) 主动取消；优化版 Job `3307` 因资源预计排到 9 月 11 日，在运行前取消。二者均未修改原始产物。
- 主日志：`/gpfs/jiuquyun/home/Frank/logs/nosync-stack-ab-605a0bd-3288.out`。
- 输出：`/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/nosync-stack-ab-605a0bd-3288`。
- 关键产物：`analysis.json`、各 arm `step_timing.json`/checkpoint/config/manifest/telemetry，以及 `nosync_stack_profile/profiler/rank0-trace-00.json.gz`（解压约 4.54 GB）。

## no-sync 吞吐结果

| 组 | 两次 arm samples/s | pooled samples/s | stereo/four samples/s | 峰值 allocated |
|---|---|---:|---:|---:|
| sync | 131.311 / 131.725 | 131.518 | 79.136 | 75.262 GiB |
| no-sync | 131.820 / 131.853 | 131.836 | 79.798 | 75.262 GiB |

- no-sync 相对提升：总体 `+0.24%`，目标 stereo/four `+0.84%`。
- 负控模式没有一致改善：mono/single `-0.76%`、mono/four `-0.21%`、stereo/single `+0.54%`，说明总体差异仍在节点/数据抖动量级。
- 两对 step-121 checkpoint 均产生相同的参数差异统计：72,020,474 个元素中 max abs `0.00811`、mean abs `3.03e-4`。这可由浮点归约顺序长期放大造成，但不能用作严格等价证明。
- 结论：DDP no-sync 方向正确，但当前端到端收益太小，不值得增加生产分支和数值轨迹变化；实验开关与实现均移除。

## copy/cat/contiguous 源码归因

stack trace 的 top operator 中，`aten::copy_` 为约 `85.81 ms/observed step`，`aten::cat` 约 `32.48 ms/observed step`。源码栈进一步表明，能直接映射到仓库 Python forward 的 copy 热点主要是：

| 源码位置 | 10 个 active microsteps 的 copy GPU 时间 | 约每 microstep |
|---|---:|---:|
| `lpips.py:170 normalize_tensor` | 66.11 ms | 6.61 ms |
| `lpips.py:90 distance_from_normalized_features` | 21.47 ms | 2.15 ms |
| `attention.py:349 forward` | 19.43 ms | 1.94 ms |
| `attention.py:57 apply_rotary_emb` | 13.42 ms | 1.34 ms |
| `attention.py:182 forward` | 9.39 ms | 0.94 ms |
| `stereo_fusion.py:105 forward` | 7.33 ms | 0.73 ms |

- LPIPS forward 还出现 4,266 次 `_to_copy`，主要 shape 为冻结 VGG/1×1 权重；这与 bf16 autocast 下冻结 FP32 权重重复转换一致，是下一项最明确的可优化 copy。
- `window_partition/window_reverse` 的显式 contiguous/clone copy 合计仅约 3.44 ms/10 microsteps，优先级低。
- 能映射到训练 Python forward 的显式 `cat` 只有 8 次、GPU 约 0.02 ms；top-level `cat` 大头来自 autograd/native 或其他上下文，不能直接删除 attention 中的 `torch.cat`。

## 下一步

不修改 attention layout 或 `cat`。只对冻结 LPIPS 权重一次性转 BF16 做同节点 A/B，保持输入、loss reduction、batch、GA 和在线 teacher 不变，同时比较吞吐、峰值显存、LPIPS 数值和 checkpoint 健康度。

## LPIPS BF16 A/B（进行中）

- 实验变量仅为冻结 LPIPS 参数及 buffer 在构造时从 FP32 一次性转为 BF16；baseline 继续由 autocast 在每次 forward 中转换冻结权重。
- 使用临时 `PERCEPTUAL_MODEL_BF16=0/1` 开关，在同一实验 SHA、同一节点上按 FP32 A → BF16 A → BF16 B → FP32 B 顺序运行。
- 运行前先通过单元测试、shell 语法检查，以及固定输入下 LPIPS loss/输入梯度的 FP32-weight-autocast 与 BF16-weight-autocast 数值对照。
- 端到端合同沿用 batch `24:24:24:12`、GA `1:1:1:2`、在线 DA3/LAS2-H；记录 pooled/per-mode samples/s、峰值显存、validation、checkpoint 和作业退出状态。
