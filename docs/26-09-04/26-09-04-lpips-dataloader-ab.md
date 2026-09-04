# LPIPS 与 DataLoader 吞吐 A/B

## 目的与约束

- 目的：在 joint-view 四模式训练合同不变时，实测 LPIPS frame microbatch、channels-last、`torch.compile` 与 DataLoader workers 对吞吐、尾延迟和显存的影响。
- 用户确认 teacher 全量 cache 因数据规模与空间成本不可行，因此本轮及后续优化结论排除 cache 路线，DA3 与 LAS2-H 均保持在线推理。
- 分支：`hezhou-las2-h`；实验代码 SHA：`e4e6e00b11afb8205f8f13983e0bf72643db6fa6`。
- 位置：H100 Slurm，单节点 8×H100 80GB；Job `3277`。
- 四模式 batch：`24:24:24:12`；梯度累积：`1:1:1:2`；每个 mode 有效 global batch 均为 192。
- 每个 arm 运行 81 logical generator updates；每个 mode 跳过前 4 次 warmup 后统计，共 65 个稳定 logical updates。比较使用 `sum(samples) / sum(interval)`，不用 step/s 或累计均值。
- GAN/W&B/media/cache 关闭；在线 DA3/LAS2-H、数据、seed、mode schedule 和验证合同固定。

## 作业与产物

- Slurm Job `3277`：`COMPLETED`、作业层 exit code `0:0`、elapsed `00:38:22`。矩阵脚本按 arm 捕获退出码，因此必须以各 arm 的 `exit_code.txt` 验收。
- 日志：`/gpfs/jiuquyun/home/Frank/logs/lpips-loader-ab-e4e6e00-3277.out`。
- 输出：`/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/lpips-loader-ab-e4e6e00-3277`。
- 汇总：输出根目录 `comparison.json`；每个 arm 保存 `resolved_config.json`、`run_manifest.json`、`step_timing.json`、1 秒 GPU telemetry 和独立退出码；成功 arm 另有 step-81 checkpoints。
- baseline checkpoint 直读：`global_step=81`、`generator_updates=81`、`batch_updates=101`；mode updates `21/20/20/20`，mode samples `4032/3840/3840/3840`，四模式有效 global batch 均为 192。

## 稳定窗口结果

| arm | workers | LPIPS 设置 | exit | aggregate samples/s | 相对 baseline | 峰值 allocated |
|---|---:|---|---:|---:|---:|---:|
| baseline | 4 | chunk 24 | 0 | 135.153 | — | 75.578 GiB |
| DataLoader | 8 | chunk 24 | 0 | 136.014 | +0.64% | 75.578 GiB |
| chunk 36 | 4 | chunk 36 | 0 | 134.947 | -0.15% | 76.420 GiB |
| chunk 48 | 4 | chunk 48 | 1 | 无有效窗口 | 淘汰 | OOM 前约 77.22 GiB allocated |
| channels-last | 4 | chunk 24 + channels-last | 0 | 137.436 | +1.69% | 75.581 GiB |
| compile | 4 | chunk 24 + `torch.compile` | 1 | 0 updates | 淘汰 | 未进入训练 |

### 各模式 aggregate samples/s

| arm | mono/single | mono/four | stereo/single | stereo/four |
|---|---:|---:|---:|---:|
| baseline | 286.710 | 90.340 | 298.353 | 83.811 |
| workers 8 | 326.108 | 100.433 | 244.008 | 79.649 |
| chunk 36 | 283.787 | 89.877 | 295.246 | 84.409 |
| channels-last | 284.469 | 92.790 | 300.704 | 85.312 |

workers 8 的总体增益仅 0.64%，且并不一致：mono 两种模式改善，但 stereo/single 与 stereo/four 分别下降约 18.2% 和 5.0%。其 stereo/single p90 interval 从 `1.100 s` 恶化到 `1.705 s`，stereo/four max 从 `3.178 s` 恶化到 `4.332 s`，不能认为它解决了 starvation。

channels-last 四模式中三种改善，但 mono/single 略降；单轮总体增益 1.69%，峰值显存基本不变。该幅度接近短窗口噪声，只能作为复测候选，不能据此默认开启。

## 失败根因

- chunk 48：8 个 rank 一致 CUDA OOM。首批异常显示每卡总容量 79.18 GiB，仅余约 36.06 MiB；PyTorch 已 allocated 77.22 GiB，再申请 48 MiB 失败。没有稳定窗口或 checkpoint。
- compile：第 0 step 首次 Inductor/Triton 编译失败。Triton 调用 `/usr/bin/gcc` 构建 `cuda_utils`，命令返回非零并触发 `torch._inductor.exc.InductorError`；GPU 利用率在编译阶段为 0%，没有进入训练。当前 H100 runtime 下不可用，且 8 rank 重复编译有明显启动成本。

## 结论与下一步

1. 不提高 LPIPS chunk：36 没有吞吐收益且增加约 0.84 GiB 峰值显存，48 明确 OOM。
2. 不把 workers 从 4 改为 8：总增益低且 stereo 尾延迟恶化。下一步若优化数据路径，应直接 profile stereo 数据解码、collate、`copy_/cat/contiguous`，而不是继续盲加 worker。
3. 不在当前 runtime 启用 `torch.compile`；若未来专门修环境，必须先解决 Triton C helper 编译，再单卡预热/共享 cache，随后重新做 8 卡完整门禁。
4. channels-last 是本轮唯一正向候选，但 +1.69% 尚不足以合入。若要继续，采用 baseline/channels-last 交错至少 3 次、扩大稳定窗口，并以置信区间和各模式一致性验收。
5. teacher cache 因空间约束永久排除后，端到端最大项仍是在线 teacher（旧 profile cycle 约 41%）和 training backward（training CUDA 约 59%）。可落地的下一轮应优先减少 teacher 在线计算量/分辨率/迭代次数的同质量 A/B，或针对 backward 中的卷积、`copy_`、`cat` 做算子级优化；这些都必须先冻结质量容差，不能只测速度。

本轮实验开关在结论产出后从生产路径移除，避免保留失败或未证实的配置入口；正式训练仍使用实验前的单一路径。
