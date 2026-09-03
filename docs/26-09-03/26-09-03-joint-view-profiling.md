# Joint-view 四模式训练 profiling

## 目的与合同

- 目的：在 mono 多视角联合 encoder 后，对当前四模式训练做阶段归因，确定速度优化优先级
- 分支：`hezhou-las2-h`
- 运行 SHA：`e0b0964b8a0503deb51cd5416cadf4004befcdd5`
- 位置：H100 Slurm，单节点 8×H100 80GB
- 模式 batch：`mono/single=24`、`mono/four=24`、`stereo/single=24`、`stereo/four=12`
- 梯度累积：`1:1:1:2`；四种 mode 的有效 global batch 均为 192
- GAN/W&B/media/cache：关闭；DA3 与 LAS2-H 均在线推理
- profiler schedule：wait 4、warmup 4、active 32、post-profile 1，共 41 logical updates；active steps 9–40 中四种 mode 各 8 次
- 埋点：DataLoader、H2D、DA3/LAS2-H teacher、encoder 子模块、decoder 子模块、各 loss、backward、gradient clipping、Adam、logging；另记录 step timing、峰值显存和 1 秒 GPU telemetry

## 作业与产物

- profiling Job `2922`：`COMPLETED`，exit code `0:0`，elapsed `00:14:33`
- CPU trace 解析 Job `2933`：`COMPLETED`，exit code `0:0`，32 active steps 完整，四种 mode 各 8 次，613,420 个 device events 全部映射成功
- 错误解析 Job `2931`：远端 shell 提前展开 `${root}` 导致参数为空，发现后立即取消；未修改 profiling 原始产物
- 日志：`/gpfs/jiuquyun/home/Frank/logs/joint-view-profile-e0b0964-2922.out`
- 输出：`/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/h100-joint-view-profile-e0b0964-2922/bs24`
- 关键文件：`step_timing.json`、`gpu_telemetry.csv`、`profile-analysis.json`、`rank0-trace-00.json.gz`、regions/top-operators JSON、三个 step-41 checkpoint、resolved config 与 run manifest
- checkpoint 直读：`global_step=41`、`generator_updates=41`、`batch_updates=51`；mode updates 为 `10/11/10/10`，每个完整 active profile window 均为 `8/8/8/8`
- validation 20/20 完成，`val/mixed/total_loss=1.240`；无 traceback/OOM，作业已释放

## 阶段耗时

下表为 active window 内每个 mode 8 次的独立 median。CPU region 用于顺序 wall 归因，CUDA region 用于训练主体内部归因；嵌套 region 不直接相加。

| mode | wall ms | DataLoader CPU ms | online teacher CPU ms | training step CPU ms | host gap ms |
|---|---:|---:|---:|---:|---:|
| mono/four | 1901.7 | 100.5 | DA3 818.3 | 946.7 | 34.2 |
| mono/single | 487.6 | 70.8 | DA3 220.2 | 177.9 | 23.6 |
| stereo/four | 2310.4 | 191.4 | LAS2-H 889.3 | 1171.5 | 36.3 |
| stereo/single | 567.0 | 54.5 | LAS2-H 237.5 | 245.0 | 15.9 |

按四种 mode median 各取一次组成 1:1:1:1 cycle，online teacher 约占 41%，training step 约占 48%，DataLoader 约占 8%。这是独立 median 的近似合成，不能当作严格可加的 timeline。

### Training step 内部 CUDA median

| mode | training step | backward | LPIPS forward | encoder | decoder | Adam |
|---|---:|---:|---:|---:|---:|---:|
| mono/four | 879.9 | 531.3 | 222.4 | 55.9 | 57.6 | 1.46 |
| mono/single | 109.8 | 57.4 | 37.4 | 4.3 | 4.5 | 0.75 |
| stereo/four | 1015.4 | 613.6 | 237.0 | 87.6 | 59.8 | 1.50 |
| stereo/single | 190.5 | 101.0 | 55.7 | 14.0 | 6.8 | 0.77 |

四模式合成后，training-step CUDA 时间中 backward 约占 59%，LPIPS forward 约占 25%，encoder+decoder forward 合计约 13%，Adam 约 0.2%。StereoFusion CUDA median 仅为 stereo/four `7.32 ms`、stereo/single `1.89 ms`，不是主要瓶颈。

### 算子、数据和设备信号

- active window 平均每 step 的主要 self CUDA operator：cuDNN convolution `129.1 ms`、`copy_` `87.0 ms`、`add_` `56.0 ms`、`mul` `52.8 ms`、`mm` `48.3 ms`、flash-attention backward `44.1 ms`、`div` `42.9 ms`、layer norm `34.6 ms`、`cat` `32.2 ms`、NCCL all-reduce `30.8 ms`
- NCCL 约占 profiler step device time的低个位数百分比，不是第一优先级；Adam 同样可以忽略
- DataLoader median 不大，但 p90 明显抖动：mono/four `1283.8 ms`、mono/single `1006.3 ms`、stereo/four `465.0 ms`、stereo/single `369.0 ms`；当前显式 `NUM_WORKERS=4`，而 launcher 默认值为 8
- active telemetry 共 384 个 GPU 样本：GPU utilization median `99%`、mean `79.0%`，功耗 median `397 W`、mean `383 W`。逐秒采样对 single-frame 短 step 分辨率有限，只用于支持存在空洞/等待，不作为 kernel 饱和度证明
- `mono/four` 峰值 PyTorch allocated/reserved 约 `75.26/75.37 GiB`，GPU telemetry memory-used median 约 `77.25 GiB`；不能靠直接放大 LPIPS chunk 或 batch 做无门禁提速

## 优化优先级

1. **先消除在线 teacher 串行路径。** 当前 DA3/LAS2-H 在每个 step 与训练主体串行，约占完整 mode cycle 的 41%。优先建设并验证全量、严格 provenance 的 teacher cache，然后增加 cache-only/fail-on-miss 训练路径并延迟或跳过 teacher 模型初始化。只开启现有 cache 会在 miss 时现场推理并压缩写 NPZ，首轮不代表稳态收益；应做 cold-build 与 warm-cache 分离 A/B。若 warm-cache 开销很低，理论上限约为减少 41% cycle wall（约 1.7×），实际需扣除 GPFS cache I/O。
2. **第二优先优化 LPIPS 和 backward，而不是 encoder/fusion。** LPIPS forward 已占 training CUDA 约 25%，并且其梯度也包含在约 59% 的 backward 中。当前按 `(view, frame)` 分块是 80GB H100 的安全修复，但带来多次 VGG 调用。应在不改变 loss mean 的前提下，先做 memory-gated 的 chunk 24/36/48 A/B，或评估 LPIPS activation/checkpoint、channels-last/compile；每个候选都必须同时验证数值/梯度和 `mono/four` 峰值，不能直接恢复 288-frame 单次调用。
3. **并行做 DataLoader 4→8 workers 的短 A/B。** 目标不是 median，而是压低 p90/max starvation；保持数据、seed、mode、batch 与 profiler schedule一致。若 8 workers 没有改善或增加 GPFS 抖动，则继续看解码与 `copy_/cat/contiguous`，而不是盲目加 worker。
4. **最后才看 encoder/decoder kernel。** forward 合计仅占 training CUDA 约 13%，StereoFusion 更小。可针对 cuDNN convolution、`copy_`、layout conversion 和 layer norm 做 channels-last/compile 小实验，但预期收益低于 teacher cache 和 LPIPS/backward。
5. **暂不投入 Adam、NCCL 或 StereoFusion 重构。** 当前测量下 Adam 约 0.2%，NCCL 低个位数，StereoFusion 每 logical update 只有数毫秒；先改这些不会显著改善端到端速度。

Profiler wall 包含 rank0 trace/shape/memory 采集以及约 8 分钟 trace 序列化写盘，只用于阶段归因。正式吞吐提升必须另跑不带 torch profiler 的同样本长窗口，用 direct `mode_samples` 与 wall time计算 samples/s，并报告分块 P10/P50/P90。
