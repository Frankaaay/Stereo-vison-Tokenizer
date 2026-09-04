# Stage A1 checkpoint 对比评测

## 目的与合同

- 目标：使用现有 Stage A1 指标，在完全相同的 UMI/LIBERO selection 上比较新 checkpoint 与上一版 baseline，并输出 3 个同样本重建案例。
- 正式质量：FP32、posterior mean、batch size 1；覆盖 UMI stereo、UMI 六个 mono camera cell、LIBERO 两个 mono camera cell；跳过 HY。
- 正式效率：H100、BF16、batch size 1、warmup 20、iterations 100、repeats 3；只统计 model-only encode/posterior/decode/end-to-end。
- 数据：只读复用上一版 baseline 的 selection、identity contract、canonical config 和 teacher 资产；selection 内容及语义哈希保持不变。
- 可视化：从冻结 selection 中选取跨数据域的 3 个固定 case，使用相同输入分别展示 baseline 与 candidate 重建。

## 代码修正

原 Stage A1 实现把上一版 checkpoint SHA、global step 和训练 update counters 写死在通用 `run`、`benchmark` 与 `report` 路径中，因此任何新 checkpoint 都会在模型加载前被拒绝。

本次修正仅泛化 provenance 校验：

- `--checkpoint-sha256` 必填，并与 checkpoint 文件实际 SHA256 严格一致；
- checkpoint 必须包含非负的 global step、epoch 和五个核心训练 counters；
- 同一报告的 9 个质量 artifact 必须具有完全一致的 checkpoint provenance，2 个效率 artifact 必须使用同一 SHA；
- canonical loader 可以位于不同用户路径，但运行 clone 必须 clean，且 HEAD 必须与 selection 冻结的 loader SHA 完全一致。

指标公式、输入域、mask、teacher、时序模式和速度 benchmark 合同均未改变。

首次 GPU smoke 还发现，mono 多视角联合训练提交后，DA3 teacher 合同已统一为 `[B,V,T,3,H,W]`，但 canonical Stage A adapter 仍输出旧的 `[B,T,3,H,W]`。这会在 `attach_online_targets()` 解包时以 `expected 6, got 5` 失败。修复仅在 canonical mono 的 `da3_images` 上补回 `V=1` 维，不改变像素、几何映射或 teacher 推理内容。

## 运行记录

- 代码分支：`hezhou-las2-h`
- 评测代码 SHA：`75f4cf6102565e6ced07cf861babb8bbc705149d`
- checkpoint：`/gpfs/jiuquyun/projects/Frank/stereo-tokenizer-checkpoints/v1/stagea-threeview-update124000/last.ckpt`
- checkpoint SHA256：`605be7940202b7f0aff2380ac4a99dfe1e15d20847ab0377d3e7b2a27352f7b7`
- checkpoint 训练计数：generator updates 124,000，discriminator updates 0，batch updates 136,000，four/single-frame updates 各 62,000；`global_step=80000`。
- 运行位置：H100 Slurm；代码、环境和 checkpoint 均来自 Frank 目录，selection、canonical config 与 teacher 权重只读复用已授权的 hezhou 资产。
- 输出根目录：`/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/stage-a1-stagea-threeview-update124000-20260903-b3bfa1c`
- 正式质量 Job：3018–3026；正式效率 Job：3027–3028。11 个 Job 均为 `COMPLETED`、exit code 0。
- 质量报告：`report/stage-a1-candidate.md`；A/B 原始差值：`report/baseline-comparison.json`；作业状态：`job-status.json`。
- 完整性校验通过：9/9 质量 artifact、2/2 效率 artifact；checkpoint、selection、代码、loader 和 teacher provenance 一致。
- 定向测试：H100 上 `tests.stereo.test_stage_a_evaluation tests.stereo.test_da3_online_teacher` 共 33/33 通过。

## 质量结果

下表为四帧聚合指标。箭头表示该指标的优劣方向，括号内是 candidate 相对 baseline 的变化。

| 数据 / 模式 | checkpoint | RGB L1 ↓ | RGB MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LIBERO mono | baseline | 0.034263 | 0.003777 | 24.666 | 0.886960 | 0.164680 |
| LIBERO mono | candidate | 0.038002 (+10.91%) | 0.003141 (-16.84%) | 26.342 (+1.676 dB) | 0.917989 (+0.03103) | 0.144711 (-12.13%) |
| UMI mono | baseline | 0.052250 | 0.006414 | 22.117 | 0.778473 | 0.199739 |
| UMI mono | candidate | 0.055372 (+5.97%) | 0.005538 (-13.65%) | 22.945 (+0.828 dB) | 0.830748 (+0.05228) | 0.173667 (-13.05%) |
| UMI stereo | baseline | 0.048049 | 0.005822 | 22.541 | 0.783413 | 0.193893 |
| UMI stereo | candidate | 0.051340 (+6.85%) | 0.004900 (-15.84%) | 23.518 (+0.977 dB) | 0.836985 (+0.05357) | 0.168932 (-12.87%) |

四帧 temporal delta L1 分别下降 20.30%、13.01%、12.33%，temporal delta LPIPS 分别下降 11.03%、9.10%、7.95%。越界像素率分别下降 82.31%、50.17%、55.70%。

单帧四个 source 的走势一致：

- LIBERO：LPIPS 下降约 12.2%–12.8%，SSIM 增加约 0.0243–0.0246，PSNR 增加约 0.90–0.94 dB，MSE 下降约 1.7%–3.6%；但 L1 上升约 20.5%–22.6%。
- UMI mono：LPIPS 下降约 13.9%–14.2%，SSIM 增加约 0.0241；PSNR 下降 0.06–0.09 dB、MSE 上升约 6.8%–8.0%，L1 上升约 18.9%–19.7%。
- UMI stereo：LPIPS 下降约 14.2%–14.5%，SSIM 增加约 0.0268–0.0270，PSNR 基本持平；MSE 上升约 6.6%–7.5%，L1 上升约 22.5%–23.3%。

## 推理速度

| 模式 | baseline e2e p50 | candidate e2e p50 | candidate 吞吐 | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: |
| mono four-frame | 8.201 ms | 8.077 ms | 123.812 samples/s | +1.53% |
| mono single-frame | 5.367 ms | 5.266 ms | 189.891 samples/s | +1.92% |
| stereo four-frame | 8.668 ms | 8.972 ms | 111.453 samples/s | -3.39% |
| stereo single-frame | 5.959 ms | 6.085 ms | 164.333 samples/s | -2.07% |

三次 repeat 下差异均在约 3.5% 内：mono 略快，stereo 略慢。当前证据只支持“总体同一速度量级”，不把这些小差值解释为确定的架构收益或退化。

## 固定案例

三组 baseline/candidate 均由同一冻结 selection、同一 `selection_index` 和同一 source 生成：

1. UMI stereo：`umi:d9a51d156ed345a3e7cd6b96b2b16249:3`，selection index 630，source 0。
2. UMI stereo：`umi:74ec16c75bd779f40d1701bd2e2fd2a4:0`，selection index 71，source 0。
3. LIBERO head-left：`libero:libero:002685:5`，selection index 173，source 0。

三个 baseline 案例均出现明显彩色散斑或局部爆色，candidate 消除了这些强伪影；candidate 的主要残余问题是纹理和局部细节更平滑。该观察与越界像素率、LPIPS 和 SSIM 的改善一致，但三个案例不替代全量指标。

## 结论

candidate 不是对 baseline 的全指标 Pareto 支配：四帧 MSE、PSNR、SSIM、LPIPS、时序一致性及输出范围稳定性均显著改善，但 L1 变差；单帧 UMI 的 MSE/L1 也变差，PSNR基本持平。它更像是一次“减少强伪影并提升感知及时序质量，以更平滑的重建和更高 L1 为代价”的变化。

因此 Stage A1 可以判定 candidate 在四帧、感知质量和稳定性方向优于 baseline，但不能写成“重建质量全面提升”。两个 checkpoint 的训练预算、effective global batch 和 GAN 配方也不同，本次 A/B 不能单独归因到某个结构改动。是否替换 baseline 应继续进入冻结 tokenizer 的下游 WAM Gate B；WAM 规模、rollout 数量和 latent ABI 以目标代码/BAS 配置审计结果为准。
