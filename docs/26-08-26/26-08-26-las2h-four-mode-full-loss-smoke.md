# LAS2-H 四模式完整 Loss Smoke 验收报告

- 日期：2026-08-26
- 分支：`hezhou/las2-h`
- commit：`45fec2e341eb83ade04c9a1c88d824e49f3c9b5f`
- v5 run：`mixed_modes_8gpu_full_loss_step4_v5_20260826`
- 配置：8 × H200，per-device batch 24，global batch 192，BF16，LPIPS/GAN/feature matching 全开启

> **范围说明**：本报告验收工程链路，不执行小样本过拟合。v5 只有四次 generator update（每模式一次），因此 loss 是 finite 快照，不是收敛曲线，也不能证明重建质量改善。

## 1. 结论

- **工程链路：通过。** LAS2-H stereo 在线 GT、DA3 mono 在线 GT、四模式 DDP、完整 LPIPS/GAN、两个 optimizer 和 checkpoint 保存均可运行。
- **Shape / 数值：通过。** 四模式真实 B=1 输入、监督和模型输出均符合合同；所有 loss finite；teacher GT 不参与梯度。
- **梯度：通过。** 所有按模式预期使用的 Encoder、Fusion、Temporal、Posterior、Decoder、RGB/depth head 和判别器均有 finite、非零梯度；未使用项与结构设计一致。
- **Checkpoint：通过。** strict load、确定性 reload、两个 optimizer/scheduler 状态以及八卡恢复后一次 G/D 更新全部通过。
- **收敛与重建质量：未证明。** 需要另一个小样本过拟合实验提供 loss 曲线和重建图。

## 2. 四模式完整 loss 快照

| 模式 | Total | RGB | Depth | Gradient | KL×w | LPIPS | Adv | FM | D loss | Step(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 双目 / 四帧 | 3.0693 | 0.3951 | 0.3889 | 0.1995 | 0.0032 | 0.9082 | 0.3079 | 0.8665 | 2.0938 | 4.064 |
| 单目 / 单帧 | 2.0470 | 0.4875 | 0.1418 | 0.1946 | 0.0032 | 0.8838 | -0.1033 | 0.4394 | 0.6338 | 0.310 |
| 单目 / 四帧 | 4.8458 | 0.4934 | 0.1698 | 0.1918 | 0.0047 | 0.8789 | 2.2129 | 0.8942 | 0.6128 | 7.436 |
| 双目 / 单帧 | 3.2132 | 0.3957 | 0.3882 | 0.1823 | 0.0037 | 0.8979 | 0.9204 | 0.4249 | 0.4192 | 0.491 |

`Lightning global_step=8` 是 generator/discriminator 两个 optimizer 各更新 4 次造成；实际 generator update 为 4。

![Loss components](/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/loss_components.png)

## 3. 时间、在线 teacher 与显存

| 模式 | Teacher | Teacher(s) | 完整 step(s) | Teacher 占比 | 峰值 allocated(GiB) | valid ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 双目 / 四帧 | LAS2-H | 1.343 | 4.064 | 33.04% | 106.16 | 44.46% |
| 单目 / 单帧 | DA3 | 0.161 | 0.310 | 51.89% | 8.34 | 56.64% |
| 单目 / 四帧 | DA3 | 0.343 | 7.436 | 4.62% | 33.28 | 56.64% |
| 双目 / 单帧 | LAS2-H | 0.202 | 0.491 | 41.08% | 24.05 | 44.52% |

- 以上 teacher 时间来自 v5 八卡训练的**单 rank、batch size 24** 日志，不是 B=1 延迟。
- stereo/four 中 LAS2-H 为 1.343 s，占完整 step 约 33%；mono/four 中 DA3 仅 0.343 s，占约 4.6%，不是 7.436 s step 的主因。
- mono/single 的 DA3 占比较高，但该模式只有一个 update，不能据此做稳定性能结论。
- v5 未启用 Kineto，`torch_profile_output_dir=None`；因此目前没有 DataLoader、H2D、Encoder/Decoder、LPIPS/GAN、backward、Adam 等更细模块时间。若需要定位 mono/four 的 7.436 s，必须补一轮带 warmup 的分阶段 profile。

![Timing and memory](/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/timing_memory.png)

## 4. Shape 与数值验收

| 模式 | 输入 | GT | Fusion contract | latent | RGB | depth | finite loss | GT requires_grad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 双目 / 四帧 | [1,3,2,3,4,256,256] | [1,3,1,4,256,256] | [1,3,4,512,16,16] | [1,3,48,1,16,16] | [1,3,3,4,256,256] | [1,3,1,4,256,256] | 通过 | False |
| 单目 / 单帧 | [1,1,1,3,1,256,256] | [1,1,1,1,256,256] | — | [1,1,48,1,16,16] | [1,1,3,1,256,256] | [1,1,1,1,256,256] | 通过 | False |
| 单目 / 四帧 | [1,1,1,3,4,256,256] | [1,1,1,4,256,256] | — | [1,1,48,1,16,16] | [1,1,3,4,256,256] | [1,1,1,4,256,256] | 通过 | False |
| 双目 / 单帧 | [1,3,2,3,1,256,256] | [1,3,1,1,256,256] | [1,3,1,512,16,16] | [1,3,48,1,16,16] | [1,3,3,1,256,256] | [1,3,1,1,256,256] | 通过 | False |

Fusion 的生产内部原生布局为 `[B,V,T,H,W,D]`，报告中的 contract 布局统一转为 `[B,V,T,D,H,W]`。

## 5. 梯度验收

表中括号为梯度 L2 norm；“按设计未使用”不算失败。

| 模块 | 双目 / 四帧 | 单目 / 单帧 | 单目 / 四帧 | 双目 / 单帧 |
| --- | --- | --- | --- | --- |
| spatial_encoder_patch | 通过 (17.9) | 通过 (29.4) | 通过 (36.2) | 通过 (18) |
| spatial_encoder_transformer | 通过 (1.02) | 通过 (2.2) | 通过 (2.07) | 通过 (1.29) |
| stereo_fusion | 通过 (0.00116) | 按设计未使用 | 按设计未使用 | 通过 (0.000546) |
| encoder_temporal_transformer | 通过 (0.917) | 按设计未使用 | 通过 (1.64) | 按设计未使用 |
| temporal_sampler | 通过 (1.52) | 按设计未使用 | 通过 (2.48) | 按设计未使用 |
| posterior_projection | 通过 (1.53) | 通过 (2.79) | 通过 (2.02) | 通过 (1.62) |
| decoder_latent_projection | 通过 (0.707) | 通过 (1.13) | 通过 (0.721) | 通过 (0.823) |
| decoder_temporal_expansion | 通过 (3.03) | 按设计未使用 | 通过 (3.07) | 按设计未使用 |
| decoder_temporal_transformer | 通过 (2.23) | 按设计未使用 | 通过 (2.17) | 按设计未使用 |
| spatial_decoder | 通过 (2.87) | 通过 (4.58) | 通过 (2.66) | 通过 (3.72) |
| rgb_head | 通过 (5.62) | 通过 (8.29) | 通过 (5.17) | 通过 (7.07) |
| depth_head | 通过 (0.28) | 通过 (0.598) | 通过 (0.243) | 通过 (0.427) |

| 判别器 | 双目 / 四帧 | 单目 / 单帧 | 单目 / 四帧 | 双目 / 单帧 |
| --- | --- | --- | --- | --- |
| image_discriminator | 通过 (25.5) | 通过 (31.3) | 通过 (24.4) | 通过 (28.6) |
| video_discriminator | 通过 (32.6) | 按设计未使用 | 通过 (45.8) | 按设计未使用 |

## 6. Checkpoint 恢复验收

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| state_dict strict=True | 通过 | missing=0, unexpected=0 |
| 确定性 latent | 通过 | max abs=0 |
| 确定性 RGB | 通过 | max abs=0 |
| 确定性 depth | 通过 | max abs=0 |
| optimizer / scheduler | 通过 | 2 optimizer + 2 scheduler |
| 恢复后继续 G/D 更新 | 通过 | G 4→5, D 4→5, global_step 8→10 |
| 参数实际变化 | 通过 | 131 tensors / 29,501,636 elements |

恢复 smoke 使用相同 8 卡、每卡 batch 24、完整 LPIPS/GAN，从原 v5 `last.ckpt` 恢复后执行下一调度模式 `mono/single_frame`。新 checkpoint 的 generator/discriminator counter 均从 4 增至 5，global_step 从 8 增至 10；131 个 floating state tensors 发生更新。

## 7. 图表与产物

- `/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/metrics_summary.csv`
- `/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/metrics_summary.json`
- `/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/shape_gradient_report.json`
- `/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/checkpoint_restore_report.json`
- `/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/loss_components.png`
- `/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/timing_memory.png`
- `/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/valid_ratio.png`
- `/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/update_counts.png`
- `/data/home/hezhou/experiments/stereo-tokenizer-las2h-smoke/las2h_v5_acceptance_report/las2h_stereo_tokenizer_smoke_report_standalone.html`

## 8. 审慎判断

1. **是否值得继续**：值得。工程链路已通过，可以进入小样本过拟合与更长 smoke。
2. **最大风险**：四模式成本差异非常大，尤其 mono/four 的慢点尚未被模块级 profile 解释；另外四次更新不能证明稳定收敛。
3. **最缺证据**：固定样本上的连续 loss 曲线、重建图，以及带 warmup/Kineto 的阶段耗时。
4. **今天可执行的最小下一步**：由独立过拟合任务在固定样本上生成 step 0/中间/最终重建图；性能侧单独 profile mono/four。
5. **置信度**：工程链路与 checkpoint 结论 95%；收敛与重建质量结论不足，暂不下判断。
