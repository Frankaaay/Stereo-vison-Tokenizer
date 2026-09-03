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

## 运行记录

评测完成后补充精确代码 SHA、Slurm Job ID、checkpoint/output/log 路径、artifact 校验、主要指标、3 个案例和结论。
