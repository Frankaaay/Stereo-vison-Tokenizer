# 单目/双目统一 Relative Log-Depth 实现记录

## 状态与边界

- 日期：2026-08-25
- 本地分支：`merged-fs-vae-single-four-profiling`
- 修改前 HEAD：`33cce26eb08756350a51aa6cf52102c39065497b`
- 状态：本地未提交实现；未 commit、未 push、未同步 H200、未启动 GPU。
- 当前完成：统一 target/loss/head、动态 `V=1|3`/`T=1|4`、显式 batch mode
  metadata、四模式确定性 sampler、stereo single-frame decode-before-teacher、
  checkpoint counters、分模式日志和 stereo 评估语义迁移。
- 2026-08-26 续作已完成 Hy mono smoke dataset/DataModule、DA3-BASE 在线 teacher、
  native cache、四模式 validation mean 和 launcher 接线；H200-1 生成 48 条固定 mono
  cache。双节点数据证据、路径、SHA、测试与 H200-2 runtime blocker 见
  `docs/26-08-26/26-08-26-hy-mono-four-mode-smoke-preparation.md`。

## 冻结模型与 loss 合同

四个 mode 为：

- `mono/single_frame`
- `mono/four_frame`
- `stereo/single_frame`
- `stereo/four_frame`

输入为 `[B,V,E,3,T,H,W]`，mono 严格要求 `V=1,E=1`，stereo 严格要求
`V=3,E=2`。输出为：

```text
rgb:                    [B,V,3,T,H,W]
raw_relative_log_depth: [B,V,1,T,H,W]
latent:                 [B,V,48,1,16,16]
```

FoundationStereo native disparity 在训练边界转换为：

```text
log(fx * baseline_m) - log(disparity)
```

DA3 native positive relative depth 转换为 `log(relative_depth)`。两种 teacher
随后都在每个 sample 内先按各 view 的有效像素求中心，再对实际 view 等权平均；
T=4 的四帧和 stereo 的三个 view 共用一个 sample center。Student 使用同一 teacher
mask 计算自己的中心，且 center 不 detach。

core loss 只替换原有两个 geometry 槽：

```text
core_loss =
    rgb_weight * rgb_loss
  + relative_depth_weight * relative_log_depth_smooth_l1
  + relative_gradient_weight * spatial_xy_gradient_smooth_l1
  + effective_kl_weight * kl_loss
```

完整 generator loss 仍为 `core + LPIPS + adversarial + feature matching`；当前
launcher 的 image/video GAN 与 feature matching 权重仍为 0，LPIPS 继续使用已有
`PERCEPTUAL_WEIGHT`。Validation 保持 `core + LPIPS`。

## 调度、checkpoint 与日志

- 统一 per-device BS=24、gradient accumulation=1、mode update ratio=1:1:1:1。
- 每四个 optimizer update 构成一个 seeded cycle；cycle 内四种 mode 各一次并打乱。
- DDP rank 共享 mode 顺序，各 rank 从各自 source index 子集取 BS24。
- checkpoint 保存四种 mode 的 update/sample counters、BS、GA 和 mode contract；旧
  disparity/metric-depth checkpoint 因 head/state/semantic contract 缺失而 strict fail。
- 日志按 `train/{mode_id}`、`val/{mode_id}` 分开记录 loss、update、sample、step time
  和 per-rank/per-mode peak memory。

## Stereo teacher/cache 边界

现有 Manifest v3 FoundationStereo disparity cache 保留，不重跑、不覆盖。在线
FoundationStereo cache 升级为 v3 metadata，绑定 eye/temporal mode、view count、
teacher family、native target representation 和实际 tensor shape。Single-frame
LeRobot 路径在视频 decode 前选择唯一 offset，FoundationStereo 只处理 T=1。

## 后续输入状态

1. Hy Lance row/schema 与 raw RGB cache 合同已核对并实现。
2. DA3-BASE source/revision/checkpoint SHA 已固定并写入严格校验。
3. 第一轮接口 smoke 显式使用 finite/positive/non-padding mask；正式 confidence 阈值
   仍等待 calibration，不能猜 `>0.5`。
4. 最佳 checkpoint 已按用户确认冻结为四模式等权 `val/mixed/total_loss`。

## 本地验证

已执行：

```text
python -m unittest tests.stereo.test_source_boundary
python -m unittest tests.stereo.test_entrypoints_source tests.stereo.test_source_boundary
python -m py_compile <全部本次修改的 Python 文件>
git diff --check
```

结果：source/AST 合同测试 27/27 通过，全部修改文件可编译，diff whitespace 检查通过。
本机 `C:\Users\Frank\miniconda3\python.exe` 没有 `torch` 与 `pytest`，因此 tensor
单测尚未运行；未为此安装依赖，也未在未经授权的 H200 上运行测试或显存 smoke。
