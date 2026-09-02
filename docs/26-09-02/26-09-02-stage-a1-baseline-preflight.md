# Stereo Tokenizer Stage A1 baseline 预检报告

## 结论

状态：**BLOCKED，尚未产生正式 baseline 指标。**

评测实现值得保留，但当前不能把 H100 新 canonical 入口上的结果称为与 checkpoint 训练合同一致的 baseline。原因不是视频不可读，而是旧测试集身份与新发布入口不一致，同时 HY 发布配置不满足固定 canonical loader 的 schema 合同。

本报告不会用任意新 episode 替代旧 test episode，也不会给出占位或推测指标。

## 固定对象

- 仓库：`/gpfs/jiuquyun/projects/hezhou/Workspace/Stereo-vison-Tokenizer`
- 分支：`hezhou-las2-h`
- Stage A1 实现提交：`941e9237196b9658f73f58607cfaa7b8990f0475`
- 实现前基线提交：`ad5d806ac681f7e8e2af00dab193763f5ae58340`
- checkpoint：`/gpfs/jiuquyun/projects/Frank/stereo-tokenizer-checkpoints/v1/stagec-update162500/last.ckpt`
- checkpoint SHA256：`a74c3b72b32dfd296157e3b6ad24d0521731517e79e75f22786bca37c47d822e`
- checkpoint 直接计数：global step 125000、generator 162500、discriminator 118500、batch 162500、single-frame 81250、four-frame 81250
- canonical loader：`/gpfs/jiuquyun/projects/hezhou/Workspace/NGADv1pp/ngad/datasets/ngad-canonical-dataloader`
- canonical loader Git SHA：`d51377ac450b0066bc0c8eb13939bcfae47275ff`

旧 manifest 只转存了 split/episode 身份合同，没有搬视频或其他数据：

| 数据集 | 身份合同 SHA256 | 旧 test episode 数 |
| --- | --- | ---: |
| UMI | `f3b7f85c32573edbb75e750cd3986fc6d68875243bbbb95f8a1d3cfa0c236a12` | 3132 |
| HY | `fc0075580bbb5a353a9ae151ad8a604a5665b78bca0bf98f92eafcb6f0a17caf` | 2897 |
| LIBERO | `283be628c3449cad895d618238742b2dd0a21b32947e9a5dc979639591b1e715` | 87 |

## 已完成实现

- 固定旧 manifest 身份合同与 SHA256 校验。
- 从旧 split 与新 canonical 的精确 episode 身份交集中，确定性选择不同 episode 的非重叠 4 帧窗口。
- 固定 10 Hz 语义帧；source frame 按实际 source FPS 计算，不硬编码 30 FPS。
- 运行时二次校验 split 身份、selection SHA、canonical YAML SHA、loader 路径/commit 和 UMI publish ledger SHA。
- 支持 mono/stereo × single/four-frame；single-frame 固定评估 4 个 source frame，four-frame 使用 posterior mean。
- mask-aware RGB L1、MSE、PSNR、SSIM、LPIPS；four-frame temporal-delta L1/LPIPS。
- 输出健康检查、样本级 P50/P90/P99、per-view/per-mode/macro 聚合与 latent ABI/压缩信息。
- DA3/LAS2-H 仅标记为 teacher-relative；没有独立 GT 时不报告真实 depth/disparity accuracy。
- rFID、RAFT warp/flicker/motion 保留到 A2；原生 4 帧 rFVD/FVMD 标记 N/A。

## H100 真实数据预检

UMI 新 canonical 视频读取成功：

- 配置 SHA256：`93b8e49c433e8ed97c79fb6902c7ca68aa12270e7d0d4653379d158852dcce86`
- publish ledger SHA256：`9dd66b359efad59abf683fe4ab3235d6c4475f854eb3c575e0d55b160b3d4450`
- canonical dataset window 数：6,782,506
- 单窗口 shape：`[4,6,3,256,256]`
- dtype/range：`float32`，`[-1,1]`
- frame offsets：`[0,1,2,3]`
- 30 Hz source frame indices：`[0,3,6,9]`
- 六个 camera 均有效

因此，`table_000/videos/.../file-000.mp4` 的路径和解码链路本身没有问题。

## 阻断证据

### UMI：旧测试身份与新入口无交集

- 旧 manifest：62,625 个唯一 episode，其中 test 3,132 个。
- H100 新 publish ledger：90,174 个唯一 `source_id`。
- 全量交集：0；test 交集同样为 0。
- 在 H100 原始 `UMI-Collectsite-KS3` 根目录中抽查旧 test ID，也未找到对应 episode。

所以新 canonical 数据“齐全可读”不等于包含 checkpoint 当时的测试 episode。当前不能生成 stereo 正式 selection。

### HY：数据覆盖不完整且 YAML 不合法

- 旧 test 涉及 `table_012/014/016/018/020`。
- H100 新入口缺少 `table_014` 数据目录和配置，影响 548/2,897 个旧 test episode。
- 其余精确身份交集有 2,349 个，理论上足够固定选择 1,024 个。
- 但当前全部 HY YAML 缺少顶层 `schema_version: ngad_canonical_dataloader_v2`。
- 固定 loader 因此按设计拒绝加载，错误为 `YAML root must contain exactly schema_version and dataset.`

没有修改共享数据配置，也没有在评测端静默绕过该 schema 检查。

### LIBERO：缺少可审计 crosswalk

旧 manifest 的 episode ID 是 suite-local，新 canonical 使用全局 episode index。当前没有冻结且可校验的新旧 crosswalk，因此实现保持 fail-close。

## 验证结果

- Stage A1 单元测试：6/6 通过。
- `py_compile`：通过。
- `git diff --check`：通过。
- H100 UMI 真实 4 帧/六相机 decode：通过。
- 正式 CUDA quality smoke：未执行；测试身份合同尚未满足。
- 正式 metrics JSON、固定可视化、退出码报告：未产生。

临时 CPU 数据预检环境是 Python 3.10 组合环境；虽然关键数据依赖为 `pyarrow 23.0.0 / pylance 10.0.0`，它不等于项目要求的 Python 3.12 完整 locked runtime，不能作为正式评测环境证据。

## 决策

1. **这个想法值得做，但需要改。** 评测合同和代码应保留；必须先修复数据身份与发布配置，再运行 baseline。
2. **最大风险：** 在 episode 身份不一致时仍把新数据结果命名为旧 checkpoint baseline，会造成不可复现且不可比较的结论。
3. **最缺的关键证据：** UMI 旧 episode ID 到 H100 新 canonical episode 的可信 crosswalk，或明确证明两者是不同发布集；另需修复 HY v2 YAML 并补齐/正式排除 `table_014`。
4. **今天可执行的最小一步：** 数据发布方先提供一份带 SHA256 的 UMI crosswalk；若确认两批数据不同，则冻结一个新的 canonical test manifest，并把它明确命名为新评测集而不是旧 split。
5. **置信度：95%。** 视频链路正常和身份交集为 0 都有直接机器检查；剩余不确定性在于发布方是否持有尚未落盘的 crosswalk。
