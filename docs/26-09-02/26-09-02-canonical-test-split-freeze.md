# Stereo Tokenizer Stage A canonical test split 冻结报告

## 结论

H100 当前 canonical 数据的全新 episode-level `train/val/test = 90/5/5` 合同已冻结。该合同不沿用旧 checkpoint manifest 的 split，也没有复制或搬运视频文件；manifest 只保存 episode identity、canonical config、episode index、长度、FPS、split 和完整 provenance。

UMI 与 LIBERO 已进一步生成 decode-audited Stage A selection 并通过两样本 preflight。HY 的完整 test manifest 已成功冻结，但 pinned canonical loader 与 HY Lance 发布数据对 `index` 的定义冲突，10,779 个 test episode 中只有 3 个通过现有 loader 的严格校验，因此不能生成正式 1,024-sample HY selection，也不能把当前状态称为完整 baseline。

## 冻结合同

- 数据入口：`/gpfs/jiuquyun/datasets/PRETRAIN_DATA/*`
- 仓库：`/gpfs/jiuquyun/projects/hezhou/Workspace/Stereo-vison-Tokenizer`
- 分支：`hezhou-las2-h`
- manifest 生成提交：`2e2409baf727f351af1bb06ad77be99e64805a1c`
- selection decode-audit 提交：`56796defb6a32a6b67b5080b72ac251699e92880`
- canonical loader：`/gpfs/jiuquyun/projects/hezhou/Workspace/NGADv1pp/ngad/datasets/ngad-canonical-dataloader`
- canonical loader SHA：`d51377ac450b0066bc0c8eb13939bcfae47275ff`
- split seed：`3407`
- split unit：episode
- split policy：按 `SHA256(seed, dataset_id, episode_id)` 全局稳定排序，再按精确 90/5/5 数量切分
- selection seed：`1234`
- selection policy：不同 episode 各取一个确定性的、4 帧对齐窗口；候选必须真实 decode 并通过 shape、mask、finite、frame identity 校验

## Manifest 结果

根目录：`/gpfs/jiuquyun/projects/hezhou/experiments/stereo-tokenizer-stage-a/contracts/canonical-v3-20260902-seed3407`

| 数据集 | 总 episode | train | val | test | manifest 语义 SHA256 | manifest 文件 SHA256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| UMI | 90,174 | 81,156 | 4,509 | 4,509 | `e4fdcd1264ce1b8c08612305701dd907569822860aaba8825264600f5cc02fe3` | `636138b262fa6df4299c7e938de6a964998d89bcbbb562907078a4dfe4c6d870` |
| HY | 215,577 | 194,019 | 10,779 | 10,779 | `0761e18b9d70b3a189618038d8bb7021966c3b2bfa9cd1750b94676e0ea00a81` | `0647dd13f01d7f5541c6e7f209b93061e31264b891011447d777f09058884a0f` |
| LIBERO | 6,500 | 5,850 | 325 | 325 | `59cea99c07cad93a7e6e922241002a6d3ae8190837941e204a356218c27cf7ae` | `51ba67cf003eb2cab4b58a6d363977652f22940ee209915dd349e126b09a808e` |

具体文件：

- UMI：`umi/umi-canonical-90-5-5-seed3407.json`
- HY：`hy/hy-canonical-90-5-5-seed3407.json`
- LIBERO：`libero/libero-canonical-90-5-5-seed3407.json`
- 每个 JSON 旁均有独立 `.sha256` sidecar。
- manifest 内保存实际 cwd、branch、commit、loader SHA、源配置及 snapshot SHA256。
- HY 的 19 份源 YAML 缺少 loader v2 要求的顶层 `schema_version`；冻结目录保存了只增加该字段的配置 snapshot，并同时记录原文件与 snapshot 的 SHA256。共享数据目录没有被修改。

## Stage A selection 与 decode 审计

selection 根目录：`/gpfs/jiuquyun/projects/hezhou/experiments/stereo-tokenizer-stage-a/20260902-stagec-update162500-baseline-v1/selections`

### UMI stereo

- selection：`umi-canonical-test-1024-seed1234.json`
- 语义 SHA256：`4a412e87244d59883ee62ad012282e09c29838b9dfe07d2109ba8efb5960929b`
- 文件 SHA256：`615f35e42d8b9904b8a7d1b308a54d8a861f8a1896f841f922b76cf827e26677`
- 为得到 1,024 个 decode-valid episode，共检查 3,249 个确定性排序候选，拒绝 2,225 个。
- 拒绝分类：1,471 个 data-shard offset 越界，515 个 LeRobot row identity 不一致，239 个 source timestamp/FPS 不一致。
- 注意：这只描述达到 1,024 个样本前检查过的候选，不代表剩余 1,260 个 test episode 都无效。
- preflight：Slurm `2212`，退出码 0；两例 shape `[3,2,3,4,256,256]`，dtype `float32`，四帧 source index 分别为 `[84,87,90,93]`、`[96,99,102,105]`。

### LIBERO mono

- selection：`libero-canonical-test-325-seed1234.json`
- 语义 SHA256：`9ff8d2adf1d9caf2d9b49d708f7a395045d62d8ccb7899bd3afd9b1316c1247c`
- 文件 SHA256：`e78ef19d8be3fe1d0d73506df99f94c739d52747287b45eb4e46acacfbde8f1c`
- 325/325 个 test episode 全部通过 decode audit，无拒绝。
- preflight：Slurm `2213`，退出码 0；两例 head-left shape `[1,1,3,4,256,256]`，dtype `float32`，四帧 source index 分别为 `[120,123,126,129]`、`[192,195,198,201]`。

### HY blocker

- 完整 manifest 和 10,779-episode test split 已冻结。
- 正式 1,024-sample selection 作业：Slurm `2207`，检查全部 10,779 个 test episode，仅 3 个通过，退出码 1；没有写出正式 selection。
- 直接证据：`table_000` episode 10 的 metadata `dataset_from_index=9116`，但物理 Lance row 在 offset 9116 的字段为 `index=9117, episode_index=10, frame_index=0`。`table_001/010/018` 抽样表现相同。
- pinned loader 当前要求 `row.index == physical_offset`，因此抛出 `Canonical Lance row identity mismatch`。episode identity 与 frame identity 本身仍匹配。
- 这是发布数据与 loader 合同冲突。评测代码没有静默删除该校验、没有绕过 pinned loader，也没有从 train/val 补样本。

## 可复现性与边界

- manifest 是新的 canonical benchmark identity，不能描述成 checkpoint 训练时的旧 test split。
- selection 是 manifest test split 的 decode-valid 子集，两层合同必须分别保存和引用。
- 当前 preflight 环境为共享 GPFS Python 3.10 诊断环境；它证明数据访问与 wrapper 合同可用，不替代最终要求的 Python 3.12 locked runtime。
- 尚未运行 checkpoint 的正式 CUDA reconstruction metrics，因此本报告不是 RGB/temporal 数值 baseline。
- 在 HY 合同冲突解决前，不应发布三数据集 macro average。

## 决策

1. **这个想法值得做，但需要修复后完成。** 新 test split 已形成可复现合同；UMI/LIBERO 可继续 baseline，HY 暂停。
2. **最大风险：** 为了凑够 HY 样本而绕过 row identity 校验，会把不满足 canonical 合同的数据混入正式结果，之后无法证明样本身份和跨版本可比性。
3. **最缺的关键证据：** HY 发布方需要明确 `index` 是物理连续行号还是保留缺口的源数据编号，并给出修复后的 Lance/metadata 或经 review 的 loader 合同变更。
4. **今天可以执行的最小一步：** 用上述 `offset=9116 / index=9117` 最小复现请数据/loader owner 定义正确语义；结论确定后只做一处上游修复，再重新跑 10,779 episode decode audit。
5. **置信度：97%。** manifest 数量、哈希、selection 审计、Slurm 退出码和 HY 行级冲突均有机器证据；不确定性只在 HY `index` 的产品语义应由发布方确认。
