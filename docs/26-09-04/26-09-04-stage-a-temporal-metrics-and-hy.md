# Stage A 时间指标与 HY Table014 排除实现记录

## 目的

在暂不执行 single-frame rFID 的前提下，为 Stage A 评测补充 optical-flow warp、static flicker、motion consistency 和 teacher-relative temporal geometry consistency；同时让 HY 复用训练路径的 production manifest/Lance reader，并显式排除当前不可用的 Table014。

## 代码与运行合同

- 分支：`hezhou-las2-h`
- 开始修改时 commit：`605a0bd176c7d4678bdee051d18a47cbe8c9ca2f`
- 提交前同步基线：`8d3c290d6bf5c14c4349ba217f592fb35e9cf393`（开发期间该分支新增并撤销了独立的 LPIPS BF16 对照改动；本变更未触碰其文件）。
- 本机工作目录：`C:\Project\Stereo-vison-Tokenizer`
- 本轮没有启动 GPU evaluation。
- RAFT-Large 共享资产：`/gpfs/jiuquyun/checkpoints/Frank/stereo-vae/runtime-assets/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth`
- RAFT-Large SHA256：`ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`；大小 `21,106,607` bytes；owner/group `Frank:ai-users`；权限 `0640`。
- 同目录 checksum sidecar：`raft_large_C_T_SKHT_V2-ff5fadd5.pth.sha256`；远端 `sha256sum -c` 已通过且无 `.part` 残留。
- RAFT 固定为 torchvision RAFT-Large，FP32，本地 checkpoint 必须通过调用方给出的 SHA256 校验；不允许运行时联网下载。
- 所有 flow 指标只在 RGB content crop 上计算，并使用目标视频的 forward/backward consistency mask。
- static：目标 flow magnitude `<=0.5 px`；dynamic：`>=1.0 px`；中间灰区不计入 static flicker 或 motion EPE。
- forward/backward consistency：`||F + warp(B,F)||^2 <= 0.01*(||F||^2+||warp(B,F)||^2)+0.5`。
- temporal geometry 在同一目标 backward-flow 对齐和 consistency mask 下比较相邻帧 relative-log geometry residual；mono 为原图 DA3 对重建图 DA3，stereo 为 LAS2-H target 对 tokenizer depth head。
- 所有时间指标均报告总体及 horizon pair 01/12/23，并报告有效 coverage。

## HY 数据路径

- 旧 test identity contract 仍是抽样边界；通过 `(normalized table_name, episode_index)` 与 production HY manifest 精确 join。
- manifest 同时支持 `.jsonl` 和 `.jsonl.gz`，文件本身必须通过调用方提供的 SHA256 校验。
- Table 命名统一为 `table_NNN`，因此 `Table014`、`table014` 和 `table_014` 都归一为 `table_014`。
- `table_014` 在 selection 中作为 `excluded_source_groups` 明确记录，包含排除 episode 数量与 episode ID 集合的语义 SHA256。
- 排除后任何 identity 未映射都会 fail closed；最终 selection 还必须覆盖所有排除后 table，避免只从部分 table 抽样。
- 解码复用训练实现的 camera contract、Lance `(episode_index, frame_index)` 查询、JPEG 解码、时间戳检查、letterbox/mask 和 `_mono_sample` ABI，不再依赖 canonical loader 的物理 row-offset `index`。

## 修改范围

- `evaluation/stage_a_metrics.py`：新增 flow warp/FB mask、static flicker、motion EPE 与 temporal geometry 指标。
- `evaluation/stage_a_data.py`：新增 HY manifest 精确 join、Table014 排除和 production Lance 解码路径。
- `evaluation/stage_a_contract.py`：允许 HY selection 使用 manifest identity 做重复窗口校验。
- `evaluation/tokenizer_stage_a.py`：新增冻结 RAFT wrapper、CLI/provenance、HY run/preflight 接线和 report v3 门禁/展示。
- `tests/stereo/test_stage_a_evaluation.py`：新增 gzip manifest、table 归一化、static/dynamic flow 与 temporal geometry 合成测试，并升级 report fixture。

## 验证状态

- `python -m py_compile evaluation/stage_a_metrics.py evaluation/stage_a_data.py evaluation/stage_a_contract.py evaluation/tokenizer_stage_a.py tests/stereo/test_stage_a_evaluation.py`：通过。
- `git diff --check`：通过；仅报告仓库既有 Windows CRLF 转换 warning。
- 本机动态单测：未运行成功，本机 Python 缺少 `torch`，首个异常为 `ModuleNotFoundError: No module named 'torch'`。
- H100 真实运行：未开始。冻结 RAFT-Large 权重已按上述共享路径与 SHA256 就绪；尚未在带 PyTorch 的正式 runtime 中运行新增合成单测或 evaluation。

## 下一步门禁

1. 在带 PyTorch 的固定运行环境先运行新增合成单测。
2. 生成 HY selection，确认 Table014 排除数量、其余 table 覆盖、join missing=0 和 decode rejected 审计。
3. 使用同一 selection/RAFT/teacher 合同重跑 baseline 与 candidate，之后生成 v3 报告；旧 v2 artifact 不与新增指标混报。
