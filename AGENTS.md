# AGENTS.md

## 问题与 bug 修复授权边界

- 只有发现会显著影响技术路线、结果语义、实验可比性、数据或资产安全，或需要用户作出新的实质性
  决策的问题时，才必须暂停并询问用户。典型情形包括：关键参数存在多种不可等价选择、代码实现需要
  重大转向或跨模块重构、配置或数据合同冲突、目标分支或 Git 状态不满足授权边界、服务器 GPU 不可用、
  缺少关键环境/依赖/数据/checkpoint、继续执行可能覆盖产物或消耗未经授权的 GPU 资源。
- 对不改变任务目标、实现路线、结果语义或外部状态的操作性小问题，应在原任务范围内直接纠正并继续，
  无需暂停询问。包括但不限于：误用当前环境不存在的只读工具、只读命令的拼写/参数/引号/转义错误、
  路径探测后改用已确认的正确路径、trailing whitespace 或其他纯格式问题，以及可安全重试且不会产生
  重复写入的瞬时只读失败。必要时在后续进度或最终回复中简短说明，不把它们升级为 blocker。
- 已获授权范围内的局部、低风险、可回退实现修正和对应定向测试，可以直接完成；但不得借此扩大到新功能、
  新公共接口、新依赖、新数据格式、跨模块重构、训练/评估启动、远端环境修改或其他未授权写操作。
- 需要暂停时，先完成解释问题所必需的最小只读取证，然后向用户说明：已确认的证据、根因或当前假设、
  影响范围、不修复的后果，以及可选修复方法。明确给出推荐方案、拟修改的文件或配置、验证方式、风险和
  回退办法，并等待用户在当前对话中明确确认具体方案后再继续。
- 判断标准是“是否需要新的实质性决策或授权”，而不是“是否出现过错误”。发现无关且不阻塞当前任务的
  问题时只记录并报告，不擅自顺带修改；发现多个需要决策的问题时逐项报告，不得借一个问题的授权修复其他问题。

## 分支与 worktree 授权边界

- 未经用户在当前任务中明确授权，禁止创建任何本地或远端 branch、worktree，禁止执行
  `git branch <new>`、`git switch -c`、`git checkout -b`、`git worktree add`，也禁止通过
  GitHub、服务器脚本或其他等价方式间接创建。
- 用户指定了目标分支或现有 worktree 时，所有开发、测试、评估和训练都必须使用该目标；
  不得以隔离开发、保持工作树 clean、分支已被其他 worktree 占用、远端环境限制或命名便利为由
  私自创建替代 branch/worktree。
- 如果继续任务确实需要新 branch/worktree，必须先停止操作，向用户说明必要性、拟用名称、基线
  commit、位置和不创建的具体 blocker，并等待用户明确同意；未收到同意不得继续任何创建动作。
- 发现目标分支有未提交修改、被其他 worktree 占用、无法 fast-forward 或不适合直接修改时，必须
  保持现场并询问用户；禁止自行通过新 branch/worktree 绕过。
- 以上约束适用于本地、H800、B300、H200、跳板机及所有用户账户，优先级高于本文其他任何
  branch/worktree 工作流建议。

## 服务器与登录身份

- H200 是本项目正式双机训练环境：
  - `ssh h200-1`：用户 `frank`，8×H200，训练时对应 `NODE_RANK=0`。
  - `ssh h200-2`：用户 `frank`，8×H200，训练时对应 `NODE_RANK=1`。
  - 两个 alias 都通过 `jump-h200-qinghua`；跳板机也使用 `frank`。
- SSH HostName、端口、密钥和 ProxyJump 以调用方的 `~/.ssh/config` 为唯一来源。仓库文档只使用 alias，不硬编码公网地址。
- 代码中的逻辑节点 ID 是 `h200-qinghua-1` 和 `h200-qinghua-2`，不要与 SSH alias `h200-1` 和 `h200-2` 混淆。

## H200 仓库与 worktree

- 两台 H200 都使用 `frank` 自己的独立完整 clone，不是 worktree：
  `/data/home/frank/projects/Stereo-vison-Tokenizer`
- 当前正式开发分支为 `frank`，upstream 为 `origin/frank`，`origin` 必须保持为：
  `https://github.com/Frankaaay/Stereo-vison-Tokenizer.git`
- 2026-08-20 只读核对基线：两端 worktree 均 clean，HEAD 均为
  `cbf99baf56316c6140009b64f48d737d73966746`。这是核对时点的事实，不是长期固定版本；每次同步或运行前仍须实时复核。

## H200 Git 同步流程

- H200 节点不能直接访问 GitHub，不得在 `h200-1` 或 `h200-2` 上直接执行 `fetch` 或 `pull`。
  GitHub fetch 必须在 `jump-h200-qinghua` 上通过两份 NFS 映射分别执行：

```bash
git -C /data-214-30-239-40/home/frank/projects/Stereo-vison-Tokenizer \
  fetch --prune origin refs/heads/frank:refs/remotes/origin/frank

git -C /data-214-30-239-42/home/frank/projects/Stereo-vison-Tokenizer \
  fetch --prune origin refs/heads/frank:refs/remotes/origin/frank
```

- 跳板机只更新两份 clone 各自的 `refs/remotes/origin/frank`；不得从跳板机执行 merge、checkout
  或修改 working tree。两份 NFS 映射对应两个 node-local clone，必须分别 fetch，不能只更新一份。
- fetch 成功后，必须登录两台 H200，在各自 clone 中以 fast-forward-only 更新 `frank` working tree：

```bash
# h200-1
git -C /data/home/frank/projects/Stereo-vison-Tokenizer \
  merge --ff-only origin/frank

# h200-2
git -C /data/home/frank/projects/Stereo-vison-Tokenizer \
  merge --ff-only origin/frank
```

- fetch 和 merge 前，先在两台 H200 分别核对：

```bash
git -C /data/home/frank/projects/Stereo-vison-Tokenizer \
  status --short --branch
git -C /data/home/frank/projects/Stereo-vison-Tokenizer \
  branch --show-current
git -C /data/home/frank/projects/Stereo-vison-Tokenizer \
  rev-parse HEAD
git -C /data/home/frank/projects/Stereo-vison-Tokenizer \
  remote get-url origin
```

- 任一节点有未提交修改、分支不是 `frank`、upstream 不是 `origin/frank`、origin URL 不符、fetch
  失败、无法 fast-forward，或两端最终 SHA 不一致时，立即停止并报告；不执行 rebase、reset、force
  checkout、删除重建，也不新建替代 branch/worktree。
- 更新完成后必须在两端再次核对 `status --short --branch` 和 `rev-parse HEAD`，确认 worktree clean、
  两端 HEAD 相同，且等于跳板机查询到的目标 `origin/frank` SHA。
- 用户指定精确 SHA 的复现或 smoke 时不跟随分支更新；先按“分支与 worktree 授权边界”确认现有目标
  branch/worktree 或取得新建授权，再在所有节点验证精确 SHA。

## H200 数据与存储拓扑

- `/data/shared` 在两台 H200 上是 node-local 视图，不是共享盘。同名路径不代表内容相同。
- 当前 Stereo OmniTokenizer 工程 pilot 数据和 GT 已分别存放在 `h200-1` 和 `h200-2` 的相同绝对路径；它们是两份 node-local 副本，不是共享存储，使用前仍需分别核对。
- 2026-08-20 同步后的双节点基线：原始 MCAP 均为 100 个，Manifest v2 均为 7 个文件，GT cache 均为 3407 个，8 个 shard success marker 均存在；原始数据、Manifest、GT 和解码审计的 rsync checksum dry-run 差异均为 0。
- 原始 MCAP 数据根：
  `/data/shared/datasets/umi_raw_data0806`
- 当前冻结的 Manifest v2 目录：
  `/data/shared/datasets/umi_raw_data0806_stereo_pilot_v2`
- 训练使用的 pilot Manifest：
  `/data/shared/datasets/umi_raw_data0806_stereo_pilot_v2/pilot_manifest.jsonl`
- Manifest 汇总：
  `/data/shared/datasets/umi_raw_data0806_stereo_pilot_v2/manifest_summary.json`
- 固定 smoke/overfit Manifest：
  - `/data/shared/datasets/umi_raw_data0806_stereo_pilot_v2/smoke_32.jsonl`
  - `/data/shared/datasets/umi_raw_data0806_stereo_pilot_v2/overfit_128.jsonl`
- FoundationStereo GT 根目录：
  `/data/shared/datasets/umi_raw_data0806_stereo_pilot_gt_v2/full`
- 每个 sample 的 GT cache 位于：
  `/data/shared/datasets/umi_raw_data0806_stereo_pilot_gt_v2/full/gt`
- 数据扫描产物：
  - `/data/shared/datasets/umi_raw_data0806_stereo_pilot_gt_v2/full/scan_v1.json`
  - `/data/shared/datasets/umi_raw_data0806_stereo_pilot_gt_v2/full/mask_candidates_v1.json`
  - `/data/shared/datasets/umi_raw_data0806_stereo_pilot_gt_v2/full/final_mask_v1.json`
- H.264 解码完整性审计：
  `/data/home/frank/runtime/stereo-gt-pilot-v2/decode_integrity_audit.json`
- 当前 Manifest v2 有 3407 个有效 sample，split 名称为 `pilot_train`。这是 100 个 MCAP 的工程 pilot，不是正式数据，也没有正式 train/validation/test 划分。
- `datasets` 或 `/data/shared/datasets` 只放数据，不放代码、脚本、虚拟环境、checkpoint 或临时工程文件。
- 本仓库不提交数据集、GT cache、checkpoint、wandb、runs、logs、虚拟环境或缓存目录。

## 当前训练资产与启动边界

- Hy runtime 已在两台 H200 建立相同的 worktree 外路径：
  - 完整 venv：`/data/home/maxliu/runtime/ngadv1-hy`
  - Python：`/data/home/maxliu/runtime/ngadv1-hy/bin/python`
  - 任务私有依赖：`/data/home/maxliu/runtime/ngadv1-hy/deps/py312-v1`
- frozen checkpoints 已在两台 H200 建立内容一致的 node-local 副本：
  - visual tokenizer：`/data/home/maxliu/artifacts/NGADv1/frozen/visual_tokenizer/checkpoint_step_00028057`
  - LAT：`/data/home/maxliu/artifacts/NGADv1/frozen/latent_action/checkpoint_step_00014029`
- 上述 runtime、deps 和 checkpoints 不依赖 NFS；launcher 默认值必须使用这些路径，不得重新指向 `crazy`。
- 当前 PI0.5 和统计资产位于：
  - tokenizer：`/data/home/maxliu/projects/NGAD/data/pi05_assets/paligemma_tokenizer.model`
  - weights：`/data/cache/models/maxliu/ngadv1/pi05_assets/pi05_base.npz`
  - Hy quantile stats：`/data/home/maxliu/experiments/ngadv1_hy_canonical_merge_260726/hy_canonical_dual_tcp_stats_v2_h1h2_train99_1m.json`
- 正式 Hy 双机训练只使用：
  `bash script/train.sh hy-16gpu`
- 未经用户明确授权，不启动训练、评估、预处理、长时间 GPU 任务或正式 resume。
- 启动前必须分别确认两端：精确 Git SHA、clean worktree、Python/依赖合同、frozen assets、Hy node-local manifest、quarantine、dataset contract、GPU 占用和所选 output 路径的实际语义。
- 两端必须使用同一代码 SHA、run ID、master port 和 resolved config，并分别设置 `NODE_RANK=0/1`。
- 发现 GPU 占用时停止并报告进程所有者和命令行；不自动 kill、清理或处理无关进程。

## 运行事实与远端写操作

- 当前运行事实按以下优先级判断：实时 Git SHA、resolved config、run manifest、data contract、实际进程和日志 > 当前代码与 launcher > 历史 docs 和 AGENTS 记录。路径、hash、commit、数据数量和运行状态属于易变事实，使用前必须重新核对；发现冲突时停止并报告，不用旧记录覆盖实时证据。
- 远端写命令遇到 SSH 255、跳板断连、本地等待失败或返回结果缺失时，不能直接认定远端操作失败，也不能盲目重试。必须先只读检查目标状态；只有确认操作未落地且重试具备幂等性时才能继续。

## 代码修改规则

- 修改前先检查 `git status --short --branch`、当前分支和 HEAD。
- 仓库代码、脚本、配置、测试和对应 `docs/` 的开发修改必须在用户明确指定的本地目标
  分支和现有 worktree 中完成；未经用户明确授权不得为此新建 branch/worktree。不得直接在
  B300、H800、H200 等服务器 clone/worktree 中编辑、commit 或 push 源码。
- 标准流程固定为：本地目标 worktree 修改与最小验证 -> 本地 commit -> push 到
  `origin` -> 服务器只读核对 status/branch/HEAD -> 服务器 fetch 后以 fast-forward
  方式更新到已推送的精确 SHA -> 在服务器运行测试、评估或训练。服务器上的运行产物、
  日志和 checkpoint 保持在仓库外输出路径。
- LIBERO 的开发和评测修复必须在本地 `libero-test` 分支的现有 worktree 完成并 push；
  B300 只允许使用 `frank` 账户的 clean `libero-test` clone 拉取已推送 SHA 后执行 eval，
  不得在 B300 现场改代码，也不得把 LIBERO 测试改动放入本地或远端 `main`。
- 服务器上的 Git 身份和上游必须按用户隔离：`frank` 的 clone/worktree 只能跟踪项目
  GitHub 远端及 `frank` 自己的 refs，不得把 `maxliu`、`jiapeilin` 或其他用户的本地
  repo、worktree、裸仓库路径配置为 `origin`、alternate、submodule source 或 fetch
  source，也不得向其他用户的 Git 目录写入 refs、safe-directory 例外或配置。
- 跨用户 checkpoint、数据和模型资产仅可在用户已授权的明确绝对路径下作为只读输入；
  代码、Git 元数据、运行输出、日志、portable 工件和临时文件必须写入当前执行用户自己的
  路径。发现 clone 正在跟踪其他用户的本地仓库时，先停止 fetch，改回项目 GitHub 远端并
  验证 branch upstream/HEAD 后才能继续测试。
- 若本地目标 worktree 不存在、目标分支含未提交修改、push 尚未成功、服务器无法
  fast-forward、或服务器 HEAD 与已推送 SHA 不一致，立即停止；不得退化为服务器现场
  修改来绕过该 blocker。
- 不丢弃未提交修改，不执行 `git reset --hard`、`git checkout -- .` 或 `git clean -fd`，除非用户明确要求。
- 禁止批量删除文件或目录；需要删除时一次只处理一个明确文件。不得使用 `rm -rf`、`Remove-Item -Recurse`、`del /s`、`rd /s` 或 `rmdir /s`。
- 只暂存具名文件，不使用 `git add .` 或 `git add -A`。
- 未经用户明确要求，不 commit、不 push、不创建 PR；即使获得授权，也必须先完成可行的最小验证。
- `docs/` 和 `README` 可随对应代码或实验记录纳入用户授权的提交。
- 禁止私自新建 branch/worktree；任何必要性都必须在创建前向用户说明并取得明确授权。

## 修改、测试与实验记录

- 代码、脚本、配置、实验设置或排查结论发生实质修改，以及完成训练、评估、调试实验或关键测试后，必须同步更新 `docs/`；没有合适文档时，新建当天 topic 文档。
- 文档路径采用 `docs/YY-MM-DD/YY-MM-DD-<topic>.md`，例如 `docs/26-05-03/26-05-03-libero-joint-debug.md`。
- 记录至少包含：改动/实验目的、分支/commit、运行位置、关键命令或脚本、checkpoint/output/log 路径、主要指标、异常现象、当前结论和下一步。
- 正在运行但尚未完成的测试也要记录，状态写为“进行中”，并给出 tmux session、日志路径和后续需要读取的结果文件。
- 每次启动训练、评估、预处理或长时间测试后，必须在用户回复和实验文档中给出 ETA。ETA 要注明估算时刻与依据（例如当前 step、目标 step、最近稳定吞吐、episode 数和单 episode 耗时），并分别说明“训练/测试主体完成”和“checkpoint 验证、closed-loop 等全部后处理完成”的预计时间；不能只报进程已启动。
- 尚未产生足够真实吞吐时，不猜测精确完成时间；先给出基于历史实测的区间并标记为初估，得到至少两个运行采样点后更新。实际吞吐、任务范围或失败重试改变时，要同步刷新 ETA。
- 最终回复前自检：如果本次有代码、脚本或实验配置修改，必须确认对应 `docs/YY-MM-DD/` 文档已同步；若确实无需记录，在最终回复中说明原因。
- 不把大日志、checkpoint、dataset 或生成视频写入文档；只记录仓库内相对路径和必要摘要。

## Agent 输出范围控制

- 用户给出的任务是唯一范围。除本文件明确要求的定向测试和实验记录外，未经用户明确要求，不新增功能、入口、配置、依赖、脚本、文档文件或抽象层。
- 修改前先查已有实现和相邻代码，复用现有模式；不要为同一能力生成旁路实现或重复工具函数。
- 采用能完成需求的最小改动：优先只改直接相关函数、调用点和测试；不做全局格式化、重命名、搬文件、清理未用代码或统一风格。
- 新文件、新依赖、新公共接口、新数据格式、跨模块重构都视为范围扩大。若不是完成任务所必需，先向用户说明必要性并等待确认。
- 修 bug 时优先复现并定位根因；补丁必须小而定向。测试只覆盖被改行为和关键边界，不扩成大型测试框架。
- 发现无关问题时只在最终回复中列出路径、现象和建议；除非用户说“顺手修”“全部修”或明确点名，不直接修改。
- 如果实现过程中 diff 明显变大、需要触碰多个无关模块，或需求理解出现歧义，暂停并用 1-3 句话说明范围变化，再等用户确认。
- 不留下占位实现、伪代码、`TODO` 替代现有逻辑，也不生成“未来可能会用”的代码。每次提交前代码必须处于可运行或可继续验证状态。
- 最终回复前自检：是否只改了必要文件；是否引入了未请求能力；是否存在重复代码、临时文件、无关格式化或未验证假设。
