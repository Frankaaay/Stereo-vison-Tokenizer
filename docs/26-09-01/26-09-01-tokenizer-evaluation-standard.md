# Tokenizer 统一评测标准文档记录

## 目的

将 Tokenizer 评测冻结为两部分、三道 Gate：Tokenizer 本体指标、冻结 Tokenizer 的轻量 WAM 离线 A/B，以及 RoboTwin 完整任务 rollout。

## Git 与位置

- 本地 worktree：`C:\Project\Stereo-vison-Tokenizer`
- 分支：`hezhou-las2-h`
- 修改前 HEAD：`e17441bc74e119d7cbcd40e4805e2fb15dbeef54`
- 主文档：`doc/Stereo Tokenizer统一评测标准.md`

## 本次变更

- 新增统一评测标准主文档；
- Gate A 覆盖 RGB、时间、Stereo/depth、bottleneck 和系统效率；
- Gate B 采用 LingBot-VA 2.0 的受控 Tokenizer A/B 原则，并使用 RepWAM 的离线诊断分组；
- Gate C 使用 RoboTwin 完整任务 rollout 成功率作为最终主要证据；
- 不预设尚未从目标 codebase 确认的 WAM 参数量、每任务 rollout 数或 A/B latent ABI；
- 增加代码基线解析门禁，要求在任何 latent 生成、WAM 训练或 rollout 前生成并审批 evaluation manifest；
- 明确 raw latent MSE 不可跨不同 latent 坐标系直接比较，要求基于各自 train split 冻结 normalization，并以 decoded future 指标作为公共空间证据；
- 明确 OLS 阈值优先来自目标 codebase，不默认写死 `0.03`。

## 验证

- 本次仅新增 Markdown 文档，不修改代码、配置、训练或评测实现；
- 验证主文档和记录文件均可读取；
- 验证 Git diff 仅包含上述两个具名文档；
- 未运行 GPU、训练、评测、RoboTwin 或远端操作。

## 未完成门禁

正式执行 Gate B/C 前仍需用户指定或确认目标 WAM/RoboTwin codebase。届时必须只读解析实际模型规模、rollout 配置和 A/B latent ABI，不能从本仓库或历史记录推断。

