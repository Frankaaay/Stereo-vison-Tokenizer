# Tokenizer/WAM 速度评测标准补充

## 目的

在统一评测标准中补充 Tokenizer 与下游 WAM 的训练、推理速度指标，并明确跨仓库的测量和汇总边界。

## 版本与位置

- 分支：`hezhou-las2-h`
- 基线 commit：`5c7e638d5d4f446dc8140cd632b7b7780351e114`
- 修改文档：`doc/Stereo Tokenizer统一评测标准.md`

## 修改内容

- 新增 Tokenizer 训练速度合同：真实样本/帧吞吐、优化器更新、显存、GPU-hours 和 time-to-quality。
- 新增 Tokenizer 推理速度合同：encode/decode 分项、P50/P95、稳定态吞吐和 peak memory。
- 新增下游 WAM 训练速度合同：samples/tokens/action chunks 吞吐、GPU-hours、time-to-quality 和 latent cache 成本。
- 新增下游 WAM 推理速度合同：observation-to-action 延迟、control Hz、horizon 延迟增长和分段耗时。
- 明确实现边界：两个仓库各自由原生 runner 产出指标，通过共享 evaluation manifest 和结果 schema 汇总，不在 Tokenizer 仓库复制或隐式调用 WAM 实现。

## 验证与结论

- 本次仅修改评测文档，没有运行训练、推理、GPU 或服务器任务。
- 速度阈值、WAM 规模、rollout 次数和 latent ABI 均未预设，后续必须从实际目标 codebase 解析并冻结。
- 使用 Markdown diff 和 `git diff --check` 做静态验证。
