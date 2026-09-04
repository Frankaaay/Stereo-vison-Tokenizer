# Stage A 不可执行指标精简

## 目的

将 `doc/Stereo Tokenizer统一评测标准.md` 的 Stage A 指标收敛到当前 Stereo Tokenizer 架构与数据合同能够产生可信结果的范围，避免把不可执行指标长期保留为默认验收项。

## 删除项

- rFVD：当前冻结 I3D-FVD 实现不支持项目原生四帧输入；通过复制、扩帧或插帧适配会改变评测对象。
- FVMD：当前没有经过验证、适用于原生四帧合同的冻结实现。
- 真实 GT disparity/depth 指标：当前数据合同没有独立真实几何 GT，因此删除 disparity EPE、D1、AbsRel、真实 GT RMSE/SILog 和 δ1/δ2/δ3。
- left-right consistency 与 stereo reprojection/warp：当前 decoder 只重建目标眼，不输出可配对评价的右眼。
- foreground/background、occluded/non-occluded、robot/end-effector、manipulation object、contact boundary 拆分：当前评测 manifest 不提供这些冻结标注。

## 保留项

- rFID：single-frame 可执行，仅缺冻结实现与正式运行。
- optical-flow warp、static flicker、motion consistency：可以在重建 RGB 上执行，属于待实现而非架构不可测。
- DA3/LAS2-H teacher-relative 几何：保留 relative log-L1、relative log-RMSE、relative log-SILog、mask coverage 和有效样本数，并明确不代表真实几何精度。
- teacher-relative temporal depth/disparity consistency：当前架构可提供四帧输出，待 teacher 时序合同冻结后补充。
- Gate B decoded-future gFVD：属于下游 WAM horizon 评测，不受 Stage A 原生四帧 rFVD 删除影响。

## 验证

- 只修改 Stage A 指标定义及本变更记录；未修改评测代码、配置或实验产物。
- Markdown 文本检查与 `git diff --check` 通过后交付。
