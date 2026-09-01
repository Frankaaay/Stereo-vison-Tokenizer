# Stereo Tokenizer 统一评测标准

## 1. 目的

本标准用于比较两个或多个视觉 Tokenizer/checkpoint，回答两个相互独立的问题：

1. Tokenizer 是否准确、稳定且高效地保存 RGB、时间和双目几何信息；
2. 冻结 Tokenizer 后，其 latent 是否真正有利于世界动作模型（WAM）的动作预测和闭环控制。

评测分为两部分：

- **第一部分  A：Tokenizer 本体指标**，直接在冻结 Tokenizer 上计算；
- **第二部分 B：轻量 WAM 离线 A/B**，冻结 Tokenizer A/B 后训练严格受控的下游模型；
- **第二部分  C：RoboTwin 完整任务 rollout**，以闭环任务成功率作最终裁决。

 A 用于排除重建、时间、几何和系统效率退化；B 用于解释 latent 的动作可用性；C 的闭环成功率是最终选择 Tokenizer 的主要证据。

## 2. 控制变量

轻量 WAM 的精确规模、每个任务的 rollout 次数，以及 Tokenizer A/B 是否具有相同 latent ABI还未确定

执行者必须先记录：
- Stereo Tokenizer A/B 的仓库路径、分支、完整 commit SHA、checkpoint 路径和直接训练计数；
- 目标 WAM/RoboTwin 仓库路径、适用的 `AGENTS.md`、分支和完整 commit SHA；
- 实际模型配置文件、实例化后的参数量、层数、宽度、attention、action head 和 conditioning；
- RoboTwin 版本、任务 split、Easy/Hard 配置、每任务 rollout 数、初始状态 seeds、episode timeout 和成功判定实现；
- A/B 的 latent shape、dtype、数值范围、posterior 策略、view/time 排列、token 数和 normalization；
- 训练数据 manifest、action normalization、horizon 定义和所有配置文件的 SHA256。

若 A/B ABI 相同，则使用完全相同的 WAM 配置直接比较。若 latent shape、token 数或维度不同，不得静默增加额外模型容量或改变序列预算；必须先冻结以下比较合同之一：

- **等 token-budget 主实验**：统一每个 observation/window 进入 WAM 的 token 数和 WAM 主干参数量；
- **native-budget 补充实验**：各 Tokenizer 使用原生 token 数，同时报告真实吞吐、显存和延迟。

## 3. 第一部分：Tokenizer 本体指标

### 3.1 固定评测合同

所有 Tokenizer/checkpoint 必须使用完全相同的：

- train/validation/test split 和数据 manifest；
- sample ID、episode ID、窗口起点和帧索引；
- 图像分辨率、归一化、裁剪和 padding；
- mono/stereo × single/four-frame 四种模式；
- posterior mean，不允许一方采样而另一方取均值；
- RGB/depth/disparity valid mask；
- metric backbone、权重、预处理和软件版本；
- 精度、硬件、样本数和聚合方法。

每次结果必须保存：

- 实际 cwd、Git branch 和完整 commit SHA；
- checkpoint 路径及 checkpoint 内直接训练计数；
- resolved config 和 manifest SHA256；
- latent shape、token 数和 dtype；
- 总样本数、有效像素数和各模式样本数；
- metrics JSON、运行日志、退出码和固定可视化案例。

### 3.2 RGB 重建

每个数据集、模式和 camera/view 分别报告：

- RGB L1 ↓；
- RGB MSE ↓；
- PSNR ↑；
- SSIM ↑；
- LPIPS ↓；
- rFID ↓：single-frame；
- rFVD ↓：four-frame。

聚合同时包含 per-camera、per-mode、macro average，以及样本级 P50/P90/P99。额外报告 NaN/Inf、无效输出、输出范围和 `abs(output)>1` 像素比例，防止少量爆点被平均指标掩盖。

### 3.3 时间一致性

four-frame 模式报告：

- temporal-delta L1 ↓；
- temporal-delta LPIPS ↓；
- optical-flow warp error ↓；
- 静态区域 flicker error ↓；
- 动态区域 motion consistency error ↓；
- FVMD ↓，当其依赖和冻结实现可用时加入。

rFVD 不单独承担时间一致性结论，因为传统 I3D-FVD 可能偏向单帧内容。至少同时保留一种显式 temporal-delta/warp/motion 指标。

### 3.4 Stereo 与深度

有独立真实 disparity/depth ground truth 时报告：

- disparity EPE ↓；
- D1 ↓；
- AbsRel ↓；
- RMSE ↓；
- SILog ↓；
- δ1/δ2/δ3 ↑；
- left-right consistency error ↓；
- stereo reprojection/warp error ↓；
- temporal depth/disparity consistency ↓。

按 foreground/background、occluded/non-occluded、robot/end-effector、manipulation object、contact boundary 和 camera/view 拆分。

没有独立 GT 时，只能报告 DA3-relative depth agreement 或 LAS2-H-relative disparity agreement，并明确标为 teacher-relative；不得将其描述为真实 depth/disparity accuracy。

### 3.5 Bottleneck 与系统效率

所有结果同时报告：

- 输入 shape、latent shape 和 dtype；
- spatial/temporal/view compression ratio；
- 每帧 token 数和每窗口总 token 数；
- latent channel/dimension；
- encode/decode latency；
- samples/s 或 frames/s；
- 峰值 allocated/reserved memory；
- Tokenizer 参数量。

## 4. 第二部分：冻结 Tokenizer 的轻量 WAM 离线 A/B （模仿lingbot va2.0）

### 4.1 方法来源与复现范围

Gate B 采用 LingBot-VA 2.0 的受控 Tokenizer A/B 原则：冻结 Tokenizer，保持 token budget、下游架构、RoboTwin 数据与后训练设置一致，并按 Easy/Hard 和不同 horizon 比较。离线的 gFVD、PSNR/SSIM、OLS 和闭环分组诊断参考 RepWAM。

本文不将尚未从目标 codebase 解析的 `300M–500M` 或 LingBot-VA 2.0 的 `1.3B` 写成强制规模。实际模型规模由代码基线解析门禁冻结；对外只声称复现受控 A/B 方法，不声称复现原论文的绝对结果，除非模型、数据和训练合同均一致。

### 4.2 Latent 生成

1. 冻结 Tokenizer A/B 的全部参数；
2. 使用同一份 RoboTwin demonstration 和完全相同的 sample/window 索引；
3. 使用 posterior mean 和 deterministic encode；
4. 分别生成 A/B latent cache；
5. 保存 sample ID、observation/action 对齐、latent ABI、Tokenizer provenance 和 cache manifest；
6. 检查 NaN/Inf、重复/缺失样本和未来信息泄漏；
7. 抽样验证缓存 latent 解码结果与在线 encode-decode 一致；
8. 基于各自训练 split 冻结 latent mean/std，不从 validation/test 拟合 normalization。

### 4.3 WAM 公平性合同

A/B 必须保持：

- 同一 WAM 主干、action head、conditioning 和初始化方法；
- 同一训练/验证 split；
- 同一 action normalization；
- 同一 batch、optimizer、学习率、scheduler、训练 steps 和有效训练 token 数；
- 同一精度、硬件类型和数据顺序策略；
- 同一 horizon 1/2/3 定义；
- 至少三个训练随机 seed；
- 除 Tokenizer 及已冻结输入 adapter 外无其他差异。

### 4.4 离线指标

#### 未来状态预测

按 horizon 1/2/3 分别报告：

- normalized future latent MSE ↓；
- normalized future latent L1 ↓；
- decoded future RGB L1 ↓；
- decoded future PSNR ↑；
- decoded future SSIM ↑；
- decoded future LPIPS ↓；
- gFVD ↓。

A/B latent 坐标系和尺度可能不同，原始 latent MSE 不可直接横向比较。必须使用各自 train split 的冻结 mean/std 标准化；跨 Tokenizer 的主要公共空间证据是 decoded future 指标。

#### 动作预测

在相同 action normalization 下报告：

- action L1 ↓；
- action MSE ↓；
- position error ↓；
- rotation error ↓；
- gripper accuracy/F1 ↑；
- OLS@代码基线阈值 ↑；
- action chunk endpoint error ↓；
- action smoothness/jerk ↓。

OLS 阈值不得默认写死为 `0.03`。优先复用目标 codebase 已实现并由其 benchmark 使用的阈值；若 codebase 没有 OLS 合同，新增阈值属于评测设计选择，需在启动前获得确认并记录依据。

结果拆分为 Easy/Hard、seen/unseen、horizon 1/2/3、短/中/长任务和训练 seed。

### 4.5 C阶段

以下条件全部满足后才进入完整 rollout：

- A/B 的三个正式 seed 均正常退出且 artifact 完整；
- 无 NaN、latent contract 违规、数据错配或未来泄漏；
- action/OLS 没有跨 seed 的稳定显著退化；
- decoded future LPIPS/gFVD 没有严重退化；
- horizon 增长曲线没有明显提前崩溃；
- 差异超出 seed 方差，或存在一致、可解释的统计趋势。


## 5. 第二部分： C — RoboTwin 完整任务 Rollout

### 5.1 固定合同

沿用 Gate B 的正式 checkpoint，不为 rollout 单独调整 A/B 超参数。根据目标 codebase 冻结：

- simulator 和 RoboTwin 版本；
- task list、Easy/Hard 和 seen/unseen split；
- 每任务 rollout 数；
- 初始状态 seeds；
- camera、control frequency 和 action chunk；
- replanning frequency、episode timeout 和成功判定；
- inference precision、硬件和随机性设置。

每任务 rollout 数必须来自目标 benchmark/codebase 的正式设置；若代码中没有唯一设置，停止并由用户确认，不由评测脚本自行选择。

### 5.2 主指标

报告：

- 全任务平均 success rate ↑；
- Easy success rate ↑；
- Hard success rate ↑；
- horizon-1/2/3 success rate ↑；
- task progress score ↑，若 codebase 有冻结定义；
- average completion steps ↓；
- timeout rate ↓；
- collision rate ↓；
- invalid-action rate ↓。

每个任务至少保存成功次数/总 rollout 数、成功率、初始 seeds、训练 seed，并在样本量允许时报告 bootstrap 95% confidence interval。模拟器或运行时失败必须与策略失败分开统计。

### 5.3 失败分类

统一归类：

- instruction grounding failure；
- target localization failure；
- grasp/contact failure；
- inaccurate placement；
- wrong-arm/wrong-object；
- temporal drift；
- action oscillation；
- collision；
- timeout；
- simulator/runtime failure。


## 7. 参考方法

- LingBot-VA 2.0：<https://arxiv.org/html/2607.08639>
- RepWAM：<https://arxiv.org/html/2606.13674>
- Cosmos World Foundation Model / Tokenizer：<https://arxiv.org/html/2501.03575>
- FVD content bias：<https://arxiv.org/html/2404.12391>
- FVMD：<https://arxiv.org/html/2407.16124>
- KITTI Stereo 2015：<https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo>

