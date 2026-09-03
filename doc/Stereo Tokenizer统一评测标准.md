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


## 6. 训练与推理速度

### 6.1 统一测速合同

每次测速必须冻结并记录：

- 代码仓库、branch、commit SHA、配置和 checkpoint；
- 数据集与 manifest、输入分辨率、视角数、帧数和有效样本数；
- GPU 型号与数量、CPU、CUDA、PyTorch、精度、编译后端和分布式设置；
- physical batch size、gradient accumulation、logical batch size 和 worker 数；
- warm-up 次数、正式测量次数、同步方式和随机 seed；
- 是否计入数据读取、CPU→GPU 传输、latent cache、后处理和可视化。

CUDA 测时必须在区间边界同步。除吞吐外，延迟同时报告 P50/P95；冷启动和稳定态分开报告。A/B 必须在相同硬件、运行时、输入合同和计时范围下测试，并保留原始逐步计时与有效样本数。

### 6.2 Tokenizer 训练速度

在本仓库的冻结训练配置下报告：

- physical micro-batch latency、logical optimizer-update latency；
- samples/s、frames/s、input pixels/s；
- generator updates/s；若包含 GAN，再单独报告 discriminator updates/s；
- 单卡与总 GPU peak memory；
- 固定训练样本量的 wall-clock 和 GPU-hours；
- time-to-quality：达到冻结 RGB、感知、时序和 stereo/depth 质量阈值所需时间与 GPU-hours。

当 gradient accumulation、手动优化或 generator/discriminator 更新频率不同时，不得只用框架 `global_step` 计算速度；应使用实际样本计数和各优化器的真实更新计数。数据加载、teacher/伪标签和模型前后向耗时可以作为分项诊断，但主表必须保留端到端训练吞吐。

### 6.3 Tokenizer 推理速度

分别测量并报告：

- encoder latency、decoder latency、encode+decode latency；
- batch=1 的 P50/P95 延迟，以及冻结 batch 下的吞吐；
- observations/s、frames/s、latent tokens/s；
- encoder、decoder和端到端 peak memory；
- 冷启动时间与 warm-up 后稳定态结果。

若部署路径只使用 encoder，则 encoder latency 是主指标，decoder latency 作为重建诊断单列；不得把部署时不会执行的 decoder 计入 observation-to-latent 延迟。随机 VAE 默认使用冻结的推理方式，例如 posterior mean，并在 manifest 中明确记录。

### 6.4 下游 WAM 训练速度

该部分在下游 WAM 仓库、其真实训练入口和冻结配置中测量，报告：

- step/s、samples/s、visual latent tokens/s、action chunks/s；
- forward、backward、optimizer update 和端到端 step latency；
- 单卡与总 GPU peak memory；
- 固定样本数或固定 token budget 的 wall-clock 与 GPU-hours；
- time-to-quality：达到冻结 future prediction loss、action L1/MSE 或 OLS 阈值所需时间与 GPU-hours；
- 一次性 latent 生成与缓存的 wall-clock、GPU-hours、产物大小和 tokens/s。

A/B 必须使用由目标 codebase 冻结的同一 WAM 架构、训练数据、训练 token/sample budget、优化器、精度和硬件。除等计算量对比外，再报告达到相同下游质量阈值时的训练成本；若 A/B latent token 数不同，不能只比较 step/s。

### 6.5 下游 WAM 推理速度

在下游真实推理和 rollout 路径中分段测量：

- 数据读取与预处理；
- 在线 Tokenizer encode，若部署时执行；
- WAM future/action forward；
- action decode/post-process；
- Tokenizer decode，只有部署或评测路径实际需要时才计入；
- 端到端 observation-to-action latency P50/P95；
- action chunk latency、control Hz、horizon 1/2/3 的延迟增长；
- peak memory 和稳定态吞吐。

同步与异步 pipeline 分开报告，并明确 observation-to-action 的起止边界。控制频率必须由端到端延迟和实际调度得到，不能仅由单个网络 forward latency 反推。

### 6.6 跨仓库实现边界与统一结果文件

当前 Tokenizer 仓库中的单个测评执行文件，可以直接完成 Tokenizer 训练/推理测速，以及定义统一结果 schema 和汇总逻辑；它不能在缺少下游 WAM 代码、配置、checkpoint、运行时和真实 rollout 入口的情况下产出可信的 WAM 训练/推理速度。

推荐采用“一个逻辑评测标准、两个仓库各自执行、一个统一结果汇总”的结构：

1. Tokenizer 仓库的 runner 生成 `tokenizer_metrics.json`；
2. 下游 WAM 仓库的 runner 生成 `wam_metrics.json`；
3. 两者引用同一份 `evaluation_manifest.json`；
4. 独立 aggregator 校验合同并生成统一 `scorecard.json`/Markdown 报告。

共享 schema 至少包含 `evaluation_id`、Tokenizer 名称与 checkpoint、各仓库 commit SHA、latent ABI、数据 manifest hash、硬件与运行时、计时范围、样本/token 数、指标单位和原始结果路径。aggregator 只负责合同校验和汇总，不代替任一仓库运行模型。


## 7. 参考方法

- LingBot-VA 2.0：<https://arxiv.org/html/2607.08639>
- RepWAM：<https://arxiv.org/html/2606.13674>
- Cosmos World Foundation Model / Tokenizer：<https://arxiv.org/html/2501.03575>
- FVD content bias：<https://arxiv.org/html/2404.12391>
- FVMD：<https://arxiv.org/html/2407.16124>
- KITTI Stereo 2015：<https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo>

