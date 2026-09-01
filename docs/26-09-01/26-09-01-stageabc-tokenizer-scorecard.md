# Stage A/B/C tokenizer 全测试集 scorecard

## 目的

在完全一致的数据、teacher、后验均值和样本选择合同下，对比 Stage A 44k、Stage B 100k、Stage C 162.5k generator updates，判断 image/video GAN 是否真正改善感知质量，或只是以像素、深度及时间一致性为代价增加锐度。额外将每个四帧 window 的第 0、1、2、3 帧分别作为 mono/stereo single-frame 输入，排查过去只评第 0 帧造成的位置偏差。

## Git 与运行位置

- 本地 worktree：`C:\Project\Stereo-vison-Tokenizer`
- 分支：`hezhou-las2-h`
- 修改前 HEAD：`71645e561eddf9ecd0015802af6e8813a60a0102`
- 运行位置：H200-1，正式启动前记录服务器同步后的精确 commit
- 运行环境：`/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260828/venv`

## Checkpoint 合同

- Stage A：`/data/home/frank/experiments/stereo-three-source-stagea-bs192-h2001-20260829-v3/train/checkpoints/epoch=0-step=44000.ckpt`；直接读取 `stereo_update_counters.generator_updates=44000`、`discriminator_updates=0`
- Stage B：`/data/home/frank/experiments/stereo-three-source-stageb-imagegan-bs192-h2001-20260830-v1/train/checkpoints/best-epoch=0-step=112000.ckpt`；直接读取 `generator_updates=100000`、`discriminator_updates=56000`
- Stage C：`/data/home/frank/experiments/stereo-three-source-stagec-videogan-bs192-h2001-20260830-v1/train/checkpoints/last.ckpt`；直接读取 `generator_updates=162500`、`discriminator_updates=118500`

checkpoint 文件名和 Lightning `global_step` 不作为 generator update 依据。

## 数据与评测合同

- Hy test：正式 `hy_formal_90_5_5_v1.jsonl` 与既有 root aliases
- LIBERO test：正式 `libero_formal_90_5_5_v1.jsonl`
- UMI test：decode-verified manifest 与既有 rectification audit SHA
- mono teacher：固定 DA3 repo/checkpoint SHA
- stereo teacher：固定 LAS2-H repo/checkpoint SHA
- 后验：mean；所有阶段固定相同 case indices
- single-frame：分别报告 `source_0`、`source_1`、`source_2`、`source_3`
- four-frame：单独报告，不把 checkpoint 名或进度条当作更新数

## 指标

- RGB：L1、global PSNR、逐帧 mean PSNR、逐帧 SSIM、逐帧 LPIPS
- four-frame 时间一致性：temporal-delta L1、temporal-delta LPIPS
- teacher-relative depth：relative-log L1、RMSE、SILog，按 view 分开
- 可视化：每个数据集固定典型 cases；四个 single-frame source 分别与同一 four-frame 重建并排输出

标准 rFID/rFVD 本轮不伪造：固定运行环境没有 `torch-fidelity`/CleanFID 或已缓存的标准 Inception 权重；VGG 特征距离不能改名为 rFID。stereo EPE/D1 和 warp error 也需要另行冻结重建后 teacher/flow 协议，本轮结果中明确列为未覆盖，而不是从现有 teacher agreement 推断。

## H200-1 资源边界

2026-09-01 13:52（Asia/Shanghai）只读快照显示 8 卡均由 `melody` 的 WAM 训练占用；GPU 3 剩余约 57.3 GiB，其他卡剩余约 50.9--52.3 GiB。用户已明确授权在现存空闲显存上叠加 eval。流程先仅在 GPU 3 做单 batch smoke 并记录峰值显存；不停止、不修改、不重启现有训练。只有 smoke 同时满足 eval 无异常、无 OOM、现有训练进程仍在，才启动正式评测。

## 状态与结果

- 本地静态编译：通过
- 本地 tensor/CUDA 单测：本机 Python 缺少 Torch；固定 H200 runtime 的定向单测 27/27 通过
- 实现 commit：`1056f81c03b305a1f558d015deef496196565980`，已推送；H200-1 clean clone 已 fast-forward 到同一 SHA
- mono/Libero smoke：单卡 GPU 3、batch 4，`exit_code=0`；四个 source 的指标与 RGB/depth 图均非空；eval 新增显存约 8.5 GiB，结束后完全释放
- stereo/UMI smoke：单卡 GPU 3、batch 12，`exit_code=0`；LAS2-H、三视角、五个模式、LPIPS/SSIM/SILog/temporal-delta 与四组可视化全部跑通；实测约 4.99 秒/批
- 正式全量评测于 2026-09-01 14:12:04 CST 启动；tmux：`stereo-stageabc-scorecard-h2001-v1`
- 输出根：`/data/home/frank/experiments/stereo-tokenizer-stageabc-scorecard-h2001-20260901-v1`
- 运行顺序：Stage A 的 Hy/LIBERO/UMI → Stage B 的 Hy/LIBERO/UMI → Stage C 的 Hy/LIBERO/UMI；8 GPU、每卡 batch 12、指标 microbatch 4，每个数据集 8 个固定 cases × 4 个 source
- 初始主体 ETA：2026-09-01 23:30 至 2026-09-02 02:00 CST；完整性校验、聚合报告和桌面产物预计再需 20--40 分钟。依据为正式参数下单卡 UMI smoke 与上一版 8 卡 evaluator 的历史吞吐，待 Stage A Hy 出现真实进度后刷新
- 启动时直接验证 Stage A `generator_updates=44000`；tmux 存活、既有 WAM 进程仍在、错误扫描为空。正式 ranks 尚处于 torchrun/model 初始化，首个进度 heartbeat 待下一次用户请求时按长任务规则复核

## 用户中止与视角合同 probe

用户发现潜在训练/推理视角合同不一致后，正式全量任务被立即停止。停止后 `stereo-stageabc-scorecard-h2001-v1` tmux、不完整 evaluator ranks 和新增 GPU 显存占用均消失；既有 WAM 训练未被操作，smoke 与部分输出保留。

Stage C 162,500-update checkpoint 加真实 UMI test window 的定向 probe 表明：

- 正式 joint 输入是 `[1,3,2,3,T,256,256]`，但 latent 是 `[1,3,48,1,16,16]`，即每个 view 一个 latent，并非三个 view 合成一个 latent。
- 当前公共 `encode(..., eye_mode="stereo")` 对 `[1,1,2,3,T,256,256]` 明确报 `ValueError: stereo eye mode requires V=3,E=2`，所以单独 stereo pair 的生产调用合同当前确实不可用。
- Encoder 的 temporal attention、spatial path 和 stereo fusion 均不跨 view；三个 view 共享权重。用进程内诊断性 V=1 contract 按 view 单独编码后，FP32 下 single/four-frame 的 latent、RGB 和 depth 都与 joint 调用对应 view 在 `atol=rtol=1e-4` 内一致。latent 最大绝对差约 `3.7e-6`--`5.6e-6`，RGB 最大差约 `3.8e-6`--`3.1e-5`。
- BF16 改变 batch cardinality 时均值差很小，但存在局部较大的最大差；这属于低精度 kernel 数值路径和当前重建异常值的放大，不能误判为跨 view 信息融合。

结论：核心网络语义允许每个 stereo pair 独立编码；真正缺口是 API 没有接收 `V=1` 和显式 `view_index/search_radius` 的合同及回归测试。生产修复前不能声称现有 checkpoint 已支持下游逐 pair 调用。

## Hy wrist 快速泛化 probe

按用户要求不跑全量，只选 4 个固定 Hy test episode。对每个 episode 同步读取 `cam_high`、`cam_left_wrist`、`cam_right_wrist`，分别测试 window source 0/1/2/3 的 single-frame 以及 four-frame。使用 Stage C 162,500-update checkpoint；不运行 DA3，指标仅覆盖 RGB。输出位于：

`/data/home/frank/experiments/stereo-tokenizer-stageabc-scorecard-h2001-20260901-v1/hy-wrist-quick-cases`

并复制到：

`C:\Users\Frank\Desktop\stereo-tokenizer-hy-wrist-quick-cases-20260901`

四个 single-frame source 的范围：

| Camera | RGB L1 | LPIPS | SSIM |
|---|---:|---:|---:|
| head | 0.01794--0.01814 | 0.04724--0.04852 | 0.88955--0.89227 |
| left wrist | 0.04231--0.04345 | 0.14209--0.14624 | 0.75643--0.76310 |
| right wrist | 0.05551--0.05995 | 0.15063--0.15430 | 0.75814--0.76186 |

four-frame 分别为：head `L1=0.02362, LPIPS=0.04919, SSIM=0.88767`；left wrist `0.04718, 0.15002, 0.74141`；right wrist `0.05906, 0.16461, 0.74933`。

腕部相机相对 head 的退化远大于 source 0/1/2/3 间差异，说明主要问题是训练只覆盖 Hy `cam_high` 带来的相机域偏移。可视化抽查与指标一致：腕部细节更模糊，并有更明显的局部高饱和彩色伪影；head 也存在物体区域的局部彩色伪影，支持继续调查 GAN 阶段异常值的必要性。该结论仅来自 4 个固定 case，用于快速方向判断，不替代全 split 统计。

## Stereo joint-vs-separate 与 GAN 爆点归因快检

使用 4 个固定 UMI test case，对 single-frame 与 four-frame 分别生成：左右眼输入、三 view joint encode 重建、逐 stereo pair 诊断 encode 重建、自动按该 case 的 p99.9 差值放大的热图；同时用同样 case 比较 Stage A/B/C。输出复制到：

`C:\Users\Frank\Desktop\stereo-joint-vs-separate-stageabc-quick-cases-20260901`

Joint-vs-separate：

- FP32：single/four 的 RGB mean abs diff 分别约 `6.17e-8`/`6.77e-8`，最大约 `2.67e-5`/`6.52e-5`；latent、RGB、depth 全部通过 `atol=rtol=1e-4`。两种 encode 在网络语义与可视重建上等价。
- BF16：RGB mean abs diff 约 `5.33e-4`/`4.16e-4`，p99.9 均约 `0.00977`，但异常像素最大差可到 `0.25`/`0.75`。整体质量与 L1 几乎不变，差异集中在模型本身已经不稳定的极端位置。热图为看清差异做了强烈自动放大，不能按热图亮度理解为可见重建误差。

同 case 的 GAN 阶段归因：

| Stage | Mode | output range | `abs(output)>1` | p99.9 abs error | max abs error |
|---|---|---|---:|---:|---:|
| A no GAN | single | `[-0.727,0.656]` | 0 | 0.354 | 0.922 |
| A no GAN | four | `[-0.598,0.699]` | 0 | 0.389 | 0.943 |
| B image GAN | single | `[-9.19,5.31]` | 0.253% | 1.772 | 8.793 |
| B image GAN | four | `[-9.50,5.72]` | 0.248% | 1.771 | 9.455 |
| C video GAN | single | `[-40.5,39.0]` | 0.157% | 1.612 | 40.098 |
| C video GAN | four | `[-86.5,64.5]` | 0.199% | 2.237 | 86.353 |

Stage A 没有 `|output|>1` 像素；Stage B 首次出现明显彩色爆点与大幅越界；Stage C 的异常像素比例略低于 Stage B，但极值幅度进一步扩大约一个数量级。因此在这 4 个固定 case 上，Image GAN 是爆点首次出现的明确阶段，Video GAN 进一步放大极值；腕部 OOD 不是其必要原因，只会改变暴露位置和严重程度。
