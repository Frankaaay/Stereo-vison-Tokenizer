# Wan2.2 VAE H100 对比评测准备

## 已对齐范围

- Wan-AI/Wan2.2-TI2V-5B 的 Wan2.2_VAE.pth 对比 GAN-off 124k checkpoint。
- H100 独立环境；UMI/LIBERO 原 selection 全部样本；RGB 与时间指标，不新增 rFID 或几何指标。
- Wan 四帧末尾重复最后一帧形成五帧，解码只评分原四帧；StereoVAE 保持原四帧。单帧四个 source 分别独立评估。报告记录不同 latent 预算。
- 用户允许保留现有未提交 Wan 架构设计文档，在当前 hezhou-las2-h 分支工作；尚未授权 commit/push。

## 当前代码与验证

- 本地 cwd：C:\Project\Stereo-vison-Tokenizer。
- H100 clone 实际位置：/gpfs/jiuquyun/projects/Frank/stereo-vae/Stereo-vison-Tokenizer（指南中的另一位置不存在）。
- 本地 fetch/pull 完成；远端 clean clone 从 18ac10c fast-forward 到 b03651418dc7dfb2c90eb3f7be35a09d6d518f48，与本地一致。
- H100 Slurm CPU：`srun --partition=h100 --qos=cpu --cpus-per-task=4 --mem=16G --time=00:10:00` 运行 `python -m pytest tests/evaluation/test_stage_a_evaluation.py -q -p no:cacheprovider`，21 passed，exit 0，16.54 秒。
- H100 Slurm CPU：通过现有 `_decode_hy_selection_record`，对旧 test 中 table_012/016/018/020 各一个 episode 解码 [0,3,6,9] 四帧，全部成功，video [3,1,3,4,256,256]、rgb_valid_mask [3,1,4,256,256]。这不是全 selection 解码或 GPU 评测证据。

## 历史 selection 与 Hy 核对

- 124k 报告根：/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/stage-a1-stagea-threeview-update124000-20260903-b3bfa1c。
- 实际 quality 目录有 9 份 UMI/LIBERO 文件，没有 Hy；历史 job-status.json 记录 COMPLETED/exit 0。
- 原 UMI selection 1024 窗口、LIBERO selection 256 窗口仍存在，文件 SHA256 与历史质量 artifact 完全一致。
- 原 selection 所在目录：/gpfs/jiuquyun/projects/hezhou/experiments/stereo-tokenizer-stage-a/20260902-stagec-update162500-baseline-v1/selections。仅只读复用原评测相关合同，不以他人代码作为执行来源。
- 已检查原 Stage A 目录列出的 v2/v3/v4/v5 quality，没有找到后续 Hy 成绩或 Hy selection；不据此宣称全服务器不存在任何 Hy 结果。
- 旧 Hy identity：contracts/checkpoint-stagec-update162500/hy-split-identities.json，旧 test 2897 episodes，table_014 占 548。
- H100 table_014 实时不存在；其余 table_012/016/018/020 均存在。排除后 2349 episodes 与 Frank 的 hy.jsonl 精确 join，missing=0。
- Frank manifest：/gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/h100-manifests-20260902-ebb013a/hy.jsonl。该 manifest 自带 train/val 划分，不作为旧 test 身份来源；必须由旧 identity join 固定 test。
- Hy 没有找到可直接复用的原 selection。已向用户询问新增冻结 selection 的规模；未擅自生成或提交全量评测。

## Wan 资产与环境

- 官方代码：/gpfs/jiuquyun/projects/Frank/stereo-vae/external/Wan2.2-42bf4cf；固定 commit 42bf4cfaa384bc21833865abc2f9e6c0e67233dc。
- 权重 revision：921dbaf3f1674a56f47e83fb80a34bac8a8f203e。
- 权重：/gpfs/jiuquyun/checkpoints/Frank/stereo-vae/wan22-ti2v-5b/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth。
- 经 .partial 下载后校验并重命名，大小 2818839170 bytes，官方 SHA256 20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36，已写校验 sidecar。
- 独立环境：/gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/wan22-eval-20260911，Python 3.12.3。
- 离线安装因缓存无 torch 失败，随后使用官方 PyTorch cu126 源安装固定版本。未修改原 train 环境。
- 独立环境安装完成；Torch 2.7.1+cu126、torchvision 0.22.1+cu126、einops 0.8.2、numpy 1.26.2、pytorch-lightning 2.5.6、torchmetrics 1.9.0、av 16.0.1、pylance 10.0.0、pyarrow 23.0.0。完整依赖冻结在环境目录 requirements.freeze.txt。
- 新环境在 Slurm CPU 中再次运行 Stage A 定向测试，21 passed，exit 0，22.35 秒。
- 新环境通过 Slurm CPU 使用官方 Wan2_2_VAE 加载真实权重，704688668 参数、FP32，exit 0；没有 GPU encode/decode 证据。
- H100 的 GAN-off checkpoint 重新计算 SHA256，与 605be7940202b7f0aff2380ac4a99dfe1e15d20847ab0377d3e7b2a27352f7b7 一致。
- Slurm Job 4997（旧环境测试）、4998（四表真实解码）均由 sacct 确认 COMPLETED/0:0。

## 待实施范围

新增 evaluation/wan/ 的官方模型适配与评测入口，复用现有 Stage A reader/指标，新增必要定向测试。训练代码不变。正式提交运行前需确定 Hy selection、完成新环境验证和 Wan 适配 smoke，并按用户授权完成代码 commit/push 与远端同步。

## 用户确认后的实现

- 用户确认 Hy 排除 table_014 后全部 2349 个 test episode 的全部有效非重叠窗口；UMI/LIBERO 使用原 selection 全部样本。
- 新增 evaluation/wan/adapter.py：固定官方 source/checkpoint SHA，FP32 posterior mean，项目值域 [-0.5,0.5] 与官方 [-1,1] 转换，逐目标视角编码，四帧补末帧为五帧、只评分前四帧，保留真实 latent。沿用官方输出 clamp；raw overshoot 不作为跨模型质量结论。
- 新增 evaluation/wan/selection.py：精确复用旧 test identity 与当前 Hy manifest，逐 episode 枚举所有窗口；不使用现有每 episode 仅一窗口抽样策略。任意缺失、越界、解码失败均停止，不静默删样本。
- 新增 evaluation/wan/run.py：同一 batch、target、mask 同时评测 GAN-off/Wan；复用 StageA1MetricSuite、冻结 VGG LPIPS 与 RAFT。逐样本保存共同输入/target/mask 校验和；输出双方指标、样本数量、latent ABI、代码/资产/环境 provenance。
- 新增 tests/evaluation/test_wan_evaluation.py：多 batch/view 值域与帧映射、非法输入、全窗口边界与 table_014 排除。
- 新增 doc/frank/h100-wan22-eval-20260911.sh：prepare（测试与冻结 Hy selection）、smoke、full。一次使用单进程/单 GPU；9 个原 UMI/LIBERO cell 与 Hy 三视角 cell 顺序执行。
- 本地 Python py_compile 与 bash -n 通过，git diff --check 通过。本地没有 torch，因此新测试和 GPU smoke 尚未运行；旧环境与新环境此前 21/21 通过不等于新增入口通过。
- 当前变更尚未 commit/push。已向用户请求只提交本次具名文件并继续 H100 eval，保留原有架构设计文档改动；未获得该项回复前不绕过 H100 已提交源码运行规则。
