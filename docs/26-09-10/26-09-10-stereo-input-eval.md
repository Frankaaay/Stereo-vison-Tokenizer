# M48/D48/S48 40k evaluation

用户批准 2026-09-09 的评测方案，并明确允许共享 H200-1 GPU。
当前 GPU 每卡约 5719 MiB 占用，先使用 GPU0、FP32/TF32 off、BS4、单进程。
保留三项用户文档改动，不纳入提交。

复用原生 LeRobot/Hy/LIBERO test loader、checkpoint strict loader、LAS2-H 和
现有 native metric functions，不改训练代码或旧 latent eval 文件。
固定每臂 40k checkpoint，原始 UMI test split 按 seed1234 哈希选 128 episodes ×8 windows。
smoke 8×2；S48 diagnostic 64×4；Hy/LIBERO RGB regression 各16×2。
不足数量时不重复凑样本，以 selection 文件实际数量为准。

每 window 评估 source0/1/2/3 与 four-frame。学生输入通过训练时的
_prepare_student_batch 路由；correct/same_left/fusion_off/shuffle/shift/time_reverse
共用正确原始输入生成的、逐样本落盘校验的 target 和 valid mask。
shift 分数使用共同内区，另记录 RGB 边界误差；time_reverse 仅 four-frame。
不执行 eye_swap。

逐 sample/view JSONL 保存 target checksum、几何、RGB、temporal、latent 和延迟指标。
统计先 episode 内等权平均，再 episode 等权及配对 bootstrap 2000 draws、95% CI。
主表包含三臂和 Hy/LIBERO 回归；诊断单独分组。编码延迟采用3次预热/5次测量，
共享 GPU 的延迟仅供描述，不能视为隔离环境的严格效率对照。
样例固定为首个哈希选中 sample，输出 PNG；shift 输出曲线。

入口：doc/frank/h2001-stereo-eval-20260910.sh smoke|full。
输出根：/data/home/frank/experiments/stereo-input-eval-h2001-20260910-v1。
先做定向单测和 smoke，检查有限指标、所有输入路径、target checksum 和峰值显存，
通过后再启动 full（main → diagnostic → regression → report）。

第一版 7ba668d 的 UMI smoke 已通过（exit0）：16 windows、2688 条记录、
峰值 allocated 3.404 GiB、主体 56.96 秒，失败日志为空。输出 v1 保留。
补充回归 smoke、深度样例、完整性断言和扰动配对 CI 后使用 v2 输出根。
评测进程限制为 GPU 总显存的15%，约21 GiB，避免对共用任务造成大幅显存挤占。
Fusion confidence/attention entropy 仅适用于执行 fusion 的模型，不进入 M48 缺失指标的
配对比较；共有指标仍严格要求相同 episode 集合。

## 正式启动

代码 SHA `249b58a3085b59ad6a990faefd08d6159739a7bc`，本机/远端源码同步，
训练实现相对原 S48 启动代码无差异。H200-1 定向测试 5/5 通过。
第二版 smoke exit0：UMI 16 windows + Hy4 + LIBERO4，共2988条记录，
逐条指标有限，UMI16个 target checksum 各自跨模型/条件唯一。
184个主对照指标 CI、235个扰动指标 CI 已生成，RGB/深度固定样例和 shift 曲线均存在。
峰值 allocated 3.404 GiB，UMI 主体56.54秒、Hy6.42秒、LIBERO4.05秒。

2026-09-10 10:05:55 CST 正式启动：tmux `stereo-eval-full-h2001-20260910`。
输出：`/data/home/frank/experiments/stereo-input-eval-h2001-20260910-v2/full`。
日志：`/data/home/frank/experiments/stereo-input-eval-h2001-20260910-v2/full.log`。
退出标记：同根 `full.exit_code.txt`。预计主体25–40分钟，报告计算再约1–2分钟，
共用 GPU 负载可能改变实际耗时。正式结果不使用 smoke 指标。

10:07:32 CST 一次健康检查：main 已到19/255 batches，峰值 allocated 3.181 GiB，
GPU0 总显存10417 MiB（含原任务），GPU1–7 保持原占用5719 MiB。
实际主选择为128 episodes、1019 windows：127 episodes各8个，1个仅3个有效窗口；
按既定方案不重复补足，三臂共用这1019个窗口，episode聚合仍等权。

## 修复空监督与共享中心

v2 正式评测在第33 batch退出（exit1）。样本6068356d9ee6a5ab3f2d1836d5dd2946:000468
右手source2有效像素0，其余视角仍有效。根因为逐view切分后调用中心化函数，
同时错误地将原始三视角共享中心改为各视角独立中心。v1/v2几何分数作废，文件保留。

修复：完整sample三视角按原函数中心化，再逐view统计。单view无监督时几何为null，
记录geometry_evaluable/geometry_coverage/geometry_valid_pixels；整sample无监督也标记null。
RGB独立计分，任何预测NaN/Inf仍直接失败。配对统计跳过null但不把它当0；
episode内先各view/source平均，再view等权，避免缺失监督改变视角权重。
三模型共享target/mask，shift族共用内区mask，保持一致的评测支持集。
增加共享中心、单view空mask、全sample空mask及缺失值配对统计测试。
修复后使用v3新输出，从同一固定样本选择重跑，不覆盖旧输出。

修复 SHA `70ce093`，H200-1 定向测试7/7通过。实际失败样本缓存mask回归通过：
source2有效像素38768/8896/0，前两视角可计分，右手几何为null。
独立合成数据检查 RGB/temporal 指标与原公式差异小于1e-5。
v3 smoke exit0、2988条记录、所有可评测指标有限，16个UMI样本target哈希跨条件一致，
峰值allocated3.404GiB。2026-09-10 10:23:27 CST 重启正式评测。
tmux `stereo-eval-full-v3-h2001-20260910`；输出根改为
`/data/home/frank/experiments/stereo-input-eval-h2001-20260910-v3`。
新旧main selection逐项完全一致，仍为1019 windows，未剔除失败样本。

10:26:46 CST 正式健康检查到52/255 batches，已越过旧故障点33，session活跃，
峰值allocated3.181GiB。原故障sample/source2/righthand在三模型中均写入几何null、
有效像素0，RGB指标正常，target checksum一致。预计剩余约25–35分钟，含诊断与报告。
