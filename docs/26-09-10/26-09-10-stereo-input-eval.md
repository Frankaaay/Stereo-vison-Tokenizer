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
