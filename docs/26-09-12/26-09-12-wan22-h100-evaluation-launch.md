# Wan2.2 / GAN-off H100 评测提交

- 用户明确授权本次 commit/push，并在 smoke 通过后启动正式评测。
- 实现 commit：7cc505b（hezhou-las2-h），已 push，H100 clean clone 已 fast-forward 同步。原有 Wan 架构设计文档改动未纳入提交。
- H100 repo：/gpfs/jiuquyun/projects/Frank/stereo-vae/Stereo-vison-Tokenizer。
- Python：/gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/wan22-eval-20260911/bin/python。
- 入口：doc/frank/h100-wan22-eval-20260911.sh prepare|smoke|full。
- 输出根：/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/wan22-ganoff124k-20260911-v1（沿用已冻结的目录名）。
- prepare Job 5231：COMPLETED/0:0；24/24 定向测试通过；Hy selection 2349 episodes、180975 windows，排除 table_014 的548 episodes。
- selection：输出根/hy-all-test-excluding-table014.json。窗口生成不等于全量解码成功；正式运行逐窗口解码，失败不跳过。
- smoke Job 5232：debug，1 GPU/4 CPU/64 GiB/1小时，依赖 afterok:5231；覆盖全部10个评测 cell，各1个共同样本。日志 smoke.log；退出标记 smoke.exit_code.txt。
- 正式 Job 5233：normal，1 GPU/4 CPU/128 GiB/48小时，依赖 afterok:5232。全部 smoke 子进程成功后才可调度；日志 full.log，退出标记 full.exit_code.txt。
- 首次申请 long/1 GPU/7天被 QOSMinGRES 拒绝，未产生正式作业。实时 QOS 显示 long 至少8 GPU；因此保持1 GPU，改用 normal 48小时，不扩大 GPU 规模。
- 提交时 smoke 因 Priority 排队，正式任务等待依赖；不能宣称 GPU smoke 已通过或正式样本已经执行。48小时是调度时限，不是实测 ETA；待 GPU 有直接吞吐证据才能估计主体和汇总耗时。
- 单帧 source0/1/2/3 与四帧均评测；两模型同 batch target/mask，Wan 四帧末尾补一帧、只评分前四帧；FP32/TF32 off/posterior mean。
- Smoke 5232 在7个 UMI cell成功后，LIBERO加载失败：共享 mapping 的 field_mask.action=false，但 element_mask.action 前10项为true。5233因此依赖不能满足。
- 用户授权 action 全部 false。在 Frank 输出目录 wan22-ganoff124k-20260911-v2 创建独立 mapping/config；field action=false、20项element action=false；相机映射与图像mask资产保持相同。共享原始配置未修改。
- v2/libero-original256-action-false.json 保留256个窗口所有非配置字段及顺序，仅更换配置路径/SHA，记录原selection/mapping SHA和修复来源；不能声称selection文件哈希未变化。
- 启动脚本改用v2输出和修复后的LIBERO selection，Hy仍指向v1冻结的180975窗口selection，不重新抽样。bash -n、git diff --check通过；修复后的512个camera-window解码验证通过前不启动GPU重试。
- checkpoint/环境/源代码哈希与原 selection 证据见前一日准备记录。结果未生成前不报告模型质量优劣。
