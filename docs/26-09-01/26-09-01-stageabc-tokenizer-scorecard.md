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
- 本地 tensor/CUDA 单测：本机 Python 缺少 Torch，转到固定 H200 runtime 执行
- H200 smoke：待执行
- 正式全量评测：待执行
- 输出、日志、指标、ETA 与结论：启动后补充
