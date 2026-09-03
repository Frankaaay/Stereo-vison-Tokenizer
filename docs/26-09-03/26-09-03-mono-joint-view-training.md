# Mono 多视角联合训练

## 目的

将 mono 训练从每个相机独立构造样本并分别调用 encoder，改为同一时间窗口内所有有效 mono 视角共同进入一次 forward。HY 固定使用 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三个视角；LIBERO 固定使用 `agentview`、`wrist` 两个视角。Stereo 路径保持 `V=3,E=2` 不变。

## 代码与合同

- 分支：`hezhou-las2-h`
- 修改起点：`52b85b15c45455a4ad54e5930c63f7a00224590a`
- 参考实现：NGADv1pp `frank/stereo-tokenizer-encoder-inference`，只读核对 SHA `fbc371b365dc6f567bea4c6fd51921811568cf79`
- HY mono sample：`[V=3,E=1,C=3,T,H,W]`
- LIBERO mono sample：`[V=2,E=1,C=3,T,H,W]`
- DA3 输入在 teacher 边界由 `[B,V,T,C,H,W]` 展平为 `[B*V,T,C,H,W]`，输出恢复为 `[B,V,1,T,H,W]`
- Encoder、decoder、posterior、重建/depth/KL/perceptual/GAN loss 均保留 view 轴；不增加跨视角 attention 或 mono fusion
- 既有 per-device batch、mode schedule 和梯度累积配置不变。`mode_samples` 继续统计 scene window 数，不改为 camera-view 数；每个 mono scene window 的实际图像负载随视角数增加
- DA3 cache 继续使用旧的逐视角 sample ID，已有合法 cache 不因 joint batching 失效

## 修改范围

- `stereo_tokenizer/pretrain_data.py`
- `stereo_tokenizer/mode_sampling.py`
- `stereo_tokenizer/model.py`
- `stereo_tokenizer/online_gt.py`
- `eval_stereo_vae.py`：共享数据集改为多视角后所需的最小 view/time 轴适配
- 对应 `tests/stereo/` 定向测试

## 验证记录

### 本地 Windows

- `python -m compileall -q stereo_tokenizer eval_stereo_vae.py tests/stereo`：通过
- `git diff --check`：通过
- 本机 Python 缺少 `torch` 和 `pytest`，未运行张量测试

### H100

运行位置：H100 Slurm，登录节点 `jiuquyun-node-11`，代码目录 `/gpfs/jiuquyun/projects/Frank/stereo-vae/Stereo-vison-Tokenizer`，冻结 runtime `/gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/train`。

- 提交并推送 SHA：`44ceaa9ae1a392a73f1af9f2d839ed6550d6ee23`
- 定向 pytest Job `2839`：`debug` QOS，1×H100，`COMPLETED`，exit code `0:0`，`70 passed, 9 subtests passed`；日志 `/gpfs/jiuquyun/home/Frank/logs/mono-joint-pytest-2839.out`
- 首次 pytest Job `2837` 未执行测试，因 Slurm `--wrap` 的 `/bin/sh` 不支持 `source` 而以 exit code `127:0` 退出；改用 POSIX `.` 后重提成功
- 四模式短实验 Job `2841`：`debug` QOS，8×H100，BS `24:24:24:12`，GA `1:1:1:2`，4 logical updates，状态 `FAILED`，exit code `1:0`；8 个 rank 均在 `mono/four_frame` 的 LPIPS 特征归一化处 OOM，未生成 checkpoint
- 实验脚本：`/gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/benchmarks-20260903/mono-joint-smoke-44ceaa9.sbatch`
- 日志：`/gpfs/jiuquyun/home/Frank/logs/mono-joint-smoke-44ceaa9-2841.out`
- 输出：`/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/mono-joint-smoke-44ceaa9-2841`
- 根因：joint-view 后 `mono/four_frame` 将 `24 x 3 views x 4 frames = 288` 张图一次展平送入 LPIPS；每卡已有约 `74.13 GiB` PyTorch allocation，LPIPS `normalize_tensor(x ** 2)` 继续申请 `4.50 GiB` 时失败。这不是 allocator 碎片或单卡故障
- 修复：保持训练 BS/GA 和联合 encoder 合同不变，仅将 LPIPS 按 `(view, frame)` 分块，每次处理 `[B,C,H,W]`，再按元素总数恢复原有全局 mean 语义
