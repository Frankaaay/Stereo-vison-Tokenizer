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
- LPIPS 修复 SHA：`fdc5831b35c8578e9f5db602b68ad7c3acaddebf`；定向 pytest Job `2854` 为 `COMPLETED`、exit code `0:0`，`21 passed, 3 warnings`
- 修复后 smoke Job `2859` 已完成 4/4 training updates 并生成 `epoch=0-step=4.ckpt` 与 `last.ckpt`，证明原 LPIPS OOM 已消失；随后 validation rank 6 的 DA3 在申请 `7.59 GiB` 时 OOM，当时 `24.18 GiB` 为 reserved but unallocated。其余 rank 已到 20/20，作业无法健康收敛，确认失败后手动取消释放 8 卡
- validation OOM 属于混合尺寸 teacher workload 的 allocator 碎片，启动脚本默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，同时保留外部显式覆盖能力；需重跑同合同 smoke 验证最终 exit code
- allocator 修复 SHA：`fb69ddd83b9a99397e5a1ed26d5b0f0ffc9e9844`；H100 `bash -n` 通过，定向 launcher + integration pytest 为 `22 passed, 3 warnings`。更宽的 source-test 另有一个与本改动无关的既有失败：测试仍要求无 source suffix 的 depth 文件名，而当前评估实现已带 suffix，本次未扩大范围修改
- 最终 smoke Job `2871`：8×H100，BS `24:24:24:12`，GA `1:1:1:2`，`COMPLETED`，exit code `0:0`，elapsed `00:04:57`；training 4/4 updates、validation 20/20，有限 `val/mixed/total_loss=1.640`
- checkpoint 直读：`global_step=4`、`generator_updates=4`、`discriminator_updates=0`，四种 mode 各 `1 update / 192 samples`，有效 global batch 均为 192；产物包含 `best-epoch=0-step=4.ckpt`、`epoch=0-step=4.ckpt`、`last.ckpt`、`resolved_config.json`、`run_manifest.json`、`step_timing.json`
- 最终日志：`/gpfs/jiuquyun/home/Frank/logs/lpips-final-smoke-fb69ddd-2871.out`；输出：`/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/lpips-final-smoke-fb69ddd-2871`。`mono/four_frame` 峰值 allocated/reserved 分别约 `75.26/75.32 GiB`，仍接近 80GB H100 上限
