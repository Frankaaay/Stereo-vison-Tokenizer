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

状态：待提交。计划先运行定向 pytest，再用既有冻结 runtime、真实 manifest 和 teacher 资产做保持原 batch 合同的四模式短 smoke。需记录精确代码 SHA、Job ID、QOS/GPU、命令、日志、exit code、loss、显存和第一条异常。
