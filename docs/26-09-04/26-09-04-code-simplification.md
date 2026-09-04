# 代码精简记录

## 目的与范围

- 基线分支：`hezhou-las2-h`
- 基线 commit：`9122eb86c5662a624ad032d9b9643e22097355ad`
- 删除已被统一 Stage A 取代的旧评估实现与 vendored 指标代码。
- 删除 Hy smoke cache 独立环境/构建脚本和无调用的 LPIPS、DataModule 接口。
- 将 Stage A 仍需的 checkpoint、teacher、batch 与可视化运行时解耦为
  `evaluation/stage_a_runtime.py`。
- 训练只保留 Hy/LIBERO/UMI 三源四模式生产路径，删除旧 LeRobot 单源入口与开关。
- 删除只接受单一取值或始终拒绝的模型配置项。
- 本次不修改教师 backend、GAN 分支及依赖集合（C 类范围）。

## 删除内容

- `evaluation/common_metrics_on_video_quality/`
- `evaluation/pytorch-fid/`
- `evaluation/fvd_external.py`
- 旧 `evaluation/README.md`
- `scripts/data/build_hy_mono_smoke_cache.py`
- `environments/hy-export/`
- `tests/stereo/test_eval_four_mode.py`
- `LPIPS.from_pretrained`、`StereoDataModule.test_dataloader`
- 旧 evaluator 的 CLI、dataset/DDP、RGB/SSIM/LPIPS accumulator 与结果汇总代码
- `four_mode_mixed_training`、旧 LeRobot 单源数据参数与分支
- `patch_embed`、`defer_temporal_pool`、`defer_spatial_pool` 和无效
  `train_epoch_repeats` 配置面

所有文件均按明确路径逐个删除，没有递归删除目录。

## 保留的生产合同

- 训练：Hy/LIBERO mono + UMI stereo 的四模式采样与 mode schedule。
- 评估：`evaluation/tokenizer_stage_a.py` 的 selection、preflight、run、benchmark、report。
- Stage A 模型调用一次接收完整视图，不恢复逐视角 encode 路径。
- 教师 backend、在线 GT cache、GAN 阶段和 checkpoint continuation/transition 合同保持原样。

## 验证

- `python -m compileall -q train_stereo_vae.py stereo_tokenizer evaluation tests`：通过。
- `python -m unittest tests.stereo.test_source_boundary tests.stereo.test_entrypoints_source`：24 项通过。
- `git diff --check`：通过（仅有仓库既有的 Windows LF/CRLF 提示）。
- 本机没有 PyTorch，未运行依赖 tensor/CUDA 的单元测试或 GPU smoke。
- Git Bash 对两个训练 launcher 执行 `bash -n`：通过。
