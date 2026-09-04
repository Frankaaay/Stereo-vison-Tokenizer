# Mode-aware 默认训练 batch 合同

## 目的

将四模式训练 launcher 的默认 per-mode batch/gradient accumulation 改为
mode-aware 合同，同时保留环境变量显式覆盖能力。

## 修改

- `mono/single_frame`：每卡 batch 48，gradient accumulation 1。
- `mono/four_frame`：每卡 batch 48，gradient accumulation 1。
- `stereo/single_frame`：每卡 batch 48，gradient accumulation 1。
- `stereo/four_frame`：每卡 batch 24，gradient accumulation 2。
- 八卡运行时四种模式的 effective global batch 均为 384；调用方必须将
  `GLOBAL_BATCH_SIZE` 显式设置为 384，现有一致性检查会拒绝不匹配配置。

## 边界

本次只修改 launcher 默认值及其源码合同测试；没有修改模型、优化器、数据、loss、
显式环境变量覆盖语义或已有 smoke 脚本，也没有启动训练或远端任务。

## 验证

- `bash -n scripts/stereo/train_stereo_vae.sh`
- `python -m unittest tests.training.test_entrypoints_source`
- `git diff --check`
