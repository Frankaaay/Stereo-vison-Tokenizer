# H100 canonical UMI manifest 与四模式训练 smoke

## 目标

- 为 H100 上的 `UMI-Collectsite-KS3-canonical-v3` 生成可追溯的 episode manifest。
- 复用现有 Hy/LIBERO manifest 入口，验证三数据源四模式 DataLoader。
- 提交单卡最小训练 smoke：四模式 batch 为 `24:24:24:12`，梯度累积为 `1:1:1:2`，各模式有效 batch 均为 24。

## 数据合同

- UMI canonical episode 总数：90,174；episode 到原始 UUID、sidecar 和 40 套六相机标定的映射由数据侧交付目录提供。
- H100 发布目录的实际结构是外层数据根下唯一的 `table_000`；episode parquet、`info.json` 和视频位于该表内，pixel mask 位于外层数据根。builder 显式校验并解析这一层级。
- 六路图像已存储为 256×256；源图像变换为 640×480 等比缩放到 256×192，再上下各补 32 像素黑边。reader 必须跳过旧路径中的二次 resize/pad。
- 每条 episode 按 `calibration_bundle_sha256` 选择 K/D/R/P；交付键 `left_wrist/right_wrist` 在 manifest 中规范为训练代码的 `lefthand/righthand`。
- rectification 状态按用户转述的数据侧确认记录为 `verified_pre_rectified / data_side_confirmed_by_user`，本任务不再执行像素级极线审计；原始审计文件 SHA256 仍写入 manifest 合同。
- pixel mask 必须为数据根下 `image_pixel_mask_umi.npz`，reader 初始化时校验其 SHA256；有效区域为 `[32:224, 0:256]`。
- Hy 当前 19 张已转换 Lance 表仍使用三路 `cam_high/cam_left_wrist/cam_right_wrist`、240×424，符合现有 reader，无需代码改动。LIBERO 继续使用现有 LeRobot v2.1 manifest 路径。

## 代码改动

- `scripts/data/build_canonical_umi_stereo_manifest.py`：读取 canonical episode parquet、数据侧映射/标定/转换 provenance，生成确定性 90/5/5 split、manifest 与 summary；拒绝覆盖既有输出。
- `stereo_tokenizer/lerobot_data.py`：对带 `stored_image` 合同的 256×256 canonical 视频跳过二次空间变换，并校验 mask 路径与哈希；旧 640×480 manifest 行为保持不变。
- `tests/stereo/test_canonical_umi_manifest.py`：覆盖标定视角名规范化、flat canonical 视频映射、reader 直通、fx/baseline 与确定性 split。

## 验证记录

- 本地 `python -m py_compile`：通过。
- 本机缺少 `pytest`、`torch` 与 `pytorch_lightning`，因此本地动态测试未运行；不能将静态检查描述为 runtime 验证。
- H100 manifest、真实视频解码、DataLoader 和 GPU smoke 结果在完成后补充，记录精确 commit、命令、Job ID、日志与输出路径。
