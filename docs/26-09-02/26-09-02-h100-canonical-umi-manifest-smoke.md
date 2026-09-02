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
- Hy 当前 19 张 Lance 表统一使用物理列 `cam_head/cam_left_wrist/cam_right_wrist`，JPEG 已存储为 256×256；有效矩形为 `[55:200,0:256]`，对应源 240×424 等比缩放并居中 padding。manifest 固化列名、mask SHA 和 bbox，reader 裁出有效矩形后复用原几何管线，避免 DA3 将黑边当成内容。LIBERO 继续使用现有 LeRobot v2.1 manifest 路径。

## 代码改动

- `scripts/data/build_canonical_umi_stereo_manifest.py`：流式读取 canonical episode parquet、数据侧映射/标定/转换 provenance，生成确定性 90/5/5 split、manifest、40 套标定 catalog 与 summary；拒绝覆盖既有输出。
- `stereo_tokenizer/lerobot_data.py`：对带 `stored_image` 合同的 256×256 canonical 视频跳过二次空间变换，并校验 mask 路径与哈希；按 catalog SHA 和 bundle SHA 解析共享标定，旧 640×480 manifest 行为保持不变。
- `tests/stereo/test_canonical_umi_manifest.py`：覆盖标定视角名规范化、flat canonical 视频映射、reader 直通、fx/baseline 与确定性 split。
- `scripts/data/smoke_pretrain_manifests.py`：由 Slurm 调用，对选定 manifest 各解码一个 single-frame 与 four-frame 样本并输出 shape、dtype、finite 和 sample ID。
- `scripts/stereo/run_h100_canonical_smoke.sh`：冻结本次单节点 8 卡 smoke 的 batch、GA、四模式顺序、teacher、loss 与日志开关，只从环境接收绝对根路径并拒绝复用输出目录。

## 验证记录

- 本地 `python -m py_compile`：通过。
- 本机缺少 `pytest`、`torch` 与 `pytorch_lightning`，因此本地动态测试未运行；不能将静态检查描述为 runtime 验证。
- H100 manifest、真实视频解码、DataLoader 和 GPU smoke 结果在完成后补充，记录精确 commit、命令、Job ID、日志与输出路径。
- 初次全量 UMI builder 在登录节点被 signal 9 终止，峰值 RSS 1,414,828 KiB。根因是一次性读取全部 parquet 并为每条 episode 重复序列化完整标定；已改为 1,024 行批流式读取和独立标定 catalog，后续生成改由 Slurm 执行。

### H100 实际结果

- H100 clean clone 与本地最终运行代码同步到 `ba496c199aea2d059d021ba241417285a5f88ad5`；canonical UMI/Hy 定向回归在前序实现 SHA 上分别通过 15 和 20 个测试，最终变更仅增加解码 smoke 入口。
- UMI manifest Job `2258`：QOS `cpu`，exit 0，57 秒，MaxRSS 1,708,620 KiB。90,174 条 episode 全部完成映射核对；17 条没有完整四帧窗口，训练 manifest 保留 90,157 条、1,661,796 samples。manifest SHA256 `18cd5f460864c21866d9d2c9690c9398248d6429eb320bc0ab2ad9644fc91bb8`，40 套标定 catalog SHA256 `2ee39845da49fad9d6cdf0aeb27cf41e37bfafcb8d91b317a208fc33ff6201da`。
- Hy/LIBERO manifest Job `2262`：exit 0，7 秒。Hy 为 215,577 records、16,703,900 windows，SHA256 `6d6f9a6cf14bc502f4471cd4c9e5617e5fe35b4d9de1412b44eb3d8293ba5497`；LIBERO 为 1,712 records、34,192 windows，SHA256 `ffa8c06d4aadf5ef1dfa72222384c5c9ef60e8b454d1cadfdeb83fe83938f4d9`。
- UMI/LIBERO 真实解码 Job `2264`：exit 0，28 秒。single/four 输出分别为 UMI `[3,2,3,1/4,256,256]`、LIBERO `[1,1,3,1/4,256,256]`，均为 float32 且 finite。
- teacher 源码 clean 且 SHA 分别为 LAS2-H `8c97bd4c4da3712c2ac60003a23201dfdb5935f4`、DA3 `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`；checkpoint SHA256 分别为 `758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4`、`e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`。
- GPU smoke 计划为单卡、4 updates、四模式权重 `1:1:1:1`、mono 数据权重 `1:1`、batch `24:24:24:12`、accumulation `1:1:1:2`。seed 1234 的顺序为 stereo/four、mono/single(LIBERO)、mono/four(Hy)、stereo/single，覆盖全部模式和三个数据源。
- 当前唯一未闭合门禁：固定 train runtime 缺少 lock 中的 `pylance==10.0.0`，因此 Hy 真实 DataLoader 与 GPU job 尚未提交。另一个 `hy-export` 环境只有 `lance==8.0.0` 且 NumPy 2.5.2，不能混入训练环境作为生产替代；需获得远端环境写入授权后在 train runtime 安装精确锁定版本，再完成 Hy decode 与 GPU smoke。

### 8 卡 smoke 提交

- 用户授权在既有 `/gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/train` 环境安装锁定依赖并将 smoke 扩为 8 卡。`uv pip install ... pylance==10.0.0` 仅新增 `pylance==10.0.0`、`lance-namespace==0.8.6` 和 `lance-namespace-urllib3-client==0.8.6`；随后 `uv pip check` 通过，Torch 2.7.1+cu126、NumPy 1.26.2、PyArrow 23.0.0、Lightning 2.5.6 和 PyAV 16.0.1 均未漂移。
- Hy 真实解码 Job `2266`：QOS `cpu`，exit 0，50 秒；single/four 输出为 `[1,1,3,1/4,256,256]`、float32、finite，train sample count 49,106,151。
- 8 卡运行源码：H100 clean clone `hezhou-las2-h@907664b61b06006bcad634fbea8fad20b9e8c460`，H100 Bash syntax 通过。debug QOS 实时上限为单作业 16 GPU；本作业申请 1 节点、8 GPU、64 CPU、512 GiB、1 小时。
- `sbatch --test-only` 通过，正式 Job ID `2269`，名称 `stereo-smoke8-907664b`。提交后快照为 `PENDING (Resources)`，Slurm 预测开始时间 `2026-09-03 00:00:43 +08:00`，候选节点 `xn01-gpu1-0062`。
- 输出目录：`/gpfs/jiuquyun/projects/Frank/stereo-vae/outputs/h100-canonical-smoke4-907664b-v1`；Slurm 日志：`/gpfs/jiuquyun/home/Frank/logs/stereo-smoke8-907664b-2269.out`。作业启动时才创建输出目录；当前未产生训练结果，不能宣称 smoke 通过。
