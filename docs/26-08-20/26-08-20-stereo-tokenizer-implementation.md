# Stereo Tokenizer 实施记录

## 状态

原主链路迁移、旁路清理和 Encoder cleanup 回归修复已在 `frank` 分支完成并 push。2026-08-20 两台 H200 均以 fast-forward-only 同步到数据生成基线 `5d5c78dda21300eabfcb5951b961da02e66d1cdd`；`h200-1` 已完成独立 RGB cache、三份 Manifest v3 和 3407 条全量数据校验，随后以 checksum 验证的非破坏性 rsync 同步到 `h200-2`。尚未启动动态单测、smoke、训练或评估。

## 目的

实现第一版结构化 `T=4` StereoTokenizer：逐帧共享 Spatial Encoder、StereoFusion 后 `4→1` temporal reduction、48-channel VAE posterior、无 anchor 的 `1 slot→4 frames` Decoder，以及 RGB/disparity/gradient/KL/LPIPS/GAN 训练合同。原主类改为 Stereo-only，不保留 legacy image-mode；Tokenizer 不实现下游 DiT patchify/unpatchify。

## 数据与配置合同

- 当前工程 pilot：100 MCAP、3407 个 `pilot_train` sample；smoke-32 与 overfit-128 为固定训练子集，没有独立 validation。
- 输入：`[B,3,2,3,4,256,256]`，帧间隔 0.1 秒，sample stride 0.4 秒。
- StereoFusion：共享权重，三个视角分别构造 mask，当前 `w=(7,7,7)`、offsets `[0..7]`、left query 向负 x 搜索。
- Disparity：`128×(softplus(raw)+eps)`，raw head bias `-2.572`；有效范围 `[0.5,112.0]`。
- Gradient：pixel disparity gradient 独立除以 `16.0 px`，不复用 128 disparity scale。
- 当前 pilot validation 显式关闭；仅在独立 validation Manifest 存在时允许 epoch 末完整验证一次。
- 首轮 smoke/overfit 启用 RGB、disparity、gradient、KL、LPIPS，GAN 显式关闭；GAN 权重与 start step 必须由后续 gate 冻结。

## 修改范围

- `OmniTokenizer/omnitokenizer.py`：原 Encoder、Decoder 和 Lightning 主类直接改为 Stereo-only VAE 主链路。
- `OmniTokenizer/modules/stereo_fusion.py`、`stereo_geometry.py`、`stereo_losses.py`：融合、几何与监督的唯一实现。
- `OmniTokenizer/data.py`：Manifest v3 + 独立 RGB/GT cache Dataset/DataModule。
- `scripts/data/build_stereo_rgb_cache.py`：RGB cache 构建与 Manifest v3 finalize。
- `vqgan_train.py`、`vqgan_eval.py`、`scripts/recons/train.sh`：Stereo-only 训练、评估与 recipe。
- `tests/stereo/`：主链路 shape、forward/backward、mask、geometry、cache 和入口边界测试。
- `doc/Stereo Tokenizer Plan.md`、`README.md`：同步最终架构、模式边界与 pilot 数据合同。
- `OmniTokenizer/stereo/*.py`：旁路实现全部删除；不再导出 `StereoTokenizer`/`StereoTokenizerConfig`。

## 验证

- `python tests/stereo/test_source_boundary.py`：通过，5/5（含 Encoder/Decoder 参数所有权回归检查）。
- `python tests/stereo/test_entrypoints_source.py`：通过，3/3。
- `python tests/stereo/test_rgb_cache_contract.py`：通过，2/2。
- 17 个主实现、入口和测试 Python 文件 AST/bytecode 编译：通过。
- 本轮相关文件 trailing-whitespace 检查：通过。
- 当前本机 Python 环境没有 PyTorch；所有 tensor/forward/backward、Dataset 和 strict checkpoint 动态测试必须在 H200 执行，当前不能记为通过。
- 本机没有可用 WSL distribution，`bash -n scripts/recons/train.sh` 未执行成功；经用户确认留到 H200 验证。
- H200 Git 同步：`h200-1`、`h200-2` 均为 clean `frank@5d5c78d`，upstream 为 `origin/frank`，origin URL 为项目 GitHub 远端。
- RGB cache：8 个 episode shard 全部成功，100 个 episode、3407/3407 个 sample，`written=3407`、`reused=0`。
- Manifest v3：full/smoke/overfit 分别为 3407/32/128 条；全量验证状态为 `PASS`。
- 全量数据交叉校验：RGB 与 GT 均为 3407 个；逐条验证路径、唯一性、NPZ key、shape、dtype、padding、GT metadata、v2→v3 字段保持和 source SHA；无临时文件残留。

## Checkpoint、输出与日志

本次未启动训练或评估，因此没有 checkpoint。数据产物与日志如下：

- RGB/Manifest v3 根：`/data/shared/datasets/umi_raw_data0806_stereo_pilot_rgb_v2`（`h200-1`、`h200-2` 各一份 node-local 副本，约 16 GB）。
- Full Manifest v3：`pilot_manifest_v3.jsonl`，SHA256 `1e12cdd448e2834f5d20a6c9a373740cbf38be65c7cb5feac053c62becd70934`。
- Smoke Manifest v3：`smoke_32_v3.jsonl`，SHA256 `3da89faae697d94a86ffc1962d47bf9b2862d5db0673abc8635753a7624b1c14`。
- Overfit Manifest v3：`overfit_128_v3.jsonl`，SHA256 `3df1278276ef855c605b774af3ff34dcb13a23ca2c8481698698e0faea86700c`。
- 运行日志：`/data/home/frank/runtime/stereo-rgb-cache-v3/shard-{0..7}.log` 与 `finalize-*.log`。
- 全量校验结果：`/data/home/frank/runtime/stereo-rgb-cache-v3/validation-full.json`，状态 `PASS`，校验耗时 83.668 秒。
- 双节点同步日志（`h200-1`）：`/data/home/frank/runtime/stereo-rgb-cache-v3/rsync-h2002-dryrun.txt`、`rsync-h2002.log`、`rsync-h2002-verify.txt`。

## 当前结论与下一步

代码、远端同步、RGB cache、Manifest v3 和数据合同校验均已完成。下一步是在用户再次确认后，先运行 H200 CPU/单卡动态单测与 32-sample smoke preflight；当前未冻结的 loss 权重、warmup、batch/accumulation、学习率与 step budget 仍须通过 smoke/overfit calibration 决定。正式训练数据仍需独立的 episode 级 train/validation/test 划分与全量重扫，不能把本 pilot 当作正式数据集。

## H200 RGB cache 与 Manifest v3 实际运行

- 运行位置：`h200-1`；代码分支/SHA：`frank@5d5c78dda21300eabfcb5951b961da02e66d1cdd`。
- Python：`/data/home/frank/runtime/foundation-stereo-v1/bin/python`；复用已存在环境，未安装或修改依赖。
- 输入：冻结 Manifest v2、100 个原始 MCAP 和 3407 个 FoundationStereo GT；三个 v2 SHA 在生成前后保持不变。
- 关键命令：8 次 `build_stereo_rgb_cache.py cache --shard-index N --shard-count 8`，随后分别对 full、smoke-32、overfit-128 执行 `finalize`。
- 主体运行：约 23:06 启动、23:08 完成；8 个 shard 的 episode 数合计 100、written 合计 3407，所有 tmux session 正常退出。
- 校验范围：逐条读取约 16 GB RGB cache 与全部 GT；RGB 合同为唯一 key `rgb`、`uint8 [3,2,3,4,256,256]`，上下各 32 像素 padding 全为 128；GT schema 为 `stereo-foundation-gt-v1`，并核对六个 key、shape/dtype、正值标定和 sample metadata。
- 数据摘要：RGB value range `[0,255]`、全局均值 `116.87274006593913`；源 MCAP 数 100；无 `.tmp-*` 文件或 cache 进程残留。
- 异常记录：系统 `/usr/bin/python3` 缺少依赖，因此经确认后复用 `frank` 自有 FoundationStereo 环境；初次 GT 非递归计数为 0，确认实际是按 100 个 episode 子目录分层，递归计数为 3407。两次 Windows→SSH 只读监控命令发生变量/CRLF 引号错误，未影响远端生成进程或数据。
- 双节点同步：先执行 checksum dry-run，确认目标为空且仅有 3410 个 regular files 待新增；随后从 `h200-1` 以 `rsync -a` 同步到 `h200-2`，未使用 `--delete`。共传输 16.10 GB、3410 个文件，主体耗时 26 秒，删除数为 0。
- 同步后验证：两端 RGB 根目录均为 `16098848056` bytes、RGB cache 均为 3407 个，三份 Manifest v3 SHA 分别一致；`rsync -acni --itemize-changes` 输出 3512 个以 `.` 开头的已核对条目，non-dot change 为 0。
- 当前结论：双节点数据链路 Gate 通过；用户已授权在同步与文档更新完成后启动动态测试和 smoke，当前尚未启动。

## 原主链路迁移进度

### 模块 1：StereoFusion

- 状态：已提交（`07384db`）。
- 唯一实现迁入 `OmniTokenizer/modules/stereo_fusion.py`；它现在属于原仓库的网络组件层，不再由旁路 tokenizer 定义。
- 迁移期间的兼容转发已随旁路目录删除。
- `tests/stereo/test_fusion.py` 已改为直接测试原仓库 modules 路径。
- 本地仅执行 AST、字节码编译和源码边界检查；Torch 动态测试按约定留到 H200。

### 模块 2：Stereo Geometry

- 状态：已提交（`47cd381`）。
- `DepthOutput` 与 `disparity_to_depth` 的唯一实现迁入 `OmniTokenizer/modules/stereo_geometry.py`。
- 转换只实现 `D=fxB/d`、标定 shape 校验和有效像素传播，不增加 Depth Head 或 depth loss。
- 迁移期间的兼容转发已随旁路目录删除。
- 几何测试已改为直接导入原仓库 modules 路径；Torch 动态数值测试按约定留到 H200。

### 模块 3：Stereo Losses

- 状态：已提交（`f3165e4`）。
- RGB、masked normalized-disparity、pixel-disparity gradient 和 posterior KL 的唯一实现迁入 `OmniTokenizer/modules/stereo_losses.py`。
- disparity 与 gradient 继续按有效像素分视角归一化后等权平均；任一视角没有有效监督时 fail closed。
- Loss 权重仍由 resolved config 显式传入，本模块没有写入待 calibration 参数的默认值。
- 迁移期间的兼容转发已随旁路目录删除。
- loss 测试已改为直接导入原仓库 modules 路径；Torch 动态数值测试按约定留到 H200。

### 模块 4：Structured Stereo Encoder

- 状态：已提交（`0ec0e4b`）。
- 结构化入口直接加入原 `OmniTokenizer_Encoder`，不再由旁路 `FrameSpatialEncoder` 或 `StereoTemporalEncoder` 定义。
- 六路四帧先合并到 batch，复用原 `to_patch_emb_first_frame` 与 `enc_spatial_transformer` 逐帧编码；随后执行 StereoFusion 和 `4×D→D` 联合线性投影。
- 原 `enc_temporal_transformer` 保留，但显式断言其输入 temporal length 为 1，因此不会在四个 raw frames 之间执行 attention。
- 新增独立 shape、mono bypass 与梯度测试；Torch 动态执行留到 H200。

### 模块 5：Structured Stereo Decoder

- 状态：已提交（`5f62c04`）。
- 结构化出口直接加入原 `OmniTokenizer_Decoder`，复用原 `dec_temporal_transformer` 和 `dec_spatial_transformer`。
- 输入严格为每视角一个 latent slot；共享 Decoder 主干后才分成 RGB 与 disparity 两个线性投影，并分别展开为四帧。
- disparity 使用 resolved per-view scale、`softplus+epsilon` 和 resolved raw bias；不增加独立 Depth Head。
- 新增双 Head shape、bias、正值与 shared-backward 测试；Torch 动态执行留到 H200。

### 模块 6：Stereo-only VAE 主类与训练核心

- 状态：已提交（`20df09b`）。
- 原 `OmniTokenizer/omnitokenizer.py::VQGAN` 已直接改为结构化 Stereo-only VAE；删除主类中的 VQ codebook、legacy image/video 分支和旧 checkpoint inflation 假设。
- Encoder 输出经原位置的 `pre_vq_conv` 产生 48-channel 对角高斯 posterior；训练默认 sample，验证、评估和日志默认使用 posterior mean。
- Decoder 输入严格为 `[B,3,48,1,H',W']`，输出 RGB 与 disparity；主损失显式组合 RGB、normalized disparity、pixel-gradient 与 KL，并保留独立 LPIPS/GAN gate。
- 所有未完成 calibration 的 loss 权重均为必填 CLI 参数，没有在主类内猜测默认值；GAN 第一阶段可通过 `--gan_enabled` 显式关闭。
- 新增主类结构化 forward、确定性 eval、backward 与无 codebook/legacy 入口测试；Torch 动态执行留到 H200。

### 模块 7：独立 RGB cache 与 Manifest v3 数据链路

- 状态：已提交（`178d8b8`）；已在 `h200-1` 完成 3407 条 cache、三份 Manifest v3 和全量校验。
- `scripts/data/build_stereo_rgb_cache.py` 只读 Manifest v2 与原始 MCAP，按 episode 解码六路 H.264，写入独立 `uint8 [3,2,3,4,256,256]` RGB cache；支持 episode 级分片和已存在 cache 的严格复用。
- `finalize` 子命令仅在 3407 个引用 cache 全部通过 shape/dtype 校验后生成新的 Manifest v3；v2 Manifest 与 FoundationStereo GT 不被覆盖。
- `OmniTokenizer/data.py::StereoManifestDataset` 直接读取 RGB/GT cache，构造 `[V,E,C,T,H,W]` RGB、`[V,1,T,H,W]` disparity 和冻结的 confidence valid mask，并返回 fx/baseline。
- pilot 不提供伪 validation；`--stereo_val_manifest` 为空时 DataModule 返回无 validation loader，正式数据可传独立 v3 Manifest 做 epoch 末完整验证。
- RGB cache 纯合同测试可在本地执行；Dataset tensor 动态测试与真实 H.264 解码留到经确认后的 H200 阶段。

### 模块 8：原训练与评估入口

- 状态：已提交（`ca4ae72`）；尚未启动训练或评估。
- `vqgan_train.py` 已移除 legacy checkpoint inflation、自动扫描恢复和 image/video dataset 分支假设；只接受 Stereo Manifest loader，训练 posterior sample，validation 每个 epoch 末完整运行一次。
- 当前 pilot 没有 validation Manifest，因此显式设置 `limit_val_batches=0`；提供正式 validation v3 Manifest 时使用 `limit_val_batches=1.0`，不做 batch 子采样。
- checkpoint 每个 epoch 保存一次；是否 resume 只由 Lightning 的显式 CLI 参数决定，不再静默选择目录中最近 checkpoint。
- `vqgan_eval.py` 使用 posterior mean 且 strict load，输出 RGB L1、分视角 disparity EPE、派生 depth AbsRel/RMSE 和有效像素数，不再计算 codebook usage。
- `scripts/recons/train.sh` 已替换为结构化 Stereo recipe；数据路径、loss 权重、KL/optimizer warmup、micro batch、gradient accumulation 和 step budget 都必须由调用方显式设置，避免在 calibration 前写入猜测值。

### 模块 9：旁路与 legacy tokenizer 清理

- 状态：已实现，随本记录的最终 cleanup commit 提交。
- 删除 `OmniTokenizer/stereo/` 下 model/training 旁路和 fusion/geometry/losses 兼容转发，删除只覆盖旁路实现的旧测试。
- 原 `OmniTokenizer_Encoder/Decoder` 中不可达的 legacy temporal patch embedding、legacy pixel projection、image-mode `forward` 和对应参数已删除。
- `OmniTokenizer/__init__.py` 不再导出旁路 `StereoTokenizer`；源码边界测试改为检查唯一主实现、无 codebook/legacy image forward、无 DiT patchify。
- 设计文档和 README 已同步说明：本分支是 Stereo-only，原 model-zoo checkpoint 和离散 codebook 下游入口与当前主类不兼容。

### 模块 10：Encoder cleanup 回归修复

- 状态：已提交（`6b13f47`）。
- 修复 cleanup commit `36febdf` 将 Decoder Head 初始化误贴入 Encoder 的问题；恢复 Encoder 对 views、frames、block 和 search radii 的校验。
- 恢复共享 `StereoFusion` 和 `LayerNorm(4D) → Linear(4D,D) → LayerNorm(D)` temporal projection；Encoder 不再引用 disparity scale/bias/epsilon 或创建 Decoder Heads。
- 新增无 Torch 依赖的 AST 回归测试，直接约束 Encoder/Decoder 的参数所有权；真实实例化与 forward 仍须在 H200 验证。

### 模块 11：Stereo 测试导入边界修复

- 状态：已实现，待 H200 动态验证。
- `h200-2` 的项目环境 `omnitokenizer-e2` 已具备 Torch 2.7.1、CUDA 12.8 和主要训练依赖，但初次导入被 legacy 数据依赖阻塞：`decord` 缺失，且当前 Python 3.12 pip 源没有 `imagenet-stubs` distribution。
- `OmniTokenizer/data.py` 不再在模块加载时导入仅供 legacy `ImageDataset` 使用的 `imagenet_stubs`；该依赖改为构造 legacy dataset 时按需导入，Stereo Manifest 主路径不再被无关依赖阻塞。
- `decord==0.6.0` 仍按原仓库数据模块合同安装到用户确认的项目环境，不通过源码伪造或旁路该依赖。
- `tests/stereo/test_source_boundary.py` 新增 AST 回归检查，约束 `imagenet_stubs` 不得重新成为 `data.py` 的顶层导入，同时保留 legacy 类内部的真实依赖。

### 模块 12：LPIPS pretrained 名称比较修复

- 状态：已实现，待 H200 动态验证。
- H200 Python 3.12 import preflight 暴露 `LPIPS.from_pretrained` 使用 `is not` 比较字符串的 `SyntaxWarning`；虽然当前 Stereo smoke 直接调用 `LPIPS()`，该 classmethod 仍可能对动态构造的同值字符串误判。
- `OmniTokenizer/modules/lpips.py` 改为 `name != "vgg_lpips"`，只修正字符串值比较，不改权重加载、网络结构、forward 或 loss 配置。
- 源码边界测试新增 AST 检查，要求该条件使用 `NotEq`，并禁止重新引入字符串 identity comparison。

### 模块 13：PyTorch Lightning 2.5 训练入口兼容

- 状态：已实现，待 H200 动态验证。
- `vqgan_train.py` 移除 Lightning 1.x 已删除的 `Trainer.add_argparse_args` 与 `Trainer.from_argparse_args`，显式声明 `devices`、`num_nodes`、`max_steps` 和 `default_root_dir`，并按 Lightning 2.5 接口构造 GPU Trainer；bf16/fp16 分别使用 `bf16-mixed`/`16-mixed`。
- `scripts/recons/train.sh` 将旧 `--gpus` 参数改为 `--devices`，仍由 `GPU_COUNT` 控制单节点可见设备数。
- `OmniTokenizer/modules/callbacks.py` 更新 `rank_zero_only` 导入和 batch-end hook 签名；本地图像与视频写到 `trainer.default_root_dir`，不再依赖 WandB logger。禁用 WandB 时不构造要求 logger 存在的 `LearningRateMonitor`。
- `tests/stereo/test_source_boundary.py` 新增无 Torch 静态回归，约束不得重新引入 Lightning 1.x Trainer API、旧 `--gpus` 参数、旧 hook 签名或 `pl_module.logger.save_dir`。
- H200 首轮完整测试发现 `tests/stereo/test_entrypoints_source.py` 仍匹配旧 `trainer_overrides` 字典文本；测试合同已改为检查 Lightning 2.5 `Trainer(...)` 的等价显式关键字，不为满足旧字符串格式回改生产代码。

### 模块 14：Update-based timm 学习率调度修复

- 状态：已实现，待 H200 3-step 动态 gate。
- 单 sample 过拟合的 CSV 证据显示 step 0 到约 step 294 的 `lr-Adam` 始终为 `0.0`，确定性 probe 的 RGB、disparity 与 depth 输出不变；这不是模型容量或数据问题，而是 optimizer 没有获得非零学习率。
- `CosineLRScheduler` 配置为 `t_in_epochs=False` 时，timm 只在 `step_update(num_updates)` 中更新参数组；原训练代码错误调用 epoch 型 `step(global_step)`，因此生成器和可选判别器 scheduler 都不生效。
- 生成器与判别器路径均改为 `step_update(self.global_step)`；源码边界测试同时禁止两个旧调用重新出现。修复不改变 optimizer、warmup/cosine 参数或 loss 配置。
