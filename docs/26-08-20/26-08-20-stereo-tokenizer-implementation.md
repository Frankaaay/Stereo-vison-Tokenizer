# Stereo Tokenizer 实施记录

## 状态

原主链路迁移和旁路清理已在本地 `frank` 分支完成，基线 commit 为 `701b619003b3e941e769269c7626dbf111d0377e`。每个实现模块独立提交；本轮未 push、未连接 H200，也未启动 cache、训练或评估。

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

- `python tests/stereo/test_source_boundary.py`：通过，4/4。
- `python tests/stereo/test_entrypoints_source.py`：通过，3/3。
- `python tests/stereo/test_rgb_cache_contract.py`：通过，2/2。
- 17 个主实现、入口和测试 Python 文件 AST/bytecode 编译：通过。
- 本轮相关文件 trailing-whitespace 检查：通过。
- 当前本机 Python 环境没有 PyTorch；所有 tensor/forward/backward、Dataset 和 strict checkpoint 动态测试必须在 H200 执行，当前不能记为通过。
- 本机没有可用 WSL distribution，`bash -n scripts/recons/train.sh` 未执行成功；经用户确认留到 H200 验证。

## Checkpoint、输出与日志

本次未启动训练或评估，因此没有 checkpoint、训练 output、tmux session 或运行日志。

## 当前结论与下一步

代码已完成本地实现与静态审查。下一步需先在 H200 生成独立 RGB cache/Manifest v3，并运行全量动态单测与 32-sample preflight；这些远端写入和 GPU 动作仍需单独确认。

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

- 状态：已提交（`178d8b8`）；尚未在 H200 生成 cache。
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
