# Stereo Tokenizer 实施记录

## 状态

本地实现与可用静态验证已完成。本记录对应本地 `frank` 分支，基线 commit 为 `701b619003b3e941e769269c7626dbf111d0377e`；实现 commit 为包含本记录的当前提交，精确 SHA 在提交后由 Git 确定并在同步记录中引用。未创建 worktree，也未连接 H200 或启动训练。

## 目的

实现第一版结构化 `T=4` StereoTokenizer：逐帧共享 Spatial Encoder、StereoFusion 后 `4→1` temporal reduction、48-channel VAE posterior、无 anchor 的 `1 slot→4 frames` Decoder，以及 RGB/disparity/gradient/KL/LPIPS/GAN 训练合同。原 `OmniTokenizer` legacy image-mode 保留，Tokenizer 不实现下游 DiT patchify/unpatchify。

## 数据与配置合同

- 当前工程 pilot：100 MCAP、3407 个 `pilot_train` sample；smoke-32 与 overfit-128 为固定训练子集，没有独立 validation。
- 输入：`[B,3,2,3,4,256,256]`，帧间隔 0.1 秒，sample stride 0.4 秒。
- StereoFusion：共享权重，三个视角分别构造 mask，当前 `w=(7,7,7)`、offsets `[0..7]`、left query 向负 x 搜索。
- Disparity：`128×(softplus(raw)+eps)`，raw head bias `-2.572`；有效范围 `[0.5,112.0]`。
- Gradient：pixel disparity gradient 独立除以 `16.0 px`，不复用 128 disparity scale。
- 当前 pilot validation 显式关闭；仅在独立 validation Manifest 存在时允许 epoch 末完整验证一次。
- 首轮 smoke/overfit 启用 RGB、disparity、gradient、KL、LPIPS，GAN 显式关闭；GAN 权重与 start step 必须由后续 gate 冻结。

## 修改范围

- `OmniTokenizer/stereo/model.py`：结构化 Encoder/Decoder、VAE posterior、raw disparity bias。
- `OmniTokenizer/stereo/fusion.py`：共享水平 cross-attention、分视角 mask、有效候选数 entropy、detached confidence gate。
- `OmniTokenizer/stereo/losses.py`：逐视角 masked loss、独立 pixel gradient scale、KL 口径。
- `OmniTokenizer/stereo/training.py`：确定性 core、LPIPS、共享 image/video discriminator、显式 GAN gate、Adam `(0.5,0.9)`、warmup-cosine 配置和 validation policy。
- `tests/stereo/`：shape、forward/backward、mask、geometry、训练 gate、优化器和源码责任边界测试。
- `doc/Stereo Tokenizer Plan.md`：同步最终架构与当前 pilot 数据合同。

## 验证

- `python tests/stereo/test_source_boundary.py`：通过，2/2。
- 对 `OmniTokenizer/stereo/*.py`、`tests/stereo/*.py` 和 `OmniTokenizer/__init__.py` 执行 AST 解析：通过，13 个文件。
- 本轮相关文件 trailing-whitespace 检查：通过。
- 当前本机 Python 环境没有 PyTorch，因此 `test_fusion.py`、`test_model.py`、`test_losses_geometry.py` 和 `test_training_stage.py` 的动态 tensor/forward/backward 测试未运行，不能记为通过。

## Checkpoint、输出与日志

本次未启动训练或评估，因此没有 checkpoint、训练 output、tmux session 或运行日志。

## 当前结论与下一步

代码已完成本地静态审查。下一步是在具备项目 PyTorch 依赖的环境运行全部 `tests/stereo/` 动态测试；通过后再进入另行授权的 smoke-32、overfit-128 与 Loss calibration。任何 H200 运行均不在本次授权范围内。

## 原主链路迁移进度

### 模块 1：StereoFusion

- 状态：已实现，等待用户审核与提交。
- 唯一实现迁入 `OmniTokenizer/modules/stereo_fusion.py`；它现在属于原仓库的网络组件层，不再由旁路 tokenizer 定义。
- `OmniTokenizer/stereo/fusion.py` 暂时只做兼容转发，保证逐模块迁移期间现有分支仍可导入；完成原主链路接入后将单独删除。
- `tests/stereo/test_fusion.py` 已改为直接测试原仓库 modules 路径。
- 本地仅执行 AST、字节码编译和源码边界检查；Torch 动态测试按约定留到 H200。

### 模块 2：Stereo Geometry

- 状态：已实现，等待用户审核与提交。
- `DepthOutput` 与 `disparity_to_depth` 的唯一实现迁入 `OmniTokenizer/modules/stereo_geometry.py`。
- 转换只实现 `D=fxB/d`、标定 shape 校验和有效像素传播，不增加 Depth Head 或 depth loss。
- `OmniTokenizer/stereo/geometry.py` 暂时只做兼容转发，完成原主链路接入后将单独删除。
- 几何测试已改为直接导入原仓库 modules 路径；Torch 动态数值测试按约定留到 H200。

### 模块 3：Stereo Losses

- 状态：已实现，等待用户审核与提交。
- RGB、masked normalized-disparity、pixel-disparity gradient 和 posterior KL 的唯一实现迁入 `OmniTokenizer/modules/stereo_losses.py`。
- disparity 与 gradient 继续按有效像素分视角归一化后等权平均；任一视角没有有效监督时 fail closed。
- Loss 权重仍由 resolved config 显式传入，本模块没有写入待 calibration 参数的默认值。
- `OmniTokenizer/stereo/losses.py` 暂时只做兼容转发，完成原主链路接入后将单独删除。
- loss 测试已改为直接导入原仓库 modules 路径；Torch 动态数值测试按约定留到 H200。
