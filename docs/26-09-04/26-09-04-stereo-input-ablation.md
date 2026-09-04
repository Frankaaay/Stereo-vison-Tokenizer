# Stereo Tokenizer 单双目训练消融

## 目的与结论边界

在同一 H200-1 数据、teacher、样本顺序、训练预算和评测合同下，从零顺序训练
M48-left-only 与 D48-same-left-trained。两者与后续同合同 S48-correct 一起用于拆分
单目路径、双路架构容量和真实左右视差的贡献。

M48 的学生网络只接收三视角左图 `E=1` 并跳过 StereoFusion；D48 保留完整
`E=2` 和 StereoFusion，但学生输入恒为 `(L,L)`。UMI DataLoader 和 LAS2-H teacher
始终先读取原始、同步、矫正的 `(L,R)`，随后才执行学生输入干预。因此 M48/D48
使用完全相同的 UMI windows、disparity target 和 valid mask，右图不会进入 M48
学生网络。

历史 H200-1 S48 update-44k 使用旧代码和 per-device BS24；H200-2 latent 消融使用
不同的 node-local manifests。它们只能作为回归参考，不能作为本轮正式 S/M/D
因果比较。正式结论仍需在 H200-1 以本轮相同合同补训 S48-correct。

## 冻结训练合同

- 分支：`hezhou-las2-h`；实际启动 commit 在启动记录中填写。
- 节点：H200-1，单节点 8 GPU，BF16。
- 顺序：M48-left-only 正常退出后，串行 wrapper 才启动 D48-same-left。
- 每个模型从零训练 40,000 generator updates；不加载任何 checkpoint。
- latent 为 `[B,3,48,1,16,16]`。
- per-device mode batch 为 `192:40:160:36`，GA 为 `1:1:1:1`；对应 8 卡
  effective global batch 为 `1536:320:1280:288`。
- mode update weights 为 `35:35:15:15`；每模型处理 35,392,000 logical samples。
- mono dataset weights Hy:LIBERO=`9:1`；UMI stereo modes 的数据、teacher target、
  schedule 和 sample counters 在 M48/D48/S48 间保持一致。
- seed 与 mode schedule seed 均为 1234。
- RGB、relative depth、relative gradient、LPIPS 权重为 `1.0/1.0/0.1/1.0`；
  KL=`1e-6`，warmup 100 updates；GAN 和 feature matching 关闭。
- generator LR/min LR 均为 `1e-4`，optimizer warmup 20 updates。
- 在线 DA3 与 LAS2-H；LAS2-H 4 iterations、pair microbatch 48；online GT cache
  关闭。
- validation 每 2,000 generator updates；checkpoint 每 5,000 updates并保存
  `last.ckpt`；W&B offline；media logging 关闭。

## 实现与门禁

- 新增 `--stereo_training_input=correct|left_only|same_left`，默认保持生产路径
  `correct`。
- DataLoader、online teacher 和原始 batch metadata 不变；仅在 teacher callback
  已附加监督后、student forward 前变换 `video`。
- `left_only` 必须产生 `V=3,E=1` 并令 encoder 的 fusion output 为 `None`；
  `same_left` 必须产生两路逐元素相同的 `V=3,E=2`，且正常执行 StereoFusion。
- checkpoint 记录该训练输入合同；非匹配合同不得 resume。
- 串行 wrapper 拒绝复用已存在输出根，并在每个模型目录记录 `exit_code.txt`；M48
  非零退出时不得启动 D48。
- 启动前要求本地与 H200-1 clean 且精确 SHA 相同、teacher source clean、资产哈希
  匹配、8 GPU 空闲、输出路径不存在，并通过定向 tensor/compile/shell 测试。

## 启动记录

首次 v1 启动没有进入 Python 或占用 GPU，M48 写入 `exit_code=127`，D48 未启动。
第一根因是冻结 venv 的 `torchrun` console script shebang 仍指向跳板机 NFS 路径
`/data-214-30-239-40/.../python`，H200-1 上该解释器不存在。已验证同一 venv 的
`python -m torch.distributed.run` 可执行；launcher 改为模块入口，不修改 DDP、训练或
数据合同。失败输出保留在
`/data/home/frank/experiments/stereo-input-ablation-permode-h2001-20260904-v1`。

修复后的 v2 输出根为
`/data/home/frank/experiments/stereo-input-ablation-permode-h2001-20260904-v2`。
完成服务器门禁后补充代码 SHA、tmux、W&B、resolved config、直接 counters、显存和
首个健康检查结果。
