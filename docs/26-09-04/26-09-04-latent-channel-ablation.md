# Stereo Tokenizer latent channel 消融

## 目的

在同一代码、数据、训练预算和评测合同下，从零训练 96、24、48 三个 latent
channel 配置，比较 Quality、Rate 和 Speed。历史 48-channel checkpoint 只作为回归
参考；正式 latent 因果比较只使用本轮三个新训练结果。

## 冻结训练合同

- 目标分支：`hezhou-las2-h`；启动 commit 在同步 H200-2 后记录。
- 节点：H200-2，单节点 8 GPU，BF16。
- 顺序：96 → 24 → 48；前一组只有正常退出后才启动下一组。
- 每组从零训练 40,000 generator updates，不加载 resume、continuation 或 stage
  transition checkpoint。
- `MODE_BATCH_SIZES=48:48:48:24`，
  `MODE_GRAD_ACCUMULATES=1:1:1:2`，四种模式 effective global batch 均为 384。
- 每组处理 15,360,000 logical samples；模式权重 `35:35:15:15`，
  mono dataset 权重 Hy:LIBERO=`9:1`。
- seed 和 mode schedule seed 均为 1234。
- RGB、relative depth、relative gradient、LPIPS 权重为 `1.0/1.0/0.1/1.0`；
  KL 权重 `1e-6`，KL warmup 100 updates；GAN 与 feature matching 关闭。
- generator LR 与 minimum LR 均为 `1e-4`，optimizer warmup 20 updates。
- 在线 DA3 与 LAS2-H teacher；LAS2-H 4 iterations、pair microbatch 48；online GT
  cache 关闭。
- validation 每 2,000 generator updates；checkpoint 每 5,000 updates 并保存
  `last.ckpt`。
- W&B logger 开启；H200 无公网时使用 offline mode，每个 latent 配置使用独立 run。

## 实现范围

- `StereoVAE` 只接受 24、48、96 三种 latent channel，其他值 fail closed。
- launcher 通过 `LATENT_CHANNELS` 选择配置，默认仍为 48。
- 串行 wrapper 固定顺序 96、24、48，拒绝复用已存在的总输出目录。
- 不修改 encoder/decoder width、latent 空间分辨率、temporal slot、loss、数据或
  optimizer 路径。

## 启动门禁

- 本地与 H200-2 必须为同一 clean commit，且远端只能 fast-forward 同步。
- 实时核对 8 张 GPU、runtime、H200-2 node-local manifests、teacher source/weights
  SHA、磁盘空间和唯一输出路径。
- H200-2 定向 tensor 测试、Python compile 与两份 shell `bash -n` 必须通过。
- 启动后观察至少 5 分钟，确认 8 ranks、W&B run、resolved config、有效 loss、
  generator update 增长、无 traceback/OOM/NaN，并记录 tmux、日志和输出路径。

## 状态

当前为本地实现阶段，尚未启动远端训练。启动后的 commit、run ID、tmux、输出、
实时计数和健康检查结果将在本文件追加。
