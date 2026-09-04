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

## Z96 首次启动与显存修复

- 首次启动 commit：`25a58c3b8184c10a553b0d6f7964e00c257541ab`。
- 首次输出：
  `/data/home/frank/experiments/stereo-latent-ablation-bs384-h2002-20260904-v1`。
- W&B offline run `2uo5ifis` 已创建，但在第一个 optimizer update 完成前失败。
- 第一根因是 mode-aware BS48 的 LPIPS forward OOM：每卡约 139.42/139.80 GiB
  已用，再申请 384 MiB 失败。8 张 GPU 随后释放，Z24/Z48 未启动。
- 保持 batch、GA、loss、精度和 optimizer 合同不变，对冻结 LPIPS forward 使用
  non-reentrant activation checkpoint；只在 backward 重算 LPIPS 特征，不改变 loss
  reduction 或 effective global batch。该修复对 Z96/Z24/Z48 三组统一生效。
- 失败输出保留，不复用、不删除；修复后使用新的唯一输出根重新启动。

## Z96 第二次启动与数值稳定修复

- 第二次输出：
  `/data/home/frank/experiments/stereo-latent-ablation-bs384-h2002-20260904-v2`。
- LPIPS activation checkpoint 将训练显存从约 139 GiB 降至约 56 GiB，Z96 完成
  三个 generator updates 后，在下一次 forward 检测到 non-finite depth prediction；
  该次停止不是 OOM，Z24/Z48 未启动。
- W&B 中失败前 RGB、depth、gradient、LPIPS 和 KL loss 均为有限值，学习率按
  warmup 从 0、5e-6、1e-5 增长。失败发生在 mono/four-frame update 之后。
- 根因复现为 LPIPS channel normalization 在全零 VGG/ReLU feature 处反向经过
  `sqrt(0)`，FP32 与 BF16 均会产生 NaN gradient。将 epsilon 放入平方根内部，
  使零特征梯度有限；正常非零特征、batch、loss 权重和 reduction 合同保持不变。
- 仅修复零点后，同合同重跑仍在首次 mono/four-frame update 后出现 non-finite
  prediction，说明 BF16 LPIPS backward 还存在其他溢出路径。LPIPS 每个 frame 的
  forward/recompute 和输入梯度改为 FP32；tokenizer 主干继续 BF16，activation
  checkpoint、batch、loss 与 optimizer 合同不变。
