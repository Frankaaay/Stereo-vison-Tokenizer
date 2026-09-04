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

正式串行训练 v11 已在 H200-2 运行，当前为 Z96；Z24/Z48 等待 Z96 正常退出后
由同一 wrapper 自动启动。训练代码、输出、W&B 和五分钟健康检查见下文。

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
- FP32 LPIPS 同合同重跑仍复现相同失败；下一次仅运行短诊断，临时启用 autograd
  anomaly detection 定位首个 non-finite backward 算子，不计入正式 ablation。
- anomaly 首个 NaN 位于 encoder temporal Transformer 的 GEGLU backward；关闭
  LPIPS 后仍复现，排除 LPIPS loss 分支。临时增加 RGB、raw depth、latent 和
  posterior 边界梯度探针，继续定位首个 Inf 的来源区间。
- 梯度探针初版插入位置错误，导致 diagnostic v7/v8 未执行真实 training step；
  其进度条不作为稳定性证据，任务已停止且探针和 anomaly 均从正式代码移除。
- 正式修复对 generator backward 使用 1/1024 静态缩放，并将 gradient clip 阈值
  同比缩放。原始 loss 日志、batch、各 loss 相对权重和 clip 后梯度方向不变；Adam
  一阶与二阶矩的同倍率缩放相消，仅用于避免 BF16 中间 backward gradient 溢出。
- 缩放版正式 schedule 仍在第 4 个真实 batch 后污染参数。临时在 clip 前、clip 后
  和 Adam step 后扫描第一个 non-finite gradient/parameter，输出精确参数名和阶段。
- 状态探针确认 optimizer/clip 前 `encoder.enc_temporal_position` 的 2048 个梯度已
  全部为 NaN，与 anomaly 的 encoder temporal GEGLU 栈一致。最终仅将四帧
  encoder/decoder temporal Transformer 固定为 FP32；空间主干与其余训练仍 BF16。
- 移除无效的整体 backward 缩放、状态探针、anomaly 和不必要的 FP32 LPIPS；保留
  LPIPS activation checkpoint 与零特征稳定 normalization。

## 正式串行训练 v11

- 训练代码 commit：`872ac395d168db637588ad776a631d1f1cda027c`。
- 启动时间：2026-09-04 17:21:59 CST；tmux：
  `stereo-latent-ablation-bs384-20260904-v11`。
- 输出根：
  `/data/home/frank/experiments/stereo-latent-ablation-bs384-h2002-20260904-v11`；
  当前 Z96 W&B offline run：`0pzk4bih`。
- launch artifact：
  `/data/home/frank/experiments/stereo-latent-ablation-bs384-h2002-20260904-v11.launch.sh`，
  SHA256 `e23c008e16f5e2f681642695a6cb585b6f36c6a9e40dcd96ad8848c231a2aef3`。
- 2026-09-04 17:27:46 CST 的五分钟门禁通过：tmux 和 8 个训练 rank 存活，
  8 卡利用率 99--100%，未检出 traceback、OOM 或 non-finite；重型 batch 下每卡
  显存约 125,421/143,771 MiB，保留约 18.3 GiB 余量。
- W&B 原始 datastore 在约 5 分钟时已记录 115 generator updates / 133 batch
  updates；四种模式计数分别为 stereo single/four 17/18、mono single/four
  41/39，证明 GA2 和四种数据路径均有真实 optimizer 更新。
- 去除前 20 个 batch 的启动段后，当前 Z96 实测为 2.033 s/batch、
  2.344 s/generator update；这是短窗口健康检查数字，不替代完整训练吞吐统计。
- 2026-08-29 的旧 mode-aware BS384 稳态为 1.226 s/batch，因此当前工程配置
  端到端约慢 65.8%。该对照包含 Z48→Z96、mono 单视角→三视角、LPIPS
  activation checkpoint 以及 temporal FP32 等多项差异，不能将 65.8% 全部归因
  于 LPIPS checkpoint。无 checkpoint 的当前同合同 Z96 在首个 optimizer update
  前 OOM，无法得到同合同吞吐基线。

## BS24 保留 LPIPS 激活测速

- 用户于 2026-09-04 18:26 CST 要求停止 v11；18:27:24 CST 已确认 tmux、
  torchrun/rank 进程全部退出，8 张 GPU 均为 0 MiB、0% 利用率。
- 新候选将四种 mode 固定为 per-device BS24、GA1，effective global batch 为 192；
  temporal Transformer 全部恢复 BF16。
- LPIPS 不再使用 activation checkpoint，保留 forward 激活供 backward 使用，
  避免 backward 重算 LPIPS feature forward。
- 先只运行 Z96 短测速，以稳定 `samples/s` 对比 v11；不同 batch 的 `step/s`
  不作为结论。通过显存和有限梯度门禁后，再决定是否作为正式三组训练合同。

### v12-bench 结果

- 代码 commit：`985df742bef45ab0c76d6a6728ce3c7287d7d355`；启动时间
  2026-09-04 18:31:44 CST；W&B offline run：`waq8kwq7`。
- 输出：
  `/data/home/frank/experiments/stereo-latent-ablation-bs192-h2002-20260904-v12-bench`；
  launch artifact SHA256：
  `57349b2214f4b4c9181f2054b1553ca14071d59ba0b6ec98a5c0ec07050a7c2d`。
- Z96 运行至 W&B 记录 177 generator updates，四种 mode 均覆盖；未检出 OOM、
  non-finite 或 traceback。重型 batch 下每卡约 89,823/143,771 MiB。
- 从 generator update 20 开始按 W&B 时间戳计算：v12 BS24/BF16 temporal/
  LPIPS 保留激活为 1.101 s/update、174.42 samples/s；v11 BS384/FP32 temporal/
  LPIPS checkpoint 为 2.275 s/update、168.81 samples/s。因此 v12 按样本吞吐快
  约 3.3%，同时显存降低约 35.6 GiB/卡；其 step/s 约快一倍主要来自 global batch
  减半，不能作为吞吐翻倍解释。
- benchmark 于 18:38:55 CST 人工停止并确认 tmux、torchrun/rank 全部退出，8 张
  GPU 均为 0 MiB、0%。`exit_code=1` 来自人工 SIGTERM，不是训练异常；尚未启动
  新的正式三组串行训练。

## Per-mode BS 48:32:48:32 可行性测试

- 按用户要求定向恢复首次 OOM 时的计算路径：LPIPS 使用普通 forward 并保留激活，
  normalization 恢复 epsilon 位于范数外的原公式，encoder/decoder temporal
  Transformer 保持 BF16；不回滚 latent ablation、三视角数据和实验记录。
- 候选物理 batch 为 `48:32:48:32`，GA 为 `1:1:1:1`；按 mode 顺序对应
  mono single/four、stereo single/four。8 卡 effective global batch 分别为
  384/256/384/256。
- runtime、launcher 和 checkpoint 已能记录并校验不等的 per-mode effective batch；
  mode schedule 仍按 update 权重 `35:35:15:15`，因此样本占比不再等于 update 占比，
  后续正式训练预算必须按真实 per-mode sample counter 对齐。
- 先在 Z96 上运行四模式短测试；每种模式分别记录显存、有限梯度和 samples/s，再决定
  是否继续向上搜索 batch 上限或启动正式训练。

### 搜索结果与正式选择

以下均为 Z96、BF16、LPIPS 保留 forward 激活、GA1；显存为 8 ranks 中最大的
 CUDA allocated 峰值，单位 GiB：

| run | per-mode BS | mono single | mono four | stereo single | stereo four | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v13 | 48:32:48:32 | 33.42 | 99.74 | 38.63 | 113.82 | 400 train + 20 val 通过 |
| v14 | 128:40:112:36 | 86.39 | 124.19 | 87.96 | 127.81 | 100 train + 20 val 通过 |
| v15 | 192:40:160:36 | 128.97 | 124.19 | 124.95 | 127.81 | 100 train + 20 val 通过 |

- v13/v14/v15 的 W&B offline run 分别为 `smo7o71r`、`85ruhlxp`、
  `is2wnvsc`；launch artifact SHA256 分别为
  `3369c9b53db6320871d5d65a04650438e4913f3cd2fce72ec1fad8b653217f95`、
  `2dcffb12ec137ef2ffe530e97b2df06446c89eb6bd215d7fb6b789302650fcf4`、
  `c8444448d7cff5f8936228885a556f5b30d5d1179dba5fe2d03e43531ce944ed`。
- 选择 `192:40:160:36` 作为正式组合：v15 最大 allocated 128.97 GiB、最大
  allocator reserved 129.30 GiB，nvidia-smi 最高约 134.36/143.77 GiB；保留约
  9.4 GiB 设备余量且通过完整 validation，不继续逼近首次 OOM 边界。
- 8 卡 effective global batch 依次为 1536/320/1280/288。40,000 updates 共处理
  35,392,000 logical samples；mode update 权重仍为 35:35:15:15，质量比较必须读取
  per-mode sample counter，不能再假设四种 mode 样本数相等。

### 正式串行训练 v16

- 运行位置：H200-2，8×H200；代码 commit
  `f3fba13f5e0585885209dc27539dcf2b3f6600a2`。
- 顺序与预算：Z96 → Z24 → Z48，每个模型 40,000 generator updates；per-mode BS
  `192:40:160:36`、GA `1:1:1:1`、更新权重 `35:35:15:15`。
- tmux：`stereo-latent-ablation-permode-20260904-v16`；output root：
  `/data/home/frank/experiments/stereo-latent-ablation-permode-h2002-20260904-v16`。
- launch artifact：
  `/data/home/frank/experiments/stereo-latent-ablation-permode-h2002-20260904-v16.launch.sh`，
  SHA256 `6735574a2a64c6e2ed74a7b14f417bf5e270362c03d73da02d29ce49bd22ad4b`。
- Z96 W&B offline run：`wts892on`。2026-09-04 20:27 的 5 分钟健康检查到
  step 104（训练计时 5:18）：8 卡持续工作，nvidia-smi 最高约 134.37 GiB/卡，
  未见 OOM、NaN、Traceback 或 CUDA error。
- 随后的直接 W&B counter 快照为 generator/batch updates 116；四模式 updates 为
  41/40/17/18，samples 为 62,976/12,800/21,760/5,184；对应最近 total loss
  为 0.923/1.057/1.288/1.268，均有限。
- 按首个模型稳定段约 2.7--2.8 秒/update 估算，每个 40k 模型约 31 小时，三个模型
  串行约 93 小时；checkpoint/validation 和模式分布会使实际完成时间小幅波动。
