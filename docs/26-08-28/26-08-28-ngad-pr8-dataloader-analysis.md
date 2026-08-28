# StereoTokenizer DataLoader 与 NGAD PR #8 对比及优化验证

## 结论摘要

本次不建议把 NGAD PR #8 的 `NGADCanonicalDataset` 整体替换进 StereoTokenizer。
两者模型合同不同，而且当前 StereoTokenizer 已经实现了 PR #8 的主要数据层设计：episode 级索引、
固定六相机顺序、只解码目标帧、worker 多进程、pinned memory、persistent workers，以及
worker-local PyAV container LRU。特别是 NGAD PR #8 的 LeRobot 路径仍在每个 camera/sample 内
执行 `with av.open(...)`，当前 tokenizer 的容器复用更适合重复小样本训练。

值得合入的是一组最小、数据层限定的修正：

1. 当前 12+12、2-GPU、每 rank 6 个 stereo 样本的容器工作集是 `6 samples × 6 MP4 = 36`
   个容器，因此默认 LRU 容量由 12 调整到 36；
2. 显式暴露 `prefetch_factor`，默认保持 PR #8 和 PyTorch 当前行为的 2；
3. 去掉 `torch.from_numpy(images.copy())` 中不必要的 NumPy 全量复制；
4. 保留 8 workers，不采用 NGAD UMI YAML 中面向另一种 T=17/SANA 负载的 2 workers；
5. 不引入 decoded-sample cache，因为它只对固定过拟合集有利、占用较多 CPU RAM，且对当前真实
   step 的可见收益不足。

**核心判断：数据生产上限明显提高，但当前 LAS2-H + VAE 训练的 DataLoader 已被预取隐藏，端到端
step 不会获得同等比例加速。**

## 基线与参考

- 目标分支：`hezhou-las2-h`
- 本地修改基线：`5b345206babe63a58acdfcb38f375059885ed8cf`
- 运行节点：`qinghua-H200-2`
- 参考仓库：`MaxLiuyy/NGADv1pp`
- PR：`#8 feat: 新增 NGAD canonical TCP 训练与部署接口`
- PR head：`34eef9bb8c54ffb4b1bcbb46f232471519c5b54e`
- 用户指定 commit：`7acc99f34481deee3582a0fb338471dfc8270e83`

经 GitHub API 和实际代码核对，`7acc99f` 本身仅新增 UMI 八卡训练 YAML 与对应测试：

```text
configs/wam/experiments/sana_wam_umi_d20_8gpu_10ep.yaml  +149
 tests/test_umi_training_config.py                        +40
```

DataLoader/Canonical Dataset 实现位于 PR #8 的其他提交和当前 head，而不在 `7acc99f` 本身。

## 哪些设计已经具备

| 设计 | NGAD PR #8 | 当前 StereoTokenizer | 判断 |
|---|---|---|---|
| episode/global window 索引 | `_episode_window_ends` + 反查 | `EpisodeSpan` + `_ends` + `bisect` | 已具备 |
| 固定六相机顺序 | canonical camera ABI | `VIDEO_KEYS` 固定 head/wrist L/R | 已具备 |
| 不跨 episode | anchor 限定 episode | `window_count` / `EpisodeSpan` | 已具备 |
| 只解码目标帧 | seek 后读取目标 index | seek 后读取 T=1/T=4 timestamp | 已具备 |
| worker 多进程 | 可配置 | 默认 8 | 当前更适合本负载 |
| `pin_memory` | 开启 | 开启 | 已具备 |
| persistent workers | worker>0 开启 | 默认开启 | 已具备 |
| `prefetch_factor` | YAML 显式为 2 | PyTorch 隐式默认 2 | 本次显式化 |
| MP4 container 复用 | 每次 `av.open` | worker-local LRU | tokenizer 更优 |
| non-blocking H2D | 显式 `.to(..., non_blocking=True)` | Lightning pinned batch transfer | 已具备 |

以下 NGAD 内容不迁移：T=17、action/proprio、TCP、Memory、SANA/FSDP、Lance/JPEG backend。
它们不属于 StereoTokenizer 的 `[B,3,2,3,T,256,256]` 输入合同。

## 实验设置

真实数据：

```text
Manifest:
/data/home/frank/experiments/stereo_lerobot_cpu_20260824_approval1/
h200_2_local_manifest_v1.jsonl

Dataset root:
/data/shared/datasets/umi_lerobot_v3_260714

Rectification audit SHA256:
41d2bfecaae85dd18f7cfd1a2a3a2177e8fd4aa8897be1cb411d85c3092a7d25
```

固定条件：

```text
seed=1234
12 个 stereo source windows
world_size=2，rank=0
per-rank batch=6
workers=8（另做 workers=4 对照）
pin_memory=True
persistent_workers=True
warmup=8/20 batches
measure=32/80 batches
```

结果目录：

```text
/data/home/hezhou/experiments/stereo-tokenizer-dataloader-ngad-pr8-20260828/
```

## 结果一：纯 stereo 数据生产上限

两次独立复跑的 samples/s 均值：

| 模式 | 原始 cache=12 + NumPy copy | cache=36，仍有 copy | 候选 cache=36 + 去 copy | 候选相对原始 |
|---|---:|---:|---:|---:|
| stereo/single_frame | 119.67 | 185.82 | 230.75 | **+92.82%** |
| stereo/four_frame | 50.92 | 59.16 | 66.82 | **+31.21%** |

分项解释：

- 仅把容器容量 12→36：single-frame `+55.28%`，four-frame `+16.18%`；
- 去除额外 NumPy copy 后，相对 cache=36 再提高约 `+24.18%` / `+12.94%`；
- 4 workers 的两次均值约为 single `92.33 samples/s`、four `30.50 samples/s`，明显低于
  8 workers，因此保留 workers=8；
- prefetch=4 对 single 没有稳定收益，four 的两次结果波动较大，不足以替代默认 2。

这里测的是生产者饱和吞吐，不等于完整训练吞吐。

## 结果二：真实四模式调度的数据等待

四模式为：

```text
mono/single_frame
mono/four_frame
stereo/single_frame
stereo/four_frame
```

### 不模拟模型计算：DataLoader 饱和消费

| 配置 | batch wait mean | median | P90 | samples/s |
|---|---:|---:|---:|---:|
| 原始 cache=12 | 22.43 ms | 0.244 ms | 47.59 ms | 267.53 |
| cache=36 | 8.89 ms | 0.209 ms | 0.294 ms | 675.23 |

这说明 cache=36 显著减少了周期性的重新打开 MP4 和长尾等待，CPU/I/O 余量更大。

### 每步有 200 ms GT+VAE 消费时间

200 ms 取自现有 LAS2-H 四模式训练约 185–260 ms/step 的量级，用于判断 prefetch 能否隐藏读取；
它是训练消费模拟，不冒充新的 GPU 端到端训练结果。

| 配置 | wait mean | median | P90 | samples/s |
|---|---:|---:|---:|---:|
| 原始 cache=12 | 0.237 ms | 0.253 ms | 0.311 ms | 29.9644 |
| cache=36 | 0.211 ms | 0.221 ms | 0.288 ms | 29.9683 |
| 最终候选（独立复测） | 0.227 ms | 0.242 ms | 0.274 ms | 29.9660 |

结论：8 workers + prefetch 已经把读取隐藏在 GPU 工作后面。cache 和 copy 优化会降低 CPU/I/O 压力、
提高数据生产上限，但在当前 step 时长下，端到端 samples/s 差异约为噪声量级，不能宣传成 31%–93%
训练加速。

## 正确性与回归

- 修改前后同一真实样本逐字节相同：
  - single SHA256：`ba9f45e7cc58711982e240c8d852b845e11259e61d95c118a89fd3acf02470b2`
  - four SHA256：`fadd13d687699a3cdabf405d20e9af1f6dfa7ef4c443f8cd4203484a3b5b0306`
- shape 保持：
  - single `[3,2,3,1,256,256]`
  - four `[3,2,3,4,256,256]`
- dtype/range 保持 `float32`、`[-0.5,0.5]`；finite ratio 为 1.0；
- 定向测试：`25 passed`；
- 完整测试：`147 passed, 9 subtests passed, 2 warnings`；
- `git diff --check` 通过。

## 最小代码改动

生产代码只涉及数据参数/读取：

```text
stereo_tokenizer/lerobot_data.py
stereo_tokenizer/data.py
scripts/stereo/train_stereo_vae.sh
train_stereo_vae.py
tests/stereo/test_entrypoints_source.py
```

没有修改 StereoVAE、LAS2-H、DA v3、loss、optimizer、模型 shape 或样本顺序。

## 决策与后续

1. **这个想法值得做、需要改，还是放弃？**
   - 需要改：保留最小数据层优化；
   - 放弃：整体替换成 NGAD canonical loader，以及把原始数据吞吐收益等同于训练加速。
2. **最大风险是什么？**
   - cache=36 会让每个 worker 最多保持 36 个 MP4 container；8 workers 时每 rank 最多约 288 个。
     单进程上限 36，低于当前 `ulimit -n=1024`，但长训仍需关注文件描述符和 worker RSS。
3. **最缺的关键证据是什么？**
   - 最新候选 commit 上真实 LAS2-H + StereoVAE 的短 Kineto 配对实验；目前 200 ms 消费实验已经强烈表明
     DataLoader 不在关键路径，但它不是 GPU 真训练替代品。
4. **今天可以执行的最小一步是什么？**
   - 使用 `workers=8, prefetch=2, video_cache=36` 跑 50–100 个真实 optimizer steps，确认
     `DataLoader wait` 仍低于 1 ms；不再做大规模 DataLoader 重构。
5. **置信度**
   - 数据输出不变：`99%`；
   - 数据生产上限提高：`90%`；
   - 当前完整训练收益很小：`85%`；
   - 综合：`90%`。
