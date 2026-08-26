# Final review P1 fixes

## 目标与范围

- 分支：`hezhou-las2-h`
- 修改基线：`f013f775d9ae9878e71105f7ab7441667bb6ca9f`
- 执行位置：Windows 本地 `C:\Project\Stereo-vison-Tokenizer`
- 目标：修复 final code review 中确认的三个 P1：stereo online GT cache provenance
  不完整、LAS2-H source 未 fail-closed、`eval_eye_mode=mono` 没有真实数据与 teacher 路径。
- 本次没有启动训练、评估、预处理、远端同步或 GPU 任务；按用户授权在验证完成后提交，
  不 push。

## 修改

### Stereo online GT cache v4

`stereo-online-foundation-gt-v4` namespace、metadata 与 callback state key 现在共同编码：

- teacher backend、LAS2 source SHA、checkpoint SHA、valid iterations；
- TensorRT engine/manifest SHA；
- `single_frame_source_index`；
- disparity min/max；
- LR consistency absolute/relative threshold。

任一 target 语义变化都会生成不同 namespace。旧 v3 cache 不会被 v4 路径命中。
DA3 cache 同步升级到 `da3-processed-relative-depth-cache-v3`，并将
`single_frame_source_index` 加入 namespace、metadata 和 callback state key；旧 v2 cache
不会被复用。

### LAS2-H source provenance

- 新增必填 `--las2_h_source_sha` / `LAS2_H_SOURCE_SHA`。
- teacher 初始化前检查外部 repo 的完整 HEAD 和 clean worktree。
- source SHA 写入 resolved config、run manifest、evaluation result 和 cache provenance。

### 正式 mono evaluation

- 新增 `HyMonoDataset`，复用 immutable Hy RGB manifest/cache 合同，不固定 48 条；
  `HyMonoSmokeDataset` 继续保留训练所需的 48 条约束。
- dataset 与 eval 显式执行 `[B,1,1,3,T,256,256]` 合同；single mode 使用配置的
  `single_frame_source_index`。
- mono eval 使用 pinned/clean DA3 source 和 checkpoint，在无 Student padding 的 DA3
  输入上推理，再通过 `GeometryMapping` 映射回 Student canvas。
- validity 仅使用 finite、positive、non-padding，不增加 confidence threshold。
- `cam_high` 指标包含 RGB L1、有效 depth pixel count、centered relative-log-depth
  L1/RMSE；DDP sample sharding 无重复且完整计数。
- mono/stereo case visualization 共用动态 view contract；visualization 可接受 single、four
  或二者同时存在。

## P2 follow-up

- 评估参数升级为 `eval_eye_mode=mono|stereo|both` 与
  `eval_temporal_mode=single_frame|four_frame|both` 的笛卡尔积，并直接复用训练侧
  `MODE_IDS`。
- 一次四模式命令内部拆为 mono/DA3 和 stereo/FoundationStereo 两个 session；同一 eye
  mode 的 single/four 每个 batch 共享一次 teacher GT。
- `metrics.json` 改为完整 mode ID key，并按 eye mode 分别记录 dataset/teacher provenance；
  每个 mode 自身也保存精确 sample count、RGB/depth 指标、valid pixels 和 provenance。
- 可视化按实际 temporal 输出动态布局，并分别写入 `visualizations/mono/` 与
  `visualizations/stereo/`，各自维护 `cases.json`。
- 请求资产、数据合同、mode ID、single-frame index 和不可覆盖输出目录均在 Student/CUDA
  初始化前校验。
- SDPA 路径现在合并 relative bias、padding mask 与 causal mask，eval 显式关闭 attention
  dropout，并删除字符串形式的 Torch 版本比较。
- four-mode 训练临时强制 `single_frame_source_index=0`；评估仍允许合法的 0..3 数据选择。

## 验证

已执行的本地验证：

```text
Python AST parse: 59 files
python -m unittest tests.stereo.test_entrypoints_source tests.stereo.test_source_boundary -v
"C:\Program Files\Git\bin\bash.exe" -n scripts/stereo/train_stereo_vae.sh
git diff --check
```

结果：59 个 Python 文件语法解析通过，24 个 source/boundary tests 通过，shell syntax 通过，
`git diff --check` 无错误。新增的四模式 orchestrator、动态可视化与 SDPA tensor tests 已写入
`tests/stereo/test_eval_four_mode.py` 和 `tests/stereo/test_attention_sdpa.py`。

本地 Python 没有 Torch/Lightning，因此 tensor tests、真实 LAS2/DA3 load、CUDA forward 和
端到端 evaluation 不能在本机执行。正式接受前仍需在已授权且具备 pinned assets 的 Torch/CUDA
环境运行：

```bash
python -m pytest -q tests
python eval_stereo_vae.py ... --eval_eye_mode mono --eval_temporal_mode both
python eval_stereo_vae.py ... --eval_eye_mode stereo --eval_temporal_mode both
```

成功标准是 exit code 0、精确 sample count、有限指标、`metrics.json`、确定性 case images，
以及无 traceback/OOM。当前状态是本地实现与静态验证阶段，不代表 GPU runtime 已通过。
