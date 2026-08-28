# H100 CPU test gate

## 目的与基线

- 目的：记录 H100 阶段 4 在 GPU Slurm 前的 CPU 单元测试 gate，以及对应的本地测试夹具修复。
- 本地分支：`hezhou-las2-h`。
- 修改基线：`d839ad5c3aed0336b5fba2fe62cf40c946455aff`。
- H100 项目：`/gpfs/jiuquyun/projects/Frank/stereo-vae/Stereo-vison-Tokenizer`。

## H100 gate 结果

两个 lock、两个 `uv pip check`、训练/Hy 环境版本、LAS2-H/DA3 source SHA 与 clean
状态、两个 teacher 权重 SHA256、`py_compile` 和 shell syntax 均通过。CPU suite 完成
146 个测试及 9 个 subtests，唯一失败为：

```text
tests/stereo/test_hy_mono_data.py::HyMonoSmokeDatasetTest::test_training_mono_subset_is_deterministic_and_strictly_nested
AttributeError: SimpleNamespace has no attribute 'single_frame_source_index'
```

阶段 4 在此 fail closed；没有提交 GPU Slurm 作业。H100 主仓库和两个外部 source
仓库均保持 clean。

## 根因与修复

生产 CLI 与四模式训练合同要求 `single_frame_source_index=0`，`StereoDataModule` 会将其
传入 `HyMonoSmokeDataset`。失败测试中的 `SimpleNamespace` 未随该必需接口更新。修复仅
在测试夹具中补充 `single_frame_source_index=0`，不为生产代码增加 fallback，也不改变
数据选择、训练语义或公共接口。

## 验证与下一步

本地 Windows Python 缺少 `pytest`，因此定向 pytest 没有进入收集，不能作为动态通过
证据；未为此临时安装或修改本地环境。AST 合同检查确认失败夹具包含常量
`single_frame_source_index=0`，`py_compile` 与 `git diff --check` 均通过。修复 commit
push 后，H100 clean clone 应 fast-forward 到精确 SHA，重新运行相同 CPU gate；只有
H100 CPU gate 全绿后才能另行提交经授权的 GPU Slurm 验收。
