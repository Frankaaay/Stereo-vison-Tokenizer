# H100 uv 国内 PyTorch 镜像迁移

## 目的与范围

- 目标：保持普通用户命令 `uv sync --locked` 不变，将 CUDA 12.6 PyTorch wheel 从官方海外源切换到已实测的阿里云镜像。
- 分支：`hezhou-las2-h`。
- 修改前基线：`d03d966a111e7ea1002abcd6779150c93f1b285d`；变更提交为本文档所在提交。
- 仅修改 `pyproject.toml`、由 uv 生成的 `uv.lock` 及本文档；不修改训练代码、依赖版本、系统代理、`/etc/uv/uv.toml` 或个人 cache。

## 来源映射

- 普通 Python 包：继续使用集群 `/etc/uv/uv.toml` 中的 TUNA。
- `torch==2.7.1`、`torchvision==0.22.1`：显式绑定 `https://mirrors.aliyun.com/pytorch-wheels/cu126/`。
- GitHub、直接 URL、Hugging Face 和 NVIDIA 独占源：本次不调整。

## 精确 wheel 测试

- 位置：H100 登录节点；测试对象为 `torch-2.7.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl`，不是镜像首页。
- 阿里云 8 MiB 分段下载：`4,155,555 B/s`，总耗时 `2.019 s`。
- 官方源在 60 秒内仅完成 `6,788,907 / 8,388,608` 字节，平均 `113,146 B/s`；阿里云约快 `36.7` 倍。
- `torch`、`torchvision` 两个锁定 wheel 均返回 HTTP `206`，精确版本覆盖完整。

## 修改与验证

- `pyproject.toml` 保留具名、`explicit = true` 的专用 index，并增加 `format = "flat"`。
- `uv.lock` 由 `uv lock` 重新生成；包版本和 CUDA 变体保持不变。
- 本地锁文件检查和 H100 空缓存安装、CUDA 导入验收将在提交后继续执行，结果追加到本文档。

## 产物与回滚

- 无 checkpoint、训练 output 或大日志；未启动或改动 Slurm 作业。
- 回滚只需恢复本提交前的 `pyproject.toml` 与 `uv.lock`；TUNA 普通包配置不受影响。

