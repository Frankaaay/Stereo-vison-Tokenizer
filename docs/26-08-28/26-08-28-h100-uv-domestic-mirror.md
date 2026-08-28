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
- 本地执行 `uv lock --check` 通过，共解析 177 个包。

## H100 灰度验收

- 状态：进行中。
- 运行位置：H100 Slurm，Job `345`，节点 `xn01-gpu1-0049`，1 张 H100 80GB；开始时间 `2026-08-28T20:27:54+08:00`。
- 关键命令：`sbatch /gpfs/jiuquyun/projects/zetyun/validation/uv-domestic-mirror-db2bb23-32f55fa-260828/validate.sh`。
- 日志：`/gpfs/jiuquyun/projects/zetyun/validation/uv-domestic-mirror-db2bb23-32f55fa-260828/logs/uv-mirror-accept-345.out`。
- 验收内容：NGAD 环境先在独立空 cache 中完成冷同步和 CUDA 检查；本项目随后使用另一份独立空 cache 执行冷 `uv sync --locked`、PyTorch/CUDA/H100 导入和一次热同步。
- 启动检查：Job 正常运行，已分配 H100；当前处于 NGAD 冷安装阶段。
- 异常记录：Job `343` 在 uv 启动前因管理员账号无权创建 `/local/cache/users/zetyun` 退出；未触碰任何用户 cache。Job `345` 改用管理员自己的 GPFS validation 目录，不修改系统权限。
- ETA：以 `2026-08-28 20:28 +08:00` 为估算时点，两个环境的冷安装主体预计 `20:38–20:53` 完成；日志回读和文档收尾预计再需约 5 分钟。尚无完整安装吞吐，属于初估。

## 产物与回滚

- 无 checkpoint、训练 output 或大日志；仅提交验收 Job `343`/`345`，未启动训练，也未修改或取消其他 Slurm 作业。
- 回滚只需恢复本提交前的 `pyproject.toml` 与 `uv.lock`；TUNA 普通包配置不受影响。
