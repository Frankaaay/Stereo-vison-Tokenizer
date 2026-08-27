# uv 双环境依赖迁移

## 目的

将 `hezhou-las2-h` 的手工 `venv + pip + requirements.txt` 安装方式迁移为两个相互
隔离、可锁定的 uv project，消除旧 `pytorch-lightning==1.5.4` 与未固定 `lightning`
并存造成的依赖歧义，同时保持当前训练、teacher、数据和 CUDA 语义不变。

## 分支与基线

- 本地分支：`hezhou-las2-h`
- 修改前 HEAD：`597dddd2057b5585743de5e7125bb70a30601645`
- 运行位置：Windows 本地 `C:\Project\Stereo-vison-Tokenizer`
- 本轮不连接 H100/H200，不安装远端环境，不启动测试、评估或训练。

## 环境合同

### 训练环境

- 声明：根目录 `pyproject.toml`
- 锁文件：根目录 `uv.lock`
- 实际 venv：仓库外 `$RUNTIME_ROOT/train`
- Python：`>=3.12,<3.13`
- 平台：Linux x86_64
- PyTorch：`2.7.1+cu126`
- torchvision：`0.22.1+cu126`
- PyTorch Lightning：`2.5.6`
- NumPy：`1.26.2`
- xFormers：`0.0.31.post1`

PyTorch 与 torchvision 只从官方 CUDA 12.6 wheel index 解析。LAS2-H 继续使用冻结源码
目录；DA3 继续检出冻结 SHA，并在 `uv sync` 后用 `uv pip install --no-deps -e` 暴露，
防止其安装过程重新解析或替换 locked dependency。

### Hy exporter 环境

- 声明：`environments/hy-export/pyproject.toml`
- 锁文件：`environments/hy-export/uv.lock`
- 实际 venv：仓库外 `$RUNTIME_ROOT/hy-export`
- Python：`>=3.12,<3.13`
- 平台：Linux x86_64
- PyLance：`8.0.0`
- PyArrow：`23.0.0`
- NumPy：`2.5.2`
- Pillow：`12.3.0`

训练与 exporter 使用两个独立 project、lock 和 venv，不尝试在同一个依赖解中兼容两套
NumPy ABI。

## 文件变化

- 新增根训练环境的 `pyproject.toml`、`uv.lock` 和 `.python-version`；
- 新增 Hy exporter 的独立 `pyproject.toml` 与 `uv.lock`；
- 删除手工维护的 `requirements.txt`，避免出现第三份依赖真相；
- 更新 `README.md` 的 uv sync、DA3 editable、双环境和目录职责说明；
- `.gitignore` 忽略误生成的 `.venv/`。

## 本地验证

本地只负责 lock 解析、TOML/lock 一致性、源码静态测试和 diff 检查。完整安装、
`uv pip check`、CUDA/import 版本打印与源码测试必须在经授权的 Linux GPU runtime 中完成；
Windows lock 解析不能替代 H100/H200 runtime acceptance。

## 当前结论与下一步

两个 lock 必须通过 `uv lock --check`。本地静态检查通过后，下一步是在新 H100 集群的
共享 GPFS 路径中创建两个全新 venv，执行 `uv sync --frozen`，再完成 DA3 editable、
`uv pip check`、版本打印、CUDA preflight 和仓库测试。任何 wheel、ABI、外部 source 或
checkpoint 不匹配都应 fail closed，不得修改 lock 后直接开始训练。
