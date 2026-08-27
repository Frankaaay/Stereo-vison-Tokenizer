# 26\-08\-27\-H100集群使用指南

> 文档性质：管理员提供的 H100 集群使用指南脱敏快照。
> 来源日期：2026-08-27。原始文件仅保留在用户本地，不纳入仓库。
> 脱敏说明：公网地址、端口和其他登录参数已移除；统一使用本机 SSH alias `h100`。
> 运行事实：路径、QOS、容量、软件版本和资源状态属于易变事实，操作前必须实时验证。

# 开始之前

当前集群提供 10 台计算节点、共 80 张 H100 80GB GPU。所有训练任务由 Slurm 统一排队和分配资源，用户不需要也不应自行挑选空闲机器。

|项目|用户需要知道的内容|
|---|---|
|登录入口|使用调用方 `~/.ssh/config` 中的 `h100` alias；地址、端口、用户名和密钥不写入仓库|
|资源管理|通过 Slurm 的 `sbatch`、`srun`、`squeue` 使用 GPU|
|共享存储|`/gpfs/jiuquyun`，所有节点可见，约 40TB|
|节点本地盘|`/local`，每台计算节点约 14TB RAID0，仅用于可重建缓存和临时数据|
|软件方式|推荐共享目录中的 uv 虚拟环境；也支持 Enroot/Pyxis 容器|
|外网访问|登录后自动配置，无需用户启动或修改代理服务|

**账号开通后即可直接使用：**管理员已经统一配置 Zsh、Oh My Zsh 及常用插件、Miniconda，并默认激活 base；外网代理和各类训练缓存目录也会自动生效，不需要手动执行 source，也不需要在每台服务器重复配置。代码和 Conda/uv 环境请放在 `/gpfs/jiuquyun/projects/$USER` 下，个人配置和小文件放在 `/gpfs/jiuquyun/home/$USER`；这些共享目录在所有计算节点上看到的是同一份内容。

**不要直接在登录节点运行训练或占用大量 CPU/内存。**登录节点只用于编辑代码、管理环境、传输小文件、提交任务和查看日志。

# 首次登录与快速检查

## 配置 SSH

公网地址、端口、用户名和密钥等连接参数只保存在调用方的 `~/.ssh/config` 或受控凭据存储中，
不写入仓库。管理员完成账号和公钥配置后，统一使用本地 alias 登录：

```bash
ssh h100
```

## 登录后的第一次检查

```bash
whoami
hostname
pwd
echo "$SHELL"
zsh --version
conda --version
echo "$CONDA_DEFAULT_ENV"
sinfo
squeue -u "$USER"
uv --version
ks3util -v
```

正常情况下，你会看到自己的用户名、登录节点名称、`/usr/bin/zsh`、Conda 版本，且 `CONDA_DEFAULT_ENV` 为 `base`。登录节点没有可供训练使用的 GPU，因此此处运行 `nvidia-smi` 不是判断集群 GPU 是否可用的方法。

## 运行第一个 GPU 检查任务

使用 debug 队列申请 1 张 GPU，并进入计算节点交互终端：

```bash
srun --qos=debug \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=00:30:00 \
  --pty bash -l
```

资源分配成功后运行：

```bash
hostname
nvidia-smi -L
echo "$CUDA_VISIBLE_DEVICES"
exit
```

应只看到分配给你的 GPU。输入 `exit` 后，交互任务结束并释放资源。

# 正确的使用方式

## 从代码到训练的标准流程

1. 登录集群，在共享项目目录中准备代码与环境。

2. 选择符合 GPU 数量和运行时长的 QOS。

3. 先用 `sbatch --test-only` 检查脚本能否被接受。

4. 使用 `sbatch` 提交，记录返回的 Job ID。

5. 通过 `squeue` 查看排队原因，通过日志文件查看运行进度。

6. 训练结束后将长期保留的 checkpoint 放共享存储或 KS3，清理无用临时文件。

## 不要绕过 Slurm

没有活跃作业时，普通用户不能直接 SSH 到计算节点；这是正常的资源隔离策略。调试正在运行的任务时，优先使用：

```bash
srun --jobid=<JobID> --overlap --pty bash -l
```

不要通过长期后台进程、共享账号或其他方式绕过 Slurm 占用 GPU。Slurm 会同时负责排队、公平性、GPU 隔离和资源释放。

# 目录与存储

## 目录用途

|路径|是否跨节点共享|适合存放|
|---|---|---|
|`/gpfs/jiuquyun/home/$USER`|是|个人配置、小型脚本、日志|
|`/gpfs/jiuquyun/projects/$USER`|是|代码、uv 环境、实验配置|
|`/gpfs/jiuquyun/checkpoints/$USER`|是|需要跨节点访问和保留的 checkpoint|
|`/gpfs/jiuquyun/datasets`|是|团队共享数据集；默认按只读方式使用|
|`/gpfs/jiuquyun/containers`|是|公共容器镜像|
|`/local/cache/users/$USER`|否|Hugging Face、Torch、Triton、uv、pip 缓存及临时文件|
|KS3 对象存储|通过网络访问|归档、备份、跨系统分发；不建议训练时逐样本读取|

## 共享数据集

团队数据集统一放在 `/gpfs/jiuquyun/datasets`，所有计算节点使用相同路径。以下路径是训练配置应引用的固定路径：

- 共享数据集默认只读使用；需要新增或修正数据时先联系管理员。

- 训练配置直接引用上述 GPFS 路径，不要在个人 projects、home 或 checkpoint 目录重复保存整套数据。

- 不要把完整数据集复制到单台节点的 `/local`；如项目确需本地预取，只缓存可重新生成的临时子集。

## 存储原则

- **代码和环境放 GPFS：**这样同一套代码与环境在所有计算节点可见，不需要逐台同步。

- **缓存和临时文件放 /local：**节点本地 RAID0 速度更高，适合可重新下载或生成的内容。

- **重要结果不要只放 /local：**本地盘不是共享盘，任务换节点后不可见，也不作为唯一备份。

- **KS3 用于归档和搬运：**训练前将所需数据放到 GPFS；训练结束后再批量上传结果，避免 200MB/s 左右的下行成为训练瓶颈。

- **checkpoint 要控制保留数量：**建议保留 last、best 和少量里程碑版本，及时删除无价值的高频快照。

## 检查空间

```bash
df -h /gpfs/jiuquyun
du -sh "/gpfs/jiuquyun/home/$USER"
du -sh "/gpfs/jiuquyun/projects/$USER"
du -sh "/gpfs/jiuquyun/checkpoints/$USER"
```

`/local` 只存在于计算节点。如需查看当前作业所在节点的本地空间，可在作业内运行：

```bash
srun df -h /local
```

# 代码与 Python 环境

## 代码放在哪里

建议每个项目放在自己的共享项目目录：

```bash
mkdir -p "/gpfs/jiuquyun/projects/$USER"
cd "/gpfs/jiuquyun/projects/$USER"
git clone <仓库地址>
cd <仓库目录>
```

集群已自动提供外网代理。正常情况下可以直接访问 GitHub、PyPI 和 Hugging Face，无需手动加载任何系统环境脚本。不要自行启动、停止或修改系统代理。访问私有仓库时，请使用自己的 SSH Key 或访问令牌，不要把令牌写入脚本、仓库或日志。

## Zsh 与默认 Conda

登录后默认使用 Zsh 和 Oh My Zsh，已安装常用补全、语法高亮、Git、Python、Conda、tmux 等插件。Miniconda 的 `base` 环境会自动激活，可用于执行 `conda`、`python` 和 `uv` 等基础工具。

请不要把项目依赖直接安装到 `base`。每个项目应建立独立环境，并保存依赖文件。

## 使用 Conda 创建共享环境

集群会自动把命名 Conda 环境放到你的共享项目目录，因此只需创建一次，在所有计算节点都可使用：

```bash
conda create -n myproject python=3.11
conda activate myproject
python --version

# 下次登录或在作业中直接激活
conda activate myproject
```

查看环境使用 `conda env list`；退出当前项目环境使用 `conda deactivate`。环境本体位于 `/gpfs/jiuquyun/projects/$USER/conda-envs`，安装包的可重建缓存会自动放到计算节点本地 RAID0。

## 使用 uv 创建项目环境

如果项目已使用 `pyproject.toml` 或 uv lock 文件，可在代码目录创建环境：

```bash
cd "/gpfs/jiuquyun/projects/$USER/<仓库目录>"
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 有 lock 文件时优先使用
uv sync --frozen
```

团队协作时应提交依赖声明和 lock 文件，而不是复制个人环境。不要使用 `sudo pip install`，也不要修改系统 Python。

## 缓存与临时文件

代理和缓存配置会在登录、`sbatch`、`srun` 以及 Pyxis 容器任务中自动生效，不需要在脚本里手动 source。Hugging Face、Torch、Triton、CUDA、uv、pip、Conda 包、ModelScope、Weights \& Biases、npm 和临时文件等可重建内容会自动写入当前计算节点的 `/local/cache/users/$USER`。

```bash
echo "$XDG_CACHE_HOME"
echo "$HF_HOME"
echo "$TORCH_HOME"
echo "$TRITON_CACHE_DIR"
echo "$CONDA_PKGS_DIRS"
echo "$TMPDIR"
```

`/local` 是计算节点本地 RAID0，适合缓存和临时文件，不适合保存唯一的 checkpoint 或重要结果。代码、Conda 环境、checkpoint 和最终输出应放在共享存储或 KS3。

## 在作业中激活项目环境

Conda 项目环境：

```bash
#!/bin/bash -l
#SBATCH ...

conda activate myproject
python train.py
```

uv 项目环境：

```bash
#!/bin/bash -l
#SBATCH ...

source "/gpfs/jiuquyun/projects/$USER/<仓库目录>/.venv/bin/activate"
python train.py
```

提交脚本使用 `#!/bin/bash -l` 时，Conda 命令和基础环境会自动准备好。

## 使用容器

需要更强环境隔离时，可以通过 Pyxis 直接在 Slurm 作业中使用公共镜像。先做最小检查：

```bash
srun --qos=debug \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --time=00:20:00 \
  --container-image=/gpfs/jiuquyun/containers/images/ubuntu-24.04.sqsh \
  bash -lc 'nvidia-smi -L; echo "$XDG_CACHE_HOME"'
```

应只看到分配给你的 GPU，且缓存路径为 `/local/cache/users/$USER`。项目容器应固定镜像版本，并在提交脚本中明确代码、数据和输出目录的挂载方式。

# QOS 与排队规则

提交任务时必须选择一个 QOS。QOS 同时规定 GPU 数量和最长运行时间，超出范围的任务会在提交时被拒绝。

|QOS|GPU 数量|最长时间|适用场景|
|---|---|---|---|
|`debug`|1–8 张|2 小时|交互调试、环境检查、短测试、单机功能验证|
|`normal`|8–16 张|1 天|常规单机或双机训练|
|`long`|16–64 张|7 天|已经稳定、确实需要长时间或多机资源的正式训练|

**选择原则：**能用 debug 完成的验证不要提交 normal；没有先通过小规模验证的任务不要直接提交 long。申请的时间和 GPU 数量应接近真实需要，较小、较准确的请求更容易被回填调度。

调度器会综合资源是否空闲、排队时间、QOS 优先级和用户历史资源使用量安排任务。同一用户长期大量占用资源后，其后续任务可能获得较低的公平份额；这是正常现象。

# 提交批处理任务

## 从系统模板开始

每个用户的家目录中提供三份模板：

```bash
ls -l ~/slurm-examples/
cp ~/slurm-examples/sbatch-debug.sh ./run-debug.sh
cp ~/slurm-examples/sbatch-normal.sh ./run-normal.sh
cp ~/slurm-examples/sbatch-long.sh ./run-long.sh
```

复制到项目目录后再修改，保留原模板用于对照。

## 一个完整的单卡调试脚本

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my-debug
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --chdir=/gpfs/jiuquyun/projects/<用户名>/<仓库目录>
#SBATCH --output=/gpfs/jiuquyun/home/<用户名>/logs/%x-%j.out

set -euo pipefail


source .venv/bin/activate

echo "job_id=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "started_at=$(date -Iseconds)"

srun python train.py <项目参数>
```

将脚本中的 `<用户名>`、`<仓库目录>` 和训练参数替换为真实值，并确保日志目录存在：

```bash
mkdir -p "/gpfs/jiuquyun/home/$USER/logs"
sbatch --test-only run-debug.sh
sbatch run-debug.sh
```

`sbatch` 成功后会返回 `Submitted batch job <JobID>`。请保存 Job ID，它是排队、日志和排障的统一标识。

## 单机 8 卡常规训练

已经完成单卡或小规模验证后，可将关键资源项改为：

```bash
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1T
#SBATCH --time=1-00:00:00
```

PyTorch DDP 项目通常使用：

```bash
srun torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  train.py <项目参数>
```

如果项目已有自己的 launcher，请遵循项目说明，但资源申请仍然必须写在 Slurm 脚本中。

## 查看、跟踪和取消任务

```bash
# 查看自己的任务
squeue -u "$USER"

# 查看指定任务的完整信息和排队原因
scontrol show job <JobID>

# 持续查看日志
tail -f "/gpfs/jiuquyun/home/$USER/logs/<日志文件>"

# 查看已结束任务的状态与资源信息
sacct -j <JobID> --format=JobID,JobName,QOS,State,Elapsed,AllocTRES,ExitCode

# 取消不再需要的任务
scancel <JobID>
```

取消后再次运行 `squeue -j <JobID>` 或 `sacct -j <JobID>`，确认任务已经退出。不要让已知无效或卡死的任务继续占用 GPU。

# 查看 GPU 历史监控

集群已经提供只读 Grafana 监控，可以查看集群、节点、单张 GPU 和 Slurm Job 的利用率、显存、温度、功耗、Tensor/DRAM 活跃度与历史曲线。监控不会影响任务调度或训练。

在自己的电脑上保持下面的 SSH 隧道运行：

```bash
ssh -N -L 3000:127.0.0.1:3000 -L 9091:127.0.0.1:9091 h100
```

然后在浏览器打开 [H100 Cluster Overview](http://127.0.0.1:3000/d/h100-gpu-overview/h100-cluster-overview)。普通用户可匿名只读查看；在 dashboard 中选择节点、GPU 或填写 Slurm Job ID，即可定位自己的训练资源和历史曲线。原始 Prometheus 查询界面为 [http://127\.0\.0\.1:9091](http://127.0.0.1:9091)。

这里的 127\.0\.0\.1 指你的个人电脑。若 SSH 隧道断开，重新运行上面的命令即可；不需要修改服务器上的监控服务。

# 交互调试

交互任务适合快速复现错误、检查张量形状、验证数据路径和测试依赖。不要把交互终端当作长期训练方式。

```bash
srun --qos=debug \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=02:00:00 \
  --pty bash -l
```

进入计算节点后：

```bash

cd "/gpfs/jiuquyun/projects/$USER/<仓库目录>"
source .venv/bin/activate
nvidia-smi
python <调试脚本>
```

终端断开会影响交互任务。需要持续运行、保留日志或排队等待的工作应改用 `sbatch`。

# 多机训练

## 先验证资源，再启动项目

第一次使用两台节点时，先提交一个只检查节点和 GPU 的 normal 任务：

```bash
#!/usr/bin/env bash
#SBATCH --job-name=check-2nodes
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1T
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/jiuquyun/home/<用户名>/logs/%x-%j.out

set -euo pipefail


srun --label bash -lc 'hostname; nvidia-smi -L'
```

确认日志中出现两个不同节点，并且每个节点看到 8 张 GPU 后，再替换为真实训练命令。

## 通用 torchrun 模板

对于“每台节点启动一个 launcher、每台使用 8 个进程”的 PyTorch DDP 项目，可以从下面的作业主体开始：

```bash

source "/gpfs/jiuquyun/projects/$USER/<仓库目录>/.venv/bin/activate"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

srun bash -lc '
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=8 \
    --node-rank="$SLURM_NODEID" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    train.py <项目参数>
'
```

这是一份通用起点，不会替代项目自身的分布式启动要求。DeepSpeed、Megatron\-LM、Ray 或其他框架应优先采用项目已验证的 Slurm 启动方式。

## 逐步扩大规模

1. 1 张 GPU 完成功能和数据链路验证。

2. 8 张 GPU 验证单机 DDP、显存、吞吐和 checkpoint。

3. 16 张 GPU 验证双机通信、断点续训和日志汇总。

4. 再按需要扩大到 32 或 64 张 GPU，并使用 long QOS。

不要在没有小规模成功记录的情况下直接申请 32–64 张 GPU。若多机训练出现卡住或异常慢，先保留 Job ID、完整日志、节点列表和启动命令，再进行排查。

集群网络参数已经统一配置。除非项目明确要求且已经验证，不要随意写死 `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA` 或禁用 IB。临时排障变量在问题定位后应移除。

# 数据传输与 KS3

## 从个人电脑传文件

小文件可以直接使用 scp：

```bash
scp <本地文件> \
  h100:/gpfs/jiuquyun/home/<用户名>/
```

目录或需要断点续传的内容建议使用 rsync：

```bash
rsync -avP -e ssh \
  <本地目录>/ \
  h100:/gpfs/jiuquyun/projects/<用户名>/<目标目录>/
```

大量数据优先从对象存储直接传到 GPFS，不要先经过个人电脑中转。

## 配置 KS3Util

集群已安装 KS3Util。向数据负责人申请最小权限的 Access Key、Secret Key、Region、Endpoint 和 Bucket 后，在自己的账号下执行：

```bash
ks3util config
chmod 600 ~/.ks3utilconfig
ks3util ls
```

配置只属于当前用户。禁止共享密钥、将配置文件提交到 Git，或把密钥写入 Slurm 脚本和日志。

## 上传和下载

```bash
# 下载单个对象到共享存储
ks3util cp ks3://<bucket>/<对象路径> \
  /gpfs/jiuquyun/projects/$USER/<目标文件>

# 下载目录
ks3util cp ks3://<bucket>/<目录>/ \
  /gpfs/jiuquyun/projects/$USER/<目录> -r

# 上传 checkpoint 目录
ks3util cp /gpfs/jiuquyun/checkpoints/$USER/<实验目录> \
  ks3://<bucket>/checkpoints/$USER/<实验目录>/ -r
```

首次大规模传输前，先用小文件验证路径和权限，再用数 GB 的非敏感文件测量实际吞吐。KS3 下行带宽约为 200MB/s，不应作为训练数据的实时读取层。

# VS Code 使用建议

可以使用 VS Code Remote SSH 连接前面配置的 `h100` 别名，并打开共享项目目录。VS Code 主要用于编辑代码、Git 操作、查看日志和提交作业。

- 不要在 VS Code 的登录节点终端直接启动训练。

- Python 解释器选择项目共享目录中的 `.venv/bin/python`。

- GPU 调试在 VS Code 终端中使用 `srun` 申请交互资源。

- 长任务使用 `sbatch`，不要依赖 VS Code 窗口持续在线。

- 不要让语言服务器扫描数据集、checkpoint 或大日志目录；在项目设置中排除这些路径。

# 常见问题

|现象|处理方法|
|---|---|
|无法 SSH 到计算节点|没有活跃作业时这是正常现象。使用 `srun` 申请资源；调试已有作业可用 `srun --jobid=<JobID> --overlap --pty bash -l`。|
|任务一直是 PD|运行 `squeue -j <JobID> -o "%.18i %.9P %.8j %.2t %.10M %.6D %R"` 查看原因。`Resources` 表示资源不足，`Priority` 表示等待调度优先级，均不代表任务出错。|
|提交时提示 QOS 或 TRES 不合法|检查 GPU 数量和时间是否落在所选 QOS 的范围内：debug 1–8/2h，normal 8–16/1d，long 16–64/7d。|
|日志文件无法创建|提前创建日志目录，并确认 `--output` 与 `--chdir` 使用自己的共享目录且有写权限。|
|作业里 Hugging Face 或 GitHub 无法访问|集群会在登录、`sbatch`、`srun` 和 Pyxis 任务中自动加载代理，无需手动 source。重新登录后重试；如仍失败，保留 Job ID 和第一条错误信息联系管理员，不要自行重启代理服务。|
|缓存仍写入共享盘|在作业中检查 `XDG_CACHE_HOME`、`HF_HOME`、`TORCH_HOME`、`TRITON_CACHE_DIR` 和 `CONDA_PKGS_DIRS`，正常应位于 `/local/cache/users/$USER`。新会话会自动加载配置，无需手动 source；路径仍不正确时请保留 Job ID 联系管理员。|
|多机训练卡住|先确认每个节点进程数、`MASTER_ADDR`、`MASTER_PORT` 和 node rank 正确；保存完整日志，不要先写死或禁用网络接口。|
|共享目录不可访问|立即停止写入，不要把重要结果临时改到 `/local` 后继续正式训练。记录时间、节点和报错并联系管理员。|
|任务异常结束但原因不清楚|运行 `sacct -j <JobID> --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocTRES`，同时检查作业日志中的第一条错误。|

# 共同使用规范

- 每个人使用自己的账号、SSH Key、Git 凭据和 KS3 凭据，不共享账号。

- 所有 GPU 工作通过 Slurm 提交；不用的任务及时取消。

- 申请真实需要的 GPU、CPU、内存和运行时间，不超量申请资源。

- 先 debug、再 normal、最后 long；先小规模稳定，再扩大到多机。

- 代码、配置、依赖 lock 文件和关键日志应可复现；实验输出目录应包含清晰的项目名、日期或 run ID。

- 共享数据集默认只读使用，不随意改名、覆盖或删除。

- 重要 checkpoint 和结果保存在 GPFS 或 KS3，不把节点本地盘当作唯一存储。

- 发现节点、存储、网络或调度异常时，保留证据并报告，不自行修改系统服务。

# 获得帮助时请提供的信息

为了快速定位问题，请一次性提供：

- 用户名与 Job ID；

- 提交脚本和实际启动命令；

- 所用 QOS、GPU 数量、节点数量；

- `scontrol show job <JobID>` 或 `sacct` 的关键输出；

- 日志路径、第一条错误及错误前后约 50 行；

- 发生时间，以及问题是否可以稳定复现。

**最短使用路径：**SSH 登录 → 在 GPFS 准备代码和 uv 环境 → 用 debug 验证 → 用 `sbatch` 提交 normal/long → 用 Job ID 跟踪 → 结果保存在 GPFS 或归档到 KS3。

# 相关资料

- [Slurm sbatch 官方文档](https://slurm.schedmd.com/sbatch.html)

- [Slurm srun 官方文档](https://slurm.schedmd.com/srun.html)

- [Slurm squeue 官方文档](https://slurm.schedmd.com/squeue.html)

- [NVIDIA Pyxis](https://github.com/NVIDIA/pyxis)

- [uv 官方文档](https://docs.astral.sh/uv/)

- [KS3Util 快速使用](https://docs.ksyun.com/documents/43947)
