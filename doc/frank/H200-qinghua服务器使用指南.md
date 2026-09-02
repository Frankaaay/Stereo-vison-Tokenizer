# H200-qinghua 服务器使用指南

> 适用账号：`Frank`/`frank`。本文中的个人开发位置、目录、数据和资产约定不作为其他用户的默认值。

本文面向 Stereo-vison-Tokenizer 项目组。服务器地址、账号、分支、资源占用和资产状态均可能变化；SSH 配置和实时检查优先于本文。本文不构成远端写入、安装、训练、评估、预处理、删除或占用 GPU 的授权。

## 1. 连接与节点

本机 `~/.ssh/config` 是 HostName、端口、用户名、密钥和 ProxyJump 的唯一来源。组员使用各自账号，不在仓库中固定登录用户。

```powershell
ssh.exe jump-h200-qinghua
ssh.exe h200-1
ssh.exe h200-2
```

- `h200-1`：8×H200；双机任务通常对应 `NODE_RANK=0`。
- `h200-2`：8×H200；双机任务通常对应 `NODE_RANK=1`。
- 两个计算节点通过 `jump-h200-qinghua` 访问。
- 代码中的逻辑节点 ID `h200-qinghua-1/2` 与 SSH alias `h200-1/2` 不是同一命名空间。

首次连接或每次重要操作前检查：

```bash
whoami
hostname
pwd
nvidia-smi
df -h /data
```

## 2. 存储拓扑

两台 H200 的 `/data` 和 `/data/shared` 都是 node-local 视图。同名路径不代表内容相同；数据、manifest、runtime、checkpoint 和输出必须在目标节点分别核对。

```text
/data/home/<username>/     用户代码、runtime 和实验目录
/data/shared/              当前节点的共享数据视图
/data/cache/               当前节点缓存
/data/tmp/                 临时文件
```

跳板机通过两份独立 NFS 挂载访问两台 H200：

```text
/data-214-30-239-40  -> H200-1 /data
/data-214-30-239-42  -> H200-2 /data
```

NFS 路径可以用于 Git fetch 和文件传输，但不能据此假定两台节点内容一致。`/data/shared/datasets` 只放数据；代码、环境、checkpoint、日志和实验输出放到当前执行用户自己的 `/data/home/<username>` 路径。

## 3. 项目仓库与 Git 同步

每位组员使用自己的独立 clone：

```text
/data/home/<username>/projects/Stereo-vison-Tokenizer
```

项目 origin 应为：

```text
https://github.com/Frankaaay/Stereo-vison-Tokenizer.git
```

目标分支和精确 SHA 由当前任务指定，不在指南中固定为某个用户分支。源码只在本地目标 worktree 修改、验证并按授权 commit/push；H200 clone 不现场编辑、commit 或 push。

同步前在目标节点检查：

```bash
git -C /data/home/<username>/projects/Stereo-vison-Tokenizer status --short --branch
git -C /data/home/<username>/projects/Stereo-vison-Tokenizer branch --show-current
git -C /data/home/<username>/projects/Stereo-vison-Tokenizer rev-parse HEAD
git -C /data/home/<username>/projects/Stereo-vison-Tokenizer remote get-url origin
```

H200 不能直接访问 GitHub。通过跳板机分别更新两份 clone 的 remote ref：

```bash
git -C /data-214-30-239-40/home/<username>/projects/Stereo-vison-Tokenizer fetch --prune origin refs/heads/<target-branch>:refs/remotes/origin/<target-branch>
git -C /data-214-30-239-42/home/<username>/projects/Stereo-vison-Tokenizer fetch --prune origin refs/heads/<target-branch>:refs/remotes/origin/<target-branch>
```

跳板机只更新 remote ref，不执行 merge、checkout 或修改 working tree。随后在需要使用的 H200 节点执行：

```bash
git -C /data/home/<username>/projects/Stereo-vison-Tokenizer merge --ff-only origin/<target-branch>
```

任一节点 dirty、分支/upstream/origin 不符、fetch 失败、不能 fast-forward 或最终 SHA 不一致时立即停止；不得 rebase、reset、force checkout、删除重建或新建替代 branch/worktree。用户指定精确 SHA 时，最终以 `rev-parse HEAD` 的实时结果为准。

## 4. Python 环境与依赖

项目优先复用经过验证的冻结 runtime，不在 H200 上临时改变依赖合同。最近验证过的项目 runtime：

|节点|runtime|
|---|---|
|H200-1|`/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260828/venv`|
|H200-2|`/data/home/frank/runtime/stereo-tokenizer-unified-v1`|

这些路径属于当前已验证资产，不设置登录用户；组员仍按本机 SSH 配置使用各自账号。使用前检查 Python、依赖、权限和目标代码 SHA；需要新环境或安装依赖时必须取得当前任务授权。

系统内网源由管理员配置，具体地址和端口不写入仓库：

```bash
python3 -m pip config list
conda config --show-sources
echo "$UV_INDEX_URL"
```

只有内网源缺少已授权依赖时，才在跳板机为目标节点准备 Python wheelhouse：

```bash
# 使用第 2 节中目标节点对应的 NFS 挂载
OFF=<node-nfs-mount>/shared/offline/wheels/<project-or-env>
mkdir -p "$OFF"
python3 -m pip download -r requirements.txt -d "$OFF"
cp requirements.txt "$OFF/"
sha256sum "$OFF"/* > "$OFF/SHA256SUMS"
```

在对应 H200 上核对哈希后离线安装：

```bash
python -m pip install --no-index --find-links /data/shared/offline/wheels/<project-or-env> -r requirements.txt
```

CUDA、PyTorch、xFormers 等二进制 wheel 必须匹配 Python、CUDA 和 Linux x86_64 合同。不得临时添加公网源、代理、VPN 或自定义 `HF_ENDPOINT`。

## 5. 当前全量训练数据合同

以下是最近验证的全量三数据源合同。使用时仍须读取目标运行的 `resolved_config.json` 和 `run_manifest.json`，并实时验证文件、哈希与 root alias。

### H200-1：最近正式训练合同

|来源|Manifest|SHA256|
|---|---|---|
|Hy|`/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260829/manifests/hy_formal_90_5_5_v1.jsonl`|`b25efc945ccd7e7afd2f1a76393ea19adde8fa072e1e9a2ca6348e0e5c1a45f9`|
|LIBERO|`/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260829/manifests/libero_formal_90_5_5_v1.jsonl`|`0299354a7225e979f6b9ff5fb3e26a975c811d2d41af44e042a6eade3f24bbf4`|
|UMI LeRobot|`/data/home/frank/runtime/umi-lerobot-decode-audit-h2001-20260829-v1/umi_lerobot_decode_verified_v1.jsonl`|`5e8f58c769549372af070a6132ad826bd7172aaeabcebebff84426e66bc2120f`|

H200-1 对应数据根：

```text
Hy primary: /data/shared/hy_embodied/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data
Hy rest:    /data/shared/hy_embodied_rest/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data
LIBERO:     /data/shared/offline/datasets/libero_mujoco3.3.2
UMI:        /data/shared/datasets/umi_lerobot_v3_260714
```

### H200-2：最近验证的全量合同

|来源|Manifest|SHA256|
|---|---|---|
|Hy|`/data/home/frank/runtime/stereo-tokenizer-pretrain-h2002-20260829-hy-odd-v1/manifests/hy_formal_90_5_5_v1.jsonl`|`055699f4b1159a6ff55e77cc1379e052fb6292cd0525c82e61cb178198d56c86`|
|LIBERO|`/data/home/frank/runtime/stereo-tokenizer-pretrain-h2002-20260829/manifests/libero_formal_90_5_5_v1.jsonl`|`0299354a7225e979f6b9ff5fb3e26a975c811d2d41af44e042a6eade3f24bbf4`|
|UMI LeRobot|`/data/home/frank/runtime/umi-lerobot-decode-audit-h2002-20260829-v1/umi_lerobot_decode_verified_v1.jsonl`|`96024f091bcf7aca844b4d4b99fad2eb6cb0f420aa693f1431340b79ac5fa53e`|

H200-2 的 Hy 表清单和 UMI 内容是 node-local 合同，不得复用 H200-1 manifest。数据根和 root aliases 必须从目标节点的 manifest 与 resolved config 实时核对。

### 在线 teacher

最近 H200-1 正式 resolved config 使用在线 teacher，并关闭在线 GT cache：

```text
online_gt_enabled:       1
online_gt_cache_enabled: 0

DA3 repo:
  /data/home/frank/runtime/depth-anything-3/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
DA3 checkpoint:
  /data/home/frank/artifacts/depth-anything-3/DA3-BASE/f4a6c9b3c95e41c82048423d3493a81ec3fa810e

LAS2-H repo:
  /data/home/hezhou/projects/LiteAnyStereo
LAS2-H checkpoint:
  /data/home/hezhou/artifacts/lite-any-stereo/checkpoints/LAS2_H.pth
```

DA3 与 LAS2-H 的 source/checkpoint SHA、运行参数和可见性必须从目标 resolved config 与实时文件核对。跨用户 LAS2-H 路径只作为已授权只读输入，不向其中写入代码、缓存或日志。

## 6. 文件传输

本机与 H200：

```powershell
scp.exe local-file h200-1:/data/tmp/
scp.exe h200-1:/data/tmp/remote-file .
```

两台 H200 之间传输前先确认源、目标和剩余空间。数据量较大时，从跳板机通过两份 NFS 挂载执行 `rsync`，且不得使用 `--delete`：

```bash
rsync -av --info=progress2 /data-214-30-239-40/<明确源路径>/ /data-214-30-239-42/<明确目标路径>/
```

远端写操作结果不确定时，先只读核对目标，不盲目重复传输。

## 7. 训练与双机检查

正式入口使用项目 launcher：

```bash
bash scripts/stereo/train_stereo_vae.sh
```

启动前必须以实际环境变量、launcher 输出、`resolved_config.json` 和 `run_manifest.json` 确认完整合同。最近 H200-1 正式合同是单机 8 GPU、每卡 BS24、GA1；这不是未来任务的默认授权或永久配置。

双机任务开始前在两台节点分别检查：

```bash
ibstat
ibdev2netdev
rdma link show
nvidia-smi topo -m
```

`MASTER_ADDR`、`MASTER_PORT`、`NODE_RANK`、NCCL 网卡和 HCA 必须来自当前已验证 launcher 或本次实时检查，不在指南中硬编码。两端必须使用相同代码 SHA、run ID、resolved config 和端口，并使用各自正确的 node-local manifest。

## 8. 启动边界与故障处理

- 未经用户明确授权，不启动训练、评估、预处理、长时间 GPU 任务或正式 resume。
- 启动前核对当前用户、主机、精确 Git SHA、clean worktree、Python/依赖、manifest、teacher、GPU 所有者和 output 路径。
- 发现 GPU 已占用时报告进程所有者和命令行，不自动 kill 或清理。
- 当前事实优先级：实时 Git/进程/日志、resolved config、run manifest 和 data contract > launcher > 历史文档。
- SSH 255、跳板断连或返回缺失时，先只读检查操作是否落地；只有确认未落地且重试幂等时才能继续。

|现象|处理|
|---|---|
|H200 无法访问公网|使用已配置内网源，或经授权从跳板机准备离线 Python 包。|
|两节点同名数据不一致|这是 node-local 存储的预期风险；分别检查路径、记录数和哈希。|
|双机初始化卡住|核对两端进程、node rank、地址/端口、IB 状态和完整首个 traceback，不先改 NCCL 参数。|
|目录无写权限|停止写入，确认当前账号和明确目标目录，不改写到其他用户路径。|
|训练异常退出|检查第一条根因 traceback、exit code、最后稳定 step、checkpoint 可读性和 GPU 是否释放。|
