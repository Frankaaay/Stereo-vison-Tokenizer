# S48 同合同补训

用户授权补训 S48-correct，并允许保留本地 `doc/Stereo Tokenizer Plan.md` 的未提交修改。
本次只提交 S48 启动脚本和本记录。模型、数据及训练代码保持 M48/D48 使用的实现。

H200-1 的 M48/D48 v3 均已有 exit_code=0，8 张 GPU 空闲，/data 剩余 5.2T。
S48 从零训练 40,000 generator updates，latent 48、8 GPU BF16、seed 1234，
per-device BS 192:40:160:36、GA1。沿用原 Hy/LIBERO/UMI manifests、teacher、loss、
LR、调度、2k validation 和 5k checkpoint，学生 UMI 输入使用正确 (L,R)。

启动入口：`doc/frank/h2001-s48-ablation-20260907.sh`。
输出：`/data/home/frank/experiments/stereo-input-ablation-s48-h2001-20260907-v1`。
tmux：`s48-ablation-h2001-20260907-v1`。
启动前进行 shell 语法检查，启动后比较 resolved config 并确认真实 update。
