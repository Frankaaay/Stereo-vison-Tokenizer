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

## 启动与健康检查

- 启动代码 SHA：`b248dfe4f060bf29494762ace15f61ca167cfc65`；本机 push 后 H200-1
  fast-forward 到同一 SHA，远端工作区 clean。训练实现相对 M48/D48 无 diff。
- 2026-09-07 17:07:29 CST 启动，W&B offline run `ubyy2onz`。
- M48/D48 的 40k checkpoint 均成功读取，generator_updates=40000，
  mode_updates=14000/14000/6000/6000，mode_samples 完全相同，exit_code=0。
- S48 与 M48 的 resolved config 逐字段比较仅有学生输入合同和输出路径不同。
  三份 manifest 实时 SHA256 与原运行一致，两个 teacher source SHA 相同且 clean。
- 17:11:29 CST 健康检查已到 7/40000；8 卡显存约 132859 MiB（129.7 GiB），
  GPU 利用率 85–100%，日志无 traceback、OOM、CUDA error、non-finite。
  未到 5k checkpoint 门槛，尚未独立解析 loss；不能据此宣称训练已收敛。
- 冷启动后 update 3–7 约 3 秒/update，仅作粗略预算，40k 主体约 34 小时，
  另加周期验证和 checkpoint 时间；无独立后处理任务。
