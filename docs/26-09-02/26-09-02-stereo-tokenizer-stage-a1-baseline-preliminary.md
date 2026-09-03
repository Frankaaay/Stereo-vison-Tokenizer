# Stereo Tokenizer Stage A1 Baseline（Preliminary）

> 状态：PRELIMINARY。HY 因 canonical Lance index/loader 合同冲突被阻断；rFID 与 RAFT 指标留待 A2。

## 实验合同与 provenance

- Artifact 根目录：`/gpfs/jiuquyun/projects/hezhou/experiments/stereo-tokenizer-stage-a/20260902-stagec-update162500-baseline-a1-v4`
- 实际 cwd：`/gpfs/jiuquyun/projects/hezhou/Workspace/Stereo-vison-Tokenizer`
- Git branch / commit：`hezhou-las2-h` / `bf5dfb8905e987717ca7c4d3142f590b15b31506`
- 未提交代码 diff：`/gpfs/jiuquyun/projects/hezhou/experiments/stereo-tokenizer-stage-a/20260902-stagec-update162500-baseline-a1-v4/source.patch`；SHA256：`24f30c5090c9d4bf17e587cae896ad69a3678710ab6d93ff239176cc64a24398`
- `git status --porcelain`：`M evaluation/stage_a_metrics.py;  M evaluation/tokenizer_stage_a.py;  M tests/stereo/test_stage_a_evaluation.py`
- Checkpoint：`/gpfs/jiuquyun/projects/Frank/stereo-tokenizer-checkpoints/v1/stagec-update162500/last.ckpt`
- Checkpoint SHA256：`a74c3b72b32dfd296157e3b6ad24d0521731517e79e75f22786bca37c47d822e`；global_step=125000；epoch=0
- 直接训练计数：`{"batch_updates": 162500, "discriminator_updates": 118500, "four_frame_updates": 81250, "generator_updates": 162500, "grad_accumulates": 1, "logical_update_contract_version": 1, "mode_batch_sizes": {"mono/four_frame": 24, "mono/single_frame": 24, "stereo/four_frame": 24, "stereo/single_frame": 24}, "mode_contract": ["mono/single_frame", "mono/four_frame", "stereo/single_frame", "stereo/four_frame"], "mode_effective_global_batch_sizes": {"mono/four_frame": 192, "mono/single_frame": 192, "stereo/four_frame": 192, "stereo/single_frame": 192}, "mode_grad_accumulates": {"mono/four_frame": 1, "mono/single_frame": 1, "stereo/four_frame": 1, "stereo/single_frame": 1}, "mode_samples": {"mono/four_frame": 10920000, "mono/single_frame": 10920000, "stereo/four_frame": 4680000, "stereo/single_frame": 4680000}, "mode_schedule_seed": 1234, "mode_update_weights": {"mono/four_frame": 7, "mono/single_frame": 7, "stereo/four_frame": 3, "stereo/single_frame": 3}, "mode_updates": {"mono/four_frame": 56875, "mono/single_frame": 56875, "stereo/four_frame": 24375, "stereo/single_frame": 24375}, "mono_dataset_weights": "9:1", "node_manifest_contracts": "{\"0\":{\"hy\":\"b25efc945ccd7e7afd2f1a76393ea19adde8fa072e1e9a2ca6348e0e5c1a45f9\",\"libero\":\"0299354a7225e979f6b9ff5fb3e26a975c811d2d41af44e042a6eade3f24bbf4\",\"umi\":\"5e8f58c769549372af070a6132ad826bd7172aaeabcebebff84426e66bc2120f\"}}", "per_device_batch_size": 24, "single_frame_updates": 81250, "world_size_contract": 8}`
- 质量：FP32；效率：BF16；posterior mean；Tokenizer `eval + inference_mode` 且运行时冻结。
- Python：`3.12.11`；GPU：`NVIDIA H100 80GB HBM3`；CUDA：`12.6`；cuDNN：`90501`
- `uv.lock` SHA256：`7542a5fcdfb3656c938302c6096d741dcd91dedab8d63c99d867d98e0b9a27ad`
- LPIPS VGG16：`/gpfs/jiuquyun/projects/hezhou/experiments/stereo-tokenizer-stage-a/runtime/metric-backbones/torch/hub/checkpoints/vgg16-397923af.pth`；SHA256：`397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`；预处理：`torchmetrics LPIPS vgg normalize=False on RGB [-1,1]`
- 关键包：av=16.0.1, numpy=1.26.2, pylance=10.0.0, pytorch-lightning=2.5.6, torch=2.7.1+cu126, torchmetrics=1.9.0, torchvision=0.22.1+cu126
- Tokenizer 参数：total=85,819,251；架构可训练=71,103,091；运行时 requires_grad=0
- DA3：source `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`；weights `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`
- LAS2-H：source `8c97bd4c4da3712c2ac60003a23201dfdb5935f4`；weights `758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4`

### 数据与哈希

| Dataset | Windows/cell | Decode checked/rejected | Selection semantic SHA256 | Selection file SHA256 | Manifest SHA256 |
| --- | ---: | ---: | --- | --- | --- |
| libero | 256 | 256/0 | `e5cf61eb8b8b27d8c5c00cac83e6b0541cb14e7ca59f088b1a06ece49d452601` | `98ead425208feec40de5b158a4cc86131846a3588dd0fbae595ede478ed4cc17` | `59cea99c07cad93a7e6e922241002a6d3ae8190837941e204a356218c27cf7ae` |
| umi | 1024 | 3249/2225 | `4a412e87244d59883ee62ad012282e09c29838b9dfe07d2109ba8efb5960929b` | `615f35e42d8b9904b8a7d1b308a54d8a861f8a1896f841f922b76cf827e26677` | `e4fdcd1264ce1b8c08612305701dd907569822860aaba8825264600f5cc02fe3` |
| HY | 0（BLOCKED） | N/A | N/A | N/A | 已冻结但不参与本次 macro |

### 覆盖矩阵

| Dataset | Eye | Camera/view cell | Windows | Modes | Macro inclusion |
| --- | --- | --- | ---: | --- | --- |
| libero | mono | observation.images.cam_head_left | 256 | single source 0/1/2/3 + four-frame | yes |
| libero | mono | observation.images.cam_left_wrist_left | 256 | single source 0/1/2/3 + four-frame | yes |
| umi | mono | observation.images.cam_head_left | 1024 | single source 0/1/2/3 + four-frame | yes |
| umi | mono | observation.images.cam_head_right | 1024 | single source 0/1/2/3 + four-frame | yes |
| umi | mono | observation.images.cam_left_wrist_left | 1024 | single source 0/1/2/3 + four-frame | yes |
| umi | mono | observation.images.cam_left_wrist_right | 1024 | single source 0/1/2/3 + four-frame | yes |
| umi | mono | observation.images.cam_right_wrist_left | 1024 | single source 0/1/2/3 + four-frame | yes |
| umi | mono | observation.images.cam_right_wrist_right | 1024 | single source 0/1/2/3 + four-frame | yes |
| umi | stereo | 3 canonical stereo pairs | 1024 | single source 0/1/2/3 + four-frame | yes |
| HY | mono | N/A | 0 | BLOCKED | no |

## RGB 重建（per camera/view）

| Dataset | Eye | Camera/view | Mode | L1 mean | P50 | P90 | P99 | MSE | PSNR | SSIM | LPIPS | RGB mask |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| libero | mono | cam_head_left | mono/four_frame | 0.046341 | 0.045431 | 0.053823 | 0.086109 | 0.228618 | 6.412 | 0.881153 | 0.160100 | 1.000000 |
| libero | mono | cam_head_left | mono/single_frame/source_0 | 0.041559 | 0.041656 | 0.047640 | 0.087240 | 0.077965 | 11.106 | 0.899926 | 0.137914 | 1.000000 |
| libero | mono | cam_head_left | mono/single_frame/source_1 | 0.041519 | 0.041221 | 0.047844 | 0.088447 | 0.078232 | 11.083 | 0.900249 | 0.137640 | 1.000000 |
| libero | mono | cam_head_left | mono/single_frame/source_2 | 0.041697 | 0.041484 | 0.048081 | 0.099947 | 0.078400 | 11.072 | 0.899750 | 0.137588 | 1.000000 |
| libero | mono | cam_head_left | mono/single_frame/source_3 | 0.041644 | 0.041304 | 0.047536 | 0.105316 | 0.078203 | 11.084 | 0.900347 | 0.137655 | 1.000000 |
| libero | mono | cam_left_wrist_left | mono/four_frame | 0.043023 | 0.039567 | 0.059749 | 0.098037 | 0.219858 | 6.587 | 0.892767 | 0.153912 | 1.000000 |
| libero | mono | cam_left_wrist_left | mono/single_frame/source_0 | 0.035323 | 0.032181 | 0.049420 | 0.090961 | 0.070060 | 11.633 | 0.911749 | 0.123506 | 1.000000 |
| libero | mono | cam_left_wrist_left | mono/single_frame/source_1 | 0.035032 | 0.031895 | 0.047746 | 0.084597 | 0.070979 | 11.529 | 0.912539 | 0.125065 | 1.000000 |
| libero | mono | cam_left_wrist_left | mono/single_frame/source_2 | 0.036356 | 0.032249 | 0.054887 | 0.091446 | 0.072184 | 11.475 | 0.910993 | 0.124679 | 1.000000 |
| libero | mono | cam_left_wrist_left | mono/single_frame/source_3 | 0.036729 | 0.032161 | 0.055020 | 0.097231 | 0.071328 | 11.576 | 0.910438 | 0.126480 | 1.000000 |
| umi | mono | cam_head_left | mono/four_frame | 0.067440 | 0.066458 | 0.078495 | 0.095786 | 0.307217 | 5.131 | 0.765784 | 0.190542 | 0.750000 |
| umi | mono | cam_head_left | mono/single_frame/source_0 | 0.058583 | 0.057897 | 0.068799 | 0.087174 | 0.105395 | 9.805 | 0.822574 | 0.156025 | 0.750000 |
| umi | mono | cam_head_left | mono/single_frame/source_1 | 0.058625 | 0.057889 | 0.069472 | 0.087228 | 0.104781 | 9.834 | 0.823063 | 0.155810 | 0.750000 |
| umi | mono | cam_head_left | mono/single_frame/source_2 | 0.058651 | 0.057745 | 0.069209 | 0.083673 | 0.104960 | 9.813 | 0.822372 | 0.156013 | 0.750000 |
| umi | mono | cam_head_left | mono/single_frame/source_3 | 0.058613 | 0.057453 | 0.069299 | 0.085388 | 0.105219 | 9.815 | 0.823050 | 0.155474 | 0.750000 |
| umi | mono | cam_head_right | mono/four_frame | 0.067986 | 0.067186 | 0.079197 | 0.095440 | 0.308201 | 5.119 | 0.762892 | 0.192106 | 0.750000 |
| umi | mono | cam_head_right | mono/single_frame/source_0 | 0.059211 | 0.058623 | 0.070246 | 0.083476 | 0.106021 | 9.770 | 0.819827 | 0.157577 | 0.750000 |
| umi | mono | cam_head_right | mono/single_frame/source_1 | 0.059030 | 0.058226 | 0.069731 | 0.084733 | 0.105495 | 9.795 | 0.820498 | 0.157052 | 0.750000 |
| umi | mono | cam_head_right | mono/single_frame/source_2 | 0.059067 | 0.058236 | 0.069386 | 0.084930 | 0.105469 | 9.797 | 0.820165 | 0.157573 | 0.750000 |
| umi | mono | cam_head_right | mono/single_frame/source_3 | 0.058993 | 0.058200 | 0.069199 | 0.083972 | 0.105759 | 9.779 | 0.820849 | 0.157231 | 0.750000 |
| umi | mono | cam_left_wrist_left | mono/four_frame | 0.065739 | 0.063311 | 0.080243 | 0.111719 | 0.304011 | 5.180 | 0.786101 | 0.185762 | 0.750000 |
| umi | mono | cam_left_wrist_left | mono/single_frame/source_0 | 0.056620 | 0.054104 | 0.069768 | 0.100491 | 0.102606 | 9.929 | 0.840367 | 0.149137 | 0.750000 |
| umi | mono | cam_left_wrist_left | mono/single_frame/source_1 | 0.056456 | 0.054022 | 0.070238 | 0.109512 | 0.103086 | 9.899 | 0.840520 | 0.149126 | 0.750000 |
| umi | mono | cam_left_wrist_left | mono/single_frame/source_2 | 0.056641 | 0.054218 | 0.071236 | 0.100780 | 0.103522 | 9.875 | 0.840256 | 0.149092 | 0.750000 |
| umi | mono | cam_left_wrist_left | mono/single_frame/source_3 | 0.056562 | 0.054456 | 0.071410 | 0.098853 | 0.102676 | 9.919 | 0.840523 | 0.149187 | 0.750000 |
| umi | mono | cam_left_wrist_right | mono/four_frame | 0.067273 | 0.065377 | 0.081927 | 0.115202 | 0.304009 | 5.180 | 0.781776 | 0.190647 | 0.750000 |
| umi | mono | cam_left_wrist_right | mono/single_frame/source_0 | 0.057824 | 0.055412 | 0.072148 | 0.108251 | 0.103323 | 9.886 | 0.837502 | 0.152738 | 0.750000 |
| umi | mono | cam_left_wrist_right | mono/single_frame/source_1 | 0.057716 | 0.055241 | 0.070523 | 0.106110 | 0.103624 | 9.890 | 0.837793 | 0.152479 | 0.750000 |
| umi | mono | cam_left_wrist_right | mono/single_frame/source_2 | 0.057713 | 0.055570 | 0.071801 | 0.106403 | 0.103230 | 9.898 | 0.837578 | 0.152684 | 0.750000 |
| umi | mono | cam_left_wrist_right | mono/single_frame/source_3 | 0.057661 | 0.055467 | 0.071703 | 0.103314 | 0.103298 | 9.885 | 0.838073 | 0.152368 | 0.750000 |
| umi | mono | cam_right_wrist_left | mono/four_frame | 0.065491 | 0.064191 | 0.079456 | 0.099932 | 0.301323 | 5.220 | 0.786018 | 0.190594 | 0.750000 |
| umi | mono | cam_right_wrist_left | mono/single_frame/source_0 | 0.056425 | 0.054448 | 0.071447 | 0.093633 | 0.100282 | 10.039 | 0.843763 | 0.148839 | 0.750000 |
| umi | mono | cam_right_wrist_left | mono/single_frame/source_1 | 0.055732 | 0.054028 | 0.069509 | 0.095354 | 0.100469 | 10.022 | 0.844622 | 0.149151 | 0.750000 |
| umi | mono | cam_right_wrist_left | mono/single_frame/source_2 | 0.056074 | 0.054466 | 0.071061 | 0.091678 | 0.100430 | 10.029 | 0.844214 | 0.149087 | 0.750000 |
| umi | mono | cam_right_wrist_left | mono/single_frame/source_3 | 0.055921 | 0.054698 | 0.070311 | 0.090084 | 0.100886 | 9.999 | 0.844522 | 0.148782 | 0.750000 |
| umi | mono | cam_right_wrist_right | mono/four_frame | 0.065313 | 0.064337 | 0.080447 | 0.099287 | 0.302123 | 5.205 | 0.788268 | 0.192748 | 0.750000 |
| umi | mono | cam_right_wrist_right | mono/single_frame/source_0 | 0.055979 | 0.054455 | 0.070562 | 0.091837 | 0.100871 | 10.007 | 0.845273 | 0.150586 | 0.750000 |
| umi | mono | cam_right_wrist_right | mono/single_frame/source_1 | 0.056192 | 0.054801 | 0.070606 | 0.090569 | 0.100583 | 10.012 | 0.845124 | 0.150855 | 0.750000 |
| umi | mono | cam_right_wrist_right | mono/single_frame/source_2 | 0.056144 | 0.054739 | 0.070865 | 0.091473 | 0.100497 | 10.039 | 0.844726 | 0.150868 | 0.750000 |
| umi | mono | cam_right_wrist_right | mono/single_frame/source_3 | 0.055784 | 0.054545 | 0.069576 | 0.088603 | 0.100300 | 10.034 | 0.845243 | 0.151176 | 0.750000 |
| umi | stereo | head | stereo/four_frame | 0.063194 | 0.062681 | 0.073655 | 0.089406 | 0.308786 | 5.109 | 0.770630 | 0.185244 | 0.750000 |
| umi | stereo | left_wrist | stereo/four_frame | 0.062016 | 0.060304 | 0.075202 | 0.101220 | 0.305160 | 5.160 | 0.789792 | 0.181396 | 0.750000 |
| umi | stereo | right_wrist | stereo/four_frame | 0.061827 | 0.060965 | 0.075948 | 0.097453 | 0.302415 | 5.203 | 0.789818 | 0.186907 | 0.750000 |
| umi | stereo | head | stereo/single_frame/source_0 | 0.053297 | 0.052688 | 0.063412 | 0.078083 | 0.107884 | 9.692 | 0.829775 | 0.149286 | 0.750000 |
| umi | stereo | left_wrist | stereo/single_frame/source_0 | 0.052242 | 0.050370 | 0.064904 | 0.094561 | 0.103959 | 9.866 | 0.845936 | 0.143990 | 0.750000 |
| umi | stereo | right_wrist | stereo/single_frame/source_0 | 0.052161 | 0.050542 | 0.065978 | 0.088829 | 0.101515 | 9.981 | 0.849478 | 0.144165 | 0.750000 |
| umi | stereo | head | stereo/single_frame/source_1 | 0.053292 | 0.052532 | 0.063438 | 0.077543 | 0.107365 | 9.712 | 0.830271 | 0.149149 | 0.750000 |
| umi | stereo | left_wrist | stereo/single_frame/source_1 | 0.052034 | 0.049751 | 0.065268 | 0.097458 | 0.104182 | 9.855 | 0.846132 | 0.143994 | 0.750000 |
| umi | stereo | right_wrist | stereo/single_frame/source_1 | 0.051741 | 0.050360 | 0.064373 | 0.087883 | 0.101484 | 9.980 | 0.850069 | 0.144515 | 0.750000 |
| umi | stereo | head | stereo/single_frame/source_2 | 0.053434 | 0.052553 | 0.063108 | 0.077797 | 0.107586 | 9.702 | 0.829658 | 0.149212 | 0.750000 |
| umi | stereo | left_wrist | stereo/single_frame/source_2 | 0.052178 | 0.050068 | 0.065081 | 0.093512 | 0.104273 | 9.854 | 0.845760 | 0.143936 | 0.750000 |
| umi | stereo | right_wrist | stereo/single_frame/source_2 | 0.051987 | 0.050527 | 0.065335 | 0.087441 | 0.101573 | 9.984 | 0.849871 | 0.144412 | 0.750000 |
| umi | stereo | head | stereo/single_frame/source_3 | 0.053322 | 0.052281 | 0.062953 | 0.078265 | 0.107725 | 9.701 | 0.830296 | 0.148929 | 0.750000 |
| umi | stereo | left_wrist | stereo/single_frame/source_3 | 0.052145 | 0.050247 | 0.065273 | 0.092478 | 0.104007 | 9.868 | 0.846125 | 0.144097 | 0.750000 |
| umi | stereo | right_wrist | stereo/single_frame/source_3 | 0.051814 | 0.050917 | 0.065145 | 0.085918 | 0.101612 | 9.980 | 0.850060 | 0.144292 | 0.750000 |

### Dataset/eye/mode 等权 macro

| Dataset | Eye | Mode | RGB L1 | MSE | PSNR | SSIM | LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| libero | mono | mono/four_frame | 0.044682 | 0.224238 | 6.499 | 0.886960 | 0.157006 |
| libero | mono | mono/single_frame/source_0 | 0.038441 | 0.074012 | 11.369 | 0.905837 | 0.130710 |
| libero | mono | mono/single_frame/source_1 | 0.038276 | 0.074606 | 11.306 | 0.906394 | 0.131353 |
| libero | mono | mono/single_frame/source_2 | 0.039026 | 0.075292 | 11.274 | 0.905371 | 0.131133 |
| libero | mono | mono/single_frame/source_3 | 0.039187 | 0.074765 | 11.330 | 0.905392 | 0.132067 |
| umi | mono | mono/four_frame | 0.066540 | 0.304481 | 5.172 | 0.778473 | 0.190400 |
| umi | mono | mono/single_frame/source_0 | 0.057440 | 0.103083 | 9.906 | 0.834885 | 0.152484 |
| umi | mono | mono/single_frame/source_1 | 0.057292 | 0.103006 | 9.909 | 0.835270 | 0.152412 |
| umi | mono | mono/single_frame/source_2 | 0.057381 | 0.103018 | 9.908 | 0.834885 | 0.152553 |
| umi | mono | mono/single_frame/source_3 | 0.057256 | 0.103023 | 9.905 | 0.835377 | 0.152370 |
| umi | stereo | stereo/four_frame | 0.062346 | 0.305454 | 5.157 | 0.783413 | 0.184516 |
| umi | stereo | stereo/single_frame/source_0 | 0.052567 | 0.104453 | 9.846 | 0.841730 | 0.145814 |
| umi | stereo | stereo/single_frame/source_1 | 0.052356 | 0.104344 | 9.849 | 0.842158 | 0.145886 |
| umi | stereo | stereo/single_frame/source_2 | 0.052533 | 0.104477 | 9.847 | 0.841763 | 0.145853 |
| umi | stereo | stereo/single_frame/source_3 | 0.052427 | 0.104448 | 9.850 | 0.842161 | 0.145773 |

## Four-frame 时间一致性

| Dataset | Eye | Camera/view | Δ L1 | Δ LPIPS | Δ01 L1/LPIPS | Δ12 L1/LPIPS | Δ23 L1/LPIPS |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| libero | mono | cam_head_left | 0.027584 | 0.148514 | 0.033830/0.148525 | 0.034533/0.160550 | 0.014388/0.136466 |
| libero | mono | cam_left_wrist_left | 0.034988 | 0.163889 | 0.040535/0.162707 | 0.041760/0.173262 | 0.022669/0.155699 |
| umi | mono | cam_head_left | 0.052716 | 0.268539 | 0.060971/0.267288 | 0.061833/0.275348 | 0.035344/0.262980 |
| umi | mono | cam_head_right | 0.052986 | 0.271411 | 0.061536/0.270199 | 0.061882/0.278119 | 0.035542/0.265914 |
| umi | mono | cam_left_wrist_left | 0.052233 | 0.251610 | 0.060500/0.249133 | 0.061696/0.262954 | 0.034503/0.242742 |
| umi | mono | cam_left_wrist_right | 0.053145 | 0.255014 | 0.061532/0.252182 | 0.062125/0.266341 | 0.035776/0.246520 |
| umi | mono | cam_right_wrist_left | 0.056549 | 0.246408 | 0.064567/0.243807 | 0.065345/0.258047 | 0.039733/0.237370 |
| umi | mono | cam_right_wrist_right | 0.055793 | 0.245259 | 0.064190/0.243213 | 0.065008/0.256853 | 0.038180/0.235710 |
| umi | stereo | head | 0.051607 | 0.262783 | 0.059983/0.261573 | 0.060349/0.269955 | 0.034490/0.256821 |
| umi | stereo | left_wrist | 0.051191 | 0.247637 | 0.059571/0.245041 | 0.060514/0.259286 | 0.033487/0.238585 |
| umi | stereo | right_wrist | 0.055169 | 0.243131 | 0.063472/0.240721 | 0.064124/0.255167 | 0.037910/0.233504 |

## Teacher-relative 几何（非真实 GT accuracy）

| Dataset | Eye | Camera/view | Mode | Metric kind | log-L1 | RMSE | SILog | Mask coverage | Valid samples |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| libero | mono | cam_head_left | mono/four_frame | reconstruction_teacher | 0.079414 | 0.111719 | 0.111719 | 1.000000 | 256 |
| libero | mono | cam_head_left | mono/single_frame/source_0 | reconstruction_teacher | 0.106625 | 0.155381 | 0.155381 | 1.000000 | 256 |
| libero | mono | cam_head_left | mono/single_frame/source_1 | reconstruction_teacher | 0.105597 | 0.153507 | 0.153507 | 1.000000 | 256 |
| libero | mono | cam_head_left | mono/single_frame/source_2 | reconstruction_teacher | 0.104421 | 0.152233 | 0.152233 | 1.000000 | 256 |
| libero | mono | cam_head_left | mono/single_frame/source_3 | reconstruction_teacher | 0.105921 | 0.152768 | 0.152768 | 1.000000 | 256 |
| libero | mono | cam_left_wrist_left | mono/four_frame | reconstruction_teacher | 0.079671 | 0.097630 | 0.097630 | 1.000000 | 256 |
| libero | mono | cam_left_wrist_left | mono/single_frame/source_0 | reconstruction_teacher | 0.121015 | 0.152649 | 0.152649 | 1.000000 | 256 |
| libero | mono | cam_left_wrist_left | mono/single_frame/source_1 | reconstruction_teacher | 0.120096 | 0.151428 | 0.151428 | 1.000000 | 256 |
| libero | mono | cam_left_wrist_left | mono/single_frame/source_2 | reconstruction_teacher | 0.122475 | 0.154862 | 0.154862 | 1.000000 | 256 |
| libero | mono | cam_left_wrist_left | mono/single_frame/source_3 | reconstruction_teacher | 0.120660 | 0.151922 | 0.151922 | 1.000000 | 256 |
| umi | mono | cam_head_left | mono/four_frame | reconstruction_teacher | 0.146792 | 0.196607 | 0.196607 | 0.750000 | 1024 |
| umi | mono | cam_head_left | mono/single_frame/source_0 | reconstruction_teacher | 0.160126 | 0.212104 | 0.212104 | 0.750000 | 1024 |
| umi | mono | cam_head_left | mono/single_frame/source_1 | reconstruction_teacher | 0.159950 | 0.212495 | 0.212495 | 0.750000 | 1024 |
| umi | mono | cam_head_left | mono/single_frame/source_2 | reconstruction_teacher | 0.160510 | 0.213057 | 0.213057 | 0.750000 | 1024 |
| umi | mono | cam_head_left | mono/single_frame/source_3 | reconstruction_teacher | 0.160455 | 0.212978 | 0.212978 | 0.750000 | 1024 |
| umi | mono | cam_head_right | mono/four_frame | reconstruction_teacher | 0.150023 | 0.201921 | 0.201921 | 0.750000 | 1024 |
| umi | mono | cam_head_right | mono/single_frame/source_0 | reconstruction_teacher | 0.165705 | 0.219675 | 0.219675 | 0.750000 | 1024 |
| umi | mono | cam_head_right | mono/single_frame/source_1 | reconstruction_teacher | 0.164139 | 0.218116 | 0.218116 | 0.750000 | 1024 |
| umi | mono | cam_head_right | mono/single_frame/source_2 | reconstruction_teacher | 0.163606 | 0.218377 | 0.218377 | 0.750000 | 1024 |
| umi | mono | cam_head_right | mono/single_frame/source_3 | reconstruction_teacher | 0.163170 | 0.217120 | 0.217120 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_left | mono/four_frame | reconstruction_teacher | 0.197353 | 0.253018 | 0.253018 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_left | mono/single_frame/source_0 | reconstruction_teacher | 0.208447 | 0.267644 | 0.267644 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_left | mono/single_frame/source_1 | reconstruction_teacher | 0.210632 | 0.269688 | 0.269688 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_left | mono/single_frame/source_2 | reconstruction_teacher | 0.209260 | 0.268140 | 0.268140 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_left | mono/single_frame/source_3 | reconstruction_teacher | 0.208820 | 0.267706 | 0.267706 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_right | mono/four_frame | reconstruction_teacher | 0.200025 | 0.257601 | 0.257601 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_right | mono/single_frame/source_0 | reconstruction_teacher | 0.213994 | 0.274123 | 0.274123 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_right | mono/single_frame/source_1 | reconstruction_teacher | 0.214010 | 0.273918 | 0.273918 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_right | mono/single_frame/source_2 | reconstruction_teacher | 0.214323 | 0.273466 | 0.273466 | 0.750000 | 1024 |
| umi | mono | cam_left_wrist_right | mono/single_frame/source_3 | reconstruction_teacher | 0.213424 | 0.272362 | 0.272362 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_left | mono/four_frame | reconstruction_teacher | 0.195156 | 0.250148 | 0.250148 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_left | mono/single_frame/source_0 | reconstruction_teacher | 0.205249 | 0.262358 | 0.262358 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_left | mono/single_frame/source_1 | reconstruction_teacher | 0.205234 | 0.263064 | 0.263064 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_left | mono/single_frame/source_2 | reconstruction_teacher | 0.205046 | 0.263193 | 0.263193 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_left | mono/single_frame/source_3 | reconstruction_teacher | 0.207247 | 0.265328 | 0.265328 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_right | mono/four_frame | reconstruction_teacher | 0.194836 | 0.250616 | 0.250616 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_right | mono/single_frame/source_0 | reconstruction_teacher | 0.196601 | 0.251337 | 0.251337 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_right | mono/single_frame/source_1 | reconstruction_teacher | 0.198700 | 0.253774 | 0.253774 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_right | mono/single_frame/source_2 | reconstruction_teacher | 0.199448 | 0.255777 | 0.255777 | 0.750000 | 1024 |
| umi | mono | cam_right_wrist_right | mono/single_frame/source_3 | reconstruction_teacher | 0.197706 | 0.252961 | 0.252961 | 0.750000 | 1024 |
| umi | stereo | head | stereo/four_frame | depth_head_teacher | 0.220685 | 0.278301 | 0.213585 | 0.526239 | 1020 |
| umi | stereo | left_wrist | stereo/four_frame | depth_head_teacher | 0.237258 | 0.317115 | 0.281658 | 0.416859 | 1024 |
| umi | stereo | right_wrist | stereo/four_frame | depth_head_teacher | 0.213092 | 0.289846 | 0.259936 | 0.378510 | 1023 |
| umi | stereo | head | stereo/single_frame/source_0 | depth_head_teacher | 0.225577 | 0.282544 | 0.218956 | 0.526239 | 1019 |
| umi | stereo | left_wrist | stereo/single_frame/source_0 | depth_head_teacher | 0.241560 | 0.317114 | 0.277141 | 0.418622 | 1021 |
| umi | stereo | right_wrist | stereo/single_frame/source_0 | depth_head_teacher | 0.215912 | 0.286056 | 0.248549 | 0.378661 | 1022 |
| umi | stereo | head | stereo/single_frame/source_1 | depth_head_teacher | 0.225889 | 0.282876 | 0.218246 | 0.528044 | 1020 |
| umi | stereo | left_wrist | stereo/single_frame/source_1 | depth_head_teacher | 0.243023 | 0.318425 | 0.277312 | 0.418617 | 1018 |
| umi | stereo | right_wrist | stereo/single_frame/source_1 | depth_head_teacher | 0.221134 | 0.291918 | 0.251521 | 0.378371 | 1021 |
| umi | stereo | head | stereo/single_frame/source_2 | depth_head_teacher | 0.225261 | 0.282268 | 0.220114 | 0.525319 | 1019 |
| umi | stereo | left_wrist | stereo/single_frame/source_2 | depth_head_teacher | 0.240560 | 0.316820 | 0.277463 | 0.415353 | 1021 |
| umi | stereo | right_wrist | stereo/single_frame/source_2 | depth_head_teacher | 0.217990 | 0.287572 | 0.250165 | 0.377930 | 1017 |
| umi | stereo | head | stereo/single_frame/source_3 | depth_head_teacher | 0.219105 | 0.274535 | 0.214810 | 0.525353 | 1020 |
| umi | stereo | left_wrist | stereo/single_frame/source_3 | depth_head_teacher | 0.241506 | 0.316472 | 0.277094 | 0.414843 | 1021 |
| umi | stereo | right_wrist | stereo/single_frame/source_3 | depth_head_teacher | 0.219764 | 0.291111 | 0.253940 | 0.379078 | 1018 |

## Bottleneck 与效率

| Eye | Mode | Encode P50/P90 ms | Posterior mean P50/P90 ms | Decode P50/P90 ms | E2E P50/P90 ms | samples/s | frames/s | Peak alloc/reserved GiB |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| mono | four_frame | 4.040/4.071 | 0.010/0.011 | 4.041/4.071 | 8.061/8.116 | 124.050 | 496.199 | 0.376/0.389 |
| mono | single_frame | 2.576/2.603 | 0.010/0.011 | 2.657/2.689 | 5.257/5.299 | 190.234 | 190.234 | 0.363/0.369 |
| stereo | four_frame | 4.980/5.002 | 0.010/0.010 | 4.004/4.035 | 8.794/8.858 | 113.716 | 454.866 | 0.534/0.660 |
| stereo | single_frame | 3.327/3.352 | 0.010/0.010 | 2.610/2.646 | 6.034/6.068 | 165.739 | 165.739 | 0.409/0.428 |

### Latent ABI

| Dataset | Eye | Camera | Mode | Input shape/dtype | Latent shape/dtype | C | Tokens/window | Tokens/input frame | Spatial × | Temporal × | View × |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| libero | mono | observation.images.cam_head_left | mono/four_frame | `[1, 1, 3, 4, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 64.000 | 256.0 | 4.0 | 1.0 |
| libero | mono | observation.images.cam_head_left | mono/single_frame/source_0 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| libero | mono | observation.images.cam_head_left | mono/single_frame/source_1 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| libero | mono | observation.images.cam_head_left | mono/single_frame/source_2 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| libero | mono | observation.images.cam_head_left | mono/single_frame/source_3 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| libero | mono | observation.images.cam_left_wrist_left | mono/four_frame | `[1, 1, 3, 4, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 64.000 | 256.0 | 4.0 | 1.0 |
| libero | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_0 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| libero | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_1 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| libero | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_2 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| libero | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_3 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_head_left | mono/four_frame | `[1, 1, 3, 4, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 64.000 | 256.0 | 4.0 | 1.0 |
| umi | mono | observation.images.cam_head_left | mono/single_frame/source_0 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_head_left | mono/single_frame/source_1 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_head_left | mono/single_frame/source_2 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_head_left | mono/single_frame/source_3 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_head_right | mono/four_frame | `[1, 1, 3, 4, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 64.000 | 256.0 | 4.0 | 1.0 |
| umi | mono | observation.images.cam_head_right | mono/single_frame/source_0 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_head_right | mono/single_frame/source_1 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_head_right | mono/single_frame/source_2 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_head_right | mono/single_frame/source_3 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_left | mono/four_frame | `[1, 1, 3, 4, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 64.000 | 256.0 | 4.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_0 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_1 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_2 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_3 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_right | mono/four_frame | `[1, 1, 3, 4, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 64.000 | 256.0 | 4.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_right | mono/single_frame/source_0 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_right | mono/single_frame/source_1 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_right | mono/single_frame/source_2 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_left_wrist_right | mono/single_frame/source_3 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_left | mono/four_frame | `[1, 1, 3, 4, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 64.000 | 256.0 | 4.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_left | mono/single_frame/source_0 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_left | mono/single_frame/source_1 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_left | mono/single_frame/source_2 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_left | mono/single_frame/source_3 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_right | mono/four_frame | `[1, 1, 3, 4, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 64.000 | 256.0 | 4.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_right | mono/single_frame/source_0 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_right | mono/single_frame/source_1 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_right | mono/single_frame/source_2 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | mono | observation.images.cam_right_wrist_right | mono/single_frame/source_3 | `[1, 1, 3, 1, 256, 256]` / `torch.float32` | `[1, 48, 1, 16, 16]` / `torch.float32` | 48 | 256 | 256.000 | 256.0 | 1.0 | 1.0 |
| umi | stereo | three canonical pairs | stereo/four_frame | `[3, 2, 3, 4, 256, 256]` / `torch.float32` | `[3, 48, 1, 16, 16]` / `torch.float32` | 48 | 768 | 192.000 | 256.0 | 4.0 | 2.0 |
| umi | stereo | three canonical pairs | stereo/single_frame/source_0 | `[3, 2, 3, 1, 256, 256]` / `torch.float32` | `[3, 48, 1, 16, 16]` / `torch.float32` | 48 | 768 | 768.000 | 256.0 | 1.0 | 2.0 |
| umi | stereo | three canonical pairs | stereo/single_frame/source_1 | `[3, 2, 3, 1, 256, 256]` / `torch.float32` | `[3, 48, 1, 16, 16]` / `torch.float32` | 48 | 768 | 768.000 | 256.0 | 1.0 | 2.0 |
| umi | stereo | three canonical pairs | stereo/single_frame/source_2 | `[3, 2, 3, 1, 256, 256]` / `torch.float32` | `[3, 48, 1, 16, 16]` / `torch.float32` | 48 | 768 | 768.000 | 256.0 | 1.0 | 2.0 |
| umi | stereo | three canonical pairs | stereo/single_frame/source_3 | `[3, 2, 3, 1, 256, 256]` / `torch.float32` | `[3, 48, 1, 16, 16]` / `torch.float32` | 48 | 768 | 768.000 | 256.0 | 1.0 | 2.0 |

## 输出健康、失败与排除样本

| Dataset | Eye | Camera | Mode | NaN | Inf | Invalid outputs | Teacher-empty views | Raw min/max | abs(output)>1 | Valid RGB values | Valid teacher pixels |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| libero | mono | observation.images.cam_head_left | mono/four_frame | 0 | 0 | 0 | 0 | -86.820450/64.991417 | 0.00194988 | 201326592 | 67108864 |
| libero | mono | observation.images.cam_head_left | mono/single_frame/source_0 | 0 | 0 | 0 | 0 | -43.801605/43.214970 | 0.00150903 | 50331648 | 16777216 |
| libero | mono | observation.images.cam_head_left | mono/single_frame/source_1 | 0 | 0 | 0 | 0 | -43.231117/43.153522 | 0.00151650 | 50331648 | 16777216 |
| libero | mono | observation.images.cam_head_left | mono/single_frame/source_2 | 0 | 0 | 0 | 0 | -43.120827/42.352318 | 0.00150728 | 50331648 | 16777216 |
| libero | mono | observation.images.cam_head_left | mono/single_frame/source_3 | 0 | 0 | 0 | 0 | -42.730122/42.281548 | 0.00150720 | 50331648 | 16777216 |
| libero | mono | observation.images.cam_left_wrist_left | mono/four_frame | 0 | 0 | 0 | 0 | -86.751846/64.710304 | 0.00193119 | 201326592 | 67108864 |
| libero | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_0 | 0 | 0 | 0 | 0 | -42.409969/41.637680 | 0.00145394 | 50331648 | 16777216 |
| libero | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_1 | 0 | 0 | 0 | 0 | -43.303570/42.714619 | 0.00146749 | 50331648 | 16777216 |
| libero | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_2 | 0 | 0 | 0 | 0 | -43.819431/42.882030 | 0.00146679 | 50331648 | 16777216 |
| libero | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_3 | 0 | 0 | 0 | 0 | -43.608147/43.038261 | 0.00145926 | 50331648 | 16777216 |
| umi | mono | observation.images.cam_head_left | mono/four_frame | 0 | 0 | 0 | 0 | -86.883881/64.906387 | 0.00202928 | 603979776 | 201326592 |
| umi | mono | observation.images.cam_head_left | mono/single_frame/source_0 | 0 | 0 | 0 | 0 | -43.684895/42.646675 | 0.00164203 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_head_left | mono/single_frame/source_1 | 0 | 0 | 0 | 0 | -43.574772/42.698967 | 0.00164509 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_head_left | mono/single_frame/source_2 | 0 | 0 | 0 | 0 | -43.529404/42.345100 | 0.00164172 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_head_left | mono/single_frame/source_3 | 0 | 0 | 0 | 0 | -43.372585/42.251106 | 0.00163910 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_head_right | mono/four_frame | 0 | 0 | 0 | 0 | -87.038551/64.910995 | 0.00203305 | 603979776 | 201326592 |
| umi | mono | observation.images.cam_head_right | mono/single_frame/source_0 | 0 | 0 | 0 | 0 | -43.360569/42.285198 | 0.00164098 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_head_right | mono/single_frame/source_1 | 0 | 0 | 0 | 0 | -43.633308/42.619213 | 0.00164270 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_head_right | mono/single_frame/source_2 | 0 | 0 | 0 | 0 | -43.901798/43.172749 | 0.00164544 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_head_right | mono/single_frame/source_3 | 0 | 0 | 0 | 0 | -43.181591/41.940979 | 0.00164800 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_left_wrist_left | mono/four_frame | 0 | 0 | 0 | 0 | -86.879013/64.870895 | 0.00199424 | 603979776 | 201326592 |
| umi | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_0 | 0 | 0 | 0 | 0 | -43.325211/42.297691 | 0.00160632 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_1 | 0 | 0 | 0 | 0 | -43.512081/42.563568 | 0.00160435 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_2 | 0 | 0 | 0 | 0 | -43.726734/42.706066 | 0.00160478 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_left_wrist_left | mono/single_frame/source_3 | 0 | 0 | 0 | 0 | -43.090057/42.513237 | 0.00160240 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_left_wrist_right | mono/four_frame | 0 | 0 | 0 | 0 | -86.965309/64.919266 | 0.00200071 | 603979776 | 201326592 |
| umi | mono | observation.images.cam_left_wrist_right | mono/single_frame/source_0 | 0 | 0 | 0 | 0 | -42.930603/42.005245 | 0.00160507 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_left_wrist_right | mono/single_frame/source_1 | 0 | 0 | 0 | 0 | -43.582115/42.737141 | 0.00160739 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_left_wrist_right | mono/single_frame/source_2 | 0 | 0 | 0 | 0 | -43.439827/42.225487 | 0.00160217 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_left_wrist_right | mono/single_frame/source_3 | 0 | 0 | 0 | 0 | -43.789352/43.133942 | 0.00160510 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_right_wrist_left | mono/four_frame | 0 | 0 | 0 | 0 | -86.966415/65.597237 | 0.00198021 | 603979776 | 201326592 |
| umi | mono | observation.images.cam_right_wrist_left | mono/single_frame/source_0 | 0 | 0 | 0 | 0 | -42.435051/41.628635 | 0.00157562 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_right_wrist_left | mono/single_frame/source_1 | 0 | 0 | 0 | 0 | -43.987152/42.976295 | 0.00157890 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_right_wrist_left | mono/single_frame/source_2 | 0 | 0 | 0 | 0 | -42.838943/41.799263 | 0.00158203 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_right_wrist_left | mono/single_frame/source_3 | 0 | 0 | 0 | 0 | -43.514740/42.981125 | 0.00157955 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_right_wrist_right | mono/four_frame | 0 | 0 | 0 | 0 | -87.095703/65.580727 | 0.00197964 | 603979776 | 201326592 |
| umi | mono | observation.images.cam_right_wrist_right | mono/single_frame/source_0 | 0 | 0 | 0 | 0 | -43.638218/42.225807 | 0.00157485 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_right_wrist_right | mono/single_frame/source_1 | 0 | 0 | 0 | 0 | -43.523060/42.563976 | 0.00157960 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_right_wrist_right | mono/single_frame/source_2 | 0 | 0 | 0 | 0 | -43.418545/42.219936 | 0.00158107 | 150994944 | 50331648 |
| umi | mono | observation.images.cam_right_wrist_right | mono/single_frame/source_3 | 0 | 0 | 0 | 0 | -43.646072/42.825951 | 0.00157321 | 150994944 | 50331648 |
| umi | stereo | three canonical pairs | stereo/four_frame | 0 | 0 | 0 | 5 | -87.202667/65.159447 | 0.00199976 | 1811939328 | 354766336 |
| umi | stereo | three canonical pairs | stereo/single_frame/source_0 | 0 | 0 | 0 | 10 | -44.371330/43.335358 | 0.00157905 | 452984832 | 88820084 |
| umi | stereo | three canonical pairs | stereo/single_frame/source_1 | 0 | 0 | 0 | 13 | -44.454636/43.490757 | 0.00158013 | 452984832 | 88921422 |
| umi | stereo | three canonical pairs | stereo/single_frame/source_2 | 0 | 0 | 0 | 15 | -44.296494/43.421337 | 0.00158078 | 452984832 | 88489841 |
| umi | stereo | three canonical pairs | stereo/single_frame/source_3 | 0 | 0 | 0 | 13 | -44.784473/44.197182 | 0.00158047 | 452984832 | 88534989 |

Teacher-empty view/frame 不影响同一固定窗口的 RGB 指标；该 view 的 teacher-relative error 缺失，几何汇总的 valid-sample count 会相应减少。
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/four_frame, sample=umi:287772baf29dc722bdef7d29126df886:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/four_frame, sample=umi:af26783a68d1d4a4e17ae6148806fe96:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/four_frame, sample=umi:e6aae8374d88c18178115896e1dce2b8:1, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/four_frame, sample=umi:e7205350a32022dbc4f3b103db63e599:2, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/four_frame, sample=umi:f54f6da3dbd8f015f98e94aa6004c3af:9, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:0f6af76e9db792b61c974753a7b97fa4:6, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:13e6224066dfa3c4b56be18531bdda6a:12, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:287772baf29dc722bdef7d29126df886:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:438e5d86513758438146be83a3889785:8, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:64e456085d2fdd337c3cbd5fcd98760f:6, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:af26783a68d1d4a4e17ae6148806fe96:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:e6aae8374d88c18178115896e1dce2b8:1, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:e6aae8374d88c18178115896e1dce2b8:1, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:e7205350a32022dbc4f3b103db63e599:2, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_0, sample=umi:f54f6da3dbd8f015f98e94aa6004c3af:9, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:2781342d156371b34f1cecc963f15fff:8, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:287772baf29dc722bdef7d29126df886:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:2c5040b9d650bfdaa02a12c86e0d5855:7, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:64e456085d2fdd337c3cbd5fcd98760f:6, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:6ef55cb9e3a061f447d8c0c373c55672:6, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:868dc9f0e05715cd0a2989385a3301e5:8, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:af26783a68d1d4a4e17ae6148806fe96:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:e6aae8374d88c18178115896e1dce2b8:1, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:e6aae8374d88c18178115896e1dce2b8:1, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:e7205350a32022dbc4f3b103db63e599:2, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:ef59b0d080fb3e9b5d5cc8b24544baea:12, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:f54f6da3dbd8f015f98e94aa6004c3af:9, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_1, sample=umi:ffc79c0a739e8ee6ee07de5ba7ee0aca:5, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:1e73c4a60530e18d75e07ef0516fb56e:7, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:287772baf29dc722bdef7d29126df886:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:64e456085d2fdd337c3cbd5fcd98760f:6, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:6ef55cb9e3a061f447d8c0c373c55672:6, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:75955c91cdcdb6894dca027b366d8f65:1, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:8445fa774321259fc93002fead8820e4:11, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:868dc9f0e05715cd0a2989385a3301e5:8, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:9bd16bd9afa9890686f2b9cd713bb495:8, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:9ca747dc894fe5a6af09c2ff16618103:5, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:af26783a68d1d4a4e17ae6148806fe96:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:b3af57c9e72ad60afd43969138a761a7:4, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:e6aae8374d88c18178115896e1dce2b8:1, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:e7205350a32022dbc4f3b103db63e599:2, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:f54f6da3dbd8f015f98e94aa6004c3af:9, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_2, sample=umi:fdacd407615f0cd4182ad18b377e0c00:9, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:2781342d156371b34f1cecc963f15fff:8, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:287772baf29dc722bdef7d29126df886:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:4d8abd81af6c74150c4825429ee90788:4, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:6ef55cb9e3a061f447d8c0c373c55672:6, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:868dc9f0e05715cd0a2989385a3301e5:8, view=left_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:ae6cb93bc2f3b2d150cd931325f583c3:12, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:af26783a68d1d4a4e17ae6148806fe96:0, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:b3af57c9e72ad60afd43969138a761a7:4, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:e6aae8374d88c18178115896e1dce2b8:1, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:e7205350a32022dbc4f3b103db63e599:2, view=head, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:e7bf2a8e9d15550128fc0322b9dfffc0:6, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:f54f6da3dbd8f015f98e94aa6004c3af:9, view=right_wrist, reason=empty_teacher_mask
- teacher exclusion: dataset=umi, eye=stereo, camera=three canonical pairs, mode=stereo/single_frame/source_3, sample=umi:fdacd407615f0cd4182ad18b377e0c00:9, view=right_wrist, reason=empty_teacher_mask

每个 selection 的 decode checked/rejected 与 rejected episode IDs SHA256 已记录在数据表；完整排除原因保存在 selection JSON。

### 固定案例

共 16 个固定窗口：UMI 8、LIBERO 8；每个窗口保存四个 source position 的原图/重建与几何图，`cases.json` 和每个 PNG 均已在报告生成时核验。

## 几何口径

- Mono：DA3 分别推理原图与重建图，报告 `reconstruction_teacher_relative_*`。
- Stereo：decoder 不重建右眼，报告 `depth_head_teacher_relative_*`；不称为 stereo 重建精度。
- 没有独立真实 depth/disparity GT，因此本报告不声称真实几何 accuracy。

## 阻断与未完成项

- 每个 selection 的 decode checked/rejected 与 rejected episode IDs SHA256 已记录；完整排除原因保存在 selection JSON。
- HY：BLOCKED，不进入任何 macro average；原因是 canonical Lance `index` 与 pinned loader 的物理 offset 合同冲突。
- rFID、RAFT warp/static flicker/motion consistency：Pending Stage A2。
- rFVD：N/A，现有冻结 I3D-FVD 实现不支持本项目原生 4 帧合同；扩帧/插帧会改变评测对象。
- FVMD：N/A，尚无经验证适用于原生 4 帧的冻结实现。

## 决策

1. **值得继续，但需要补 A2 与 HY 修复。** A1 可作为 preliminary baseline，不能表述成完整 Stage A。
2. **最大风险：** 当前几何指标只有 teacher-relative 证据；若误写为真实 depth/disparity accuracy，会得到错误结论。
3. **最缺的关键证据：** 独立真实几何 GT，以及 rFID/显式 warp-motion 指标；HY 的 identity/offset 合同也仍缺上游确认。
4. **今天可执行的最小下一步：** 固定同一批 16 个案例做人工异常审查，并为 A2 锁定 rFID 与 RAFT 的权重、预处理和版本。
5. **置信度：80%（中等）。** 对 A1 已报告数字和可复现合同置信度较高；因 HY、A2 与独立几何 GT 缺失，不给高置信度。
