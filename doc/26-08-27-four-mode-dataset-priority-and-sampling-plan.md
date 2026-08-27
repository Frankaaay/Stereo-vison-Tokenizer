# 四模式 Tokenizer 预训练数据优先级与采样方案

官方数据源补充核对：

- ABC-130k：<https://huggingface.co/datasets/XDOF/ABC-130k>
- T-Rex：<https://huggingface.co/datasets/zekaiwang/trex_dataset>
- HiFi-UMI-2K：<https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K>
- RealOmni：<https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData>
- Daimon-Infinity：<https://modelscope.cn/datasets/daimonrobotics/Daimon-Infinity>
- Open-AoE：<https://github.com/ant-research/Open-AoE>
- EgoSuite-Open100K：<https://www.humanoidsdata.com/datasets/egosuite-open100k>

## 2. 当前 teacher 合同：单目 DA3，双目 LAS2-H

| eye mode | 当前正式 teacher | 固定参数 | 说明 |
| --- | --- | --- | --- |
| mono | DA3-BASE | `process_res=504`、`upper_bound_resize`、DA3 不接收 VAE padding | 生成单目 relative-depth 监督 |
| stereo | LAS2-H | `max_disp=192`、`valid_iters=4`、左右双向推理 | 生成双目 disparity，之后经过 LR consistency、disparity range 和 non-padding mask |

## 3. 优先级定义

本文 P0-P3 评价的是“数据本身是否适合用于对应 eye mode”，不评价当前固定 `V=3` 数据接口。
如果数据是有效的一组或多组双目，后续可以把代码改成支持可变 view 数和 view mask；不能因为
当前实现固定三组双目而把真实双目数据降级。

| 等级 | 定义 |
| --- | --- |
| P0 | 相机类型、连续视频、公开数据或已落盘 schema 很明确，可以立即设计正式 adapter；完成标准质量门禁后即可训练 |
| P1 | 高概率可用，但仍缺完整落盘、子集占比、标定发布情况或一个关键样本核验 |
| P2 | 理论可能可用，但当前缺少稳定下载、视觉 schema、同步/标定证据或原始轨迹 |
| P3 | 当前不适合作为该 eye mode 数据，或只有静态候选、论文、评测资产而没有可用连续视觉轨迹 |

时长列严格区分发布方声称值、表内实测值和未知值。`待统计`、episode 数和 frame 数不能擅自换算为小时。

## 4. 单目数据优先级表

### 4.1 单目判定原则

连续 RGB 视频不需要相机标定即可用于单目两种模式。多相机数据按 camera key 拆成独立单目流；
不能把不同相机、不同 episode 或不同机器人配置硬拼成四帧。当前单帧路径仍从四帧 source window
选定一帧，因此只有孤立图片、没有 episode 时间轴的数据并非现成可用。

| 优先级 | 数据集 | 时长/规模口径 | 可覆盖模式 | 数据使用建议与门禁 |
| --- | --- | --- | --- | --- |
| P0 | Hy-Embodied-0.5-VLA-Data | 2,163 h | mono/single、mono/four | `cam_high`、左右 wrist 分成三路单目；复核两节点完整性和 missing-key 补片 |
| P0 | ABC-130k | 发布方 3,590.7 h；当前表内本地 52 episodes、约 1.7 h | mono/single、mono/four | top、left wrist、right wrist 分流；MCAP/H.264/H.265 解码并按时间戳建窗口 |
| P0 | T-Rex | 50.7 h | mono/single、mono/four | `head_left`、`left_wrist`、`right_wrist` 分流；20 路触觉视频不混入 RGB |
| P0 | RoboMIND2.0-Agilex | 16.7 h | mono/single、mono/four | front/left/right RGB 分流；HDF5 adapter |
| P0 | RoboMIND2.0-Agilex-mobile | 10.2 h | mono/single、mono/four | 同上，保留 mobile domain 标签 |
| P0 | RoboMIND2.0-Ark | 14.0 h | mono/single、mono/four | left/right/top 分流 |
| P0 | RoboMIND2.0-Ark-mobile | 5.9 h | mono/single、mono/four | left/right/top 分流，保留 mobile domain 标签 |
| P0 | RoboMIND2.0-Franka Part-1~5 | 29.8 h | mono/single、mono/four | front/left/right/top/wrist 各自采样；按 Part 处理相机缺失 |
| P0 | RoboMIND2.0-Franka-sim | 2.8 h | mono/single、mono/four | 六路相机分流；sim 与 real 分域 |
| P0 | RoboMIND2.0-Tianyi | 20.8 h | mono/single、mono/four | `camera_top` |
| P0 | RoboMIND2.0-Tianyi-mobile | 7.9 h | mono/single、mono/four | `camera_top`，保留 mobile domain 标签 |
| P0 | RoboMIND2.0-Tienkung | 15.1 h | mono/single、mono/four | `camera_top` |
| P0 | RoboMIND2.0-Tienkung-sim | 23.4 h | mono/single、mono/four | `camera_head`；sim 与 real 分域 |
| P0 | RoboMIND2.0-UR5 | 43.3 h | mono/single、mono/four | 六路相机分流；7.231 FPS 下明确时间 stride |
| P0 | HiFi-UMI-2K | 发布集 2,000 h；源语料 20,000+h | mono/single、mono/four | 六路视频都可独立作为单目流；使用 `valid.frame` 并保持硬件同步时间轴 |
| P0 | RealOmni / 10Kh-RealOmin-OpenData | 13,000+h | mono/single、mono/four | 每个 gripper 鱼眼相机分别使用；双手相机不能互相拼成固定双目 |
| P0 | Daimon-Infinity | 首批 1,000 h；目标规模更大 | mono/single、mono/four | 按 DataTac/DataClaw/DataDex 配置拆分；每路普通 RGB 独立采样 |
| P0 | Open-AoE-2000H | 官方首发约 2,000 h；表内旧汇总曾写约 694 h 已准备 | mono/single、mono/four | 手机第一人称连续视频；实际落盘时重新统计有效时长，过滤非操作和严重抖动片段 |
| P1 | EgoSuite-Open100K | 分阶段目标/发布口径 100,000 h | mono/single、mono/four | head-view、wrist-view 分流；核验实际已发布小时数和 no-resale 许可 |
| P1 | HuMI | 已抽样 HF 子任务约 1.29 h；全量 827 demos、总时长待统计 | mono/single、mono/four | camera0/camera1 分流；视频完整落盘后确认真实 FPS、episode 边界 |
| P1 | UMI-3D | 4.6K demos；总时长待统计 | mono/single、mono/four | 下载 Zarr 后读取 RGB 分辨率、FPS 和有效帧 |
| P1 | RoboMIND2.0-UR5-Dex | 表内写 7.3 h，但视觉 schema、路径和格式状态为空 | 预计 mono/single、mono/four | 先验证 HDF5 相机字段和真实落盘状态 |
| P1 | RoboCasa365 | 600+h human + 1,600+h auto | 预计 mono/single、mono/four | human/auto、sim/real、embodiment 分域；抽样验证 365 版本发布物 |
| P1 | MIKASA-Robo-VLA | 约 22,500 trajectories、6M+ timesteps；总时长待统计 | 预计 mono/single、mono/four | 先确认每个 timestep 是否都有连续 RGB |
| P1 | RoboMME | 约 21.3 h | 预计 mono/single、mono/four | 只取训练型连续轨迹；先抽一个 episode 核对 RGB/action/schema |
| P1 | Deform360 | >215 h、1,980 multi-view sequences | mono/single、mono/four | 每路 multi-view camera 可先独立作为单目；下载后核对时序和许可 |
| P1 | WatchAct | 3,000+ long-horizon instances；总时长待统计 | 预计 mono/single、mono/four | 需确认原始视频端点、许可和连续帧 |
| P2 | TAMEn | 724 demos；总时长待统计 | 预计 mono/single、mono/four | 数据仍按阶段发布；当前主要证据是采集代码和 10 FPS topic |
| P2 | BiCoord | 待统计 | 待定 | 无稳定视觉 schema，先取单个 episode |
| P2 | DuoBench | 待统计 | 待定 | 无稳定视觉 schema，先取单个 episode |
| P2 | RMBench | 评测型，无统一总时长 | 待定 | 只有存在原始连续 RGB 时才进入训练池 |
| P2 | RoboDojo | 待统计 | 待定 | sim/real 分域；下载入口和 schema 未稳定 |
| P2 | RoboMemArena | 评测型，无统一训练时长 | 预计 mono/four 更有价值 | 仅取原始长轨迹，不能把评测结果页当训练数据 |
| P2 | UniVTAC | 生成型/触觉 benchmark，无统一总时长 | 待定 | RGB 与触觉严格分模态，确认是否有连续视觉 episode |
| P2 | HABIT | 164.19 h、10,563 episodes | 待定 | 论文给出规模，但表中没有原始数据端点 |
| P2 | RoboTacDex | 6,000 trajectories；总时长待统计 | 待定 | 未确认原始数据端点 |
| P2 | TactiDex | 待统计 | 待定 | 需确认公开工件、RGB 流和许可 |
| P2 | ViTacWorld | 待统计 | 待定 | 未确认独立原始轨迹发布 |
| P2 | PREFAIL | 待统计 | 待定 | 未确认原始连续视觉数据 |
| P2 | Real-IKEA | 待统计 | 待定 | 未确认原始轨迹发布 |
| P2 | YUBI | 8,434 h interaction data；时长为论文口径 | 待定 | 当前有硬件/论文规模，未确认独立原始数据下载 |
| P2 | UMI-Bench 1.0 | 约 20K demonstrations；总时长未发布 | 待定 | 确认训练数据工件与许可后再接入 |
| P2 | OmniUMI | 待统计 | 预计 mono/single、mono/four | RGB-D 的 RGB 通道可作单目；当前没有原始数据 URL |
| P2 | TacUMI | 论文分割实验约 0.50 h、约 30K frames | 预计 mono/single、mono/four | 16.67 Hz；未确认独立语料发布 |
| P2 | UMI-Underwater | 陆地约 6.70 h；水下约 15 h，后者含 reset | 预计 mono/single、mono/four | 陆地/水下分域；仓库未提供原始轨迹，15 h 不能直接当有效示范时长 |
| P2 | HoMMI | 481 demonstrations；总时长未公布 | 预计 mono/single、mono/four | first-person/hand view 分流；未确认数据工件 |
| P3 | GraspIT | 2.3M simulated grasp candidates，不是时间序列时长 | 当前四模式不可直接使用 | 静态候选不满足当前四帧 source-window 合同 |
| P3 | TableVerse | 待统计 | 当前不可确定 | 只有场景/感知候选信息，没有连续动作轨迹证据 |

### 4.2 单目不是零预处理

单目主要是视频数据工程，确实不需要双目的几何预处理，但仍必须：

1. 按 camera key 拆独立流，统一 RGB、旋转和色彩空间；
2. 验证 episode、frame index、timestamp，四帧窗口不得跨 episode；
3. 按真实 FPS 定义 temporal stride，不能让 7 FPS 与 60 FPS 的“四个连续帧”代表完全不同的隐含时间尺度；
4. 过滤坏帧、黑屏、严重失焦、长静止和重复片段；
5. Student 做通用 letterbox，DA3 使用独立 resize/归一化，DA3 不接收 Student/VAE padding；
6. 保留 dataset、camera、episode、frame/timestamp、源文件 hash 和转换合同 provenance；
7. 原生 RGB-D depth 暂不自动替换 DA3；若要引入，应另立监督合同和可比实验。

相比之下，双目还要额外完成左右同步、内外参、baseline、畸变校正、rectification audit、
LAS2-H 双向推理、LR consistency 和 disparity valid-mask 质量门禁。

## 5. 双目数据优先级表

### 5.1 双目判定原则

“有两个相机”或“有内参”都不够。可用于双目训练的数据至少需要：

- 同一刚性 rig 上可识别的 left/right 图像；
- 左右时间同步，或有足够精确的 timestamp 可匹配；
- 双眼 intrinsics、distortion、相对 extrinsics/baseline；
- 可校正到共同极线几何，或有可信的 pre-rectified 证明；
- 连续 episode 中至少能构造四个双目时刻；
- 校正后有效重叠 FOV 足够，LAS2-H disparity 和 LR-consistency valid mask 不塌缩。

以下只评价数据本身。即使只有一组双目，只要上述条件成立，就可以把模型和 dataset adapter 改成
可变 `V` 或 `view_mask` 后使用；不允许复制同一双目 pair 三次伪造三视角。

| 优先级 | 数据集/候选相机组 | 时长/规模口径 | 数据可用性判断 | 必须完成的双目门禁 |
| --- | --- | --- | --- | --- |
| P0 | ABC-130k / ZED-X top stereo | ABC 总计 3,590.7 h；ZED-X 子集时长待统计；当前本地仅约 1.7 h 部分 episode | 官方明确 ZED-X station 为顶部双目，MCAP 含相机 calibration topic；可用于 stereo/single、stereo/four | 筛选 ZED-X episode；确认双眼编码/stream；读取 K/D/R/P 和 baseline；左右同步；rectification audit；LAS2-H valid-mask gate |
| P0 | HiFi-UMI-2K / head stereo pair | 发布集 2,000 h | 官方明确 head left/right，所有传感器硬件同步误差 <40 us；可用于两种双目模式 | 确认发布 metadata 中完整双目标定和 baseline；校正审计；过滤 `valid.frame=false`；LAS2-H gate |
| P1 | HiFi-UMI-2K / left-hand up/down 与 right-hand up/down | 发布集 2,000 h | 每只手两路非平行鱼眼且硬件同步，可能形成两组广角双目，但不能直接按标准平行双目处理 | 核对每只手两相机刚性外参、重叠 FOV、鱼眼模型和可校正区域；分别统计 LAS2-H 有效像素率 |
| P1 | Daimon-Infinity / DM-DataDex headset left/right | 首批数据总计 1,000 h；DataDex 子集时长待统计 | 官方结构含 headset left/right，系统说明含 stereo camera + IMU；高概率可用 | 确认左右图像是同步原始流；读取 episode calibration、baseline；按 configuration 隔离；rectification/LAS2-H gate |
| P2 | Deform360 / multi-view camera pairs | >215 h、1,980 sequences | 多视角序列有潜力，但表中没有确认刚性相机、同步或标定发布 | 下载样本；枚举固定 camera pair；核对 K/D/外参/timestamp；只保留校正后有效重叠区域 |
| P2 | HuMI / camera0_rgb + camera1_rgb | 已抽样子任务约 1.29 h；全量 827 demos、总时长待统计 | 两路 60 FPS 相机，但当前没有证据证明它们是固定左右眼 | 视频落盘；核对 rig 安装、同步、内外参和 baseline；通过后可升级 |
| P2 | RoboMIND2.0-Agilex / camera_left + camera_right | 16.7 h | 可能是固定环境相机对，但名称不能证明双目 | 从 HDF5/设备 metadata 提取标定与 timestamp；检查视场重叠和极线误差 |
| P2 | RoboMIND2.0-Agilex-mobile / camera_left + camera_right | 10.2 h | 同上 | 同上，另检查移动底盘振动下标定是否稳定 |
| P2 | RoboMIND2.0-Ark / camera_left + camera_right | 14.0 h | 可能是固定环境相机对 | 样本级标定、同步和 rectification audit |
| P2 | RoboMIND2.0-Ark-mobile / camera_left + camera_right | 5.9 h | 可能是固定环境相机对 | 同上，另检查移动平台标定稳定性 |
| P2 | RoboMIND2.0-Franka Part-1~5 / 固定 left-right 候选 | 29.8 h | 多相机有候选 pair，但各 Part camera set 不一致 | 按 Part 枚举 pair；禁止把两个 wrist 相机跨移动机械臂当固定双目；核对标定和同步 |
| P2 | RoboMIND2.0-Franka-sim / 仿真固定相机 pair | 2.8 h | 仿真可能直接提供完整 camera matrix | 从仿真配置导出 K/外参/baseline；sim 单独分域；校正和 LAS2-H gate |
| P2 | RoboMIND2.0-UR5 / 固定 left-right 候选 | 43.3 h | 多相机有候选 pair | 区分固定环境相机与两臂 wrist 相机；仅固定刚性 pair 可用 |
| P2 | TAMEn / left_camera + right_camera | 724 demos；总时长待统计 | 有 left/right RGB topic，但可能分别随双手运动 | 数据发布后确认 rig；若分别装在两只相对运动的手上则不能作为固定双目 |
| P2 | RoboCasa365 / 仿真固定相机 pair | 600+h human + 1,600+h auto | 仿真通常可以获取完整 camera matrix，但当前 365 schema 未核验 | 从 scene config 枚举固定 pair；sim 域隔离；校正与 LAS2-H gate |
| P2 | OmniUMI / RGB-D 与 external view | 总时长待统计 | 多视角有潜力，当前没有原始数据和标定 schema | 等实际数据；确认刚性 pair、同步、标定和连续帧 |

### 5.2 当前证据下不能算双目的数据

| 优先级 | 数据集 | 时长/规模口径 | 不能算双目的原因 |
| --- | --- | --- | --- |
| P3 | T-Rex | 50.7 h | 落盘普通 RGB 只有 `head_left`、`left_wrist`、`right_wrist`；没有 head right 图像。Sheet3 的“双目”标签与详细 schema 冲突 |
| P3 | RealOmni | 13,000+h | 每个 gripper 一台鱼眼相机；左右 gripper 会相对运动，不构成固定 baseline |
| P3 | Hy-Embodied-0.5-VLA-Data | 2,163 h | high、left wrist、right wrist 是三路独立单目视角 |
| P3 | UMI-3D | 4.6K demos；总时长待统计 | 当前只有单路 RGB |
| P3 | RoboMIND2.0-Tianyi | 20.8 h | 单相机 |
| P3 | RoboMIND2.0-Tianyi-mobile | 7.9 h | 单相机 |
| P3 | RoboMIND2.0-Tienkung | 15.1 h | 单相机 |
| P3 | RoboMIND2.0-Tienkung-sim | 23.4 h | 单相机 |
| P3 | EgoSuite-Open100K | 分阶段 100,000 h 口径 | 当前公开描述以单目第一人称为主，没有双目证据 |
| P3 | Open-AoE-2000H | 约 2,000 h | 智能手机第一人称单目，没有双目 rig 证据 |
| P3 | Daimon-Infinity / 两个独立 gripper RGB | 首批总计 1,000 h | 两个可相对运动的 gripper camera 不能互相组成固定双目；只有 DataDex headset left/right 单独列为候选 |
| P3 | 其余 P1/P2 单目候选 | 见单目表 | 当前没有可核验的同步、刚性 left/right、完整标定和 baseline；不能凭“多相机”或相机名称升级 |

## 7. 推荐的分阶段单双目混合方案

| 训练阶段 | mono/single | mono/four | stereo/single | stereo/four | eye-mode 合计 | 目的 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| smoke/功能验证 | 25% | 25% | 25% | 25% | mono 50% / stereo 50% | 只验证四条路径、DDP、显存和 resume，不代表正式比例 |
| 正式前 0-20% | 40% | 40% | 10% | 10% | mono 80% / stereo 20% | 充分利用大规模单目，先建立外观与时间表征，同时不断开双目分支 |
| 正式中间 20-80% | 35% | 35% | 15% | 15% | mono 70% / stereo 30% | 稳定训练 stereo fusion，控制双目重复率 |
| 正式最后 80-100% | 30% | 30% | 20% | 20% | mono 60% / stereo 40% | 末期提高几何和跨眼能力，避免最终 checkpoint 退化成单目模型 |

上述百分比是 optimizer update 比例，不是原始图像数。一个 `stereo/four` sample 的输入图像数量远高于
`mono/single`，所以 20% 双目 update 已经是强双目信号。正式启用前还要用实际 step time、LAS2-H
teacher time、峰值显存和每个 mode 的有效监督像素率复核计算预算。

### 7.1 确定性实现建议

继续使用 deterministic cycle，不在每个 rank 独立调用普通随机数：

| 阶段 | cycle 长度 | 每 cycle 精确 mode 计数 |
| --- | ---: | --- |
| 0-20% | 10 | `4,4,1,1` |
| 20-80% | 20 | `7,7,3,3` |
| 80-100% | 10 | `3,3,2,2` |

每个 cycle 用 `seed + phase_id + cycle_index` 确定性 shuffle。checkpoint/run manifest 必须冻结：

- 三个 phase 边界；
- 每阶段 ratio/cycle counts；
- schedule seed；
- 当前 global update 和每 mode occurrence；
- world size、per-device batch 和 gradient accumulation。

resume 必须恢复到完全相同的 mode 序列；ratio 或 phase boundary 改变时不能 strict resume。

### 7.2 mode 内的数据集采样

不要把所有数据集按总帧数直接 concatenate。推荐三级采样：

```text
mode -> dataset -> camera/episode/window
```

- dataset 初始权重使用 `w_i ∝ sqrt(unique_valid_windows_i)`，避免 100K h 级数据线性淹没小而高质量的数据；
- 单个 dataset 在同一 eye mode 内初始上限建议 25%；
- 同一数据集有六路相机时，不应仅因相机数多就自动获得六倍 dataset 权重；
- P0 数据进入正式池，P1 先通过 adapter/corruption/teacher-valid gate，P2 只进入独立 pilot；
- 记录 dataset/camera/mode 的 unique window、sample count、repeat factor 和 effective epochs；
- 双目池重复率过高时降低该 dataset 权重，但 stereo eye-mode 全程保留下限，不能出现 mono-only tail。


### 7.3 必须保留的启动陷阱

`scripts/stereo/train_stereo_vae.sh` 当前仍有：

```bash
FOUNDATION_STEREO_BACKEND="${FOUNDATION_STEREO_BACKEND:-pytorch}"
```

因此不能依赖 Python 入口默认值。通过 shell launcher 启动正式实验时，必须显式 pin：

```bash
export FOUNDATION_STEREO_BACKEND=las2_h
export LAS2_H_REPO=...
export LAS2_H_SOURCE_SHA=...
export LAS2_H_CHECKPOINT=...
export LAS2_H_CHECKPOINT_SHA256=...
export LAS2_H_VALID_ITERS=4
export LAS2_H_MAX_DISP=192
```

否则 shell 会显式向 Python 传 `pytorch`，从而跑回 FoundationStereo。代码中的
`FoundationStereoOnlineTeacher`、`OnlineFoundationGTCallback`、
`foundation_stereo_backend` 和部分 `teacher_kind="foundation_stereo"` 是兼容历史后端的
泛化命名，不能据此判断实际运行的是 FoundationStereo。最终运行事实必须读取
`resolved_config.json` 和 run metadata 中的 `online_gt.backend`；正式 LAS2-H 实验必须为
`las2_h`。

## 8. H200 x8、GAN-off 下的 P0+P1 一遍工期估算

### 8.1 计算口径

本节只使用 2026-08-27 的 H200 x8、global batch 192、GAN-off 稳态实测：

| mode | 中位 step time | global sample-executions/s |
| --- | ---: | ---: |
| mono/single | 157.96 ms | 1,215.5 |
| mono/four | 530.04 ms | 362.2 |
| stereo/single | 353.65 ms | 542.9 |
| stereo/four | 1,457.48 ms | 131.7 |

当前 LeRobot 正式窗口合同是 30 FPS、`frame_offsets=[0,3,6,9]`、
`START_STRIDE=12`，所以每条连续相机时间线每小时约产生：

```text
30 * 3600 / 12 = 9,000 windows/hour
```

这里的“一遍”定义为：每个 unique window 被分配给一个 temporal mode 执行一次；不是让
`single` 和 `four` 各自都再完整看一遍。同一数据集的多个相机流如果都独立枚举，则必须按
相机流数增加 unique windows，不能仍按原始 episode 小时数计算。

正式三阶段比例在完整训练区间上的积分为：

```text
mono/single   = 35%
mono/four     = 35%
stereo/single = 15%
stereo/four   = 15%
```

由实测 step time 得到加权平均 `512.47 ms/step`，即约 `374.66` global
sample-executions/s。其中 mono 覆盖速度约 `262.26 windows/s`，stereo 执行速度约
`112.40 windows/s`。

### 8.2 当前可量化数据量

| 口径 | 时长 | 说明 |
| --- | ---: | --- |
| P0 合计 | 23,994.3 h | 按表中发布/首批小时口径求和 |
| 已量化 P1（不含 EgoSuite） | 至少 2,443.6 h | UR5-Dex 7.3 + RoboCasa365 2,200 + RoboMME 21.3 + Deform360 至少 215 |
| P0 + 已量化 P1（不含 EgoSuite） | 至少 26,437.9 h | HuMI 全量、UMI-3D、MIKASA、WatchAct 未知时长未计入 |
| 再计入 EgoSuite 100K 分阶段目标 | 至少 126,437.9 h | 100K 是目标/发布口径，不等于当前已经全部落盘可训 |

### 8.3 估算结果

| 场景 | H200 x8 理想训练时间 | 含义 |
| --- | ---: | --- |
| P0 + 已量化 P1，不含 EgoSuite，保持正式分阶段混合 | **约 10.50 天** | mono 数据完整覆盖一遍，同时按阶段比例持续抽 stereo |
| 再计入 EgoSuite 100K，保持正式分阶段混合 | **约 50.22 天** | 当前最适合作为“全部 P0+P1 一遍”的预算下限 |
| P0 + 已量化 P1，只跑 mono 且 single/four 各占一半 | 约 4.93 天 | 纯吞吐下限，不是建议的正式训练 |
| 再计入 EgoSuite 100K，只跑 mono | 约 23.60 天 | 纯吞吐下限，会完全停止 stereo 分支更新 |

正式混合的 50.22 天不是“所有数据已知后的精确 ETA”，而是单条代表时间线口径的计算下限。
它尚未计入：未知时长的四个 P1 数据集、每个数据集的多相机独立流放大、损坏/过滤后的有效时长变化、
checkpoint/validation/重启，以及正式数据解码相对 smoke cache 的 I/O 差异。可用下式在 manifest 统计后
直接更新：

```text
formal_days = unique_mono_windows / 262.26 / 86400
            = effective_mono_camera_hours * 9000 / 262.26 / 86400
```

因此每增加 10,000 个有效 mono camera-hours，正式分阶段混合约增加 **3.97 天**。

### 8.4 双目池覆盖和重复率

按 single/four 各占一半单独遍历 stereo pool，HiFi head 的 2,000 pair-hours 约需
**0.98 天**；若左右手两组 P1 也通过门禁，三组共 6,000 pair-hours 约需 **2.95 天**。
ABC ZED-X 和 Daimon DataDex 的真实 stereo 子集时长仍未知，暂时不能加入精确总数。

在正式三阶段积分比例下，mono/stereo 总执行数为 70/30。覆盖 26,437.9 个 mono
timeline-hours 时，会执行相当于约 11,330.5 stereo pair-hours；计入 EgoSuite 100K 后则约
54,187.7 pair-hours。即使把 ABC 的全部 3,590.7 h、Daimon 的全部 1,000 h 和 HiFi 三组
pair 的 6,000 h 都当作可用双目，当前候选上界也只有 10,590.7 pair-hours；因此前一个
11,330.5 pair-hours 预算已经足够让当前表内 stereo P0+P1 至少覆盖一遍。若合格 stereo pool
只有 HiFi head 的 2,000 pair-hours，等效重复约
5.67x / 27.09x；即使三组 HiFi 共 6,000 pair-hours 都合格，也约为 1.89x / 9.03x。
这说明正式训练必须记录每个 stereo dataset 的 repeat factor，并在 ABC/DataDex 子集时长落盘后
重新决定 stereo 上限；不能用简单 concatenate 让训练尾段变成无控制的重复采样。
