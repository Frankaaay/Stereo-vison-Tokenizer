#!/usr/bin/env bash
set -euo pipefail

cd /data/home/frank/projects/Stereo-vison-Tokenizer
source /data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260828/venv/bin/activate
export PYTHONUNBUFFERED=1

export ABLATION_OUTPUT_ROOT=/data/home/frank/experiments/stereo-input-ablation-permode-h2001-20260904-v1
export OUTPUT_ROOT=/data/home/frank/experiments/stereo-input-ablation-permode-h2001-20260904-v1/placeholder
export GPU_COUNT=8
export PER_DEVICE_BATCH_SIZE=24
export GRAD_ACCUMULATES=1
export MAX_STEPS=40000
export MODE_UPDATES_PER_EPOCH=40000
export LEARNING_RATE=1e-4
export MIN_LEARNING_RATE=1e-4
export WARMUP_STEPS=20
export KL_WARMUP_STEPS=100
export RGB_WEIGHT=1.0
export RELATIVE_DEPTH_WEIGHT=1.0
export RELATIVE_GRADIENT_WEIGHT=0.1
export KL_WEIGHT=1e-6
export PERCEPTUAL_WEIGHT=1.0
export SINGLE_FRAME_SOURCE_INDEX=0
export MODE_UPDATE_WEIGHTS=35:35:15:15
export MODE_BATCH_SIZES=192:40:160:36
export MODE_GRAD_ACCUMULATES=1:1:1:1
export MONO_DATASET_WEIGHTS=9:1
export CHECKPOINT_EVERY_N_STEPS=5000
export ONLINE_VAL_CHECK_INTERVAL_STEPS=2000
export NUM_WORKERS=8
export PREFETCH_FACTOR=2
export DISABLE_MEDIA_LOGGING=1
export WANDB_MODE=offline

export HY_MANIFEST=/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260829/manifests/hy_formal_90_5_5_v1.jsonl
export HY_ROOT_ALIASES='{"hy_primary":"/data/shared/hy_embodied/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data","hy_rest":"/data/shared/hy_embodied_rest/Hy-Embodied/huggingface_tencent_Hy-Embodied-0.5-VLA-Data"}'
export LIBERO_MANIFEST=/data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260829/manifests/libero_formal_90_5_5_v1.jsonl
export LIBERO_ROOT_ALIASES='{"libero":"/data/shared/offline/datasets/libero_mujoco3.3.2"}'
export UMI_MANIFEST=/data/home/frank/runtime/umi-lerobot-decode-audit-h2001-20260829-v1/umi_lerobot_decode_verified_v1.jsonl
export UMI_DATASET_ROOT=/data/shared/datasets/umi_lerobot_v3_260714
export UMI_RECTIFICATION_AUDIT_SHA256=41d2bfecaae85dd18f7cfd1a2a3a2177e8fd4aa8897be1cb411d85c3092a7d25
export NODE_MANIFEST_CONTRACTS='{"0":{"hy":"b25efc945ccd7e7afd2f1a76393ea19adde8fa072e1e9a2ca6348e0e5c1a45f9","libero":"0299354a7225e979f6b9ff5fb3e26a975c811d2d41af44e042a6eade3f24bbf4","umi":"5e8f58c769549372af070a6132ad826bd7172aaeabcebebff84426e66bc2120f"}}'

export DA3_REPO=/data/home/frank/runtime/depth-anything-3/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
export DA3_SOURCE_SHA=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
export DA3_CHECKPOINT=/data/home/frank/artifacts/depth-anything-3/DA3-BASE/f4a6c9b3c95e41c82048423d3493a81ec3fa810e
export DA3_CHECKPOINT_SHA256=e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5
export FOUNDATION_STEREO_BACKEND=las2_h
export LAS2_H_REPO=/data/home/frank/runtime/lite-any-stereo-8c97bd4-clean
export LAS2_H_SOURCE_SHA=8c97bd4c4da3712c2ac60003a23201dfdb5935f4
export LAS2_H_CHECKPOINT=/data/home/frank/artifacts/lite-any-stereo/LAS2_H.pth
export LAS2_H_CHECKPOINT_SHA256=758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4
export LAS2_H_VALID_ITERS=4
export LAS2_H_MAX_DISP=192
export FOUNDATION_STEREO_PAIR_MICROBATCH=48
export ONLINE_GT_CACHE_ENABLED=0

exec bash scripts/stereo/run_stereo_input_ablation_serial.sh
