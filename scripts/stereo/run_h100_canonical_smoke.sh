#!/usr/bin/env bash
set -euo pipefail

: "${OUTPUT_ROOT:?set a fresh output root}"
: "${MANIFEST_ROOT:?set the generated H100 manifest root}"
: "${HY_ROOT:?set the canonical Hy dataset root}"
: "${LIBERO_ROOT:?set the LIBERO suite root}"
: "${UMI_ROOT:?set the canonical UMI dataset root}"
: "${TRAIN_ENV:?set the existing H100 training environment}"
: "${EXTERNAL_ROOT:?set the frozen teacher source root}"
: "${TEACHER_ROOT:?set the frozen teacher checkpoint root}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse smoke output: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

export PATH="${TRAIN_ENV}/bin:${PATH}"
export OUTPUT_ROOT
export DISTRIBUTED_MODE=single
export NUM_NODES=1
export GPU_COUNT=8
export GLOBAL_BATCH_SIZE=192
export PER_DEVICE_BATCH_SIZE=24
export GRAD_ACCUMULATES=1
export MODE_BATCH_SIZES=24:24:24:12
export MODE_GRAD_ACCUMULATES=1:1:1:2
export FOUR_MODE_MIXED_TRAINING=1
export MAX_STEPS=4
export MODE_UPDATES_PER_EPOCH=4
export MODE_UPDATE_WEIGHTS=1:1:1:1
export MONO_DATASET_WEIGHTS=1:1
export MODE_SCHEDULE_SEED=1234
export SINGLE_FRAME_SOURCE_INDEX=0

export HY_MANIFEST="${MANIFEST_ROOT}/hy.jsonl"
export HY_ROOT_ALIASES="{\"hy_primary\":\"${HY_ROOT}\"}"
export LIBERO_MANIFEST="${MANIFEST_ROOT}/libero.jsonl"
export LIBERO_ROOT_ALIASES="{\"libero_primary\":\"${LIBERO_ROOT}\"}"
export UMI_MANIFEST="${MANIFEST_ROOT}/umi-canonical.jsonl"
export UMI_DATASET_ROOT="${UMI_ROOT}"
export UMI_RECTIFICATION_AUDIT_SHA256=f9ccd6464df57a6cc10b7dfae62b34a7a1dbca04f0e76ce512239eee624dfdef

export FOUNDATION_STEREO_BACKEND=las2_h
export LAS2_H_REPO="${EXTERNAL_ROOT}/LiteAnyStereo"
export LAS2_H_SOURCE_SHA=8c97bd4c4da3712c2ac60003a23201dfdb5935f4
export LAS2_H_CHECKPOINT="${TEACHER_ROOT}/las2-h/LAS2_H.pth"
export LAS2_H_CHECKPOINT_SHA256=758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4
export LAS2_H_VALID_ITERS=4
export LAS2_H_MAX_DISP=192
export FOUNDATION_STEREO_PAIR_MICROBATCH=48
export DA3_REPO="${EXTERNAL_ROOT}/depth-anything-3"
export DA3_SOURCE_SHA=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
export DA3_CHECKPOINT="${TEACHER_ROOT}/DA3-BASE"
export DA3_CHECKPOINT_SHA256=e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5
export ONLINE_GT_CACHE_ENABLED=0
export ONLINE_VAL_CHECK_INTERVAL_STEPS=4

export LEARNING_RATE=1e-4
export MIN_LEARNING_RATE=1e-4
export WARMUP_STEPS=20
export KL_WARMUP_STEPS=100
export RGB_WEIGHT=1.0
export RELATIVE_DEPTH_WEIGHT=1.0
export RELATIVE_GRADIENT_WEIGHT=0.1
export KL_WEIGHT=1e-6
export PERCEPTUAL_WEIGHT=1.0
export GAN_ENABLED=0
export IMAGE_GAN_WEIGHT=0
export VIDEO_GAN_WEIGHT=0
export GAN_FEAT_WEIGHT=0
export CHECKPOINT_EVERY_N_STEPS=4
export NUM_WORKERS=4
export PREFETCH_FACTOR=2
export DISABLE_WANDB=1
export DISABLE_MEDIA_LOGGING=1

bash scripts/stereo/train_stereo_vae.sh
