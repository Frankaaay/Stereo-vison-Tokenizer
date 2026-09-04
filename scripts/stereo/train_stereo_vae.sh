#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${OUTPUT_ROOT:?set a repository-external output directory}"
: "${GPU_COUNT:?set the number of visible GPUs}"
: "${GLOBAL_BATCH_SIZE:?set the intended global batch size}"
: "${PER_DEVICE_BATCH_SIZE:?set the per-GPU micro batch}"
: "${GRAD_ACCUMULATES:?set gradient accumulation}"
: "${MAX_STEPS:?set the smoke/overfit step budget}"
: "${LEARNING_RATE:?set the generator learning rate}"
: "${MIN_LEARNING_RATE:?set the cosine minimum learning rate}"
: "${WARMUP_STEPS:?set optimizer warmup steps}"
: "${KL_WARMUP_STEPS:?set KL warmup steps}"
: "${RGB_WEIGHT:?set the calibrated RGB loss weight}"
: "${RELATIVE_DEPTH_WEIGHT:?set the existing geometry weight for relative depth}"
: "${RELATIVE_GRADIENT_WEIGHT:?set the existing geometry weight for relative gradient}"
: "${KL_WEIGHT:?set the calibrated KL loss weight}"
: "${PERCEPTUAL_WEIGHT:?set the calibrated LPIPS weight}"
: "${SINGLE_FRAME_SOURCE_INDEX:?set the current source frame index}"

DISTRIBUTED_MODE="${DISTRIBUTED_MODE:-single}"
NUM_NODES="${NUM_NODES:-1}"
case "${DISTRIBUTED_MODE}" in
  single)
    if [[ "${NUM_NODES}" != "1" ]]; then
      echo "single distributed mode requires NUM_NODES=1" >&2
      exit 2
    fi
    ;;
  ib)
    if [[ "${NUM_NODES}" != "2" ]]; then
      echo "ib distributed mode requires NUM_NODES=2" >&2
      exit 2
    fi
    : "${NODE_RANK:?set NODE_RANK=0 on h200-1 and NODE_RANK=1 on h200-2}"
    : "${MASTER_ADDR:?set the h200-1 bond0 address}"
    : "${MASTER_PORT:?set a unique rendezvous port}"
    if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
      echo "ib distributed mode requires NODE_RANK=0 or 1" >&2
      exit 2
    fi
    if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]]; then
      echo "MASTER_PORT must be numeric" >&2
      exit 2
    fi
    ;;
  *)
    echo "DISTRIBUTED_MODE must be single or ib" >&2
    exit 2
    ;;
esac

GAN_ENABLED="${GAN_ENABLED:-0}"
IMAGE_GAN_WEIGHT="${IMAGE_GAN_WEIGHT:-0}"
VIDEO_GAN_WEIGHT="${VIDEO_GAN_WEIGHT:-0}"
GAN_FEAT_WEIGHT="${GAN_FEAT_WEIGHT:-0}"
DISCRIMINATOR_ITER_START="${DISCRIMINATOR_ITER_START:-50000}"
GAN_ARGS=()
if [[ "${GAN_ENABLED}" == "1" ]]; then
  GAN_ARGS+=(--gan_enabled)
elif [[ "${GAN_ENABLED}" != "0" ]]; then
  echo "GAN_ENABLED must be 0 or 1" >&2
  exit 2
fi

MODE_UPDATE_WEIGHTS="${MODE_UPDATE_WEIGHTS:-35:35:15:15}"
MONO_DATASET_WEIGHTS="${MONO_DATASET_WEIGHTS:-9:1}"
FOUR_MODE_MIXED_TRAINING="${FOUR_MODE_MIXED_TRAINING:-0}"
MODE_BATCH_SIZES="${MODE_BATCH_SIZES:-${PER_DEVICE_BATCH_SIZE}:${PER_DEVICE_BATCH_SIZE}:${PER_DEVICE_BATCH_SIZE}:${PER_DEVICE_BATCH_SIZE}}"
MODE_GRAD_ACCUMULATES="${MODE_GRAD_ACCUMULATES:-${GRAD_ACCUMULATES}:${GRAD_ACCUMULATES}:${GRAD_ACCUMULATES}:${GRAD_ACCUMULATES}}"
if [[ "${FOUR_MODE_MIXED_TRAINING}" == "1" ]]; then
  if [[ "${GRAD_ACCUMULATES}" != "1" ]]; then
    echo "four-mode training keeps GRAD_ACCUMULATES=1; use MODE_GRAD_ACCUMULATES" >&2
    exit 2
  fi
  IFS=: read -r -a MODE_BATCH_SIZE_VALUES <<< "${MODE_BATCH_SIZES}"
  IFS=: read -r -a MODE_GRAD_ACCUMULATE_VALUES <<< "${MODE_GRAD_ACCUMULATES}"
  if [[ "${#MODE_BATCH_SIZE_VALUES[@]}" -ne 4 || "${#MODE_GRAD_ACCUMULATE_VALUES[@]}" -ne 4 ]]; then
    echo "MODE_BATCH_SIZES and MODE_GRAD_ACCUMULATES require four colon-separated values" >&2
    exit 2
  fi
  for value in "${MODE_BATCH_SIZE_VALUES[@]}" "${MODE_GRAD_ACCUMULATE_VALUES[@]}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
      echo "per-mode batch and accumulation values must be positive integers" >&2
      exit 2
    fi
  done
  if [[ "${MODE_GRAD_ACCUMULATE_VALUES[0]}" != "1" || "${MODE_GRAD_ACCUMULATE_VALUES[1]}" != "1" ]]; then
    echo "mono modes currently require accumulation factor 1" >&2
    exit 2
  fi
fi

DATA_ARGS=()
ONLINE_GT_ARGS=()
if [[ "${FOUR_MODE_MIXED_TRAINING}" != "1" ]]; then
  : "${LEROBOT_EPISODE_MANIFEST:?set the episode-level LeRobot manifest}"
  : "${LEROBOT_DATASET_ROOT:?set the H1-local LeRobot root}"
  : "${LEROBOT_RECTIFICATION_AUDIT_SHA256:?set the rectification audit SHA256}"
fi
FOUNDATION_STEREO_BACKEND="${FOUNDATION_STEREO_BACKEND:-pytorch}"
FOUNDATION_BACKEND_ARGS=()
if [[ "${FOUNDATION_STEREO_BACKEND}" == "las2_h" ]]; then
  : "${LAS2_H_REPO:?set the LiteAnyStereo repository}"
  : "${LAS2_H_SOURCE_SHA:?set the full LiteAnyStereo source Git SHA}"
  : "${LAS2_H_CHECKPOINT:?set the LAS2-H checkpoint}"
  : "${LAS2_H_CHECKPOINT_SHA256:?set the LAS2-H checkpoint SHA256}"
  LAS2_H_VALID_ITERS="${LAS2_H_VALID_ITERS:-4}"
  LAS2_H_MAX_DISP="${LAS2_H_MAX_DISP:-192}"
  FOUNDATION_BACKEND_ARGS=(
    --las2_h_repo "${LAS2_H_REPO}"
    --las2_h_source_sha "${LAS2_H_SOURCE_SHA}"
    --las2_h_checkpoint "${LAS2_H_CHECKPOINT}"
    --las2_h_checkpoint_sha256 "${LAS2_H_CHECKPOINT_SHA256}"
    --las2_h_valid_iters "${LAS2_H_VALID_ITERS}"
    --las2_h_max_disp "${LAS2_H_MAX_DISP}"
  )
elif [[ "${FOUNDATION_STEREO_BACKEND}" == "pytorch" ]]; then
  : "${FOUNDATION_STEREO_CHECKPOINT_SHA256:?set the checkpoint SHA256}"
  : "${FOUNDATION_STEREO_REPO:?set the FoundationStereo repository}"
  : "${FOUNDATION_STEREO_CHECKPOINT:?set the ViT-L checkpoint}"
  FOUNDATION_STEREO_VALID_ITERS="${FOUNDATION_STEREO_VALID_ITERS:-16}"
  FOUNDATION_BACKEND_ARGS+=(
    --foundation_stereo_repo "${FOUNDATION_STEREO_REPO}"
    --foundation_stereo_checkpoint "${FOUNDATION_STEREO_CHECKPOINT}"
    --foundation_stereo_checkpoint_sha256 "${FOUNDATION_STEREO_CHECKPOINT_SHA256}"
    --foundation_stereo_valid_iters "${FOUNDATION_STEREO_VALID_ITERS}"
  )
elif [[ "${FOUNDATION_STEREO_BACKEND}" == "tensorrt" ]]; then
  : "${FOUNDATION_STEREO_CHECKPOINT_SHA256:?set the checkpoint SHA256}"
  : "${FOUNDATION_STEREO_ENGINE:?set the TensorRT engine path}"
  : "${FOUNDATION_STEREO_ENGINE_SHA256:?set the TensorRT engine SHA256}"
  : "${FOUNDATION_STEREO_ENGINE_MANIFEST:?set the engine manifest path}"
  : "${FOUNDATION_STEREO_ENGINE_MANIFEST_SHA256:?set the engine manifest SHA256}"
  FOUNDATION_STEREO_VALID_ITERS="${FOUNDATION_STEREO_VALID_ITERS:-32}"
  if [[ "${FOUNDATION_STEREO_VALID_ITERS}" != "32" ]]; then
    echo "TensorRT FoundationStereo is frozen to 32 iterations" >&2
    exit 2
  fi
  FOUNDATION_BACKEND_ARGS+=(
    --foundation_stereo_engine "${FOUNDATION_STEREO_ENGINE}"
    --foundation_stereo_engine_sha256 "${FOUNDATION_STEREO_ENGINE_SHA256}"
    --foundation_stereo_engine_manifest "${FOUNDATION_STEREO_ENGINE_MANIFEST}"
    --foundation_stereo_engine_manifest_sha256 "${FOUNDATION_STEREO_ENGINE_MANIFEST_SHA256}"
    --foundation_stereo_checkpoint_sha256 "${FOUNDATION_STEREO_CHECKPOINT_SHA256}"
    --foundation_stereo_valid_iters "${FOUNDATION_STEREO_VALID_ITERS}"
  )
else
  echo "unsupported FOUNDATION_STEREO_BACKEND=${FOUNDATION_STEREO_BACKEND}" >&2
  exit 2
fi
if [[ "${FOUR_MODE_MIXED_TRAINING}" != "1" ]]; then
  DATA_ARGS+=(
    --lerobot_episode_manifest "${LEROBOT_EPISODE_MANIFEST}"
    --lerobot_dataset_root "${LEROBOT_DATASET_ROOT}"
    --lerobot_rectification_audit_sha256 "${LEROBOT_RECTIFICATION_AUDIT_SHA256}"
    --lerobot_video_cache_capacity "${LEROBOT_VIDEO_CACHE_CAPACITY:-12}"
    --lerobot_maximum_timestamp_error_s "${LEROBOT_MAXIMUM_TIMESTAMP_ERROR_S:-0.05}"
    --lerobot_val_sample_limit "${LEROBOT_VAL_SAMPLE_LIMIT:-512}"
  )
fi
ONLINE_GT_CACHE_ENABLED="${ONLINE_GT_CACHE_ENABLED:-0}"
if [[ "${ONLINE_GT_CACHE_ENABLED}" == "1" ]]; then
  : "${ONLINE_GT_CACHE_ROOT:?set a repository-external online GT cache root}"
fi
ONLINE_GT_ARGS+=(
  --online_gt_enabled 1
  --foundation_stereo_backend "${FOUNDATION_STEREO_BACKEND}"
  --foundation_stereo_pair_microbatch "${FOUNDATION_STEREO_PAIR_MICROBATCH:-48}"
  "${FOUNDATION_BACKEND_ARGS[@]}"
  --online_gt_cache_enabled "${ONLINE_GT_CACHE_ENABLED}"
  --online_gt_cache_root "${ONLINE_GT_CACHE_ROOT:-}"
  --online_val_check_interval_steps "${ONLINE_VAL_CHECK_INTERVAL_STEPS:-500}"
)
WORLD_SIZE=$((NUM_NODES * GPU_COUNT))
if [[ "${FOUR_MODE_MIXED_TRAINING}" == "1" ]]; then
  for index in 0 1 2 3; do
    EXPECTED_MODE_GLOBAL_BATCH_SIZE=$((WORLD_SIZE * MODE_BATCH_SIZE_VALUES[index] * MODE_GRAD_ACCUMULATE_VALUES[index]))
    if [[ "${EXPECTED_MODE_GLOBAL_BATCH_SIZE}" -ne "${GLOBAL_BATCH_SIZE}" ]]; then
      echo "mode ${index} global batch mismatch: expected ${EXPECTED_MODE_GLOBAL_BATCH_SIZE}, configured ${GLOBAL_BATCH_SIZE}" >&2
      exit 2
    fi
  done
else
  EXPECTED_GLOBAL_BATCH_SIZE=$((WORLD_SIZE * PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATES))
  if [[ "${EXPECTED_GLOBAL_BATCH_SIZE}" -ne "${GLOBAL_BATCH_SIZE}" ]]; then
    echo "global batch mismatch: expected ${EXPECTED_GLOBAL_BATCH_SIZE}, configured ${GLOBAL_BATCH_SIZE}" >&2
    exit 2
  fi
fi

WANDB_ARGS=()
if [[ "${DISABLE_WANDB:-0}" == "1" ]]; then
  WANDB_ARGS+=(--disable_wandb)
fi

MEDIA_ARGS=()
if [[ "${DISABLE_MEDIA_LOGGING:-0}" == "1" ]]; then
  MEDIA_ARGS+=(--disable_media_logging)
fi

TIMING_ARGS=()
if [[ -n "${STEP_TIMING_OUTPUT:-}" ]]; then
  TIMING_ARGS+=(
    --step_timing_output "${STEP_TIMING_OUTPUT}"
    --step_timing_warmup "${STEP_TIMING_WARMUP:-5}"
  )
fi

MIXED_MODE_ARGS=()
if [[ "${FOUR_MODE_MIXED_TRAINING}" == "1" ]]; then
  : "${HY_MANIFEST:?set the node-local Hy manifest}"
  : "${HY_ROOT_ALIASES:?set the node-local Hy root-alias JSON}"
  : "${LIBERO_MANIFEST:?set the node-local LIBERO manifest}"
  : "${LIBERO_ROOT_ALIASES:?set the node-local LIBERO root-alias JSON}"
  : "${UMI_MANIFEST:?set the node-local UMI manifest}"
  : "${UMI_DATASET_ROOT:?set the node-local UMI LeRobot dataset root}"
  : "${UMI_RECTIFICATION_AUDIT_SHA256:?set the UMI rectification audit SHA256}"
  : "${DA3_REPO:?set the pinned Depth Anything 3 source repository}"
  : "${DA3_SOURCE_SHA:?set the full Depth Anything 3 source SHA}"
  : "${DA3_CHECKPOINT:?set the pinned DA3-BASE checkpoint directory}"
  : "${DA3_CHECKPOINT_SHA256:?set the DA3-BASE model.safetensors SHA256}"
  MODE_UPDATES_PER_EPOCH="${MODE_UPDATES_PER_EPOCH:-${MAX_STEPS}}"
  if (( MODE_UPDATES_PER_EPOCH < MAX_STEPS )); then
    echo "MODE_UPDATES_PER_EPOCH must cover MAX_STEPS" >&2
    exit 2
  fi
  if [[ "${NUM_NODES}" == "2" && -z "${NODE_MANIFEST_CONTRACTS:-}" ]]; then
    echo "dual-node training requires NODE_MANIFEST_CONTRACTS" >&2
    exit 2
  fi
  MIXED_MODE_ARGS+=(
    --four_mode_mixed_training 1
    --hy_manifest "${HY_MANIFEST}"
    --hy_root_aliases "${HY_ROOT_ALIASES}"
    --libero_manifest "${LIBERO_MANIFEST}"
    --libero_root_aliases "${LIBERO_ROOT_ALIASES}"
    --umi_manifest "${UMI_MANIFEST}"
    --umi_dataset_root "${UMI_DATASET_ROOT}"
    --umi_rectification_audit_sha256 "${UMI_RECTIFICATION_AUDIT_SHA256}"
    --mode_schedule_seed "${MODE_SCHEDULE_SEED:-1234}"
    --mode_update_weights "${MODE_UPDATE_WEIGHTS}"
    --mode_batch_sizes "${MODE_BATCH_SIZES}"
    --mode_grad_accumulates "${MODE_GRAD_ACCUMULATES}"
    --mono_dataset_weights "${MONO_DATASET_WEIGHTS}"
    --node_manifest_contracts "${NODE_MANIFEST_CONTRACTS:-}"
    --mode_updates_per_epoch "${MODE_UPDATES_PER_EPOCH}"
    --da3_repo "${DA3_REPO}"
    --da3_source_sha "${DA3_SOURCE_SHA}"
    --da3_checkpoint "${DA3_CHECKPOINT}"
    --da3_checkpoint_sha256 "${DA3_CHECKPOINT_SHA256}"
    --da3_process_res 504
    --da3_process_res_method upper_bound_resize
    --da3_confidence_mask_mode finite_positive_non_padding
  )
elif [[ "${FOUR_MODE_MIXED_TRAINING}" != "0" ]]; then
  echo "FOUR_MODE_MIXED_TRAINING must be 0 or 1" >&2
  exit 2
fi

PROFILE_ARGS=()
if [[ -n "${TORCH_PROFILE_OUTPUT_DIR:-}" ]]; then
  PROFILE_ARGS+=(
    --torch_profile_output_dir "${TORCH_PROFILE_OUTPUT_DIR}"
    --torch_profile_wait "${TORCH_PROFILE_WAIT:-5}"
    --torch_profile_warmup "${TORCH_PROFILE_WARMUP:-2}"
    --torch_profile_active "${TORCH_PROFILE_ACTIVE:-4}"
    --torch_profile_with_stack "${TORCH_PROFILE_WITH_STACK:-0}"
  )
fi

RESUME_ARGS=()
CHECKPOINT_ARG_COUNT=0
for checkpoint_value in \
  "${RESUME_FROM_CHECKPOINT:-}" \
  "${CONTINUATION_CHECKPOINT:-}" \
  "${STAGE_TRANSITION_CHECKPOINT:-}" \
  "${DISCRIMINATOR_EXPANSION_CHECKPOINT:-}"; do
  if [[ -n "${checkpoint_value}" ]]; then
    CHECKPOINT_ARG_COUNT=$((CHECKPOINT_ARG_COUNT + 1))
  fi
done
if (( CHECKPOINT_ARG_COUNT > 1 )); then
  echo "checkpoint inputs are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${CONTINUATION_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--continuation_checkpoint "${CONTINUATION_CHECKPOINT}")
fi
if [[ -n "${STAGE_TRANSITION_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--stage_transition_checkpoint "${STAGE_TRANSITION_CHECKPOINT}")
fi
if [[ -n "${DISCRIMINATOR_EXPANSION_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--discriminator_expansion_checkpoint "${DISCRIMINATOR_EXPANSION_CHECKPOINT}")
fi

TRAIN_LAUNCHER=(python3)
if [[ "${DISTRIBUTED_MODE}" == "single" && "${GPU_COUNT}" -gt 1 ]]; then
  TRAIN_LAUNCHER=(
    torchrun
    --standalone
    --nnodes 1
    --nproc_per_node "${GPU_COUNT}"
  )
elif [[ "${DISTRIBUTED_MODE}" == "ib" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
  export NODE_RANK MASTER_ADDR MASTER_PORT
  export NCCL_IB_DISABLE=0
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=bond0}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
  export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-${OUTPUT_ROOT}/nccl-%h-%p.log}"
  TRAIN_LAUNCHER=(
    torchrun
    --nnodes "${NUM_NODES}"
    --nproc_per_node "${GPU_COUNT}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
  )
fi

"${TRAIN_LAUNCHER[@]}" train_stereo_vae.py \
  "${DATA_ARGS[@]}" \
  "${ONLINE_GT_ARGS[@]}" \
  "${MIXED_MODE_ARGS[@]}" \
  --resolution 256 \
  --sequence_length 4 \
  --image_channels 3 \
  --patch_embed linear \
  --patch_size 16 \
  --spatial_depth 4 \
  --temporal_depth 4 \
  --embedding_dim 512 \
  --latent_channels 48 \
  --enc_block ttww \
  --dec_block tttt \
  --twod_window_size 8 \
  --spatial_pos rope \
  --causal_in_peg \
  --peg_backend conv2d_t1_slice \
  --dim_head 64 \
  --heads 8 \
  --initialize_vit \
  --stereo_num_views 3 \
  --stereo_num_frames 4 \
  --single_frame_source_index "${SINGLE_FRAME_SOURCE_INDEX}" \
  --stereo_search_radii 7 7 7 \
  --stereo_search_direction left \
  --stereo_disparity_min_px 0.5 \
  --stereo_disparity_max_px 112.0 \
  --stereo_lr_error_abs_threshold_px 1.0 \
  --stereo_lr_error_relative_threshold 0.05 \
  --rgb_weight "${RGB_WEIGHT}" \
  --relative_depth_weight "${RELATIVE_DEPTH_WEIGHT}" \
  --relative_gradient_weight "${RELATIVE_GRADIENT_WEIGHT}" \
  --relative_depth_epsilon 1e-6 \
  --kl_weight "${KL_WEIGHT}" \
  --perceptual_weight "${PERCEPTUAL_WEIGHT}" \
  --accumulation_no_sync "${ACCUMULATION_NO_SYNC:-1}" \
  --image_gan_weight "${IMAGE_GAN_WEIGHT}" \
  --video_gan_weight "${VIDEO_GAN_WEIGHT}" \
  --gan_feat_weight "${GAN_FEAT_WEIGHT}" \
  --discriminator_iter_start "${DISCRIMINATOR_ITER_START}" \
  --recon_loss_type l1 \
  --smooth_l1_beta 1.0 \
  --batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --prefetch_factor "${PREFETCH_FACTOR:-2}" \
  --pin_memory 1 \
  --persistent_workers 1 \
  --train_epoch_repeats "${TRAIN_EPOCH_REPEATS:-1}" \
  --grad_accumulates "${GRAD_ACCUMULATES}" \
  --grad_clip_val 1.0 \
  --lr "${LEARNING_RATE}" \
  --lr_min "${MIN_LEARNING_RATE}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --kl_warmup_steps "${KL_WARMUP_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --checkpoint_every_n_steps "${CHECKPOINT_EVERY_N_STEPS:-500}" \
  --default_root_dir "${OUTPUT_ROOT}" \
  --devices "${GPU_COUNT}" \
  --num_nodes "${NUM_NODES}" \
  --distributed_mode "${DISTRIBUTED_MODE}" \
  --bf16 \
  "${GAN_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${MEDIA_ARGS[@]}" \
  "${TIMING_ARGS[@]}" \
  "${PROFILE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

if [[ "${DISTRIBUTED_MODE}" == "ib" ]]; then
  shopt -s nullglob
  NCCL_LOGS=("${OUTPUT_ROOT}"/nccl-*.log)
  if (( ${#NCCL_LOGS[@]} == 0 )); then
    echo "ib run produced no NCCL debug logs" >&2
    exit 3
  fi
  if ! grep -q "NET/IB" "${NCCL_LOGS[@]}"; then
    echo "NCCL logs do not prove NET/IB transport; refusing success" >&2
    exit 3
  fi
fi
