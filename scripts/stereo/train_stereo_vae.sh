#!/usr/bin/env bash
set -euo pipefail

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

MODE_UPDATE_RATIO="${MODE_UPDATE_RATIO:-1:1:1:1}"
FOUR_MODE_MIXED_TRAINING="${FOUR_MODE_MIXED_TRAINING:-0}"
if [[ "${PER_DEVICE_BATCH_SIZE}" != "24" || "${GRAD_ACCUMULATES}" != "1" ]]; then
  echo "four-mode training is frozen to per-device BS24 and GA1" >&2
  exit 2
fi
if [[ "${MODE_UPDATE_RATIO}" != "1:1:1:1" ]]; then
  echo "four-mode training requires MODE_UPDATE_RATIO=1:1:1:1" >&2
  exit 2
fi

STEREO_DATA_BACKEND="${STEREO_DATA_BACKEND:-manifest_v3}"
DATA_ARGS=(--stereo_data_backend "${STEREO_DATA_BACKEND}")
ONLINE_GT_ARGS=()
if [[ "${STEREO_DATA_BACKEND}" == "manifest_v3" ]]; then
  : "${STEREO_TRAIN_MANIFEST:?set a Manifest v3 training path}"
  : "${STEREO_RGB_ROOT:?set the independent RGB cache root}"
  : "${STEREO_GT_ROOT:?set the FoundationStereo GT root}"
  DATA_ARGS+=(
    --stereo_train_manifest "${STEREO_TRAIN_MANIFEST}"
    --stereo_rgb_root "${STEREO_RGB_ROOT}"
    --stereo_gt_root "${STEREO_GT_ROOT}"
  )
elif [[ "${STEREO_DATA_BACKEND}" == "lerobot_online" ]]; then
  : "${LEROBOT_EPISODE_MANIFEST:?set the episode-level LeRobot manifest}"
  : "${LEROBOT_DATASET_ROOT:?set the H1-local LeRobot root}"
  : "${LEROBOT_RECTIFICATION_AUDIT_SHA256:?set the rectification audit SHA256}"
  : "${FOUNDATION_STEREO_CHECKPOINT_SHA256:?set the checkpoint SHA256}"
  FOUNDATION_STEREO_BACKEND="${FOUNDATION_STEREO_BACKEND:-pytorch}"
  FOUNDATION_BACKEND_ARGS=()
  if [[ "${FOUNDATION_STEREO_BACKEND}" == "pytorch" ]]; then
    : "${FOUNDATION_STEREO_REPO:?set the FoundationStereo repository}"
    : "${FOUNDATION_STEREO_CHECKPOINT:?set the ViT-L checkpoint}"
    FOUNDATION_STEREO_VALID_ITERS="${FOUNDATION_STEREO_VALID_ITERS:-16}"
    FOUNDATION_BACKEND_ARGS+=(
      --foundation_stereo_repo "${FOUNDATION_STEREO_REPO}"
      --foundation_stereo_checkpoint "${FOUNDATION_STEREO_CHECKPOINT}"
    )
  elif [[ "${FOUNDATION_STEREO_BACKEND}" == "tensorrt" ]]; then
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
    )
  else
    echo "unsupported FOUNDATION_STEREO_BACKEND=${FOUNDATION_STEREO_BACKEND}" >&2
    exit 2
  fi
  DATA_ARGS+=(
    --lerobot_episode_manifest "${LEROBOT_EPISODE_MANIFEST}"
    --lerobot_dataset_root "${LEROBOT_DATASET_ROOT}"
    --lerobot_rectification_audit_sha256 "${LEROBOT_RECTIFICATION_AUDIT_SHA256}"
    --lerobot_video_cache_capacity "${LEROBOT_VIDEO_CACHE_CAPACITY:-12}"
    --lerobot_maximum_timestamp_error_s "${LEROBOT_MAXIMUM_TIMESTAMP_ERROR_S:-0.05}"
    --lerobot_val_sample_limit "${LEROBOT_VAL_SAMPLE_LIMIT:-512}"
  )
  ONLINE_GT_CACHE_ENABLED="${ONLINE_GT_CACHE_ENABLED:-0}"
  if [[ "${ONLINE_GT_CACHE_ENABLED}" == "1" ]]; then
    : "${ONLINE_GT_CACHE_ROOT:?set a repository-external online GT cache root}"
  fi
  ONLINE_GT_ARGS+=(
    --online_gt_enabled 1
    --foundation_stereo_backend "${FOUNDATION_STEREO_BACKEND}"
    --foundation_stereo_checkpoint_sha256 "${FOUNDATION_STEREO_CHECKPOINT_SHA256}"
    --foundation_stereo_valid_iters "${FOUNDATION_STEREO_VALID_ITERS}"
    --foundation_stereo_pair_microbatch "${FOUNDATION_STEREO_PAIR_MICROBATCH:-48}"
    --online_gt_cache_enabled "${ONLINE_GT_CACHE_ENABLED}"
    --online_gt_cache_root "${ONLINE_GT_CACHE_ROOT:-}"
    --online_val_check_interval_steps "${ONLINE_VAL_CHECK_INTERVAL_STEPS:-500}"
    "${FOUNDATION_BACKEND_ARGS[@]}"
  )
else
  echo "unsupported STEREO_DATA_BACKEND=${STEREO_DATA_BACKEND}" >&2
  exit 2
fi

EXPECTED_GLOBAL_BATCH_SIZE=$((GPU_COUNT * PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATES))
if [[ "${EXPECTED_GLOBAL_BATCH_SIZE}" -ne "${GLOBAL_BATCH_SIZE}" ]]; then
  echo "global batch mismatch: expected ${EXPECTED_GLOBAL_BATCH_SIZE}, configured ${GLOBAL_BATCH_SIZE}" >&2
  exit 2
fi

VALIDATION_ARGS=()
if [[ "${STEREO_DATA_BACKEND}" == "manifest_v3" && -n "${STEREO_VAL_MANIFEST:-}" ]]; then
  VALIDATION_ARGS+=(--stereo_val_manifest "${STEREO_VAL_MANIFEST}")
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
  if [[ "${STEREO_DATA_BACKEND}" != "lerobot_online" ]]; then
    echo "four-mode smoke requires STEREO_DATA_BACKEND=lerobot_online" >&2
    exit 2
  fi
  : "${MONO_SMOKE_MANIFEST:?set the node-local Hy mono manifest}"
  : "${MONO_SMOKE_CACHE_ROOT:?set the node-local Hy mono cache root}"
  : "${DA3_REPO:?set the pinned Depth Anything 3 source repository}"
  : "${DA3_SOURCE_SHA:?set the full Depth Anything 3 source SHA}"
  : "${DA3_CHECKPOINT:?set the pinned DA3-BASE checkpoint directory}"
  : "${DA3_CHECKPOINT_SHA256:?set the DA3-BASE model.safetensors SHA256}"
  MODE_UPDATES_PER_EPOCH="${MODE_UPDATES_PER_EPOCH:-${MAX_STEPS}}"
  if (( MODE_UPDATES_PER_EPOCH < MAX_STEPS || MODE_UPDATES_PER_EPOCH % 4 != 0 )); then
    echo "MODE_UPDATES_PER_EPOCH must cover MAX_STEPS and be divisible by four" >&2
    exit 2
  fi
  MIXED_MODE_ARGS+=(
    --four_mode_mixed_training 1
    --mono_train_manifest "${MONO_SMOKE_MANIFEST}"
    --mono_val_manifest "${MONO_SMOKE_MANIFEST}"
    --mono_cache_root "${MONO_SMOKE_CACHE_ROOT}"
    --mixed_stereo_sample_limit 48
    --mode_schedule_seed "${MODE_SCHEDULE_SEED:-1234}"
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
  )
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

python3 train_stereo_vae.py \
  "${DATA_ARGS[@]}" \
  "${ONLINE_GT_ARGS[@]}" \
  "${MIXED_MODE_ARGS[@]}" \
  "${VALIDATION_ARGS[@]}" \
  --resolution 256 \
  --sequence_length 4 \
  --image_channels 3 \
  --patch_embed linear \
  --patch_size 16 \
  --temporal_patch_size 4 \
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
  --stereo_mode stereo \
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
  --image_gan_weight 0 \
  --video_gan_weight 0 \
  --gan_feat_weight 0 \
  --recon_loss_type l1 \
  --smooth_l1_beta 1.0 \
  --batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS:-8}" \
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
  --bf16 \
  "${WANDB_ARGS[@]}" \
  "${MEDIA_ARGS[@]}" \
  "${TIMING_ARGS[@]}" \
  "${PROFILE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
