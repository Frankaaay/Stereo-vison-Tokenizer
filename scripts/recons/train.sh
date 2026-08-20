#!/usr/bin/env bash
set -euo pipefail

: "${STEREO_TRAIN_MANIFEST:?set a Manifest v3 training path}"
: "${STEREO_RGB_ROOT:?set the independent RGB cache root}"
: "${STEREO_GT_ROOT:?set the FoundationStereo GT root}"
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
: "${DISPARITY_WEIGHT:?set the calibrated disparity loss weight}"
: "${GRADIENT_WEIGHT:?set the calibrated gradient loss weight}"
: "${KL_WEIGHT:?set the calibrated KL loss weight}"
: "${PERCEPTUAL_WEIGHT:?set the calibrated LPIPS weight}"

EXPECTED_GLOBAL_BATCH_SIZE=$((GPU_COUNT * PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATES))
if [[ "${EXPECTED_GLOBAL_BATCH_SIZE}" -ne "${GLOBAL_BATCH_SIZE}" ]]; then
  echo "global batch mismatch: expected ${EXPECTED_GLOBAL_BATCH_SIZE}, configured ${GLOBAL_BATCH_SIZE}" >&2
  exit 2
fi

VALIDATION_ARGS=()
if [[ -n "${STEREO_VAL_MANIFEST:-}" ]]; then
  VALIDATION_ARGS+=(--stereo_val_manifest "${STEREO_VAL_MANIFEST}")
fi

WANDB_ARGS=()
if [[ "${DISABLE_WANDB:-0}" == "1" ]]; then
  WANDB_ARGS+=(--disable_wandb)
fi

python3 vqgan_train.py \
  --tokenizer omnitokenizer \
  --loader_type stereo_manifest \
  --stereo_train_manifest "${STEREO_TRAIN_MANIFEST}" \
  --stereo_rgb_root "${STEREO_RGB_ROOT}" \
  --stereo_gt_root "${STEREO_GT_ROOT}" \
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
  --codebook_dim 48 \
  --enc_block ttww \
  --dec_block tttt \
  --twod_window_size 8 \
  --spatial_pos rope \
  --causal_in_temporal_transformer \
  --causal_in_peg \
  --dim_head 64 \
  --heads 8 \
  --initialize_vit \
  --stereo_num_views 3 \
  --stereo_num_frames 4 \
  --stereo_search_radii 7 7 7 \
  --stereo_search_direction left \
  --stereo_disparity_scale 128 128 128 \
  --stereo_disparity_bias -2.572 \
  --stereo_disparity_epsilon 1e-6 \
  --stereo_mode stereo \
  --stereo_disparity_min_px 0.5 \
  --stereo_disparity_max_px 112.0 \
  --stereo_lr_error_abs_threshold_px 1.0 \
  --stereo_lr_error_relative_threshold 0.05 \
  --geometry_gradient_scale_px 16.0 \
  --rgb_weight "${RGB_WEIGHT}" \
  --disparity_weight "${DISPARITY_WEIGHT}" \
  --gradient_weight "${GRADIENT_WEIGHT}" \
  --kl_weight "${KL_WEIGHT}" \
  --perceptual_weight "${PERCEPTUAL_WEIGHT}" \
  --image_gan_weight 0 \
  --video_gan_weight 0 \
  --gan_feat_weight 0 \
  --recon_loss_type l1 \
  --smooth_l1_beta 1.0 \
  --batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --grad_accumulates "${GRAD_ACCUMULATES}" \
  --grad_clip_val 1.0 \
  --lr "${LEARNING_RATE}" \
  --lr_min "${MIN_LEARNING_RATE}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --kl_warmup_steps "${KL_WARMUP_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --default_root_dir "${OUTPUT_ROOT}" \
  --gpus "${GPU_COUNT}" \
  --bf16 \
  "${WANDB_ARGS[@]}"
