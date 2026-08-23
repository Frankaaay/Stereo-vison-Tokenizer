#!/usr/bin/env bash
set -euo pipefail

: "${STEREO_TRAIN_MANIFEST:?set the frozen selected-8 Manifest v3 path}"
: "${STEREO_RGB_ROOT:?set the independent RGB cache root}"
: "${STEREO_GT_ROOT:?set the FoundationStereo GT root}"
: "${OUTPUT_ROOT:?set a new repository-external output directory}"
: "${EXPECTED_GIT_SHA:?set the exact profiling branch SHA}"
: "${EXPECTED_MANIFEST_SHA256:?set the selected-8 manifest SHA256}"
: "${PHYSICAL_GPU:?set the physical H200 GPU index}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="${REPO_ROOT:-/data/home/frank/projects/Stereo-vison-Tokenizer}"
TORCH_HOME="${TORCH_HOME:-/home/frank/.cache/torch}"
PROFILE_PEG_BACKEND="${PROFILE_PEG_BACKEND:-conv3d_contiguous}"
PROFILE_DATASET_MODE="${PROFILE_DATASET_MODE:-selected8}"
PROFILE_NUM_WORKERS="${PROFILE_NUM_WORKERS:-0}"
PROFILE_PRELOAD_DATA="${PROFILE_PRELOAD_DATA:-0}"
PROFILE_PIN_MEMORY="${PROFILE_PIN_MEMORY:-0}"
PROFILE_LPIPS_GT_CACHE="${PROFILE_LPIPS_GT_CACHE:-0}"

test "$(git -C "${REPO_ROOT}" branch --show-current)" = frank-profiling
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_GIT_SHA}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain)"
test -x "${PYTHON_BIN}"
test -f "${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth"

if nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "GPU_PRECHECK_FAILED: GPU ${PHYSICAL_GPU} has a compute process" >&2
  exit 41
fi

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export TORCH_HOME

exec /usr/bin/time -v "${PYTHON_BIN}" profile_stereo_step.py \
  --seed 1234 \
  --bf16 \
  --disable_wandb \
  --devices 1 \
  --num_nodes 1 \
  --max_steps 5000 \
  --profile_updates 40 \
  --profile_wait 15 \
  --profile_warmup 5 \
  --profile_active 10 \
  --profile_dataset_mode "${PROFILE_DATASET_MODE}" \
  --profile_peg_backend "${PROFILE_PEG_BACKEND}" \
  --profile_preload_data "${PROFILE_PRELOAD_DATA}" \
  --profile_pin_memory "${PROFILE_PIN_MEMORY}" \
  --profile_lpips_gt_cache "${PROFILE_LPIPS_GT_CACHE}" \
  --default_root_dir "${OUTPUT_ROOT}" \
  --embedding_dim 512 \
  --lr 0.0001 \
  --lr_min 0.0001 \
  --warmup_steps 20 \
  --warmup_lr_init 0.0 \
  --image_gan_weight 0.0 \
  --video_gan_weight 0.0 \
  --gan_feat_weight 0.0 \
  --perceptual_weight 1.0 \
  --kl_weight 0.000001 \
  --kl_warmup_steps 100 \
  --initialize_vit \
  --norm_type group \
  --recon_loss_type l1 \
  --patch_size 16 \
  --patch_embed linear \
  --enc_block ttww \
  --dec_block tttt \
  --twod_window_size 8 \
  --temporal_patch_size 4 \
  --spatial_pos rope \
  --spatial_depth 4 \
  --temporal_depth 4 \
  --causal_in_temporal_transformer \
  --causal_in_peg \
  --dim_head 64 \
  --heads 8 \
  --attn_dropout 0.0 \
  --ff_dropout 0.0 \
  --ff_mult 4.0 \
  --latent_channels 48 \
  --stereo_num_views 3 \
  --stereo_num_frames 4 \
  --stereo_search_radii 7 7 7 \
  --stereo_search_direction left \
  --stereo_disparity_scale 128.0 128.0 128.0 \
  --stereo_disparity_bias -2.572 \
  --stereo_disparity_epsilon 0.000001 \
  --stereo_mode stereo \
  --rgb_weight 1.0 \
  --disparity_weight 1.0 \
  --gradient_weight 0.1 \
  --geometry_gradient_scale_px 16.0 \
  --smooth_l1_beta 1.0 \
  --grad_accumulates 1 \
  --grad_clip_val 1.0 \
  --sequence_length 4 \
  --resolution 256 \
  --batch_size 8 \
  --num_workers "${PROFILE_NUM_WORKERS}" \
  --image_channels 3 \
  --stereo_train_manifest "${STEREO_TRAIN_MANIFEST}" \
  --stereo_rgb_root "${STEREO_RGB_ROOT}" \
  --stereo_gt_root "${STEREO_GT_ROOT}" \
  --stereo_disparity_min_px 0.5 \
  --stereo_disparity_max_px 112.0 \
  --stereo_lr_error_abs_threshold_px 1.0 \
  --stereo_lr_error_relative_threshold 0.05 \
  --expected_git_sha "${EXPECTED_GIT_SHA}" \
  --expected_manifest_sha256 "${EXPECTED_MANIFEST_SHA256}"
