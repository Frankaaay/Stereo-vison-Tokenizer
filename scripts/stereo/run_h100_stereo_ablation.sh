#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "H100 evaluation must run inside a Slurm allocation" >&2
  exit 2
fi

: "${REPO_ROOT:?set the clean H100 repository path}"
: "${EXPECTED_GIT_SHA:?set the exact committed evaluation SHA}"
: "${RUN_DIR:?set a new repository-external run directory}"
: "${PROFILE:?set smoke, diagnostic, or main}"
: "${GPU_COUNT:?set the number of Slurm GPUs}"
: "${TOKENIZER_PYTHON:?set the tokenizer environment Python}"
: "${STEREO_VAE_CHECKPOINT:?set the Stage A checkpoint path}"
: "${STEREO_VAE_CHECKPOINT_SHA256:?set the Stage A checkpoint SHA256}"
: "${CANONICAL_V3_ROOT:?set the canonical-v3 dataset root}"
: "${CANONICAL_V3_MANIFEST:?set the generated canonical-v3 manifest}"
: "${DATA_GATE_JSON:?set the passing data-gate JSON}"
: "${TEACHER_CACHE_DIR:?set the shared immutable teacher-cache directory}"
: "${LAS2_H_REPO:?set the clean pinned LAS2-H Git clone}"
: "${LAS2_H_SOURCE_SHA:?set the full LAS2-H source SHA}"
: "${LAS2_H_CHECKPOINT:?set the LAS2-H checkpoint}"
: "${LAS2_H_CHECKPOINT_SHA256:?set the LAS2-H checkpoint SHA256}"

case "${PROFILE}" in
  smoke)
    EPISODE_COUNT=8
    WINDOWS_PER_EPISODE=2
    NUM_VISUALIZATIONS="${NUM_VISUALIZATIONS:-2}"
    ;;
  diagnostic)
    EPISODE_COUNT=64
    WINDOWS_PER_EPISODE=4
    NUM_VISUALIZATIONS="${NUM_VISUALIZATIONS:-8}"
    ;;
  main)
    EPISODE_COUNT=128
    WINDOWS_PER_EPISODE=8
    NUM_VISUALIZATIONS="${NUM_VISUALIZATIONS:-24}"
    ;;
  *)
    echo "PROFILE must be smoke, diagnostic, or main" >&2
    exit 2
    ;;
esac

if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing to overwrite existing RUN_DIR=${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

cd "${REPO_ROOT}"
CURRENT_SHA="$(git rev-parse HEAD)"
if [[ "${CURRENT_SHA}" != "${EXPECTED_GIT_SHA}" ]]; then
  echo "repository SHA mismatch: ${CURRENT_SHA} != ${EXPECTED_GIT_SHA}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "H100 repository is dirty" >&2
  exit 2
fi
if [[ "$(sha256sum "${STEREO_VAE_CHECKPOINT}" | awk '{print $1}')" != "${STEREO_VAE_CHECKPOINT_SHA256}" ]]; then
  echo "Stage A checkpoint SHA256 mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "${LAS2_H_CHECKPOINT}" | awk '{print $1}')" != "${LAS2_H_CHECKPOINT_SHA256}" ]]; then
  echo "LAS2-H checkpoint SHA256 mismatch" >&2
  exit 2
fi
DATA_GATE_SHA256="$(sha256sum "${DATA_GATE_JSON}" | awk '{print $1}')"
TORCHRUN="$(dirname "${TOKENIZER_PYTHON}")/torchrun"
if [[ ! -x "${TOKENIZER_PYTHON}" || ! -x "${TORCHRUN}" ]]; then
  echo "tokenizer Python/torchrun is unavailable" >&2
  exit 2
fi

env | sort > "${RUN_DIR}/environment.txt"
git status --short --branch > "${RUN_DIR}/git-status.txt"
nvidia-smi -L > "${RUN_DIR}/gpu-inventory.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader \
  > "${RUN_DIR}/gpu-details.csv"

"${TORCHRUN}" --standalone --nproc_per_node "${GPU_COUNT}" \
  eval_stereo_vae.py \
  --stereo_vae_ckpt "${STEREO_VAE_CHECKPOINT}" \
  --eval_split test \
  --device cuda \
  --output_json "${RUN_DIR}/evaluation.json" \
  --bf16 \
  --eval_eye_mode stereo \
  --eval_temporal_mode both \
  --ablation-condition real_stereo copy_left fusion_off wrong_right shift_right time_reverse \
  --right-shift-px -32 -16 16 32 \
  --teacher-cache-dir "${TEACHER_CACHE_DIR}" \
  --paired-output-dir "${RUN_DIR}/paired" \
  --report-dir "${RUN_DIR}/report" \
  --visualization_dir "${RUN_DIR}/visualizations" \
  --num_visualizations "${NUM_VISUALIZATIONS}" \
  --canonical-v3-manifest "${CANONICAL_V3_MANIFEST}" \
  --canonical-v3-root "${CANONICAL_V3_ROOT}" \
  --canonical-v3-pixel-mask "${CANONICAL_V3_ROOT}/image_pixel_mask_umi.npz" \
  --data-gate-json "${DATA_GATE_JSON}" \
  --ablation-episode-count "${EPISODE_COUNT}" \
  --ablation-windows-per-episode "${WINDOWS_PER_EPISODE}" \
  --bootstrap-iterations 10000 \
  --foundation_stereo_backend las2_h \
  --foundation_stereo_pair_microbatch "${TEACHER_PAIR_MICROBATCH:-12}" \
  --las2_h_repo "${LAS2_H_REPO}" \
  --las2_h_source_sha "${LAS2_H_SOURCE_SHA}" \
  --las2_h_checkpoint "${LAS2_H_CHECKPOINT}" \
  --las2_h_checkpoint_sha256 "${LAS2_H_CHECKPOINT_SHA256}" \
  --las2_h_valid_iters 4 \
  --las2_h_max_disp 192 \
  --lerobot_rectification_audit_sha256 "${DATA_GATE_SHA256}" \
  --lerobot_video_cache_capacity "${VIDEO_CACHE_CAPACITY:-12}" \
  --lerobot_maximum_timestamp_error_s 0.05 \
  --stereo_disparity_min_px 0.5 \
  --stereo_disparity_max_px 112.0 \
  --stereo_lr_error_abs_threshold_px 1.0 \
  --stereo_lr_error_relative_threshold 0.05 \
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
  --stereo_num_views 3 \
  --stereo_num_frames 4 \
  --single_frame_source_index 0 \
  --stereo_search_radii 7 7 7 \
  --stereo_search_direction left \
  --rgb_weight 1.0 \
  --relative_depth_weight 1.0 \
  --relative_gradient_weight 0.1 \
  --relative_depth_epsilon 1e-6 \
  --kl_weight 1e-6 \
  --perceptual_weight 1.0 \
  --image_gan_weight 0.0 \
  --video_gan_weight 0.0 \
  --gan_feat_weight 0.0 \
  --recon_loss_type l1 \
  --smooth_l1_beta 1.0 \
  --batch_size "${EVAL_BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-2}" \
  --pin_memory 1 \
  --persistent_workers 1 \
  --metric_frame_microbatch "${METRIC_FRAME_MICROBATCH:-12}"
