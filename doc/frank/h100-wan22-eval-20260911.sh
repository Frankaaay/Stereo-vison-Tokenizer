#!/usr/bin/env bash
set -euo pipefail
ROOT=/gpfs/jiuquyun/projects/Frank/stereo-vae
REPO=$ROOT/Stereo-vison-Tokenizer
PY=$ROOT/runtime/wan22-eval-20260911/bin/python
OUT=$ROOT/outputs/wan22-ganoff124k-20260911-v3
OLD=/gpfs/jiuquyun/projects/hezhou/experiments/stereo-tokenizer-stage-a
ASSETS=/gpfs/jiuquyun/checkpoints/Frank/stereo-vae
export TORCH_HOME=$ASSETS/runtime-assets/torch
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
cd "$REPO"
phase=${1:?prepare or smoke or full}
mkdir -p "$OUT"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/$phase.exit_code.txt"' EXIT
if [[ "$phase" == prepare ]]; then
    "$PY" -m pytest tests/evaluation/test_stage_a_evaluation.py tests/evaluation/test_wan_evaluation.py -q -p no:cacheprovider
    exit 0
fi
[[ "$phase" == smoke || "$phase" == full ]]
mkdir -p "$OUT/$phase"
extra=()
if [[ "$phase" == smoke ]]; then extra=(--max_batches 1); fi
run_cell() {
    local dataset=$1 eye=$2 name=$3 selection=$4
    shift 4
    "$PY" -u -m evaluation.wan.run \
      --stage-a-dataset-id "$dataset" --eval_eye_mode "$eye" \
      --stage-a-selection "$selection" \
      --canonical-loader-root /gpfs/jiuquyun/projects/Frank/NGADv1pp-pr17/ngad/datasets/ngad-canonical-dataloader \
      --stereo_vae_ckpt /gpfs/jiuquyun/projects/Frank/stereo-tokenizer-checkpoints/v1/stagea-threeview-update124000/last.ckpt \
      --checkpoint-sha256 605be7940202b7f0aff2380ac4a99dfe1e15d20847ab0377d3e7b2a27352f7b7 \
      --wan-source-root "$ROOT/external/Wan2.2-42bf4cf" \
      --wan-checkpoint "$ASSETS/wan22-ti2v-5b/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth" \
      --raft-checkpoint "$TORCH_HOME/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
      --raft-checkpoint-sha256 ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322 \
      --rgb-only --num_visualizations 0 --batch_size 1 --num_workers 0 \
      --eval_temporal_mode both --single_frame_source_indices 0 1 2 3 \
      --output_json "$OUT/$phase/$name.json" "${extra[@]}" "$@"
}
for cam in head_left head_right left_wrist_left left_wrist_right right_wrist_left right_wrist_right; do
    run_cell umi mono "umi-$cam" "$OLD/20260902-stagec-update162500-baseline-v1/selections/umi-canonical-test-1024-seed1234.json" --stage-a-camera-key "observation.images.cam_$cam"
done
run_cell umi stereo umi-stereo "$OLD/20260902-stagec-update162500-baseline-v1/selections/umi-canonical-test-1024-seed1234.json"
for cam in head_left left_wrist_left; do
    run_cell libero mono "libero-$cam" "$ROOT/outputs/wan22-ganoff124k-20260911-v2/libero-original256-action-false.json" --stage-a-camera-key "observation.images.cam_$cam"
done
run_cell hy mono hy-threeview "$ROOT/outputs/wan22-ganoff124k-20260911-v1/hy-all-test-excluding-table014.json" \
  --hy_root_aliases '{"hy_primary":"/gpfs/jiuquyun/datasets/PRETRAIN_DATA/Hy-Embodied-0.5-VLA-Data"}'
