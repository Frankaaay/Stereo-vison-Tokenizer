#!/usr/bin/env bash
set -euo pipefail
cd /data/home/frank/projects/Stereo-vison-Tokenizer
source /data/home/frank/runtime/stereo-tokenizer-unified-v1/bin/activate
export PYTHONPATH=/data/home/frank/projects/Stereo-vison-Tokenizer:/data/home/frank/runtime/hy-lance-export-v1/lib/python3.12/site-packages:/data/home/frank/runtime/stereo-tokenizer-wandb-overlay-v1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=/data/home/frank/runtime/lite-any-stereo-8c97bd4-clean
root=/data/home/frank/experiments/stereo-latent-eval-h2002-20260908-v1
train_root=/data/home/frank/experiments/stereo-latent-ablation-permode-h2002-20260904-v16
test ! -e "$root"
mkdir -p "$root"
trap 'rc=$?; printf "%s\n" "$rc" > "$root/exit_code.txt"' EXIT
for phase in smoke full; do
  for dataset in libero umi hy; do
    extra=()
    if [[ "$phase" == smoke ]]; then extra=(--max-batches 1); fi
    torchrun --standalone --nnodes 1 --nproc_per_node 8 doc/frank/h2002-latent-eval-20260908.py \
      --run-root "$train_root" --output "$root/$phase-$dataset.json" \
      --dataset "$dataset" --batch-size 4 "${extra[@]}" \
      > "$root/$phase-$dataset.log" 2>&1
  done
done
