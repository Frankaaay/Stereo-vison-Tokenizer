#!/usr/bin/env bash
set -euo pipefail
cd /data/home/frank/projects/Stereo-vison-Tokenizer
source /data/home/frank/runtime/stereo-tokenizer-pretrain-h2001-20260828/venv/bin/activate
export PYTHONPATH=/data/home/frank/projects/Stereo-vison-Tokenizer
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
phase="${1:?smoke or full}"
case "$phase" in
  smoke) extra=(--smoke) ;;
  full) extra=() ;;
  *) exit 2 ;;
esac
root=/data/home/frank/experiments/stereo-input-eval-h2001-20260910-v3
output="$root/$phase"
mkdir -p "$root"
test ! -e "$output"
trap 'rc=$?; printf "%s\n" "$rc" > "$root/$phase.exit_code.txt"' EXIT
python3 doc/frank/h2001-stereo-eval-20260910.py --output "$output" --batch-size 4 "${extra[@]}" > "$root/$phase.log" 2>&1
