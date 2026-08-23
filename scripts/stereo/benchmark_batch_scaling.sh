#!/usr/bin/env bash
set -euo pipefail

: "${TELEMETRY_OUTPUT:?set the nvidia-smi telemetry CSV path}"

mkdir -p "$(dirname "${TELEMETRY_OUTPUT}")"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
  --format=csv,noheader,nounits \
  --loop=1 >"${TELEMETRY_OUTPUT}" &
telemetry_pid=$!

stop_telemetry() {
  kill "${telemetry_pid}" 2>/dev/null || true
  wait "${telemetry_pid}" 2>/dev/null || true
}
trap stop_telemetry EXIT

bash scripts/stereo/train_stereo_vae.sh
