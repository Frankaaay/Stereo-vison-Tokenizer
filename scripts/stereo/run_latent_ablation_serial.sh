#!/usr/bin/env bash
set -euo pipefail

: "${ABLATION_OUTPUT_ROOT:?set a new repository-external ablation output root}"

if [[ -e "${ABLATION_OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse existing ABLATION_OUTPUT_ROOT=${ABLATION_OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p "${ABLATION_OUTPUT_ROOT}"

for latent_channels in 96 24 48; do
  export LATENT_CHANNELS="${latent_channels}"
  export OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT}/z${latent_channels}"
  mkdir "${OUTPUT_ROOT}"
  set +e
  bash scripts/stereo/train_stereo_vae.sh 2>&1 | tee "${OUTPUT_ROOT}/run.log"
  train_status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "${train_status}" > "${OUTPUT_ROOT}/exit_code.txt"
  if (( train_status != 0 )); then
    exit "${train_status}"
  fi
done
