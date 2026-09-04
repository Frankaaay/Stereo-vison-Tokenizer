#!/usr/bin/env bash
set -euo pipefail

: "${ABLATION_OUTPUT_ROOT:?set a new repository-external ablation output root}"

if [[ -e "${ABLATION_OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse existing ABLATION_OUTPUT_ROOT=${ABLATION_OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p "${ABLATION_OUTPUT_ROOT}"
export LATENT_CHANNELS=48

for condition in left_only same_left; do
  export STEREO_TRAINING_INPUT="${condition}"
  if [[ "${condition}" == "left_only" ]]; then
    run_name="m48-left-only"
  else
    run_name="d48-same-left"
  fi
  export OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT}/${run_name}"
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
