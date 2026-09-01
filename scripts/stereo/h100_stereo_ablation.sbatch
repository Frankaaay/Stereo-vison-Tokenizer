#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT before sbatch --export}"
exec "${REPO_ROOT}/scripts/stereo/run_h100_stereo_ablation.sh"
