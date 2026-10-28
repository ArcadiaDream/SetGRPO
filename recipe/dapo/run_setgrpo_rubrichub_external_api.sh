#!/usr/bin/env bash

# Formal RubricHub SetGRPO launcher with the external API grader.
# The API key is intentionally not accepted here; inject it through the
# environment as STAGE2_API_KEY or OPENAI_API_KEY before launching.

set -u

: "${MODEL_PATH:?set MODEL_PATH}"
: "${TRAIN_FILE:?set TRAIN_FILE}"
: "${CONDA_ENV:-}"
: "${STAGE2_API_KEY:=${OPENAI_API_KEY:-}}"
: "${STAGE2_API_BASE_URL:=https://sub.anzhiyu.com/v1}"
: "${STAGE2_API_MODEL:=gpt-5.6-luna}"

if [[ -z "${STAGE2_API_KEY}" ]]; then
  echo "ERROR: STAGE2_API_KEY or OPENAI_API_KEY must be supplied in the environment" >&2
  exit 2
fi

export SETGRPO_GRADER_BACKEND=external_api
export STAGE2_API_BASE_URL STAGE2_API_MODEL STAGE2_API_KEY
export STAGE2_API_TIMEOUT="${STAGE2_API_TIMEOUT:-240}"
export STAGE2_API_MAX_RETRIES="${STAGE2_API_MAX_RETRIES:-5}"
export STAGE2_API_CONCURRENCY="${STAGE2_API_CONCURRENCY:-8}"
export STAGE2_API_TEMPERATURE="${STAGE2_API_TEMPERATURE:-0}"
export STAGE2_API_PREFLIGHT_TIMEOUT="${STAGE2_API_PREFLIGHT_TIMEOUT:-180}"

echo "[SetGRPO] grader_backend=${SETGRPO_GRADER_BACKEND}"
echo "[SetGRPO] grader_endpoint=${STAGE2_API_BASE_URL} model=${STAGE2_API_MODEL}"
echo "[SetGRPO] grader_concurrency=${STAGE2_API_CONCURRENCY} timeout=${STAGE2_API_TIMEOUT}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # The inherited environment is intentionally preserved when the caller has
  # already activated a different environment; otherwise activate CONDA_ENV.
  if [[ -n "${CONDA_ENV:-}" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python scripts/check_stage2_external_api.py \
  --timeout "$STAGE2_API_PREFLIGHT_TIMEOUT" \
  --max-tokens 1

exec "$(dirname "${BASH_SOURCE[0]}")/run_setgrpo_rubrichub_online.sh" "$@"
