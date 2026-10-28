#!/bin/bash
# Do not enable global fail-fast in the training launcher; report failures from
# the actual training process instead of terminating on a probe/optional check.
# set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
TRAIN_FILE="${TRAIN_FILE:?set TRAIN_FILE}"
VAL_FILE="${VAL_FILE:-}"
WARMUP_ENABLED="${WARMUP_ENABLED:-true}"
CERTIFICATE_PATH="${CERTIFICATE_PATH:-}"
CALIBRATION_ROWS="${CALIBRATION_ROWS:-}"
RISK_ROWS="${RISK_ROWS:-}"
FROZEN_RULE_PATH="${FROZEN_RULE_PATH:-${CERTIFICATE_PATH%/*}/frozen_rule.json}"
SETGRPO_GRADER_BACKEND="${SETGRPO_GRADER_BACKEND:-external_api}"
if [[ "${SETGRPO_GRADER_BACKEND}" == "external_api" || "${SETGRPO_GRADER_BACKEND}" == "luna" || "${SETGRPO_GRADER_BACKEND}" == "stage2" ]]; then
  : "${STAGE2_API_BASE_URL:=https://sub.anzhiyu.com/v1}"
  : "${STAGE2_API_MODEL:=gpt-5.6-luna}"
else
  VLLM_BASE_URL="${VLLM_BASE_URL:?set VLLM_BASE_URL to RubricHub grader endpoints}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-setgrpo_rubrichub_online}"
REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-health_bench/scaleai_batch_reward_fn.py}"
REWARD_FUNCTION_NAME="${REWARD_FUNCTION_NAME:-compute_score_batched}"
REWARD_MANAGER="${REWARD_MANAGER:-batch_dapo}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-setgrpo}"
ONLINE_ENABLED="${ONLINE_ENABLED:-true}"
SETGRPO_VARIANT="${SETGRPO_VARIANT:-full}"
SETGRPO_DISCARD_RATE="${SETGRPO_DISCARD_RATE:-0.0}"
PROJECTION_DIM="${PROJECTION_DIM:-256}"
export SETGRPO_GRADER_BACKEND STAGE2_API_BASE_URL STAGE2_API_MODEL
if [[ -n "${VLLM_BASE_URL:-}" ]]; then export VLLM_BASE_URL; fi

# Fail early if the Hydra config is an older copy missing fields used below.
# This is clearer than letting Hydra fail after importing the full trainer.
SETGRPO_CONFIG_FILE="${SETGRPO_CONFIG_FILE:-${PWD}/recipe/dapo/config/dapo_trainer.yaml}"
if [[ ! -f "${SETGRPO_CONFIG_FILE}" ]]; then
  echo "ERROR: missing SetGRPO Hydra config: ${SETGRPO_CONFIG_FILE}" >&2
  exit 2
fi
for required_key in "variant:" "discard_rate:" "projection_dim:"; do
  grep -Eq "^[[:space:]]+${required_key}" "${SETGRPO_CONFIG_FILE}" || {
    echo "ERROR: outdated SetGRPO config ${SETGRPO_CONFIG_FILE}; missing ${required_key}" >&2
    echo "SYNC recipe/dapo/config/dapo_trainer.yaml before submitting" >&2
    exit 2
  }
done

if [[ "${ONLINE_ENABLED}" == "true" && "${WARMUP_ENABLED}" != "true" ]]; then
  : "${CERTIFICATE_PATH:?set CERTIFICATE_PATH when WARMUP_ENABLED=false}"
  : "${CALIBRATION_ROWS:?set CALIBRATION_ROWS when WARMUP_ENABLED=false}"
  : "${RISK_ROWS:?set RISK_ROWS when WARMUP_ENABLED=false}"
  python3 -m setgrpo_preflight.freeze_rules --calibration "${CALIBRATION_ROWS}" --risk "${RISK_ROWS}" --output "${FROZEN_RULE_PATH}"
fi

ONLINE_ARGS=""
if [[ "${ONLINE_ENABLED}" == "true" ]]; then
  ONLINE_ARGS="algorithm.setgrpo.online.enabled=true algorithm.setgrpo.online.provider_type=online_certificate algorithm.setgrpo.online.certificate_path=${CERTIFICATE_PATH} algorithm.setgrpo.online.rule_path=${FROZEN_RULE_PATH} algorithm.setgrpo.online.certificate_key=${CERTIFICATE_KEY:-microbatch_id} algorithm.setgrpo.online.variant=${SETGRPO_VARIANT} algorithm.setgrpo.online.discard_rate=${SETGRPO_DISCARD_RATE} algorithm.setgrpo.online.allow_reward_proxy=false algorithm.setgrpo.online.warmup.enabled=${WARMUP_ENABLED} algorithm.setgrpo.online.warmup.output_dir=${WARMUP_OUTPUT_DIR:-${EXPERIMENT_NAME}_warmup} algorithm.setgrpo.online.warmup.alpha=${WARMUP_ALPHA:-0.1} algorithm.setgrpo.online.warmup.beta=${WARMUP_BETA:-0.25} algorithm.setgrpo.online.warmup.gamma=${WARMUP_GAMMA:-0.25} algorithm.setgrpo.online.warmup.shape_shrinkage=${WARMUP_SHAPE_SHRINKAGE:-0.1} algorithm.setgrpo.online.warmup.shape_condition_cap=${WARMUP_SHAPE_CONDITION_CAP:-1000.0}"
else
  ONLINE_ARGS="algorithm.setgrpo.online.enabled=false"
fi

DATA_ARGS=(data.train_files="${TRAIN_FILE}")
if [[ -n "${VAL_FILE}" ]]; then
  DATA_ARGS+=(data.val_files="${VAL_FILE}")
fi

python3 -m recipe.dapo.main_dapo \
  "${DATA_ARGS[@]}" \
  data.prompt_key=prompt \
  data.truncation=left \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
  data.gen_batch_size="${GEN_BATCH_SIZE:-32}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.setgrpo.online.projection_dim="${PROJECTION_DIM}" \
  ${ONLINE_ARGS} \
  reward_model.reward_manager="${REWARD_MANAGER}" \
  custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
  custom_reward_function.name="${REWARD_FUNCTION_NAME}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.strategy=fsdp \
  critic.strategy=fsdp \
  trainer.project_name="setgrpo_rubrichub" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.val_before_train=false \
  "$@"
