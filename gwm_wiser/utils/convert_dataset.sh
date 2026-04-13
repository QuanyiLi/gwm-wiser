#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_PREFIX="${ENV_PREFIX:-${REPO_ROOT}/.conda/envs/gwm-openvla-oft}"
TRAIN_MANUAL_DIR="${TRAIN_MANUAL_DIR:-${MANUAL_DIR:-${REPO_ROOT}/datasets/train}}"

# Auto-derive DATA_DIR from MANUAL_DIR: place rlds output as sibling of input dir
# e.g. .../wise_dataset_0.4.3/merged_train -> .../wise_dataset_0.4.3/rlds_train
#       .../wise_dataset_0.4.3/foo_test    -> .../wise_dataset_0.4.3/rlds_test
#       .../wise_dataset_0.4.3/mydata      -> .../wise_dataset_0.4.3/rlds
if [[ -z "${DATA_DIR:-}" ]]; then
  _parent_dir="$(dirname "${TRAIN_MANUAL_DIR}")"
  _leaf_dir="$(basename "${TRAIN_MANUAL_DIR}")"
  if [[ "${_leaf_dir}" == *train* ]]; then
    DATA_DIR="${_parent_dir}/rlds_train"
  elif [[ "${_leaf_dir}" == *test* ]]; then
    DATA_DIR="${_parent_dir}/rlds_test"
  else
    DATA_DIR="${_parent_dir}/rlds"
  fi
fi
DATASET_BUILDER_PATH="${DATASET_BUILDER_PATH:-${REPO_ROOT}/gwm_wiser/utils/rlds_converters/maniskill_dataset_converted_externally_to_rlds}"
DATASET_NAME="${DATASET_NAME:-maniskill_dataset_converted_externally_to_rlds}"
DATASET_VERSION="${DATASET_VERSION:-0.0.3}"
DATASET_IMPORT="${DATASET_IMPORT:-rlds_converters.${DATASET_NAME}.dataset_builder}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

export GWM_RLDS_MAX_EPISODES="${GWM_RLDS_MAX_EPISODES:-0}"
export GWM_RLDS_MAX_STEPS_PER_EPISODE="${GWM_RLDS_MAX_STEPS_PER_EPISODE:-0}"
export GWM_RLDS_EPISODE_IDS="${GWM_RLDS_EPISODE_IDS:-}"

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    NUM_PROCESSES="$(nproc)"
  else
    NUM_PROCESSES=1
  fi
fi
if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  NUM_PROCESSES=1
fi

DATASET_INFO_PATH="${DATA_DIR}/${DATASET_NAME}/${DATASET_VERSION}/dataset_info.json"
if [[ -f "${DATASET_INFO_PATH}" && "${FORCE_REBUILD}" != "1" ]]; then
  echo "[convert_dataset] skip: existing converted dataset found at ${DATASET_INFO_PATH}"
  echo "[convert_dataset] set FORCE_REBUILD=1 to rebuild"
  exit 0
fi

echo "----------------------------------------"
echo "TFDS build (RLDS)"
echo "Env Prefix:   ${ENV_PREFIX}"
echo "Train Dir:    ${TRAIN_MANUAL_DIR}"
echo "Data Dir:     ${DATA_DIR}"
echo "Builder Path: ${DATASET_BUILDER_PATH}"
echo "Dataset Name: ${DATASET_NAME}"
echo "Dataset Import:${DATASET_IMPORT}"
echo "Max Episodes: ${GWM_RLDS_MAX_EPISODES}"
echo "Max Steps/Ep: ${GWM_RLDS_MAX_STEPS_PER_EPISODE}"
echo "Episode IDs:  ${GWM_RLDS_EPISODE_IDS:-<none>}"
echo "Processes:    ${NUM_PROCESSES}"
echo "Force Rebuild:${FORCE_REBUILD}"
echo "----------------------------------------"

# Create a temporary PYTHONPATH directory with only a symlink to rlds_converters,
# so that lerobot.py in utils/ doesn't shadow the installed lerobot package.
_TFDS_PYTHONPATH="$(mktemp -d)"
ln -s "${REPO_ROOT}/gwm_wiser/utils/rlds_converters" "${_TFDS_PYTHONPATH}/rlds_converters"
trap 'rm -rf "${_TFDS_PYTHONPATH}"' EXIT

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 \
  conda run --no-capture-output -p "${ENV_PREFIX}" \
    env PYTHONPATH="${_TFDS_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
    tfds build "${DATASET_NAME}" \
    --imports="${DATASET_IMPORT}" \
    --manual_dir="${TRAIN_MANUAL_DIR}" \
    --data_dir="${DATA_DIR}" \
    --num-processes="${NUM_PROCESSES}"
else
  CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 \
  conda run --no-capture-output -p "${ENV_PREFIX}" \
    tfds build "${DATASET_BUILDER_PATH}" \
    --manual_dir="${TRAIN_MANUAL_DIR}" \
    --data_dir="${DATA_DIR}" \
    --num-processes="${NUM_PROCESSES}"
fi

echo "[convert_dataset] done"
