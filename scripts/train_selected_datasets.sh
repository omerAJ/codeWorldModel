#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-maincoder}"
CONFIG_PATH="${CONFIG_PATH:-configs/train.maincoder_1b.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python}"
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="outputs/batch_train_${TIMESTAMP_UTC}"
mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python binary not found or not executable: ${PYTHON_BIN}"
  echo "Set PYTHON_BIN explicitly, e.g. PYTHON_BIN=/home/city/miniconda3/envs/maincoder/bin/python"
  exit 1
fi

# Additional --set overrides can be passed through this script.
EXTRA_OVERRIDES=("$@")

declare -a DATASETS=(
  "humaneval_extended_v1_filtered:data/runs/20260226T053436Z_humaneval_extended_v1_filtered/transition_dataset.jsonl"
  "mbpp_extended_v1_valid_balanced_tasktext:data/runs/20260226_mbpp_extended_v1_valid_balanced_tasktext/transition_dataset.jsonl"
  "mbpp_balanced_tasktext:data/runs/20260226_mbpp_balanced_tasktext/transition_dataset.jsonl"
  "humaneval_balanced_tasktext:data/runs/humaneval/humaneval_transition_balanced.jsonl"
)

echo "Batch timestamp (UTC): ${TIMESTAMP_UTC}"
echo "Conda env: ${CONDA_ENV}"
echo "Python bin: ${PYTHON_BIN}"
echo "Config: ${CONFIG_PATH}"
echo "Logs: ${LOG_DIR}"
printf "run_name\tdataset_path\tstatus\n" > "${LOG_DIR}/summary.tsv"

for item in "${DATASETS[@]}"; do
  name="${item%%:*}"
  path="${item#*:}"
  run_name="${TIMESTAMP_UTC}_${name}"
  log_file="${LOG_DIR}/${run_name}.log"

  if [[ ! -f "${path}" ]]; then
    echo "MISSING DATASET: ${path}"
    printf "%s\t%s\tmissing_dataset\n" "${run_name}" "${path}" >> "${LOG_DIR}/summary.tsv"
    continue
  fi

  echo "=== Starting ${run_name} ==="
  echo "Dataset: ${path}"
  echo "Log: ${log_file}"

  cmd=(
    "${PYTHON_BIN}" scripts/train.py
    --config "${CONFIG_PATH}"
    --set "data.path=${path}"
    --set "output.run_name=${run_name}"
  )

  if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_OVERRIDES[@]}")
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "${log_file}"
  status=$?
  set -e

  if [[ ${status} -eq 0 ]]; then
    printf "%s\t%s\tok\n" "${run_name}" "${path}" >> "${LOG_DIR}/summary.tsv"
    echo "=== Completed ${run_name} ==="
  else
    printf "%s\t%s\tfailed(${status})\n" "${run_name}" "${path}" >> "${LOG_DIR}/summary.tsv"
    echo "=== Failed ${run_name} (exit ${status}) ==="
  fi
done

echo "Batch summary: ${LOG_DIR}/summary.tsv"
