#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="scripts/baselines/celeba"
METHODS=(fjord fluid papaya pyramidfl swiftfl)
MODEL="convnet2"
DATASET="celeba"

CONFIG_FILES=()
for m in "${METHODS[@]}"; do
  CONFIG_FILES+=("${BASE_DIR}/${m}_${MODEL}_on_${DATASET}.yaml")
done

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
  LOG_DIR="${CONFIG_FILE/scripts\/baselines/scripts\/logs}"
  LOG_FILE="${LOG_DIR/.yaml/.log}"
  LOG_DIR="$(dirname "$LOG_FILE")"

  mkdir -p "$LOG_DIR"

  echo "Running: $CONFIG_FILE"
  nohup python federatedscope/main.py --cfg "$CONFIG_FILE" > "$LOG_FILE" 2>&1
done
