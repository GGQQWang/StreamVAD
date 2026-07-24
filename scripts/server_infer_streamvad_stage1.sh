#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

echo "Stage1 inference entry is pending model-runner integration."
echo "Use this checkpoint once a StreamMind-compatible inference wrapper is added:"
echo "  STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-output/streamvad_stage1_lora}"
echo "Expected generation format: <think>...</think><answer>Abnormal</answer>."
