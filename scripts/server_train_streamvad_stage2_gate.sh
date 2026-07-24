#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

echo "Stage2 Gate training is intentionally not launched by this placeholder."
echo "Required before enabling:"
echo "1. Wire StreamVAD gate_action hold/trigger rows into a StreamMind cls training adapter."
echo "2. Resolve gate class ids from configuration, not tokenizer token ids."
echo "3. Load STAGE1_CHECKPOINT and train only the cognition-gate parameters."
echo
echo "Expected inputs:"
echo "  STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-output/streamvad_stage1_lora}"
echo "  STREAMVAD_STAGE2_JSONL=${STREAMVAD_STAGE2_JSONL:-data/streamvad_weak_supervision/streamvad_stage2_train.jsonl}"
