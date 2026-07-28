#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

: "${STAGE1_CHECKPOINT:?set STAGE1_CHECKPOINT to a StreamVAD LoRA checkpoint directory, e.g. output/streamvad_stage1_lora/checkpoint-500}"
: "${MODEL_PATH:?set MODEL_PATH to the server VideoLLaMA2 base checkpoint path}"
: "${VISION_TOWER:?set VISION_TOWER to the server CLIP vision tower path}"

args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  args+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  args+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ -n "${PRED_JSONL:-}" ]]; then
  args+=(--output-jsonl "${PRED_JSONL}")
fi
if [[ "${FLASH_ATTN:-0}" == "1" ]]; then
  args+=(--flash-attn)
fi

python tools/infer_stage1_val.py \
  --checkpoint "${STAGE1_CHECKPOINT}" \
  --base-model "${MODEL_PATH}" \
  --vision-tower "${VISION_TOWER}" \
  --val-jsonl "${STREAMVAD_STAGE1_VAL_JSONL:-data/streamvad_weak_supervision/streamvad_stage1_val.jsonl}" \
  --streammind-root "${STREAMMIND_ROOT:-${STREAMVAD_ROOT:-${REPO_ROOT}}/StreamMind}" \
  --num-frames "${NUM_FRAMES:-32}" \
  "${args[@]}"
