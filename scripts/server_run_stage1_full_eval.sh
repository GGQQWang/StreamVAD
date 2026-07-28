#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

: "${STAGE1_CHECKPOINT:?set STAGE1_CHECKPOINT to a StreamVAD LoRA checkpoint directory}"
: "${MODEL_PATH:?set MODEL_PATH to the server VideoLLaMA2 base checkpoint path}"
: "${VISION_TOWER:?set VISION_TOWER to the server CLIP vision tower path}"
: "${STREAMVAD_STAGE1_VAL_JSONL:?set STREAMVAD_STAGE1_VAL_JSONL to the Stage1 validation JSONL}"

unset MAX_SAMPLES

run_name="${RUN_NAME:-stage1_full_eval}"
checkpoint_name="$(basename "${STAGE1_CHECKPOINT}")"
output_dir="${FULL_EVAL_OUTPUT_DIR:-$(dirname "${STAGE1_CHECKPOINT}")}"

export PRED_JSONL="${PRED_JSONL:-${output_dir}/${run_name}_${checkpoint_name}_predictions.jsonl}"
export REVIEW_MD="${REVIEW_MD:-${output_dir}/${run_name}_${checkpoint_name}_manual_review.md}"

echo "Running full Stage1 inference"
echo "  checkpoint: ${STAGE1_CHECKPOINT}"
echo "  val_jsonl:   ${STREAMVAD_STAGE1_VAL_JSONL}"
echo "  pred_jsonl:  ${PRED_JSONL}"
echo "  review_md:   ${REVIEW_MD}"

scripts/server_infer_streamvad_stage1.sh
scripts/server_eval_streamvad_stage1.sh
scripts/server_export_stage1_review.sh
