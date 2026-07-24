#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

: "${VADR1_JSONL:=Vad-R1/data/Vad-Reasoning-SFT-train.jsonl}"
: "${STREAMVAD_DATA_DIR:=data/streamvad_weak_supervision}"
: "${FPS:=30}"

extra_args=()
if [[ -n "${PATH_PREFIX_FROM:-}" || -n "${PATH_PREFIX_TO:-}" ]]; then
  extra_args+=(--path-prefix-from "${PATH_PREFIX_FROM:?set PATH_PREFIX_FROM with PATH_PREFIX_TO}")
  extra_args+=(--path-prefix-to "${PATH_PREFIX_TO:?set PATH_PREFIX_TO with PATH_PREFIX_FROM}")
fi
if [[ -n "${REQUIRE_RELIABLE_TIMING:-}" ]]; then
  extra_args+=(--require-reliable-timing)
fi

python3 tools/build_streamvad_weak_supervision.py \
  --input-jsonl "${VADR1_JSONL}" \
  --output-dir "${STREAMVAD_DATA_DIR}" \
  --fps "${FPS}" \
  "${extra_args[@]}"

python3 tools/smoke_test_streamvad_batches.py \
  --stage1-jsonl "${STREAMVAD_DATA_DIR}/streamvad_stage1_train.jsonl" \
  --stage2-jsonl "${STREAMVAD_DATA_DIR}/streamvad_stage2_train.jsonl"
