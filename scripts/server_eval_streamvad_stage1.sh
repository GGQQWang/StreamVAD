#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

echo "Stage1 evaluation entry is pending generated-output JSONL integration."
echo "Minimum metrics to report after inference is wired:"
echo "  answer-format validity"
echo "  abnormal-answer recall on event clips"
echo "  CoT audit sample quality"
echo "  invalid or missing <think>/<answer> tag rate"
