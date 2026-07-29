#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

: "${PRED_JSONL:?set PRED_JSONL to the JSONL written by tools/infer_stage1_val.py --output-jsonl}"

REVIEW_MD="${REVIEW_MD:-${PRED_JSONL%.jsonl}_review.md}"

python - "${PRED_JSONL}" "${REVIEW_MD}" <<'PY'
import json
import sys
from pathlib import Path

pred_path = Path(sys.argv[1])
review_path = Path(sys.argv[2])

rows = []
with pred_path.open("r", encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row["_line_no"] = line_no
        rows.append(row)

if not rows:
    raise ValueError(f"{pred_path}: no prediction rows")

review_path.parent.mkdir(parents=True, exist_ok=True)

def block(value):
    if value is None:
        value = ""
    value = str(value).strip()
    return value if value else "(empty)"

with review_path.open("w", encoding="utf-8") as out:
    out.write(f"# StreamVAD Stage1 Manual Review\n\n")
    out.write(f"- Predictions: `{pred_path}`\n")
    out.write(f"- Rows: `{len(rows)}`\n\n")
    out.write("Use `human_label` for your manual judgment: `normal`, `abnormal`, or `unclear`.\n\n")

    for row in rows:
        key = row.get("video_key") or row.get("video_id") or row.get("video") or f"line:{row['_line_no']}"
        out.write(f"## Sample {row.get('index', row['_line_no'])}: {key}\n\n")
        out.write(f"- human_label:\n")
        out.write(f"- gt: `{row.get('gt')}`\n")
        out.write(f"- parsed_pred: `{row.get('pred')}`\n")
        out.write(f"- parse_method: `{row.get('decision_parse_method', 'unknown')}`\n")
        out.write(f"- miss: `{row.get('miss')}` false_alarm: `{row.get('false_alarm')}` unparsed: `{row.get('unparsed')}` skipped: `{row.get('skipped', False)}`\n")
        if row.get("jsonl_path") or row.get("jsonl_line"):
            out.write(f"- source_jsonl: `{row.get('jsonl_path')}` line `{row.get('jsonl_line')}`\n")
        out.write(f"- video: `{row.get('video')}`\n")
        if row.get("original_video"):
            out.write(f"- original_video: `{row.get('original_video')}`\n")
        out.write(f"- window: `{row.get('clip_start')}` to `{row.get('clip_end')}`; event: `{row.get('event_start_sec')}` to `{row.get('event_end_sec')}`\n\n")

        if row.get("observation") or row.get("reason"):
            out.write("### GT Evidence\n\n")
            if row.get("observation"):
                out.write(f"Observation:\n\n> {block(row.get('observation')).replace(chr(10), ' ')}\n\n")
            if row.get("reason"):
                out.write(f"Reason:\n\n> {block(row.get('reason')).replace(chr(10), ' ')}\n\n")
        elif row.get("target_text"):
            out.write("### GT Target Text\n\n")
            out.write("```text\n")
            out.write(block(row.get("target_text")))
            out.write("\n```\n\n")

        if row.get("original_answer"):
            out.write("### Original Answer\n\n")
            out.write("```text\n")
            out.write(block(row.get("original_answer")))
            out.write("\n```\n\n")

        out.write("### Model Output\n\n")
        out.write("```text\n")
        out.write(block(row.get("output")))
        out.write("\n```\n\n")

        if row.get("error"):
            out.write("### Error\n\n")
            out.write("```text\n")
            out.write(block(row.get("error")))
            out.write("\n```\n\n")

print(f"Review file written to: {review_path}")
PY
