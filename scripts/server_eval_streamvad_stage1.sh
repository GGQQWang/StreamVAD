#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

: "${PRED_JSONL:?set PRED_JSONL to the JSONL written by tools/infer_stage1_val.py --output-jsonl}"

python - "${PRED_JSONL}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
rows = []
with path.open("r", encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        missing = {"gt", "pred", "correct", "miss", "false_alarm", "unparsed"} - set(row)
        if missing:
            raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
        rows.append(row)

if not rows:
    raise ValueError(f"{path}: no prediction rows")

total = len(rows)
skipped = [row for row in rows if row.get("skipped")]
scored_rows = [row for row in rows if not row.get("skipped")]
correct = sum(bool(row["correct"]) for row in scored_rows)
labels = Counter(row["gt"] for row in scored_rows)
correct_by_label = Counter(row["gt"] for row in scored_rows if row["correct"])
misses = [row for row in scored_rows if row["miss"]]
false_alarms = [row for row in scored_rows if row["false_alarm"]]
unparsed = [row for row in scored_rows if row["unparsed"]]
parse_methods = Counter(row.get("decision_parse_method", "unknown") for row in scored_rows)

def pct(num, den):
    return f"{num / den:.2%}" if den else "n/a"

print(f"Predictions: {path}")
print(f"Rows: {total}  scored: {len(scored_rows)}  skipped: {len(skipped)}")
print(f"Accuracy: {correct}/{len(scored_rows)} = {pct(correct, len(scored_rows))}")
for label in ("normal", "abnormal"):
    count = labels[label]
    hit = correct_by_label[label]
    print(f"  {label}: {hit}/{count} = {pct(hit, count)}")
print(f"Miss rate: {len(misses)}/{labels['abnormal']} = {pct(len(misses), labels['abnormal'])}")
print(f"False alarm rate: {len(false_alarms)}/{labels['normal']} = {pct(len(false_alarms), labels['normal'])}")
print(f"Unparsed answers: {len(unparsed)}")
print("Parse methods:")
for method, count in sorted(parse_methods.items()):
    print(f"  {method}: {count}")

if misses:
    print("First miss examples:")
    for row in misses[:10]:
        key = row.get("video_key") or row.get("video_id") or row.get("video")
        print(f"  index={row.get('index')} key={key} pred={row.get('pred')}")
PY
