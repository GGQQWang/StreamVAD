#!/usr/bin/env python3
"""Sample StreamVAD Stage 1 compact CoT rows for manual audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


AUDIT_FIELDS = [
    "video_id",
    "video",
    "original_source",
    "original_think",
    "original_answer",
    "scene_prior",
    "observation",
    "reason",
    "event_start_sec",
    "event_end_sec",
    "answer",
    "needs_review",
    "review_status",
    "review_note",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_input_file"] = str(path)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in AUDIT_FIELDS})


def source_bucket(row: dict[str, Any]) -> str:
    original_source = str(row.get("original_source") or "").strip()
    if original_source:
        return original_source
    video = str(row.get("video") or "")
    parts = Path(video).parts
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def stratified_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[source_bucket(row)].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    selected: list[dict[str, Any]] = []
    bucket_names = sorted(buckets)
    while len(selected) < count and any(buckets.values()):
        for name in bucket_names:
            if len(selected) >= count:
                break
            if buckets[name]:
                selected.append(buckets[name].pop())
    rng.shuffle(selected)
    return selected


def make_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": row.get("video_id"),
        "video": row.get("video"),
        "original_source": row.get("original_source") or source_bucket(row),
        "original_think": row.get("original_think", ""),
        "original_answer": row.get("original_answer", ""),
        "scene_prior": row.get("scene_prior", ""),
        "observation": row.get("observation", ""),
        "reason": row.get("reason", ""),
        "event_start_sec": row.get("event_start_sec", ""),
        "event_end_sec": row.get("event_end_sec", ""),
        "answer": row.get("answer", ""),
        "needs_review": row.get("needs_review", ""),
        "review_status": "",
        "review_note": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, nargs="+", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--normal-count", type=int, default=50)
    parser.add_argument("--abnormal-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [row for path in args.input_jsonl for row in read_jsonl(path)]
    normal_rows = [row for row in rows if str(row.get("answer", "")).lower() == "normal"]
    abnormal_rows = [row for row in rows if str(row.get("answer", "")).lower() == "abnormal"]

    sampled = stratified_sample(normal_rows, args.normal_count, args.seed)
    sampled.extend(stratified_sample(abnormal_rows, args.abnormal_count, args.seed + 1))
    audit_rows = [make_audit_row(row) for row in sampled]

    write_csv(args.output_csv, audit_rows)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, audit_rows)

    print("=== StreamVAD Compact CoT Audit Sample ===")
    print(f"input_rows: {len(rows)}")
    print(f"normal_available: {len(normal_rows)}")
    print(f"abnormal_available: {len(abnormal_rows)}")
    print(f"normal_sampled: {sum(1 for row in audit_rows if row['answer'] == 'normal')}")
    print(f"abnormal_sampled: {sum(1 for row in audit_rows if row['answer'] == 'abnormal')}")
    print(f"wrote_csv: {args.output_csv}")
    if args.output_jsonl:
        print(f"wrote_jsonl: {args.output_jsonl}")


if __name__ == "__main__":
    main()
