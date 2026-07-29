#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

: "${PRED_JSONL:?set PRED_JSONL to the JSONL written by tools/infer_stage1_val.py --output-jsonl}"

FRAME_DIR="${FRAME_DIR:-${PRED_JSONL%.jsonl}_frames}"
MAX_ROWS="${MAX_ROWS:-0}"

python - "${PRED_JSONL}" "${FRAME_DIR}" "${MAX_ROWS}" <<'PY'
import json
import sys
from pathlib import Path

from decord import VideoReader, cpu
from PIL import Image, ImageDraw

pred_path = Path(sys.argv[1])
frame_dir = Path(sys.argv[2])
max_rows = int(sys.argv[3])

rows = []
with pred_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
if max_rows > 0:
    rows = rows[:max_rows]
if not rows:
    raise ValueError(f"{pred_path}: no prediction rows")

frame_dir.mkdir(parents=True, exist_ok=True)

def safe_name(value):
    text = str(value or "sample")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:120]

for row in rows:
    video = Path(str(row.get("video") or ""))
    if not video.exists():
        print(f"skip missing video: index={row.get('index')} video={video}")
        continue
    clip_start = float(row.get("clip_start") or 0.0)
    clip_end = float(row.get("clip_end") or clip_start)
    if clip_end <= clip_start:
        print(f"skip invalid window: index={row.get('index')} [{clip_start}, {clip_end}]")
        continue

    vr = VideoReader(str(video), ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    times = [clip_start, (clip_start + clip_end) / 2.0, clip_end]
    frame_ids = [
        max(0, min(len(vr) - 1, int(round(t * fps))))
        for t in times
    ]
    frames = vr.get_batch(frame_ids).asnumpy()
    images = []
    for label, t, fid, frame in zip(("start", "middle", "end"), times, frame_ids, frames):
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 30), fill=(0, 0, 0))
        draw.text((8, 8), f"{label} t={t:.2f}s frame={fid}", fill=(255, 255, 255))
        images.append(image)

    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width

    key = row.get("video_key") or row.get("video_id") or video.stem
    out_path = frame_dir / f"{int(row.get('index') or 0):05d}_{safe_name(key)}.jpg"
    sheet.save(out_path, quality=90)
    print(out_path)
PY
