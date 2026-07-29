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

def resize_for_sheet(image, max_width=420):
    if image.width <= max_width:
        return image
    height = max(1, round(image.height * (max_width / image.width)))
    return image.resize((max_width, height))

def make_sheet(images):
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width
    return sheet

def load_inference_frame_ids(vr, clip_start, clip_end):
    fps = float(vr.get_avg_fps())
    start_frame = int(max(0, clip_start * fps - 1))
    end_frame = int(clip_end * fps + 1)
    if end_frame <= start_frame or clip_start >= clip_end:
        raise ValueError(f"invalid video segment [{clip_start}, {clip_end}]")
    seg_size = max(1, int(fps / 2))
    frame_ids = list(range(start_frame, min(end_frame, len(vr)), seg_size))
    if not frame_ids:
        raise ValueError(f"no frames sampled for segment [{clip_start}, {clip_end}]")
    if len(frame_ids) > 64:
        keep = []
        for i in range(64):
            pos = round(i * (len(frame_ids) - 1) / 63)
            keep.append(frame_ids[pos])
        frame_ids = keep
    return frame_ids

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
    checkpoint_times = [clip_start, (clip_start + clip_end) / 2.0, clip_end]
    checkpoint_frame_ids = [
        max(0, min(len(vr) - 1, int(round(t * fps))))
        for t in checkpoint_times
    ]
    sampled_frame_ids = load_inference_frame_ids(vr, clip_start, clip_end)
    sampled_preview_ids = sampled_frame_ids
    if len(sampled_preview_ids) > 12:
        sampled_preview_ids = [
            sampled_frame_ids[round(i * (len(sampled_frame_ids) - 1) / 11)]
            for i in range(12)
        ]

    frames = vr.get_batch(checkpoint_frame_ids).asnumpy()
    images = []
    for label, t, fid, frame in zip(("start", "middle", "end"), checkpoint_times, checkpoint_frame_ids, frames):
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 30), fill=(0, 0, 0))
        draw.text((8, 8), f"{label} t={t:.2f}s frame={fid}", fill=(255, 255, 255))
        images.append(resize_for_sheet(image))

    sampled_frames = vr.get_batch(sampled_preview_ids).asnumpy()
    sampled_images = []
    for fid, frame in zip(sampled_preview_ids, sampled_frames):
        image = resize_for_sheet(Image.fromarray(frame).convert("RGB"), max_width=260)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
        draw.text((6, 6), f"sample frame={fid}", fill=(255, 255, 255))
        sampled_images.append(image)

    key = row.get("video_key") or row.get("video_id") or video.stem
    stem = f"{int(row.get('index') or 0):05d}_{safe_name(key)}"
    checkpoint_path = frame_dir / f"{stem}_clip_checkpoints.jpg"
    sampled_path = frame_dir / f"{stem}_inference_samples.jpg"
    meta_path = frame_dir / f"{stem}_frames.json"
    make_sheet(images).save(checkpoint_path, quality=90)
    make_sheet(sampled_images).save(sampled_path, quality=90)
    meta_path.write_text(json.dumps({
        "index": row.get("index"),
        "video_key": row.get("video_key"),
        "video": str(video),
        "original_video": row.get("original_video"),
        "clip_start": clip_start,
        "clip_end": clip_end,
        "fps": fps,
        "num_frames": len(vr),
        "checkpoint_frame_ids": checkpoint_frame_ids,
        "inference_frame_ids": sampled_frame_ids,
        "inference_preview_frame_ids": sampled_preview_ids,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(checkpoint_path)
    print(sampled_path)
    print(meta_path)
PY
