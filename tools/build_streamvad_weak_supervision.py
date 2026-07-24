#!/usr/bin/env python3
"""Build StreamVAD weak-supervision JSONL files from VAD-R1 JSONL data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


TAG_RE = re.compile(r"<[^>]+>")
IGNORE_INDEX = -100


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalized_video_key(row: dict[str, Any]) -> str:
    video_id = str(row.get("video") or "").strip()
    if video_id:
        return f"video:{video_id}"
    raw_path = str(row.get("path") or "").strip()
    if raw_path:
        return f"path:{Path(raw_path).as_posix()}"
    return f"line:{row.get('_line_no')}"


def original_video_path(row: dict[str, Any]) -> str:
    return str(row.get("path") or row.get("video") or "")


def resolve_video_path(row: dict[str, Any], args: argparse.Namespace) -> str:
    path = original_video_path(row)
    prefix_from = args.path_prefix_from
    prefix_to = args.path_prefix_to
    if not path or not prefix_from:
        return path
    normalized_path = Path(path).as_posix()
    normalized_from = Path(prefix_from).as_posix().rstrip("/")
    normalized_to = Path(prefix_to).as_posix().rstrip("/") if prefix_to else ""
    if normalized_path == normalized_from:
        return normalized_to
    if normalized_path.startswith(normalized_from + "/"):
        suffix = normalized_path[len(normalized_from) :].lstrip("/")
        return str(Path(normalized_to) / suffix) if normalized_to else suffix
    return path


def extract_tag(text: str | None, tag: str) -> str:
    if not text:
        return ""
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    return clean_text(match.group(1)) if match else ""


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n-")


def strip_low_value_reasoning(text: str) -> str:
    """Keep visible facts while removing common consequence/norm-heavy clauses."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    banned = (
        "could lead to",
        "could cause",
        "as a result",
        "violat",
        "social norm",
        "societal",
        "psychological",
        "emotional distress",
        "potential negative",
        "broader implication",
        "expected to",
        "considered abnormal because",
    )
    kept = [s.strip() for s in sentences if s.strip() and not any(b in s.lower() for b in banned)]
    return " ".join(kept).strip() or text.strip()


def infer_label(row: dict[str, Any]) -> str:
    anomaly_type = str(row.get("anomaly_type", "")).strip().lower()
    if anomaly_type == "normal":
        return "normal"
    answer_which = extract_tag(row.get("answer"), "which").lower().strip(". ")
    if answer_which == "normal":
        return "normal"
    return "abnormal"


def extract_compact_supervision(row: dict[str, Any]) -> tuple[str, str, str, bool]:
    think = row.get("think") or ""
    answer = row.get("answer") or ""
    label = infer_label(row)

    step1 = extract_tag(think, "step1")
    step2 = extract_tag(think, "step2")
    what = extract_tag(answer, "what")
    why = extract_tag(answer, "why")

    scene_prior = strip_low_value_reasoning(step1)
    if label == "abnormal":
        observation_parts = [step2, what]
    else:
        observation_parts = [what, step1]
    observation = strip_low_value_reasoning(" ".join(p for p in observation_parts if p))

    needs_review = False
    if not scene_prior or not observation:
        needs_review = True
    if not think.strip() or not answer.strip():
        needs_review = True

    reason = strip_low_value_reasoning(why) or scene_prior
    return scene_prior, observation, reason, needs_review


def probe_video_duration(path: str) -> tuple[float | None, float | None]:
    if not path or not os.path.exists(path):
        return None, None
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None, None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,duration",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        streams = data.get("streams") or []
        if not streams:
            return None, None
        stream = streams[0]
        duration = float(stream["duration"]) if stream.get("duration") else None
        fps = None
        rate = stream.get("avg_frame_rate")
        if rate and "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            fps = float(num) / den_f if den_f else None
        return duration, fps
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        return None, None


def get_video_timing(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    path = resolve_video_path(row, args)
    total_frames = to_float(row.get("total_frames"))
    probed_duration, probed_fps = probe_video_duration(path)
    fps = probed_fps or args.fps
    if probed_duration is not None:
        duration = probed_duration
        source = "ffprobe"
        reliable = probed_fps is not None
    elif total_frames and fps > 0:
        duration = total_frames / fps
        source = "total_frames_over_fps"
        reliable = False
    else:
        duration = None
        source = "unknown"
        reliable = False
    return {
        "duration_sec": duration,
        "fps": fps,
        "timing_source": source,
        "timing_reliable": reliable,
        "video_exists": bool(path and os.path.exists(path)),
    }


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def boundary_info(row: dict[str, Any], fps: float, duration_sec: float | None) -> dict[str, Any]:
    start = to_float(row.get("start"))
    end = to_float(row.get("end"))
    total_frames = to_float(row.get("total_frames"))
    info = {
        "original_start": start,
        "original_end": end,
        "start_sec": None,
        "end_sec": None,
        "missing": start is None or end is None,
        "start_lt_0": False,
        "end_gt_video_length": False,
        "start_gte_end": False,
        "valid": False,
    }
    if start is None or end is None:
        return info
    info["start_lt_0"] = start < 0
    if total_frames is not None:
        info["end_gt_video_length"] = end > total_frames
    elif duration_sec is not None and fps > 0:
        info["end_gt_video_length"] = (end / fps) > duration_sec
    info["start_gte_end"] = start >= end
    info["start_sec"] = start / fps if fps > 0 else None
    info["end_sec"] = end / fps if fps > 0 else None
    info["valid"] = not (info["missing"] or info["start_lt_0"] or info["end_gt_video_length"] or info["start_gte_end"])
    return info


def make_target_text(observation: str, reason: str) -> str:
    reason_text = reason[:1].lower() + reason[1:] if reason else "the visible behavior departs from normal activity."
    return (
        "<think>\n"
        f"The clip shows {observation}\n"
        f"The behavior is abnormal because {reason_text}\n"
        "</think>\n"
        "<answer>\n"
        "Abnormal\n"
        "</answer>"
    )


def make_stage1(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    label = infer_label(row)
    if label != "abnormal":
        return None
    timing = get_video_timing(row, args)
    duration = timing["duration_sec"]
    scene_prior, observation, reason, needs_review = extract_compact_supervision(row)
    boundary = boundary_info(row, timing["fps"], duration)

    if label == "abnormal" and boundary["valid"] and duration is not None:
        clip_start = clamp(boundary["start_sec"] - args.pre_context_sec, 0.0, duration)
        clip_end = clamp(boundary["end_sec"] + args.post_context_sec, 0.0, duration)
    else:
        return None

    if clip_end is None:
        return None

    return {
        "source": "vadr1",
        "original_source": row.get("source"),
        "video": resolve_video_path(row, args),
        "video_id": row.get("video"),
        "video_key": normalized_video_key(row),
        "original_video": original_video_path(row),
        "clip_start": round(clip_start, 6) if clip_start is not None else None,
        "clip_end": round(clip_end, 6) if clip_end is not None else None,
        "event_start_sec": round(boundary["start_sec"], 6),
        "event_end_sec": round(boundary["end_sec"], 6),
        "event_token_fractions": [0.1, 0.5, 0.9],
        "event_token_policy": "ordered_start_middle_end",
        "scene_prior": scene_prior,
        "observation": observation,
        "reason": reason,
        "answer": "abnormal",
        "target_text": make_target_text(observation, reason),
        "original_think": row.get("think"),
        "original_answer": row.get("answer"),
        "original_start": boundary["original_start"],
        "original_end": boundary["original_end"],
        "original_unit": "frame_index",
        "total_frames": row.get("total_frames"),
        "timing_source": timing["timing_source"],
        "timing_reliable": timing["timing_reliable"],
        "fps": timing["fps"],
        "needs_review": needs_review,
    }


def distance_to_interval_point(point: float, start: float, end: float) -> float:
    if start <= point <= end:
        return 0.0
    return min(abs(point - start), abs(point - end))


def make_stage2(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    label = infer_label(row)
    timing = get_video_timing(row, args)
    duration = timing["duration_sec"]
    if duration is None or duration <= 0:
        return []

    boundary = boundary_info(row, timing["fps"], duration)
    n_chunks = max(1, math.ceil(duration / args.chunk_duration_sec))
    chunks: list[dict[str, Any]] = []
    for idx in range(n_chunks):
        chunk_start = idx * args.chunk_duration_sec
        chunk_end = min(duration, chunk_start + args.chunk_duration_sec)
        chunks.append(
            {
                "source": "vadr1",
                "original_source": row.get("source"),
                "video": resolve_video_path(row, args),
                "video_id": row.get("video"),
                "video_key": normalized_video_key(row),
                "original_video": original_video_path(row),
                "chunk_start": round(chunk_start, 6),
                "chunk_end": round(chunk_end, 6),
                "gate_label": 0,
                "gate_action": "hold",
                "gate_text": "hold",
                "trigger_reason": "none",
                "trigger_sources": [],
                "original_start": boundary["original_start"],
                "original_end": boundary["original_end"],
                "original_unit": "frame_index",
                "total_frames": row.get("total_frames"),
                "timing_source": timing["timing_source"],
                "timing_reliable": timing["timing_reliable"],
                "fps": timing["fps"],
                "weak_supervision": "anomaly-boundary weak supervision",
            }
        )
    mark_trigger(chunks, 0, "initialization")
    if label == "abnormal" and boundary["valid"]:
        selected: dict[str, int] = {}
        selected["anomaly_start"] = nearest_chunk_index(chunks, boundary["start_sec"])
        selected["anomaly_end"] = nearest_chunk_index(chunks, boundary["end_sec"])
        ignored = set()
        for reason, point in (("anomaly_start", boundary["start_sec"]), ("anomaly_end", boundary["end_sec"])):
            if point is None:
                continue
            chosen_idx = selected[reason]
            mark_trigger(chunks, chosen_idx, reason)
            for idx, chunk in enumerate(chunks):
                distance = distance_to_interval_point(point, chunk["chunk_start"], chunk["chunk_end"])
                if distance <= args.boundary_radius_sec and idx != chosen_idx:
                    ignored.add(idx)
        for idx in ignored:
            if chunks[idx]["gate_label"] != 1:
                chunks[idx]["gate_label"] = IGNORE_INDEX
                chunks[idx]["gate_action"] = "ignore"
                chunks[idx]["gate_text"] = "ignore"
    return chunks


def nearest_chunk_index(chunks: list[dict[str, Any]], point: float | None) -> int:
    if point is None:
        return 0
    return min(
        range(len(chunks)),
        key=lambda idx: (
            distance_to_interval_point(point, chunks[idx]["chunk_start"], chunks[idx]["chunk_end"]),
            idx,
        ),
    )


def mark_trigger(chunks: list[dict[str, Any]], idx: int, reason: str) -> None:
    chunk = chunks[idx]
    sources = chunk.setdefault("trigger_sources", [])
    if reason not in sources:
        sources.append(reason)
    chunk["gate_label"] = 1
    chunk["gate_action"] = "trigger"
    chunk["gate_text"] = "trigger"
    chunk["trigger_reason"] = choose_final_trigger_reason(sources)


def choose_final_trigger_reason(sources: list[str]) -> str:
    for reason in ("anomaly_start", "anomaly_end", "initialization"):
        if reason in sources:
            return reason
    return "none"


def split_video_groups(rows: list[dict[str, Any]], train_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(normalized_video_key(row), []).append(row)
    shuffled_keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(shuffled_keys)
    cut = int(len(shuffled_keys) * train_ratio)
    train_keys = set(shuffled_keys[:cut])
    train_rows = [row for row in rows if normalized_video_key(row) in train_keys]
    val_rows = [row for row in rows if normalized_video_key(row) not in train_keys]
    assert_no_split_leakage(train_rows, val_rows)
    return train_rows, val_rows


def assert_no_split_leakage(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    train_ids = {normalized_video_key(row) for row in train_rows}
    val_ids = {normalized_video_key(row) for row in val_rows}
    overlap = train_ids & val_ids
    if overlap:
        examples = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"video-level split leakage detected: {examples}")


def collect_input_stats(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    labels = Counter(infer_label(row) for row in rows)
    stats["video_labels"] = labels
    stats["unique_video_keys"] = len({normalized_video_key(row) for row in rows})
    stats["duplicate_video_keys"] = sum(count > 1 for count in Counter(normalized_video_key(row) for row in rows).values())
    stats["duplicate_paths"] = sum(count > 1 for count in Counter(str(row.get("path") or "") for row in rows).values())
    stats["missing_think_answer"] = sum(not (row.get("think") or "").strip() or not (row.get("answer") or "").strip() for row in rows)
    stats["video_path_missing"] = sum(not os.path.exists(resolve_video_path(row, args)) for row in rows)
    stats["timing_unreliable"] = sum(not get_video_timing(row, args)["timing_reliable"] for row in rows)
    invalid = Counter()
    for row in rows:
        timing = get_video_timing(row, args)
        boundary = boundary_info(row, timing["fps"], timing["duration_sec"])
        if infer_label(row) == "abnormal" and boundary["missing"]:
            invalid["missing_start_end_abnormal"] += 1
        if boundary["start_lt_0"]:
            invalid["start_lt_0"] += 1
        if boundary["end_gt_video_length"]:
            invalid["end_gt_video_length"] += 1
        if boundary["start_gte_end"]:
            invalid["start_gte_end"] += 1
    stats["boundaries"] = invalid
    return stats


def report(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    stage1_train: list[dict[str, Any]],
    stage1_val: list[dict[str, Any]],
    stage2_train: list[dict[str, Any]],
    stage2_val: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    stats = collect_input_stats(rows, args)
    stage1 = stage1_train + stage1_val
    stage2 = stage2_train + stage2_val
    stage1_labels = Counter(row["answer"] for row in stage1)
    stage2_labels = Counter(row["gate_text"] for row in stage2)
    reasons = Counter(row["trigger_reason"] for row in stage2)
    trigger_sources = Counter(source for row in stage2 for source in row.get("trigger_sources", []))
    trigger_collisions = sum(1 for row in stage2 if len(row.get("trigger_sources", [])) > 1)
    needs_review = sum(1 for row in stage1 if row["needs_review"])
    total_stage2 = max(len(stage2), 1)

    print("=== StreamVAD Weak Supervision Report ===")
    print(f"input_samples: {len(rows)}")
    print(f"unique_video_keys: {stats['unique_video_keys']}")
    print(f"duplicate_video_keys: {stats['duplicate_video_keys']}")
    print(f"duplicate_paths: {stats['duplicate_paths']}")
    print(f"train_videos: {len({normalized_video_key(row) for row in train_rows})}")
    print(f"val_videos: {len({normalized_video_key(row) for row in val_rows})}")
    print(f"train_video_samples: {len(train_rows)}")
    print(f"val_video_samples: {len(val_rows)}")
    print(f"stage1_train_samples: {len(stage1_train)}")
    print(f"stage1_val_samples: {len(stage1_val)}")
    print(f"stage2_train_samples: {len(stage2_train)}")
    print(f"stage2_val_samples: {len(stage2_val)}")
    print(f"normal_videos: {stats['video_labels'].get('normal', 0)}")
    print(f"abnormal_videos: {stats['video_labels'].get('abnormal', 0)}")
    print(f"stage1_normal_samples: {stage1_labels.get('normal', 0)}")
    print(f"stage1_abnormal_samples: {stage1_labels.get('abnormal', 0)}")
    print(f"stage2_trigger: {stage2_labels.get('trigger', 0)} ({stage2_labels.get('trigger', 0) / total_stage2:.4f})")
    print(f"stage2_hold: {stage2_labels.get('hold', 0)} ({stage2_labels.get('hold', 0) / total_stage2:.4f})")
    print(f"stage2_ignore: {stage2_labels.get('ignore', 0)} ({stage2_labels.get('ignore', 0) / total_stage2:.4f})")
    hold = stage2_labels.get("hold", 0)
    trigger = stage2_labels.get("trigger", 0)
    ratio = trigger / hold if hold else float("inf")
    print(f"stage2_trigger_to_hold_ratio: {ratio:.6f}")
    print(f"trigger_initialization: {reasons.get('initialization', 0)}")
    print(f"trigger_anomaly_start: {reasons.get('anomaly_start', 0)}")
    print(f"trigger_anomaly_end: {reasons.get('anomaly_end', 0)}")
    print(f"trigger_source_initialization: {trigger_sources.get('initialization', 0)}")
    print(f"trigger_source_anomaly_start: {trigger_sources.get('anomaly_start', 0)}")
    print(f"trigger_source_anomaly_end: {trigger_sources.get('anomaly_end', 0)}")
    print(f"trigger_collision_chunks: {trigger_collisions}")
    print(f"missing_start_end_abnormal: {stats['boundaries'].get('missing_start_end_abnormal', 0)}")
    print(f"invalid_start_lt_0: {stats['boundaries'].get('start_lt_0', 0)}")
    print(f"invalid_end_gt_video_length: {stats['boundaries'].get('end_gt_video_length', 0)}")
    print(f"invalid_start_gte_end: {stats['boundaries'].get('start_gte_end', 0)}")
    print(f"video_path_missing: {stats['video_path_missing']}")
    print(f"timing_unreliable: {stats['timing_unreliable']}")
    print(f"missing_think_or_answer: {stats['missing_think_answer']}")
    print(f"needs_review: {needs_review}")
    print_weight_recommendations(stage2_train, args)
    print()
    if stats["timing_unreliable"]:
        print("Timing note: this run used fallback timing for at least one sample. Without real videos and ffprobe FPS, generated chunks are suitable for format validation only, not final training.")
        print()
    print("Known limitation: 该 Gate 数据只监督异常开始、异常结束和初始状态，不包含异常事件内部阶段变化，因此目前学习的是异常边界触发，而不是完整的语义观察变化触发。")
    print()
    print_examples(stage1, label="normal", n=5, seed=args.seed)
    print_examples(stage1, label="abnormal", n=5, seed=args.seed + 1)


def print_weight_recommendations(stage2_train: list[dict[str, Any]], args: argparse.Namespace) -> None:
    weights = compute_class_weights(stage2_train, args.weight_strategy, args.manual_class_weights, args.effective_beta)
    print("stage2_weight_strategy:", args.weight_strategy)
    print("recommended_cross_entropy_weight_hold:", f"{weights[0]:.6f}")
    print("recommended_cross_entropy_weight_trigger:", f"{weights[1]:.6f}")
    print("ignore_index:", IGNORE_INDEX)
    print("config_example:", json.dumps({"gate_loss_weight": [round(weights[0], 6), round(weights[1], 6)], "ignore_index": IGNORE_INDEX}, ensure_ascii=False))


def compute_class_weights(
    stage2_rows: list[dict[str, Any]],
    strategy: str,
    manual: str | None,
    beta: float,
) -> tuple[float, float]:
    counts = Counter(row["gate_label"] for row in stage2_rows if row["gate_label"] != IGNORE_INDEX)
    hold = counts.get(0, 0)
    trigger = counts.get(1, 0)
    if manual:
        return parse_manual_class_weights(manual)
    if hold == 0 or trigger == 0:
        return 1.0, 1.0
    if strategy == "manual":
        return 1.0, 1.0
    if strategy == "inverse_frequency":
        return 1.0 / hold, 1.0 / trigger
    if strategy == "normalized_inverse_frequency":
        total = hold + trigger
        return total / (2.0 * hold), total / (2.0 * trigger)
    if strategy == "effective_number":
        hold_eff = (1.0 - beta**hold) / (1.0 - beta)
        trigger_eff = (1.0 - beta**trigger) / (1.0 - beta)
        hold_w = 1.0 / hold_eff
        trigger_w = 1.0 / trigger_eff
        scale = 2.0 / (hold_w + trigger_w)
        return hold_w * scale, trigger_w * scale
    raise ValueError(f"unknown weight strategy: {strategy}")


def parse_manual_class_weights(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--manual-class-weights must use 'hold,trigger' format")
    try:
        hold_weight, trigger_weight = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError("--manual-class-weights values must be numeric") from exc
    if hold_weight <= 0 or trigger_weight <= 0:
        raise ValueError("--manual-class-weights values must be positive")
    return hold_weight, trigger_weight


def print_examples(stage1: list[dict[str, Any]], label: str, n: int, seed: int) -> None:
    rows = [row for row in stage1 if row["answer"] == label]
    rng = random.Random(seed)
    rng.shuffle(rows)
    print(f"--- sample_stage1_{label} ---")
    for row in rows[:n]:
        preview = {
            "video_id": row["video_id"],
            "clip_start": row["clip_start"],
            "clip_end": row["clip_end"],
            "scene_prior": row["scene_prior"][:180],
            "observation": row["observation"][:180],
            "answer": row["answer"],
            "needs_review": row["needs_review"],
        }
        print(json.dumps(preview, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pre-context-sec", type=float, default=2.0)
    parser.add_argument("--post-context-sec", type=float, default=2.0)
    parser.add_argument("--chunk-duration-sec", type=float, default=1.0)
    parser.add_argument("--boundary-radius-sec", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback fps when video probing is unavailable.")
    parser.add_argument("--path-prefix-from", help="Original video path prefix to rewrite, for example /media/wbf/VA-Reasoning-SFT.")
    parser.add_argument("--path-prefix-to", help="Local video path prefix used with --path-prefix-from.")
    parser.add_argument("--require-reliable-timing", action="store_true")
    parser.add_argument(
        "--weight-strategy",
        choices=("manual", "inverse_frequency", "normalized_inverse_frequency", "effective_number"),
        default="normalized_inverse_frequency",
    )
    parser.add_argument("--manual-class-weights", help="Comma-separated hold,trigger weights, for example 0.15,0.85.")
    parser.add_argument("--effective-beta", type=float, default=0.9999)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.chunk_duration_sec <= 0:
        raise ValueError("--chunk-duration-sec must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 0.0 < args.effective_beta < 1.0:
        raise ValueError("--effective-beta must be between 0 and 1")
    if args.manual_class_weights:
        parse_manual_class_weights(args.manual_class_weights)
    if bool(args.path_prefix_from) != bool(args.path_prefix_to):
        raise ValueError("--path-prefix-from and --path-prefix-to must be provided together")

    rows = read_jsonl(args.input_jsonl)
    unreliable_rows = [row for row in rows if not get_video_timing(row, args)["timing_reliable"]]
    if args.require_reliable_timing and unreliable_rows and not args.dry_run:
        examples = ", ".join(str(row.get("video") or row.get("path")) for row in unreliable_rows[:5])
        raise ValueError(f"--require-reliable-timing blocked {len(unreliable_rows)} rows without ffprobe timing; examples: {examples}")

    train_rows, val_rows = split_video_groups(rows, args.train_ratio, args.seed)
    stage1_train = [sample for row in train_rows if (sample := make_stage1(row, args)) is not None]
    stage1_val = [sample for row in val_rows if (sample := make_stage1(row, args)) is not None]
    stage2_train = [chunk for row in train_rows for chunk in make_stage2(row, args)]
    stage2_val = [chunk for row in val_rows for chunk in make_stage2(row, args)]
    report(rows, train_rows, val_rows, stage1_train, stage1_val, stage2_train, stage2_val, args)

    if args.dry_run:
        print("dry_run: no files written")
        return

    write_jsonl(args.output_dir / "streamvad_stage1_train.jsonl", stage1_train)
    write_jsonl(args.output_dir / "streamvad_stage1_val.jsonl", stage1_val)
    write_jsonl(args.output_dir / "streamvad_stage2_train.jsonl", stage2_train)
    write_jsonl(args.output_dir / "streamvad_stage2_val.jsonl", stage2_val)
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
