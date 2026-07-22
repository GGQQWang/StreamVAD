"""PyTorch-style datasets for StreamVAD weak-supervision JSONL files.

The datasets intentionally expose only training-safe fields by default. Audit
fields such as original CoT text, original boundaries, trigger reasons, and
video ids are kept out of returned samples unless ``include_metadata=True`` is
requested for debugging.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


IGNORE_INDEX = -100
GATE_LABELS = {0, 1, IGNORE_INDEX}


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no}: invalid JSON: {exc}") from exc
            row["_jsonl_path"] = str(jsonl_path)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def read_many_jsonl(paths: Iterable[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _require_fields(row: dict[str, Any], fields: tuple[str, ...], dataset_name: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
        raise ValueError(f"{dataset_name}: missing fields {missing} at {location}")


def _as_float(row: dict[str, Any], field: str, dataset_name: str) -> float:
    value = row.get(field)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
        raise ValueError(f"{dataset_name}: {field} must be numeric at {location}, got {value!r}") from exc
    if not math.isfinite(number):
        location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
        raise ValueError(f"{dataset_name}: {field} must be finite at {location}, got {value!r}")
    return number


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _validate_window(start: float, end: float, row: dict[str, Any], dataset_name: str) -> None:
    if start < 0:
        location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
        raise ValueError(f"{dataset_name}: window start < 0 at {location}")
    if end <= start:
        location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
        raise ValueError(f"{dataset_name}: window end <= start at {location}")


def _load_video_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int,
) -> Any:
    """Decode a video segment with torchvision and sample frames uniformly.

    ``torchvision`` is imported lazily so metadata-only debugging can run in a
    lightweight environment. Returned frames are shaped ``T,H,W,C`` as uint8.
    Training code can permute/normalize them according to the visual encoder.
    """
    try:
        import torch
        from torchvision.io import read_video
    except ImportError as exc:
        raise ImportError("decode_video=True requires torch and torchvision") from exc

    try:
        frames, _, _ = read_video(video_path, start_pts=start_sec, end_pts=end_sec, pts_unit="sec")
    except Exception:
        return _load_video_clip_ffmpeg(video_path, start_sec, end_sec, num_frames)
    if frames.numel() == 0:
        raise ValueError(f"no frames decoded from {video_path} between {start_sec} and {end_sec}")
    if num_frames <= 0:
        return frames
    indices = torch.linspace(0, frames.shape[0] - 1, steps=num_frames).round().long()
    return frames.index_select(0, indices)


def _probe_video_size(video_path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        video_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
    width_text, height_text = result.stdout.strip().split("x", 1)
    return int(width_text), int(height_text)


def _load_video_clip_ffmpeg(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int,
) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("decode_video=True requires torch") from exc

    if num_frames <= 0:
        raise ValueError("ffmpeg fallback requires num_frames > 0")
    duration = max(end_sec - start_sec, 1e-3)
    width, height = _probe_video_size(video_path)
    fps = max(num_frames / duration, 1e-3)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        video_path,
        "-vf",
        f"fps={fps:.8f},format=rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    frame_size = width * height * 3
    if frame_size <= 0 or len(result.stdout) < frame_size:
        raise ValueError(f"ffmpeg decoded no frames from {video_path} between {start_sec} and {end_sec}")
    usable = len(result.stdout) - (len(result.stdout) % frame_size)
    frames = torch.frombuffer(bytearray(result.stdout[:usable]), dtype=torch.uint8)
    frames = frames.reshape(-1, height, width, 3)
    if frames.shape[0] == num_frames:
        return frames
    indices = torch.linspace(0, frames.shape[0] - 1, steps=num_frames).round().long()
    return frames.index_select(0, indices)


class StreamVADStage1Dataset:
    """Dataset for StreamVAD Stage 1 compact text supervision.

    Default returned fields are exactly:

    - ``video``
    - ``clip_start``
    - ``clip_end``
    - ``target_text``
    - optional ``frames`` when ``decode_video=True``
    """

    REQUIRED_FIELDS = ("video", "clip_start", "clip_end", "target_text")

    def __init__(
        self,
        jsonl_paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
        *,
        decode_video: bool = False,
        num_frames: int = 32,
        require_video_exists: bool = True,
        require_reliable_timing: bool = False,
        drop_needs_review: bool = False,
        include_metadata: bool = False,
    ) -> None:
        if isinstance(jsonl_paths, (str, os.PathLike)):
            paths = [jsonl_paths]
        else:
            paths = list(jsonl_paths)
        rows = read_many_jsonl(paths)
        if drop_needs_review:
            rows = [row for row in rows if not _as_bool(row.get("needs_review"))]
        self.rows = rows
        self.decode_video = decode_video
        self.num_frames = num_frames
        self.require_video_exists = require_video_exists
        self.require_reliable_timing = require_reliable_timing
        self.include_metadata = include_metadata
        self._validate_rows()

    def _validate_rows(self) -> None:
        for row in self.rows:
            _require_fields(row, self.REQUIRED_FIELDS, self.__class__.__name__)
            clip_start = _as_float(row, "clip_start", self.__class__.__name__)
            clip_end = _as_float(row, "clip_end", self.__class__.__name__)
            _validate_window(clip_start, clip_end, row, self.__class__.__name__)
            target_text = str(row.get("target_text") or "").strip()
            if not target_text:
                location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
                raise ValueError(f"{self.__class__.__name__}: target_text is empty at {location}")
            if self.require_reliable_timing and not _as_bool(row.get("timing_reliable")):
                location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
                raise ValueError(f"{self.__class__.__name__}: unreliable timing at {location}")
            if self.require_video_exists and not os.path.exists(str(row["video"])):
                location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
                raise FileNotFoundError(f"{self.__class__.__name__}: video not found at {location}: {row['video']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        clip_start = _as_float(row, "clip_start", self.__class__.__name__)
        clip_end = _as_float(row, "clip_end", self.__class__.__name__)
        sample: dict[str, Any] = {
            "video": str(row["video"]),
            "clip_start": clip_start,
            "clip_end": clip_end,
            "target_text": str(row["target_text"]),
        }
        if self.decode_video:
            sample["frames"] = _load_video_clip(str(row["video"]), clip_start, clip_end, self.num_frames)
        if self.include_metadata:
            sample["metadata"] = _stage1_debug_metadata(row)
        return sample


class StreamVADStage2GateDataset:
    """Dataset for StreamVAD Stage 2 Cognition Gate supervision.

    Default returned fields are exactly:

    - ``video``
    - ``chunk_start``
    - ``chunk_end``
    - ``gate_label``
    - optional ``frames`` when ``decode_video=True``
    """

    REQUIRED_FIELDS = ("video", "chunk_start", "chunk_end", "gate_label")

    def __init__(
        self,
        jsonl_paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
        *,
        decode_video: bool = False,
        num_frames: int = 8,
        require_video_exists: bool = True,
        require_reliable_timing: bool = False,
        include_ignore: bool = True,
        include_metadata: bool = False,
    ) -> None:
        if isinstance(jsonl_paths, (str, os.PathLike)):
            paths = [jsonl_paths]
        else:
            paths = list(jsonl_paths)
        rows = read_many_jsonl(paths)
        if not include_ignore:
            rows = [row for row in rows if int(row.get("gate_label")) != IGNORE_INDEX]
        self.rows = rows
        self.decode_video = decode_video
        self.num_frames = num_frames
        self.require_video_exists = require_video_exists
        self.require_reliable_timing = require_reliable_timing
        self.include_metadata = include_metadata
        self._validate_rows()

    def _validate_rows(self) -> None:
        for row in self.rows:
            _require_fields(row, self.REQUIRED_FIELDS, self.__class__.__name__)
            chunk_start = _as_float(row, "chunk_start", self.__class__.__name__)
            chunk_end = _as_float(row, "chunk_end", self.__class__.__name__)
            _validate_window(chunk_start, chunk_end, row, self.__class__.__name__)
            label = int(row["gate_label"])
            if label not in GATE_LABELS:
                location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
                raise ValueError(f"{self.__class__.__name__}: invalid gate_label {label} at {location}")
            if self.require_reliable_timing and not _as_bool(row.get("timing_reliable")):
                location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
                raise ValueError(f"{self.__class__.__name__}: unreliable timing at {location}")
            if self.require_video_exists and not os.path.exists(str(row["video"])):
                location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
                raise FileNotFoundError(f"{self.__class__.__name__}: video not found at {location}: {row['video']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        chunk_start = _as_float(row, "chunk_start", self.__class__.__name__)
        chunk_end = _as_float(row, "chunk_end", self.__class__.__name__)
        sample: dict[str, Any] = {
            "video": str(row["video"]),
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "gate_label": int(row["gate_label"]),
        }
        if self.decode_video:
            sample["frames"] = _load_video_clip(str(row["video"]), chunk_start, chunk_end, self.num_frames)
        if self.include_metadata:
            sample["metadata"] = _stage2_debug_metadata(row)
        return sample


def _stage1_debug_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": row.get("video_id"),
        "video_key": row.get("video_key"),
        "answer": row.get("answer"),
        "needs_review": row.get("needs_review"),
        "timing_reliable": row.get("timing_reliable"),
    }


def _stage2_debug_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": row.get("video_id"),
        "video_key": row.get("video_key"),
        "gate_text": row.get("gate_text"),
        "trigger_reason": row.get("trigger_reason"),
        "trigger_sources": row.get("trigger_sources"),
        "timing_reliable": row.get("timing_reliable"),
    }


def collate_stage1(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "video": [sample["video"] for sample in samples],
        "clip_start": [sample["clip_start"] for sample in samples],
        "clip_end": [sample["clip_end"] for sample in samples],
        "target_text": [sample["target_text"] for sample in samples],
    }
    if samples and "frames" in samples[0]:
        batch["frames"] = _stack_frames([sample["frames"] for sample in samples])
    if samples and "metadata" in samples[0]:
        batch["metadata"] = [sample["metadata"] for sample in samples]
    return batch


def collate_stage2_gate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [sample["gate_label"] for sample in samples]
    batch: dict[str, Any] = {
        "video": [sample["video"] for sample in samples],
        "chunk_start": [sample["chunk_start"] for sample in samples],
        "chunk_end": [sample["chunk_end"] for sample in samples],
        "gate_label": _tensor_long(labels),
    }
    if samples and "frames" in samples[0]:
        batch["frames"] = _stack_frames([sample["frames"] for sample in samples])
    if samples and "metadata" in samples[0]:
        batch["metadata"] = [sample["metadata"] for sample in samples]
    return batch


def _tensor_long(values: list[int]) -> Any:
    try:
        import torch
    except ImportError:
        return values
    return torch.tensor(values, dtype=torch.long)


def _stack_frames(frames: list[Any]) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("collating decoded frames requires torch") from exc
    return torch.stack(frames, dim=0)
