"""StreamVAD JSONL reader and StreamMind-compatible tokenizer for Stage 1.

Each JSONL record must contain:

    video, clip_start, clip_end, event_start_sec, event_end_sec,
    event_token_fractions, target_text
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from streammind import conversation as conversation_lib
from streammind.constants import IGNORE_INDEX, MMODAL_TOKEN_INDEX


def read_streamvad_jsonl(data_path: str) -> list[dict[str, Any]]:
    """Read a StreamVAD Stage 1 JSONL file.

    Returns a list of rows.  ``data_path`` may be a single ``.jsonl`` file or a
    directory containing ``*.jsonl`` files.
    """
    path = Path(data_path)
    if path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"no .jsonl files found in {data_path}")
    else:
        jsonl_files = [path]

    rows: list[dict[str, Any]] = []
    for jsonl_file in jsonl_files:
        with jsonl_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{jsonl_file}:{line_no}: invalid JSON: {exc}") from exc
                row["_jsonl_path"] = str(jsonl_file)
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def validate_streamvad_row(row: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a required field is missing or invalid."""
    required = (
        "video",
        "clip_start",
        "clip_end",
        "event_start_sec",
        "event_end_sec",
        "event_token_fractions",
        "target_text",
    )
    location = f"{row.get('_jsonl_path', '<unknown>')}:{row.get('_line_no', '?')}"
    missing = [f for f in required if f not in row]
    if missing:
        raise ValueError(f"streamvad row missing fields {missing} at {location}")

    if not os.path.exists(str(row["video"])):
        raise FileNotFoundError(f"video not found at {location}: {row['video']}")

    clip_dur = float(row["clip_end"]) - float(row["clip_start"])
    if clip_dur <= 0:
        raise ValueError(f"invalid clip window at {location}")

    fractions = list(row["event_token_fractions"])
    if len(fractions) == 0:
        raise ValueError(f"event_token_fractions empty at {location}")
    for f in fractions:
        if not 0.0 <= float(f) <= 1.0:
            raise ValueError(f"event_token_fraction {f} not in [0,1] at {location}")

    if not str(row.get("target_text", "")).strip():
        raise ValueError(f"target_text empty at {location}")


STAGE1_HUMAN_PROMPT = (
    "<video>\n"
    "You are an anomaly understanding assistant.\n\n"
    "Given the streaming perception tokens extracted from a video event,\n"
    "analyze the observed behavior.\n\n"
    "Please provide:\n"
    "1. What is happening in the scene.\n"
    "2. What behavior patterns are observed.\n"
    "3. Whether the observed behavior deviates from normal patterns and why.\n"
    "4. The final anomaly decision token.\n\n"
    "If no abnormal behavior is observed, clearly state that the behavior is normal.\n"
    "Do not hallucinate or invent anomalies without sufficient visual evidence."
)


def preprocess_llama_2_streamvad(
    target_text: str,
    video_path: str,
    timestamp: list[float],
    tokenizer: Any,
    data_type: str,
) -> dict[str, Any]:
    """Tokenize a single StreamVAD clip for Stage 1.

    Follows the same conversation template and masking strategy as
    ``preprocess_llama_2_soccer``, but accepts a single ``target_text``
    instead of a caption list.
    """
    MODAL_list = ["VIDEO"]
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    sources = [[
        {"from": "human", "value": STAGE1_HUMAN_PROMPT},
        {"from": "gpt", "value": target_text},
    ]]

    conversations = []
    for source in sources:
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    import torch
    from streammind.mm_utils import tokenizer_MMODAL_token

    input_ids = torch.stack(
        [
            tokenizer_MMODAL_token(
                prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[i]], return_tensors="pt"
            )
            for i, prompt in enumerate(conversations)
        ],
        dim=0,
    )

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(
                tokenizer_MMODAL_token(rou, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[0]])
            )
            instruction_len = (
                len(tokenizer_MMODAL_token(parts[0], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[0]])) - 2
            )
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return {
        "labels": targets,
        "video": None,
        "input_ids": input_ids,
        "timestamp": timestamp,
        "caption_info": [target_text],
        "video_path": video_path,
        "past_review_caption": None,
        "data_type": data_type,
        "model_type": "llm",
    }


def compute_event_token_indices(
    clip_start: float,
    clip_end: float,
    event_start: float,
    event_end: float,
    num_frames: int,
    fractions: list[float],
) -> list[int]:
    """Map event time fractions to frame indices in the decoded clip."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if clip_end <= clip_start:
        raise ValueError("clip_end must be greater than clip_start")
    if not (clip_start <= event_start < event_end <= clip_end):
        raise ValueError("event range must be inside clip range")

    clip_duration = clip_end - clip_start
    event_duration = max(event_end - event_start, 1e-6)
    indices: list[int] = []
    for frac in fractions:
        frac = float(frac)
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"event token fraction must be in [0, 1], got {frac}")
        event_time = event_start + event_duration * frac
        relative = (event_time - clip_start) / clip_duration
        indices.append(max(0, min(num_frames - 1, round(relative * (num_frames - 1)))))
    return indices
