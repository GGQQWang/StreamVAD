"""Adapters from StreamVAD loader samples to StreamMind-style batch objects."""

from __future__ import annotations

from typing import Any


DEFAULT_STAGE1_PROMPT = (
    "<video>\n"
    "Describe the scene prior, the visible observation, and whether the video is normal or abnormal."
)


def build_streammind_stage1_batch(
    samples: list[dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    image_processor: Any | None = None,
    prompt: str = DEFAULT_STAGE1_PROMPT,
) -> dict[str, Any]:
    """Build a Stage 1 batch without exposing audit-only fields.

    When ``tokenizer`` is omitted, the batch contains raw prompt and target text
    for smoke tests. When a tokenizer is supplied later, this function can be
    extended to emit StreamMind ``input_ids`` and language-model labels.
    """
    videos = [_maybe_process_frames(sample.get("frames"), image_processor) for sample in samples]
    batch: dict[str, Any] = {
        "images": [videos, ["video"] * len(videos)],
        "video": [sample["video"] for sample in samples],
        "clip_start": [sample["clip_start"] for sample in samples],
        "clip_end": [sample["clip_end"] for sample in samples],
        "prompt": [prompt for _ in samples],
        "target_text": [sample["target_text"] for sample in samples],
        "task": "streamvad_stage1",
    }
    if tokenizer is not None:
        batch.update(_tokenize_stage1_text(batch["prompt"], batch["target_text"], tokenizer))
    return batch


def build_streammind_stage2_gate_batch(
    samples: list[dict[str, Any]],
    *,
    image_processor: Any | None = None,
) -> dict[str, Any]:
    """Build a Stage 2 Gate batch with ``gate_labels`` preserved.

    Returned labels may contain ``-100`` and must be used with
    ``ignore_index=-100`` in the Gate loss.
    """
    videos = [_maybe_process_frames(sample.get("frames"), image_processor) for sample in samples]
    labels = [int(sample["gate_label"]) for sample in samples]
    return {
        "images": [videos, ["video"] * len(videos)],
        "video": [sample["video"] for sample in samples],
        "chunk_start": [sample["chunk_start"] for sample in samples],
        "chunk_end": [sample["chunk_end"] for sample in samples],
        "gate_labels": _tensor_long(labels),
        "task": "streamvad_stage2_gate",
    }


def _maybe_process_frames(frames: Any, image_processor: Any | None) -> Any:
    if frames is None or image_processor is None:
        return frames
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("image_processor path requires pillow") from exc

    images = [Image.fromarray(frame.cpu().numpy() if hasattr(frame, "cpu") else frame) for frame in frames]
    return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]


def _tokenize_stage1_text(prompts: list[str], targets: list[str], tokenizer: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("tokenized Stage 1 batches require torch") from exc

    input_ids = []
    labels = []
    pad_id = tokenizer.pad_token_id
    ignore_index = -100
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        full_ids = tokenizer(prompt + "\n" + target, add_special_tokens=True).input_ids
        cur_input = torch.tensor(full_ids, dtype=torch.long)
        cur_labels = cur_input.clone()
        cur_labels[: len(prompt_ids)] = ignore_index
        input_ids.append(cur_input)
        labels.append(cur_labels)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=ignore_index)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(pad_id),
    }


def _tensor_long(values: list[int]) -> Any:
    try:
        import torch
    except ImportError:
        return values
    return torch.tensor(values, dtype=torch.long)
