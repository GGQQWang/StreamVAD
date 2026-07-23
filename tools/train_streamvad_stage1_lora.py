#!/usr/bin/env python3
"""Train StreamVAD Stage 1 with StreamMind's trainer and a StreamVAD JSONL.

This script does not modify StreamMind model code. It patches only the training
data module at runtime so StreamMind can consume StreamVAD Stage 1 records:

    video, clip_start, clip_end, target_text

The model, LoRA injection, EPFE/Mamba projector, and checkpoint saving behavior
remain StreamMind's original implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    custom_args, remaining_argv = _parse_custom_args(sys.argv[1:])
    streammind_root = custom_args.streammind_root.resolve()
    if not streammind_root.exists():
        raise FileNotFoundError(f"StreamMind root not found: {streammind_root}")
    if str(streammind_root) not in sys.path:
        sys.path.insert(0, str(streammind_root))

    # StreamMind's training code parses sys.argv itself with HfArgumentParser.
    sys.argv = [sys.argv[0], *remaining_argv]

    import streammind.train_new_stream as train_new_stream

    train_new_stream.make_supervised_stream_data_module = _build_streamvad_data_module_factory(
        max_samples=custom_args.streamvad_max_samples,
        require_reliable_timing=not custom_args.allow_unreliable_timing,
    )
    train_new_stream.train(attn_implementation="flash_attention_2" if custom_args.flash_attn else None)


def _parse_custom_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--streammind-root",
        type=Path,
        default=REPO_ROOT / "StreamMind",
        help="Path to the cloned StreamMind repository.",
    )
    parser.add_argument(
        "--streamvad-max-samples",
        type=int,
        default=None,
        help="Optional cap for tiny smoke/overfit runs.",
    )
    parser.add_argument(
        "--allow-unreliable-timing",
        action="store_true",
        help="Allow samples whose timing_reliable flag is false.",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use StreamMind's flash_attention_2 training path.",
    )
    return parser.parse_known_args(argv)


def _build_streamvad_data_module_factory(*, max_samples: int | None, require_reliable_timing: bool):
    def make_streamvad_data_module(tokenizer: Any, data_args: Any) -> dict[str, Any]:
        dataset = StreamVADStage1StreamMindDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            image_processor=data_args.video_processor,
            num_frames=data_args.num_frames or 32,
            data_type=data_args.data_type,
            max_samples=max_samples,
            require_reliable_timing=require_reliable_timing,
        )
        return {
            "train_dataset": dataset,
            "eval_dataset": None,
            "data_collator": StreamVADStage1StreamMindCollator(),
        }

    return make_streamvad_data_module


class StreamVADStage1StreamMindDataset:
    """StreamMind-compatible wrapper around StreamVAD Stage 1 JSONL records."""

    def __init__(
        self,
        *,
        data_path: str,
        tokenizer: Any,
        image_processor: Any,
        num_frames: int,
        data_type: str,
        max_samples: int | None,
        require_reliable_timing: bool,
    ) -> None:
        from streamvad.data import StreamVADStage1Dataset

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_type = data_type
        self.dataset = StreamVADStage1Dataset(
            data_path,
            decode_video=True,
            require_video_exists=True,
            require_reliable_timing=require_reliable_timing,
            num_frames=num_frames,
            include_audit_fields=False,
        )
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("--streamvad-max-samples must be positive")
            self.indices = list(range(min(max_samples, len(self.dataset))))
        else:
            self.indices = list(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from data.soccer_data import preprocess_llama_2_soccer
        from streamvad.data import build_streammind_stage1_batch

        row = self.dataset[self.indices[index]]
        processed = build_streammind_stage1_batch(
            [row],
            image_processor=self.image_processor,
            expand_square=True,
        )
        data_dict = preprocess_llama_2_soccer(
            caption_data=[row["target_text"]],
            video_data=row["video"],
            timestamp=[row["clip_end"]],
            tokenizer=self.tokenizer,
            data_type=self.data_type,
        )
        data_dict["video"] = [processed["images"][0][0]]
        data_dict["timestamp"] = [row["clip_end"]]
        data_dict["caption_info"] = [row["target_text"]]
        data_dict["video_path"] = row["video"]
        data_dict["past_review_caption"] = None
        data_dict["data_type"] = self.data_type
        data_dict["model_type"] = "llm"
        return data_dict


class StreamVADStage1StreamMindCollator:
    """Single-video collator matching StreamMind's streaming batch contract."""

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        if len(instances) != 1:
            raise ValueError("StreamMind streaming Stage 1 currently expects per-device batch size 1")
        instance = instances[0]
        return {
            "timestamp": instance["timestamp"],
            "labels": instance["labels"],
            "input_ids": instance["input_ids"],
            "caption_info": instance["caption_info"],
            "video_path": instance["video_path"],
            "images": [instance["video"], ["video"]],
            "attention_mask": None,
            "past_review_caption": instance["past_review_caption"],
            "data_type": instance["data_type"],
            "model_type": instance["model_type"],
        }


if __name__ == "__main__":
    main()
