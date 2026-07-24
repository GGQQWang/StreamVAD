#!/usr/bin/env python3
"""Train StreamVAD Stage 1 with StreamMind's trainer and a StreamVAD JSONL.

This script does not edit StreamMind source files. It patches the training data
module at runtime so StreamMind can consume StreamVAD Stage 1 records:

    video, clip_start, clip_end, event_start_sec, event_end_sec, target_text

The wrapper also patches the multimodal prepare function at runtime so only the
three ordered event perception tokens are inserted into the LLM prompt.
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

    _patch_streamvad_event_token_selection()

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
            include_metadata=False,
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
        data_dict["event_token_indices"] = compute_event_token_indices(
            clip_start=row["clip_start"],
            clip_end=row["clip_end"],
            event_start=row["event_start_sec"],
            event_end=row["event_end_sec"],
            num_frames=processed["images"][0][0].shape[0],
            fractions=row["event_token_fractions"],
        )
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
            "event_token_indices": instance["event_token_indices"],
        }


def compute_event_token_indices(
    *,
    clip_start: float,
    clip_end: float,
    event_start: float,
    event_end: float,
    num_frames: int,
    fractions: list[float],
) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if clip_end <= clip_start:
        raise ValueError("clip_end must be greater than clip_start")
    if not (clip_start <= event_start < event_end <= clip_end):
        raise ValueError("event range must be inside clip range")

    clip_duration = clip_end - clip_start
    event_duration = event_end - event_start
    indices: list[int] = []
    for fraction in fractions:
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"event token fraction must be in [0, 1], got {fraction}")
        event_time = event_start + event_duration * fraction
        relative = (event_time - clip_start) / clip_duration
        indices.append(max(0, min(num_frames - 1, round(relative * (num_frames - 1)))))
    return indices


def _patch_streamvad_event_token_selection() -> None:
    import torch
    from streammind.constants import IGNORE_INDEX, MMODAL_TOKEN_INDEX
    from streammind.model.videollama2_arch import Videollama2MetaForCausalLM

    if getattr(Videollama2MetaForCausalLM, "_streamvad_event_patch", False):
        return

    original = Videollama2MetaForCausalLM.prepare_inputs_labels_for_multimodal_score_stream

    def prepare_with_event_tokens(
        self: Any,
        input_ids: Any,
        attention_mask: Any,
        past_key_values: Any,
        labels: Any,
        X_modalities: Any,
        timestamp: Any,
        sample_per: float = 0.5,
        sample_type: str = "all",
        **kwargs: Any,
    ):
        event_token_indices = kwargs.pop("event_token_indices", None)
        if event_token_indices is None:
            return original(
                self,
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                X_modalities,
                timestamp,
                sample_per=sample_per,
                sample_type=sample_type,
                **kwargs,
            )

        model_type = kwargs.pop("model_type", None)
        data_type = kwargs.pop("data_type", None)
        if model_type == "cls":
            kwargs["model_type"] = model_type
            kwargs["data_type"] = data_type
            return original(
                self,
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                X_modalities,
                timestamp,
                sample_per=sample_per,
                sample_type=sample_type,
                **kwargs,
            )

        Xs, keys = X_modalities
        X_features, feature_idx = self.encode_images_or_videos_score_cls_video_cls_autoregressive(
            Xs,
            cls_training=False,
            cls_inference=False,
            prompt_time_input_ids=input_ids,
            prompt_time_lable=labels,
        )
        start_feature_idx = [0] + feature_idx[:-1]
        normalized_event_indices = _normalize_event_indices(event_token_indices)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            X_token_indices = _find_video_token_indices(cur_input_ids, keys, MMODAL_TOKEN_INDEX, torch)
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while X_token_indices.numel() > 0:
                cur_X_features = X_features[0][start_feature_idx[cur_X_idx] : feature_idx[cur_X_idx]]
                event_indices = normalized_event_indices[min(cur_X_idx, len(normalized_event_indices) - 1)]
                cur_X_features = _select_event_features(cur_X_features, event_indices, torch)

                X_token_start = X_token_indices[0]
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start]))
                cur_new_input_embeds.append(cur_X_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:X_token_start])
                    cur_new_labels.append(
                        torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
                    )
                    cur_labels = cur_labels[X_token_start + 1 :]

                cur_X_idx += 1
                cur_input_ids = cur_input_ids[X_token_start + 1 :]
                X_token_indices = _find_video_token_indices(cur_input_ids, keys, MMODAL_TOKEN_INDEX, torch)

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            new_input_embeds.append(torch.cat(cur_new_input_embeds, dim=0))
            if labels is not None:
                new_labels.append(torch.cat(cur_new_labels, dim=0))

        new_input_embeds, new_labels, attention_mask = _pad_streammind_embeds(
            new_input_embeds,
            new_labels,
            attention_mask,
            labels,
            input_ids,
            IGNORE_INDEX,
            torch,
        )
        return None, attention_mask, past_key_values, new_input_embeds, new_labels, None

    Videollama2MetaForCausalLM.prepare_inputs_labels_for_multimodal_score_stream = prepare_with_event_tokens
    Videollama2MetaForCausalLM._streamvad_event_patch = True


def _find_video_token_indices(cur_input_ids: Any, keys: list[str], mmodal_token_index: dict[str, int], torch_module: Any) -> Any:
    return torch_module.where(
        torch_module.any(
            torch_module.stack([cur_input_ids == mmodal_token_index[key.upper()] for key in keys]),
            dim=0,
        )
    )[0]


def _normalize_event_indices(event_token_indices: Any) -> list[list[int]]:
    if hasattr(event_token_indices, "detach"):
        event_token_indices = event_token_indices.detach().cpu().tolist()
    if event_token_indices and isinstance(event_token_indices[0], int):
        return [list(event_token_indices)]
    return [list(indices) for indices in event_token_indices]


def _select_event_features(cur_X_features: Any, event_indices: list[int], torch_module: Any) -> Any:
    if cur_X_features.shape[0] == 0:
        raise ValueError("cannot select event tokens from an empty visual feature sequence")
    clipped = [max(0, min(cur_X_features.shape[0] - 1, int(index))) for index in event_indices]
    index_tensor = torch_module.tensor(clipped, dtype=torch_module.long, device=cur_X_features.device)
    return cur_X_features.index_select(0, index_tensor)


def _pad_streammind_embeds(
    new_input_embeds: list[Any],
    new_labels: list[Any] | None,
    attention_mask: Any,
    labels: Any,
    input_ids: Any,
    ignore_index: int,
    torch_module: Any,
) -> tuple[Any, Any, Any]:
    if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
        max_len = max(x.shape[0] for x in new_input_embeds)
        aligned_embeds = []
        for cur_new_embed in new_input_embeds:
            pad = torch_module.zeros(
                (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                dtype=cur_new_embed.dtype,
                device=cur_new_embed.device,
            )
            aligned_embeds.append(torch_module.cat((cur_new_embed, pad), dim=0))
        new_input_embeds = torch_module.stack(aligned_embeds, dim=0)

        if new_labels is not None:
            raw_labels = new_labels
            aligned_labels = []
            for cur_new_label in new_labels:
                pad = torch_module.full(
                    (max_len - cur_new_label.shape[0],),
                    ignore_index,
                    dtype=cur_new_label.dtype,
                    device=cur_new_label.device,
                )
                aligned_labels.append(torch_module.cat((cur_new_label, pad), dim=0))
            new_labels = torch_module.stack(aligned_labels, dim=0)
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_raw_label, cur_aligned_label in zip(attention_mask, raw_labels, new_labels):
                    left = torch_module.full(
                        (cur_raw_label.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    right = torch_module.full(
                        (cur_aligned_label.shape[0] - cur_raw_label.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attention_mask.append(torch_module.cat((left, cur_attention_mask, right), dim=0))
                attention_mask = torch_module.stack(new_attention_mask, dim=0)
    else:
        new_input_embeds = torch_module.stack(new_input_embeds, dim=0)
        if new_labels is not None:
            new_labels = torch_module.stack(new_labels, dim=0)
        if attention_mask is not None:
            left = torch_module.full(
                (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch_module.cat((left, attention_mask), dim=1)
        else:
            attention_mask = torch_module.full(
                (new_input_embeds.shape[0], new_input_embeds.shape[1]),
                1,
                device=new_input_embeds.device,
            )
    return new_input_embeds, new_labels, attention_mask


if __name__ == "__main__":
    main()
