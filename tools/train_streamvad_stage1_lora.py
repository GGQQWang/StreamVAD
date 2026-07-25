#!/usr/bin/env python3
"""Train StreamVAD Stage 1 with StreamMind's trainer using StreamVAD JSONL data.

StreamVAD Stage 1 aligns the EPFE streaming perception module with abnormal
event descriptions.  Three ordered event-perception tokens (event start /
mid / end) are inserted into the LLM prompt so the model can attend to the
temporal structure of an anomaly.

This script does NOT edit StreamMind source files.  It patches only the
multimodal prepare function at runtime to use event-aware token selection.

Usage (server, 1 GPU)::

    torchrun --nproc_per_node=1 tools/train_streamvad_stage1_lora.py \\
        --streammind-root StreamMind \\
        --flash-attn \\
        --streamvad_dataset True \\
        --data_path /path/to/streamvad_stage1_train.jsonl \\
        --data_type train \\
        --model_name_or_path /path/to/base/model \\
        --vision_tower /path/to/clip-vit-large-patch14-336 \\
        --mm_projector_type mamba \\
        --lora_enable True \\
        --lora_r 64 --lora_alpha 16 \\
        --num_train_epochs 3 \\
        --per_device_train_batch_size 1 \\
        --gradient_accumulation_steps 8 \\
        --bf16 True \\
        --output_dir /path/to/output
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

    sys.argv = [sys.argv[0], *remaining_argv]

    import streammind.train_new_stream as train_new_stream

    _patch_streamvad_encoder()
    _patch_streamvad_event_token_selection()

    if custom_args.streamvad_max_samples is not None:
        _patch_streamvad_max_samples(custom_args.streamvad_max_samples)

    train_new_stream.train(
        attn_implementation="flash_attention_2" if custom_args.flash_attn else None
    )


def _parse_custom_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--streammind-root",
        type=Path,
        default=REPO_ROOT / "StreamMind",
        help="Path to the cloned StreamMind repository.",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use StreamMind's flash_attention_2 training path.",
    )
    parser.add_argument(
        "--streamvad-max-samples",
        type=int,
        default=None,
        help="Cap number of samples for smoke/overfit runs (applied after dataset init).",
    )
    return parser.parse_known_args(argv)


# ---------------------------------------------------------------------------
# max-samples cap for smoke tests
# ---------------------------------------------------------------------------


def _patch_streamvad_max_samples(max_samples: int) -> None:
    from data.datasets import LazySupervisedDataset

    if getattr(LazySupervisedDataset, "_streamvad_max_samples_patch", False):
        return

    original_init = LazySupervisedDataset.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if hasattr(self, "streamvad_rows"):
            self.streamvad_rows = self.streamvad_rows[:max_samples]

    LazySupervisedDataset.__init__ = patched_init
    LazySupervisedDataset._streamvad_max_samples_patch = True


# ---------------------------------------------------------------------------
# 3-token event insertion patch
# ---------------------------------------------------------------------------


def _patch_streamvad_encoder() -> None:
    """Prevent None kwargs from being passed to mm_projector.forward."""
    from streammind.model.videollama2_arch import Videollama2MetaForCausalLM

    if getattr(Videollama2MetaForCausalLM, "_streamvad_encoder_patch", False):
        return

    original = Videollama2MetaForCausalLM.encode_images_or_videos_score_cls_video_cls_autoregressive

    def safe_encoder(
        self: Any,
        images_or_videos: Any,
        cls_inference: bool = False,
        cls_training: bool = False,
        caption_info: Any = None,
        prompt_time_input_ids: Any = None,
        prompt_time_lable: Any = None,
    ) -> Any:
        from itertools import accumulate
        import einops
        import torch as _torch

        frames_features_list = []
        frames_features_shape = []
        for images_or_video in images_or_videos:
            num_frames = images_or_video.shape[0]
            videos = images_or_video.unsqueeze(0)
            assert len(videos.size()) == 5
            frames = einops.rearrange(videos, "b t c h w -> (b t) c h w")
            if frames.shape[0] > 600:
                frames = frames[-600:]
            frames_features = self.get_model().get_vision_tower()(frames)
            frames_features = einops.rearrange(
                frames_features, "(b t) n h -> b t n h", b=videos.size(0)
            )
            frames_features_list.append(frames_features)
            frames_features_shape.append(frames_features.shape[1])

        frames_features_shape = list(accumulate(frames_features_shape))
        frames_features = _torch.cat(frames_features_list, dim=1)

        # Only pass kwargs that are not None
        exactor_kwargs: dict[str, Any] = dict(
            cls_inference=cls_inference,
            cls_training=cls_training,
            frames_features_shape=frames_features_shape,
        )
        if prompt_time_input_ids is not None:
            exactor_kwargs["prompt_time_input_ids"] = prompt_time_input_ids
        if prompt_time_lable is not None:
            exactor_kwargs["prompt_time_lable"] = prompt_time_lable

        exactor_output = self.temporal_aggregator(frames_features, **exactor_kwargs)
        return exactor_output, frames_features_shape

    Videollama2MetaForCausalLM.encode_images_or_videos_score_cls_video_cls_autoregressive = safe_encoder
    Videollama2MetaForCausalLM._streamvad_encoder_patch = True


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
                self, input_ids, attention_mask, past_key_values, labels,
                X_modalities, timestamp, sample_per=sample_per, sample_type=sample_type,
                **kwargs,
            )

        model_type = kwargs.pop("model_type", None)
        data_type = kwargs.pop("data_type", None)
        if model_type == "cls":
            kwargs["model_type"] = model_type
            kwargs["data_type"] = data_type
            return original(
                self, input_ids, attention_mask, past_key_values, labels,
                X_modalities, timestamp, sample_per=sample_per, sample_type=sample_type,
                **kwargs,
            )

        Xs, keys = X_modalities
        X_features, feature_idx = (
            self.encode_images_or_videos_score_cls_video_cls_autoregressive(
                Xs,
                cls_training=False,
                cls_inference=False,
            )
        )
        start_feature_idx = [0] + feature_idx[:-1]
        normalized_event_indices = _normalize_event_indices(event_token_indices)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            X_token_indices = _find_video_token_indices(
                cur_input_ids, keys, MMODAL_TOKEN_INDEX, torch
            )
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while X_token_indices.numel() > 0:
                cur_X_features = X_features[0][
                    start_feature_idx[cur_X_idx] : feature_idx[cur_X_idx]
                ]
                event_indices = normalized_event_indices[
                    min(cur_X_idx, len(normalized_event_indices) - 1)
                ]
                cur_X_features = _select_event_features(
                    cur_X_features, event_indices, torch
                )

                X_token_start = X_token_indices[0]
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids[:X_token_start])
                )
                cur_new_input_embeds.append(cur_X_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:X_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_X_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_labels = cur_labels[X_token_start + 1:]

                cur_X_idx += 1
                cur_input_ids = cur_input_ids[X_token_start + 1:]
                X_token_indices = _find_video_token_indices(
                    cur_input_ids, keys, MMODAL_TOKEN_INDEX, torch
                )

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids)
                )
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            new_input_embeds.append(torch.cat(cur_new_input_embeds, dim=0))
            if labels is not None:
                new_labels.append(torch.cat(cur_new_labels, dim=0))

        new_input_embeds, new_labels, attention_mask = _pad_streammind_embeds(
            new_input_embeds, new_labels, attention_mask,
            labels, input_ids, IGNORE_INDEX, torch,
        )
        return None, attention_mask, past_key_values, new_input_embeds, new_labels, None

    Videollama2MetaForCausalLM.prepare_inputs_labels_for_multimodal_score_stream = (
        prepare_with_event_tokens
    )
    Videollama2MetaForCausalLM._streamvad_event_patch = True


def _find_video_token_indices(
    cur_input_ids: Any,
    keys: list[str],
    mmodal_token_index: dict[str, int],
    torch_module: Any,
) -> Any:
    return torch_module.where(
        torch_module.any(
            torch_module.stack(
                [cur_input_ids == mmodal_token_index[key.upper()] for key in keys]
            ),
            dim=0,
        )
    )[0]


def _normalize_event_indices(event_token_indices: Any) -> list[list[int]]:
    if hasattr(event_token_indices, "detach"):
        event_token_indices = event_token_indices.detach().cpu().tolist()
    if event_token_indices and isinstance(event_token_indices[0], int):
        return [list(event_token_indices)]
    return [list(indices) for indices in event_token_indices]


def _select_event_features(
    cur_X_features: Any, event_indices: list[int], torch_module: Any
) -> Any:
    if cur_X_features.shape[0] == 0:
        raise ValueError("cannot select event tokens from empty visual feature sequence")
    clipped = [
        max(0, min(cur_X_features.shape[0] - 1, int(index)))
        for index in event_indices
    ]
    index_tensor = torch_module.tensor(
        clipped, dtype=torch_module.long, device=cur_X_features.device
    )
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
                for cur_attention_mask, cur_raw_label, cur_aligned_label in zip(
                    attention_mask, raw_labels, new_labels
                ):
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
                    new_attention_mask.append(
                        torch_module.cat((left, cur_attention_mask, right), dim=0)
                    )
                attention_mask = torch_module.stack(new_attention_mask, dim=0)
    else:
        new_input_embeds = torch_module.stack(new_input_embeds, dim=0)
        if new_labels is not None:
            new_labels = torch_module.stack(new_labels, dim=0)
        if attention_mask is not None:
            left = torch_module.full(
                (
                    attention_mask.shape[0],
                    new_input_embeds.shape[1] - input_ids.shape[1],
                ),
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
