#!/usr/bin/env python3
"""Inference: run Stage 1 checkpoint on validation JSONL and compute accuracy.

Usage::

  python tools/infer_stage1_val.py \
    --checkpoint /path/to/checkpoint \
    --val-jsonl /path/to/val.jsonl \
    --streammind-root StreamMind \
    --max-samples 20

Requires GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--streammind-root", type=Path, default=REPO_ROOT / "StreamMind")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=32)
    return p.parse_args()


def load_model(checkpoint_dir: Path, streammind_root: Path) -> Any:
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from peft import PeftModel

    sys.path.insert(0, str(streammind_root))
    from streammind.model.language_model.videollama2_mistral import (
        Videollama2MistralForCausalLM,
    )
    from streammind.model.videollama2_arch import Videollama2MetaModel

    # Delete old projector + create mamba (same as training patch)
    original_init = Videollama2MetaModel.initialize_vision_modules

    def mega_init(self, model_args, fsdp=None):
        import gc

        if hasattr(self, "mm_projector") and self.mm_projector is not None:
            self.mm_projector.to("cpu")
            del self.mm_projector
            gc.collect()
            torch.cuda.empty_cache()
        original_init(self, model_args, fsdp=fsdp)
        if hasattr(self.mm_projector, "cls_net") and self.mm_projector.cls_net is not None:
            self.mm_projector.cls_net = None
            torch.cuda.empty_cache()

    Videollama2MetaModel.initialize_vision_modules = mega_init

    # Load base model
    cfg = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    cfg._attn_implementation = "flash_attention_2"

    model = Videollama2MistralForCausalLM.from_pretrained(
        str(checkpoint_dir),
        config=cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    # Init vision + mamba projector
    model.get_model().initialize_vision_modules(
        type(
            "Args",
            (),
            {
                "vision_tower": str(checkpoint_dir),
                "mm_projector_type": "mamba",
                "mm_vision_select_layer": -2,
                "mm_vision_select_feature": "patch",
                "mm_use_im_start_end": False,
                "mm_use_im_patch_token": False,
                "pretrain_mm_mlp_adapter": None,
                "tune_mm_mlp_adapter": False,
            },
        )(),
    )

    # Move vision tower + mamba projector to GPU
    vt = model.get_vision_tower()
    vt.to(dtype=torch.bfloat16, device=0)
    for p in vt.parameters():
        p.requires_grad = False
    model.get_model().mm_projector.to(device=0, dtype=torch.bfloat16)

    # Load EPFE (mamba projector) weights from non_lora_trainables.bin
    nlt_path = checkpoint_dir / "non_lora_trainables.bin"
    if nlt_path.exists():
        nlt = torch.load(nlt_path, map_location="cpu")
        nlt = {(k[11:] if k.startswith("base_model.") else k): v for k, v in nlt.items()}
        if any(k.startswith("model.model.") for k in nlt):
            nlt = {(k[6:] if k.startswith("model.") else k): v for k, v in nlt.items()}
        model.load_state_dict(nlt, strict=False)
        print("non_lora_trainables loaded.")

    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # Add video modal token (same as training)
    from streammind.constants import DEFAULT_MMODAL_TOKEN
    tokenizer.add_tokens(list(DEFAULT_MMODAL_TOKEN.values()), special_tokens=True)

    return model, tokenizer, vt.image_processor


def decord_load_video(video_path: str, start_sec: float, end_sec: float, num_frames: int, image_processor: Any) -> Any:
    """Load video with decord (matching training pipeline)."""
    import numpy as np
    from PIL import Image
    from decord import VideoReader, cpu
    from streammind.mm_utils import expand2square

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    video_fps = float(vr.get_avg_fps())
    start_frame = int(max(0, start_sec * video_fps - 1))
    end_frame = int(end_sec * video_fps + 1)
    if end_frame <= start_frame or start_sec >= end_sec:
        raise ValueError(f"invalid video segment: {video_path} [{start_sec},{end_sec}]")

    seg_size = max(1, int(video_fps / 2))  # cur_fps=2 like training
    frame_ids = np.arange(start_frame, min(end_frame, len(vr)), seg_size, dtype=int)
    frames_arr = vr.get_batch(frame_ids).asnumpy()
    if frames_arr.shape[0] == 0:
        raise ValueError(f"no frames decoded from {video_path}")

    bg = tuple(int(c * 255) for c in image_processor.image_mean[:3])
    images = []
    for f in frames_arr:
        img = Image.fromarray(f)
        img = expand2square(img, bg)
        images.append(img)
    if len(images) > num_frames:
        import torch
        indices = torch.linspace(0, len(images) - 1, steps=num_frames).round().long()
        images = [images[i] for i in indices]
    pixel_values = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return pixel_values


def generate_text(
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    pixel_values: Any,
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    import torch

    # Tokenize prompt
    from streammind.constants import MMODAL_TOKEN_INDEX
    from streammind.mm_utils import tokenizer_MMODAL_token

    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX["VIDEO"], return_tensors="pt")
    input_ids = input_ids.to("cuda:0")
    pixel_values = pixel_values.to(device="cuda:0", dtype=model.dtype)

    # Run CLIP in chunks
    import einops

    VT_CHUNK = 8
    all_features = []
    for i in range(0, pixel_values.shape[0], VT_CHUNK):
        chunk = pixel_values[i : i + VT_CHUNK]
        with torch.no_grad():
            feat = model.get_model().get_vision_tower()(chunk)
        all_features.append(feat)
    frames_features = torch.cat(all_features, dim=0)
    frames_features = einops.rearrange(frames_features, "(b t) n h -> b t n h", b=1)

    # Run mamba projector
    from itertools import accumulate

    shape = [frames_features.shape[1]]
    shape = list(accumulate(shape))
    with torch.no_grad():
        proj_out = model.get_model().mm_projector(
            frames_features,
            cls_training=False,
            cls_inference=False,
            frames_features_shape=shape,
        )

    # Find <video> token position and replace with projected features
    video_idx = (input_ids == MMODAL_TOKEN_INDEX["VIDEO"]).nonzero(as_tuple=True)[1]
    if video_idx.numel() > 0:
        vpos = video_idx[0].item()
        prefix_embeds = model.get_model().embed_tokens(input_ids[:, :vpos])
        suffix_embeds = model.get_model().embed_tokens(input_ids[:, vpos + 1 :])
        inputs_embeds = torch.cat([prefix_embeds, proj_out[0], suffix_embeds], dim=1)
    else:
        inputs_embeds = model.get_model().embed_tokens(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def extract_decision(text: str) -> str | None:
    m = re.search(r"<answer>\s*(Normal|Abnormal)\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fallback: last occurrence
    if "normal" in text.lower()[-200:]:
        return "normal"
    if "abnormal" in text.lower()[-200:]:
        return "abnormal"
    return None


def main():
    import torch

    args = parse_args()
    if str(args.streammind_root) not in sys.path:
        sys.path.insert(0, str(args.streammind_root))

    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, image_processor = load_model(args.checkpoint, args.streammind_root)
    print("Model loaded.")

    from data.streamvad_data import STAGE1_HUMAN_PROMPT, read_streamvad_jsonl, validate_streamvad_row

    rows = read_streamvad_jsonl(str(args.val_jsonl))
    n = min(args.max_samples or len(rows), len(rows))
    print(f"Val samples: {n}")

    correct = 0
    total = 0
    skipped = 0
    by_label = {"normal": [0, 0], "abnormal": [0, 0]}

    for i in range(n):
        row = rows[i]
        gt = row.get("answer", "").lower()

        try:
            pixel_values = decord_load_video(
                str(row["video"]),
                float(row["clip_start"]),
                float(row["clip_end"]),
                args.num_frames,
                image_processor,
            )
            output = generate_text(
                model, tokenizer, image_processor,
                pixel_values=pixel_values,
                prompt=STAGE1_HUMAN_PROMPT,
            )
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            skipped += 1
            continue

        pred = extract_decision(output)
        total += 1
        if gt in by_label:
            by_label[gt][0] += 1
        if pred == gt:
            correct += 1
            if gt in by_label:
                by_label[gt][1] += 1
            status = "OK"
        else:
            status = f"WRONG (pred={pred}, gt={gt})"

        # Print preview every 10th
        if i % 10 == 0 or pred != gt:
            print(f"  [{i}] {status}")
            print(f"       gt={gt}  output={output[-120:].replace(chr(10), ' ')}")

    print()
    print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
    for label, (tc, cc) in by_label.items():
        print(f"  {label}: {cc}/{tc} = {cc / tc:.2%}" if tc else f"  {label}: 0 samples")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
