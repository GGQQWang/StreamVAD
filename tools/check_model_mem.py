#!/usr/bin/env python3
"""Server-side memory probe for the StreamVAD Stage1 base model stack.

This script intentionally does not contain machine-specific paths. It only
loads the base VideoLLaMA2 model and CLIP vision tower supplied by arguments,
then reports CUDA memory after each step. Use it before a training run when
you need a quick capacity check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streammind-root", type=Path, default=REPO_ROOT / "StreamMind")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--projector-type", default="mamba")
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def show_memory(tag: str, torch_module: object) -> None:
    cuda = torch_module.cuda
    alloc = cuda.memory_allocated() / 1e9
    reserved = cuda.memory_reserved() / 1e9
    peak = cuda.max_memory_allocated() / 1e9
    print(f"[{tag}] alloc={alloc:.1f}GB reserved={reserved:.1f}GB peak={peak:.1f}GB")


def main() -> None:
    args = parse_args()
    if not args.streammind_root.exists():
        raise FileNotFoundError(f"StreamMind root not found: {args.streammind_root}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"model path not found: {args.model_path}")
    if not args.vision_tower.exists():
        raise FileNotFoundError(f"vision tower path not found: {args.vision_tower}")

    sys.path.insert(0, str(args.streammind_root))

    import torch
    from transformers import AutoConfig, BitsAndBytesConfig
    from streammind.model.language_model.videollama2_mistral import (
        Videollama2MistralForCausalLM,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this server-side memory probe")

    show_memory("start", torch)
    cfg = AutoConfig.from_pretrained(str(args.model_path), trust_remote_code=True)
    if args.flash_attn:
        cfg._attn_implementation = "flash_attention_2"

    model_kwargs = {
        "config": cfg,
        "torch_dtype": torch.float16,
        "device_map": {"": 0},
    }
    if args.load_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = Videollama2MistralForCausalLM.from_pretrained(
        str(args.model_path),
        **model_kwargs,
    )
    show_memory("base model loaded", torch)
    print(f"base params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    model.get_model().initialize_vision_modules(
        type(
            "Args",
            (),
            {
                "vision_tower": str(args.vision_tower),
                "mm_projector_type": args.projector_type,
                "mm_vision_select_layer": -2,
                "mm_vision_select_feature": "patch",
                "mm_use_im_start_end": False,
                "mm_use_im_patch_token": False,
                "pretrain_mm_mlp_adapter": None,
                "tune_mm_mlp_adapter": False,
            },
        )(),
    )
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.float16, device=0)
    show_memory("vision tower loaded", torch)
    print(f"vision params: {sum(p.numel() for p in vision_tower.parameters()) / 1e9:.2f}B")


if __name__ == "__main__":
    main()
