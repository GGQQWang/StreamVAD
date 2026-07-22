#!/usr/bin/env python3
"""Smoke test StreamVAD loaders and StreamMind-style batch adapters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamvad.data import (
    StreamVADStage1Dataset,
    StreamVADStage2GateDataset,
    build_streammind_stage1_batch,
    build_streammind_stage2_gate_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-jsonl", required=True, type=Path)
    parser.add_argument("--stage2-jsonl", required=True, type=Path)
    parser.add_argument("--decode-video", action="store_true")
    parser.add_argument("--require-video-exists", action="store_true")
    parser.add_argument("--require-reliable-timing", action="store_true")
    parser.add_argument("--stage1-num-frames", type=int, default=32)
    parser.add_argument("--stage2-num-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    s1 = StreamVADStage1Dataset(
        args.stage1_jsonl,
        decode_video=args.decode_video,
        require_video_exists=args.require_video_exists,
        require_reliable_timing=args.require_reliable_timing,
        num_frames=args.stage1_num_frames,
    )
    s2 = StreamVADStage2GateDataset(
        args.stage2_jsonl,
        decode_video=args.decode_video,
        require_video_exists=args.require_video_exists,
        require_reliable_timing=args.require_reliable_timing,
        num_frames=args.stage2_num_frames,
    )

    stage1_samples = [s1[idx] for idx in range(min(args.batch_size, len(s1)))]
    stage2_samples = [s2[idx] for idx in range(min(args.batch_size, len(s2)))]
    stage1_batch = build_streammind_stage1_batch(stage1_samples)
    stage2_batch = build_streammind_stage2_gate_batch(stage2_samples)

    print("=== Stage 1 ===")
    print("dataset_len:", len(s1))
    print("sample_keys:", sorted(stage1_samples[0]))
    print("batch_keys:", sorted(stage1_batch))
    print("images_modalities:", stage1_batch["images"][1])
    print("first_video:", stage1_batch["video"][0])
    print("first_clip:", stage1_batch["clip_start"][0], stage1_batch["clip_end"][0])
    print("first_target_preview:", stage1_batch["target_text"][0][:180].replace("\n", "\\n"))
    if args.decode_video:
        print("first_frames_shape:", tuple(stage1_samples[0]["frames"].shape))

    print("=== Stage 2 ===")
    print("dataset_len:", len(s2))
    print("sample_keys:", sorted(stage2_samples[0]))
    print("batch_keys:", sorted(stage2_batch))
    print("images_modalities:", stage2_batch["images"][1])
    print("first_video:", stage2_batch["video"][0])
    print("first_chunk:", stage2_batch["chunk_start"][0], stage2_batch["chunk_end"][0])
    print("gate_labels:", stage2_batch["gate_labels"])
    if args.decode_video:
        print("first_frames_shape:", tuple(stage2_samples[0]["frames"].shape))


if __name__ == "__main__":
    main()
