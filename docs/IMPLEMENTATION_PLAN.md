# StreamVAD Implementation Plan

## Current Architecture

The repository is organized as a thin StreamVAD layer around two upstream code drops:

- `StreamMind/`: upstream streaming VLM training/inference prototype.
- `Vad-R1/`: vendored VAD reasoning annotations and reference scripts.
- `streamvad/`: local StreamVAD data loaders and StreamMind batch adapters.
- `tools/`: local data construction, smoke checks, and Stage1 training wrapper.
- `scripts/`: server-side execution entrypoints for setup, data building, training, inference, and evaluation.
- `docs/`: audit notes, data schema notes, and implementation plans.

Upstream commits recorded during this pass:

- StreamMind: `d873dc5559d3bb9457882fe92c5898449cb4d8d4`
- Vad-R1 vendored state in parent repository: `85fc8553c813db48dcd2fb26982b1c581714b3b5`

`Vad-R1/` is not an independent git checkout in this workspace, so the exact local state is recorded by the parent repository commit that introduced the vendored files.

`STREAMVAD_CODEX_TASK.md` was required by `AGENTS.md`, but it is not present in the workspace. Work continued from the user request and the repository-local instructions.

## Stage1 Goal

Start from StreamMind Stage1 reproduction, then replace its soccer caption supervision with StreamVAD abnormal-video supervision derived from VAD-R1.

Stage1 v0 is abnormal-event semantic alignment. It uses only abnormal rows with validated `[start, end]` boundaries. CLIP sees the sampled clip, EPFE/Mamba produces a perception-token sequence, and the LLM receives only three ordered event perception tokens selected from the anomaly start/middle/end phases.

The Stage1 language model target is:

```text
<think>
The clip shows ...
The behavior is abnormal because ...
</think>
<answer>
Abnormal
</answer>
```

Event token selection:

- Map VAD-R1 frame `start/end` to seconds using validated FPS.
- Sample the clip with the configured Stage1 video frame count.
- Select token positions at `0.1L`, `0.5L`, and `0.9L` within the sampled anomaly interval.
- Keep the three visual tokens in order instead of mean-pooling them.

## Implemented Local Path

1. `tools/build_streamvad_weak_supervision.py`
   - Parses VAD-R1 JSONL.
   - Builds abnormal-only Stage1 compact CoT rows with event boundaries and `target_text`.
   - Builds Stage2 gate rows with semantic `gate_action` values: `hold`, `trigger`, `ignore`.
   - Keeps `gate_label` as a configured class id for loss code: `hold=0`, `trigger=1`, `ignore=-100`.

2. `streamvad/data/datasets.py`
   - Provides Stage1 and Stage2 datasets.
   - Supports mock/no-video execution with `require_video_exists=False`.
   - Validates Stage1 event ranges are inside clip ranges.
   - Validates Stage2 `gate_action` and `gate_label` agreement.

3. `streamvad/data/streammind_adapter.py`
   - Builds raw StreamMind-style batches for local smoke checks.
   - Keeps Stage1 prompt aligned with event-token CoT output.
   - Preserves Stage2 semantic gate actions.

4. `tools/train_streamvad_stage1_lora.py`
   - Runtime-patches StreamMind's data module so Stage1 training can consume StreamVAD JSONL while leaving upstream StreamMind files unchanged.
   - Runtime-patches visual-token insertion so the LLM receives only the three ordered event perception tokens.

5. `tests/test_streamvad_data.py`
   - Covers mock Stage1 conversion.
   - Covers event range validation and event token index computation.
   - Covers hold/trigger construction.
   - Covers gate action/class-id mismatch rejection.
   - Covers CLI mock execution.

## Stage2 Design Boundary

Stage2 is the cognition gate. It must answer only:

- `hold`: keep current cognition state.
- `trigger`: invoke the cognition model because new evidence may change the current interpretation.

The gate must not directly classify normal vs abnormal. The current generated Stage2 rows satisfy this interface at the data level, but the StreamMind cls trainer is not yet patched to consume `gate_action` rows. `scripts/server_train_streamvad_stage2_gate.sh` is therefore a placeholder that documents the required connection instead of launching an incorrect training job.

## Server Execution Order

1. Set up environment:

```bash
scripts/server_setup_streamvad.sh
```

2. Build derived JSONL after real videos are available:

```bash
VADR1_JSONL=Vad-R1/data/Vad-Reasoning-SFT-train.jsonl \
PATH_PREFIX_FROM=/original/vadr1/video/root \
PATH_PREFIX_TO=/server/vadr1/video/root \
REQUIRE_RELIABLE_TIMING=1 \
scripts/server_build_streamvad_data.sh
```

3. Run a small Stage1 overfit/smoke job:

```bash
MODEL_PATH=/path/to/VideoLLaMA2-7B \
VISION_TOWER=/path/to/clip-vit-large-patch14-336 \
MAX_SAMPLES=16 \
tools/run_streamvad_stage1_lora_smoke.sh
```

4. Run full Stage1:

```bash
MODEL_PATH=/path/to/VideoLLaMA2-7B \
VISION_TOWER=/path/to/clip-vit-large-patch14-336 \
scripts/server_train_streamvad_stage1.sh
```

5. Stage2, inference, and evaluation:

```bash
scripts/server_train_streamvad_stage2_gate.sh
scripts/server_infer_streamvad_stage1.sh
scripts/server_eval_streamvad_stage1.sh
```

These last three are explicit server-side placeholders until the model-runner and cls-trainer adapters are implemented.

## Expected Outputs

- Data build:
  - `data/streamvad_weak_supervision/streamvad_stage1_train.jsonl`
  - `data/streamvad_weak_supervision/streamvad_stage1_val.jsonl`
  - `data/streamvad_weak_supervision/streamvad_stage2_train.jsonl`
  - `data/streamvad_weak_supervision/streamvad_stage2_val.jsonl`

- Stage1 training:
  - LoRA checkpoints under `output/streamvad_stage1_lora` unless `OUTPUT_DIR` overrides it.
  - Generated targets should use `<think>...</think><answer>Abnormal</answer>`.

## Remaining Work

- Implement a StreamVAD Stage2 cls adapter that trains hold/trigger without using tokenizer token ids.
- Add Stage1 inference JSONL writer and CoT/answer evaluation.
- Add Stage2 trigger metrics: precision, recall, F1, boundary latency, duplicate trigger rate.
