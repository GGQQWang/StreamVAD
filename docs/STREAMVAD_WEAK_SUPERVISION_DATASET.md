# StreamVAD Weak Supervision Dataset

This document describes the first StreamVAD weak-supervision dataset generated from VAD-R1 SFT JSONL annotations. It is a derived data prototype for StreamVAD. It does not use VAD-R1 model weights, VAD-R1 GRPO training, or any StreamMind model-structure changes.

## Source Data

The converter is:

- `tools/build_streamvad_weak_supervision.py`

The intended inputs are VAD-R1 SFT JSONL files:

- `Vad-R1/data/Vad-Reasoning-SFT-train.jsonl`
- `Vad-R1/data/Vad-Reasoning-SFT-test.jsonl`

The local audit found that VAD-R1 SFT rows contain `source`, `video`, `anomaly_type`, `path`, `think`, `answer`, and `total_frames`. Abnormal rows also contain `start` and `end`.

`start` and `end` are frame indices. They are converted to seconds by real video FPS when `ffprobe` can read the video. If the video is unavailable, the converter can fall back to `total_frames / --fps`, but that fallback is only acceptable for format validation.

## Split Rule

The converter now splits before data expansion:

```text
VAD-R1 video-level rows
-> group by normalized video key
-> split train/val by video key
-> generate Stage 1 and Stage 2 rows inside each split
```

The normalized video key uses the original `video` field when present, otherwise the normalized original path. The script checks for train/val leakage and raises an error if the same video key appears in both splits.

The current VAD-R1 SFT train JSONL has no duplicate `video` or `path` entries, but the script still reports duplicate video keys and duplicate paths.

## Stage 1 Compact CoT

Stage 1 rows provide compact text supervision:

```text
Scene prior:
<normal scene behavior or rules>

Observation:
<visible people, actions, and events>

Answer:
Normal or Abnormal
```

Important rules:

- `scene_prior` and `observation` are extracted from VAD-R1 `think` and `answer`.
- The converter does not generate observation text from the filename or `anomaly_type`.
- Long consequences, social impact language, repeated descriptions, and weakly visible reasoning are filtered by simple rules.
- If text extraction is unreliable, `needs_review` is set to `true`.
- The original `think` and `answer` are preserved in Stage 1 output for audit.

Key Stage 1 fields:

- `source`: `vadr1`
- `original_source`: original VAD-R1 source directory/dataset label
- `video`: rewritten local video path when path-prefix rewrite is configured
- `original_video`: original VAD-R1 path before rewrite
- `video_id`, `video_key`
- `clip_start`, `clip_end`
- `scene_prior`, `observation`, `answer`, `target_text`
- `original_think`, `original_answer`
- `original_start`, `original_end`, `original_unit`
- `total_frames`, `fps`, `timing_source`, `timing_reliable`
- `needs_review`

For abnormal samples, the video window is:

```text
[start - pre_context_sec, end + post_context_sec]
```

Defaults:

- `--pre-context-sec 2.0`
- `--post-context-sec 2.0`

For normal samples, the current version uses the full short video duration when timing is available.

## Stage 2 Gate Data

Stage 2 is only for Cognition Gate weak supervision:

```text
0 = silence
1 = response
-100 = ignore
```

The data should be called:

```text
anomaly-boundary weak supervision
```

It is not complete observation-change supervision.

Key Stage 2 fields:

- `source`: `vadr1`
- `original_source`: original VAD-R1 source
- `video`: rewritten local video path when path-prefix rewrite is configured
- `original_video`: original VAD-R1 path before rewrite
- `video_id`, `video_key`
- `chunk_start`, `chunk_end`
- `gate_label`: `0`, `1`, or `-100`
- `gate_text`: `silence`, `response`, or `ignore`
- `trigger_reason`: final selected reason, one of `initialization`, `anomaly_start`, `anomaly_end`, `none`
- `trigger_sources`: all trigger sources covered by this chunk
- `original_start`, `original_end`, `original_unit`
- `total_frames`, `fps`, `timing_source`, `timing_reliable`
- `weak_supervision`: `anomaly-boundary weak supervision`

Videos are split into fixed chunks. Default:

```text
--chunk-duration-sec 1.0
```

### Stage 2 Label Rule

For normal videos:

- First valid chunk: `response`, `trigger_reason = initialization`.
- Later chunks: `silence`, `trigger_reason = none`.

For abnormal videos:

- First valid chunk: `response`.
- Exactly one chunk nearest to `start` is selected as `response`.
- Exactly one chunk nearest to `end` is selected as `response`.
- Boundary-near chunks that were not selected are `ignore`, not `silence`.
- Other stable chunks are `silence`.

Boundary-near ambiguity is controlled by:

```text
--boundary-radius-sec 1.0
```

If initialization, anomaly start, and anomaly end fall in the same chunk, only one row is emitted. The single-value `trigger_reason` records the final selected reason, and `trigger_sources` records all covered trigger sources.

The `ignore` label must be excluded from gate training loss with:

```text
ignore_index = -100
```

## Timing Reliability

The converter emits:

- `timing_source`: `ffprobe`, `total_frames_over_fps`, or `unknown`
- `timing_reliable`: `true` only when the local video exists and `ffprobe` obtains real FPS/timing

Formal Stage 2 training data should use:

```bash
--require-reliable-timing
```

With this option, non-dry-run generation raises an error if any row lacks reliable `ffprobe` timing. In the current no-video workspace, generation without this flag is only for format validation.

## Class Weights

Stage 2 is highly imbalanced, so the first version keeps weighted cross-entropy rather than downsampling silence. The old StreamMind-style `0.15/0.85` weights should not be assumed correct for this data.

The converter reports training-set class weights while excluding `-100` ignore rows. Supported strategies:

- `--manual-class-weights silence,response`
- `--weight-strategy inverse_frequency`
- `--weight-strategy normalized_inverse_frequency`
- `--weight-strategy effective_number`

Example reported config:

```json
{"gate_loss_weight": [0.506535, 38.754727], "ignore_index": -100}
```

Validation should not rely on accuracy alone. At minimum, record:

- response precision
- response recall
- response F1
- whether each abnormal start boundary is triggered at least once
- average trigger latency
- duplicate trigger count

## Compact CoT Audit

The audit sampler is:

- `tools/audit_streamvad_compact_cot.py`

It reads Stage 1 converted JSONL and writes a human-review table. Default sampling is 50 normal and 50 abnormal rows, stratified across `original_source` when possible.

Example:

```bash
python3 tools/audit_streamvad_compact_cot.py \
  --input-jsonl data/streamvad_weak_supervision/streamvad_stage1_train.jsonl \
                data/streamvad_weak_supervision/streamvad_stage1_val.jsonl \
  --output-csv data/streamvad_weak_supervision/compact_cot_audit.csv \
  --output-jsonl data/streamvad_weak_supervision/compact_cot_audit.jsonl \
  --seed 42
```

Audit rows include `video_id`, `video`, `original_think`, `original_answer`, `scene_prior`, `observation`, `answer`, `needs_review`, `review_status`, and `review_note`.

Reviewers should check whether:

- `scene_prior` only describes normal scene rules.
- `scene_prior` accidentally includes abnormal events.
- `observation` is based on visible video facts.
- future outcomes, social consequences, or subjective inference leaked into compact text.
- text is overly long, repeated, empty, or incorrectly truncated.
- normal/abnormal labels match the original answer.
- filename or `anomaly_type` leaked the answer.

Do not hand-edit generated training JSONL. If audit finds a systematic problem, change the converter and regenerate.

## Current Local Run

Command:

```bash
python3 tools/build_streamvad_weak_supervision.py \
  --input-jsonl Vad-R1/data/Vad-Reasoning-SFT-train.jsonl \
  --output-dir data/streamvad_weak_supervision
```

Statistics:

- Input samples: `1755`
- Unique video keys: `1755`
- Duplicate video keys: `0`
- Duplicate paths: `0`
- Train videos: `1579`
- Val videos: `176`
- Stage 1 train samples: `1579`
- Stage 1 val samples: `176`
- Stage 2 train samples: `216404`
- Stage 2 val samples: `28330`
- Normal videos: `941`
- Abnormal videos: `814`
- Stage 1 normal samples: `941`
- Stage 1 abnormal samples: `814`
- Stage 2 response chunks: `3052`
- Stage 2 silence chunks: `238066`
- Stage 2 ignore chunks: `3616`
- Response/silence ratio: `0.012820`
- Final `trigger_reason=initialization`: `1425`
- Final `trigger_reason=anomaly_start`: `814`
- Final `trigger_reason=anomaly_end`: `813`
- `trigger_sources` initialization: `1755`
- `trigger_sources` anomaly_start: `814`
- `trigger_sources` anomaly_end: `814`
- Trigger collision chunks: `331`
- Missing abnormal `start/end`: `0`
- Invalid `start < 0`: `0`
- Invalid `end > video length`: `0`
- Invalid `start >= end`: `0`
- Missing local video paths: `1755`
- Unreliable timing rows: `1755`
- Missing `think` or `answer`: `0`
- `needs_review`: `1`
- Recommended normalized inverse-frequency train weights: silence `0.506535`, response `38.754727`

Because all local video paths are missing, this generated output is a format-validation artifact. It should be regenerated with real videos and `--require-reliable-timing` before formal Stage 2 training.

## Rebuild Commands

Dry-run format check:

```bash
python3 tools/build_streamvad_weak_supervision.py \
  --input-jsonl Vad-R1/data/Vad-Reasoning-SFT-train.jsonl \
  --output-dir data/streamvad_weak_supervision \
  --dry-run
```

Format-validation generation without local videos:

```bash
python3 tools/build_streamvad_weak_supervision.py \
  --input-jsonl Vad-R1/data/Vad-Reasoning-SFT-train.jsonl \
  --output-dir data/streamvad_weak_supervision \
  --pre-context-sec 2.0 \
  --post-context-sec 2.0 \
  --chunk-duration-sec 1.0 \
  --boundary-radius-sec 1.0 \
  --train-ratio 0.9 \
  --seed 42 \
  --fps 30.0
```

Formal generation after videos are available and paths are valid:

```bash
python3 tools/build_streamvad_weak_supervision.py \
  --input-jsonl Vad-R1/data/Vad-Reasoning-SFT-train.jsonl \
  --output-dir data/streamvad_weak_supervision \
  --path-prefix-from /media/wbf/VA-Reasoning-SFT \
  --path-prefix-to /path/to/local/Vad-Reasoning-SFT \
  --pre-context-sec 2.0 \
  --post-context-sec 2.0 \
  --chunk-duration-sec 1.0 \
  --boundary-radius-sec 1.0 \
  --train-ratio 0.9 \
  --seed 42 \
  --weight-strategy normalized_inverse_frequency \
  --require-reliable-timing
```

## Video Files

VAD-R1 JSONL paths use the original author machine root, for example:

```text
/media/wbf/VA-Reasoning-SFT/UCF/Burglary073_x264.mp4
```

On a new machine, download the official VAD-R1 Hugging Face dataset and map:

```text
/media/wbf/VA-Reasoning-SFT
```

to the local video root containing directories such as:

```text
ECVA/
UCF/
NEW/
TAD/
SH/
XD/
UB/
```

The converter supports this mapping without editing the original VAD-R1 JSONL:

```bash
--path-prefix-from /media/wbf/VA-Reasoning-SFT \
--path-prefix-to /path/to/local/Vad-Reasoning-SFT
```

All generated rows preserve the original path in `original_video` and write the resolved local path to `video`.

## Known Limitations

该 Gate 数据只监督异常开始、异常结束和初始状态，不包含异常事件内部阶段变化，因此目前学习的是异常边界触发，而不是完整的语义观察变化触发。

Additional limitations:

- The current local generated files use fallback FPS because no videos are present.
- Formal Stage 2 data must be regenerated with real video FPS and `--require-reliable-timing`.
- Stage 1 compact text is rule-extracted and still needs human audit.
- Normal videos do not contain event boundaries, so only initialization is supervised as response.
- `ignore` chunks require downstream training code to use `ignore_index=-100`; the current work does not modify training code.
- A loader still needs to map `clip_start/clip_end` and `chunk_start/chunk_end` into actual video decoding.
