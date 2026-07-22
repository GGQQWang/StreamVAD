# Phase 2 Implementation Plan

Scope: next phase only. This plan covers data parsing, compact P2C conversion, and hold/trigger data construction. It intentionally does not modify model architecture, training code, or inference code.

## Goals

- Build a unified parser for upstream StreamMind-style caption boundaries and Vad-R1-style anomaly reasoning JSONL.
- Convert Vad-R1 full-video P2C into compact, prefix-safe supervision records.
- Construct hold/trigger labels suitable for a StreamMind-like Gate without inheriting hardcoded token ids or full-video leakage.

## Non-Goals

- No model changes.
- No tokenizer changes.
- No training launch.
- No dataset download.
- No checkpoint conversion.
- No GPU/runtime environment setup.

## 1. Data Parsing

### Inputs

Support local metadata files only:

- Vad-R1 SFT JSONL rows with fields confirmed in `Vad-R1/data/Vad-Reasoning-SFT-train.jsonl`:
  - `source`
  - `video`
  - `anomaly_type`
  - `path`
  - `total_frames`
  - `think`
  - `answer`
  - optional `start`, `end` for abnormal videos
- StreamMind-style caption boundary rows or derived rows:
  - video path/id
  - ordered timestamp or frame boundary list
  - caption text per boundary

### Parser Outputs

Create an intermediate schema, not tied to either upstream project:

```json
{
  "sample_id": "string",
  "source": "string",
  "video": "string",
  "video_path": "string",
  "total_frames": 0,
  "segments": [
    {
      "start_frame": 0,
      "end_frame": 0,
      "label": "hold|trigger",
      "text": "optional compact target"
    }
  ],
  "anomaly": {
    "type": "Normal|Abnormal|category",
    "start_frame": null,
    "end_frame": null
  },
  "reasoning": {
    "think": "string",
    "answer": "string",
    "p2c": {}
  }
}
```

Implementation notes:

- Parse `think` and `answer` with tag-aware extraction, not ad hoc substring splitting.
- Validate abnormal rows require `start`, `end`, and `total_frames`.
- Validate normal rows may omit `start/end`; represent normal anomaly range as `null`, not `[0,total_frames]`.
- Preserve original text fields for traceability.
- Emit explicit parse errors with `sample_id`, `video`, and missing/malformed field names.

## 2. Compact P2C Conversion

### Required Tags

For abnormal Vad-R1 records, extract:

- `step1`: scene description
- `step2`: abnormal event description
- `step3`: abnormal recognition
- `step4`: causal/social reasoning
- `which`
- `what`
- `when`
- `where`
- `why`
- `how`

For normal Vad-R1 records, extract:

- `step1`
- `step2`
- `which`
- `what`
- `why`

### Compact Format

Produce a concise P2C target that avoids future-only phrasing when used before the observation point:

Abnormal compact target:

```text
<p2c>
<scene>...</scene>
<event>...</event>
<type>...</type>
<time>[s,e]</time>
<place>...</place>
<reason>...</reason>
</p2c>
```

Normal compact target:

```text
<p2c>
<scene>...</scene>
<event>normal activity</event>
<type>Normal</type>
<reason>...</reason>
</p2c>
```

Rules:

- Keep compact text short enough for prompt/state use.
- Do not include `how` in early prefix targets unless the prefix has reached or passed the anomaly end frame.
- Do not expose full-video `step4` before the anomaly event has completed.
- Keep original `when` only in records whose observation window reaches `end_frame`; for earlier prefixes use relative status such as `not_yet_observable`, `onset_observable`, or omit time target.

## 3. Hold/Trigger Construction

### Labels

Use semantic labels in generated data:

- `hold`: no response should be emitted at this frame/window.
- `trigger`: response should be emitted at this frame/window.

Do not write numeric token ids such as `32000` or `32001` into data artifacts. Numeric ids must be resolved later from tokenizer special tokens if a training implementation needs them.

### From StreamMind-Style Boundaries

For a sequence of caption boundary intervals:

- Frames/windows before the next caption boundary are `hold`.
- The last sampled frame/window at the boundary is `trigger`.
- The trigger target text is the caption or compact P2C assigned to that boundary.

This mirrors the audited behavior of `StreamMind/streammind/model/multimodal_projector/builder.py:Video_Mamba_seq.forward`, but keeps labels semantic.

### From Vad-R1 Anomaly Rows

For abnormal rows:

- Before `start_frame`: `hold`.
- Around `start_frame`: optional weak `trigger_candidate` if we want onset detection.
- At or after `end_frame`: `trigger` with compact abnormal P2C.
- Between start and end: use `hold` or `in_event_hold` unless we explicitly decide to support partial event narration.

For normal rows:

- All sampled windows are `hold`, unless a periodic normal summary dataset is deliberately constructed.
- If normal summaries are needed, create separate summary triggers at deterministic endpoints and mark them as a different task type, not anomaly triggers.

### Sampling

Initial deterministic sampling plan:

- Convert frame ranges from `start/end/total_frames`.
- Sample fixed windows at a configurable FPS or frame stride.
- Ensure each abnormal sample includes:
  - pre-event negative windows
  - onset/event windows
  - post-event trigger window
- Ensure normal samples include balanced hold windows from early/middle/late portions.

Hard-negative plan:

- For abnormal videos, sample visually adjacent pre-event windows as `hold`.
- For normal videos, sample high-motion or scene-change windows as `hold` when scene-change metadata is available.

## 4. Leakage Controls

Mandatory checks:

- A record whose window end is before anomaly `end_frame` must not contain final full-video `answer`.
- A record whose window end is before anomaly `start_frame` must not contain anomaly category, `what`, `where`, `why`, or `how`.
- Full `think+answer` from Vad-R1 may only be used as source text for post-event compact targets.
- Keep `source_text_span` or `derived_from` metadata so each compact target can be traced to original tags.

Reject or quarantine:

- Abnormal rows with missing or invalid `start/end`.
- Rows where parsed `<when>` disagrees materially with `start/end/total_frames`.
- Rows where required tags are missing or malformed.

## 5. Deliverables For Next Phase

Recommended new artifacts:

- `tools/parse_vadr1_jsonl.py`: parse and validate Vad-R1 rows into intermediate JSONL.
- `tools/build_compact_p2c.py`: extract compact P2C fields from parsed rows.
- `tools/build_hold_trigger_data.py`: construct semantic hold/trigger windows.
- `docs/DATA_SCHEMA.md`: document intermediate and output schemas.
- Small fixture files under `tests/fixtures/` made from synthetic or tiny hand-written rows, not copied full datasets.

Recommended tests:

- Tag parser handles normal and abnormal CoT.
- Missing `start/end` on abnormal rows fails validation.
- Normal rows without `start/end` pass validation.
- Compact P2C omits future-only fields for pre-event windows.
- Hold/trigger builder never emits numeric token ids.
- Boundary construction marks exactly one trigger per configured post-event target.

## 6. Open Decisions

- Whether trigger should fire at anomaly onset, anomaly end, or both.
- Whether compact P2C should be generated by extraction only or allow a later model-assisted summarization pass.
- Window size and FPS for initial CPU-only preprocessing.
- Whether normal videos should ever produce summary triggers, or remain pure hold examples for Gate training.
