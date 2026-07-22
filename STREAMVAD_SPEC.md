# StreamVAD Implementation Task

## 0. Execution Directive

You are acting as a senior machine-learning systems engineer.

Implement a complete, reviewable, locally testable codebase for **StreamVAD** by adapting StreamMind and using Vad-R1 as an annotation and teacher-data source.

Do not return only a design or plan. Inspect the upstream code, create an implementation plan, implement the code, write tests, run all locally feasible checks, review the diff, and document server execution.

The current machine is only for code generation and lightweight testing. Heavy data processing and training will be performed manually on a remote Linux GPU server.

Do not locally:

* Download model checkpoints.
* Download the Vad-R1 videos.
* Install CUDA, FlashAttention, DeepSpeed, or large Conda environments.
* Run actual StreamMind or Vad-R1 model training.
* Run full video decoding.
* Assume that a GPU is present.
* Execute any script under `scripts/server/`.

Generate those scripts, but do not run them.

When information is missing, inspect the repository and make conservative, documented assumptions. Ask a question only when implementation is genuinely blocked. Do not stop at the planning stage.

---

# 1. Research Objective

The research hypothesis is:

> Continuous perception does not require continuous cognition.

StreamVAD continuously processes a video stream and maintains a temporally persistent visual state. It does not invoke the large language model at every frame or clip.

The system contains three conceptual components.

## 1.1 Streaming Perception

The StreamMind-style streaming perception module continuously updates a visual state from incoming video frames.

Its responsibility is to preserve:

* Scene context.
* Active entities.
* Ongoing actions.
* Temporal relations.
* Event continuity.
* Short-term evidence across repeated frames.
* Event identity across brief occlusion or irrelevant interruption.

It must not directly emit the final normal/abnormal decision.

## 1.2 Cognition Gate

The gate predicts exactly two semantic classes:

```text
hold
trigger
```

Definitions:

### `hold`

The newly observed evidence does not meaningfully change the existing event interpretation.

Examples include:

* Repeated or near-duplicate frames.
* The same action continuing.
* The same anomalous event continuing.
* A brief occlusion.
* A short black screen.
* A short irrelevant interruption.
* The same event becoming visible again after a brief interruption.
* Small visual motion that does not change the event semantics.

### `trigger`

The newly observed evidence may require the current event interpretation to be updated.

Examples include:

* A new event begins.
* A new subject enters and changes the event.
* A relevant action begins.
* The relation between subjects changes.
* Evidence becomes sufficient to revise an earlier interpretation.
* The event meaningfully changes phase.
* One event is replaced by a different event.
* An ongoing event ends.
* The scene changes in a way that invalidates the previous interpretation.

The gate must not be trained as a normal/abnormal classifier.

Do not implement the semantic substitution:

```text
silence -> normal
response -> abnormal
```

That substitution is conceptually incorrect.

The required interpretation is:

```text
silence/response timing mechanism
        becomes
hold/trigger cognition-update mechanism
```

## 1.3 Cognition Module

Only invoke the cognition module when:

* The gate outputs `trigger`; or
* The stream reaches its initial forced bootstrap point; or
* A stale cognition state must be refreshed after a long interruption.

The cognition output must follow this compact Perception-to-Cognition structure:

```xml
<scene_prior>
The normal behavior expected in the currently observed scene.
</scene_prior>
<observation>
The currently visible entities, actions, relations, and temporal changes.
</observation>
<deviation>
Whether and how the observation deviates from the expected pattern.
</deviation>
<decision>
normal
</decision>
```

or:

```xml
<scene_prior>
...
</scene_prior>
<observation>
...
</observation>
<deviation>
...
</deviation>
<decision>
abnormal
</decision>
```

The required decision vocabulary is:

```text
normal
abnormal
```

Do not require Vad-R1’s longer discussion of social norms, moral evaluation, or potential harm.

The code may preserve the original Vad-R1 text in provenance fields, but those sections are not required cognition targets.

---

# 2. Fixed Prompts

The first version is a generic VAD system.

Do not support arbitrary user prompts in the initial implementation.

Store prompts in configuration files instead of embedding them directly in model code.

## 2.1 Gate Prompt

Use the following default gate prompt:

```text
Continuously monitor the video stream and maintain the current event
interpretation across repeated, occluded, or briefly irrelevant frames.

Output trigger only when newly observed visual evidence may meaningfully
change the current event interpretation. Otherwise, output hold.

A continuing instance of the same event should normally produce hold.
A brief interruption followed by the same event should normally produce
hold. A new event, a meaningful event change, or an event ending should
produce trigger.

Do not determine whether the event is normal or abnormal.
```

## 2.2 Cognition Prompt

Use the following default cognition prompt:

```text
Analyze only the video evidence observed up to the current time.

Return exactly four XML fields:
scene_prior, observation, deviation, and decision.

The decision must be either normal or abnormal.

Do not use future frames. Do not add social-norm discussion, moral
judgment, or speculative consequences unless they are necessary to
distinguish the observed behavior.
```

Store the prompts under:

```text
configs/prompts/gate_prompt.txt
configs/prompts/cognition_prompt.txt
```

---

# 3. Repository Acquisition and Workspace

## 3.1 Desired Layout

Use the following workspace layout:

```text
StreamVAD-workspace/
├── AGENTS.md
├── STREAMVAD_CODEX_TASK.md
├── StreamVAD/
├── third_party/
│   └── Vad-R1/
├── scripts/
│   ├── local/
│   └── server/
├── reproducibility/
└── README_WORKSPACE.md
```

`StreamVAD/` is the working repository derived from StreamMind.

`third_party/Vad-R1/` is a read-only reference repository. Do not modify it.

## 3.2 Generate Repository Setup Script

Create:

```text
scripts/local/00_clone_upstreams.sh
```

It must perform the equivalent of:

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:-$(pwd)}"

mkdir -p "${WORKSPACE_ROOT}/third_party"
mkdir -p "${WORKSPACE_ROOT}/reproducibility"

if [[ ! -d "${WORKSPACE_ROOT}/StreamVAD/.git" ]]; then
  git clone https://github.com/xinding-bot/StreamMind.git \
    "${WORKSPACE_ROOT}/StreamVAD"
fi

if [[ ! -d "${WORKSPACE_ROOT}/third_party/Vad-R1/.git" ]]; then
  git clone https://github.com/wbfwonderful/Vad-R1.git \
    "${WORKSPACE_ROOT}/third_party/Vad-R1"
fi

git -C "${WORKSPACE_ROOT}/StreamVAD" remote rename origin upstream 2>/dev/null || true

git -C "${WORKSPACE_ROOT}/StreamVAD" rev-parse HEAD \
  > "${WORKSPACE_ROOT}/reproducibility/streammind_commit.txt"

git -C "${WORKSPACE_ROOT}/third_party/Vad-R1" rev-parse HEAD \
  > "${WORKSPACE_ROOT}/reproducibility/vadr1_commit.txt"
```

Improve the script as needed:

* Make repeated execution idempotent.
* Detect an existing `upstream` remote.
* Do not overwrite unrelated repositories.
* Print the resolved commit hashes.
* Fail clearly when Git or network access is unavailable.
* Never delete existing directories automatically.

Do not execute this script unless cloning is necessary and network access is allowed in the current Codex environment.

If both repositories already exist, inspect them in place.

---

# 4. Mandatory Upstream Code Audit

Before changing implementation files, inspect the relevant upstream code in batches.

At minimum, inspect:

```text
StreamVAD/README.md
StreamVAD/requirements.txt
StreamVAD/pyproject.toml
StreamVAD/scripts/custom/finetune_stage1.sh
StreamVAD/scripts/custom/finetune_stage2.sh
StreamVAD/scripts/custom/eval/evaluate.sh
StreamVAD/data/datasets.py
StreamVAD/data/soccer_data.py
StreamVAD/streammind/train_flash_attn_score.py
StreamVAD/streammind/train_new_stream.py
StreamVAD/streammind/streammind_trainer_score.py
StreamVAD/streammind/model/multimodal_projector/builder.py
StreamVAD/streammind/model/**/*.py
third_party/Vad-R1/README.md
third_party/Vad-R1/setup.sh
third_party/Vad-R1/src/r1-v/src/open_r1/sft_video.py
third_party/Vad-R1/src/scripts/run_sft_video.sh
third_party/Vad-R1/inference/*
third_party/Vad-R1/evaluation/*
```

Search both repositories for:

```text
32000
32001
silence
response
caption_target
eos_target
soccer_dataset_train_cls
soccer_dataset_train_llm
cls_net
prompt_time_input_ids
prompt_time_lable
data/MatchTime
/mnt/
/home/
pdb.set_trace
torch.save
Vad-Reasoning-SFT
SYSTEM_PROMPT
USER_PROMPT
```

Create:

```text
StreamVAD/docs/UPSTREAM_CODE_AUDIT.md
```

The audit must document:

1. Exact StreamMind and Vad-R1 commit hashes.
2. Actual Stage 1 entry point.
3. Actual Stage 2 entry point.
4. Actual model class used by the training scripts.
5. Exact parameters trained in original Stage 1.
6. Exact parameters trained in original Stage 2.
7. How the original gate labels are generated.
8. How prompt text is inserted into the original gate.
9. Every relevant hard-coded token ID.
10. Every relevant hard-coded path.
11. Every active debugger call.
12. Every unconditional debug `torch.save`.
13. Current class-weight implementation.
14. Existing batching assumptions.
15. Whether the original implementation assumes batch size 1.
16. Existing legacy datasets and collators.
17. Vad-R1 SFT prompt structure.
18. Vad-R1 record fields.
19. Environment incompatibilities between both repositories.
20. Repository, model, and dataset licensing uncertainties.

Do not assume that source-code header comments alone establish the licensing status of the entire Vad-R1 repository or dataset. Record uncertainty explicitly.

---

# 5. Implementation Strategy

Do not rewrite StreamMind wholesale.

Add an isolated StreamVAD path while preserving legacy behavior where practical.

Use a feature flag or explicit task configuration such as:

```text
task_type = legacy_streammind
task_type = stream_vad_stage1
task_type = stream_vad_stage2
task_type = stream_vad_inference
```

Prefer new files with small integration changes.

Recommended modules:

```text
StreamVAD/streammind/stream_vad/
├── __init__.py
├── schemas.py
├── prompts.py
├── cognition_gate.py
├── cognition_output.py
├── state.py
├── streaming_engine.py
├── interruption.py
└── metrics.py

StreamVAD/data/stream_vad/
├── __init__.py
├── schemas.py
├── path_resolver.py
├── metadata.py
├── vadr1_reader.py
├── compact_p2c.py
├── stage1_builder.py
├── stage2_builder.py
├── semantic_change.py
├── augmentations.py
├── dataset.py
└── collator.py
```

Adjust paths after inspecting actual package conventions.

---

# 6. Configuration System

Create:

```text
StreamVAD/configs/stream_vad/base.yaml
StreamVAD/configs/stream_vad/stage1.yaml
StreamVAD/configs/stream_vad/stage2.yaml
StreamVAD/configs/stream_vad/inference.yaml
StreamVAD/configs/stream_vad/evaluation.yaml
StreamVAD/configs/prompts/gate_prompt.txt
StreamVAD/configs/prompts/cognition_prompt.txt
StreamVAD/.env.example
```

Do not hard-code server paths.

Use environment variables:

```bash
STREAMVAD_ROOT
VADR1_REPO_ROOT
DATA_ROOT
VADR1_DATA_ROOT
MODEL_ROOT
OUTPUT_ROOT
HF_HOME
HF_TOKEN
MODEL_NAME_OR_PATH
VISION_TOWER
STAGE1_CHECKPOINT
STAGE2_CHECKPOINT
```

Every server script must validate required variables with syntax equivalent to:

```bash
: "${DATA_ROOT:?DATA_ROOT is required}"
```

Configuration must include:

```yaml
seed: 42

task:
  name: stream_vad

prompts:
  gate_prompt_path: configs/prompts/gate_prompt.txt
  cognition_prompt_path: configs/prompts/cognition_prompt.txt

gate:
  labels:
    hold: 0
    trigger: 1
  architecture: mistral4
  class_weights: null
  conditioning: visual_state_only
  trigger_threshold: 0.5
  cooldown_seconds: 1.0
  minimum_context_seconds: 2.0
  state_stale_after_seconds: 5.0

cognition:
  decisions:
    normal: 0
    abnormal: 1
  output_format: compact_p2c_xml

data:
  annotation_unit: auto
  fps_fallback: null
  fail_on_missing_video: true
  unresolved_policy: exclude

local:
  allow_heavy_execution: false
```

Support these optional gate-conditioning modes:

```text
visual_state_only
visual_plus_previous_cognition
```

Implement `visual_state_only` first.

Design the interface for `visual_plus_previous_cognition`, but do not make it mandatory for the initial smoke-test path.

---

# 7. Server Script Layout

Create, but do not run locally:

```text
scripts/server/00_check_server.sh
scripts/server/01_create_streamvad_environment.sh
scripts/server/02_create_vadr1_teacher_environment.sh
scripts/server/03_download_vadr1_sft.sh
scripts/server/04_audit_vadr1_sft.sh
scripts/server/05_build_compact_p2c.sh
scripts/server/06_build_stage1_dataset.sh
scripts/server/07_generate_prefix_teacher_labels.sh
scripts/server/08_build_stage2_dataset.sh
scripts/server/09_smoke_test_stage1.sh
scripts/server/10_train_stage1.sh
scripts/server/11_smoke_test_stage2.sh
scripts/server/12_train_stage2.sh
scripts/server/13_run_streaming_inference.sh
scripts/server/14_evaluate_streamvad.sh
```

Each script must:

* Use `#!/usr/bin/env bash`.
* Use `set -euo pipefail`.
* Validate environment variables.
* Print resolved paths.
* Support `--help` where appropriate.
* Avoid unquoted variable expansions.
* Avoid hard-coded CUDA device IDs.
* Avoid automatically launching full training from setup scripts.
* Save executed configuration and command lines to the output directory.
* Record Git commit hashes.
* Fail clearly on missing dependencies.
* Never modify raw data.

---

# 8. Server Environment Generation

## 8.1 StreamVAD Environment

Generate a server environment script based on the actual upstream requirements.

The official StreamMind baseline currently expects a GPU-oriented environment. Do not install it locally.

The script should broadly perform:

```bash
conda create -n streamvad python=3.10 -y
conda activate streamvad

cd "${STREAMVAD_ROOT}"
pip install -r requirements.txt
pip install flash-attn==2.5.8 --no-build-isolation
pip install pytest pydantic pyyaml jsonlines huggingface_hub \
  sentence-transformers scikit-learn
```

Do not blindly assume this set is sufficient.

After auditing imports:

* Add only required dependencies.
* Avoid installing Vad-R1’s custom transformers into this environment.
* Record `pip freeze`.
* Record PyTorch and CUDA versions.
* Validate FlashAttention compatibility.
* Provide a `--skip-flash-attn` option for smoke testing when possible.

## 8.2 Vad-R1 Teacher Environment

Generate a separate optional teacher environment script.

It should follow the actual Vad-R1 repository instructions, including its Python version, setup script, Qwen video utility, and provided transformers package.

Do not merge this environment with the StreamVAD environment.

The teacher environment is only required for:

* Prefix-specific compact P2C generation.
* Semantic event-signature generation.
* Semantic `hold/trigger` pseudo-label construction.

The core StreamVAD repository must still support rule-based and mock-data tests without the teacher environment.

---

# 9. Vad-R1 SFT Download Script

Create `scripts/server/03_download_vadr1_sft.sh`.

It must download only SFT metadata and SFT videos, not the entire dataset repository.

Use commands equivalent to:

```bash
hf download wbfwonderful/Vad-R1 \
  Vad-Reasoning-SFT-train.jsonl \
  Vad-Reasoning-SFT-test.jsonl \
  --repo-type dataset \
  --local-dir "${VADR1_DATA_ROOT}"

hf download wbfwonderful/Vad-R1 \
  --repo-type dataset \
  --include "Vad-Reasoning-SFT/**" \
  --local-dir "${VADR1_DATA_ROOT}"
```

Requirements:

* Check free disk space before downloading.
* Print a warning that the whole repository is much larger and must not be downloaded unintentionally.
* Support `HF_HOME`.
* Support `HF_TOKEN` without printing it.
* Support `--metadata-only`.
* Support `--dry-run`.
* Confirm expected files after completion.
* Generate a download manifest.
* Do not run automatically during environment setup.

---

# 10. Raw Dataset Audit

Create:

```text
StreamVAD/tools/audit_vadr1_sft.py
```

Use typed configuration and structured logging.

## 10.1 Required Fields

Validate:

```text
source
video
anomaly_type
path
total_frames
think
answer
```

Abnormal records may additionally contain:

```text
start
end
```

Do not assume that `start/end` are frames until validated.

## 10.2 Annotation Unit Detection

Support:

```text
auto
frame
second
normalized
```

In `auto` mode, inspect:

* Value ranges.
* Integer versus floating-point values.
* Relation to `total_frames`.
* Relation to decoded duration.
* Relation to FPS.
* Samples across multiple sources.

Do not silently choose a unit when evidence is inconsistent.

Produce:

```text
annotation_unit_status
annotation_unit_inferred
annotation_unit_confidence
annotation_unit_evidence
```

Require an explicit override when inference is ambiguous.

## 10.3 Path Resolution

Never edit the original JSONL.

Create a resolver that:

1. Accepts an explicit local SFT root.
2. Attempts to preserve the path suffix following `Vad-Reasoning-SFT/`.
3. Checks exact relative paths.
4. Uses a generated file manifest.
5. Allows basename matching only when the basename is unique.
6. Logs every fallback.
7. Rejects ambiguous matches.

Do not silently bind a record to the first file with the same basename.

## 10.4 Video Metadata

Use a pluggable metadata provider.

Server implementation may use:

* `ffprobe`; or
* Decord; or
* PyAV.

Local unit tests must inject mock metadata without opening real videos.

Collect:

```text
actual_frame_count
fps
duration_seconds
width
height
codec
decode_status
```

Compare `actual_frame_count` with `total_frames`.

## 10.5 Identity and Leakage

Create stable IDs:

```text
{official_split}:{source}:{video}
```

Also create a content identity using, where feasible:

* Relative canonical path.
* File size.
* Partial or complete hash.
* Normalized video stem.

Check:

* Duplicate records.
* Duplicate files.
* Same source video across train and test.
* Near-identical path aliases.
* Invalid intervals.
* Missing files.
* Unreadable files.
* Empty reasoning fields.

## 10.6 Audit Outputs

Generate:

```text
data/reports/vadr1_sft_audit_summary.json
data/reports/vadr1_sft_statistics.md
data/reports/vadr1_sft_errors.jsonl
data/reports/vadr1_sft_manifest.jsonl
data/reports/vadr1_sft_unit_analysis.json
```

Do not continue automatically when:

* Annotation units are unresolved.
* More than a configurable fraction of videos is missing.
* Split leakage is detected.
* Invalid intervals exceed the configured tolerance.

---

# 11. Typed Data Schemas

Use Pydantic or equivalent explicit validation.

Create schemas for:

```text
RawVadR1Record
ResolvedVadR1Record
CompactP2C
Stage1Example
EventSignature
GateExample
GateSequence
CognitionState
StreamingStepOutput
```

## 11.1 Compact P2C

```json
{
  "scene_prior": "string",
  "observation": "string",
  "deviation": "string",
  "decision": "normal"
}
```

Validation rules:

* All text fields must be non-empty.
* `decision` must be `normal` or `abnormal`.
* No unknown top-level fields unless explicitly allowed.
* Preserve provenance separately.
* Do not infer `abnormal` only because `start/end` exist without checking the official label.

## 11.2 Gate Labels

Use internal class IDs only:

```python
GATE_HOLD_ID = 0
GATE_TRIGGER_ID = 1
```

These IDs are not tokenizer IDs.

---

# 12. Compact P2C Construction

Create:

```text
StreamVAD/tools/build_compact_p2c.py
StreamVAD/data/stream_vad/compact_p2c.py
```

Support:

```text
--mode rule
--mode teacher
```

Implement `rule` mode locally.

Prepare, but do not locally run, `teacher` mode.

## 12.1 Rule Parsing

Parse Vad-R1’s structured `think` and `answer` fields.

For abnormal videos:

* Scene-description content becomes `scene_prior`.
* Abnormal-event description becomes `observation`.
* Abnormal-event recognition becomes `deviation`.
* Decision becomes `abnormal`.

For normal videos:

* Scene and expected behavior become `scene_prior`.
* Current normal activity becomes `observation`.
* Explanation of consistency with expected behavior becomes `deviation`.
* Decision becomes `normal`.

Exclude from the required target:

* Potential harm.
* Social norms.
* Moral judgment.
* Generic safety commentary.
* Repetitive final-answer wording.

Do not delete original text from provenance.

## 12.2 Parsing Failures

Every record must receive one status:

```text
parsed
partially_parsed
unresolved
invalid
```

Do not generate plausible-looking empty defaults.

Write unresolved records to:

```text
data/reports/compact_p2c_unresolved.jsonl
```

Default training policy:

```text
exclude unresolved
```

Record:

```text
source_record_id
source_text_hash
parser_version
parse_status
missing_fields
warnings
```

## 12.3 Teacher Mode

Teacher mode should accept a server-side inference command or model adapter.

It must:

* Generate schema-constrained compact P2C.
* Retry only on format failure, with a bounded retry count.
* Validate output.
* Cache results.
* Record model ID, checkpoint, prompt hash, decoding settings, and timestamp.
* Never overwrite rule-parsed outputs without preserving both versions.

---

# 13. Stage 1 Dataset Design

Stage 1 teaches the StreamMind perception representation and cognition model how to encode and express anomaly-relevant reasoning.

It must not create future-leaking online targets.

Implement two explicit modes.

## 13.1 Mode A: `full_annotation_alignment`

This is the first runnable baseline.

For an abnormal video:

* Resolve the annotated abnormal interval.
* Include configurable pre-event context.
* End the visual input at the annotated event end.
* Use compact P2C derived from the full annotated event.
* Do not claim that this sample trains earliest anomaly onset detection.
* Mark it as full-event alignment.

For a normal video:

* Use the full video or a uniformly sampled representation of the full video.
* Use the compact normal P2C.
* Mark it as full-video alignment.

This mode is allowed to use full-event reasoning because its visual input contains the full event covered by that reasoning.

## 13.2 Mode B: `causal_prefix_teacher`

This is the intended online-training mode.

Construct prefixes ending at selected causal time points such as:

```text
pre-event
candidate onset
early event
mid event
event offset
post-event
```

For every prefix:

* The visual input ends at `observation_end_frame`.
* The target must describe only evidence visible at or before that frame.
* Do not reuse the full-video Vad-R1 reasoning as the target unless a validator confirms every statement is already observable.
* Generate prefix-specific compact P2C with the teacher environment.
* Store the teacher prompt, model, and provenance.
* Reject outputs containing future timestamps or future-only events.

## 13.3 No-Future-Leakage Contract

Every Stage 1 record must include:

```json
{
  "observation_start_frame": 0,
  "observation_end_frame": 100,
  "target_evidence_end_frame": 100,
  "uses_future_frames": false,
  "construction_mode": "causal_prefix_teacher"
}
```

Require:

```text
target_evidence_end_frame <= observation_end_frame
```

Add tests for this invariant.

## 13.4 Split Policy

Preserve the official Vad-R1 test split.

Create validation data only from the official train split.

Group by source-video identity.

Use seed 42.

Stratify where feasible by:

```text
source
normal/abnormal
anomaly_type
```

Never put prefixes or augmented variants of the same source video into different splits.

## 13.5 Stage 1 Output

Generate:

```text
data/processed/stage1/train.jsonl
data/processed/stage1/val.jsonl
data/processed/stage1/test.jsonl
data/reports/stage1_statistics.md
```

Example:

```json
{
  "id": "train:UCF:example:full_event",
  "video_path": "/resolved/path/video.mp4",
  "source_video_id": "UCF:example",
  "observation_start_frame": 1200,
  "observation_end_frame": 2400,
  "fps": 30.0,
  "prompt_id": "generic_cognition_v1",
  "target": {
    "scene_prior": "Pedestrians normally remain on the sidewalk...",
    "observation": "A pedestrian enters the roadway...",
    "deviation": "The action conflicts with the expected traffic pattern...",
    "decision": "abnormal"
  },
  "construction_mode": "full_annotation_alignment",
  "uses_future_frames": false,
  "provenance": {
    "original_record_id": "train:UCF:example",
    "annotation_source": "Vad-Reasoning-SFT",
    "compact_parser_version": "v1"
  }
}
```

---

# 14. Stage 2 Cognition-Gate Dataset

Stage 2 trains whether cognition must be updated.

Do not use only abnormal boundaries as the final gate dataset.

Implement three label modes.

## 14.1 `boundary_baseline`

Purpose:

* Code validation.
* Initial baseline.
* Smoke testing.

Label:

* Near anomaly onset: `trigger`.
* Near anomaly offset: `trigger`.
* Stable periods: `hold`.

Clearly document:

> This mode can leak anomaly-boundary information into the gate and may make the gate behave like a VAD classifier. It is a baseline, not the final intended supervision.

## 14.2 `semantic_teacher`

This is the preferred mode.

Divide a stream into ordered windows or causal prefixes.

For each cognition point, generate an `EventSignature`:

```json
{
  "scene": "road intersection",
  "entities": ["pedestrian", "vehicles"],
  "actions": ["pedestrian waiting"],
  "relations": ["pedestrian beside roadway"],
  "phase": "ongoing",
  "decision": "normal",
  "summary": "A pedestrian waits beside the crossing."
}
```

Compare the current signature with the previously accepted cognition state.

Label `trigger` when one or more meaningful changes occur:

* New event identity.
* New relevant subject.
* New relevant action.
* Changed subject-object relation.
* Event phase begins.
* Event phase ends.
* Existing explanation becomes invalid.
* Decision changes.
* An event is replaced by a different event.
* Previously insufficient evidence becomes sufficient.

Label `hold` when:

* The same event continues.
* Only visual appearance changes.
* The same action is repeated.
* A short occlusion occurs.
* A short irrelevant segment occurs.
* The same event resumes after a short interruption.
* Semantic differences are below the configured threshold.

Do not trigger solely because the teacher paraphrases the same event differently.

Use structured fields and embeddings, not raw-string equality.

## 14.3 `hybrid`

Combine:

* High-confidence official anomaly boundaries.
* Semantic-teacher event changes.
* Normal-event changes.
* Persistence filtering.
* Manual or heuristic interruption constraints.

Every label must include provenance and confidence.

## 14.4 Normal Videos Must Contain Triggers

A gate dataset in which all normal windows are `hold` is invalid.

It would allow the gate to learn:

```text
trigger approximately means abnormal
```

Normal videos must include meaningful normal-event transitions, such as:

* A person enters.
* A person begins walking.
* A vehicle arrives.
* A routine action begins.
* A routine action ends.
* A scene changes while remaining normal.

Use semantic-teacher labels for final data.

A visual-change proxy may be implemented as an auxiliary baseline, but it must not be presented as ground-truth cognition change.

## 14.5 Temporal Persistence

Prevent one-frame noise from becoming a trigger.

Configuration must support:

```yaml
gate_data:
  stride_seconds: 0.5
  trigger_tolerance_seconds: 1.0
  minimum_change_persistence_windows: 2
  minimum_trigger_separation_seconds: 1.5
  interruption_max_seconds: 2.0
```

## 14.6 Stage 2 Records

Use ordered sequences, not only shuffled independent frames.

Example:

```json
{
  "sequence_id": "train:UCF:example:sequence_000",
  "source_video_id": "UCF:example",
  "prompt_id": "generic_gate_v1",
  "steps": [
    {
      "frame_start": 0,
      "frame_end": 30,
      "label": "trigger",
      "label_id": 1,
      "label_source": "bootstrap",
      "confidence": 1.0
    },
    {
      "frame_start": 30,
      "frame_end": 45,
      "label": "hold",
      "label_id": 0,
      "label_source": "semantic_teacher",
      "confidence": 0.95
    }
  ]
}
```

Audit-only fields such as the true anomaly label must never be passed as model input.

---

# 15. Interruption and Resume Construction

The system must support:

```text
event A
-> brief interruption
-> event A continues
```

The expected gate sequence is generally:

```text
trigger
hold
hold
```

not:

```text
trigger
trigger
trigger
```

Implement train-only augmentations:

```text
frame repetition
frame dropping
short black-screen insertion
partial occlusion
short normal-video insertion
short unrelated-frame insertion
temporary low-quality frames
```

Each augmentation record must contain:

```text
augmentation_type
interruption_start
interruption_end
interruption_duration
source_of_inserted_content
preserve_event_identity
expected_resume_behavior
```

## 15.1 Positive Resume Example

```text
event A starts
event A continues
brief interruption
event A reappears
```

Labels:

```text
trigger
hold
hold
hold
```

## 15.2 Negative Resume Example

```text
event A starts
brief interruption
different event B appears
```

Labels:

```text
trigger
hold
trigger
```

## 15.3 Scene Reset

A true camera cut or long outage may invalidate the existing state.

Implement internal state validity:

```text
valid
stale
reset_required
```

Do not add `unknown` as a required cognition decision in the first version.

When the interruption exceeds the configured timeout:

* Mark the previous cognition state stale.
* Do not silently preserve it indefinitely.
* Force cognition when usable visual evidence returns.
* Record the stale interval in inference output.

Do not count artificially augmented videos as unmodified validation or test examples.

---

# 16. Cognition Gate Refactor

The current StreamMind gate implementation contains model, prompt assembly, token embeddings, class labels, and legacy timing logic in a tightly coupled path.

Create a cleaner module:

```text
StreamVAD/streammind/stream_vad/cognition_gate.py
```

## 16.1 Required Interface

Implement a typed output:

```python
@dataclass
class GateOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.LongTensor
```

Implement a typed batch:

```python
@dataclass
class GateBatch:
    visual_states: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.LongTensor | None
    prompt_input_ids: torch.LongTensor | None = None
    previous_cognition_embedding: torch.Tensor | None = None
```

Expected logits:

```text
[batch, time, 2]
```

or a clearly documented equivalent.

## 16.2 Architectures

Support:

```text
mistral4
mlp
linear
```

`mistral4` should reproduce the original high-level idea:

* A shallow 4-layer Mistral-style gate.
* Output dimension 2.
* Initialized using the appropriate upstream mechanism where possible.

`linear` and `mlp` are required for lightweight tests and ablations.

## 16.3 Remove Coupling

For the StreamVAD path:

* Do not use `32000` or `32001` as gate labels.
* Do not search for hard-coded tokenizer IDs to find insertion positions.
* Do not encode gate labels through the main tokenizer.
* Do not append class IDs to a general-language vocabulary.
* Do not hard-code class weights.
* Do not hard-code the system prompt inside `builder.py`.
* Do not assume a single example per batch.
* Do not use tensor value comparison as an attention mask for floating-point embeddings.
* Do not leave active `pdb.set_trace`.
* Do not unconditionally save debug tensors to `/home/...`.

Preserve the original code behind an explicitly named legacy path when practical.

## 16.4 Class Weights

Support:

```yaml
gate:
  class_weights: null
```

and optional explicit values:

```yaml
gate:
  class_weights: [1.0, 2.0]
```

Also support computing weights from the training manifest.

Record the selected weights.

## 16.5 Trainable Parameters

For Stage 2, freeze all parameters except the new gate by default.

Require a programmatic assertion:

```python
unexpected_trainable = [
    name for name, parameter in model.named_parameters()
    if parameter.requires_grad and not name.startswith(ALLOWED_GATE_PREFIX)
]
assert not unexpected_trainable
```

Do not rely only on substring matching without reporting the resulting parameter list.

Write the trainable parameter names and counts to a file before training.

---

# 17. Stage 1 Training Integration

Create explicit arguments rather than overloading SoccerNet flags.

Example:

```python
stream_vad_dataset: bool = False
stream_vad_stage: str | None = None
stream_vad_manifest: str | None = None
stream_vad_config: str | None = None
```

Avoid naming the new path:

```text
soccer_dataset_train_llm
soccer_dataset_train_cls
```

Legacy flags may remain for compatibility.

## 17.1 Default Stage 1 Policy

Preserve the original StreamMind Stage 1 trainability policy as a named baseline after auditing it.

Add configurable alternatives:

```text
streammind_full
projector_only
projector_plus_lora
```

Do not silently change which parameters train.

Before training, print and save:

```text
trainable parameter names
trainable parameter count
frozen parameter count
model checkpoint
vision checkpoint
dataset manifest
configuration hash
```

---

# 18. Stage 2 Training Integration

Stage 2 must initialize from the Stage 1 checkpoint.

It must:

* Freeze the foundation model.
* Freeze the vision tower.
* Freeze the streaming perception module.
* Freeze the language model.
* Train only the cognition gate unless explicitly configured otherwise.
* Use ordered gate sequences.
* Compute loss only at valid gate positions.
* Respect padding masks.
* Support class imbalance.
* Save gate-only checkpoints and a complete inference bundle.
* Verify that non-gate gradients are absent.

Create:

```text
StreamVAD/scripts/stream_vad/train_stage1.sh
StreamVAD/scripts/stream_vad/train_stage2.sh
```

The scripts under `scripts/server/` may call these repository scripts.

---

# 19. Streaming Inference State Machine

Create:

```text
StreamVAD/streammind/stream_vad/state.py
StreamVAD/streammind/stream_vad/streaming_engine.py
StreamVAD/streammind/inference_stream_vad.py
```

## 19.1 State

Use a structure similar to:

```python
@dataclass
class CognitionState:
    initialized: bool = False
    validity: str = "valid"
    last_decision: str | None = None
    last_abnormal_probability: float | None = None
    last_scene_prior: str | None = None
    last_observation: str | None = None
    last_deviation: str | None = None
    last_summary: str | None = None
    last_trigger_frame: int | None = None
    last_visible_frame: int | None = None
    active_event_id: str | None = None
```

## 19.2 Bootstrap

The first decision cannot depend on waiting indefinitely for a trained trigger.

After `minimum_context_seconds`:

* Force one cognition call.
* Mark its gate source as `bootstrap`.
* Initialize the cognition state.
* Exclude bootstrap from ordinary false-trigger metrics or report it separately.

## 19.3 Per-Step Logic

Conceptually:

```python
update_streaming_visual_state()

if cognition_state_is_stale:
    if sufficient_visual_evidence_has_returned:
        force_trigger("stale_state_refresh")
elif gate_predicts_trigger:
    invoke_cognition()
else:
    preserve_previous_cognition_state()
```

## 19.4 Hold Behavior

On `hold`:

* Do not invoke the LLM.
* Preserve the last decision.
* Preserve the last abnormal probability.
* Preserve the active event identity.
* Update visibility and freshness metadata.
* Do not clear state because of a short occlusion.

## 19.5 Trigger Behavior

On `trigger`:

* Invoke cognition using evidence only up to the current time.
* Validate the four-field output.
* Obtain a continuous abnormal score from the decision-token logits or a clearly documented calibrated method.
* Update the state.
* Create or update an event ID.
* Record why the trigger occurred when available.

## 19.6 Trigger Stabilization

Support:

```text
threshold
hysteresis
cooldown
minimum persistence
maximum hold duration
```

Do not hide these behaviors in ad hoc inference code.

## 19.7 Inference Output

Write JSONL:

```json
{
  "video_id": "example",
  "frame_index": 1820,
  "timestamp_seconds": 60.67,
  "gate_prediction": "hold",
  "trigger_probability": 0.08,
  "trigger_source": "model",
  "llm_invoked": false,
  "maintained_decision": "abnormal",
  "maintained_abnormal_probability": 0.91,
  "state_validity": "valid",
  "active_event_id": "event_001"
}
```

For a trigger:

```json
{
  "video_id": "example",
  "frame_index": 1860,
  "gate_prediction": "trigger",
  "trigger_probability": 0.87,
  "trigger_source": "model",
  "llm_invoked": true,
  "cognition": {
    "scene_prior": "...",
    "observation": "...",
    "deviation": "...",
    "decision": "normal"
  },
  "abnormal_probability": 0.12,
  "active_event_id": null
}
```

---

# 20. Frame-Level Scores

Do not use trigger probability as anomaly probability.

They represent different questions:

```text
trigger probability:
Should cognition be updated?

abnormal probability:
Does the cognition model judge the current state abnormal?
```

Between cognition calls, define the frame-level anomaly score as the most recently valid cognition score:

```python
frame_anomaly_score[t] = last_valid_abnormal_probability
```

During a stale-state interval:

* Preserve the score only with a `stale=true` flag; or
* Mask the interval according to evaluation configuration.

Do not silently treat stale scores as fresh predictions.

---

# 21. Evaluation

Create:

```text
StreamVAD/evaluation/evaluate_stream_vad.py
StreamVAD/evaluation/gate_metrics.py
StreamVAD/evaluation/cognition_metrics.py
StreamVAD/evaluation/resume_metrics.py
```

## 21.1 Gate Metrics

Report:

```text
trigger precision
trigger recall
trigger F1
trigger average precision
average trigger delay
false triggers per minute
missed semantic changes
duplicate trigger rate
trigger rate per minute
LLM calls per minute
LLM-call reduction relative to every-window cognition
```

Evaluate against the selected gate-label source and state which source was used.

## 21.2 Cognition Metrics

Report:

```text
normal/abnormal accuracy
balanced accuracy
macro F1
XML/schema validity rate
field completeness
reasoning-decision consistency
event-description consistency
```

## 21.3 End-to-End VAD Metrics

Report:

```text
frame-level ROC-AUC
frame-level PR-AUC
event-level precision
event-level recall
event-level F1
anomaly onset delay
anomaly offset delay
false alarm duration
```

Use maintained cognition scores, not gate scores.

## 21.4 Interruption and Resume Metrics

Report:

```text
same-event resume consistency
state retention through brief occlusion
unnecessary re-trigger rate after same-event resume
correct trigger rate when event B replaces event A
state-staleness handling accuracy
```

## 21.5 Efficiency

Report:

```text
streaming FPS
gate latency
cognition latency
LLM invocation count
LLM invocation reduction
peak GPU memory
average GPU memory
```

## 21.6 Baselines and Ablations

Prepare evaluation configurations for:

```text
Every-window cognition
Original StreamMind-style endpoint gate
Boundary-baseline hold/trigger gate
Semantic-teacher hold/trigger gate
Linear gate
MLP gate
Mistral4 gate
No interruption augmentation
With interruption augmentation
Visual-state-only conditioning
Visual-plus-previous-cognition conditioning
```

Do not fabricate results.

---

# 22. Local Test Fixtures

Create:

```text
StreamVAD/tests/fixtures/mock_vadr1_train.jsonl
StreamVAD/tests/fixtures/mock_vadr1_test.jsonl
StreamVAD/tests/fixtures/mock_manifest.jsonl
StreamVAD/tests/fixtures/mock_compact_p2c.jsonl
```

Do not require real video files for most local tests.

Use an injected metadata provider:

```python
class FakeVideoMetadataProvider:
    ...
```

For tensor-level gate tests:

* Use small random tensors.
* Use the `linear` or `mlp` gate.
* Skip tests cleanly when PyTorch is unavailable.
* Do not import FlashAttention during schema-only tests.

---

# 23. Required Tests

Create at least:

```text
tests/test_vadr1_schema.py
tests/test_annotation_unit_detection.py
tests/test_path_resolution.py
tests/test_split_leakage.py
tests/test_compact_p2c_parser.py
tests/test_stage1_no_future_leakage.py
tests/test_gate_label_ids.py
tests/test_gate_normal_triggers.py
tests/test_gate_sequence_order.py
tests/test_interruption_resume.py
tests/test_event_replacement_trigger.py
tests/test_state_staleness.py
tests/test_gate_output_shape.py
tests/test_trainable_parameters.py
tests/test_prompt_loading.py
tests/test_streaming_state_machine.py
tests/test_legacy_imports.py
tests/test_server_scripts_static.py
```

## 23.1 Critical Assertions

Tests must verify:

1. Gate labels are only `hold/trigger`.
2. Gate IDs are exactly `0/1`.
3. Gate IDs are independent of tokenizer vocabulary.
4. Stage 1 causal targets do not use future evidence.
5. Source videos do not leak across splits.
6. Normal videos may contain valid triggers.
7. Same-event continuation is `hold`.
8. Same-event resume after a short interruption is `hold`.
9. Different-event replacement is `trigger`.
10. Long interruption marks state stale.
11. Stage 2 exposes only permitted trainable parameters.
12. Raw JSONL files are never rewritten.
13. Missing data produces an explicit error or report.
14. Legacy StreamMind imports remain usable.
15. Server scripts do not contain developer-specific absolute paths.
16. Local tests do not initiate network downloads.

---

# 24. Local Validation Commands

Run only locally feasible commands, such as:

```bash
python -m compileall StreamVAD/data/stream_vad
python -m compileall StreamVAD/streammind/stream_vad
pytest -q
```

Also run when available:

```bash
ruff check .
shellcheck scripts/server/*.sh
```

Do not install large dependencies merely to make optional checks run.

When a check is blocked by the local environment:

* Record the exact missing dependency.
* Mark the check as server-only.
* Do not falsely report success.

---

# 25. Server Smoke Tests

Generate scripts but do not run them locally.

## 25.1 Stage 1 Smoke Test

Use:

* 2 normal samples.
* 2 abnormal samples.
* One forward pass.
* One backward pass.
* No checkpoint download inside the training process.
* Explicit user-provided checkpoint paths.
* Output trainable parameter report.
* Output loss and tensor shapes.

## 25.2 Stage 2 Smoke Test

Use:

* At least one `hold`.
* At least one `trigger`.
* A short ordered sequence.
* One forward pass.
* One backward pass.
* Gradient assertion for non-gate parameters.
* Gate logits and loss validation.

## 25.3 Streaming Smoke Test

Use a short video and verify:

```text
bootstrap cognition
hold
brief interruption
hold
same event resumes
hold
event changes
trigger
```

Save the complete inference log.

---

# 26. Documentation

Create:

```text
StreamVAD/README_STREAMVAD.md
StreamVAD/docs/UPSTREAM_CODE_AUDIT.md
StreamVAD/docs/IMPLEMENTATION_PLAN.md
StreamVAD/docs/ARCHITECTURE.md
StreamVAD/docs/DATASET_DESIGN.md
StreamVAD/docs/GATE_LABELING.md
StreamVAD/docs/TRAINING.md
StreamVAD/docs/SERVER_RUNBOOK.md
StreamVAD/docs/EVALUATION.md
StreamVAD/docs/KNOWN_LIMITATIONS.md
StreamVAD/docs/LICENSE_AUDIT.md
```

## 26.1 Server Runbook

Give the exact manual order:

```text
00_check_server.sh
01_create_streamvad_environment.sh
02_create_vadr1_teacher_environment.sh
03_download_vadr1_sft.sh
04_audit_vadr1_sft.sh
05_build_compact_p2c.sh
06_build_stage1_dataset.sh
07_generate_prefix_teacher_labels.sh
08_build_stage2_dataset.sh
09_smoke_test_stage1.sh
10_train_stage1.sh
11_smoke_test_stage2.sh
12_train_stage2.sh
13_run_streaming_inference.sh
14_evaluate_streamvad.sh
```

Explain which steps are optional for the baseline and which are necessary for semantic-teacher supervision.

## 26.2 Known Limitations

Explicitly document:

1. Vad-R1 reasoning is primarily full-video reasoning, not naturally causal prefix reasoning.
2. Full-video reasoning must not be attached to an early prefix without validation.
3. `boundary_baseline` may turn the gate into an anomaly-boundary detector.
4. Semantic teacher labels are pseudo-labels.
5. Normal-event trigger quality depends on teacher quality.
6. A brief interruption and a true event replacement may be difficult to distinguish.
7. State persistence may propagate a wrong cognition decision.
8. The initial implementation does not support arbitrary user queries.
9. Frame-level anomaly scores are held from the latest cognition call.
10. StreamMind, foundation-model, Vad-R1, and dataset licensing require separate review.
11. Local tests cannot validate CUDA kernels or full-model training.
12. Server hardware requirements are not known and must remain configurable.

---

# 27. Implementation Phases

Maintain `docs/IMPLEMENTATION_PLAN.md`.

Use phases:

## Phase 1: Audit and Scaffolding

* Inspect repositories.
* Record commits.
* Create package structure.
* Create configs and schemas.
* Create local fixtures.

## Phase 2: Dataset Tooling

* Raw reader.
* Path resolver.
* Metadata interface.
* Audit tool.
* Compact P2C parser.
* Stage 1 builder.
* Stage 2 builder.
* Semantic-label interfaces.
* Interruption augmentation.

## Phase 3: Model Integration

* Cognition gate module.
* Data collator.
* Training argument integration.
* Trainable-parameter policy.
* Legacy compatibility.

## Phase 4: Streaming Inference

* State machine.
* Bootstrap.
* Hold/trigger handling.
* Staleness.
* Cognition output validation.
* JSONL logs.

## Phase 5: Evaluation and Scripts

* Metrics.
* Server scripts.
* Runbook.
* Smoke-test scripts.

## Phase 6: Verification

* Compile.
* Unit tests.
* Static checks.
* Diff review.
* Documentation consistency review.

Do not end the task while plan items remain silently pending.

Every item must be marked:

```text
completed
blocked
cancelled
```

with a reason where relevant.

---

# 28. Acceptance Criteria

The local implementation is complete only when all of the following are true:

* Both upstream repositories are obtainable through generated scripts.
* Exact upstream commit hashes can be recorded.
* Vad-R1 raw records can be parsed without mutating source JSONL.
* Annotation-unit ambiguity is handled explicitly.
* Compact P2C parsing is typed and auditable.
* Stage 1 supports full-event baseline and causal-prefix teacher modes.
* Causal Stage 1 records enforce no-future-leakage invariants.
* Stage 2 supports boundary, semantic, and hybrid label modes.
* Normal-video triggers are supported.
* Interruption-resume examples are represented.
* Gate class IDs are independent from tokenizer IDs.
* The new gate does not rely on `32000/32001`.
* Stage 2 freezes all non-gate parameters by default.
* The inference state machine performs bootstrap, hold, trigger, resume, and stale-state handling.
* Frame anomaly scores come from cognition, not trigger probability.
* Local fixtures and tests exist.
* Local feasible tests have been run.
* Server scripts exist but have not been run locally.
* All absolute developer paths in the new StreamVAD path are eliminated.
* Existing upstream debug breakpoints do not affect the StreamVAD path.
* Documentation clearly distinguishes implemented features from server-only or future work.

---

# 29. Final Codex Response

At the end, provide:

1. What was implemented.
2. The final repository tree for new files.
3. Important upstream problems discovered.
4. Important design decisions.
5. Exact local checks run and results.
6. Checks skipped because they require the GPU server.
7. Exact first server command the user should run.
8. Any genuine blockers.
9. A short diff-risk review.

Do not paste the full contents of every generated file in the final response. Reference their paths.

Do not claim that real training, real dataset processing, or GPU smoke tests succeeded unless they were actually executed on the server.
