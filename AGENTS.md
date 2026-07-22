# AGENTS.md

## Project Purpose

This repository implements StreamVAD, a research prototype derived from StreamMind and supervised with Vad-R1 annotations.

The core principle is:

> Continuous perception does not require continuous cognition.

The streaming perception module maintains temporal visual state. A cognition gate predicts only whether new evidence requires the current interpretation to be updated:

* `hold`: keep the current cognition state.
* `trigger`: invoke the cognition model because new evidence may change the current interpretation.

The gate must not directly classify the video as normal or abnormal.

## Development Environment

The current machine is a local code-development environment.

Do not perform heavyweight operations locally:

* Do not download model checkpoints.
* Do not download the Vad-R1 video dataset.
* Do not install CUDA, FlashAttention, DeepSpeed, or large model environments.
* Do not run GPU training.
* Do not run full video preprocessing.
* Do not assume that an NVIDIA GPU, FFmpeg, Conda, or real datasets are available.
* Do not write machine-specific absolute paths.

Generate server scripts for all heavyweight operations instead.

Local verification may use:

* Static code inspection.
* `python -m compileall`.
* Lightweight unit tests.
* Mock JSONL records.
* Mock metadata providers.
* CPU-only tests that do not load foundation-model checkpoints.

## Engineering Rules

* Inspect relevant upstream files before editing.
* Preserve existing StreamMind behavior unless a new StreamVAD-specific path is enabled.
* Prefer adding isolated StreamVAD modules over rewriting large upstream files.
* Keep legacy StreamMind dataset and gate paths available where practical.
* Do not silently swallow malformed data.
* Do not use broad exception handlers as success-shaped fallbacks.
* Do not mutate raw Vad-R1 JSONL files.
* Do not infer annotation units without validating them.
* Do not use future video frames to construct an online training target.
* Keep tokenizer token IDs separate from gate class IDs.
* Use configuration values instead of hard-coded paths, token IDs, thresholds, or class weights.
* Do not use destructive Git commands.
* Do not revert unrelated user changes.
* Do not commit unless explicitly requested.

## Required Workflow

For this task:

1. Read `STREAMVAD_CODEX_TASK.md` completely.
2. Audit the two upstream repositories and record the exact commit hashes.
3. Create and maintain `docs/IMPLEMENTATION_PLAN.md`.
4. Continue from planning into implementation; do not stop after producing only a plan.
5. Implement the code, tests, configuration files, server scripts, and documentation.
6. Run all locally feasible checks.
7. Review the final diff for broken imports, hard-coded paths, data leakage, and unintended upstream regressions.
8. Report completed work, blocked server-only checks, and exact commands the user should run on the server.

## Definition of Done

The local task is complete only when:

* The complete StreamVAD code structure exists.
* Data builders support mock execution.
* The gate interface uses `hold/trigger`.
* Causal data-leakage checks exist.
* Local tests pass or each failure is explicitly explained.
* Server setup, download, preprocessing, training, inference, and evaluation scripts are present but have not been run locally.
* Documentation explains assumptions, limitations, execution order, and expected server outputs.
