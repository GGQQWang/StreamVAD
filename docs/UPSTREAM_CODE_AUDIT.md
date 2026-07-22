# Upstream Code Audit

Scope: static code audit only. No datasets, checkpoints, GPU setup, package installation, or training were run.

Audited repositories:

- `StreamMind/` cloned from `https://github.com/xinding-bot/StreamMind.git`
- `Vad-R1/` cloned from `https://github.com/wbfwonderful/Vad-R1.git`

## 1. StreamMind Training Entries

### Stage 1

- Entry script: `StreamMind/scripts/custom/finetune_stage1.sh`.
- Python entry: `StreamMind/streammind/train_flash_attn_score.py`, which imports and calls `streammind.train_new_stream.train(attn_implementation="flash_attention_2")`.
- Effective dataset flags in the script:
  - `--soccer_dataset True`
  - `--soccer_dataset_train_llm True`
- Important training parameters from `scripts/custom/finetune_stage1.sh`:
  - `WORLD_SIZE=1`, `NPROC_PER_NODE=1`
  - `GLOBAL_BATCH_SIZE=2`, `GRADIENT_ACCUMULATION_STEPS=2`, so `LOCAL_BATCH_SIZE=1`
  - `--model_name_or_path VideoLLaMA2-7B`
  - `--vision_tower clip-vit-large-patch14-336`
  - `--mm_projector_type mamba`
  - `--num_frames 32`
  - `--num_train_epochs 1`
  - `--learning_rate 2e-5`
  - `--bf16 True`, `--tf32 True`, `--fp16 False`
  - `--deepspeed scripts/zero2.json`

Confirmed behavior:

- `streammind.train_new_stream.train` constructs `LazySupervisedDataset` through `make_supervised_stream_data_module`, using `DataCollatorForstreamDataset`.
- With `soccer_dataset_train_llm=True`, `data.datasets.LazySupervisedDataset.__getitem__` calls `data.soccer_data.preprocess_llama_2_soccer`.
- `data.soccer_data.preprocess_llama_2_soccer` builds alternating `human: <video>\n` and `gpt: caption` turns, tokenizes with `tokenizer_MMODAL_token`, and masks the human/instruction side with `IGNORE_INDEX`.
- Stage 1 is therefore caption/response LM training over per-segment soccer captions, not Gate training.

### Stage 2

- Entry script: `StreamMind/scripts/custom/finetune_stage2.sh`.
- Python entry: same as Stage 1, `streammind/train_flash_attn_score.py` -> `streammind.train_new_stream.train`.
- Effective dataset flags in the script:
  - `--soccer_dataset True`
  - `--soccer_dataset_train_cls True`
- Important training parameters from `scripts/custom/finetune_stage2.sh`:
  - Same single-node/single-process setup and batch math as Stage 1.
  - `--learning_rate 2e-6`
  - Other core model/data arguments match Stage 1.

Confirmed behavior:

- `DataArguments` in `data/datasets.py` does not declare `soccer_dataset_train_cls`; that flag is declared under `ModelArguments` in `streammind/train_new_stream.py`.
- With `soccer_dataset_train_llm` absent/false, `data.datasets.LazySupervisedDataset.__getitem__` calls `data.soccer_data.preprocess_llama_2_soccer_cls`.
- `data.soccer_data.preprocess_llama_2_soccer_cls` returns `input_ids=torch.tensor([0])`, `labels=None`, `caption_info=caption_data`, `video_path`, `timestamp`, `model_type="cls"`.
- `streammind.train_new_stream.train` checks `model_args.soccer_dataset_train_cls`; when true, it freezes the whole model and then sets `requires_grad=True` only for parameters whose name under `model.get_model().mm_projector.named_parameters()` contains `"cls"`.
- `streammind.model.language_model.videollama2_mistral.Videollama2MistralForCausalLM.forward` returns `cls_output` directly when `prepare_inputs_labels_for_multimodal_score_stream` produces it, so Trainer optimizes the Gate loss returned by the Gate CausalLM.

Uncertain:

- The script names say Stage 1 and Stage 2, but there is no explicit stage enum in code. The stage distinction is inferred from the two shell scripts and their flags.
- `scripts/custom/finetune_stage2.sh` does not pass `--pretrain_model_name_or_path`; from the script alone, Stage 2 does not explicitly load Stage 1 output. It may rely on `VideoLLaMA2-7B` or external checkpoint path changes outside this repository, but that is not confirmed by code.

## 2. StreamMind Data Loading

Primary classes/functions:

- `data.datasets.DataArguments`
- `data.datasets.LazySupervisedDataset`
- `data.datasets.DataCollatorForstreamDataset`
- `data.soccer_data.preprocess_llama_2_soccer`
- `data.soccer_data.preprocess_llama_2_soccer_cls`
- `data.live_data.preprocess_llama_2_live`
- `data.live_data.preprocess_llama_2_live_cls`

Soccer data path:

- `LazySupervisedDataset.__init__` searches `data/MatchTime/features_video` for `1_224p.mkv` and `2_224p.mkv`.
- It converts video paths to labels with `data.soccer_data.trans_video_2_json`, replacing `features_video` with `dataset/MatchTime/{data_type}` and replacing half video filenames with `Labels-caption.json`.
- `preprocess_caption_only_caption_data_soccer` reads `annotations`, filters by match half, parses `gameTime`, reverses timestamp/caption order, and builds `timestamp_dict`, `start_timestamp_dict`, `caption_dict`, `half_dict`.
- `__getitem__` loads frame intervals with `process_soccer_video`, then chooses LM or cls preprocessing based on `data_args.soccer_dataset_train_llm`.

Live/Ego4D data path:

- `data.live_data.preprocess_llama_2_live` converts roles `stream`, `user`, `assistant` into prompt turns for LM training.
- `data.live_data.preprocess_llama_2_live_cls` builds a single user prompt from the first user content plus `<video>\n`, and a single assistant label `</silence>`.
- `data.datasets.LazySupervisedDataset.__init__` has additional live/Ego4D/LTA branches, but the provided Stage 1/2 scripts only set `soccer_dataset`.

## 3. StreamMind Stage 2 Gate

Primary classes/functions:

- `streammind.model.multimodal_projector.builder.Video_Mamba_seq`
- `streammind.model.multimodal_projector.builder.ClsNet`
- `streammind.model.multimodal_projector.builder.MistralForCausalLM_cls`
- `streammind.model.videollama2_arch.prepare_inputs_labels_for_multimodal_score_stream`
- `streammind.model.language_model.videollama2_mistral.Videollama2MistralForCausalLM.forward`

Network structure:

- `Video_Mamba_seq` applies:
  - frame patch mean pooling over dimension `l`
  - `PreNet`
  - `VideoMamba`
  - `PostNet`
  - optional `ClsNet`
- `ClsNet` wraps `MistralForCausalLM_cls`.
- `ClsNet.__init__` creates a fresh `MistralConfig` with `vocab_size=2` and `num_hidden_layers=4`, then constructs `MistralForCausalLM_cls`.
- Gate is therefore a small causal LM over embedded frame/prompt sequences with 2 output ids, not a plain binary linear classifier.

Gate input:

- In cls mode, `data.soccer_data.preprocess_llama_2_soccer_cls` provides video frame chunks, captions, timestamps, and `model_type="cls"` but no normal LM labels.
- `streammind.model.videollama2_arch.prepare_inputs_labels_for_multimodal_score_stream` calls `encode_images_or_videos_score_cls_video_cls_autoregressive`, which routes visual features through the Mamba projector.
- In `Video_Mamba_seq.forward`, when `prompt_time_input_ids` and `prompt_time_lable` are available, the Gate input is built by inserting a single frame feature into the prompt template around the original `<video>` position and target prediction position.
- Otherwise, the fallback Gate input is pairs of `[frame_feature, class_token_embedding]`.
- SoccerNet Stage 2 actually takes the fallback branch: `data.soccer_data.preprocess_llama_2_soccer_cls` returns `input_ids=torch.tensor([0])` and `labels=None`, so `Video_Mamba_seq.forward` sees `prompt_time_input_ids.numel() == 1` and does not enter the prompt-template branch.

Gate labels and loss:

- `Video_Mamba_seq.forward` labels non-final frames in a segment as silence and the final frame of a segment as response.
- On the current SoccerNet Stage 2 path, those labels are class ids `0` for hold/silence and `1` for trigger/response in the fallback branch.
- `MistralForCausalLM_cls.forward` computes standard shifted causal LM loss over `config.vocab_size=2`.
- It uses weighted `CrossEntropyLoss`; the last two weights are `0.15` and `0.85`.
- Because `config.vocab_size=2`, these are the weights for class id `0` and class id `1`.

Parameter freezing:

- `streammind.train_new_stream.train` first optionally freezes the backbone if `freeze_backbone` is set.
- In Stage 2, `model_args.soccer_dataset_train_cls=True` triggers:
  - `model.requires_grad_(False)`
  - then only `model.get_model().mm_projector` parameters whose names contain `"cls"` are trainable.
- If not Stage 2 cls training, the same function freezes all `mm_projector` parameters whose names contain `"cls"`.

Uncertain:

- There are several projector builder variants (`builder.py`, `builder_all_clstoken.py`, `builder_two_clstoken.py`, `builder_219.py`). The active file depends on `build_vision_projector` imports and local package wiring. The observed default import path for `mm_projector_type=mamba` points to `streammind/model/multimodal_projector/builder.py`, but this should be rechecked when packaging StreamVAD.

## 4. `silence/response` Label Generation

Confirmed locations:

- `streammind.train_new_stream.train` adds tokenizer special tokens: `["</silence>", "</response>"]`.
- `data.live_data.preprocess_llama_2_live_cls` explicitly creates a cls sample with assistant value `"</silence>"`.
- `streammind.model.multimodal_projector.builder.Video_Mamba_seq.forward` creates Gate labels:
  - non-final frames: hardcoded `32000`
  - final frame: hardcoded `32001`
- Comments in `Video_Mamba_seq.forward` state: `32000` is `</silence>`, `32001` is `</response>`.

For soccer Stage 2:

- The dataset does not contain literal `silence/response` fields.
- `data.soccer_data.preprocess_llama_2_soccer_cls` passes caption boundaries through `caption_info` and `timestamp`.
- `Video_Mamba_seq.forward` derives the original StreamMind Gate labels structurally from caption boundaries: every frame before a caption boundary is labeled as `silence`, and the last frame before the caption is labeled as `response`. StreamVAD plans to reinterpret or rebuild this decision as `hold/trigger`; that is a downstream semantic change, not the original upstream label name.
- In the current SoccerNet Stage 2 fallback branch, these are emitted as `0/1` class labels, not as `32000/32001` tokenizer ids.

Uncertain:

- No external annotation field named `silence` or `response` was found for the Stage 2 soccer path. The labels are generated internally from segment boundaries.

## 5. `32000/32001` Meaning and Hardcoding

Confirmed hardcoding:

- `streammind.train_new_stream.train` adds `</silence>` and `</response>` as tokenizer additional special tokens.
- `streammind.model.multimodal_projector.builder.Video_Mamba_seq.forward` has a prompt-template branch that searches `prompt_time_lable == 32000` and creates labels `32000` / `32001`.
- Equivalent hardcoding exists in `builder_all_clstoken.py` and `builder_two_clstoken.py`.

Meaning:

- `32000` means `</silence>`.
- `32001` means `</response>`.
- This is only correct if the base tokenizer assigns the first two added special tokens those ids. For a Mistral tokenizer with original vocab size 32000, that is plausible and matches code comments.
- In the current SoccerNet Stage 2 path, `32000/32001` are not converted to `0/1`; instead, that prompt-template branch is not triggered, and the fallback branch directly constructs `0/1` labels.

Risk:

- This is not robust to tokenizer changes, added tokens in different order, or a base model with a different vocab size. StreamVAD should resolve ids with `tokenizer.convert_tokens_to_ids("</silence>")` and `tokenizer.convert_tokens_to_ids("</response>")` instead of inheriting numeric constants.
- If the prompt-template branch were executed as written, it would pass labels `32000/32001` to `MistralForCausalLM_cls`, whose logits have final dimension `2` because `ClsNet.__init__` sets `mis_config.vocab_size=2`. That would conflict with binary `CrossEntropyLoss` target bounds and should be avoided or fixed in any downstream StreamVAD refactor.

## 6. Prompt, Streaming State, and Interruption Semantics

Primary functions:

- `streammind.model.language_model.videollama2_mistral.Videollama2MistralForCausalLM.stream_generate`
- `streammind.model.language_model.videollama2_mistral.Videollama2MistralForCausalLM.stream_generate_demo`
- `streammind.model.videollama2_arch.prepare_inputs_labels_for_multimodal_score_stream_inference`
- `streammind.model.videollama2_arch.prepare_inputs_labels_for_multimodal_score_stream_inference_demo`
- `streammind.model.videollama2_arch.Videollama2MetaForCausalLM.encode_images_or_videos_score_cls_inference_allframe_demo`
- `streammind.model.multimodal_projector.builder.Video_Mamba_seq.forward`
- `streammind.model.multimodal_projector.ssm.VideoMamba.forward`
- `streammind/eval/inference_video_score_stream.py`

Confirmed behavior:

- `Videollama2MistralForCausalLM.__init__` initializes mutable per-instance stream fields: `self.frame_feature`, `self.past_review_caption`, `self.past_review_caption_list`, and `self.interval_id_list`.
- `stream_generate` tokenizes `self.past_review_caption` when present and passes it as `past_review_caption`; after LLM generation it appends the decoded output string back into `self.past_review_caption`.
- `prepare_inputs_labels_for_multimodal_score_stream_inference` inserts `past_review_caption` as text embeddings before the current visual features. This makes it generated language history, not the primary visual event memory.
- The active `stream_generate_demo` path passes `self.frame_feature` and `self.interval_id_list` into `prepare_inputs_labels_for_multimodal_score_stream_inference_demo`.
- `encode_images_or_videos_score_cls_inference_allframe_demo` extracts CLIP frame features into `frames_features`, concatenates `past_frames_features` when present, computes `interval_id = frames_features.shape[1]`, and sends the accumulated tensor through `temporal_aggregator`.
- `temporal_aggregator` routes `mm_projector_type` values containing `mamba` into `self.get_model().mm_projector(...)`.
- `Video_Mamba_seq.forward` reduces the CLIP patch dimension with `torch.mean(x, dim=2)`, runs `pre_net`, `self.mamba_model`, and `post_net`, then returns either frame-level projected features or Gate features depending on flags.
- `VideoMamba.forward` iterates through SSM blocks with `hidden_states` and `residual`, accepts optional `inference_params`, and returns `hidden_states`; the audited StreamMind wrapper calls `self.mamba_model(x)` without passing an external cache object.
- The Gate output is softmaxed and `argmax` is used:
  - `pred == 0`: return `None` and skip LLM generation.
  - `pred == 1`: construct LLM input embeddings and call `super().generate`.
- Eval code in `streammind/eval/inference_video_score_stream.py` resets `model.frame_feature = None` and `model.past_review_caption = None` between videos.

### Semantic Interruption and Event Refocusing

This means "interruption" in the sense of occlusion, noise frames, or short irrelevant content inside the same running video stream is primarily a visual-temporal modeling issue, not a process-resume issue.

Code-confirmed facts:

- `streammind/model/videollama2_arch.py`, function `encode_images_or_videos_score_cls_inference_allframe_demo`, variables `frames_features`, `past_frames_features`, and `interval_id`: the demo stream path keeps accumulating CLIP frame features by concatenating previous `self.frame_feature` with the current frame features before Mamba projection.
- `streammind/model/videollama2_arch.py`, function `prepare_inputs_labels_for_multimodal_score_stream_inference_demo`, variables `frames_features`, `interval_id`, and `interval_id_list`: on `pred == 0`, the accumulated `frames_features` and current `interval_id` are returned to the caller; on `pred == 1`, the code appends `interval_id` into `interval_id_list` and slices `X_features` between prior trigger boundaries and the current boundary for the LLM input.
- `streammind/model/multimodal_projector/builder.py`, class `Video_Mamba_seq`, function `forward`, variable `x`: the projector turns per-frame CLIP features into temporal features through `self.mamba_model(x)`.
- `streammind/model/multimodal_projector/ssm.py`, class `VideoMamba`, function `forward`, variables `hidden_states`, `residual`, and `inference_params`: the Mamba/SSM stack internally propagates hidden activations across the input sequence during a forward pass. The current wrapper does not expose a named `H_t`, perception-memory object, or saved recurrent state at the StreamMind API level.
- `streammind/model/language_model/videollama2_mistral.py`, class `Videollama2MistralForCausalLM`, function `stream_generate`, variables `past_caption_ids` and `self.past_review_caption`: `past_review_caption` is tokenized generated text and is appended only after LLM output. It should be described as language-output history used in prompt construction, not as the main visual state that lets the model refocus after occlusion.

Paper-supported but not fully code-confirmed:

- The StreamMind paper describes EPFE as taking `CLIP(v_t)` plus the previous hidden state `H_{t-1}` to produce a perception token and update its internal state, and says perception tokens are stored in perception memory. The public code implements a Mamba temporal projector over accumulated CLIP features, but it does not expose explicit variables named `H_t`, `perception token`, or `perception memory`.
- The paper claims that EPFE perception tokens can distinguish unrelated frames and refocus on the main event after irrelevant content. In code, the closest evidence is accumulated `frames_features` plus Mamba temporal projection and, in demo mode, `interval_id_list`-based slicing of projected features. The exact paper-level "perception memory" abstraction is not directly identifiable as a single public-code variable; `frame_feature` is an accumulated CLIP feature tensor and should not be treated as strictly equivalent to paper perception memory.

Conclusion:

- For semantic interruption, visual event continuity should be attributed to the visual stream path: accumulated `frame_feature`/`frames_features`, Mamba-based temporal projection in `Video_Mamba_seq`/`VideoMamba`, and trigger-boundary bookkeeping through `interval_id_list` in demo mode. `past_review_caption` can condition later language generation with previous text, but it is not the main mechanism for visual event refocusing.

### Process Interruption and State Persistence

This is a separate case: program exit, model destruction, and later model reload.

Code-confirmed facts:

- `streammind/model/language_model/videollama2_mistral.py`, class `Videollama2MistralForCausalLM`, function `__init__`, variables `self.frame_feature`, `self.past_review_caption`, `self.past_review_caption_list`, and `self.interval_id_list`: streaming state is stored as ordinary in-memory model-instance attributes.
- `streammind/eval/inference_video_score_stream.py`, functions `run_inference` and `run_inference_caption_metric`, variables `model.frame_feature` and `model.past_review_caption`: the evaluator explicitly clears these fields between videos.
- `streammind/model/multimodal_projector/ssm.py`, class `VideoMamba`, function `allocate_inference_cache`, variable `inference_params`: the Mamba module has a cache allocation method, but the StreamMind projector path in `Video_Mamba_seq.forward` calls `self.mamba_model(x)` without passing, returning, serializing, or restoring `inference_params`.
- Repository-wide search found training-checkpoint `save_state`/`torch.save` calls, but no formal serialization protocol for per-video stream state such as `frame_feature`, `past_review_caption`, `interval_id_list`, a video cursor, or Mamba `inference_params`.

Uncertain:

- The public code does not define a formal process-level "interrupt and continue" or checkpoint-resume API for online inference. Process restart recovery is therefore not supported or not found in the audited code.
- A future StreamVAD implementation that needs restart-safe inference would need to define what to persist and restore, likely including video position, pending visual features or derived perception representations, trigger boundaries, and language history. The upstream code does not specify that contract.

## 7. Vad-R1 SFT

Primary files:

- `Vad-R1/src/scripts/run_sft_video.sh`
- `Vad-R1/src/r1-v/src/open_r1/sft_video.py`
- `Vad-R1/data/Vad-Reasoning-SFT-train.jsonl`
- `Vad-R1/data/Vad-Reasoning-SFT-test.jsonl`

Training entry:

- Shell entry: `src/scripts/run_sft_video.sh`
- Python entry: `src/r1-v/src/open_r1/sft_video.py`
- Important script parameters:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`
  - `torchrun --nproc_per_node=4`
  - `--dataset_name "/home/wbf/VideoAnomaly-R/data/Vad-Reasoning-SFT-train.jsonl"`
  - `--per_device_train_batch_size 1`
  - `--gradient_accumulation_steps 2`
  - `--learning_rate 1e-6`
  - `--num_train_epochs 4`
  - `--bf16`
  - `--gradient_checkpointing true`
  - `--attn_implementation flash_attention_2`
  - `--deepspeed local_scripts/zero2.json`
  - base model path points to a local Qwen2.5-VL directory.

JSONL fields confirmed from data:

- Abnormal rows include: `source`, `video`, `anomaly_type`, `start`, `end`, `total_frames`, `path`, `think`, `answer`.
- Normal rows include: `source`, `video`, `anomaly_type`, `total_frames`, `path`, `think`, `answer`; no `start/end` in the sampled normal row.

SFT conversion:

- `sft_video.prepare_dataset` builds messages:
  - system text: `You are a multimodal reasoning assistant tasked with analyzing videos.`
  - user content: one video item from `example["path"]` plus a fixed long `USER_PROMPT`
  - assistant content: `example["think"] + "\n" + example["answer"]`
- `sft_video.collate_fn` applies `processor.apply_chat_template`, calls `process_vision_info`, and masks pad tokens plus visual tokens in labels.
- The full assistant CoT and answer remain supervised labels.

Normal CoT format:

- Prompt requires `<think><step1>...</step1><step2>...</step2></think>` and `<answer>` with `<which>Normal</which>`, `<what>`, `<why>`.
- Confirmed in `data/Vad-Reasoning-SFT-train.jsonl` normal sample around video `01_015`.

Abnormal CoT format:

- Prompt requires `<think>` with `<step1>` through `<step4>`.
- `<answer>` includes `<which>`, `<what>`, `<when>[s,e]</when>`, `<where>`, `<why>`, `<how>`.
- Confirmed in initial abnormal samples in `data/Vad-Reasoning-SFT-train.jsonl`.

## 8. Vad-R1 `start/end` Use and Future-Leakage Risk

Confirmed `start/end` use:

- `sft_video.prepare_dataset` does not read `start`, `end`, `total_frames`, or `anomaly_type`; it only uses `path`, `think`, and `answer`.
- `evaluation/1-evaluate_detection.py.load_jsonl_as_dict_gt` converts abnormal `start/end/total_frames` into normalized GT time ranges. Normal videos use `[0, 1]`.
- `src/r1-v/src/open_r1/trainer/anomaly_grpo_trainer.py.extract_time` parses generated `<when>[s,e]</when>`.
- `anomaly_grpo_trainer.compute_loss` uses parsed generated time as `ignored_intervals` and calls `process_vision_info(..., ignored_intervals=ignored_intervals)`.
- `src/qwen-vl-utils/src/qwen_vl_utils/vision_process.py.smart_nframes_with_ignored_intervals` removes frames in that normalized interval and samples remaining frames.

Future-information risk:

- For SFT, the model input is the whole video path processed by `process_vision_info`; there is no streaming prefix restriction. The assistant label includes the full CoT and final `<when>`, so this is full-video reasoning, not online VAD.
- The CoT can mention events and consequences after the anomaly onset because it is trained against complete-video annotations.
- For a streaming StreamVAD setting, directly using full `think+answer` at early timestamps would leak future information unless truncated/converted into prefix-observable labels.

Uncertain:

- The dataset generation process for `think`/`answer` is not fully included. Therefore, whether the human/GPT annotation procedure itself used future frames beyond each statement is not verifiable from code. The presence of whole-video `path` SFT and complete final answers is sufficient to flag leakage risk for streaming adaptation.

## 9. Dependencies and Licenses

StreamMind:

- `StreamMind/README.md` says Python >= 3.10, PyTorch >= 2.5.1, CUDA >= 11.8, transformers >= 4.44.2, tokenizers >= 0.19.1, plus `flash-attn==2.5.8`.
- `StreamMind/requirements.txt` pins older versions including `torch==2.2.0`, `torchvision==0.17.0`, `transformers==4.40.0`, `deepspeed==0.13.1`, `mamba-ssm==1.2.0.post1`.
- `StreamMind/pyproject.toml` also pins `torch==2.2.0`, `transformers==4.40.0`, `tokenizers==0.19.1`, `deepspeed==0.13.1`, `peft==0.4.0`, `bitsandbytes==0.43.0`, `gradio==3.50.0`, etc.
- License file: `StreamMind/LICENSE` is Apache License 2.0.
- README license note additionally says the service is a research preview for non-commercial use only and subject to LLaMA/Mistral/model/data terms. This is a material extra constraint for downstream use.

Vad-R1:

- `Vad-R1/setup.sh` installs `src/r1-v` editable with `.[dev]`, then installs `wandb==0.18.3`, `tensorboardx`, `qwen_vl_utils`, `torchvision`, `flash-attn`, and `vllm==0.7.2`.
- `Vad-R1/src/r1-v/setup.py` requires Python `>=3.10.9` and dependencies including `accelerate>=1.2.1`, `bitsandbytes>=0.43.0`, `datasets>=3.2.0`, `deepspeed==0.15.4`, `einops>=0.8.0`, `liger_kernel==0.5.2`, `torch>=2.5.1`, `trl==0.16.0`, `vllm==0.7.2`.
- README says the experiments can train on 4 A100 80G GPUs and require installing a provided `transformers-main.zip`.
- License file found at `Vad-R1/src/r1-v/LICENSE` is Apache License 2.0.
- No top-level `Vad-R1/LICENSE` was found by `find`; the license is under `src/r1-v/`.

## 10. StreamVAD-Relevant Takeaways

- StreamMind's original Gate labels are caption-boundary-derived `silence/response` labels, not explicit dataset fields. StreamVAD plans to remodel the Gate decision as `hold/trigger`.
- StreamMind's `32000/32001` prompt-template branch must not be copied as numeric ids; current SoccerNet Stage 2 avoids it by using the fallback `0/1` branch.
- StreamMind's streaming resume state is in Python object fields, not a durable protocol.
- Vad-R1 SFT is full-video P2C supervision; it is useful for compact reasoning style, but must be converted into prefix-safe labels for streaming.
- Vad-R1 `start/end` is reliable for full-video temporal GT/evaluation, but using final `think+answer` before the end frame would leak future information.
