#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

: "${MODEL_PATH:?set MODEL_PATH to the server VideoLLaMA2 checkpoint path}"
: "${VISION_TOWER:?set VISION_TOWER to the server CLIP vision tower path}"

streamvad_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  streamvad_args+=(--streamvad-max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${ALLOW_UNRELIABLE_TIMING:-}" ]]; then
  streamvad_args+=(--allow-unreliable-timing)
fi

torchrun \
  --nnodes "${NNODES:-1}" \
  --nproc_per_node "${NPROC_PER_NODE:-1}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --master_port "${MASTER_PORT:-16677}" \
  --node_rank "${NODE_RANK:-0}" \
  tools/train_streamvad_stage1_lora.py \
  --streammind-root "${STREAMMIND_ROOT:-${STREAMVAD_ROOT:-${REPO_ROOT}}/StreamMind}" \
  "${streamvad_args[@]}" \
  --data_path "${STREAMVAD_STAGE1_JSONL:-data/streamvad_weak_supervision/streamvad_stage1_train.jsonl}" \
  --output_dir "${OUTPUT_DIR:-output/streamvad_stage1_lora}" \
  --deepspeed "${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.json}" \
  --version v1_mistral \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --freeze_backbone True \
  --lora_enable True \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --mm_projector_type mamba \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --num_frames "${NUM_FRAMES:-32}" \
  --bf16 "${BF16:-True}" \
  --tf32 "${TF32:-True}" \
  --fp16 "${FP16:-False}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${LOCAL_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-500}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --model_max_length "${MODEL_MAX_LENGTH:-2048}" \
  --gradient_checkpointing True \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --report_to "${REPORT_TO:-none}" \
  --run_name "${RUN_NAME:-streamvad_stage1_lora}"
