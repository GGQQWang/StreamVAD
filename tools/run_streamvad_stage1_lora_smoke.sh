#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"

torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  --master_addr 127.0.0.1 \
  --master_port "${MASTER_PORT:-16677}" \
  --node_rank 0 \
  tools/train_streamvad_stage1_lora.py \
  --streammind-root "${STREAMMIND_ROOT:-${STREAMVAD_ROOT:-${REPO_ROOT}}/StreamMind}" \
  --streamvad-max-samples "${MAX_SAMPLES}" \
  --data_path "${STREAMVAD_STAGE1_JSONL:-data/streamvad_weak_supervision/streamvad_stage1_train.jsonl}" \
  --output_dir "${OUTPUT_DIR:-output/streamvad_stage1_lora_smoke}" \
  --deepspeed "${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.json}" \
  --version v1_mistral \
  --model_name_or_path "${MODEL_PATH:?set MODEL_PATH to the server VideoLLaMA2 checkpoint path}" \
  --vision_tower "${VISION_TOWER:?set VISION_TOWER to the server CLIP vision tower path}" \
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
  --bf16 True \
  --tf32 True \
  --fp16 False \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${LOCAL_BATCH_SIZE}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 8 \
  --save_total_limit 2 \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --report_to "${REPORT_TO:-none}" \
  --run_name streamvad_stage1_lora_smoke
