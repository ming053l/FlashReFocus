#!/bin/bash
# Decoder refiner on Stage I latents: 15k steps, batch 1 x gradient accumulation 2 x 2 GPUs, bf16, initialised from the
# released refiner (pass --refiner_init none to start from the SD1.5 decoder). --lambda_sem is the inference omega.
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --num_processes 2 --multi_gpu --mixed_precision bf16 train_refiner.py \
  --stage1_ckpt "${STAGE1_CKPT:-runs/deblur/checkpoints/model_40000.pkl}" --lambda_sem 1.25 --output_dir runs/refiner \
  --train_batch_size 1 --gradient_accumulation_steps 2 --max_train_steps 15000 --grad_ckpt "$@"
