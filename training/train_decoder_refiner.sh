#!/bin/bash
# Decoder refiner on Stage I latents at omega = 1.25: 15k steps, batch 1 x gradient accumulation 2 x 2 GPUs, bf16,
# initialised from the released refiner (pass --refiner_init none to start from the SD1.5 decoder).
# Run from the repository root.
set -e
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --num_processes 2 --multi_gpu --mixed_precision bf16 training/train_decoder_refiner.py \
  --stage1_checkpoint "${STAGE1_CHECKPOINT:-${FLASHREFOCUS_WORK:-work}/runs/deblurring/checkpoints/model_40000.pkl}" \
  --detail_weight 1.25 --output_dir ${FLASHREFOCUS_WORK:-work}/runs/decoder_refiner \
  --train_batch_size 1 --gradient_accumulation_steps 2 --max_train_steps 15000 --grad_ckpt "$@"
