#!/bin/bash
# Stage I training of the released model (weights/stage1.pkl):
#   1. pixel stage, 20k steps on 2 GPUs (batch 4 x gradient accumulation 2 x 2 GPUs = 16): rank-16 pixel LoRA kept at
#      the scale of the rank-4 initialisation, rank-4 LoRA on the VAE encoder, trainable U-Net conv_in;
#   2. detail stage from step 20k to 40k on one GPU (batch 4 x gradient accumulation 4 = 16).
# Weights and data locations: see paths.py. If NCCL fails with illegal memory accesses, keep P2P/SHM disabled.
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
PIX="--pix_steps 20000 --lora_rank_unet_pix 16 --pix_lora_scaling 2 --pix_lora_a_std 0.25 --vae_lora_rank 4 --train_conv_in"

accelerate launch --num_processes 2 --multi_gpu --mixed_precision fp16 train_deblur.py \
  --output_dir runs/stage1_pixel $PIX --max_train_steps 20000

accelerate launch --num_processes 1 --mixed_precision fp16 train_deblur.py \
  --output_dir runs/stage1 --resume runs/stage1_pixel/checkpoints/model_20000.pkl $PIX \
  --train_batch_size 4 --gradient_accumulation_steps 4 --max_train_steps 40000
