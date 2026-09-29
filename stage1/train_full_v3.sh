#!/bin/bash
# v3 = v2 with the pixel-LoRA scaling fixed: pixel stage 20k steps, rank-16 pixel LoRA kept at the released
# LoRA's scale (scaling 2, A ~ N(0, 0.25^2)), rank-4 VAE-encoder LoRA, trainable UNet conv_in; then the
# semantic stage. (v2, rank 16 at peft's default scaling 8/16, learned 4-16x slower per step: stopped at 936.)
# Validation adds held-out LFDOF test images and a sharpness ratio.
# NCCL P2P/SHM transports crash with illegal memory access on this machine; socket transport is used instead.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
~/miniconda3/envs/pisa/bin/accelerate launch --num_processes 2 --multi_gpu --mixed_precision fp16 \
  train_pisa_defocus.py --output_dir runs/train_pisa_defocus_v3 \
  --pix_steps 20000 --lora_rank_unet_pix 16 --pix_lora_scaling 2 --pix_lora_a_std 0.25 \
  --vae_lora_rank 4 --train_conv_in "$@"
