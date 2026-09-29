#!/bin/bash
# Semantic-stage branches from the v3 pixel stage (model_20000.pkl), one GPU each (batch 4 x accum 4 = 16):
#   train_branch_v3.sh s1 0            official semantic recipe (L2 + 2 LPIPS + 1 CSD)
#   train_branch_v3.sh s2 1 --lambda_gan 0.1   + adversarial loss (Real-ESRGAN U-Net discriminator)
cd "$(dirname "$0")"
name=$1; gpu=$2; shift 2
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu
~/miniconda3/envs/pisa/bin/accelerate launch --num_processes 1 --mixed_precision fp16 \
  train_pisa_defocus.py --output_dir runs/v3_$name --resume runs/train_pisa_defocus_v3/checkpoints/model_20000.pkl \
  --pix_steps 20000 --lora_rank_unet_pix 16 --pix_lora_scaling 2 --pix_lora_a_std 0.25 --vae_lora_rank 4 --train_conv_in \
  --train_batch_size 4 --gradient_accumulation_steps 4 --max_train_steps 60000 "$@"
