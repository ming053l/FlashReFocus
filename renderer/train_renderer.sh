#!/bin/bash
# Stage II training of the released renderer (weights/renderer.pt):
#   1. pre-training on synthetic multi-focal Hypersim scenes, 3 epochs;
#   2. adaptation to RealBokeh pairs, 5 epochs, with Depth Anything V2 disparity (cache it first with
#      generate_da2_depth.py) and separate learning rates for the defocus modules and the backbone.
# Data locations: see paths.py (FLASHREFOCUS_DATA).
set -e
cd "$(dirname "$0")"
DATA=${FLASHREFOCUS_DATA:-../data}

python train_hypersim_dma_p.py --data-dir "$DATA/Hypersim_processed" \
  --size small --crop-size 512 --batch-size 4 --epochs 3 --lr 5e-4 --zero-aperture-batch-p 0.1 --amp \
  --attn-version defocus --depth-map-dropout-p 0.0 --inverse-depth --focus-map-type gaussian \
  --save-dir checkpoints/hypersim

python train_realbokeh_dma_p.py --data-dir "$DATA/RealBokeh" --depth-map-dir da2_depth --depth-mode disparity \
  --size small --crop-size 512 --batch-size 4 --num-workers 4 --epochs 5 --depth-defocus-lr 5e-4 --backbone-lr 5e-5 \
  --zero-aperture-batch-p 0.1 --amp --attn-version defocus --depth-map-dropout-p 0.0 --focus-map-type gaussian \
  --pretrained checkpoints/hypersim/best.pt --save-dir checkpoints/realbokeh
