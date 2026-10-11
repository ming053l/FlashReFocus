#!/bin/bash
# Stage II training of the released renderer (weights/renderer.pt):
#   1. pre-training on synthetic multi-focal Hypersim scenes, 3 epochs;
#   2. adaptation to RealBokeh pairs, 5 epochs, with Depth Anything V2 depth (cache it first with
#      generate_da2_depth.py) and a 10x larger learning rate for the defocus-conditioning parameters.
# Data locations: see paths.py (FLASHREFOCUS_DATA).
set -e
cd "$(dirname "$0")"

python train_renderer.py --phase hypersim --epochs 3 --lr 5e-4 --amp --save-dir checkpoints/hypersim

python train_renderer.py --phase realbokeh --epochs 5 --lr 5e-5 --defocus-lr 5e-4 --amp \
  --pretrained checkpoints/hypersim/best.pt --save-dir checkpoints/realbokeh
