#!/bin/bash
# Re-evaluate FlashReFocus rendering with bf16 autocast (adopted setting; the bokeh tables were already rendered this way):
# RealBokeh validation, MODEST ladders, LVCorr on LFDOF and 3CReal. Quality only (timing is done separately).
B=/ssd1/chingheng/Defocus/refocus_bench; cd $B; export CUDA_VISIBLE_DEVICES=0; PY=~/miniconda3/envs/pisa/bin/python
( $PY rbval_ours.py --root realbokeh_val --out rbval_out/ours_bf16 > rbval_out/ours_bf16.log 2>&1; echo "$(date +%T) rbval $?" ) &
( for ap in F16.0 F9.0 F5.0 F2.8; do $PY rbval_ours.py --root modest_eval/$ap --out rbval_out/modest_ours_bf16_$ap > rbval_out/modest_ours_bf16_$ap.log 2>&1; echo "$(date +%T) modest $ap $?"; done
  $PY lvcorr.py --method derefocus --tag _bf16 > lvcorr/derefocus_bf16.log 2>&1; echo "$(date +%T) lvcorr lfdof $?"
  $PY lvcorr.py --method derefocus --bench 3creal_lvcorr --tag _3creal_bf16 > lvcorr/derefocus_3creal_bf16.log 2>&1; echo "$(date +%T) lvcorr 3creal $?" ) &
wait; echo ALL_DONE
