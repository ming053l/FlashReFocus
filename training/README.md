# Training

Run from the repository root. Install and download the inference weights as in the [top-level README](../README.md#setup).

## Additional weights

```bash
# RAM++ for the Stage I prompts
huggingface-cli download xinyu1205/recognize-anything-plus-model ram_plus_swin_large_14m.pth --local-dir weights
```

The initial LoRAs for Stage I training (`pisa_sr.pkl`) go to `third_party/PiSA-SR/preset/models/PiSA-SR/`, as described in
that repository. The released checkpoints are trained models, not a complete training-resume snapshot for all stages.

## Data

Refocusing your own photos needs no dataset. For training and evaluation, download the datasets into `data/` (or set
`FLASHREFOCUS_DATA`) with the folder names below, then build the evaluation sets:

| Folder in `data/` | Dataset | Used for | Source |
|---|---|---|---|
| `dd_dp_dataset_png/` (`train_c`, `val_c`, `test_c`, each with `source/`, `target/`) | DPDD, Canon PNG version | Stage I training, deblurring evaluation | [github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) |
| `RealDOF/` (`source/`, `target/`) | RealDOF test set | deblurring evaluation | [github.com/codeslake/IFAN](https://github.com/codeslake/IFAN) (Datasets) |
| `LFDOF/` (`train_data/`, `test_data/`) | LFDOF | Stage I training, LVCorr benchmark | [sweb.cityu.edu.hk/miullam/AIFNET](https://sweb.cityu.edu.hk/miullam/AIFNET/) |
| `div2k/DIV2K_train_HR/` | DIV2K train HR | Stage I training (synthetic blur) | [data.vision.ee.ethz.ch/cvl/DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| `RealBokeh/` (`train/`, `validation/`, `test/`) | RealBokeh_3MP | Stage I and renderer training, evaluation | `huggingface-cli download timseizinger/RealBokeh_3MP --repo-type dataset --local-dir data/RealBokeh` |
| `EBB/` (`train/`, `validation/`) | EBB! | bokeh-rendering evaluation | [aiff22.github.io/pynet-bokeh.html](https://aiff22.github.io/pynet-bokeh.html#dataset) |
| `3CReal/validation/` (`original/`, `bokeh/`, `focus_point.json`) | [3CReal](https://huggingface.co/datasets/ming0531/3CReal) (ours, 89 pairs, CC BY-NC 4.0) | bokeh-rendering evaluation | `huggingface-cli download ming0531/3CReal --repo-type dataset --local-dir data/3CReal/validation` |
| `MODEST/` | MODEST | aperture-series evaluation (optional) | `huggingface-cli download independent-vision-lab/MODEST --repo-type dataset --local-dir data/MODEST` |
| `Hypersim_processed/` | Hypersim scenes with rendered bokeh targets | renderer pre-training | scenes from [github.com/apple/ml-hypersim](https://github.com/apple/ml-hypersim); target rendering script to be released |

Stage I trains on 4,179 RealBokeh pairs of the top-1000 scenes (`training/configs/realbokeh_deblurring_top1000.csv`);
this manifest and the test pairs in `annotations/deblurring_pairs/` store paths relative to `data/`. Our focal-point annotations for
EBB!, RealBokeh, 3CReal and LFDOF are in `annotations/`; `scripts/prepare_data.py` copies them next to the images and
builds the evaluation folders:

```bash
python scripts/prepare_data.py check   # which datasets are in place
python scripts/prepare_data.py all     # EBB_Val176/, RealBokeh_val_f2.0/, work/lfdof_refocus/, RealBokeh and 3CReal focal points
python scripts/cache_depth.py          # cache Depth Anything V2 depth of RealBokeh for renderer training
```

Evaluation outputs go into `work/` (`FLASHREFOCUS_WORK`); `flashrefocus/paths.py` lists every location.

## Stages

| Stage (paper section) | Script | Data |
|---|---|---|
| Stage I: deblurring (Sec. 3.1) | `training/train_deblurring.sh` (pixel phase, then detail phase) | DPDD, RealBokeh, LFDOF, DIV2K (`training/configs/deblurring_data.yaml`) |
| Stage I: decoder refiner (Sec. 3.1) | `training/train_decoder_refiner.sh` | same mixture, Stage I latents at ω = 1.25 |
| Stage II: bokeh rendering (Sec. 3.2) | `training/train_renderer.sh` (Hypersim pre-training, RealBokeh adaptation) | Hypersim with BokehMe targets, RealBokeh |
