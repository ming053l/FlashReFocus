# FlashReFocus: Interactive Image Refocusing in Seconds

Single-image refocusing moves the focal plane and changes the depth of field of a photograph after capture. FlashReFocus
splits it by what each part needs: deblurring has to recover lost detail, so it uses a one-step diffusion model, run
**once** per photograph; rendering new blur is determined by depth and the focal plane, so every edit is rendered by an
explicit 1.3M-parameter network that leaves in-focus content untouched.

- **Stage I, deblurring.** One-step diffusion on Stable Diffusion 2.1 with a pixel LoRA for fidelity and a
  detail LoRA weighted by ω at inference, trained on a mixture of real and synthetic defocus pairs, plus a decoder
  refiner that reinjects structure of the blurred input lost in VAE decoding.
- **Stage II, rendering.** A residual encoder–decoder conditioned on a defocus map from Depth Anything V2 depth, the
  focal point and the blur strength. Defocus-bias attention relates pixels by how similarly they are defocused rather
  than by their distance in the image. Pre-trained on synthetic multi-focal Hypersim scenes, adapted to RealBokeh pairs.
- **Edits without retraining.** Focal point, blur strength, two focal planes and tilt-shift, all from one restored image.

## Repository layout

```
paths.py      locations of weights, data and outputs (environment variables, see below)
pipeline/     refocusing pipeline (fast_refocus.py), two focal planes, tilt-shift, cost benchmark
deblur/       Stage I model (model.py), inference, training data (deblur_data.py), Stage I and decoder-refiner training
refiner/      decoder refiner (VAEMultiScaleFusionDecoder)
renderer/     Stage II renderer: model (method/), training data (render_data.py), training, reference inference
eval/         evaluation: bokeh-rendering tables, LVCorr, focus-sweep consistency, MODEST aperture series
annotations/  our focal-point annotations and evaluation subsets (EBB!, RealBokeh, 3CReal, LFDOF)
tools/        data preparation
third_party/  base one-step diffusion code, Depth Anything V2, RAM++ (git submodules)
```

## Setup

```bash
git clone --recursive <this repository> FlashReFocus && cd FlashReFocus
conda create -n flashrefocus python=3.10 && conda activate flashrefocus
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Weights

Weights go into `weights/` (or set `FLASHREFOCUS_WEIGHTS`). The public ones download with `huggingface-cli`
(`pip install huggingface_hub`):

```bash
huggingface-cli download sd2-community/stable-diffusion-2-1-base --local-dir weights/stable-diffusion-2-1-base
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 --include "vae/*" --local-dir weights/stable-diffusion-v1-5
huggingface-cli download depth-anything/Depth-Anything-V2-Large depth_anything_v2_vitl.pth --local-dir weights
# Stage I training only: RAM++ for the prompts, and the pretrained one-step LoRAs used as initialisation
huggingface-cli download xinyu1205/recognize-anything-plus-model ram_plus_swin_large_14m.pth --local-dir weights
```

The initial LoRAs for Stage I training (`pisa_sr.pkl`) go to `third_party/PiSA-SR/preset/models/PiSA-SR/`, as described in
that repository. The FlashReFocus weights (`stage1.pkl`, `refiner.pt`, `renderer.pt`) will be released with the paper.

```
weights/
  stable-diffusion-2-1-base/      Stage I backbone
  stable-diffusion-v1-5/vae/      decoder refiner backbone
  depth_anything_v2_vitl.pth      depth for the renderer
  stage1.pkl  refiner.pt  renderer.pt
  ram_plus_swin_large_14m.pth     Stage I training only
```

### Data

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
| `3CReal/validation/` (`original/`, `bokeh/`) | 3CReal (ours, 89 pairs) | bokeh-rendering evaluation | released with the paper |
| `MODEST/` | MODEST | aperture-series evaluation (optional) | `huggingface-cli download independent-vision-lab/MODEST --repo-type dataset --local-dir data/MODEST` |
| `Hypersim_processed/` | Hypersim scenes with rendered bokeh targets | renderer pre-training | scenes from [github.com/apple/ml-hypersim](https://github.com/apple/ml-hypersim); target rendering script to be released |

Stage I trains on 4,179 RealBokeh pairs of the top-1000 scenes (`deblur/configs/deblur_train_realbokeh_top1000_laplacian.csv`);
these manifests and the test pairs in `deblur/pairs/` store paths relative to `data/`. Our focal-point annotations for
EBB!, RealBokeh, 3CReal and LFDOF are in `annotations/`; `tools/prepare_data.py` copies them next to the images and
builds the evaluation folders:

```bash
python tools/prepare_data.py check   # which datasets are in place
python tools/prepare_data.py all     # EBB_Val176/, RealBokeh_val_f2.0/, work/lfdof_refocus/, RealBokeh and 3CReal focal points
python renderer/generate_da2_depth.py          # cache Depth Anything V2 depth of RealBokeh for renderer training
```

Evaluation outputs go into `work/` (`FLASHREFOCUS_WORK`); `paths.py` lists every location.

## Refocusing

```python
import sys; sys.path.insert(0, "pipeline")
import fast_refocus as FR

s1 = FR.FastStageI()                       # Stage I with the refiner, omega = 1.25 (s1.omega = ... to change)
s1("photo.png", "aif.png")                 # restore the all-in-focus image once
rd = FR.FastRenderer()                     # Stage II
rd.set_image("aif.png")                    # depth is estimated once and reused
rd.render(x, y, 0.5, "refocused.png")      # every edit: focal point (x, y), strength in [0, 1]
```

Image sides should be multiples of 16. Two focal planes and tilt-shift edit the depth map given to the renderer
(`rd.render(x, y, s, depth=rd.two_plane_depth(a, b))`, `rd.tilt_shift_depth(row)`), also as scripts:

```bash
python pipeline/dual_focus.py aif.png xA yA xB yB 0.8 out/dual
python pipeline/tiltshift.py photo.png out/tilt --row 0.3 --strength 0.6
```

For scripts and notebooks, `deblur/model.py` exposes the restoration directly:
`Deblurrer(load_stage1(omega=1.25), load_refiner())(read_image("photo.png"))`.

Outputs are bit-reproducible with `FR.setup_backend(False)`; `setup_backend(True)` (cuDNN autotuning) is faster for a
fixed input size, but may pick different convolution algorithms between runs, changing outputs by a few 1/255.
`FastStageI(merge_vae_lora=True)` merges the VAE-encoder LoRA for a ~20% faster encoder (outputs ~60 dB from the exact ones).

## Training

| Stage | Script | Data |
|---|---|---|
| Stage I | `deblur/train_deblur.sh` (pixel stage, then detail stage) | DPDD, RealBokeh (`deblur/configs/*.csv`), LFDOF, DIV2K |
| Decoder refiner | `deblur/train_refiner.sh` | same mixture, Stage I latents at ω = 1.25 |
| Renderer | `renderer/train_renderer.sh` (Hypersim pre-training, RealBokeh adaptation) | Hypersim with BokehMe targets, RealBokeh |

## Evaluation

| Result | Scripts |
|---|---|
| Defocus deblurring (DPDD, RealDOF) | `deblur/deblur.py --pairs deblur/pairs/<set>_test.json`, `eval/score_folders.py` |
| Bokeh rendering (EBB! Val176, 3CReal, RealBokeh val.) | `eval/rbval_ours.py`, `eval/eval_bokeh_table.py`, `eval/score_folders.py` |
| Strength control (LVCorr) | `eval/lvcorr.py` |
| Focus-sweep consistency | `eval/rack_ours.py`, `eval/rack_genrefocus.py`, `eval/rack_match_blur*.py`, `eval/rack_analyze.py`, `eval/rack_metrics_kmatch.py` |
| MODEST aperture series | `eval/build_modest_series.py`, `eval/modest_analyze.py` |
| Effect of the defocus bias on attention (Appendix C.1) | `eval/attention_bias.py` |
| Renderer residual, defocus embedding and strength scale (Appendix C.1) | `eval/renderer_internals.py` |
| Where the detail branch acts (Appendix F) | `eval/detail_branch.py` |
| Depth source: DA-V2 S/B/L and Depth Pro (Appendix G) | `eval/depth_source.py` |
| Cost | `deblur/cost_stage1.py`, `pipeline/bench_fast_refocus.py` |

Outputs of compared methods are produced with their own code (under `baselines/`, `FLASHREFOCUS_BASELINES`) and
scored with the same scripts.

## Acknowledgements

The code builds on [PiSA-SR](https://github.com/csslc/PiSA-SR), [Bokehlicious](https://github.com/TimSeizinger/Bokehlicious),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[RAM++](https://github.com/xinyu1205/recognize-anything); their licenses apply to the corresponding parts.
