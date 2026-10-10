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
stage1/       Stage I training and inference, decoder-refiner training, data configuration and test pairs
refiner/      decoder refiner (VAEMultiScaleFusionDecoder)
renderer/     Stage II renderer: model, data loaders, training, inference, evaluation
eval/         evaluation: bokeh-rendering tables, LVCorr, focus-sweep consistency, MODEST aperture ladders
third_party/  base one-step diffusion code, Depth Anything V2, RAM++ (git submodules)
```

## Setup

```bash
git clone --recursive <this repository> FlashReFocus && cd FlashReFocus
conda create -n flashrefocus python=3.10 && conda activate flashrefocus
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Weights go into `weights/` (or set `FLASHREFOCUS_WEIGHTS`):

```
weights/
  stable-diffusion-2-1-base/      Stable Diffusion 2.1 base (Stage I)
  stable-diffusion-v1-5/          Stable Diffusion 1.5, only vae/ is used (decoder refiner)
  depth_anything_v2_vitl.pth      Depth Anything V2 ViT-L
  stage1.pkl                      FlashReFocus Stage I LoRAs
  refiner.pt                      FlashReFocus decoder refiner
  renderer.pt                     FlashReFocus renderer
  ram_plus_swin_large_14m.pth     RAM++ (Stage I training only)
```

The FlashReFocus weights will be released with the paper. Datasets go into `data/` (`FLASHREFOCUS_DATA`) and
evaluation outputs into `work/` (`FLASHREFOCUS_WORK`); `paths.py` lists the expected folders.

## Refocusing

```python
import sys; sys.path.insert(0, "pipeline")
import fast_refocus as FR

s1 = FR.FastStageI()                       # Stage I with the refiner, omega = 1.25
s1("photo.png", "aif.png")                 # restore the all-in-focus image once
rd = FR.FastRenderer()                     # Stage II
rd.set_image("aif.png")                    # depth is estimated once and reused
rd.render(x, y, 0.5, "refocused.png")      # every edit: focal point (x, y), strength in [0, 1]
```

Image sides should be multiples of 16. Two focal planes and tilt-shift edit the depth map given to the renderer:

```bash
python pipeline/dual_focus.py aif.png xA yA xB yB 0.8 out/dual
python pipeline/tiltshift.py photo.png out/tilt --row 0.3 --strength 0.6
```

## Training

| Stage | Script | Data |
|---|---|---|
| Stage I | `stage1/train_stage1.sh` (pixel stage, then detail stage) | DPDD, RealBokeh (`stage1/configs/*.csv`), LFDOF, DIV2K |
| Decoder refiner | `stage1/train_refiner.sh` | same mixture, Stage I latents at ω = 1.25 |
| Renderer | `renderer/train_renderer.sh` (Hypersim pre-training, RealBokeh adaptation) | Hypersim with BokehMe targets, RealBokeh |

## Evaluation

| Result | Scripts |
|---|---|
| Defocus deblurring (DPDD, RealDOF) | `stage1/deblur.py --pairs stage1/pairs/<set>_test.json`, `eval/score_folders.py` |
| Bokeh rendering (EBB! Val176, 3CReal, RealBokeh val.) | `eval/rbval_ours.py`, `renderer/test_focal_depth_k_search_p.py`, `eval/eval_bokeh_table.py`, `eval/score_folders.py` |
| Strength control (LVCorr) | `eval/lvcorr.py` |
| Focus-sweep consistency | `eval/rack_ours.py`, `eval/rack_genrefocus.py`, `eval/rack_match_blur*.py`, `eval/rack_analyze.py`, `eval/rack_metrics_kmatch.py` |
| MODEST aperture ladders | `eval/build_modest_ladder.py`, `eval/modest_analyze.py` |
| Cost | `stage1/cost_stage1.py`, `pipeline/bench_fast_refocus.py` |

Outputs of compared methods are produced with their own code (under `baselines/`, `FLASHREFOCUS_BASELINES`) and
scored with the same scripts.

## Acknowledgements

The code builds on [PiSA-SR](https://github.com/csslc/PiSA-SR), [Bokehlicious](https://github.com/TimSeizinger/Bokehlicious),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[RAM++](https://github.com/xinyu1205/recognize-anything); their licenses apply to the corresponding parts.
