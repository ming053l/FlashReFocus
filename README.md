# FlashReFocus: Interactive Image Refocusing in Seconds

FlashReFocus changes the focal plane and depth of field of a single photograph after capture. Diffusion is applied only where information is lost: a one-step latent diffusion model restores an all-in-focus (AiF) image **once**, and a 1.3M-parameter renderer then produces **every** focal setting from a depth-derived defocus map. A complete refocus takes about 2.5 s at 1536×1536, and each further edit about half a second.

- **Stage I – one-step deblurring.** PiSA-SR on Stable Diffusion 2.1 with a pixel LoRA (fidelity) and a detail LoRA (weighted by ω at inference), trained on mixed real and synthetic defocus pairs, plus an optional input-guided decoder refiner that restores text and thin edges lost in VAE decoding.
- **Stage II – controllable rendering.** A NAF-style encoder–decoder with defocus-bias attention, conditioned on Depth Anything V2 depth, a focal point, and a blur strength; pre-trained on synthetic multi-focal Hypersim scenes and adapted to RealBokeh pairs.
- **Controls.** Focal point, blur strength, two focal planes and tilt-shift by depth editing, all from the same restored image.

Project page: [`docs/`](docs/index.html) (can be served with GitHub Pages).

## Repository layout

```
stage1/     Stage I training and inference (PiSA-SR based), refiner fine-tuning, cost measurement
  common/   dataset, options, losses and the RAM++ tagger used for CSD prompts
  configs/  training data configurations
  pairs/    held-out / test pairs used for model selection and Table 1
refiner/    input-guided decoder refiner (VAEMultiScaleFusionDecoder)
renderer/   Stage II renderer (built on Bokehlicious): model, data, training, inference
pipeline/   end-to-end refocusing (fast path), two focal planes, tilt-shift, benchmarks
eval/       bokeh / refocusing evaluation: SSIM-searched strength protocol, LVCorr,
            focus-sweep consistency, MODEST aperture ladders, metric scripts
docs/       project page
third_party/ PiSA-SR and Depth-Anything-V2 (git submodules)
```

## Setup

```bash
git clone --recursive git@github.com:ming053l/FlashReFocus.git
conda create -n flashrefocus python=3.10 && conda activate flashrefocus
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install diffusers==0.25.0 transformers==4.46.0 accelerate==1.13.0 peft==0.9.0 lpips torchmetrics pyiqa opencv-python scipy tensorboard
pip install -r renderer/requirements.txt
```

Model weights required:
- Stable Diffusion 2.1 base (Stage I) and 1.5 VAE (refiner), from Hugging Face.
- PiSA-SR released LoRAs (initialization), see `third_party/PiSA-SR`.
- Depth Anything V2 ViT-L, see `third_party/Depth-Anything-V2`.
- FlashReFocus checkpoints (Stage I, refiner, renderer): *to be released*.

> **Note.** The scripts still contain absolute paths of our training machines (e.g. `/ssd1/...`) as defaults. Pass the corresponding command-line arguments or edit the defaults before running.

## Usage

**Deblur (Stage I, ω = 1.25, with refiner):**
```bash
python stage1/run_pisa_refiner.py --pairs <pairs.json> --data_root <root> \
  --ckpt <stage1.pkl> --refiner <refiner.pt> --lambda_sem 1.25 --out <out_dir>
```

**Refocus (restore once, render any focal point):**
```python
import fast_refocus as FR
FR.setup_backend(True)
s1 = FR.FastStageI("<stage1.pkl>", "<refiner.pt>", lambda_sem=1.25)
s1("input.png", "aif.png")                      # once per image
rd = FR.FastRenderer()                          # bf16 autocast by default
rd.set_image("aif.png")                         # depth is estimated once and cached
rd.render(focus_x, focus_y, strength, "refocused.png")   # per edit, strength in [0, 1]
```

**Two focal planes:** `python pipeline/dual_focus.py aif.png xA yA xB yB 1.5 out/prefix`

## Training

- Stage I: `stage1/train_full_v3.sh` (pixel phase, then detail phase), refiner: `stage1/train_refiner_pisa.py`.
- Renderer: `renderer/train_hypersim_dma_p.py` (synthetic pre-training) and `renderer/train_A5.sh` (RealBokeh adaptation with a differential learning rate).

## Evaluation

- Deblurring: `stage1/eval_ckpt.py`, held-out selection by the rank sum of PSNR, LPIPS, DISTS, and FID.
- Bokeh rendering: `eval/rbval_ours.py` (per-image SSIM-searched strength), `eval/eval_bokeh_table.py`, `eval/score_folders.py`, `eval/lvcorr.py`.
- Focus-sweep consistency: `eval/rack_*.py`; MODEST aperture ladders: `eval/build_modest_ladder.py`, `eval/modest_analyze.py`.

## Acknowledgements

This code builds on [PiSA-SR](https://github.com/csslc/PiSA-SR), [Bokehlicious](https://github.com/TimSeizinger/Bokehlicious), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and [RAM++](https://github.com/xinyu1205/recognize-anything). Please also follow their licenses.

## Citation

```bibtex
@article{flashrefocus2026,
  title  = {FlashReFocus: Interactive Image Refocusing in Seconds},
  author = {TBD},
  year   = {2026}
}
```
