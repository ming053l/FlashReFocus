"""Locations of weights, datasets and outputs, shared by every script in this repository.

Defaults are folders inside the repository; set an environment variable to point elsewhere:

  FLASHREFOCUS_WEIGHTS  weights/   stable-diffusion-2-1-base/, stable-diffusion-v1-5/ (VAE only),
                                   stage1.pkl, refiner.pt, renderer.pt (FlashReFocus weights),
                                   depth_anything_v2_vitl.pth, ram_plus_swin_large_14m.pth (training only)
  FLASHREFOCUS_DATA     data/      DPDD (dd_dp_dataset_png/), RealDOF/, div2k/, 3CReal/, RealBokeh/, LFDOF/
  FLASHREFOCUS_WORK     work/      evaluation outputs and intermediate files
  FLASHREFOCUS_BASELINES baselines/ code of compared methods (GenRefocus, Depth Pro, ...), for the evaluation scripts
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("FLASHREFOCUS_WEIGHTS", REPO / "weights"))
DATA = Path(os.environ.get("FLASHREFOCUS_DATA", REPO / "data"))
WORK = Path(os.environ.get("FLASHREFOCUS_WORK", REPO / "work"))
BASELINES = Path(os.environ.get("FLASHREFOCUS_BASELINES", REPO / "baselines"))

SD21 = WEIGHTS / "stable-diffusion-2-1-base"
SD15 = WEIGHTS / "stable-diffusion-v1-5"
STAGE1 = WEIGHTS / "stage1.pkl"
REFINER = WEIGHTS / "refiner.pt"
RENDERER = WEIGHTS / "renderer.pt"
DA2 = WEIGHTS / "depth_anything_v2_vitl.pth"
RAM_PLUS = WEIGHTS / "ram_plus_swin_large_14m.pth"

PISA_SR = REPO / "third_party" / "PiSA-SR"
DA2_ROOT = REPO / "third_party" / "Depth-Anything-V2"
RAM_ROOT = REPO / "third_party" / "recognize-anything"   # RAM++ tagger (Stage I training prompts)


def add_to_path():
    """Make the repository's packages importable from any script: refiner (as a package), stage1 and stage1/common,
    renderer, pipeline, eval, and the submodules. recognize-anything precedes third_party/PiSA-SR, whose own older `ram`
    package must not shadow RAM++."""
    for p in (DA2_ROOT, PISA_SR, RAM_ROOT, REPO / "eval", REPO / "pipeline", REPO / "renderer", REPO / "stage1" / "common",
              REPO / "stage1", REPO):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
