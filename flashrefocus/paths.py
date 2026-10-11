"""Locations of weights, datasets and outputs. Defaults are folders inside the repository; set an environment variable
to point elsewhere:

  FLASHREFOCUS_WEIGHTS    weights/    stable-diffusion-2-1-base/, stable-diffusion-v1-5/ (VAE only),
                                      stage1.pkl, refiner.pt, renderer.pt (FlashReFocus weights),
                                      depth_anything_v2_vitl.pth, ram_plus_swin_large_14m.pth (Stage I training only)
  FLASHREFOCUS_DATA       data/       DPDD (dd_dp_dataset_png/), RealDOF/, div2k/, 3CReal/, RealBokeh/, LFDOF/
  FLASHREFOCUS_WORK       work/       evaluation outputs and intermediate files
  FLASHREFOCUS_BASELINES  baselines/  code of compared methods (GenRefocus, Depth Pro, ...), for the evaluation scripts
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("FLASHREFOCUS_WEIGHTS", REPO / "weights"))
DATA = Path(os.environ.get("FLASHREFOCUS_DATA", REPO / "data"))
WORK = Path(os.environ.get("FLASHREFOCUS_WORK", REPO / "work"))
BASELINES = Path(os.environ.get("FLASHREFOCUS_BASELINES", REPO / "baselines"))

SD21 = WEIGHTS / "stable-diffusion-2-1-base"
SD15 = WEIGHTS / "stable-diffusion-v1-5"
STAGE1 = WEIGHTS / "stage1.pkl"
REFINER = WEIGHTS / "refiner.pt"
RENDERER = WEIGHTS / "renderer.pt"
DEPTH_ANYTHING_V2 = WEIGHTS / "depth_anything_v2_vitl.pth"
RAM_PLUS = WEIGHTS / "ram_plus_swin_large_14m.pth"

PISA_SR = REPO / "third_party" / "PiSA-SR"
DEPTH_ANYTHING_V2_CODE = REPO / "third_party" / "Depth-Anything-V2"
RAM_PLUS_CODE = REPO / "third_party" / "recognize-anything"


def use_third_party(*roots):
    """Make third-party code importable (git submodules under third_party/). recognize-anything must precede
    PiSA-SR, whose older `ram` package would otherwise shadow RAM++."""
    for root in roots:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
