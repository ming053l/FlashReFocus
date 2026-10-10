"""Renderer adaptation to real RealBokeh pairs, starting from the Hypersim pre-training (--pretrained), with a separate
learning rate for the defocus-conditioning modules (--depth-defocus-lr) and the backbone (--backbone-lr).
See train_renderer.sh for the configuration of the released renderer."""
import os
import sys
import types

import torch

from dataset.loader_focal_depth import RealBokehFocalDepth
from dataset.util import Mode
from method import config_dma, model_dma_p


def _arg_value(name: str, default: str | None = None):
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


def _pop_arg_value(name: str, default: str | None = None):
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    value = default
    if idx + 1 < len(sys.argv):
        value = sys.argv[idx + 1]
        del sys.argv[idx : idx + 2]
    else:
        del sys.argv[idx]
    return value


def _mode_value(mode):
    return mode if isinstance(mode, Mode) else Mode(mode)


crop_size = int(_arg_value("--crop-size", "0") or 0)
if crop_size <= 0 and "-h" not in sys.argv and "--help" not in sys.argv:
    raise ValueError(
        "train_realbokeh_dma_p.py uses loader-side augmentation, so "
        "--crop-size must be > 0 to define the random crop size."
    )
os.environ["BOKEHLICIOUS_DMA_IMAGE_SIZE"] = str(crop_size)

DEPTH_MAP_DROPOUT_P = float(_pop_arg_value("--depth-map-dropout-p", "0.0") or 0.0)
if not 0.0 <= DEPTH_MAP_DROPOUT_P <= 1.0:
    raise ValueError("--depth-map-dropout-p must be in [0, 1].")

if (
    "--save-dir" not in sys.argv
    and "-h" not in sys.argv
    and "--help" not in sys.argv
):
    sys.argv.extend(["--save-dir", "checkpoints/train_realbokeh_dma_p"])


def _maybe_dropout_depth_map(depth_map: torch.Tensor | None, probability: float):
    if depth_map is None or probability <= 0.0:
        return depth_map
    batch_size = depth_map.size(0)
    if probability >= 1.0:
        drop_mask = torch.ones(batch_size, device=depth_map.device, dtype=torch.bool)
    else:
        drop_mask = torch.rand(batch_size, device=depth_map.device) < probability
    if not bool(drop_mask.any()):
        return depth_map
    drop_mask = drop_mask.view(batch_size, *([1] * (depth_map.dim() - 1)))
    return torch.where(drop_mask, torch.zeros_like(depth_map), depth_map)


class Bokehlicious(model_dma_p.Bokehlicious):
    def forward(
        self,
        source,
        bokeh_strength=None,
        pos_map=None,
        focus_map=None,
        depth_map=None,
        disp_focus=None,
        **kwargs,
    ):
        kwargs.pop("bokeh_strength_map", None)
        if focus_map is None:
            focus_map = pos_map
        if self.training:
            depth_map = _maybe_dropout_depth_map(depth_map, DEPTH_MAP_DROPOUT_P)
        return super().forward(
            source=source,
            bokeh_strength=bokeh_strength,
            focus_map=focus_map,
            depth_map=depth_map,
            disp_focus=disp_focus,
            **kwargs,
        )


BokehliciousDMA = Bokehlicious


class RealBokeh(RealBokehFocalDepth):
    """Adapter matching train.py's expected dataset.loader.RealBokeh API."""

    def __init__(
        self,
        data_path,
        mode,
        binary_bokeh: bool = False,
        defocus_deblur_mode: bool = False,
        device: str = "cuda",
        challenge: bool = False,
        include_source: bool = False,
        **kwargs,
    ):
        if defocus_deblur_mode:
            raise ValueError(
                "train_realbokeh_dma_p.py does not support --defocus-deblur."
            )
        if challenge:
            raise ValueError("train_realbokeh_dma_p.py does not support challenge.")
        if binary_bokeh:
            print("--binary-bokeh is ignored by RealBokehFocalDepth.")
        if include_source:
            print("--include-source is ignored by RealBokehFocalDepth.")

        mode = _mode_value(mode)
        image_size = int(os.environ.get("BOKEHLICIOUS_DMA_IMAGE_SIZE", "0") or 0)
        super().__init__(
            data_path=data_path,
            mode=mode,
            device=device,
            use_augmentation=mode == Mode.TRAIN,
            image_size=image_size if mode == Mode.TRAIN else None,
            **kwargs,
        )

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        sample["pos_map"] = sample["focus_map"]
        return sample


loader_realbokeh_focal = types.ModuleType("dataset.loader")
loader_realbokeh_focal.RealBokeh = RealBokeh

model_dma_p_focus = types.ModuleType("method.model")
model_dma_p_focus.Bokehlicious = Bokehlicious
model_dma_p_focus.BokehliciousDMA = BokehliciousDMA

sys.modules["dataset.loader"] = loader_realbokeh_focal
sys.modules["method.config"] = config_dma
sys.modules["method.model"] = model_dma_p_focus

import train  # noqa: E402


def _disable_train_script_crop(
    inputs,
    targets,
    pos_map,
    bokeh_strength_map,
    crop_size,
    depth_map=None,
):
    return inputs, targets, pos_map, bokeh_strength_map, depth_map


if __name__ == "__main__":
    train.random_crop_train_tensors = _disable_train_script_crop
    print(f"Using depth map dropout p={DEPTH_MAP_DROPOUT_P}")
    train.main()
