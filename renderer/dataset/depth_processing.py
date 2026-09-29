from pathlib import Path
from typing import Union

import torch

from dataset.loader_dma import load_relative_map


DEPTH_MODES = {
    "auto",
    "metric_scaled",
    "metric_minmax",
    "metric_inverse",
    "disparity",
    "raw",
}


def normalize_minmax(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    min_value = tensor.amin(dim=(-2, -1), keepdim=True)
    max_value = tensor.amax(dim=(-2, -1), keepdim=True)
    denom = (max_value - min_value).clamp_min(eps)
    return ((tensor - min_value) / denom).clamp(0.0, 1.0)


def to_inverse_minmax(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inverse = 1.0 / tensor.clamp_min(eps)
    return normalize_minmax(inverse, eps=eps)


def infer_depth_mode(
    depth_mode: str,
    depth_map_dir: Union[str, Path],
    normalize_depth_map: bool = False,
    inverse_depth: bool = False,
) -> str:
    """
    Resolve the effective depth preprocessing mode.

    Modes:
    - metric_scaled: metric-like depth divided by depth_normalize_scale.
    - metric_minmax: metric-like depth percentile-clipped then min-max normalized.
    - metric_inverse: metric-like depth converted to inverse depth then min-max normalized.
    - disparity: disparity/relative inverse-depth-like map, percentile min-max normalized.
      If depth_map_dir is da2_depth, DA2 relative depth is converted by
      1 - min-max(depth) to match disparity direction.
    - raw: percentile-clipped raw tensor, no scale or min-max normalization.
    - auto: choose from depth_map_dir.

    normalize_depth_map/inverse_depth are legacy switches kept for old callers.
    """
    depth_mode = str(depth_mode).lower()
    if depth_mode not in DEPTH_MODES:
        raise ValueError(
            f"depth_mode must be one of {sorted(DEPTH_MODES)}, got {depth_mode!r}"
        )

    if depth_mode != "auto":
        return depth_mode

    if inverse_depth:
        return "metric_inverse"
    if normalize_depth_map:
        return "metric_minmax"

    dir_name = Path(depth_map_dir).name.lower()
    if "da2" in dir_name or "midas" in dir_name or "disparity" in dir_name:
        return "disparity"
    if "norm" in dir_name:
        return "metric_minmax"
    return "metric_scaled"


def load_depth_map(
    depth_path: Union[str, Path],
    size: tuple[int, int],
    depth_mode: str,
    depth_map_dir: Union[str, Path],
    clip_percentiles: tuple[float, float] = (0.0, 100.0),
    depth_normalize_scale: float = 20.0,
    normalize_depth_map: bool = False,
    inverse_depth: bool = False,
) -> torch.Tensor:
    effective_mode = infer_depth_mode(
        depth_mode=depth_mode,
        depth_map_dir=depth_map_dir,
        normalize_depth_map=normalize_depth_map,
        inverse_depth=inverse_depth,
    )
    dir_name = Path(depth_map_dir).name.lower()
    is_da2_depth = "da2" in dir_name
    normalize_after_clip = effective_mode in {
        "metric_minmax",
        "disparity",
    }
    depth_map = load_relative_map(
        depth_path,
        size,
        clip_percentiles=clip_percentiles,
        normalize_after_clip=normalize_after_clip,
        ensure_unit_range=False,
    )

    if is_da2_depth and effective_mode == "disparity":
        return 1.0 - depth_map
    if effective_mode == "metric_scaled":
        if depth_normalize_scale <= 0:
            raise ValueError(
                "depth_normalize_scale must be > 0 for metric_scaled mode, got "
                f"{depth_normalize_scale}"
            )
        return depth_map / float(depth_normalize_scale)
    if effective_mode in {"metric_minmax", "disparity", "raw"}:
        return depth_map
    if effective_mode == "metric_inverse":
        return to_inverse_minmax(depth_map)

    raise ValueError(f"Unsupported effective depth mode: {effective_mode}")
