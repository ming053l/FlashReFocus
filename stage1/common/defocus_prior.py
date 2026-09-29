from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def _ensure_4d(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    if tensor.ndim == 4:
        return tensor, 4
    if tensor.ndim == 3:
        return tensor.unsqueeze(0), 3
    raise ValueError(f"Expected [C,H,W] or [B,C,H,W] tensor, got {tuple(tensor.shape)}")


def _restore_ndim(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    if ndim == 3:
        return tensor.squeeze(0)
    return tensor


def _sanitize_map(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return tensor.clamp(0.0, 1.0)


def _clean_raw_map(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _resize_if_needed(
    tensor: torch.Tensor, target_size: Optional[Tuple[int, int]], is_mask: bool = False
) -> torch.Tensor:
    if target_size is None or tuple(tensor.shape[-2:]) == tuple(target_size):
        return tensor
    mode = "nearest" if is_mask else "bilinear"
    kwargs = {} if is_mask else {"align_corners": False}
    return F.interpolate(tensor, size=target_size, mode=mode, **kwargs)


def _normalize_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    batch = tensor.shape[0]
    flat = tensor.view(batch, -1)
    mins = flat.min(dim=1).values.view(batch, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(batch, 1, 1, 1)
    return ((tensor - mins) / (maxs - mins + 1e-8)).clamp(0.0, 1.0)


def parse_band_thresholds(thresholds: Sequence[float] | float | None) -> Tuple[float, float]:
    if thresholds is None:
        return (0.33, 0.66)
    if isinstance(thresholds, (int, float)):
        value = float(thresholds)
        return (value, value)
    values = [float(v) for v in thresholds]
    if len(values) != 2:
        raise ValueError(f"`defocus_band_thresholds` expects 2 values, got {values}")
    low, high = values
    if low > high:
        low, high = high, low
    return (max(0.0, min(1.0, low)), max(0.0, min(1.0, high)))


def build_strength_prior(
    defocus_map: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    tensor, ndim = _ensure_4d(_sanitize_map(defocus_map))
    tensor = _resize_if_needed(tensor, target_size, is_mask=False)
    tensor = gaussian_blur(tensor, blur_kernel_size, blur_sigma)
    tensor = _normalize_per_sample(tensor)
    return _restore_ndim(tensor, ndim)


def build_rank_prior(
    defocus_map: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> torch.Tensor:
    tensor, ndim = _ensure_4d(_sanitize_map(defocus_map))
    tensor = _resize_if_needed(tensor, target_size, is_mask=False)
    flat = tensor.view(tensor.shape[0], -1)
    q_low = torch.quantile(flat, lower_quantile, dim=1).view(-1, 1, 1, 1)
    q_high = torch.quantile(flat, upper_quantile, dim=1).view(-1, 1, 1, 1)
    tensor = ((tensor - q_low) / (q_high - q_low + 1e-8)).clamp(0.0, 1.0)
    return _restore_ndim(tensor, ndim)


def build_depth_rank_prior(
    depth_map: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    tensor, ndim = _ensure_4d(_sanitize_map(depth_map))
    tensor = _resize_if_needed(tensor, target_size, is_mask=False)
    flat = tensor.view(tensor.shape[0], -1)
    median = torch.quantile(flat, 0.5, dim=1).view(-1, 1, 1, 1)
    tensor = (tensor - median).abs()
    tensor = _normalize_per_sample(tensor)
    return _restore_ndim(tensor, ndim)


def build_band_masks(
    prior_base: torch.Tensor,
    thresholds: Sequence[float] | float | None = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    low_thresh, high_thresh = parse_band_thresholds(thresholds)
    tensor, ndim = _ensure_4d(_sanitize_map(prior_base))
    tensor = _resize_if_needed(tensor, target_size, is_mask=False)
    low = (tensor < low_thresh).float()
    mid = ((tensor >= low_thresh) & (tensor < high_thresh)).float()
    high = (tensor >= high_thresh).float()
    band = torch.cat([low, mid, high], dim=1)
    return _restore_ndim(band, ndim)


def build_prior_bundle(
    defocus_map: torch.Tensor,
    mode: str = "raw",
    thresholds: Sequence[float] | float | None = None,
    depth_map: Optional[torch.Tensor] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    raw = _clean_raw_map(defocus_map)
    sanitized = _sanitize_map(raw)
    strength = build_strength_prior(sanitized, target_size=target_size)
    rank = build_rank_prior(sanitized, target_size=target_size)
    band_masks = build_band_masks(
        strength,
        thresholds=thresholds,
        target_size=target_size,
    )
    high_band = band_masks[..., 2:3, :, :] if band_masks.ndim == 4 else band_masks[2:3]

    depth_rank = None
    if depth_map is not None:
        depth_rank = build_depth_rank_prior(depth_map, target_size=target_size)

    if mode == "raw":
        selected = _resize_if_needed(_ensure_4d(raw)[0], target_size, is_mask=False)
        selected = _restore_ndim(selected, raw.ndim)
    elif mode == "strength":
        selected = strength
    elif mode == "rank":
        selected = rank
    elif mode == "band":
        selected = high_band
    elif mode == "depth_rank":
        if depth_rank is None:
            raise KeyError("`depth_rank` prior mode requires a depth map for this sample.")
        selected = depth_rank
    else:
        raise ValueError(f"Unsupported defocus prior mode: {mode}")

    bundle: Dict[str, torch.Tensor] = {
        "raw_map": selected if mode == "raw" else raw,
        "strength_map": strength,
        "rank_map": rank,
        "band_masks": band_masks,
        "high_band_mask": high_band,
        "selected_prior": selected if mode == "raw" else _sanitize_map(selected),
    }
    if depth_rank is not None:
        bundle["depth_rank_map"] = depth_rank
    return bundle


def apply_prior_ablation(prior_map: torch.Tensor, mode: str = "none") -> torch.Tensor:
    if mode in {None, "none", "true"}:
        return prior_map
    tensor, ndim = _ensure_4d(_sanitize_map(prior_map))
    if mode == "zero":
        return _restore_ndim(torch.zeros_like(tensor), ndim)
    if mode == "shuffle":
        shuffled = tensor.clone()
        batch = shuffled.shape[0]
        for batch_idx in range(batch):
            perm = torch.randperm(
                shuffled.shape[-2] * shuffled.shape[-1], device=shuffled.device
            )
            flat = shuffled[batch_idx].reshape(shuffled.shape[1], -1)
            flat = flat[:, perm]
            shuffled[batch_idx] = flat.reshape_as(shuffled[batch_idx])
        return _restore_ndim(shuffled, ndim)
    raise ValueError(f"Unsupported defocus prior ablation mode: {mode}")


def build_band_visualization(band_masks: torch.Tensor) -> torch.Tensor:
    tensor, ndim = _ensure_4d(band_masks.float().clamp(0.0, 1.0))
    if tensor.shape[1] != 3:
        raise ValueError(
            f"Band visualization expects 3 channels [low, mid, high], got {tensor.shape[1]}"
        )
    rgb = torch.stack(
        [tensor[:, 2], tensor[:, 1], tensor[:, 0]],
        dim=1,
    )
    return _restore_ndim(rgb.clamp(0.0, 1.0), ndim)


def average_non_nan(values: Sequence[float]) -> float:
    valid = [float(v) for v in values if v == v]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def summarize_metrics(metric_dicts: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    keys = set().union(*(metric.keys() for metric in metric_dicts))
    return {key: average_non_nan([metric.get(key, float("nan")) for metric in metric_dicts]) for key in sorted(keys)}
