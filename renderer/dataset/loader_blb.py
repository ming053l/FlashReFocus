import json
import os
import re
from pathlib import Path
from typing import Optional, Union

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from dataset.util import get_pos_map

_BOKEH_RE = re.compile(r"^bokeh_(\d+)_(\d+)\.(exr|jpg|jpeg|png)$", re.IGNORECASE)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_exr_rgb(path: Path, gamma: float = 2.2) -> torch.Tensor:
    image = cv2.imread(str(path), -1)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = image[..., :3].astype(np.float32)
    image = np.maximum(image, 0.0) ** (1.0 / gamma)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def _read_standard_rgb(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def _read_rgb(path: Path, gamma: float = 2.2) -> torch.Tensor:
    if path.suffix.lower() == ".exr":
        return _read_exr_rgb(path, gamma=gamma)
    return _read_standard_rgb(path)


def _normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    tensor_min = tensor.amin()
    tensor_max = tensor.amax()
    return ((tensor - tensor_min) / (tensor_max - tensor_min + eps)).clamp(0.0, 1.0)


def _normalize_tensor_with_value(
    tensor: torch.Tensor,
    value: float,
    percentiles: tuple[float, float] = (0.0, 100.0),
    valid_min: Optional[float] = None,
    valid_max: Optional[float] = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, float]:
    lo_p, hi_p = percentiles
    if not (0.0 <= lo_p < hi_p <= 100.0):
        raise ValueError(f"Invalid percentiles: {percentiles}")
    valid = torch.isfinite(tensor)
    if valid_min is not None:
        valid = valid & (tensor >= float(valid_min))
    if valid_max is not None:
        valid = valid & (tensor <= float(valid_max))
    valid_values = tensor[valid]
    if valid_values.numel() == 0:
        raise ValueError(
            f"No valid values left after applying valid_min={valid_min}, "
            f"valid_max={valid_max}"
        )
    if lo_p == 0.0 and hi_p == 100.0:
        lo = valid_values.amin()
        hi = valid_values.amax()
    else:
        flat = valid_values.flatten()
        q = torch.tensor([lo_p / 100.0, hi_p / 100.0], device=flat.device)
        lo, hi = torch.quantile(flat, q)
    denom = hi - lo + eps
    normalized = ((tensor.clamp(float(lo), float(hi)) - lo) / denom).clamp(0.0, 1.0)
    normalized_value = ((torch.tensor(float(value)) - lo) / denom).clamp(0.0, 1.0)
    return normalized, float(normalized_value.item())


def _read_depth(path: Path) -> torch.Tensor:
    depth = cv2.imread(str(path), -1)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth map: {path}")
    depth = depth[..., 0].astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(depth[None])).float()


def _read_depth_as_disparity(path: Path, eps: float = 1e-6) -> torch.Tensor:
    depth = _read_depth(path).numpy()[0]
    disp = 1.0 / np.maximum(depth, eps)
    return torch.from_numpy(np.ascontiguousarray(disp[None])).float()


def _parse_bokeh_name(path: Path) -> tuple[int, int]:
    match = _BOKEH_RE.match(path.name)
    if match is None:
        raise ValueError(f"Invalid BLB bokeh filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def _resize_tensor(
    tensor: torch.Tensor,
    size: tuple[int, int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    return TF.resize(tensor, list(size), interpolation=interpolation, antialias=True)


def _resize_longest_side(
    tensor: torch.Tensor,
    max_dim: Optional[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    if max_dim is None or max_dim <= 0:
        return tensor
    height, width = tensor.shape[-2:]
    longest = max(height, width)
    if longest <= max_dim:
        return tensor
    scale = float(max_dim) / float(longest)
    new_h = max(4, int(round(height * scale)))
    new_w = max(4, int(round(width * scale)))
    new_h -= new_h % 4
    new_w -= new_w % 4
    return _resize_tensor(tensor, (new_h, new_w), interpolation=interpolation)


def _make_pos_map(
    height: int, width: int, device: Union[str, torch.device]
) -> torch.Tensor:
    pos_x, pos_y = get_pos_map(width, height)
    return torch.cat((pos_x.unsqueeze(0), pos_y.unsqueeze(0)), dim=0).to(device)


class BLBBokeh(Dataset):
    """
    BLB dataset loader for Bokehlicious/DMA-style evaluation.

    Each sample pairs image.exr with one bokeh_<blur_idx>_<focus_idx>.exr target.
    The default model inputs match the user's Hypersim training convention:
        depth_map: 1 - normalized disparity
        disp_focus: 1 - normalized focal disparity (legacy model argument name)
        bokeh_strength: clamp(K / k_normalizer, 0, 1)
        bokeh_strength_map: abs(depth_map - disp_focus)

    The raw BLB signed defocus map K * (disp - disp_focus) is also returned as
    signed_defocus_map for analysis, but it is not used as the default model input.
    """

    def __init__(
        self,
        root: Union[str, Path] = "/ssd4/chingheng/BLB",
        split: str = "data",
        device: Union[str, torch.device] = "cuda",
        target_ext: str = "exr",
        scene_ids: Optional[list[Union[str, int]]] = None,
        blur_indices: Optional[list[int]] = None,
        focus_indices: Optional[list[int]] = None,
        k_normalizer: float = 80.0,
        gamma: float = 2.2,
        max_dim: Optional[int] = None,
        clip_source: bool = False,
        defocus_map_mode: str = "normalized_abs_depth",
        k_values: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0),
        signed_defocus_k_source: str = "level",
        depth_clip_percentiles: tuple[float, float] = (0.0, 100.0),
        disparity_clip_percentiles: tuple[float, float] = (0.0, 100.0),
        depth_valid_min: Optional[float] = None,
        depth_valid_max: Optional[float] = None,
        disparity_valid_min: Optional[float] = None,
        disparity_valid_max: Optional[float] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.data_dir = self.root / split
        self._device = device
        self.target_ext = target_ext.lower().lstrip(".")
        self.k_normalizer = float(k_normalizer)
        self.gamma = float(gamma)
        self.max_dim = max_dim
        self.clip_source = bool(clip_source)
        self.defocus_map_mode = defocus_map_mode
        self.k_values = tuple(float(v) for v in k_values)
        self.signed_defocus_k_source = str(signed_defocus_k_source)
        self.depth_clip_percentiles = depth_clip_percentiles
        self.disparity_clip_percentiles = disparity_clip_percentiles
        self.depth_valid_min = depth_valid_min
        self.depth_valid_max = depth_valid_max
        self.disparity_valid_min = disparity_valid_min
        self.disparity_valid_max = disparity_valid_max

        if self.k_normalizer <= 0:
            raise ValueError(f"k_normalizer must be > 0, got {self.k_normalizer}")
        if self.defocus_map_mode not in {
            "normalized_abs_depth",
            "normalized_abs_disp",
            "abs_disp",
            "abs_k_disp",
            "signed_k_disp",
        }:
            raise ValueError(
                "defocus_map_mode must be one of: normalized_abs_depth, "
                "normalized_abs_disp, abs_disp, abs_k_disp, signed_k_disp"
            )
        if self.signed_defocus_k_source not in {"level", "blur_parameter"}:
            raise ValueError(
                "signed_defocus_k_source must be 'level' or 'blur_parameter', got "
                f"{self.signed_defocus_k_source!r}"
            )
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"BLB data directory not found: {self.data_dir}")

        wanted_scenes = {str(scene_id) for scene_id in scene_ids} if scene_ids else None
        wanted_blurs = set(blur_indices) if blur_indices is not None else None
        wanted_focus = set(focus_indices) if focus_indices is not None else None

        self.items = []
        for scene_dir in sorted(
            path for path in self.data_dir.iterdir() if path.is_dir()
        ):
            if wanted_scenes is not None and scene_dir.name not in wanted_scenes:
                continue
            info_path = scene_dir / "info.json"
            source_path = scene_dir / "image.exr"
            depth_path = scene_dir / "depth.exr"
            if (
                not info_path.is_file()
                or not source_path.is_file()
                or not depth_path.is_file()
            ):
                continue
            info = _load_json(info_path)
            focus_distances = [float(v) for v in info["focus_distances"]]
            blur_parameters = [float(v) for v in info["blur_parameters"]]
            f_stops = [float(v) for v in info.get("f_stops", [])]

            for target_path in sorted(scene_dir.glob(f"bokeh_*.{self.target_ext}")):
                blur_idx, focus_idx = _parse_bokeh_name(target_path)
                if wanted_blurs is not None and blur_idx not in wanted_blurs:
                    continue
                if wanted_focus is not None and focus_idx not in wanted_focus:
                    continue
                if blur_idx >= len(blur_parameters) or focus_idx >= len(
                    focus_distances
                ):
                    continue
                self.items.append(
                    {
                        "scene_id": scene_dir.name,
                        "source_path": source_path,
                        "target_path": target_path,
                        "depth_path": depth_path,
                        "info_path": info_path,
                        "blur_idx": blur_idx,
                        "focus_idx": focus_idx,
                        "K": self.k_values[blur_idx],
                        "blur_parameter": blur_parameters[blur_idx],
                        "f_stop": f_stops[blur_idx] if blur_idx < len(f_stops) else 0.0,
                        "focus_distance": focus_distances[focus_idx],
                    }
                )

        if not self.items:
            raise FileNotFoundError(
                f"No BLB samples found under {self.data_dir} with target_ext={self.target_ext!r}"
            )
        print(
            f"BLBBokeh initialized with {len(self.items)} samples from {self.data_dir}"
        )

    def __len__(self):
        return len(self.items)

    def _embedding(self, k_value: float) -> float:
        return max(0.0, min(float(k_value) / self.k_normalizer, 1.0))

    def _defocus_map(
        self,
        normalized_depth: torch.Tensor,
        normalized_focus_depth: float,
        disparity: torch.Tensor,
        disp_focus: float,
        normalized_disparity: torch.Tensor,
        normalized_disp_focus: float,
        signed_defocus: torch.Tensor,
        k_value: float,
    ) -> torch.Tensor:
        if self.defocus_map_mode == "normalized_abs_depth":
            return (normalized_depth - float(normalized_focus_depth)).abs()
        if self.defocus_map_mode == "normalized_abs_disp":
            return (normalized_disparity - float(normalized_disp_focus)).abs()
        if self.defocus_map_mode == "abs_disp":
            return (disparity - float(disp_focus)).abs()
        if self.defocus_map_mode == "abs_k_disp":
            return signed_defocus.abs()
        return signed_defocus

    def __getitem__(self, index: int):
        item = self.items[index]
        source = _read_rgb(item["source_path"], gamma=self.gamma)
        target = _read_rgb(item["target_path"], gamma=self.gamma)
        depth = _read_depth(item["depth_path"])
        disparity = _read_depth_as_disparity(item["depth_path"])

        source = _resize_longest_side(source, self.max_dim)
        target = _resize_longest_side(target, self.max_dim)
        depth = _resize_longest_side(depth, self.max_dim)
        disparity = _resize_longest_side(disparity, self.max_dim)

        if self.clip_source:
            source = source.clamp(0.0, 1.0)

        k_value = float(item["K"])
        focus_depth = float(item["focus_distance"])
        disp_focus = 1.0 / max(float(item["focus_distance"]), 1e-6)
        normalized_depth, normalized_focus_depth = _normalize_tensor_with_value(
            depth,
            focus_depth,
            percentiles=self.depth_clip_percentiles,
            valid_min=self.depth_valid_min,
            valid_max=self.depth_valid_max,
        )
        normalized_disparity, normalized_disp_focus = _normalize_tensor_with_value(
            disparity,
            disp_focus,
            percentiles=self.disparity_clip_percentiles,
            valid_min=self.disparity_valid_min,
            valid_max=self.disparity_valid_max,
        )
        input_depth = 1.0 - normalized_disparity
        input_focus_depth = 1.0 - normalized_disp_focus
        signed_defocus_k = (
            float(item["blur_parameter"])
            if self.signed_defocus_k_source == "blur_parameter"
            else k_value
        )
        signed_defocus = signed_defocus_k * (disparity - disp_focus)
        bokeh_strength_map = self._defocus_map(
            normalized_depth=input_depth,
            normalized_focus_depth=input_focus_depth,
            disparity=disparity,
            disp_focus=disp_focus,
            normalized_disparity=normalized_disparity,
            normalized_disp_focus=normalized_disp_focus,
            signed_defocus=signed_defocus,
            k_value=k_value,
        )

        _, height, width = source.shape
        image_name = f"{item['scene_id']}_{item['target_path'].stem}"
        sample = {
            "source": source.to(self._device),
            "target": target.to(self._device),
            "bokeh_strength": torch.tensor(
                self._embedding(k_value), dtype=torch.float32, device=self._device
            ),
            "pos_map": _make_pos_map(height, width, self._device),
            "bokeh_strength_map": bokeh_strength_map.to(self._device),
            "pseudo_defocus_map": bokeh_strength_map.to(self._device),
            "depth_map": input_depth.to(self._device),
            "disp_focus": torch.tensor(
                input_focus_depth, dtype=torch.float32, device=self._device
            ),
            "raw_depth": depth.to(self._device),
            "raw_focus_depth": torch.tensor(
                focus_depth, dtype=torch.float32, device=self._device
            ),
            "normalized_disparity": normalized_disparity.to(self._device),
            "normalized_disp_focus": torch.tensor(
                normalized_disp_focus, dtype=torch.float32, device=self._device
            ),
            "raw_disparity": disparity.to(self._device),
            "raw_disp_focus": torch.tensor(
                disp_focus, dtype=torch.float32, device=self._device
            ),
            "signed_defocus_map": signed_defocus.to(self._device),
            "K": torch.tensor(k_value, dtype=torch.float32),
            "blur_parameter": torch.tensor(
                float(item["blur_parameter"]), dtype=torch.float32
            ),
            "signed_defocus_k": torch.tensor(signed_defocus_k, dtype=torch.float32),
            "f_stop": torch.tensor(float(item["f_stop"]), dtype=torch.float32),
            "focus_distance": torch.tensor(
                float(item["focus_distance"]), dtype=torch.float32
            ),
            "blur_idx": torch.tensor(int(item["blur_idx"]), dtype=torch.int64),
            "focus_idx": torch.tensor(int(item["focus_idx"]), dtype=torch.int64),
            "scene_id": item["scene_id"],
            "image_name": image_name,
            "source_path": str(item["source_path"]),
            "target_path": str(item["target_path"]),
            "depth_path": str(item["depth_path"]),
        }
        return sample


RealBokeh = BLBBokeh
