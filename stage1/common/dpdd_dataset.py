import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

from defocus_prior import build_prior_bundle
from train_utils import format_aperture_prompt

# datasets live under FLASHREFOCUS_DATA (default <repo>/data), see paths.py
DATA_ROOT = Path(os.environ.get("FLASHREFOCUS_DATA", Path(__file__).resolve().parents[2] / "data"))


def _normalize_image_sizes(
    image_size: Optional[int | Sequence[int] | Sequence[Sequence[int]]],
) -> Optional[list[tuple[int, int]]]:
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return [(int(image_size), int(image_size))]
    if isinstance(image_size, str):
        raise TypeError(
            "image_size must be an int, [height, width], or a list of [height, width] pairs."
        )
    values = list(image_size)
    if len(values) == 0:
        return None
    if len(values) == 2 and all(isinstance(value, int) for value in values):
        return [(int(values[0]), int(values[1]))]

    sizes = []
    for value in values:
        if isinstance(value, int):
            sizes.append((int(value), int(value)))
            continue
        pair = list(value)
        if len(pair) != 2:
            raise ValueError(
                "Each train resolution must be [height, width]; "
                f"got {value!r}."
            )
        height, width = int(pair[0]), int(pair[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid non-positive resolution: {(height, width)}")
        sizes.append((height, width))
    return sizes


def _list_image_files(folder: Path, exts: Sequence[str]) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _index_by_stem(paths: Sequence[Path], name: str) -> dict[str, Path]:
    indexed = {}
    duplicates = []
    for path in paths:
        if path.stem in indexed:
            duplicates.append(path.stem)
        indexed[path.stem] = path
    if duplicates:
        raise ValueError(
            f"Duplicate {name} file stem(s) found: {sorted(set(duplicates))[:5]}"
        )
    return indexed


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return TF.to_tensor(image)


def _resolve_manifest_path(manifest_root: Path, value: str) -> Path:
    """Relative manifest entries are resolved against the data root (FLASHREFOCUS_DATA, default <repo>/data) when
    they exist there, otherwise against the manifest's own folder."""
    path = Path(value)
    if not path.is_absolute():
        path = DATA_ROOT / path if (DATA_ROOT / path).exists() else manifest_root / path
    return path


def _load_gray_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    tensor = TF.to_tensor(image)
    min_val = tensor.amin()
    max_val = tensor.amax()
    if (max_val - min_val) < 1e-8:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)


def _load_gray_raw_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    return TF.to_tensor(image)


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.amin()
    max_val = tensor.amax()
    if (max_val - min_val) < 1e-8:
        return torch.zeros_like(tensor)
    return ((tensor - min_val) / (max_val - min_val)).clamp(0.0, 1.0)


def _load_depth_or_disp_tensor(path: Path, depth_mode: str) -> torch.Tensor:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    depth_or_disp = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
    if depth_mode == "metric_depth":
        depth = depth_or_disp / 1000.0
        disp = 1.0 / depth.clamp_min(1e-6)
    elif depth_mode == "normalized_depth":
        depth = _normalize_tensor(depth_or_disp)
        disp = 1.0 - depth
    elif depth_mode == "disparity":
        disp = depth_or_disp
    else:
        raise ValueError(f"Unsupported depth mode: {depth_mode}")
    return _normalize_tensor(disp)


def _build_hypersim_defocus_map(
    depth_path: Path,
    k_value: float,
    disp_focus: float,
    defocus_scale: float,
    depth_mode: str,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    disp = _load_depth_or_disp_tensor(depth_path, depth_mode)
    scale = float(defocus_scale) if normalize else 1.0
    signed_defocus = float(k_value) * (disp - float(disp_focus)) / scale
    defocus_strength = signed_defocus.abs()
    if normalize:
        defocus_strength = _normalize_tensor(defocus_strength)
    return defocus_strength, disp


def _normalize_task_mode(task_mode: str) -> str:
    task_mode = str(task_mode).lower()
    aliases = {
        "deblur": "defocus_deblurring",
        "defocus_deblur": "defocus_deblurring",
        "defocus_deblurring": "defocus_deblurring",
        "bokeh": "bokeh_rendering",
        "bokeh_render": "bokeh_rendering",
        "bokeh_rendering": "bokeh_rendering",
    }
    if task_mode not in aliases:
        raise ValueError(
            "Unsupported task_mode: "
            f"{task_mode!r}. Use 'defocus_deblurring' or 'bokeh_rendering'."
        )
    return aliases[task_mode]


def _disk_kernel(
    radius: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    radius = max(float(radius), 0.5)
    kernel_radius = max(1, int(round(radius)))
    coords = torch.arange(-kernel_radius, kernel_radius + 1, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = ((xx.square() + yy.square()) <= radius * radius).to(dtype)
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def apply_disk_blur(image: torch.Tensor, radius: float) -> torch.Tensor:
    """Apply a per-channel disk-like blur to a CHW image tensor in [0, 1]."""
    if radius <= 0:
        return image
    kernel = _disk_kernel(radius, dtype=image.dtype, device=image.device)
    kernel = kernel.expand(image.size(0), 1, -1, -1)
    padding = kernel.shape[-1] // 2
    image_b = image.unsqueeze(0)
    image_b = F.pad(image_b, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(image_b, kernel, groups=image.size(0)).squeeze(0)


def apply_gaussian_blur(image: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0:
        return image
    kernel_size = max(3, int(round(radius * 2.0)) * 2 + 1)
    sigma = max(float(radius) * 0.5, 0.1)
    return TF.gaussian_blur(
        image,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    )


def _motion_kernel(
    radius: float,
    angle_degrees: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    kernel_radius = max(1, int(round(radius)))
    size = kernel_radius * 2 + 1
    coords = torch.arange(size, dtype=dtype, device=device) - kernel_radius
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    theta = torch.tensor(angle_degrees * np.pi / 180.0, dtype=dtype, device=device)
    distance_along = xx * torch.cos(theta) + yy * torch.sin(theta)
    distance_cross = -xx * torch.sin(theta) + yy * torch.cos(theta)
    kernel = (
        (distance_along.abs() <= kernel_radius) & (distance_cross.abs() <= 0.5)
    ).to(dtype)
    if kernel.sum() < 1:
        kernel[kernel_radius, kernel_radius] = 1.0
    return kernel / kernel.sum().clamp_min(1e-8)


def apply_motion_blur(
    image: torch.Tensor,
    radius: float,
    angle_degrees: Optional[float] = None,
) -> torch.Tensor:
    if radius <= 0:
        return image
    if angle_degrees is None:
        angle_degrees = random.uniform(0.0, 180.0)
    kernel = _motion_kernel(
        radius,
        angle_degrees=angle_degrees,
        dtype=image.dtype,
        device=image.device,
    )
    kernel = kernel.expand(image.size(0), 1, -1, -1)
    padding = kernel.shape[-1] // 2
    image_b = image.unsqueeze(0)
    image_b = F.pad(image_b, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(image_b, kernel, groups=image.size(0)).squeeze(0)


def make_smooth_random_map(
    height: int,
    width: int,
    dtype: torch.dtype,
    low_res_size: int = 16,
) -> torch.Tensor:
    low_h = max(2, min(low_res_size, height))
    low_w = max(2, min(low_res_size, width))
    random_map = torch.rand((1, 1, low_h, low_w), dtype=dtype)
    blur_map = F.interpolate(
        random_map,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)
    return _normalize_tensor(blur_map)


def make_directional_gradient_map(
    height: int,
    width: int,
    dtype: torch.dtype,
    direction: str = "top_to_bottom",
) -> torch.Tensor:
    direction = str(direction).lower()
    if direction == "random":
        direction = random.choice(
            [
                "top_to_bottom",
                "bottom_to_top",
                "left_to_right",
                "right_to_left",
                "top_left_to_bottom_right",
                "top_right_to_bottom_left",
                "center_to_edge",
                "edge_to_center",
                "all",
            ]
        )

    if direction in {"all", "all_ones", "ones", "constant"}:
        return torch.ones((1, height, width), dtype=dtype)

    y = torch.linspace(0.0, 1.0, height, dtype=dtype).view(1, height, 1)
    x = torch.linspace(0.0, 1.0, width, dtype=dtype).view(1, 1, width)
    if direction in {"top_to_bottom", "upper_to_lower", "vertical"}:
        blur_map = y.expand(1, height, width)
    elif direction in {"bottom_to_top", "lower_to_upper"}:
        blur_map = 1.0 - y.expand(1, height, width)
    elif direction in {"left_to_right", "horizontal"}:
        blur_map = x.expand(1, height, width)
    elif direction == "right_to_left":
        blur_map = 1.0 - x.expand(1, height, width)
    elif direction in {"top_left_to_bottom_right", "tl_to_br"}:
        blur_map = 0.5 * (x.expand(1, height, width) + y.expand(1, height, width))
    elif direction in {"top_right_to_bottom_left", "tr_to_bl"}:
        blur_map = 0.5 * (
            (1.0 - x.expand(1, height, width)) + y.expand(1, height, width)
        )
    elif direction == "center_to_edge":
        yy = y.expand(1, height, width) - 0.5
        xx = x.expand(1, height, width) - 0.5
        blur_map = torch.sqrt(xx.square() + yy.square())
    elif direction == "edge_to_center":
        yy = y.expand(1, height, width) - 0.5
        xx = x.expand(1, height, width) - 0.5
        blur_map = 1.0 - torch.sqrt(xx.square() + yy.square())
    else:
        raise ValueError(f"Unsupported synthetic gradient direction: {direction}")
    return _normalize_tensor(blur_map)


def synthetic_defocus_blur(
    clear: torch.Tensor,
    radius_min: float = 1.0,
    radius_max: float = 8.0,
    spatially_varying: bool = True,
    blur_type: str = "disk",
    blur_map_mode: str = "random",
    gradient_direction: str = "top_to_bottom",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic defocus pair from a clear CHW image tensor in [0, 1]."""
    radius_min = max(0.0, float(radius_min))
    radius_max = max(radius_min, float(radius_max))
    radius = random.uniform(radius_min, radius_max)
    if blur_type in {"mixed", "both"}:
        blur_type = random.choice(["disk", "gaussian", "motion"])
    if blur_type == "gaussian":
        blur_fn = apply_gaussian_blur
    elif blur_type == "motion":
        blur_fn = apply_motion_blur
    else:
        blur_fn = apply_disk_blur

    if (not spatially_varying) or blur_map_mode in {"constant", "uniform", "patch"}:
        blur_map = torch.full(
            (1, *clear.shape[-2:]),
            radius / max(radius_max, 1e-8),
            dtype=clear.dtype,
        )
        return blur_fn(clear, radius).clamp(0.0, 1.0), blur_map.clamp(0.0, 1.0)

    height, width = clear.shape[-2:]
    if blur_map_mode in {"gradient", "linear", "directional"}:
        blur_map = make_directional_gradient_map(
            height,
            width,
            dtype=clear.dtype,
            direction=gradient_direction,
        )
    else:
        blur_map = make_smooth_random_map(height, width, dtype=clear.dtype)
    radii = [radius_min, 0.5 * (radius_min + radius_max), radius_max]
    blurred = torch.stack([blur_fn(clear, r) for r in radii], dim=0)
    centers = torch.tensor([0.0, 0.5, 1.0], dtype=clear.dtype).view(3, 1, 1, 1)
    weights = (1.0 - (blur_map.unsqueeze(0) - centers).abs() * 2.0).clamp_min(0.0)
    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
    synthetic = (weights * blurred).sum(dim=0)
    return synthetic.clamp(0.0, 1.0), blur_map.clamp(0.0, 1.0)


class DPDDDataset(Dataset):
    def __init__(
        self,
        root: str = str(DATA_ROOT / "dd_dp_dataset_png"),
        split: str = "train",
        prediction_root: Optional[str] = None,
        image_size: Optional[int] = None,
        use_augmentation: Optional[bool] = None,
        random_scale_range: Tuple[float, float] = (0.2, 1.0),
        random_hflip_p: float = 0.5,
        center_crop: bool = False,
        min_divisor: int = 8,
        defocus_band_thresholds: Tuple[float, float] = (0.33, 0.66),
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.root = Path(root)
        self.split = split
        self.prediction_root = (
            Path(prediction_root) if prediction_root is not None else None
        )
        self.image_size = image_size
        self.image_sizes = _normalize_image_sizes(image_size)
        self.use_augmentation = (
            split == "train" if use_augmentation is None else use_augmentation
        )
        self.random_scale_range = random_scale_range
        self.random_hflip_p = random_hflip_p
        self.center_crop = center_crop
        self.min_divisor = min_divisor
        self.defocus_band_thresholds = defocus_band_thresholds
        self.exts = tuple(ext.lower() for ext in exts)

        self.split_dir = self.root / f"{split}_c"
        self.source_dir = self.split_dir / "source"
        self.target_dir = self.split_dir / "target"
        self.map_dir = self.split_dir / "pseudo_defocus_map_from_points"
        self.depth_map_dir = self.split_dir / "depth_map"
        self.prediction_dir = (
            self.prediction_root / split / "predictions"
            if self.prediction_root is not None
            else None
        )

        for folder in (self.source_dir, self.target_dir):
            if not folder.exists():
                raise FileNotFoundError(f"Missing dataset folder: {folder}")

        self.source_paths = _list_image_files(self.source_dir, self.exts)
        target_candidates = _list_image_files(self.target_dir, self.exts)
        map_candidates = (
            _list_image_files(self.map_dir, self.exts) if self.map_dir.exists() else []
        )
        depth_map_candidates = (
            _list_image_files(self.depth_map_dir, self.exts)
            if self.depth_map_dir.exists()
            else []
        )
        self.target_paths = target_candidates
        self.map_by_stem = _index_by_stem(map_candidates, "pseudo defocus map")
        self.depth_map_by_stem = _index_by_stem(depth_map_candidates, "depth map")
        self.prediction_by_stem = {}
        if self.prediction_dir is not None:
            if not self.prediction_dir.exists():
                raise FileNotFoundError(
                    f"Missing prediction folder: {self.prediction_dir}"
                )
            prediction_candidates = _list_image_files(self.prediction_dir, self.exts)
            self.prediction_by_stem = _index_by_stem(
                prediction_candidates, "prediction"
            )
            missing_predictions = [
                path.stem
                for path in self.source_paths
                if path.stem not in self.prediction_by_stem
            ]
            if missing_predictions:
                raise FileNotFoundError(
                    f"Missing predictions for {len(missing_predictions)} files in {self.prediction_dir}. "
                    f"Examples: {missing_predictions[:5]}"
                )

        if len(self.source_paths) == 0:
            raise FileNotFoundError(f"No source images found in {self.source_dir}")
        if len(self.target_paths) == 0:
            raise FileNotFoundError(f"No target images found in {self.target_dir}")
        if len(self.target_paths) != len(self.source_paths):
            raise FileNotFoundError(
                "Source / target counts do not match, so target metrics cannot be aligned. "
                f"source={len(self.source_paths)}, target={len(self.target_paths)}. "
                "Defocus maps may be missing and will be one-filled, but every source still needs a target."
            )
        missing_maps = [
            path.stem for path in self.source_paths if path.stem not in self.map_by_stem
        ]
        if missing_maps:
            print(
                "Warning: Missing pseudo defocus maps for "
                f"{len(missing_maps)} source images in {self.map_dir}. "
                "Using one-filled defocus maps for those samples. "
                f"Examples: {missing_maps[:5]}"
            )

    def __len__(self) -> int:
        return len(self.source_paths)

    def _resize_tensor(
        self, tensor: torch.Tensor, size: Tuple[int, int], is_mask: bool
    ) -> torch.Tensor:
        interpolation = (
            InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
        )
        return TF.resize(
            tensor,
            list(size),
            interpolation=interpolation,
            antialias=not is_mask,
        )

    def _sample_image_size(self) -> Optional[tuple[int, int]]:
        image_sizes = getattr(self, "image_sizes", None)
        if image_sizes is None and hasattr(self, "image_size"):
            image_sizes = _normalize_image_sizes(self.image_size)
            self.image_sizes = image_sizes
        if image_sizes is None:
            return None
        return random.choice(image_sizes)

    def _joint_random_scale(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        defocus_map: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        image_size = self._sample_image_size()
        if image_size is None:
            return source, target, defocus_map, prediction, depth_map

        crop_h, crop_w = image_size
        scale = random.uniform(*self.random_scale_range)
        height, width = source.shape[-2:]
        scaled_h = max(crop_h, int(round(height * scale)))
        scaled_w = max(crop_w, int(round(width * scale)))
        size = (scaled_h, scaled_w)
        source = self._resize_tensor(source, size, is_mask=False)
        target = self._resize_tensor(target, size, is_mask=False)
        defocus_map = self._resize_tensor(defocus_map, size, is_mask=False)
        if prediction is not None:
            prediction = self._resize_tensor(prediction, size, is_mask=False)
        if depth_map is not None:
            depth_map = self._resize_tensor(depth_map, size, is_mask=False)
        return source, target, defocus_map, prediction, depth_map

    def _joint_random_crop(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        defocus_map: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        image_size = self._sample_image_size()
        if image_size is None:
            return source, target, defocus_map, prediction, depth_map

        crop_h, crop_w = image_size
        height, width = source.shape[-2:]
        if height < crop_h or width < crop_w:
            size = (max(crop_h, height), max(crop_w, width))
            source = self._resize_tensor(source, size, is_mask=False)
            target = self._resize_tensor(target, size, is_mask=False)
            defocus_map = self._resize_tensor(defocus_map, size, is_mask=False)
            if prediction is not None:
                prediction = self._resize_tensor(prediction, size, is_mask=False)
            if depth_map is not None:
                depth_map = self._resize_tensor(depth_map, size, is_mask=False)
            height, width = source.shape[-2:]

        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        source = TF.crop(source, top, left, crop_h, crop_w)
        target = TF.crop(target, top, left, crop_h, crop_w)
        defocus_map = TF.crop(defocus_map, top, left, crop_h, crop_w)
        if prediction is not None:
            prediction = TF.crop(
                prediction, top, left, crop_h, crop_w
            )
        if depth_map is not None:
            depth_map = TF.crop(depth_map, top, left, crop_h, crop_w)
        return source, target, defocus_map, prediction, depth_map

    def _joint_random_hflip(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        defocus_map: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        if random.random() >= self.random_hflip_p:
            return source, target, defocus_map, prediction, depth_map
        source = TF.hflip(source)
        target = TF.hflip(target)
        defocus_map = TF.hflip(defocus_map)
        if prediction is not None:
            prediction = TF.hflip(prediction)
        if depth_map is not None:
            depth_map = TF.hflip(depth_map)
        return source, target, defocus_map, prediction, depth_map

    def _maybe_resize(self, tensor: torch.Tensor, is_mask: bool) -> torch.Tensor:
        image_size = self._sample_image_size()
        if image_size is None:
            return tensor
        return self._resize_tensor(tensor, image_size, is_mask)

    def _maybe_center_crop(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        defocus_map: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        image_size = self._sample_image_size()
        if not self.center_crop or image_size is None:
            return source, target, defocus_map, prediction, depth_map

        source = TF.center_crop(source, list(image_size))
        target = TF.center_crop(target, list(image_size))
        defocus_map = TF.center_crop(defocus_map, list(image_size))
        if prediction is not None:
            prediction = TF.center_crop(prediction, list(image_size))
        if depth_map is not None:
            depth_map = TF.center_crop(depth_map, list(image_size))
        return source, target, defocus_map, prediction, depth_map

    def _crop_to_divisible(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.min_divisor is None or self.min_divisor <= 1:
            return tensor
        height, width = tensor.shape[-2:]
        new_height = height - (height % self.min_divisor)
        new_width = width - (width % self.min_divisor)
        if new_height == height and new_width == width:
            return tensor
        if new_height <= 0 or new_width <= 0:
            raise ValueError(
                f"Image size {(height, width)} is smaller than divisor {self.min_divisor}"
            )
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        return TF.crop(tensor, top, left, new_height, new_width)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_path = self.source_paths[index]
        target_path = self.target_paths[index]
        map_path = self.map_by_stem.get(source_path.stem)
        depth_map_path = self.depth_map_by_stem.get(source_path.stem)
        prediction_path = self.prediction_by_stem.get(source_path.stem)

        source = _load_rgb_tensor(source_path)
        target = _load_rgb_tensor(target_path)
        if map_path is None:
            pseudo_defocus_map = torch.ones(
                (1, source.shape[-2], source.shape[-1]), dtype=source.dtype
            )
        else:
            pseudo_defocus_map = _load_gray_tensor(map_path)
        prediction = (
            _load_rgb_tensor(prediction_path) if prediction_path is not None else None
        )
        depth_map = (
            _load_gray_tensor(depth_map_path) if depth_map_path is not None else None
        )

        if self.use_augmentation:
            (
                source,
                target,
                pseudo_defocus_map,
                prediction,
                depth_map,
            ) = self._joint_random_scale(
                source,
                target,
                pseudo_defocus_map,
                prediction,
                depth_map,
            )
            (
                source,
                target,
                pseudo_defocus_map,
                prediction,
                depth_map,
            ) = self._joint_random_crop(
                source,
                target,
                pseudo_defocus_map,
                prediction,
                depth_map,
            )
            (
                source,
                target,
                pseudo_defocus_map,
                prediction,
                depth_map,
            ) = self._joint_random_hflip(
                source,
                target,
                pseudo_defocus_map,
                prediction,
                depth_map,
            )
        else:
            source = self._maybe_resize(source, is_mask=False)
            target = self._maybe_resize(target, is_mask=False)
            pseudo_defocus_map = self._maybe_resize(pseudo_defocus_map, is_mask=False)
            if prediction is not None:
                prediction = self._maybe_resize(prediction, is_mask=False)
            if depth_map is not None:
                depth_map = self._maybe_resize(depth_map, is_mask=False)

        source, target, pseudo_defocus_map, prediction, depth_map = (
            self._maybe_center_crop(
                source,
                target,
                pseudo_defocus_map,
                prediction,
                depth_map,
            )
        )
        source = self._crop_to_divisible(source)
        target = self._crop_to_divisible(target)
        pseudo_defocus_map = self._crop_to_divisible(pseudo_defocus_map)
        if prediction is not None:
            prediction = self._crop_to_divisible(prediction)
        if depth_map is not None:
            depth_map = self._crop_to_divisible(depth_map)

        sample = {
            "source": source,
            "target": target,
            "defocus_map": pseudo_defocus_map,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "index": index,
        }
        if depth_map is not None:
            sample["depth_map"] = depth_map
        if prediction is not None:
            sample["prediction"] = prediction
        return sample


class HypersimBokehDataset(DPDDDataset):
    def __init__(
        self,
        root: str = str(DATA_ROOT / "Hypersim_preprocessed"),
        split: str = "train",
        image_size: Optional[int] = None,
        use_augmentation: Optional[bool] = None,
        random_scale_range: Tuple[float, float] = (0.2, 1.0),
        random_hflip_p: float = 0.5,
        center_crop: bool = False,
        min_divisor: int = 8,
        defocus_band_thresholds: Tuple[float, float] = (0.33, 0.66),
        metadata_name: str = "bokeh_random_metadata.json",
        defocus_scale: float = 10.0,
        depth_mode: str = "metric_depth",
        task_mode: str = "defocus_deblurring",
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.use_augmentation = (
            split == "train" if use_augmentation is None else use_augmentation
        )
        self.random_scale_range = random_scale_range
        self.random_hflip_p = random_hflip_p
        self.center_crop = center_crop
        self.min_divisor = min_divisor
        self.defocus_band_thresholds = defocus_band_thresholds
        self.metadata_name = metadata_name
        self.defocus_scale = float(defocus_scale)
        self.depth_mode = depth_mode
        self.task_mode = _normalize_task_mode(task_mode)

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing Hypersim split folder: {split_dir}")

        self.items = []
        for metadata_path in sorted(split_dir.rglob(metadata_name)):
            with metadata_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            for item in payload.get("items", []):
                clear_path = Path(item["rgb_path"])
                bokeh_path = Path(item["output_path"])
                depth_path = Path(item["depth_path"])
                if (
                    not clear_path.is_file()
                    or not bokeh_path.is_file()
                    or not depth_path.is_file()
                ):
                    continue
                self.items.append(
                    {
                        "clear_path": clear_path,
                        "bokeh_path": bokeh_path,
                        "depth_path": depth_path,
                        "metadata_path": metadata_path,
                        "K": float(item["K"]),
                        "disp_focus": float(item["disp_focus"]),
                        "pair_id": item.get("pair_id", bokeh_path.stem),
                    }
                )
        if len(self.items) == 0:
            raise FileNotFoundError(
                f"No valid Hypersim bokeh samples found under {split_dir} with {metadata_name}."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        if self.task_mode == "bokeh_rendering":
            source_path = item["clear_path"]
            target_path = item["bokeh_path"]
        else:
            source_path = item["bokeh_path"]
            target_path = item["clear_path"]
        depth_path = item["depth_path"]

        source = _load_rgb_tensor(source_path)
        target = _load_rgb_tensor(target_path)
        defocus_map, depth_map = _build_hypersim_defocus_map(
            depth_path,
            item["K"],
            item["disp_focus"],
            self.defocus_scale,
            self.depth_mode,
            normalize=self.task_mode != "bokeh_rendering",
        )

        if self.use_augmentation:
            source, target, defocus_map, _, depth_map = self._joint_random_scale(
                source, target, defocus_map, None, depth_map
            )
            source, target, defocus_map, _, depth_map = self._joint_random_crop(
                source, target, defocus_map, None, depth_map
            )
            source, target, defocus_map, _, depth_map = self._joint_random_hflip(
                source, target, defocus_map, None, depth_map
            )
        else:
            source = self._maybe_resize(source, is_mask=False)
            target = self._maybe_resize(target, is_mask=False)
            defocus_map = self._maybe_resize(defocus_map, is_mask=False)
            depth_map = self._maybe_resize(depth_map, is_mask=False)

        source, target, defocus_map, _, depth_map = self._maybe_center_crop(
            source, target, defocus_map, None, depth_map
        )
        source = self._crop_to_divisible(source)
        target = self._crop_to_divisible(target)
        defocus_map = self._crop_to_divisible(defocus_map)
        if self.task_mode != "bokeh_rendering":
            defocus_map = defocus_map.clamp(0.0, 1.0)
        depth_map = self._crop_to_divisible(depth_map)

        sample = {
            "source": source,
            "target": target,
            "defocus_map": defocus_map,
            "depth_map": depth_map,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "index": index,
        }
        return sample


class RealBokehDataset(DPDDDataset):
    def __init__(
        self,
        root: str = str(DATA_ROOT / "RealBokeh"),
        split: str = "train",
        image_size: Optional[int] = None,
        use_augmentation: Optional[bool] = None,
        random_scale_range: Tuple[float, float] = (0.2, 1.0),
        random_hflip_p: float = 0.5,
        center_crop: bool = False,
        min_divisor: int = 8,
        defocus_band_thresholds: Tuple[float, float] = (0.33, 0.66),
        metadata_dir_name: str = "metadata",
        task_mode: str = "bokeh_rendering",
        defocus_scale: float = 50.0,
    ):
        self.root = Path(root)
        self.split = "validation" if split == "val" else split
        self.image_size = image_size
        self.use_augmentation = (
            self.split == "train" if use_augmentation is None else use_augmentation
        )
        self.random_scale_range = random_scale_range
        self.random_hflip_p = random_hflip_p
        self.center_crop = center_crop
        self.min_divisor = min_divisor
        self.defocus_band_thresholds = defocus_band_thresholds
        self.metadata_dir_name = metadata_dir_name
        self.task_mode = _normalize_task_mode(task_mode)
        self.defocus_scale = float(defocus_scale)

        self.split_dir = self.root / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing RealBokeh split folder: {self.split_dir}")

        metadata_dir = self.split_dir / metadata_dir_name
        if not metadata_dir.exists():
            raise FileNotFoundError(
                f"Missing RealBokeh metadata folder: {metadata_dir}"
            )

        self.items = []
        for metadata_path in sorted(
            metadata_dir.glob("*.json"),
            key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
        ):
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            source_path = self.split_dir / metadata["source_image"]
            if not source_path.is_file():
                continue

            target_images = metadata.get("target_images", [])
            target_avs = metadata.get("target_avs", [])
            if len(target_images) != len(target_avs):
                raise ValueError(
                    f"RealBokeh metadata target_images/target_avs length mismatch: {metadata_path}"
                )

            scene_id = str(metadata.get("id", metadata_path.stem))
            source_prior_path = (
                self.split_dir / "pseudo_defocus_map" / f"{source_path.stem}.png"
            )
            for target_rel, target_av in zip(target_images, target_avs):
                target_path = self.split_dir / target_rel
                if not target_path.is_file():
                    continue
                self.items.append(
                    {
                        "source_path": source_path,
                        "target_path": target_path,
                        "source_prior_path": source_prior_path,
                        "metadata_path": metadata_path,
                        "scene_id": scene_id,
                        "target_av": float(target_av),
                        "source_av": float(metadata.get("source_av", 22.0)),
                        "pair_id": Path(target_rel).stem,
                    }
                )
        if len(self.items) == 0:
            raise FileNotFoundError(
                f"No valid RealBokeh samples found under {self.split_dir}."
            )

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _aperture_embedding(target_av: float) -> float:
        return max(0.0, min(1.0, 2.0 / max(float(target_av), 1e-6)))

    def _build_realbokeh_defocus_map(
        self,
        item: Dict[str, object],
        size: Tuple[int, int],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Optional[Path], bool]:
        source_prior_path = item["source_prior_path"]
        if isinstance(source_prior_path, Path) and source_prior_path.is_file():
            defocus_map = _load_gray_raw_tensor(source_prior_path)
            return defocus_map.clamp_min(0.0), source_prior_path, True
        height, width = size
        return (
            torch.ones((1, height, width), dtype=dtype),
            None,
            False,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        if self.task_mode == "bokeh_rendering":
            source_path = item["source_path"]
            target_path = item["target_path"]
        else:
            source_path = item["target_path"]
            target_path = item["source_path"]

        source = _load_rgb_tensor(source_path)
        target = _load_rgb_tensor(target_path)
        defocus_map, defocus_map_path, has_defocus_map = (
            self._build_realbokeh_defocus_map(
                item,
                size=tuple(source.shape[-2:]),
                dtype=source.dtype,
            )
        )

        if self.use_augmentation:
            source, target, defocus_map, _, _ = self._joint_random_scale(
                source, target, defocus_map, None, None
            )
            source, target, defocus_map, _, _ = self._joint_random_crop(
                source, target, defocus_map, None, None
            )
            source, target, defocus_map, _, _ = self._joint_random_hflip(
                source, target, defocus_map, None, None
            )
        else:
            source = self._maybe_resize(source, is_mask=False)
            target = self._maybe_resize(target, is_mask=False)
            defocus_map = self._maybe_resize(defocus_map, is_mask=False)

        source, target, defocus_map, _, _ = self._maybe_center_crop(
            source, target, defocus_map, None, None
        )
        source = self._crop_to_divisible(source)
        target = self._crop_to_divisible(target)
        defocus_map = self._crop_to_divisible(defocus_map).clamp_min(0.0)
        if self.task_mode != "bokeh_rendering":
            defocus_map = defocus_map.clamp(0.0, 1.0)

        sample = {
            "source": source,
            "target": target,
            "pseudo_defocus_map": defocus_map,
            "blurred": source,
            "sharp": target,
            "defocus_map": defocus_map,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "pseudo_defocus_map_path": (
                str(defocus_map_path) if defocus_map_path is not None else None
            ),
            "has_defocus_map": has_defocus_map,
            "metadata_path": str(item["metadata_path"]),
            "scene_id": item["scene_id"],
            "pair_id": item["pair_id"],
            "target_av": item["target_av"],
            "source_av": item["source_av"],
            "aperture_embedding": self._aperture_embedding(float(item["target_av"])),
            "defocus_scale": self.defocus_scale,
            "caption": format_aperture_prompt(item["target_av"]),
            "index": index,
        }
        prior_bundle = build_prior_bundle(
            defocus_map,
            mode="raw",
            thresholds=self.defocus_band_thresholds,
        )
        sample["defocus_strength_map"] = prior_bundle["strength_map"]
        sample["defocus_rank_map"] = prior_bundle["rank_map"]
        sample["defocus_band_masks"] = prior_bundle["band_masks"]
        sample["defocus_high_band_mask"] = prior_bundle["high_band_mask"]
        return sample


class RealDOFDataset(DPDDDataset):
    def __init__(
        self,
        root: str = str(DATA_ROOT / "RealDOF"),
        split: str = "test",
        image_size: Optional[int] = None,
        use_augmentation: Optional[bool] = None,
        random_scale_range: Tuple[float, float] = (0.2, 1.0),
        random_hflip_p: float = 0.5,
        center_crop: bool = False,
        min_divisor: int = 8,
        defocus_band_thresholds: Tuple[float, float] = (0.33, 0.66),
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.use_augmentation = False if use_augmentation is None else use_augmentation
        self.random_scale_range = random_scale_range
        self.random_hflip_p = random_hflip_p
        self.center_crop = center_crop
        self.min_divisor = min_divisor
        self.defocus_band_thresholds = defocus_band_thresholds
        self.exts = tuple(ext.lower() for ext in exts)

        self.source_dir = self.root / "source"
        self.target_dir = self.root / "target"
        self.map_dir = self.root / "pseudo_defocus_map"
        for folder in (self.source_dir, self.target_dir):
            if not folder.exists():
                raise FileNotFoundError(f"Missing RealDOF dataset folder: {folder}")

        source_candidates = _list_image_files(self.source_dir, self.exts)
        target_candidates = _list_image_files(self.target_dir, self.exts)
        map_candidates = (
            _list_image_files(self.map_dir, self.exts) if self.map_dir.exists() else []
        )
        target_by_stem = _index_by_stem(target_candidates, "RealDOF target")
        self.map_by_stem = _index_by_stem(map_candidates, "RealDOF pseudo defocus map")
        missing_targets = [
            path.stem for path in source_candidates if path.stem not in target_by_stem
        ]
        if missing_targets:
            raise FileNotFoundError(
                f"Missing RealDOF targets for {len(missing_targets)} source images. "
                f"Examples: {missing_targets[:5]}"
            )
        if len(source_candidates) == 0:
            raise FileNotFoundError(
                f"No RealDOF source images found in {self.source_dir}"
            )

        self.items = [
            (source_path, target_by_stem[source_path.stem])
            for source_path in source_candidates
        ]
        missing_maps = [
            path.stem for path in source_candidates if path.stem not in self.map_by_stem
        ]
        if missing_maps:
            print(
                "Warning: Missing RealDOF pseudo defocus maps for "
                f"{len(missing_maps)} source images in {self.map_dir}. "
                "Using one-filled defocus maps for those samples. "
                f"Examples: {missing_maps[:5]}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_path, target_path = self.items[index]
        map_path = self.map_by_stem.get(source_path.stem)
        source = _load_rgb_tensor(source_path)
        target = _load_rgb_tensor(target_path)
        if map_path is None:
            defocus_map = torch.ones(
                (1, source.shape[-2], source.shape[-1]), dtype=source.dtype
            )
            has_defocus_map = False
        else:
            defocus_map = _load_gray_tensor(map_path)
            has_defocus_map = True

        if self.use_augmentation:
            source, target, defocus_map, _, _ = self._joint_random_scale(
                source, target, defocus_map, None, None
            )
            source, target, defocus_map, _, _ = self._joint_random_crop(
                source, target, defocus_map, None, None
            )
            source, target, defocus_map, _, _ = self._joint_random_hflip(
                source, target, defocus_map, None, None
            )
        else:
            source = self._maybe_resize(source, is_mask=False)
            target = self._maybe_resize(target, is_mask=False)
            defocus_map = self._maybe_resize(defocus_map, is_mask=False)

        source, target, defocus_map, _, _ = self._maybe_center_crop(
            source, target, defocus_map, None, None
        )
        source = self._crop_to_divisible(source)
        target = self._crop_to_divisible(target)
        defocus_map = self._crop_to_divisible(defocus_map).clamp(0.0, 1.0)

        return {
            "source": source,
            "target": target,
            "blurred": source,
            "sharp": target,
            "defocus_map": defocus_map,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "pseudo_defocus_map_path": str(map_path) if map_path is not None else None,
            "has_defocus_map": has_defocus_map,
            "caption": "a clear source image",
            "index": index,
        }


class ClearImageSyntheticDefocusDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: Optional[int] = None,
        use_augmentation: bool = True,
        random_scale_range: Tuple[float, float] = (0.2, 1.0),
        random_hflip_p: float = 0.5,
        center_crop: bool = False,
        min_divisor: int = 8,
        radius_min: float = 1.0,
        radius_max: float = 8.0,
        center_crop_ratio: float = 0.6,
        spatially_varying: bool = True,
        blur_type: str = "mixed",
        blur_map_mode: str = "gradient",
        gradient_direction: str = "random",
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.image_sizes = _normalize_image_sizes(image_size)
        self.use_augmentation = use_augmentation
        self.random_scale_range = random_scale_range
        self.random_hflip_p = random_hflip_p
        self.center_crop = center_crop
        self.min_divisor = min_divisor
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.center_crop_ratio = float(center_crop_ratio)
        self.spatially_varying = bool(spatially_varying)
        self.blur_type = blur_type
        self.blur_map_mode = blur_map_mode
        self.gradient_direction = gradient_direction
        self.exts = tuple(ext.lower() for ext in exts)

        if not self.root.exists():
            raise FileNotFoundError(f"Missing clear image folder: {self.root}")
        self.image_paths = [
            path
            for path in sorted(self.root.rglob("*"))
            if path.is_file() and path.suffix.lower() in self.exts
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No clear images found in {self.root}")
        if not 0.0 < self.center_crop_ratio <= 1.0:
            raise ValueError(
                "center_crop_ratio must be in the range (0, 1], "
                f"got {self.center_crop_ratio}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resize_tensor(
        self, tensor: torch.Tensor, size: Tuple[int, int], is_mask: bool
    ) -> torch.Tensor:
        interpolation = (
            InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
        )
        return TF.resize(
            tensor,
            list(size),
            interpolation=interpolation,
            antialias=not is_mask,
        )

    def _sample_image_size(self) -> Optional[tuple[int, int]]:
        image_sizes = getattr(self, "image_sizes", None)
        if image_sizes is None and hasattr(self, "image_size"):
            image_sizes = _normalize_image_sizes(self.image_size)
            self.image_sizes = image_sizes
        if image_sizes is None:
            return None
        return random.choice(image_sizes)

    def _maybe_random_scale(self, image: torch.Tensor) -> torch.Tensor:
        image_size = self._sample_image_size()
        if image_size is None:
            return image
        crop_h, crop_w = image_size
        scale = random.uniform(*self.random_scale_range)
        height, width = image.shape[-2:]
        scaled_h = max(crop_h, int(round(height * scale)))
        scaled_w = max(crop_w, int(round(width * scale)))
        return self._resize_tensor(image, (scaled_h, scaled_w), is_mask=False)

    def _maybe_random_crop(self, image: torch.Tensor) -> torch.Tensor:
        image_size = self._sample_image_size()
        if image_size is None:
            return image
        crop_h, crop_w = image_size
        height, width = image.shape[-2:]
        if height < crop_h or width < crop_w:
            size = (max(crop_h, height), max(crop_w, width))
            image = self._resize_tensor(image, size, is_mask=False)
            height, width = image.shape[-2:]
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        return TF.crop(image, top, left, crop_h, crop_w)

    def _center_crop_by_ratio(self, image: torch.Tensor) -> torch.Tensor:
        if self.center_crop_ratio >= 1.0:
            return image
        height, width = image.shape[-2:]
        crop_h = max(1, int(round(height * self.center_crop_ratio)))
        crop_w = max(1, int(round(width * self.center_crop_ratio)))
        return TF.center_crop(image, [crop_h, crop_w])

    def _maybe_resize(self, image: torch.Tensor) -> torch.Tensor:
        image_size = self._sample_image_size()
        if image_size is None:
            return image
        return self._resize_tensor(image, image_size, is_mask=False)

    def _maybe_center_crop(self, image: torch.Tensor) -> torch.Tensor:
        image_size = self._sample_image_size()
        if not self.center_crop or image_size is None:
            return image
        return TF.center_crop(image, list(image_size))

    def _crop_to_divisible(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.min_divisor is None or self.min_divisor <= 1:
            return tensor
        height, width = tensor.shape[-2:]
        new_height = height - (height % self.min_divisor)
        new_width = width - (width % self.min_divisor)
        if new_height == height and new_width == width:
            return tensor
        if new_height <= 0 or new_width <= 0:
            raise ValueError(
                f"Image size {(height, width)} is smaller than divisor {self.min_divisor}"
            )
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        return TF.crop(tensor, top, left, new_height, new_width)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        clear_path = self.image_paths[index]
        clear = _load_rgb_tensor(clear_path)
        clear = self._center_crop_by_ratio(clear)

        if self.use_augmentation:
            clear = self._maybe_random_scale(clear)
            clear = self._maybe_random_crop(clear)
            if random.random() < self.random_hflip_p:
                clear = TF.hflip(clear)
        else:
            clear = self._maybe_resize(clear)

        clear = self._maybe_center_crop(clear)
        clear = self._crop_to_divisible(clear)
        source, blur_map = synthetic_defocus_blur(
            clear,
            radius_min=self.radius_min,
            radius_max=self.radius_max,
            spatially_varying=self.spatially_varying,
            blur_type=self.blur_type,
            blur_map_mode=self.blur_map_mode,
            gradient_direction=self.gradient_direction,
        )
        blur_map = self._crop_to_divisible(blur_map)

        return {
            "source": source,
            "target": clear,
            "defocus_map": blur_map,
            "blur_map": blur_map,
            "depth_map": blur_map,
            "source_path": str(clear_path),
            "target_path": str(clear_path),
            "batch_type": "synthetic",
            "sample_id": index,
            "index": index,
        }


class DeblurringDataset(DPDDDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Optional[int] = None,
        use_augmentation: Optional[bool] = None,
        random_scale_range: Tuple[float, float] = (0.2, 1.0),
        random_hflip_p: float = 0.5,
        center_crop: bool = False,
        min_divisor: int = 8,
        defocus_band_thresholds: Tuple[float, float] = (0.33, 0.66),
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.use_augmentation = (
            split == "train" if use_augmentation is None else use_augmentation
        )
        self.random_scale_range = random_scale_range
        self.random_hflip_p = random_hflip_p
        self.center_crop = center_crop
        self.min_divisor = min_divisor
        self.defocus_band_thresholds = defocus_band_thresholds

        manifest_path = self._resolve_manifest(split)
        self.manifest_path = manifest_path
        manifest_root = manifest_path.parent
        self.items = []
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header: {manifest_path}")
            required = {"source", "target"}
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Manifest {manifest_path} is missing required column(s): {sorted(missing)}"
                )
            for row_idx, row in enumerate(reader, start=2):
                source_path = _resolve_manifest_path(manifest_root, row["source"])
                target_path = _resolve_manifest_path(manifest_root, row["target"])
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"Missing source image from {manifest_path}:{row_idx}: {source_path}"
                    )
                if not target_path.is_file():
                    raise FileNotFoundError(
                        f"Missing target image from {manifest_path}:{row_idx}: {target_path}"
                    )
                self.items.append((source_path, target_path))

        if len(self.items) == 0:
            raise FileNotFoundError(f"No source/target pairs found in {manifest_path}")

    def _resolve_manifest(self, split: str) -> Path:
        if self.root.is_file():
            return self.root

        candidates = []
        if split == "train":
            candidates.extend(["train.csv", "deblur_train.csv"])
        elif split in {"val", "validation"}:
            candidates.extend(["val.csv", "validation.csv", "deblur_val.csv"])
        elif split == "test":
            candidates.extend(["test.csv", "deblur_test.csv"])
        candidates.append(f"{split}.csv")

        for name in candidates:
            path = self.root / name
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Could not find a manifest CSV for split {split!r} under {self.root}. "
            f"Tried: {candidates}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_path, target_path = self.items[index]
        source = _load_rgb_tensor(source_path)
        target = _load_rgb_tensor(target_path)
        defocus_map = torch.ones(
            (1, source.shape[-2], source.shape[-1]), dtype=source.dtype
        )

        if self.use_augmentation:
            source, target, defocus_map, _, _ = self._joint_random_scale(
                source, target, defocus_map, None, None
            )
            source, target, defocus_map, _, _ = self._joint_random_crop(
                source, target, defocus_map, None, None
            )
            source, target, defocus_map, _, _ = self._joint_random_hflip(
                source, target, defocus_map, None, None
            )
        else:
            source = self._maybe_resize(source, is_mask=False)
            target = self._maybe_resize(target, is_mask=False)
            defocus_map = self._maybe_resize(defocus_map, is_mask=False)

        source, target, defocus_map, _, _ = self._maybe_center_crop(
            source, target, defocus_map, None, None
        )
        source = self._crop_to_divisible(source)
        target = self._crop_to_divisible(target)
        defocus_map = self._crop_to_divisible(defocus_map).clamp(0.0, 1.0)

        return {
            "source": source,
            "target": target,
            "blurred": source,
            "sharp": target,
            "defocus_map": defocus_map,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "caption": "a clear source image",
            "index": index,
        }


class LFDOFDataset(DPDDDataset):
    def __init__(
        self,
        root: str = str(DATA_ROOT / "LFDOF"),
        split: str = "train",
        image_size: Optional[int] = None,
        use_augmentation: Optional[bool] = None,
        random_scale_range: Tuple[float, float] = (0.4, 1.0),
        random_hflip_p: float = 0.5,
        center_crop: bool = False,
        min_divisor: int = 8,
        defocus_band_thresholds: Tuple[float, float] = (0.33, 0.66),
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.use_augmentation = (
            split == "train" if use_augmentation is None else use_augmentation
        )
        self.random_scale_range = random_scale_range
        self.random_hflip_p = random_hflip_p
        self.center_crop = center_crop
        self.min_divisor = min_divisor
        self.defocus_band_thresholds = defocus_band_thresholds
        self.exts = tuple(ext.lower() for ext in exts)

        if self.root.name in {"train_data", "test_data"}:
            self.split_dir = self.root
        else:
            split_dir_name = "train_data" if split == "train" else "test_data"
            self.split_dir = self.root / split_dir_name
            if not self.split_dir.exists() and split != "train":
                self.split_dir = self.root / "train_data"
        self.input_dir = self.split_dir / "input"
        self.target_dir = self.split_dir / "ground_truth"
        for folder in (self.input_dir, self.target_dir):
            if not folder.exists():
                raise FileNotFoundError(f"Missing LFDOF dataset folder: {folder}")

        target_candidates = _list_image_files(self.target_dir, self.exts)
        target_by_stem = _index_by_stem(target_candidates, "LFDOF ground truth")
        self.items = []
        for scene_dir in sorted(p for p in self.input_dir.iterdir() if p.is_dir()):
            target_path = target_by_stem.get(scene_dir.name)
            if target_path is None:
                continue
            for source_path in _list_image_files(scene_dir, self.exts):
                self.items.append((source_path, target_path))
        if not self.items:
            raise FileNotFoundError(
                f"No LFDOF source/target pairs found under {self.split_dir}"
            )

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _aperture_embedding(path: Path) -> float:
        match = re.search(r"_ap(\d+)_", path.stem)
        return float(match.group(1)) if match else 1.0

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_path, target_path = self.items[index]
        source = _load_rgb_tensor(source_path)
        target = _load_rgb_tensor(target_path)
        defocus_map = torch.ones(
            (1, source.shape[-2], source.shape[-1]), dtype=source.dtype
        )

        if self.use_augmentation:
            source, target, defocus_map, _, _ = self._joint_random_scale(
                source, target, defocus_map, None, None
            )
            source, target, defocus_map, _, _ = self._joint_random_crop(
                source, target, defocus_map, None, None
            )
            source, target, defocus_map, _, _ = self._joint_random_hflip(
                source, target, defocus_map, None, None
            )
        else:
            source = self._maybe_resize(source, is_mask=False)
            target = self._maybe_resize(target, is_mask=False)
            defocus_map = self._maybe_resize(defocus_map, is_mask=False)

        source, target, defocus_map, _, _ = self._maybe_center_crop(
            source, target, defocus_map, None, None
        )
        source = self._crop_to_divisible(source)
        target = self._crop_to_divisible(target)
        defocus_map = self._crop_to_divisible(defocus_map).clamp(0.0, 1.0)

        return {
            "source": source,
            "target": target,
            "blurred": source,
            "sharp": target,
            "defocus_map": defocus_map,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "caption": "a sharp, clear image",
            "aperture_embedding": self._aperture_embedding(source_path),
            "index": index,
        }


class MixedDeblurDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Optional[int] = None,
        use_augmentation: Optional[bool] = None,
        random_scale_range: Tuple[float, float] = (0.4, 1.0),
        random_hflip_p: float = 0.5,
        center_crop: bool = False,
        min_divisor: int = 8,
        defocus_band_thresholds: Tuple[float, float] = (0.33, 0.66),
        dpdd_root: Optional[str] = None,
        deblurring_root: Optional[str] = None,
        lfdof_root: Optional[str] = None,
        clear_data_dir: Optional[str] = None,
        dpdd_prob: float = 0.34,
        deblurring_prob: float = 0.33,
        lfdof_prob: float = 0.0,
        synthetic_prob: float = 0.33,
        synthetic_blur_radius_min: float = 1.0,
        synthetic_blur_radius_max: float = 8.0,
        synthetic_spatially_varying: bool = True,
        synthetic_blur_type: str = "mixed",
        synthetic_blur_map_mode: str = "gradient",
        synthetic_gradient_direction: str = "random",
    ):
        self.root = Path(root)
        self.split = split
        self.use_augmentation = (
            split == "train" if use_augmentation is None else use_augmentation
        )

        dpdd_root = dpdd_root or str(DATA_ROOT / "dd_dp_dataset_png")
        deblurring_root = (
            deblurring_root
            or str(Path(__file__).resolve().parents[1] / "configs/deblur_train_realbokeh_top1000_laplacian.csv")
        )
        lfdof_root = lfdof_root or str(DATA_ROOT / "LFDOF")
        if clear_data_dir is None:
            dpdd_target_dir = Path(dpdd_root) / f"{split}_c" / "target"
            clear_data_dir = str(
                dpdd_target_dir if dpdd_target_dir.exists() else dpdd_root
            )

        common_kwargs = {
            "split": split,
            "image_size": image_size,
            "use_augmentation": self.use_augmentation,
            "random_scale_range": random_scale_range,
            "random_hflip_p": random_hflip_p,
            "center_crop": center_crop,
            "min_divisor": min_divisor,
            "defocus_band_thresholds": defocus_band_thresholds,
        }
        self.datasets: list[tuple[str, Dataset]] = [
            (
                "dpdd",
                DPDDDataset(
                    root=dpdd_root,
                    **common_kwargs,
                ),
            ),
            (
                "deblurring",
                DeblurringDataset(
                    root=deblurring_root,
                    **common_kwargs,
                ),
            ),
        ]
        if float(lfdof_prob) > 0.0:
            self.datasets.append(
                (
                    "lfdof",
                    LFDOFDataset(
                        root=lfdof_root,
                        **common_kwargs,
                    ),
                )
            )
        self.datasets.append(
            (
                "synthetic",
                ClearImageSyntheticDefocusDataset(
                    root=clear_data_dir,
                    image_size=image_size,
                    use_augmentation=self.use_augmentation,
                    random_scale_range=random_scale_range,
                    random_hflip_p=random_hflip_p,
                    center_crop=center_crop,
                    min_divisor=min_divisor,
                    radius_min=synthetic_blur_radius_min,
                    radius_max=synthetic_blur_radius_max,
                    spatially_varying=synthetic_spatially_varying,
                    blur_type=synthetic_blur_type,
                    blur_map_mode=synthetic_blur_map_mode,
                    gradient_direction=synthetic_gradient_direction,
                ),
            )
        )
        raw_weights = {
            "dpdd": max(float(dpdd_prob), 0.0),
            "deblurring": max(float(deblurring_prob), 0.0),
            "lfdof": max(float(lfdof_prob), 0.0),
            "synthetic": max(float(synthetic_prob), 0.0),
        }
        raw_weights = {name: raw_weights[name] for name, _ in self.datasets}
        total = sum(raw_weights.values())
        if total <= 0:
            raise ValueError("At least one mixed dataset probability must be > 0.")
        self.weights = [raw_weights[name] / total for name, _ in self.datasets]
        self.length = max(len(dataset) for _, dataset in self.datasets)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.use_augmentation:
            dataset_idx = random.choices(
                range(len(self.datasets)),
                weights=self.weights,
                k=1,
            )[0]
            name, dataset = self.datasets[dataset_idx]
            sample_index = random.randrange(len(dataset))
        else:
            dataset_idx = index % len(self.datasets)
            name, dataset = self.datasets[dataset_idx]
            sample_index = (index // len(self.datasets)) % len(dataset)

        sample = dict(dataset[sample_index])
        sample["batch_type"] = name
        sample["sample_id"] = index
        sample["mixed_source_index"] = sample_index
        return sample


def build_training_dataset(
    args,
    split: str,
    image_size: Optional[int],
    use_augmentation: Optional[bool],
    center_crop: bool = False,
) -> Dataset:
    dataset_name = str(getattr(args, "dataset_name", "dpdd") or "dpdd").lower()
    common_kwargs = {
        "root": args.train_data_dir,
        "split": split,
        "image_size": image_size,
        "use_augmentation": use_augmentation,
        "center_crop": center_crop,
        "defocus_band_thresholds": tuple(args.defocus_band_thresholds),
    }
    if dataset_name in {"dpdd", "dd_dp", "dddp"}:
        return DPDDDataset(**common_kwargs)
    if dataset_name in {"deblurring", "deblur", "manifest", "manifest_deblurring"}:
        return DeblurringDataset(**common_kwargs)
    if dataset_name in {"lfdof", "lf_dof"}:
        return LFDOFDataset(**common_kwargs)
    if dataset_name in {"hypersim", "hypersim_bokeh"}:
        return HypersimBokehDataset(
            **common_kwargs,
            metadata_name=args.hypersim_metadata_name,
            defocus_scale=args.hypersim_defocus_scale,
            depth_mode=args.hypersim_depth_mode,
            task_mode=args.task_mode,
        )
    if dataset_name in {"realbokeh", "real_bokeh"}:
        return RealBokehDataset(
            **common_kwargs,
            task_mode=args.task_mode,
            defocus_scale=args.realbokeh_defocus_scale,
        )
    if dataset_name in {"realdof", "real_dof"}:
        return RealDOFDataset(**common_kwargs)
    if dataset_name in {"mixed"}:
        if split != "train":
            dpdd_kwargs = dict(common_kwargs)
            dpdd_kwargs["root"] = (
                getattr(args, "mixed_dpdd_data_dir", None)
                or str(DATA_ROOT / "dd_dp_dataset_png")
            )
            return DPDDDataset(**dpdd_kwargs)
        return MixedDeblurDataset(
            **common_kwargs,
            dpdd_root=getattr(args, "mixed_dpdd_data_dir", None),
            deblurring_root=getattr(args, "mixed_deblurring_data_dir", None),
            lfdof_root=getattr(args, "lfdof_data_dir", None),
            clear_data_dir=getattr(args, "clear_data_dir", None),
            dpdd_prob=getattr(args, "mixed_dpdd_prob", 0.34),
            deblurring_prob=getattr(args, "mixed_deblurring_prob", 0.33),
            lfdof_prob=getattr(args, "lfdof_prob", 0.0),
            synthetic_prob=getattr(args, "synthetic_prob", 0.33),
            synthetic_blur_radius_min=getattr(args, "synthetic_blur_radius_min", 1.0),
            synthetic_blur_radius_max=getattr(args, "synthetic_blur_radius_max", 8.0),
            synthetic_spatially_varying=getattr(
                args,
                "synthetic_spatially_varying",
                True,
            ),
            synthetic_blur_type=getattr(args, "synthetic_blur_type", "mixed"),
            synthetic_blur_map_mode=getattr(
                args, "synthetic_blur_map_mode", "gradient"
            ),
            synthetic_gradient_direction=getattr(
                args,
                "synthetic_gradient_direction",
                "random",
            ),
        )
    raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")
