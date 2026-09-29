import json
import os
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode, to_tensor

from dataset.util import (
    Mode,
    build_input_dict,
    generate_maps,
    make_point_focus_map,
)


def _to_inverse_depth_tensor(
    tensor: torch.Tensor,
    value: Optional[float] = None,
    eps: float = 1e-6,
    min_valid_depth: float = 0.1,
) -> tuple[torch.Tensor, Optional[float]]:
    valid = torch.isfinite(tensor) & (tensor >= min_valid_depth)
    if bool(valid.any()):
        farthest_valid = tensor[valid].max()
        safe_tensor = torch.where(valid, tensor, farthest_valid)
    else:
        safe_tensor = tensor.clamp_min(eps)
    inverse = (1.0 / safe_tensor.clamp_min(eps)).clamp(0.0, 1.0)
    inverse_value = None if value is None else 1.0 / max(float(value), eps)
    if inverse_value is not None:
        inverse_value = max(0.0, min(float(inverse_value), 1.0))
    return inverse, inverse_value


def _load_rgb_tensor_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_depth_or_disp_tensor(
    path: Path,
    depth_mode: str,
) -> torch.Tensor:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    depth_or_disp = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)

    if depth_mode == "metric_depth":
        return depth_or_disp / 1000.0
    elif depth_mode == "normalized_depth":
        return depth_or_disp
    elif depth_mode == "disparity":
        return depth_or_disp
    else:
        raise ValueError(
            f"Unsupported depth_mode {depth_mode!r}. "
            "Use metric_depth, normalized_depth, or disparity."
        )


def _resolve_path(path_value: str, metadata_path: Path, split_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if path.is_absolute():
        parts = path.parts
        if split_dir.name in parts:
            split_idx = parts.index(split_dir.name)
            split_relative = Path(*parts[split_idx + 1 :])
            remapped = split_dir / split_relative
            if remapped.is_file():
                return remapped
        return path
    metadata_relative = metadata_path.parent / path
    if metadata_relative.is_file():
        return metadata_relative
    return split_dir / path


def _resize_tensor(
    tensor: torch.Tensor,
    size: tuple[int, int],
    interpolation=InterpolationMode.BILINEAR,
) -> torch.Tensor:
    return TF.resize(tensor, list(size), interpolation=interpolation, antialias=True)


def _joint_random_scale(
    source: torch.Tensor,
    target: torch.Tensor,
    pos_map: torch.Tensor,
    defocus_map: torch.Tensor,
    depth_map: torch.Tensor,
    image_size: Optional[int],
    random_scale_range: tuple[float, float],
    focus_map: Optional[torch.Tensor] = None,
):
    if image_size is None:
        if focus_map is not None:
            return source, target, pos_map, defocus_map, depth_map, focus_map
        return source, target, pos_map, defocus_map, depth_map

    scale = random.uniform(*random_scale_range)
    height, width = source.shape[-2:]
    scaled_h = max(image_size, int(round(height * scale)))
    scaled_w = max(image_size, int(round(width * scale)))
    size = (scaled_h, scaled_w)
    result = (
        _resize_tensor(source, size),
        _resize_tensor(target, size),
        _resize_tensor(pos_map, size),
        _resize_tensor(defocus_map, size),
        _resize_tensor(depth_map, size),
    )
    if focus_map is not None:
        result = result + (_resize_tensor(focus_map, size),)
    return result


def _joint_random_crop(
    source: torch.Tensor,
    target: torch.Tensor,
    pos_map: torch.Tensor,
    defocus_map: torch.Tensor,
    depth_map: torch.Tensor,
    image_size: Optional[int],
    focus_map: Optional[torch.Tensor] = None,
):
    if image_size is None:
        if focus_map is not None:
            return source, target, pos_map, defocus_map, depth_map, focus_map
        return source, target, pos_map, defocus_map, depth_map

    height, width = source.shape[-2:]
    if height < image_size or width < image_size:
        size = (max(image_size, height), max(image_size, width))
        source = _resize_tensor(source, size)
        target = _resize_tensor(target, size)
        pos_map = _resize_tensor(pos_map, size)
        defocus_map = _resize_tensor(defocus_map, size)
        depth_map = _resize_tensor(depth_map, size)
        focus_map = _resize_tensor(focus_map, size)
        height, width = source.shape[-2:]

    top = random.randint(0, height - image_size)
    left = random.randint(0, width - image_size)
    result = (
        TF.crop(source, top, left, image_size, image_size),
        TF.crop(target, top, left, image_size, image_size),
        TF.crop(pos_map, top, left, image_size, image_size),
        TF.crop(defocus_map, top, left, image_size, image_size),
        TF.crop(depth_map, top, left, image_size, image_size),
    )
    if focus_map is not None:
        result = result + (TF.crop(focus_map, top, left, image_size, image_size),)
    return result


def _joint_random_hflip(
    source: torch.Tensor,
    target: torch.Tensor,
    pos_map: torch.Tensor,
    defocus_map: torch.Tensor,
    depth_map: torch.Tensor,
    random_hflip_p: float,
    focus_map: Optional[torch.Tensor] = None,
):
    if random.random() >= random_hflip_p:
        if focus_map is not None:
            return source, target, pos_map, defocus_map, depth_map, focus_map
        return source, target, pos_map, defocus_map, depth_map
    result = (
        TF.hflip(source),
        TF.hflip(target),
        TF.hflip(pos_map),
        TF.hflip(defocus_map),
        TF.hflip(depth_map),
    )
    if focus_map is not None:
        result = result + (TF.hflip(focus_map),)
    return result


def _joint_random_rotate90(
    source: torch.Tensor,
    target: torch.Tensor,
    pos_map: torch.Tensor,
    defocus_map: torch.Tensor,
    depth_map: torch.Tensor,
    focus_map: Optional[torch.Tensor] = None,
):
    k = random.randint(0, 3)
    if k == 0:
        if focus_map is not None:
            return source, target, pos_map, defocus_map, depth_map, focus_map
        return source, target, pos_map, defocus_map, depth_map
    dims = (-2, -1)
    result = (
        torch.rot90(source, k, dims),
        torch.rot90(target, k, dims),
        torch.rot90(pos_map, k, dims),
        torch.rot90(defocus_map, k, dims),
        torch.rot90(depth_map, k, dims),
    )
    if focus_map is not None:
        result = result + (torch.rot90(focus_map, k, dims),)
    return result


class HypersimBokehDMA(Dataset):
    """
    Hypersim synthetic bokeh loader for Bokehlicious DMA training.

    Expected metadata item keys follow AutoRefocusV2's HypersimBokehDataset:
    rgb_path, output_path, depth_path, K, disp_focus, and optional pair_id.
    The depth file is used both to build the synthetic defocus supervision map
    and as the model input for focal-distance learning.
    """

    def __init__(
        self,
        root: Union[
            str, Path
        ] = "/ssddd/chingheng/Deblurring-Dataset/Hypersim_preprocessed",
        split: str = "train",
        metadata_name: str = "bokeh_random_metadata.json",
        device: str = "cuda",
        image_size: Optional[int] = None,
        use_augmentation: Optional[bool] = None,
        random_scale_range: tuple[float, float] = (0.4, 1.0),
        random_hflip_p: float = 0.5,
        random_rotate90: bool = True,
        depth_mode: str = "metric_depth",
        inverse_depth: bool = False,
        task_mode: str = "bokeh_rendering",
        k_normalizer: float = 100.0,
        use_k_in_defocus: bool = False,
        focus_plane_tolerance: float = 1e-6,
        focus_map_type: str = "cosine",
        focus_map_gaussian_sigma: Optional[float] = None,
        focus_map_gaussian_sigma_scale: float = 3.0,
    ):
        self.root = Path(root)
        self.split = split
        self.metadata_name = metadata_name
        self._device = device
        self.depth_mode = depth_mode
        self.inverse_depth = bool(inverse_depth)
        self.task_mode = str(task_mode).lower()
        self.k_normalizer = float(k_normalizer)
        self.use_k_in_defocus = bool(use_k_in_defocus)
        self.focus_plane_tolerance = float(focus_plane_tolerance)
        self.focus_map_type = str(focus_map_type).lower()
        if self.focus_map_type == "gaussain":
            self.focus_map_type = "gaussian"
        self.focus_map_gaussian_sigma = focus_map_gaussian_sigma
        self.focus_map_gaussian_sigma_scale = float(focus_map_gaussian_sigma_scale)
        self.use_augmentation = (
            split == "train" if use_augmentation is None else use_augmentation
        )
        if image_size is None:
            env_image_size = os.environ.get("BOKEHLICIOUS_DMA_IMAGE_SIZE")
            image_size = int(env_image_size) if env_image_size else None
        self.image_size = image_size
        self.random_scale_range = random_scale_range
        self.random_hflip_p = float(random_hflip_p)
        self.random_rotate90 = bool(random_rotate90)

        if self.k_normalizer <= 0:
            raise ValueError(f"k_normalizer must be > 0, got {self.k_normalizer}")
        if self.inverse_depth and self.depth_mode == "disparity":
            raise ValueError(
                "inverse_depth=True is not supported with depth_mode='disparity' "
                "because disparity is already inverse-depth-like."
            )
        if self.focus_plane_tolerance < 0.0:
            raise ValueError(
                "focus_plane_tolerance must be >= 0, got "
                f"{self.focus_plane_tolerance}"
            )
        if self.focus_map_type not in {"cosine", "gaussian"}:
            raise ValueError('focus_map_type must be "cosine" or "gaussian".')
        if self.focus_map_gaussian_sigma is not None:
            self.focus_map_gaussian_sigma = float(self.focus_map_gaussian_sigma)
            if self.focus_map_gaussian_sigma <= 0:
                raise ValueError("focus_map_gaussian_sigma must be > 0.")
        if self.focus_map_gaussian_sigma_scale <= 0:
            raise ValueError("focus_map_gaussian_sigma_scale must be > 0.")
        if self.task_mode not in {"bokeh_rendering", "defocus_deblurring"}:
            raise ValueError(
                "task_mode must be bokeh_rendering or defocus_deblurring, got "
                f"{task_mode!r}"
            )
        if self.use_augmentation and self.image_size is None:
            raise ValueError("image_size must be set when use_augmentation=True")

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing Hypersim split folder: {split_dir}")
        self.split_dir = split_dir

        self.items = []
        for metadata_path in sorted(split_dir.rglob(metadata_name)):
            with metadata_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            for item in payload.get("items", []):
                clear_path = _resolve_path(item["rgb_path"], metadata_path, split_dir)
                bokeh_path = _resolve_path(
                    item["output_path"], metadata_path, split_dir
                )
                depth_path = _resolve_path(item["depth_path"], metadata_path, split_dir)
                if (
                    not clear_path.is_file()
                    or not bokeh_path.is_file()
                    or not depth_path.is_file()
                ):
                    continue
                disp_focus = float(item["disp_focus"])
                if self.inverse_depth and disp_focus <= 1e-6:
                    continue
                self.items.append(
                    {
                        "clear_path": clear_path,
                        "bokeh_path": bokeh_path,
                        "depth_path": depth_path,
                        "K": float(item["K"]),
                        "disp_focus": disp_focus,
                        "pair_id": item.get("pair_id", bokeh_path.stem),
                        "metadata_path": metadata_path,
                    }
                )

        if len(self.items) == 0:
            raise FileNotFoundError(
                f"No valid Hypersim bokeh samples found under {split_dir} "
                f"with metadata {metadata_name}."
            )

        print(
            f"HypersimBokehDMA initialized in {split} split with "
            f"{len(self.items)} samples!"
        )

    def __len__(self):
        return len(self.items)

    def _aperture_embedding(self, k_value: float) -> float:
        return max(0.0, min(float(k_value) / self.k_normalizer, 1.0))

    def _build_defocus_map(
        self,
        depth_map: torch.Tensor,
        disp_focus: float,
        aperture_embedding: float,
    ) -> torch.Tensor:
        defocus_map = torch.abs(depth_map - float(disp_focus))
        if self.use_k_in_defocus:
            defocus_map = defocus_map * float(aperture_embedding)
        return defocus_map

    def _sample_focus_pixel_from_disp(
        self,
        depth_map: torch.Tensor,
        disp_focus: float,
    ) -> tuple[int, int]:
        _, _, width = depth_map.shape
        distance = torch.abs(depth_map - float(disp_focus)).flatten()
        min_distance = distance.min()
        candidates = torch.nonzero(
            distance <= min_distance + self.focus_plane_tolerance,
            as_tuple=False,
        ).flatten()
        if candidates.numel() == 0:
            flat_idx = torch.argmin(distance).item()
        else:
            choice = torch.randint(candidates.numel(), (1,)).item()
            flat_idx = candidates[choice].item()
        return flat_idx % width, flat_idx // width

    def __getitem__(self, index: int):
        item = self.items[index]
        if self.task_mode == "bokeh_rendering":
            source_path = item["clear_path"]
            target_path = item["bokeh_path"]
        else:
            source_path = item["bokeh_path"]
            target_path = item["clear_path"]

        source = _load_rgb_tensor_image(source_path)
        target = _load_rgb_tensor_image(target_path)
        depth_map = _load_depth_or_disp_tensor(
            item["depth_path"],
            self.depth_mode,
        )
        disp_focus = float(item["disp_focus"])
        if self.inverse_depth:
            depth_map, disp_focus = _to_inverse_depth_tensor(depth_map, disp_focus)
            depth_map = 1.0 - depth_map
            disp_focus = 1.0 - float(disp_focus)
        aperture_embedding = self._aperture_embedding(item["K"])
        defocus_map = self._build_defocus_map(
            depth_map,
            disp_focus=disp_focus,
            aperture_embedding=aperture_embedding,
        )

        maps = generate_maps(source, aperture_embedding, target=target)
        sample = build_input_dict(
            maps, aperture_embedding, item["pair_id"], self._device
        )
        sample["bokeh_strength_map"] = defocus_map.to(self._device)
        sample["pseudo_defocus_map"] = sample["bokeh_strength_map"]
        sample["depth_map"] = depth_map.to(self._device)
        focus_pixel = self._sample_focus_pixel_from_disp(depth_map, disp_focus)
        sample["focus_map"] = make_point_focus_map(
            depth_map.shape[-1],
            depth_map.shape[-2],
            focus_pixel[0],
            focus_pixel[1],
            radius=max(depth_map.shape[-2:]),
            map_type=self.focus_map_type,
            gaussian_sigma=self.focus_map_gaussian_sigma,
            gaussian_sigma_scale=self.focus_map_gaussian_sigma_scale,
        ).to(self._device)

        if self.use_augmentation:
            (
                sample["source"],
                sample["target"],
                sample["pos_map"],
                sample["bokeh_strength_map"],
                sample["depth_map"],
                sample["focus_map"],
            ) = _joint_random_scale(
                sample["source"],
                sample["target"],
                sample["pos_map"],
                sample["bokeh_strength_map"],
                sample["depth_map"],
                self.image_size,
                self.random_scale_range,
                focus_map=sample["focus_map"],
            )
            (
                sample["source"],
                sample["target"],
                sample["pos_map"],
                sample["bokeh_strength_map"],
                sample["depth_map"],
                sample["focus_map"],
            ) = _joint_random_crop(
                sample["source"],
                sample["target"],
                sample["pos_map"],
                sample["bokeh_strength_map"],
                sample["depth_map"],
                self.image_size,
                focus_map=sample["focus_map"],
            )
            (
                sample["source"],
                sample["target"],
                sample["pos_map"],
                sample["bokeh_strength_map"],
                sample["depth_map"],
                sample["focus_map"],
            ) = _joint_random_hflip(
                sample["source"],
                sample["target"],
                sample["pos_map"],
                sample["bokeh_strength_map"],
                sample["depth_map"],
                self.random_hflip_p,
                focus_map=sample["focus_map"],
            )
            if self.random_rotate90:
                (
                    sample["source"],
                    sample["target"],
                    sample["pos_map"],
                    sample["bokeh_strength_map"],
                    sample["depth_map"],
                    sample["focus_map"],
                ) = _joint_random_rotate90(
                    sample["source"],
                    sample["target"],
                    sample["pos_map"],
                    sample["bokeh_strength_map"],
                    sample["depth_map"],
                    focus_map=sample["focus_map"],
                )
            sample["bokeh_strength_map"] = self._build_defocus_map(
                sample["depth_map"],
                disp_focus=disp_focus,
                aperture_embedding=aperture_embedding,
            ).to(self._device)
            sample["pseudo_defocus_map"] = sample["bokeh_strength_map"]
            flat_focus_idx = torch.argmax(sample["focus_map"]).item()
            focus_w = sample["focus_map"].shape[-1]
            focus_pixel = (flat_focus_idx % focus_w, flat_focus_idx // focus_w)

        sample["K"] = torch.tensor(float(item["K"]), dtype=torch.float32)
        sample["disp_focus"] = torch.tensor(float(disp_focus), dtype=torch.float32)
        sample["aperture_embedding"] = torch.tensor(
            float(aperture_embedding), dtype=torch.float32
        )
        sample["focus_point"] = torch.tensor(focus_pixel, dtype=torch.float32)
        sample["source_path"] = str(source_path)
        sample["target_path"] = str(target_path)
        sample["depth_path"] = str(item["depth_path"])
        sample["metadata_path"] = str(item["metadata_path"])
        sample["pair_id"] = str(item["pair_id"])
        return sample


HypersimBokehDataset = HypersimBokehDMA


class RealBokeh(HypersimBokehDMA):
    """Compatibility wrapper for train.py/train_dma.py style dataset construction."""

    def __init__(
        self,
        data_path: Union[str, Path],
        mode: Mode,
        binary_bokeh: bool = False,
        defocus_deblur_mode: bool = False,
        device: str = "cuda",
        challenge: bool = False,
        include_source: bool = False,
        **kwargs,
    ):
        split = mode.value if isinstance(mode, Mode) else str(mode)
        split = "val" if split == "validation" else split
        kwargs.pop("depth_map_dir", None)
        kwargs.pop("depth_normalize_scale", None)
        super().__init__(
            root=data_path,
            split=split,
            device=device,
            task_mode=(
                "defocus_deblurring" if defocus_deblur_mode else "bokeh_rendering"
            ),
            **kwargs,
        )
