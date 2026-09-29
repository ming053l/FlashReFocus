import random
import os
from json import load
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode, to_tensor

from dataset.util import (
    Mode,
    build_input_dict,
    calculate_aperture_embedding,
    crop_to_divisible,
    generate_maps,
)


def _ensure_unit_range(
    map_tensor: torch.Tensor,
    source_mode: str,
) -> torch.Tensor:
    map_tensor = torch.nan_to_num(map_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    min_value = map_tensor.min()
    max_value = map_tensor.max()

    if min_value >= 0.0 and max_value <= 1.0:
        return map_tensor

    if min_value >= 0.0:
        if source_mode in {"L", "P", "RGB", "RGBA"} and max_value <= 255.0:
            return (map_tensor / 255.0).clamp(0.0, 1.0)
        if source_mode.startswith("I;16") and max_value <= 65535.0:
            return (map_tensor / 65535.0).clamp(0.0, 1.0)

    return map_tensor.clamp(0.0, 1.0)


def load_pseudo_defocus_map(
    map_path: Union[str, Path],
    size: tuple[int, int],
    clip_percentiles: tuple[float, float] = (0.0, 100.0),
    normalize_after_clip: bool = False,
    ensure_unit_range: bool = True,
) -> torch.Tensor:
    """
    Load a pseudo defocus map as [1, H, W], clip by percentiles, then optionally
    normalize to [0, 1]. The returned tensor is kept in [0, 1] by default.
    """
    map_path = Path(map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Pseudo defocus map not found: {map_path}")

    image = Image.open(map_path)
    source_mode = image.mode
    defocus = image.convert("F")
    if defocus.size != size:
        defocus = defocus.resize(size, Image.BILINEAR)

    defocus = to_tensor(defocus).float()

    low_pct, high_pct = clip_percentiles
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise ValueError(
            "clip_percentiles must satisfy 0 <= low < high <= 100, got "
            f"{clip_percentiles}"
        )

    flat = defocus.flatten()
    low = torch.quantile(flat, low_pct / 100.0)
    high = torch.quantile(flat, high_pct / 100.0)
    defocus = defocus.clamp(min=low.item(), max=high.item())

    if normalize_after_clip:
        denom = high - low
        if denom.abs() > 1e-12:
            defocus = (defocus - low) / denom
        else:
            defocus = torch.zeros_like(defocus)

    if ensure_unit_range:
        defocus = _ensure_unit_range(defocus, source_mode)

    return defocus


def load_relative_map(
    map_path: Union[str, Path],
    size: tuple[int, int],
    clip_percentiles: tuple[float, float] = (0.0, 100.0),
    normalize_after_clip: bool = False,
    ensure_unit_range: bool = True,
) -> torch.Tensor:
    return load_pseudo_defocus_map(
        map_path=map_path,
        size=size,
        clip_percentiles=clip_percentiles,
        normalize_after_clip=normalize_after_clip,
        ensure_unit_range=ensure_unit_range,
    )


def _resize_tensor(
    tensor: Optional[torch.Tensor],
    size: tuple[int, int],
    interpolation=InterpolationMode.BILINEAR,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return TF.resize(tensor, list(size), interpolation=interpolation, antialias=True)


def _joint_random_scale(
    source: torch.Tensor,
    target: torch.Tensor,
    pos_map: torch.Tensor,
    defocus_map: torch.Tensor,
    depth_map: Optional[torch.Tensor],
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
    depth_map: Optional[torch.Tensor],
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
        (
            TF.crop(depth_map, top, left, image_size, image_size)
            if depth_map is not None
            else None
        ),
    )
    if focus_map is not None:
        result = result + (TF.crop(focus_map, top, left, image_size, image_size),)
    return result


def _joint_random_hflip(
    source: torch.Tensor,
    target: torch.Tensor,
    pos_map: torch.Tensor,
    defocus_map: torch.Tensor,
    depth_map: Optional[torch.Tensor],
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
        TF.hflip(depth_map) if depth_map is not None else None,
    )
    if focus_map is not None:
        result = result + (TF.hflip(focus_map),)
    return result


def _joint_random_rotate90(
    source: torch.Tensor,
    target: torch.Tensor,
    pos_map: torch.Tensor,
    defocus_map: torch.Tensor,
    depth_map: Optional[torch.Tensor],
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
        torch.rot90(depth_map, k, dims) if depth_map is not None else None,
    )
    if focus_map is not None:
        result = result + (torch.rot90(focus_map, k, dims),)
    return result


class RealBokeh(Dataset):
    """
    RealBokeh loader for DMA models.

    It keeps the original scalar bokeh_strength/aperture embedding, but replaces
    bokeh_strength_map with pseudo_defocus_map/<source_stem>.png after [5, 95]
    percentile clipping.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        mode: Mode,
        binary_bokeh: bool = False,
        defocus_deblur_mode: bool = False,
        device: str = "cuda",
        challenge: bool = False,
        include_source: bool = False,
        pseudo_defocus_dir: str = "pseudo_defocus_map",
        depth_map_dir: str = "in_depth",
        clip_percentiles: tuple[float, float] = (0.0, 100.0),
        normalize_defocus_map: bool = False,
        use_augmentation: Optional[bool] = None,
        image_size: Optional[int] = None,
        random_scale_range: tuple[float, float] = (0.4, 1.0),
        random_hflip_p: float = 0.5,
        random_rotate90: bool = True,
        require_depth_map: bool = True,
    ):
        self._data_path: Path = (
            Path(data_path) if isinstance(data_path, str) else data_path
        )
        assert (
            self._data_path.exists()
        ), f"Data directory {self._data_path.absolute()} does not exist!"

        self.defocus_deblur_mode = defocus_deblur_mode
        if self.defocus_deblur_mode:
            print("Dataset is in Defocus Deblur mode!")

        self._mode = mode
        self.challenge = challenge
        self.include_source = include_source
        self._iteration_mode = "sample"
        self._binary_bokeh = binary_bokeh
        self.pseudo_defocus_dir = pseudo_defocus_dir
        self.depth_map_dir = depth_map_dir
        self.clip_percentiles = clip_percentiles
        self.normalize_defocus_map = normalize_defocus_map
        self.use_augmentation = (
            self._mode == Mode.TRAIN if use_augmentation is None else use_augmentation
        )
        if image_size is None:
            env_image_size = os.environ.get("BOKEHLICIOUS_DMA_IMAGE_SIZE")
            image_size = int(env_image_size) if env_image_size else None
        self.image_size = image_size
        self.random_scale_range = random_scale_range
        self.random_hflip_p = float(random_hflip_p)
        self.random_rotate90 = bool(random_rotate90)
        self.require_depth_map = bool(require_depth_map)

        if self._binary_bokeh:
            print("Using binary bokeh strength of F22.0 and F2.0!")
            self._iteration_mode = "scene"
        if self.use_augmentation and self.image_size is None:
            raise ValueError("image_size must be set when use_augmentation=True")

        self._mode_dir = self._data_path.joinpath(self._mode.value)
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Mode directory {self._mode_dir} does not exist!")

        metadata_dir = self._mode_dir.joinpath("metadata")
        self._scene_list = sorted(
            [load(open(f)) for f in metadata_dir.glob("*.json")],
            key=lambda x: x["id"],
        )
        if len(self._scene_list) == 0:
            raise FileNotFoundError(f"No metadata files found in {metadata_dir}!")

        defocus_dir = self._mode_dir.joinpath(self.pseudo_defocus_dir)
        if not defocus_dir.exists():
            raise FileNotFoundError(
                f"Pseudo defocus map directory {defocus_dir} does not exist!"
            )
        self._depth_dir = self._mode_dir.joinpath(self.depth_map_dir)
        self._has_depth_map = self._depth_dir.exists()
        if self.require_depth_map and not self._has_depth_map:
            raise FileNotFoundError(
                f"Depth map directory {self._depth_dir} does not exist!"
            )

        self._sample_list = []
        for scene_id, metadata in enumerate(self._scene_list):
            for sample_id in range(len(metadata["target_images"])):
                self._sample_list.append((scene_id, sample_id))
            if self._mode == Mode.TRAIN and self.include_source:
                self._sample_list.append((scene_id, None))

        self._device = device

        print(
            f"RealBokeh DMA Dataloader initialized in {mode} mode with "
            f"{len(self._scene_list)} scenes and {len(self._sample_list)} samples!"
        )

    def __len__(self):
        if self._iteration_mode == "sample":
            return len(self._sample_list)
        return len(self._scene_list)

    def _pseudo_defocus_path(self, metadata: dict) -> Path:
        source_stem = Path(metadata["source_image"]).stem
        return self._mode_dir.joinpath(self.pseudo_defocus_dir, f"{source_stem}.png")

    def _depth_map_path(self, metadata: dict) -> Path:
        source_stem = Path(metadata["source_image"]).stem
        return self._depth_dir.joinpath(f"{source_stem}.JPG")

    def __getitem__(self, index: int):
        if self._iteration_mode == "sample":
            scene_id, target_idx = self._sample_list[index]
            metadata: dict = self._scene_list[scene_id]
        else:
            metadata: dict = self._scene_list[index]
            if self._mode == Mode.TRAIN and self.include_source:
                target_idx = random.randrange(len(metadata["target_images"]) + 1) - 1
            else:
                target_idx = 0

        source = Image.open(self._mode_dir.joinpath(metadata["source_image"]))
        defocus_map = load_pseudo_defocus_map(
            self._pseudo_defocus_path(metadata),
            source.size,
            clip_percentiles=self.clip_percentiles,
            normalize_after_clip=False,
        )
        depth_map = None
        if self._has_depth_map:
            depth_map = load_relative_map(
                self._depth_map_path(metadata),
                source.size,
                clip_percentiles=self.clip_percentiles,
                normalize_after_clip=self.normalize_defocus_map,
            )

        if target_idx is None or target_idx == -1:
            target = source.copy() if not self.challenge else None
            tgt_av = metadata["source_av"]
            target_name = Path(metadata["source_image"]).stem
        else:
            target = (
                Image.open(
                    self._mode_dir.joinpath(metadata["target_images"][target_idx])
                )
                if not self.challenge
                else None
            )
            tgt_av = metadata["target_avs"][target_idx]
            target_name = Path(metadata["target_images"][target_idx]).stem

        if self.defocus_deblur_mode:
            tmp = source
            source = target
            target = tmp

        aperture_embedding = calculate_aperture_embedding(tgt_av)
        maps = generate_maps(source, aperture_embedding, target=target)
        return_dict = build_input_dict(
            maps, aperture_embedding, target_name, self._device
        )
        return_dict["bokeh_strength_map"] = defocus_map.to(self._device)
        return_dict["pseudo_defocus_map"] = return_dict["bokeh_strength_map"]
        if depth_map is not None:
            return_dict["depth_map"] = depth_map.to(self._device)

        if self.use_augmentation:
            (
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
            ) = _joint_random_scale(
                return_dict["source"],
                return_dict["target"],
                return_dict["pos_map"],
                return_dict["bokeh_strength_map"],
                return_dict.get("depth_map"),
                self.image_size,
                self.random_scale_range,
            )
            (
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
            ) = _joint_random_crop(
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
                self.image_size,
            )
            (
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
            ) = _joint_random_hflip(
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
                self.random_hflip_p,
            )
            if self.random_rotate90:
                (
                    source_t,
                    target_t,
                    pos_map_t,
                    defocus_t,
                    depth_t,
                ) = _joint_random_rotate90(
                    source_t,
                    target_t,
                    pos_map_t,
                    defocus_t,
                    depth_t,
                )

            return_dict["source"] = source_t
            return_dict["target"] = target_t
            return_dict["pos_map"] = pos_map_t
            return_dict["bokeh_strength_map"] = defocus_t
            return_dict["pseudo_defocus_map"] = return_dict["bokeh_strength_map"]
            if depth_t is None:
                return_dict.pop("depth_map", None)
            else:
                return_dict["depth_map"] = depth_t

        return return_dict


class EBB(Dataset):
    def __init__(self, data_path: Union[str, Path], mode: Mode, device: str = "cuda"):
        self._data_path: Path = (
            Path(data_path) if isinstance(data_path, str) else data_path
        )
        self._mode = mode
        self._device = device

        self._mode_dir = self._data_path.joinpath(self._mode.value)
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Mode directory {self._mode_dir} does not exist!")

        self._image_list = sorted(
            [Path(f) for f in self._mode_dir.joinpath("in").glob("*.jpg")],
            key=lambda x: x.stem,
        )
        self._gt_list = sorted(
            [Path(f) for f in self._mode_dir.joinpath("gt").glob("*.jpg")],
            key=lambda x: x.stem,
        )

        if len(self._image_list) == 0:
            raise FileNotFoundError(
                f"No images found in {self._mode_dir.joinpath('in')}!"
            )
        if len(self._gt_list) == 0:
            raise FileNotFoundError(
                f"No images found in {self._mode_dir.joinpath('gt')}!"
            )
        if len(self._image_list) != len(self._gt_list):
            raise ValueError(
                f"Number of images and gt images do not match in {self._mode_dir}, "
                f"{len(self._image_list)} inputs != {len(self._gt_list)} gts!"
            )

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index: int):
        source = crop_to_divisible(Image.open(self._image_list[index]), divisor=4)
        target = crop_to_divisible(Image.open(self._gt_list[index]), divisor=4)

        aperture_embedding = calculate_aperture_embedding(2.0)
        maps = generate_maps(source, aperture_embedding, target=target)

        return build_input_dict(
            maps, aperture_embedding, self._image_list[index].name, self._device
        )
