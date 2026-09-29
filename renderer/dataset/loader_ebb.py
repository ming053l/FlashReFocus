import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union

from dataset.depth_processing import infer_depth_mode, load_depth_map
from dataset.util import (
    Mode,
    calculate_aperture_embedding,
    generate_maps,
    build_input_dict,
    crop_to_divisible,
    make_point_focus_map,
)


def _image_sort_key(path: Path):
    return int(path.stem) if path.stem.isdigit() else path.stem


def ebb_image_paths(data_path: Union[str, Path], mode: Mode, folder: str):
    data_path = Path(data_path) if isinstance(data_path, str) else data_path
    image_dir = data_path / mode.value / folder
    if not image_dir.is_dir():
        raise FileNotFoundError(f"EBB {folder} directory not found: {image_dir}")
    return sorted(image_dir.glob("*.jpg"), key=_image_sort_key)


def load_ebb_focus_annotations(path: Union[str, Path]):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    annotations = data.get("annotations", data)
    if not isinstance(annotations, dict):
        raise ValueError(f"Unsupported annotation format: {path}")
    return {str(key): value for key, value in annotations.items()}


class EBB(Dataset):

    def __init__(self, data_path: Union[str, Path], mode: Mode, device: str = "cuda"):
        self._data_path: Path = (
            Path(data_path) if isinstance(data_path, str) else data_path
        )
        self._mode = mode
        self._device = device

        # initialize dir
        self._mode_dir = self._data_path.joinpath(self._mode.value)
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Mode directory {self._mode_dir} does not exist!")

        # Initialize input and gt lists
        self._image_list = sorted(
            [Path(f) for f in self._mode_dir.joinpath("original").glob("*.jpg")],
            key=_image_sort_key,
        )
        self._gt_list = sorted(
            [Path(f) for f in self._mode_dir.joinpath("bokeh").glob("*.jpg")],
            key=_image_sort_key,
        )

        if len(self._image_list) == 0:
            raise FileNotFoundError(
                f"No images found in {self._mode_dir.joinpath('original')}!"
            )
        if len(self._gt_list) == 0:
            raise FileNotFoundError(
                f"No images found in {self._mode_dir.joinpath('bokeh')}!"
            )
        if len(self._image_list) != len(self._gt_list):
            raise ValueError(
                f"Number of images and gt images do not match in {self._mode_dir}, {len(self._image_list)} inputs != {len(self._gt_list)} gts!"
            )

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index: int):
        source = crop_to_divisible(Image.open(self._image_list[index]), divisor=4)
        target = crop_to_divisible(Image.open(self._gt_list[index]), divisor=4)

        aperture_embedding = calculate_aperture_embedding(2.0)

        maps = generate_maps(source, aperture_embedding, target=target)

        return_dict = build_input_dict(
            maps, aperture_embedding, self._image_list[index].name, self._device
        )
        width, height = source.size
        return_dict["focus_map"] = make_point_focus_map(
            width,
            height,
            (width - 1) / 2.0,
            (height - 1) / 2.0,
            radius=max(height, width),
        ).to(self._device)

        return return_dict


class EBBFocalDepth(Dataset):
    """
    EBB loader for DMA-v3 focal-depth evaluation.

    It reads EBB original/bokeh pairs, a focus annotation JSON, and cached depth
    maps. The returned bokeh_strength_map is abs(depth - focal_depth), matching
    RealBokehFocalDepth's contract for model_dma_v3.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        mode: Mode,
        annotation_file: Union[str, Path] = None,
        device: str = "cpu",
        depth_map_dir: str = "da3_depth",
        depth_mode: str = "auto",
        clip_percentiles: tuple[float, float] = (0.0, 100.0),
        normalize_depth_map: bool = False,
        depth_normalize_scale: float = 20.0,
        inverse_depth: bool = False,
        focus_map_type: str = "cosine",
        focus_map_gaussian_sigma: float | None = None,
        focus_map_gaussian_sigma_scale: float = 3.0,
    ):
        self._data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self._mode = mode
        self._mode_dir = self._data_path / self._mode.value
        self._device = device
        self.depth_map_dir = depth_map_dir
        self._depth_dir = self._mode_dir / depth_map_dir
        self.depth_mode = str(depth_mode).lower()
        self.effective_depth_mode = infer_depth_mode(
            self.depth_mode,
            depth_map_dir,
            normalize_depth_map=normalize_depth_map,
            inverse_depth=inverse_depth,
        )
        self.clip_percentiles = clip_percentiles
        self.normalize_depth_map = normalize_depth_map
        self.depth_normalize_scale = float(depth_normalize_scale)
        self.inverse_depth = bool(inverse_depth)
        self.focus_map_type = str(focus_map_type).lower()
        if self.focus_map_type == "gaussain":
            self.focus_map_type = "gaussian"
        self.focus_map_gaussian_sigma = focus_map_gaussian_sigma
        self.focus_map_gaussian_sigma_scale = float(focus_map_gaussian_sigma_scale)
        self._annotation_file = (
            Path(annotation_file)
            if annotation_file is not None
            else self._mode_dir / "focus_points_f2.json"
        )

        if not self._mode_dir.is_dir():
            raise FileNotFoundError(f"EBB split directory not found: {self._mode_dir}")
        if not self._depth_dir.is_dir():
            raise FileNotFoundError(f"EBB depth directory not found: {self._depth_dir}")
        if not self._annotation_file.exists():
            raise FileNotFoundError(
                f"EBB focus annotation not found: {self._annotation_file}"
            )
        if self.depth_normalize_scale <= 0:
            raise ValueError(
                "depth_normalize_scale must be > 0, got "
                f"{self.depth_normalize_scale}"
            )
        if self.focus_map_type not in {"cosine", "gaussian"}:
            raise ValueError('focus_map_type must be "cosine" or "gaussian".')
        if self.focus_map_gaussian_sigma is not None:
            self.focus_map_gaussian_sigma = float(self.focus_map_gaussian_sigma)
            if self.focus_map_gaussian_sigma <= 0:
                raise ValueError("focus_map_gaussian_sigma must be > 0.")
        if self.focus_map_gaussian_sigma_scale <= 0:
            raise ValueError("focus_map_gaussian_sigma_scale must be > 0.")

        annotations = load_ebb_focus_annotations(self._annotation_file)
        bokeh_dir = self._mode_dir / "bokeh"
        depth_suffixes = (".tiff", ".tif", ".png")
        if self.normalize_depth_map:
            depth_suffixes = depth_suffixes + (".JPG", ".jpg")
        self._sample_list = []
        for source_path in ebb_image_paths(self._data_path, self._mode, "original"):
            image_id = source_path.stem
            if image_id not in annotations:
                continue
            target_path = bokeh_dir / source_path.name
            depth_path = None
            for suffix in depth_suffixes:
                candidate = self._depth_dir / f"{image_id}{suffix}"
                if candidate.exists():
                    depth_path = candidate
                    break
            if not target_path.exists():
                raise FileNotFoundError(f"EBB bokeh image not found: {target_path}")
            if depth_path is None:
                raise FileNotFoundError(
                    f"EBB depth map not found for {image_id} in {self._depth_dir}"
                )
            self._sample_list.append(
                {
                    "image_id": image_id,
                    "source_path": source_path,
                    "target_path": target_path,
                    "depth_path": depth_path,
                    "annotation": annotations[image_id],
                }
            )
        if not self._sample_list:
            raise ValueError(
                f"No annotated EBB samples found in {self._annotation_file}"
            )
        print(
            "EBB focal-depth dataloader initialized with "
            f"{len(self._sample_list)} samples, "
            f"depth_dir={depth_map_dir}, depth_mode={self.effective_depth_mode}"
        )

    def __len__(self):
        return len(self._sample_list)

    @staticmethod
    def _focal_pixel(annotation: dict, depth_map: torch.Tensor) -> tuple[int, int]:
        _, height, width = depth_map.shape
        ann_width = float(annotation.get("width", width))
        ann_height = float(annotation.get("height", height))
        x = float(annotation["x"]) * width / ann_width
        y = float(annotation["y"]) * height / ann_height
        x_idx = min(max(int(round(x)), 0), width - 1)
        y_idx = min(max(int(round(y)), 0), height - 1)
        return x_idx, y_idx

    def __getitem__(self, index: int):
        sample = self._sample_list[index]
        source = Image.open(sample["source_path"]).convert("RGB")
        target = Image.open(sample["target_path"]).convert("RGB")
        if target.size != source.size:
            target = target.resize(source.size, Image.Resampling.BILINEAR)

        aperture_embedding = calculate_aperture_embedding(2.0)
        maps = generate_maps(source, aperture_embedding, target=target)
        return_dict = build_input_dict(
            maps,
            aperture_embedding,
            sample["source_path"].name,
            self._device,
        )

        depth_map = load_depth_map(
            sample["depth_path"],
            source.size,
            depth_mode=self.depth_mode,
            depth_map_dir=self.depth_map_dir,
            clip_percentiles=self.clip_percentiles,
            depth_normalize_scale=self.depth_normalize_scale,
            normalize_depth_map=self.normalize_depth_map,
            inverse_depth=self.inverse_depth,
        )
        x_idx, y_idx = self._focal_pixel(sample["annotation"], depth_map)
        focal_depth = depth_map[:, y_idx, x_idx].view(1, 1, 1)
        defocus_map = (depth_map - focal_depth).abs()
        focus_map = make_point_focus_map(
            depth_map.shape[-1],
            depth_map.shape[-2],
            x_idx,
            y_idx,
            radius=max(depth_map.shape[-2:]),
            map_type=self.focus_map_type,
            gaussian_sigma=self.focus_map_gaussian_sigma,
            gaussian_sigma_scale=self.focus_map_gaussian_sigma_scale,
        )

        return_dict["bokeh_strength_map"] = defocus_map.to(self._device)
        return_dict["pseudo_defocus_map"] = return_dict["bokeh_strength_map"]
        return_dict["depth_map"] = depth_map.to(self._device)
        return_dict["focus_map"] = focus_map.to(self._device)
        return_dict["focal_depth"] = depth_map[:, y_idx, x_idx].to(self._device)
        return_dict["disp_focus"] = return_dict["focal_depth"]
        return_dict["focus_point"] = torch.tensor(
            (x_idx, y_idx),
            dtype=torch.float32,
            device=self._device,
        )
        return_dict["scene_id"] = sample["image_id"]
        return return_dict
