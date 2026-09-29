from json import load
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset.depth_processing import infer_depth_mode, load_depth_map
from dataset.loader_dma import (
    _joint_random_crop,
    _joint_random_hflip,
    _joint_random_rotate90,
    _joint_random_scale,
)
from dataset.util import (
    Mode,
    build_input_dict,
    calculate_aperture_embedding,
    generate_maps,
    make_point_focus_map,
)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return load(f)


def _annotation_path_value(annotation: dict, *keys: str) -> Optional[str]:
    for key in keys:
        value = annotation.get(key)
        if value:
            return value
    return None


class RealBokehFocalDepth(Dataset):
    """
    RealBokeh loader that keeps only manually annotated pairs and builds a
    defocus map from relative depth:

        abs(in_depth - in_depth[focal_y, focal_x])

    The returned dict follows the Bokehlicious/DMA loader contract. The computed
    map is assigned to both "bokeh_strength_map" and "pseudo_defocus_map".
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        mode: Mode,
        annotation_file: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        depth_map_dir: str = "in_depth",
        depth_mode: str = "auto",
        clip_percentiles: tuple[float, float] = (0.0, 100.0),
        normalize_depth_map: bool = False,
        depth_normalize_scale: float = 20.0,
        use_augmentation: Optional[bool] = None,
        image_size: Optional[int] = None,
        random_scale_range: tuple[float, float] = (0.4, 1.0),
        random_hflip_p: float = 0.5,
        random_rotate90: bool = True,
        target_apertures: str = "all",
        skip_missing: bool = True,
        focus_map_type: str = "cosine",
        focus_map_gaussian_sigma: Optional[float] = None,
        focus_map_gaussian_sigma_scale: float = 3.0,
        inverse_depth: bool = False,
    ):
        self._data_path = Path(data_path)
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Data directory {self._data_path.absolute()} does not exist!"
            )

        self._mode = mode
        self._mode_dir = self._data_path / self._mode.value
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Mode directory {self._mode_dir} does not exist!")

        self._depth_dir = self._mode_dir / depth_map_dir
        if not self._depth_dir.exists():
            raise FileNotFoundError(
                f"Depth map directory {self._depth_dir} does not exist!"
            )

        self._annotation_file = (
            Path(annotation_file)
            if annotation_file is not None
            else self._mode_dir / "focus_points_f2.json"
        )
        if not self._annotation_file.exists():
            raise FileNotFoundError(
                f"Focus annotation file {self._annotation_file} does not exist!"
            )

        self._device = device
        self.depth_map_dir = depth_map_dir
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
        self.use_augmentation = (
            self._mode == Mode.TRAIN if use_augmentation is None else use_augmentation
        )
        self.image_size = image_size
        self.random_scale_range = random_scale_range
        self.random_hflip_p = float(random_hflip_p)
        self.random_rotate90 = bool(random_rotate90)
        self.focus_map_type = str(focus_map_type).lower()
        if self.focus_map_type == "gaussain":
            self.focus_map_type = "gaussian"
        self.focus_map_gaussian_sigma = focus_map_gaussian_sigma
        self.focus_map_gaussian_sigma_scale = float(focus_map_gaussian_sigma_scale)
        if target_apertures not in {"all", "annotated"}:
            raise ValueError('target_apertures must be "all" or "annotated".')
        self.target_apertures = target_apertures
        if self.use_augmentation and self.image_size is None:
            raise ValueError("image_size must be set when use_augmentation=True")
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

        metadata_dir = self._mode_dir / "metadata"
        self._metadata_by_id = {
            str(metadata["id"]): metadata
            for metadata in (
                _load_json(path) for path in sorted(metadata_dir.glob("*.json"))
            )
        }
        if not self._metadata_by_id:
            raise FileNotFoundError(f"No metadata files found in {metadata_dir}!")

        annotations_blob = _load_json(self._annotation_file)
        annotations = annotations_blob.get("annotations", annotations_blob)
        if not isinstance(annotations, dict):
            raise ValueError(
                "Focus annotation file must contain an 'annotations' dict or be a dict."
            )

        self._sample_list = self._build_sample_list(annotations, skip_missing)
        if not self._sample_list:
            raise ValueError(
                f"No usable annotated samples found in {self._annotation_file}."
            )

        print(
            f"RealBokeh focal-depth Dataloader initialized in {mode} mode with "
            f"{len(self._sample_list)} annotated samples, "
            f"depth_dir={depth_map_dir}, depth_mode={self.effective_depth_mode}!"
        )

    def _build_sample_list(self, annotations: dict, skip_missing: bool) -> list[dict]:
        samples = []
        for scene_id, annotation in sorted(
            annotations.items(),
            key=lambda item: (int(item[0]) if str(item[0]).isdigit() else str(item[0])),
        ):
            metadata = self._metadata_by_id.get(str(scene_id))
            if metadata is None:
                if skip_missing:
                    continue
                raise FileNotFoundError(
                    f"No metadata found for annotated scene {scene_id}"
                )

            source_rel = _annotation_path_value(annotation, "f22_path", "source_path")
            if source_rel is None:
                source_rel = f"{self._mode.value}/{metadata['source_image']}"

            source_path = self._resolve_dataset_path(source_rel)
            depth_path = self._depth_path(metadata, source_path)
            if not source_path.exists() or not depth_path.exists():
                if skip_missing:
                    continue
                missing = [
                    str(path) for path in (source_path, depth_path) if not path.exists()
                ]
                raise FileNotFoundError(
                    "Missing annotated sample files: " + ", ".join(missing)
                )

            for target_rel, target_av in self._target_pairs(metadata, annotation):
                target_path = self._resolve_dataset_path(target_rel)
                if not target_path.exists():
                    if skip_missing:
                        continue
                    raise FileNotFoundError(
                        f"Missing annotated target file: {target_path}"
                    )
                samples.append(
                    {
                        "scene_id": str(scene_id),
                        "metadata": metadata,
                        "annotation": annotation,
                        "source_path": source_path,
                        "target_path": target_path,
                        "target_av": float(target_av),
                        "depth_path": depth_path,
                    }
                )
        return samples

    def _target_pairs(
        self, metadata: dict, annotation: dict
    ) -> list[tuple[str, float]]:
        if self.target_apertures == "all":
            return [
                (f"{self._mode.value}/{target_path}", target_av)
                for target_path, target_av in zip(
                    metadata["target_images"], metadata["target_avs"]
                )
            ]

        target_rel = _annotation_path_value(
            annotation, "f2_path", "annotation_path", "target_path"
        )
        if target_rel is None:
            target_rel = f"{self._mode.value}/{metadata['target_images'][0]}"
        return [(target_rel, float(annotation.get("annotation_aperture", 2.0)))]

    def _resolve_dataset_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == self._mode.value:
            return self._data_path / path
        return self._mode_dir / path

    def _depth_path(self, metadata: dict, source_path: Path) -> Path:
        source_stem = Path(metadata.get("source_image", source_path.name)).stem
        for suffix in (".tiff", ".tif", ".png", ".JPG", ".jpg"):
            candidate = self._depth_dir / f"{source_stem}{suffix}"
            if candidate.exists():
                return candidate
        return self._depth_dir / source_path.name

    def __len__(self):
        return len(self._sample_list)

    def _focal_pixel(
        self, annotation: dict, depth_map: torch.Tensor
    ) -> tuple[int, int]:
        _, height, width = depth_map.shape
        ann_width = float(annotation.get("width", width))
        ann_height = float(annotation.get("height", height))
        x = float(annotation["x"]) * width / ann_width
        y = float(annotation["y"]) * height / ann_height
        x_idx = min(max(int(round(x)), 0), width - 1)
        y_idx = min(max(int(round(y)), 0), height - 1)
        return x_idx, y_idx

    def _build_focal_defocus_map(
        self,
        depth_path: Path,
        source_size: tuple[int, int],
        annotation: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        depth_map = load_depth_map(
            depth_path,
            source_size,
            depth_mode=self.depth_mode,
            depth_map_dir=self.depth_map_dir,
            clip_percentiles=self.clip_percentiles,
            depth_normalize_scale=self.depth_normalize_scale,
            normalize_depth_map=self.normalize_depth_map,
            inverse_depth=self.inverse_depth,
        )
        x_idx, y_idx = self._focal_pixel(annotation, depth_map)
        focal_depth = depth_map[:, y_idx, x_idx].view(1, 1, 1)
        defocus_map = (depth_map - focal_depth).abs()
        return defocus_map, depth_map, (x_idx, y_idx)

    def __getitem__(self, index: int):
        sample = self._sample_list[index]
        # metadata = sample["metadata"]
        annotation = sample["annotation"]

        source = Image.open(sample["source_path"]).convert("RGB")
        target = Image.open(sample["target_path"]).convert("RGB")
        target_name = sample["target_path"].stem
        target_av = sample["target_av"]

        aperture_embedding = calculate_aperture_embedding(target_av)
        maps = generate_maps(source, aperture_embedding, target=target)
        return_dict = build_input_dict(
            maps, aperture_embedding, target_name, self._device
        )

        defocus_map, depth_map, focal_pixel = self._build_focal_defocus_map(
            sample["depth_path"], source.size, annotation
        )
        return_dict["bokeh_strength_map"] = defocus_map.to(self._device)
        return_dict["pseudo_defocus_map"] = return_dict["bokeh_strength_map"]
        return_dict["depth_map"] = depth_map.to(self._device)
        return_dict["focal_depth"] = depth_map[:, focal_pixel[1], focal_pixel[0]].to(
            self._device
        )
        return_dict["disp_focus"] = return_dict["focal_depth"]
        return_dict["focus_point"] = torch.tensor(
            focal_pixel, dtype=torch.float32, device=self._device
        )
        return_dict["focus_map"] = make_point_focus_map(
            depth_map.shape[-1],
            depth_map.shape[-2],
            focal_pixel[0],
            focal_pixel[1],
            radius=max(depth_map.shape[-2:]),
            map_type=self.focus_map_type,
            gaussian_sigma=self.focus_map_gaussian_sigma,
            gaussian_sigma_scale=self.focus_map_gaussian_sigma_scale,
        ).to(self._device)
        return_dict["scene_id"] = sample["scene_id"]

        if self.use_augmentation:
            (
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
                focus_map_t,
            ) = _joint_random_scale(
                return_dict["source"],
                return_dict["target"],
                return_dict["pos_map"],
                return_dict["bokeh_strength_map"],
                return_dict["depth_map"],
                self.image_size,
                self.random_scale_range,
                focus_map=return_dict["focus_map"],
            )
            (
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
                focus_map_t,
            ) = _joint_random_crop(
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
                self.image_size,
                focus_map=focus_map_t,
            )
            (
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
                focus_map_t,
            ) = _joint_random_hflip(
                source_t,
                target_t,
                pos_map_t,
                defocus_t,
                depth_t,
                self.random_hflip_p,
                focus_map=focus_map_t,
            )
            if self.random_rotate90:
                (
                    source_t,
                    target_t,
                    pos_map_t,
                    defocus_t,
                    depth_t,
                    focus_map_t,
                ) = _joint_random_rotate90(
                    source_t,
                    target_t,
                    pos_map_t,
                    defocus_t,
                    depth_t,
                    focus_map=focus_map_t,
                )

            return_dict["source"] = source_t
            return_dict["target"] = target_t
            return_dict["pos_map"] = pos_map_t
            return_dict["focus_map"] = focus_map_t
            disp_focus = return_dict["disp_focus"].to(
                device=depth_t.device,
                dtype=depth_t.dtype,
            )
            return_dict["bokeh_strength_map"] = (
                depth_t - disp_focus.view(1, 1, 1)
            ).abs()
            return_dict["pseudo_defocus_map"] = return_dict["bokeh_strength_map"]
            return_dict["depth_map"] = depth_t
            flat_focus_idx = torch.argmax(focus_map_t).item()
            focus_w = focus_map_t.shape[-1]
            focus_pixel = (flat_focus_idx % focus_w, flat_focus_idx // focus_w)
            return_dict["focus_point"] = torch.tensor(
                focus_pixel, dtype=torch.float32, device=self._device
            )

        return return_dict
