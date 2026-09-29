from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from json import load
from typing import Union, Optional
import random
from torchvision.transforms.functional import to_tensor

from .util import (
    Mode,
    calculate_aperture_embedding,
    generate_maps,
    build_input_dict,
    crop_to_divisible,
)


class RealBokeh(Dataset):
    """
    Dataset class for the RealBokeh dataset.
        :param data_path: Path to the dataset directory with train/val/test subdirectories. Only test is needed for testing!
        :param mode: one of the modes defined in the Mode enum.
        :param binary_bokeh: whether to use binary bokeh strength of F22.0 and F2.0, RealBokeh_bin in the paper.
        :param defocus_deblur_mode: If true, reverses input and outputs, RealDefocus in the paper
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        mode: Mode,
        binary_bokeh: bool = False,
        defocus_deblur_mode: bool = False,
        device: str = "cuda",
        challenge: bool = False,
        crop_size: Optional[int] = None,
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
        scale_p: float = 0.0,
        include_source: bool = False,
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
        self.challenge = challenge  # Activate if used in the context of a challenge.
        # Iterate over individual samples of each scene for validation and test, over scenes where a random sample of a
        self._iteration_mode = "sample"

        self._binary_bokeh = binary_bokeh
        if self._binary_bokeh:
            print("Using binary bokeh strength of F22.0 and F2.0!")
            self._iteration_mode = "scene"

        # initialize dir
        self._mode_dir = self._data_path.joinpath(self._mode.value)
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Mode directory {self._mode_dir} does not exist!")

        # load metadata list
        self._scene_list = sorted(
            [load(open(f)) for f in self._mode_dir.joinpath("metadata").glob("*.json")],
            key=lambda x: x["id"],
        )
        if len(self._scene_list) == 0:
            raise FileNotFoundError(
                f"No metadata files found in {self._mode_dir.joinpath('metadata')}!"
            )

        self.include_source = include_source and self._mode == Mode.TRAIN
        self._sample_list = []
        for scene_id, metadata in enumerate(self._scene_list):
            target_images, _ = self._targets_with_source(metadata, self.include_source)
            for sample_id in range(len(target_images)):
                self._sample_list.append((scene_id, sample_id))

        self._device = device
        self._crop_size = crop_size
        self._hflip_p = hflip_p
        self._vflip_p = vflip_p
        self._scale_p = scale_p

        print(
            f"RealBokeh Dataloader initialized in {mode} mode with {len(self._scene_list)} scenes and {len(self._sample_list)} samples!"
        )

    def __len__(self):
        # Since for Validation and Test we are iterating over every sample of every scene, we need to return the
        # length of the sample list, otherwise we return the length of the scene list.
        if self._iteration_mode == "sample":
            return len(self._sample_list)
        else:
            return len(self._scene_list)

    def __getitem__(self, index: int):
        if self._iteration_mode == "sample":
            metadata: dict = self._scene_list[self._sample_list[index][0]]
            target_images, target_avs = self._targets_with_source(
                metadata, self.include_source
            )
            sample_idx = self._sample_list[index][1]

            target = (
                Image.open(self._mode_dir.joinpath(target_images[sample_idx]))
                if not self.challenge
                else None
            )
            tgt_av = target_avs[sample_idx]

            target_name = Path(target_images[sample_idx]).stem
        else:
            metadata: dict = self._scene_list[index]
            target_images, target_avs = self._targets_with_source(
                metadata, self.include_source
            )

            target_idx = 0
            target = (
                Image.open(self._mode_dir.joinpath(target_images[target_idx]))
                if not self.challenge
                else None
            )
            tgt_av = target_avs[target_idx]

            target_name = Path(target_images[target_idx]).stem

        source = Image.open(self._mode_dir.joinpath(metadata["source_image"]))
        depth_path = self._mode_dir.joinpath(
            self._resolve_depth_path(metadata["source_image"])
        )
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        depth = Image.open(depth_path).convert("L")

        # Training-only augmentation.
        if self._mode == Mode.TRAIN and not self.challenge:
            if self._scale_p > 0.0 and random.random() < self._scale_p:
                source, target, depth = self._random_scale_triplet(
                    source, target, depth, min_size=self._crop_size
                )
            if self._crop_size is not None:
                source, target, depth = self._random_crop_triplet(
                    source, target, depth, self._crop_size
                )
            if self._hflip_p > 0.0 and random.random() < self._hflip_p:
                source, target, depth = self._hflip_triplet(source, target, depth)
            if self._vflip_p > 0.0 and random.random() < self._vflip_p:
                source, target, depth = self._vflip_triplet(source, target, depth)

        if self.defocus_deblur_mode:
            tmp = source
            source = target
            target = tmp

        # Embed metadata and generate auxiliary maps

        aperture_embedding = calculate_aperture_embedding(tgt_av)

        maps = generate_maps(source, aperture_embedding, target=target)

        return_dict = build_input_dict(
            maps, aperture_embedding, target_name, self._device
        )
        return_dict["depth_map"] = to_tensor(depth).to(self._device)

        return return_dict

    @staticmethod
    def _random_crop_pair(source: Image.Image, target: Image.Image, crop_size: int):
        w, h = source.size
        if w < crop_size or h < crop_size:
            raise ValueError(
                f"Crop size {crop_size} is larger than image size {(w, h)}."
            )
        if target is not None and target.size != (w, h):
            raise ValueError("Source and target must have the same size for cropping.")
        left = random.randint(0, w - crop_size)
        top = random.randint(0, h - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        source = source.crop((left, top, right, bottom))
        if target is not None:
            target = target.crop((left, top, right, bottom))
        return source, target

    @staticmethod
    def _hflip_pair(source: Image.Image, target: Image.Image):
        source = source.transpose(Image.FLIP_LEFT_RIGHT)
        if target is not None:
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        return source, target

    @staticmethod
    def _random_crop_triplet(
        source: Image.Image, target: Image.Image, depth: Image.Image, crop_size: int
    ):
        w, h = source.size
        if w < crop_size or h < crop_size:
            raise ValueError(
                f"Crop size {crop_size} is larger than image size {(w, h)}."
            )
        if target is not None and target.size != (w, h):
            raise ValueError("Source and target must have the same size for cropping.")
        if depth.size != (w, h):
            raise ValueError("Source and depth must have the same size for cropping.")

        left = random.randint(0, w - crop_size)
        top = random.randint(0, h - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        source = source.crop((left, top, right, bottom))
        if target is not None:
            target = target.crop((left, top, right, bottom))
        depth = depth.crop((left, top, right, bottom))
        return source, target, depth

    @staticmethod
    def _hflip_triplet(source: Image.Image, target: Image.Image, depth: Image.Image):
        source = source.transpose(Image.FLIP_LEFT_RIGHT)
        if target is not None:
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        return source, target, depth

    @staticmethod
    def _vflip_triplet(source: Image.Image, target: Image.Image, depth: Image.Image):
        source = source.transpose(Image.FLIP_TOP_BOTTOM)
        if target is not None:
            target = target.transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        return source, target, depth

    @staticmethod
    def _random_scale_triplet(
        source: Image.Image,
        target: Image.Image,
        depth: Image.Image,
        scale_range=(0.8, 1.1),
        min_size: int | None = None,
        src_resample=Image.BICUBIC,
        tgt_resample=Image.BICUBIC,
        depth_resample=Image.BILINEAR,
    ):

        w, h = source.size
        if target is not None and target.size != (w, h):
            raise ValueError("Source and target must have the same size for scaling.")
        if depth.size != (w, h):
            raise ValueError("Source and depth must have the same size for scaling.")

        s_min, s_max = scale_range
        if s_min <= 0 or s_max <= 0 or s_min > s_max:
            raise ValueError(f"Invalid scale_range={scale_range}.")

        scale = random.uniform(s_min, s_max)

        if min_size is not None:
            min_scale_needed = min_size / min(w, h)
            if scale < min_scale_needed:
                scale = min_scale_needed

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        source = source.resize((new_w, new_h), resample=src_resample)
        if target is not None:
            target = target.resize((new_w, new_h), resample=tgt_resample)
        depth = depth.resize((new_w, new_h), resample=depth_resample)

        return source, target, depth

    @staticmethod
    def _targets_with_source(metadata: dict, include_source: bool):
        # Keep original target ordering; optionally append source as an additional supervised target.
        target_images = list(metadata["target_images"])
        target_avs = list(metadata["target_avs"])
        if include_source:
            target_images.append(metadata["source_image"])
            target_avs.append(metadata.get("source_av", 22.0))
        return target_images, target_avs

    @staticmethod
    def _resolve_depth_path(source_rel_path: str) -> Path:
        source_rel = Path(source_rel_path)
        if len(source_rel.parts) > 0 and source_rel.parts[0] == "in":
            depth_rel = Path("in_depth").joinpath(*source_rel.parts[1:])
        else:
            depth_rel = Path("in_depth").joinpath(source_rel.name)
        return depth_rel


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

        return return_dict
