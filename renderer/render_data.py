"""Training pairs of the renderer (Appendix C.2). Every sample is a dict with the image I_AiF ("source"), the target
I_bokeh ("target"), the strength K ("strength", shape (1,)), the depth D ("depth", 1 x H x W, in [0, 1], 1 = far) and
the focal depth D_f ("focal_depth", shape (1,)).

  HypersimPairs   synthetic multi-focal targets rendered by BokehMe (metadata items: rgb_path, output_path, depth_path,
                  K, disp_focus); D = 1 - min(1 / Z, 1) from the metric depth Z in meters, K = K_BokehMe / 100
  RealBokehPairs  RealBokeh pairs at their annotated focal points; D = 1 - min-max(Depth Anything V2 output), cached by
                  generate_da2_depth.py, and K = 2 / N for a target at f/N, as in Bokehlicious

Training samples are augmented jointly: random rescaling by [0.4, 1], a random crop, a horizontal flip with
probability 0.5 and a rotation by a random multiple of 90 degrees."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode, to_tensor


def augment(tensors, crop_size, scale_range=(0.4, 1.0), hflip_p=0.5):
    """Same random rescale, crop, horizontal flip and 90-degree rotation for all (c, h, w) tensors."""
    resize = lambda ts, size: [TF.resize(t, list(size), interpolation=InterpolationMode.BILINEAR, antialias=True)
                               for t in ts]
    scale = random.uniform(*scale_range)
    h, w = tensors[0].shape[-2:]
    tensors = resize(tensors, (max(crop_size, int(round(h * scale))), max(crop_size, int(round(w * scale)))))
    h, w = tensors[0].shape[-2:]
    top, left = random.randint(0, h - crop_size), random.randint(0, w - crop_size)
    tensors = [TF.crop(t, top, left, crop_size, crop_size) for t in tensors]
    if random.random() < hflip_p:
        tensors = [TF.hflip(t) for t in tensors]
    k = random.randint(0, 3)
    if k:
        tensors = [torch.rot90(t, k, (-2, -1)) for t in tensors]
    return tensors


def load_rgb(path):
    return to_tensor(Image.open(path).convert("RGB"))


def sample(source, target, strength, depth, focal_depth, crop_size=None):
    if crop_size:
        source, target, depth = augment([source, target, depth], crop_size)
    return {"source": source, "target": target, "strength": torch.tensor([float(strength)]), "depth": depth,
            "focal_depth": torch.tensor([float(focal_depth)])}


class HypersimPairs(Dataset):
    def __init__(self, root, split="train", crop_size=None, metadata_name="bokeh_random_metadata.json"):
        split_dir = Path(root) / split
        self.crop_size = crop_size
        self.items = []
        for meta_path in sorted(split_dir.rglob(metadata_name)):
            for item in json.loads(meta_path.read_text()).get("items", []):
                files = [self._resolve(item[k], meta_path, split_dir) for k in ("rgb_path", "output_path", "depth_path")]
                if all(f.is_file() for f in files) and float(item["disp_focus"]) > 1e-6:
                    self.items.append((*files, float(item["K"]), float(item["disp_focus"])))
        if not self.items:
            raise FileNotFoundError(f"no Hypersim samples with {metadata_name} under {split_dir}")

    @staticmethod
    def _resolve(value, meta_path, split_dir):
        path = Path(value)
        if path.is_file():
            return path
        if path.is_absolute():
            if split_dir.name in path.parts:
                remapped = split_dir / Path(*path.parts[path.parts.index(split_dir.name) + 1:])
                if remapped.is_file():
                    return remapped
            return path
        return meta_path.parent / path if (meta_path.parent / path).is_file() else split_dir / path

    @staticmethod
    def _depth(path, focal_metric_depth, min_valid=0.1, eps=1e-6):
        """D = 1 - min(1 / Z, 1), Z in meters (depth PNG in mm); invalid pixels take the farthest valid depth."""
        array = np.asarray(Image.open(path))
        z = torch.from_numpy((array[..., 0] if array.ndim == 3 else array).astype(np.float32)).unsqueeze(0) / 1000.0
        valid = torch.isfinite(z) & (z >= min_valid)
        z = torch.where(valid, z, z[valid].max()) if bool(valid.any()) else z.clamp_min(eps)
        depth = 1.0 - (1.0 / z.clamp_min(eps)).clamp(0.0, 1.0)
        focal_depth = 1.0 - max(0.0, min(1.0 / max(focal_metric_depth, eps), 1.0))
        return depth, focal_depth

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        rgb, bokeh, depth_path, k, focal_metric_depth = self.items[index]
        depth, focal_depth = self._depth(depth_path, focal_metric_depth)
        return sample(load_rgb(rgb), load_rgb(bokeh), max(0.0, min(k / 100.0, 1.0)), depth, focal_depth,
                      self.crop_size)


class RealBokehPairs(Dataset):
    """Every target aperture of the annotated RealBokeh scenes of a split ("train" or "validation")."""

    def __init__(self, root, split="train", crop_size=None, depth_dir="da2_depth"):
        split_dir = Path(root) / split
        self.crop_size = crop_size
        metadata = {str(m["id"]): m for m in (json.loads(p.read_text())
                                              for p in sorted((split_dir / "metadata").glob("*.json")))}
        annotations = json.loads((split_dir / "focus_points_f2.json").read_text())
        annotations = annotations.get("annotations", annotations)
        resolve = lambda rel: Path(root) / rel if Path(rel).parts[0] == split else split_dir / rel
        self.items = []
        for scene, ann in sorted(annotations.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
            meta = metadata.get(str(scene))
            if meta is None:
                continue
            source = resolve(ann.get("f22_path") or ann.get("source_path") or f"{split}/{meta['source_image']}")
            depth = self._depth_path(split_dir / depth_dir, meta, source)
            if not source.exists() or not depth.exists():
                continue
            for target_rel, f_number in zip(meta["target_images"], meta["target_avs"]):
                target = resolve(f"{split}/{target_rel}")
                if target.exists():
                    self.items.append((source, target, 2.0 / float(f_number), depth, ann))

    @staticmethod
    def _depth_path(depth_dir, meta, source):
        stem = Path(meta.get("source_image", source.name)).stem
        for suffix in (".tiff", ".tif", ".png", ".JPG", ".jpg"):
            if (depth_dir / f"{stem}{suffix}").exists():
                return depth_dir / f"{stem}{suffix}"
        return depth_dir / source.name

    @staticmethod
    def _depth(path, size):
        """D = 1 - min-max(DA2 output), resized to the image."""
        d = Image.open(path).convert("F")
        d = to_tensor(d if d.size == size else d.resize(size, Image.BILINEAR)).float()
        lo, hi = d.min(), d.max()
        return 1.0 - ((d - lo) / (hi - lo) if (hi - lo).abs() > 1e-12 else torch.zeros_like(d))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        source_path, target_path, strength, depth_path, ann = self.items[index]
        source = Image.open(source_path).convert("RGB")
        depth = self._depth(depth_path, source.size)
        _, h, w = depth.shape
        x = min(max(int(round(float(ann["x"]) * w / float(ann.get("width", w)))), 0), w - 1)
        y = min(max(int(round(float(ann["y"]) * h / float(ann.get("height", h)))), 0), h - 1)
        return sample(to_tensor(source), load_rgb(target_path), strength, depth, float(depth[0, y, x]), self.crop_size)
