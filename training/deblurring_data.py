"""Stage I training mixture (Sec. 3.1, Appendix B): (A) DPDD, (B) RealBokeh pairs reversed into deblurring pairs
(the CSV manifest of the 1,000 sharpest scenes), (C) LFDOF and (D) DIV2K with synthetic spatially varying blur, drawn
per sample with the probabilities of configs/deblurring_data.yaml. Every sample is {"source": blurred, "target": sharp},
both 3 x H x W in [0, 1].

Augmentation, shared by all sources: random rescaling by [0.4, 1], a crop of the training resolution and a horizontal
flip with probability 0.5. The crop size is drawn from the list of training resolutions (one entry here) at each of
these steps, as in the released training, which keeps its random stream (training runs are reproducible only with
the same sequence of random draws)."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from flashrefocus import paths

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DIRECTIONS = ["top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left", "top_left_to_bottom_right",
              "top_right_to_bottom_left", "center_to_edge", "edge_to_center", "uniform"]


def load_config(path):
    """YAML with ${FLASHREFOCUS_REPO} and ${FLASHREFOCUS_DATA} expanded (see flashrefocus/paths.py)."""
    import yaml
    text = Path(path).read_text()
    for key, value in {"FLASHREFOCUS_REPO": str(paths.REPO), "FLASHREFOCUS_DATA": str(paths.DATA)}.items():
        text = text.replace("${" + key + "}", value)
    return yaml.safe_load(text)


def load_rgb(path):
    return TF.to_tensor(Image.open(path).convert("RGB"))


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def resize(t, size):
    return TF.resize(t, list(size), interpolation=InterpolationMode.BILINEAR, antialias=True)


def crop_to_multiple(t, multiple=8):
    h, w = t.shape[-2:]
    nh, nw = h - h % multiple, w - w % multiple
    return t if (nh, nw) == (h, w) else TF.crop(t, (h - nh) // 2, (w - nw) // 2, nh, nw)


def augment(tensors, sizes, scale_range=(0.4, 1.0), hflip_p=0.5):
    """Same random rescale, crop and horizontal flip for all (c, h, w) tensors; sizes: training resolutions (h, w)."""
    crop_h, crop_w = random.choice(sizes)
    scale = random.uniform(*scale_range)
    h, w = tensors[0].shape[-2:]
    tensors = [resize(t, (max(crop_h, int(round(h * scale))), max(crop_w, int(round(w * scale))))) for t in tensors]
    crop_h, crop_w = random.choice(sizes)
    h, w = tensors[0].shape[-2:]
    if h < crop_h or w < crop_w:
        tensors = [resize(t, (max(crop_h, h), max(crop_w, w))) for t in tensors]
        h, w = tensors[0].shape[-2:]
    top, left = random.randint(0, h - crop_h), random.randint(0, w - crop_w)
    tensors = [TF.crop(t, top, left, crop_h, crop_w) for t in tensors]
    if random.random() < hflip_p:
        tensors = [TF.hflip(t) for t in tensors]
    random.choice(sizes)  # unused draw: the released training drew a size for an optional center crop, here off
    return [crop_to_multiple(t) for t in tensors]


class PairDataset(Dataset):
    """Aligned (blurred, sharp) image pairs."""

    def __init__(self, pairs, sizes):
        if not pairs:
            raise FileNotFoundError("no image pairs found")
        self.pairs, self.sizes = pairs, sizes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = augment([load_rgb(p) for p in self.pairs[index]], self.sizes)
        return {"source": source, "target": target}


def dpdd_pairs(root):
    """DPDD training split: train_c/source and train_c/target, matched by sorted order."""
    return list(zip(list_images(Path(root) / "train_c" / "source"), list_images(Path(root) / "train_c" / "target")))


def manifest_pairs(csv_path, data_root):
    """CSV with source,target columns; relative paths are taken under the data root, else next to the CSV."""
    csv_path = Path(csv_path)
    resolve = lambda v: Path(v) if Path(v).is_absolute() else (
        data_root / v if (data_root / v).exists() else csv_path.parent / v)
    with csv_path.open(newline="", encoding="utf-8") as f:
        return [(resolve(row["source"]), resolve(row["target"])) for row in csv.DictReader(f)]


def lfdof_pairs(root):
    """LFDOF train_data: every refocused slice input/<scene>/* with its all-in-focus ground_truth/<scene>.*."""
    split = Path(root) / "train_data"
    targets = {p.stem: p for p in list_images(split / "ground_truth")}
    return [(src, targets[scene.name]) for scene in sorted(p for p in (split / "input").iterdir() if p.is_dir())
            if scene.name in targets for src in list_images(scene)]


# ---- synthetic spatially varying blur (Eq. 7) ----

def disk_blur(image, radius):
    radius = max(float(radius), 0.5)
    r = max(1, int(round(radius)))
    coords = torch.arange(-r, r + 1, dtype=image.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = ((xx.square() + yy.square()) <= radius * radius).to(image.dtype)
    return _filter(image, kernel / kernel.sum().clamp_min(1e-8))


def gaussian_blur(image, radius):
    size = max(3, int(round(radius * 2.0)) * 2 + 1)
    sigma = max(float(radius) * 0.5, 0.1)
    return TF.gaussian_blur(image, kernel_size=[size, size], sigma=[sigma, sigma])


def motion_blur(image, radius):
    """Line kernel of length 2r + 1 at a random angle in [0, 180)."""
    angle = random.uniform(0.0, 180.0)
    r = max(1, int(round(radius)))
    coords = torch.arange(2 * r + 1, dtype=image.dtype) - r
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    theta = torch.tensor(angle * np.pi / 180.0, dtype=image.dtype)
    along, across = xx * torch.cos(theta) + yy * torch.sin(theta), -xx * torch.sin(theta) + yy * torch.cos(theta)
    kernel = ((along.abs() <= r) & (across.abs() <= 0.5)).to(image.dtype)
    if kernel.sum() < 1:
        kernel[r, r] = 1.0
    return _filter(image, kernel / kernel.sum().clamp_min(1e-8))


def _filter(image, kernel):
    kernel = kernel.expand(image.size(0), 1, -1, -1)
    pad = kernel.shape[-1] // 2
    return F.conv2d(F.pad(image.unsqueeze(0), (pad,) * 4, mode="reflect"), kernel, groups=image.size(0)).squeeze(0)


def gradient_map(h, w, direction, dtype):
    """M_s: a linear or radial ramp in [0, 1] in one of eight directions, or 1 everywhere ("uniform")."""
    if direction == "uniform":
        return torch.ones((1, h, w), dtype=dtype)
    y = torch.linspace(0.0, 1.0, h, dtype=dtype).view(1, h, 1).expand(1, h, w)
    x = torch.linspace(0.0, 1.0, w, dtype=dtype).view(1, 1, w).expand(1, h, w)
    radial = torch.sqrt((x - 0.5).square() + (y - 0.5).square())
    m = {"top_to_bottom": y, "bottom_to_top": 1.0 - y, "left_to_right": x, "right_to_left": 1.0 - x,
         "top_left_to_bottom_right": 0.5 * (x + y), "top_right_to_bottom_left": 0.5 * ((1.0 - x) + y),
         "center_to_edge": radial, "edge_to_center": 1.0 - radial}[direction]
    lo, hi = m.amin(), m.amax()
    return torch.zeros_like(m) if (hi - lo) < 1e-8 else ((m - lo) / (hi - lo)).clamp(0.0, 1.0)


def synthetic_blur(sharp, radius_min, radius_max):
    """Eq. 7: one kernel type (disk, Gaussian or motion) at radii r_k = min, mid, max, blended per pixel with
    weights w_k that interpolate the radii along M_s."""
    random.uniform(radius_min, radius_max)  # unused draw, kept for the random stream of the released training
    blur = {"disk": disk_blur, "gaussian": gaussian_blur, "motion": motion_blur}[
        random.choice(["disk", "gaussian", "motion"])]
    m_s = gradient_map(*sharp.shape[-2:], random.choice(DIRECTIONS), sharp.dtype)
    radii = [radius_min, 0.5 * (radius_min + radius_max), radius_max]
    blurred = torch.stack([blur(sharp, r) for r in radii], dim=0)
    centers = torch.tensor([0.0, 0.5, 1.0], dtype=sharp.dtype).view(3, 1, 1, 1)
    w = (1.0 - (m_s.unsqueeze(0) - centers).abs() * 2.0).clamp_min(0.0)
    w = w / w.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return (w * blurred).sum(dim=0).clamp(0.0, 1.0)


class SyntheticBlurDataset(Dataset):
    """Sharp images (center 60% of each DIV2K image), augmented, then blurred with synthetic_blur."""

    def __init__(self, root, sizes, radius_min=1.0, radius_max=10.0, center_ratio=0.6):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise FileNotFoundError(f"no images in {root}")
        self.sizes, self.radius_min, self.radius_max, self.center_ratio = sizes, radius_min, radius_max, center_ratio

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sharp = load_rgb(self.paths[index])
        h, w = sharp.shape[-2:]
        sharp = TF.center_crop(sharp, [max(1, int(round(h * self.center_ratio))),
                                       max(1, int(round(w * self.center_ratio)))])
        (sharp,) = augment([sharp], self.sizes)
        return {"source": synthetic_blur(sharp, self.radius_min, self.radius_max), "target": sharp}


class MixedDeblurringDataset(Dataset):
    """Draws a source with the configured probabilities, then a random sample of it."""

    def __init__(self, config, resolution):
        sizes = [(resolution, resolution)]
        data_root = paths.DATA
        c = config
        self.sources = [("dpdd", PairDataset(dpdd_pairs(c["dpdd"]["root"]), sizes), c["dpdd"]["p"]),
                        ("realbokeh", PairDataset(manifest_pairs(c["realbokeh"]["manifest"], data_root), sizes),
                         c["realbokeh"]["p"]),
                        ("lfdof", PairDataset(lfdof_pairs(c["lfdof"]["root"]), sizes), c["lfdof"]["p"]),
                        ("synthetic", SyntheticBlurDataset(c["synthetic"]["root"], sizes, c["synthetic"]["radius_min"],
                                                           c["synthetic"]["radius_max"]), c["synthetic"]["p"])]
        total = sum(p for _, _, p in self.sources)
        self.weights = [p / total for _, _, p in self.sources]
        self.length = max(len(d) for _, d, _ in self.sources)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        _, dataset, _ = self.sources[random.choices(range(len(self.sources)), weights=self.weights, k=1)[0]]
        return dataset[random.randrange(len(dataset))]


def build_deblurring_dataset(config_path, resolution):
    return MixedDeblurringDataset(load_config(config_path), resolution)
