"""ComfyUI nodes for FlashReFocus.

Load the models once, prepare a photograph (Stage I restoration and depth, once per image), build the focus with any
chain of depth edits, and render it (about half a second per edit). Image batches are treated as video frames: depth
and focal depth are smoothed over time.

Types passed between nodes:
  FLASHREFOCUS_MODEL  the Refocuser
  FLASHREFOCUS_SCENE  {"images": [1x3xHxW], "depths": [1xHxW]} per frame, on the CPU, sides multiples of 16
  FLASHREFOCUS_FOCUS  {"depths": [1xHxW], "focal": [float]} per frame: the depth map to render and its focal depth
Coordinates are fractions of the image width and height (0 = left / top, 1 = right / bottom)."""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
CATEGORY = "FlashReFocus"
WEIGHT_REPO = "ming0531/FlashReFocus-weights"
_MODELS = {}


# ---------------------------------------------------------------------------------------------------------------- utils
def _fetch_weights(weights, restoration=True):
    """Download the weights that are missing from `weights` (public backbones and the FlashReFocus checkpoints). Without
    restoration only Depth Anything V2 and the renderer are needed."""
    from huggingface_hub import hf_hub_download, snapshot_download
    w = Path(weights)
    w.mkdir(parents=True, exist_ok=True)
    if not (w / "depth_anything_v2_vitl.pth").exists():
        hf_hub_download("depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth", local_dir=w)
    if not (w / "renderer.pt").exists():
        hf_hub_download(WEIGHT_REPO, "renderer.pt", local_dir=w)
    if not restoration:
        return
    if not (w / "stable-diffusion-2-1-base" / "unet").exists():
        sd = ["model_index.json", "scheduler/scheduler_config.json", "tokenizer/*", "text_encoder/config.json",
              "text_encoder/model.safetensors", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
              "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
        snapshot_download("sd2-community/stable-diffusion-2-1-base", allow_patterns=sd, local_dir=w / "stable-diffusion-2-1-base")
    if not (w / "stable-diffusion-v1-5" / "vae").exists():
        snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
                          allow_patterns=["vae/config.json", "vae/diffusion_pytorch_model.safetensors"],
                          local_dir=w / "stable-diffusion-v1-5")
    for f in ("stage1.pkl", "refiner.pt"):
        if not (w / f).exists():
            hf_hub_download(WEIGHT_REPO, f, local_dir=w)


def _resize_frames(images, long_side):
    """ComfyUI IMAGE (B, H, W, 3) -> B x 3 x h x w with the long side at most `long_side`, sides multiples of 16."""
    _, h, w, _ = images.shape
    s = min(1.0, long_side / max(h, w))
    nh, nw = max(16, int(h * s) // 16 * 16), max(16, int(w * s) // 16 * 16)
    x = images.permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def _to_comfy(frames):
    """List of 1x3xHxW -> IMAGE (B, H, W, 3) on the CPU."""
    return torch.cat([f.float().cpu() for f in frames]).permute(0, 2, 3, 1).clamp(0, 1)


def _depth_preview(depths):
    """Depth maps (1 = far) as images with near bright, as in the paper figures."""
    return torch.cat([(1 - d.float().cpu())[..., None].expand(-1, -1, -1, 3) for d in depths])


def _masks_for(mask, n, h, w):
    """ComfyUI MASK (B, H, W) -> n boolean masks 1 x h x w at the working size (one mask is reused for every frame)."""
    m = mask if mask.dim() == 3 else mask[None]
    m = F.interpolate(m[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0] > 0.5
    return [m[min(i, m.shape[0] - 1)][None] for i in range(n)]


def _smooth(values, alpha):
    """Exponential moving average over frames (alpha = weight of the previous value; 0 = off)."""
    out, prev = [], None
    for v in values:
        prev = v if prev is None or alpha <= 0 else alpha * prev + (1 - alpha) * v
        out.append(prev)
    return out


def _focal_at(depth, x, y):
    from flashrefocus.rendering.depth_edits import depth_at
    _, h, w = depth.shape
    return float(depth_at(depth, x * (w - 1), y * (h - 1)))


def _focus(depths, focal):
    return {"depths": list(depths), "focal": [float(f) for f in focal]}


def _xy(default_x=0.5, default_y=0.5):
    return {"x": ("FLOAT", {"default": default_x, "min": 0.0, "max": 1.0, "step": 0.01}),
            "y": ("FLOAT", {"default": default_y, "min": 0.0, "max": 1.0, "step": 0.01})}


# ------------------------------------------------------------------------------------------------------- load / prepare
class FlashReFocusLoader:
    """Loads Stage I (one-step restoration), Depth Anything V2 and the renderer. With restoration off, only depth and the
    renderer are loaded, for photographs and videos that are already sharp (portrait mode). weights_dir: empty uses
    $FLASHREFOCUS_WEIGHTS or weights/ in the repository; auto_download fetches missing weights from Hugging Face.
    fused_attention runs the renderer's attention as one Triton kernel (same output up to rounding; much less memory
    and faster at high resolution); it falls back to the reference code without Triton. compile_restoration compiles
    Stage I's full-resolution parts with torch.compile (about 35-40% faster restoration; the first image takes one to
    two minutes to compile): "dynamic" serves any image size, "static" is a little faster but recompiles per size."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "restoration": ("BOOLEAN", {"default": True}),
            "detail_weight": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 1.75, "step": 0.05}),
            "weights_dir": ("STRING", {"default": ""}),
            "auto_download": ("BOOLEAN", {"default": True}),
        }, "optional": {                      # optional, so workflows saved before it existed still load
            "fused_attention": ("BOOLEAN", {"default": True}),
            "compile_restoration": (["off", "dynamic", "static"], {"default": "off"}),
        }}

    RETURN_TYPES = ("FLASHREFOCUS_MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, restoration, detail_weight, weights_dir, auto_download, fused_attention=True,
             compile_restoration="off"):
        # empty: $FLASHREFOCUS_WEIGHTS if set, else weights/ in the repository
        weights_dir = weights_dir.strip() or os.environ.get("FLASHREFOCUS_WEIGHTS", str(REPO / "weights"))
        key = (restoration, weights_dir, fused_attention, compile_restoration)
        if key not in _MODELS:
            if auto_download:
                _fetch_weights(weights_dir, restoration)
            os.environ["FLASHREFOCUS_WEIGHTS"] = weights_dir          # read by flashrefocus.paths at import
            from flashrefocus import Refocuser
            _MODELS.clear()
            _MODELS[key] = Refocuser.from_pretrained(detail_weight=detail_weight, deblurring=restoration,
                                                     fused=fused_attention,
                                                     compile=False if compile_restoration == "off" else compile_restoration)
        model = _MODELS[key]
        if model.deblurrer is not None:
            model.deblurrer.detail_weight = float(detail_weight)
        return (model,)


class FlashReFocusPrepare:
    """Stage I restoration (skipped when already_sharp) and depth, once per image or video frame. Batches are treated
    as video: the depth is smoothed over time with an exponential moving average (temporal_smoothing = weight of the
    previous frame; 0 = off). Outputs the scene, the restored image and its depth (near is bright)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("FLASHREFOCUS_MODEL",),
            "image": ("IMAGE",),
            "long_side": ("INT", {"default": 1024, "min": 256, "max": 3072, "step": 16}),
            "already_sharp": ("BOOLEAN", {"default": False}),
            "temporal_smoothing": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.95, "step": 0.05}),
        }}

    RETURN_TYPES = ("FLASHREFOCUS_SCENE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("scene", "restored", "depth")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    @torch.no_grad()
    def prepare(self, model, image, long_side, already_sharp, temporal_smoothing):
        if not already_sharp and model.deblurrer is None:
            raise ValueError("This model was loaded without restoration: set already_sharp, or load with restoration on.")
        frames = _resize_frames(image, long_side)
        device = model.renderer.mean.device
        images, depths = [], []
        for i in range(frames.shape[0]):
            scene = model.prepare(frames[i:i + 1].to(device), deblur=not already_sharp)
            images.append(scene.image.cpu())
            depths.append(scene.depth.float().cpu())
        depths = _smooth(depths, temporal_smoothing if len(depths) > 1 else 0.0)
        scene = {"images": images, "depths": depths}
        return (scene, _to_comfy(images), _depth_preview(depths))


# ------------------------------------------------------------------------------------------------------------- focus
class FlashReFocusFocusPoint:
    """Focus on a point: the focal depth is the depth at (x, y). For video it is read per frame and smoothed."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scene": ("FLASHREFOCUS_SCENE",), **_xy(),
                             "temporal_smoothing": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.95, "step": 0.05})}}

    RETURN_TYPES = ("FLASHREFOCUS_FOCUS",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, scene, x, y, temporal_smoothing):
        focal = _smooth([_focal_at(d, x, y) for d in scene["depths"]], temporal_smoothing)
        return (_focus(scene["depths"], focal),)


class FlashReFocusFocusMask:
    """Focus on a masked subject (for example a SAM or CLIPSeg mask, or a mask batch that tracks the subject through a
    video): the focal depth is the median depth inside the mask, and the subject can be kept sharp as a whole."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scene": ("FLASHREFOCUS_SCENE",), "mask": ("MASK",),
                             "keep_subject_sharp": ("BOOLEAN", {"default": True}),
                             "temporal_smoothing": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.95, "step": 0.05})}}

    RETURN_TYPES = ("FLASHREFOCUS_FOCUS",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, scene, mask, keep_subject_sharp, temporal_smoothing):
        from flashrefocus.rendering import depth_edits as E
        depths = scene["depths"]
        _, h, w = depths[0].shape
        masks = _masks_for(mask, len(depths), h, w)
        focal = [float(d[m].median()) if m.any() else float(d.median()) for d, m in zip(depths, masks)]
        focal = _smooth(focal, temporal_smoothing)
        out = [E.keep_sharp(d, m, f) if keep_subject_sharp else d for d, m, f in zip(depths, masks, focal)]
        return (_focus(out, focal),)


class FlashReFocusRadialFocus:
    """Lensbaby-style radial focus: a sharp disc around (x, y) with blur growing outward. Replaces the depth map."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scene": ("FLASHREFOCUS_SCENE",), **_xy(),
                             "radius": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "falloff": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 10.0, "step": 0.1})}}

    RETURN_TYPES = ("FLASHREFOCUS_FOCUS",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, scene, x, y, radius, falloff):
        from flashrefocus.rendering import depth_edits as E
        _, h, w = scene["depths"][0].shape
        d = E.radial_focus(h, w, x, y, radius, falloff)
        return (_focus([d] * len(scene["depths"]), [0.5] * len(scene["depths"])),)


class FlashReFocusTiltShift:
    """Tilt-shift at any angle: a sharp band through (x, y), rotated by `angle` degrees. Replaces the depth map."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scene": ("FLASHREFOCUS_SCENE",), **_xy(),
                             "angle": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                             "band": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01}),
                             "falloff": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1})}}

    RETURN_TYPES = ("FLASHREFOCUS_FOCUS",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, scene, x, y, angle, band, falloff):
        from flashrefocus.rendering import depth_edits as E
        _, h, w = scene["depths"][0].shape
        d = E.tilted_band(h, w, x, y, angle, band, falloff)
        return (_focus([d] * len(scene["depths"]), [0.5] * len(scene["depths"])),)


# ------------------------------------------------------------------------------------------------------- depth edits
class FlashReFocusFocusRange:
    """A thick focal plane: every depth within `width` of the focal depth stays sharp. Range and strength are separate
    controls, which one physical lens cannot separate."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"focus": ("FLASHREFOCUS_FOCUS",),
                             "width": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})}}

    RETURN_TYPES = ("FLASHREFOCUS_FOCUS",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, focus, width):
        from flashrefocus.rendering import depth_edits as E
        return (_focus([E.focus_range(d, f, width) for d, f in zip(focus["depths"], focus["focal"])], focus["focal"]),)


class FlashReFocusOneSided:
    """Blur only one side of the focal plane: keep the foreground sharp (only the background blurs) or keep the
    background sharp (only the foreground blurs)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"focus": ("FLASHREFOCUS_FOCUS",),
                             "keep_sharp": (["foreground", "background"],)}}

    RETURN_TYPES = ("FLASHREFOCUS_FOCUS",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, focus, keep_sharp):
        from flashrefocus.rendering import depth_edits as E
        return (_focus([E.one_sided(d, f, keep_sharp) for d, f in zip(focus["depths"], focus["focal"])], focus["focal"]),)


class FlashReFocusAddFocalPlane:
    """Add a focal plane at (x, y): both the current focal plane and the new one stay sharp, and the depths between
    them blur. Chain several nodes for three or more planes."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"focus": ("FLASHREFOCUS_FOCUS",), **_xy(0.5, 0.8)}}

    RETURN_TYPES = ("FLASHREFOCUS_FOCUS",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, focus, x, y):
        from flashrefocus.rendering import depth_edits as E
        planes = focus.get("planes") or [[f] for f in focus["focal"]]
        bases = focus.get("base") or focus["depths"]
        planes = [p + [_focal_at(b, x, y)] for p, b in zip(planes, bases)]
        out = [E.focal_planes(b, p) for b, p in zip(bases, planes)]
        result = _focus(out, [p[0] for p in planes])
        result.update(planes=planes, base=bases)
        return (result,)


class FlashReFocusMaskEdit:
    """Rewrite the depth inside a mask (SAM, CLIPSeg, or drawn):
      keep_sharp      put the masked region on the focal plane
      push_far        move it to the far background (remove a photobomber: he melts into the bokeh, no inpainting)
      push_near       move it to the nearest depth
      move_away       move it at least `amount` away from the focal plane (optical privacy blur for faces)
      set_depth       set it to depth `amount` (0 = near, 1 = far), for example to give a flat picture a front layer
      isolate         keep it sharp and move everything else at least `amount` away (strong subject separation)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"focus": ("FLASHREFOCUS_FOCUS",), "mask": ("MASK",),
                             "mode": (["keep_sharp", "push_far", "push_near", "move_away", "set_depth", "isolate"],),
                             "amount": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01})}}

    RETURN_TYPES = ("FLASHREFOCUS_FOCUS",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, focus, mask, mode, amount):
        from flashrefocus.rendering import depth_edits as E
        depths = focus["depths"]
        _, h, w = depths[0].shape
        masks = _masks_for(mask, len(depths), h, w)
        ops = {"keep_sharp": lambda d, m, f: E.keep_sharp(d, m, f),
               "push_far": lambda d, m, f: E.set_mask_depth(d, m, 1.0),
               "push_near": lambda d, m, f: E.set_mask_depth(d, m, 0.0),
               "move_away": lambda d, m, f: E.move_away(d, m, f, amount),
               "set_depth": lambda d, m, f: E.set_mask_depth(d, m, amount),
               "isolate": lambda d, m, f: E.isolate(d, m, f, amount)}
        out = [ops[mode](d, m, f) for d, m, f in zip(depths, masks, focus["focal"])]
        return (_focus(out, focus["focal"]),)


# ------------------------------------------------------------------------------------------------------------ render
class FlashReFocusRender:
    """Stage II: renders the focus with blur strength K (0 = sharp; 1 = the strongest blur seen in training; values
    above 1 extrapolate). Outputs the result and the depth map that was rendered (near is bright)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("FLASHREFOCUS_MODEL",), "scene": ("FLASHREFOCUS_SCENE",),
                             "focus": ("FLASHREFOCUS_FOCUS",),
                             "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01})}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "rendered_depth")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    @torch.no_grad()
    def render(self, model, scene, focus, strength):
        from flashrefocus import Scene
        device = model.renderer.mean.device
        out = []
        for img, d, f in zip(scene["images"], focus["depths"], focus["focal"]):
            s = Scene(img.to(device), d.to(device))
            out.append(model.render(s, strength, f).cpu())
        return (_to_comfy(out), _depth_preview(focus["depths"]))



# ----------------------------------------------------------------------------------------------- video spotlight
class FlashReFocusSAM3Subject:
    """SAM 3 subject masks for a video (image batch): segments and tracks every instance of `prompt` through the clip
    and picks one track as the subject (most_central, largest, or the track under point x, y on the first frame
    where one covers it). Outputs the subject mask and the union of the others (for Spotlight). Needs
    transformers >= 5.0 and access to facebook/sam3; segmentation runs at `long_side` at most."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "prompt": ("STRING", {"default": "person"}),
                             "select": (["most_central", "largest", "point"],), **_xy(),
                             "long_side": ("INT", {"default": 1280, "min": 256, "max": 2048, "step": 16}),
                             "chunk": ("INT", {"default": 300, "min": 16, "max": 2000, "step": 1})}}

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("subject", "others")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, prompt, select, x, y, long_side, chunk):
        from flashrefocus.segment import track_subject
        _, h, w, _ = images.shape
        s = min(1.0, long_side / max(h, w))
        frames = images.permute(0, 3, 1, 2).float()
        if s < 1:
            frames = F.interpolate(frames, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False,
                                   antialias=True)
        frames = (frames.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        subject, others = track_subject(frames, prompt.strip() or "person", select, (x, y), chunk, device=device,
                                        dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        return (subject.float(), others.float())


class FlashReFocusSpotlight:
    """Spotlight on a subject in a video or photograph: the focal depth is the median depth inside the subject mask
    (smoothed over frames), the mask is grown by `grow` pixels into nearby pixels at the subject's depth that belong
    to no other person (missed hands and hair), and every pixel outside it moves at least `gap` away from the focal
    plane, so the people around the subject blur strongly. Connect `others` (for example from SAM 3 Subject) so the
    mask never grows into a neighbour. Frames without the subject are rendered with their own depth."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scene": ("FLASHREFOCUS_SCENE",), "subject": ("MASK",),
                             "gap": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "grow": ("INT", {"default": 10, "min": 0, "max": 64, "step": 1}),
                             "temporal_smoothing": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.95, "step": 0.05})},
                "optional": {"others": ("MASK",)}}

    RETURN_TYPES = ("FLASHREFOCUS_FOCUS", "MASK")
    RETURN_NAMES = ("focus", "subject_grown")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, scene, subject, gap, grow, temporal_smoothing, others=None):
        from flashrefocus.rendering import depth_edits as E
        depths = scene["depths"]
        _, h, w = depths[0].shape
        masks = _masks_for(subject, len(depths), h, w)
        other = _masks_for(others, len(depths), h, w) if others is not None else [None] * len(depths)
        present = [bool(m.sum() > 50) for m in masks]
        focal = [float(d[m].median()) if p else float(d.median()) for d, m, p in zip(depths, masks, present)]
        focal = _smooth(focal, temporal_smoothing)
        out, grown = [], []
        for d, m, o, f, p in zip(depths, masks, other, focal, present):
            if p:
                d2, m2 = E.spotlight(d, m, f, gap, o, grow)
            else:
                d2, m2 = d, m
            out.append(d2)
            grown.append(m2.float().view(h, w))
        return (_focus(out, focal), torch.stack(grown))


class FlashReFocusOverlayFill:
    """Removes a burnt-in overlay (title, caption, logo; white = overlay) by filling it from its surroundings, before
    Prepare: otherwise the renderer blurs the letters together with the scene and they glow and flicker as the focus
    changes. Put the sharp overlay back after Render with Overlay Paste Back."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, mask):
        from flashrefocus.overlays import fill_region
        x = image.permute(0, 3, 1, 2).float()
        m = _batch_mask(mask, x.shape[0], *x.shape[-2:])
        return (fill_region(x, m).permute(0, 2, 3, 1).clamp(0, 1),)


class FlashReFocusOverlayPasteBack:
    """Pastes the overlay (white in the mask) from the original frames back onto the rendered frames, with a soft
    edge. The original and the mask are resized to the rendered size."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "original": ("IMAGE",), "mask": ("MASK",),
                             "feather": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, original, mask, feather):
        from flashrefocus.overlays import paste_back
        r = image.permute(0, 3, 1, 2).float()
        o = F.interpolate(original.permute(0, 3, 1, 2).float(), size=r.shape[-2:], mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 1)
        if o.shape[0] < r.shape[0]:
            o = o[torch.arange(r.shape[0]).clamp(max=o.shape[0] - 1)]
        m = _batch_mask(mask, r.shape[0], *r.shape[-2:])
        return (paste_back(r, o, m, feather).permute(0, 2, 3, 1).clamp(0, 1),)


def _batch_mask(mask, n, h, w):
    """ComfyUI MASK -> (n, 1, h, w) float in [0, 1] (one mask is reused for every frame)."""
    m = mask if mask.dim() == 3 else mask[None]
    m = F.interpolate(m[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)
    return m[torch.arange(n).clamp(max=m.shape[0] - 1)]

NODE_CLASS_MAPPINGS = {
    "FlashReFocusLoader": FlashReFocusLoader,
    "FlashReFocusPrepare": FlashReFocusPrepare,
    "FlashReFocusFocusPoint": FlashReFocusFocusPoint,
    "FlashReFocusFocusMask": FlashReFocusFocusMask,
    "FlashReFocusRadialFocus": FlashReFocusRadialFocus,
    "FlashReFocusTiltShift": FlashReFocusTiltShift,
    "FlashReFocusFocusRange": FlashReFocusFocusRange,
    "FlashReFocusOneSided": FlashReFocusOneSided,
    "FlashReFocusAddFocalPlane": FlashReFocusAddFocalPlane,
    "FlashReFocusMaskEdit": FlashReFocusMaskEdit,
    "FlashReFocusRender": FlashReFocusRender,
    "FlashReFocusSAM3Subject": FlashReFocusSAM3Subject,
    "FlashReFocusSpotlight": FlashReFocusSpotlight,
    "FlashReFocusOverlayFill": FlashReFocusOverlayFill,
    "FlashReFocusOverlayPasteBack": FlashReFocusOverlayPasteBack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashReFocusLoader": "FlashReFocus Loader",
    "FlashReFocusPrepare": "FlashReFocus Prepare (restore + depth)",
    "FlashReFocusFocusPoint": "FlashReFocus Focus Point",
    "FlashReFocusFocusMask": "FlashReFocus Focus on Mask",
    "FlashReFocusRadialFocus": "FlashReFocus Radial Focus",
    "FlashReFocusTiltShift": "FlashReFocus Tilt-Shift",
    "FlashReFocusFocusRange": "FlashReFocus Focus Range",
    "FlashReFocusOneSided": "FlashReFocus One-Sided Blur",
    "FlashReFocusAddFocalPlane": "FlashReFocus Add Focal Plane",
    "FlashReFocusMaskEdit": "FlashReFocus Mask Depth Edit",
    "FlashReFocusRender": "FlashReFocus Render",
    "FlashReFocusSAM3Subject": "FlashReFocus SAM 3 Subject (video)",
    "FlashReFocusSpotlight": "FlashReFocus Spotlight",
    "FlashReFocusOverlayFill": "FlashReFocus Overlay Fill (before Prepare)",
    "FlashReFocusOverlayPasteBack": "FlashReFocus Overlay Paste Back (after Render)",
}
