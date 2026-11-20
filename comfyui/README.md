# FlashReFocus for ComfyUI

Custom nodes that bring every FlashReFocus control into ComfyUI: refocus a photograph after capture, change the
aperture, and edit the explicit defocus map (focus range, several focal planes, one-sided blur, radial focus,
tilt-shift at any angle, mask-based edits) on photographs and videos, and keep one subject sharp in a video with a
spotlight whose subject SAM 3 finds from a text prompt. Restoration and depth run once per image; every later edit only
re-runs the lightweight renderer (about half a second at 1536×1536), which the loader runs with a fused Triton
attention kernel by default (same output, much less memory at high resolution).

## Installation

1. Clone the repository with its submodules and link this folder into ComfyUI:

   ```bash
   git clone --recursive https://github.com/ming053l/FlashReFocus
   ln -s "$PWD/FlashReFocus/comfyui" ComfyUI/custom_nodes/ComfyUI-FlashReFocus
   ```

   On Windows, copy or junction the folder instead (`mklink /J`). The nodes put the repository root on the import path,
   so `pip install -e .` is not required.

2. Install the dependencies into the Python environment that runs ComfyUI:

   ```bash
   pip install -r FlashReFocus/requirements.txt
   ```

   FlashReFocus pins `diffusers==0.25.0` and `transformers` 4.46 (needed by the PiSA-SR code of Stage I). Recent
   ComfyUI releases ask for a newer `transformers`; the core nodes and the FlashReFocus nodes work with 4.46, but if
   other custom nodes need a newer version, run FlashReFocus in a separate ComfyUI environment. Tested with Python
   3.10, torch 2.7.1 (CUDA 12.8) and ComfyUI from November 2026.

   **SAM 3 Subject** needs `transformers` >= 4.57 and access to the gated
   [facebook/sam3](https://huggingface.co/facebook/sam3) checkpoint (accept its terms, then `huggingface-cli login`).
   With restoration off (sharp video, which is what the spotlight workflows use), FlashReFocus does not load Stage I,
   so these nodes run with a recent `transformers`; workflows 04-06 were tested with Python 3.11, torch 2.11 (CUDA
   12.8), `transformers` 5.23 and ComfyUI from November 2026. Install `kernels` (`pip install kernels`) for SAM 3's
   mask post-processing (hole filling, small-fragment removal).

3. Weights are downloaded automatically on first use (about 7 GB: Stable Diffusion 2.1 base, the SD 1.5 VAE,
   Depth Anything V2 Large and the three FlashReFocus checkpoints) into `FlashReFocus/weights/`. To use another
   folder, set `FLASHREFOCUS_WEIGHTS` or the `weights_dir` field of the loader. To download them in advance, see
   [Weights](../README.md#weights).

## Example workflows

Drag a file from [`workflows/`](workflows) onto the ComfyUI canvas, put your inputs into `ComfyUI/input/` under the
names below (or pick other files in the load nodes), and press *Queue*.

| Workflow | What it shows | Inputs |
|---|---|---|
| `01_refocus_photo.json` | Restore a defocused photograph once, then refocus it on a point. Also saves the restored image and its depth. | `photo.jpg` |
| `02_depth_effects.json` | Five edits of one scene from a single preparation: focus range, one-sided blur, two focal planes, radial focus, tilt-shift. | `photo.jpg` |
| `03_mask_edits.json` | On a sharp photograph (restoration off): keep a masked subject sharp and push everything else away, or push the masked region into the far background. | `sharp_photo.jpg`, `subject_mask.png` (white = subject) |
| `04_video_spotlight.json` | Video spotlight with a mask video: the subject stays sharp in every frame and everyone else is pushed out of focus; depth and focal depth are smoothed over time. | `video.mp4`, `video_mask.mp4` (white = subject) |
| `05_video_spotlight_sam3.json` | The same without a mask: SAM 3 finds every person (prompt "person"), picks the subject (most central; or largest, or the one under a point) and follows it; also saves the subject mask as a video. | `video.mp4` |
| `06_title_overlay.json` | A video with a burnt-in title: the title is filled in before depth and rendering and pasted back sharp afterwards, so it does not glow or flicker. | `title_video.mp4`, `title_mask.png` (white = title) |

Masks can come from any mask node, for example Segment Anything (SAM / SAM 2) or CLIPSeg custom nodes, a drawn mask
from the image editor, or `LoadImageMask`. For video, connect a mask batch with one mask per frame (for example from
**SAM 3 Subject**, or a SAM 2 video node); a single mask is reused for every frame.

## Nodes

All nodes are in the **FlashReFocus** category. Coordinates `x`, `y` are fractions of the image width and height
(0 = left / top, 1 = right / bottom), so they do not depend on the working resolution.

| Node | Purpose |
|---|---|
| **FlashReFocus Loader** | Loads Stage I (one-step restoration), Depth Anything V2 and the renderer. Turn `restoration` off for photographs and videos that are already sharp (portrait mode); only depth and the renderer are loaded (and downloaded) then. `detail_weight` (ω) trades pixel fidelity (lower) for perceptual detail (higher); 1.25 is the released setting. `fused_attention` (on by default) runs the renderer's attention as one Triton kernel. `compile_restoration` (`dynamic` or `static`) compiles Stage I with `torch.compile` for about 35-40% faster restoration; the first image waits one to two minutes for compilation. |
| **FlashReFocus Prepare (restore + depth)** | Resizes to `long_side` (sides multiples of 16), restores the image (unless `already_sharp`) and estimates depth, once per image or frame. Outputs the scene, the restored image and its depth (near is bright). For an image batch (video), `temporal_smoothing` smooths the depth over time. |
| **FlashReFocus Focus Point** | Focal plane at the depth of point (x, y). |
| **FlashReFocus Focus on Mask** | Focal plane at the median depth inside a mask; optionally keeps the whole subject sharp. With a mask batch it follows the subject through a video. |
| **FlashReFocus Focus Range** | A thick focal plane: everything within `width` of the focal depth stays sharp. |
| **FlashReFocus One-Sided Blur** | Keep the foreground sharp and blur only the background, or the reverse. |
| **FlashReFocus Add Focal Plane** | Adds another sharp plane at (x, y); the depths in between blur. Chain several for three or more planes. |
| **FlashReFocus Radial Focus** | A sharp disc around (x, y) with blur growing outward (Lensbaby look). |
| **FlashReFocus Tilt-Shift** | A sharp band through (x, y) at any `angle`, with `band` half-width and `falloff`. |
| **FlashReFocus Mask Depth Edit** | Rewrites the depth inside a mask: `keep_sharp`, `push_far` (remove a photobomber without inpainting), `push_near`, `move_away` (optical privacy blur for faces), `set_depth` (give a flat picture a front layer), `isolate` (keep the subject sharp and move everything else at least `amount` away). |
| **FlashReFocus Render** | Renders the focus with blur `strength` K (0 = sharp, 1 = the strongest training blur, up to 2 extrapolates). Outputs the image and the depth map that was rendered. |
| **FlashReFocus SAM 3 Subject (video)** | SAM 3 segments and tracks every instance of `prompt` (default "person") in an image batch and picks one as the subject: `most_central`, `largest`, or `point` (the track under x, y). Outputs the subject mask and the union of the others. Long clips run in chunks of `chunk` frames. |
| **FlashReFocus Spotlight** | Keeps a subject sharp and pushes everything else at least `gap` from the focal plane (focal depth = median depth in the mask, smoothed over frames). The mask grows by `grow` pixels into nearby pixels at the subject's depth that belong to no one in `others`, recovering hands and hair the segmentation missed. Also outputs the grown mask. |
| **FlashReFocus Overlay Fill (before Prepare)** | Fills a burnt-in title, caption or logo (white in the mask) from its surroundings, so the renderer does not blur it into a flickering glow. |
| **FlashReFocus Overlay Paste Back (after Render)** | Pastes the overlay from the original frames back onto the rendered ones, with a soft edge. |

Edits chain: start from **Focus Point**, **Focus on Mask**, **Spotlight**, **Radial Focus** or **Tilt-Shift**, add any number of
**Focus Range**, **One-Sided Blur**, **Add Focal Plane** and **Mask Depth Edit** nodes, and end with **Render**. Changing
only the focus nodes or the strength re-runs the renderer alone, because ComfyUI caches the loader and the prepared
scene.

## Notes

- The renderer was trained with one focal plane and one strength per image; every depth edit above is zero-shot.
- Video is processed frame by frame with temporally smoothed depth and focal depth. For defocused video, restoration
  runs independently per frame and may flicker; the video examples in the paper use sharp clips.
- Restored detail comes partly from a diffusion prior; do not treat the output as an exact record of the scene.
