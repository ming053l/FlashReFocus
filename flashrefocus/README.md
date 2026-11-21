# `flashrefocus` package: options and details

The [top-level README](../README.md) covers the common use. This page lists the remaining options of the Python API.

## Loading

`Refocuser.from_pretrained(...)` loads Stage I (restoration), the depth estimator and the renderer.

| Argument | Default | Effect |
|---|---|---|
| `deblurring` | `True` | `False` loads only the depth estimator and the renderer, for images that are already sharp. Stage I and its pinned `diffusers` / `transformers` are then not imported, so this also runs next to a recent `transformers` (needed for SAM 3). |
| `detail_weight` | `1.25` | ω, the blend of the pixel and pixel + detail predictions of Stage I (higher: more synthesized detail). |
| `refiner` | released refiner | `None` decodes with the SD 2.1 VAE decoder instead of the decoder refiner. |
| `refiner_dtype` | `bfloat16` | The cost benchmark (paper, Table 6) uses `float16`. |
| `render_dtype` | `bfloat16` | `None` renders in fp32. |
| `channels_last` | `False` | The cost benchmark uses `True`. |
| `merge_vae_lora` | `False` | Merges the VAE-encoder LoRA: about 20% faster encoder, outputs about 60 dB from the exact ones. |
| `fused` | `False` | Fused attention kernel for the renderer, see [Speed-ups](#speed-ups). |
| `compile` | `False` | `"static"` or `"dynamic"`: compiles Stage I, see [Speed-ups](#speed-ups). |

Stage I alone: `flashrefocus.deblurring.Deblurrer.from_pretrained()(x)` restores an image `x` in [-1, 1].
`Deblurrer.from_pretrained(separate_unets=True)` keeps the pixel and pixel + detail U-Nets as two modules instead of
swapping weights in place (same output, twice the U-Net memory), as the Hugging Face demo does on ZeroGPU.

## Scenes and rendering

- `refocuser.prepare(image, deblur=True)` returns a `Scene`: the restored image (quantized to 8 bits, as when it is
  saved and reloaded) and its depth. `deblur=False` skips Stage I. Image sides must be multiples of 16
  (`image_io.read_image(path, device, multiple=16)` or `image_io.resize_to_multiple` pad or resize).
- `scene.focal_depth_at(x, y)` gives the focal depth D_f of a pixel.
- `refocuser.render(scene, strength, focal_depth, depth=None)` returns the result (b, 3, H, W) in [0, 1]. The strength K
  is trained in [0, 1]; larger values extrapolate. `depth` replaces the scene depth, for the edits in
  [`rendering/depth_edits.py`](rendering/depth_edits.py) (listed in the top-level README under Applications).
- `render` takes the focal depth rather than a pixel, so a video can be processed as one `Scene` per frame, with D_f
  and K chosen per frame (for example smoothed over time).
- `refocuser.render_batch(images, depths, strengths, focal_depths)` renders several frames in one call, with one
  strength and focal depth per frame. It gives the same result as rendering them one by one. At video resolutions one
  frame already fills a large GPU, so it is a convenience, not a speed-up.

## Depth edits

Every effect replaces the depth passed to `render`. The renderer was trained with one focal plane and one strength per
image, so all of them are zero-shot.

| Effect | `depth_edits` function | ComfyUI node |
|---|---|---|
| Focus range: a thick focal plane, separate from the aperture | `focus_range` | Focus Range |
| Radial (Lensbaby-style) focus around a point | `radial_focus` | Radial Focus |
| Tilt-shift band at any angle | `tilted_band` (`tilt_shift` for horizontal) | Tilt-Shift |
| Blur only the background, or only the foreground | `one_sided` | One-Sided Blur |
| Two, three or more focal planes | `focal_planes` (`two_focal_planes`) | Add Focal Plane |
| Keep a masked subject sharp while the focus moves | `keep_sharp` | Focus on Mask, Mask Depth Edit |
| Remove a photobomber: push him into the background bokeh (no inpainting) | `set_mask_depth(…, 1.0)` | Mask Depth Edit (`push_far`) |
| Optical privacy blur for faces or plates | `move_away` | Mask Depth Edit (`move_away`) |
| Per-layer blur, or a front layer for a flat picture | `set_mask_depth` | Mask Depth Edit (`set_depth`) |
| Isolate a subject: everything else at least a set distance from focus | `isolate` | Mask Depth Edit (`isolate`) |
| Portrait mode on a sharp photograph (skip Stage I) | `prepare(image, deblur=False)` | Prepare (`already_sharp`) |
| Video focus pull | one `Scene` per frame, smoothed depth and focus | Prepare + Focus Point with an image batch |
| Video spotlight (one subject sharp, everyone else blurred) | `spotlight` + `flashrefocus.segment.track_subject` (SAM 3) | SAM 3 Subject + Spotlight |
| Burnt-in titles kept sharp in video | `flashrefocus.overlays.fill_region` / `paste_back` | Overlay Fill + Overlay Paste Back |

In Python:

```python
from flashrefocus.rendering import depth_edits

_, H, W = scene.depth.shape                                  # scene = refocuser.prepare(...); focal_depth = scene.focal_depth_at(x, y)
edits = {
    "focus_range": (depth_edits.focus_range(scene.depth, focal_depth, width=0.2), focal_depth),
    "background_only": (depth_edits.one_sided(scene.depth, focal_depth, keep="foreground"), focal_depth),
    "three_planes": (depth_edits.focal_planes(scene.depth, [focal_depth, scene.focal_depth_at(x2, y2), 0.9]), focal_depth),
    "radial": (depth_edits.radial_focus(H, W, 0.5, 0.6, radius=0.12, device="cuda"), 0.5),
    "tilt_shift_20deg": (depth_edits.tilted_band(H, W, 0.5, 0.6, angle=20, device="cuda"), 0.5),
    "keep_sharp": (depth_edits.keep_sharp(scene.depth, mask, focal_depth), focal_depth),   # mask: H x W bool
    "remove_photobomber": (depth_edits.set_mask_depth(scene.depth, mask, 1.0), focal_depth),
    "privacy_blur": (depth_edits.move_away(scene.depth, mask, focal_depth, 0.3), focal_depth),
    "isolate_subject": (depth_edits.isolate(scene.depth, mask, focal_depth, 0.25), focal_depth),
}
for name, (depth, d_f) in edits.items():
    save_image(refocuser.render(scene, 0.8, d_f, depth=depth), f"{name}.png")   # about half a second each
```

`mask` is an H x W bool tensor from any segmenter, for example Segment Anything from a click or a box, or CLIPSeg from
a text prompt.

## Video spotlight

```python
from flashrefocus.segment import track_subject        # SAM 3: transformers >= 5.0, access to facebook/sam3
from flashrefocus.rendering import depth_edits

refocuser = Refocuser.from_pretrained(deblurring=False, fused=True)
subject, others = track_subject(frames, prompt="person", select="most_central")   # frames: (n, h, w, 3) uint8
for frame, m, o in zip(frames_as_tensors, subject, others):
    scene = refocuser.prepare(frame, deblur=False)
    d_f = float(scene.depth[0][m].median())           # smooth over frames in practice
    depth, _ = depth_edits.spotlight(scene.depth, m[None], d_f, gap=0.6, others=o[None], grow=10)
    out = refocuser.render(scene, 1.5, d_f, depth=depth)
```

- `track_subject(frames, prompt, select, point, chunk)` tracks every instance of `prompt` with SAM 3, in chunks of
  `chunk` frames. It returns the chosen subject's masks and the union of the other tracks. `select` is
  `"most_central"`, `"largest"` or `"point"` (the track under `point`, given as fractions of width and height). The
  subject is carried from one chunk to the next by mask overlap.
- `spotlight(depth, mask, focal_depth, gap, others, grow)` sets the subject to the focal depth and moves every other
  pixel at least `gap` from it. First it grows the mask by up to `grow` pixels into neighbouring pixels at the
  subject's depth that are not in `others` (hands and hair that segmentation misses).
- Burnt-in titles: `overlays.fill_region(image, mask)` fills the title from its surroundings before rendering, so the
  renderer does not blur the letters into a flickering glow. `overlays.paste_back(result, original, mask)` then puts
  the sharp title back.

## Writing results

- `image_io.save_image(x, path, background=True)` copies the image to the CPU at once and compresses and writes the PNG
  on a background thread (the same file, byte for byte). `image_io.wait_for_writes()` waits for pending files.
- `video.VideoWriter(path, fps, codec="auto", quality=18, audio_from=None)` streams frames into one ffmpeg encode from
  a background thread, so encoding overlaps rendering. It picks the first encoder that works: NVENC, libx264,
  libopenh264, then mpeg4. `audio_from=` copies the audio of the source clip. ffmpeg is taken from
  `$FLASHREFOCUS_FFMPEG`, `imageio-ffmpeg`, or `PATH`.

```python
from flashrefocus.video import VideoWriter

with VideoWriter("refocused.mp4", fps=30, audio_from="clip.mp4") as writer:
    for scene, d_f in zip(scenes, focal_depths):
        writer.write(refocuser.render(scene, 1.0, d_f))
```

## Speed-ups

Both options are off by default, so the reported numbers are reproduced exactly. ComfyUI and the demo turn `fused` on.

- **`fused=True`** runs the renderer's defocus-bias attention as one Triton kernel
  ([`rendering/fused.py`](rendering/fused.py)). The kernel computes the defocus bias from the per-token defocus values
  and uses an online softmax, so the per-row and per-column bias and attention matrices are never stored. Weights are
  unchanged, and outputs match the reference path to about 61 dB in bf16 and 80 dB in fp32. On an RTX PRO 6000,
  the renderer alone takes 72 ms instead of 170 ms at 1536×1536. At 3840×1728 it takes 229 ms instead of 740 ms, and
  peak memory drops from 16.1 to 3.0 GiB. Without Triton, or on CPU, the reference code runs.
  `rendering.fused.enable_fused(renderer)` switches a renderer that is already loaded.
- **`compile="static"`** (or `"dynamic"`) compiles Stage I's two full-resolution parts, the decoder refiner and the VAE
  encoder, with `torch.compile`. Restoration is about 35-40% faster after a one-time compilation of one to two
  minutes, and outputs match the eager path to about 57-62 dB. `"static"` is fastest for many images of one size and
  recompiles for a new size. `"dynamic"` compiles once for all sizes and is about 10% slower.

## Reproducibility

Outputs are bit-reproducible with `torch.backends.cudnn.benchmark = False`. cuDNN autotuning is faster for a fixed
input size, but it may pick different convolution algorithms between runs, which changes outputs by a few 1/255.

## Tests

The regression test restores two small photographs in `tests/data/` and compares the restored image, depth, refocused
image, two focal planes and tilt-shift with `tests/reference/`: bit-identical on the machine that made the references,
PSNR >= 45 dB elsewhere (floating-point differences between GPUs and library versions). Regenerate the references with
`python tests/regression_cases.py` only when an output changes on purpose.
