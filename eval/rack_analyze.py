#!/usr/bin/env python
"""Rack-focus comparison of FlashReFocus and GenRefocus (outputs of rack_ours.py / rack_genrefocus.py) and material:
  metrics  - in-focus fidelity: PSNR and LPIPS against the AiF over pixels near the focal depth;
             edit consistency: between consecutive focal settings, pixels whose expected blur is unchanged
             (| |d - f_i| - |d - f_i+1| | < tau) should stay the same -> PSNR / mean abs difference between the frames there.
             Masks use FlashReFocus' DA2 depth for both methods (GenRefocus uses Depth Pro internally, so this is slightly
             in its disfavour; tau is loose).
  figures  - comparison strips (far -> near) and consecutive-frame difference maps;
  videos   - rack-focus and strength sweeps (mp4 + gif) and a side-by-side comparison, for the project page."""
import json
import subprocess
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import os
R = Path(os.environ.get("RACK_OUT", "/ssd1/chingheng/Defocus/refocus_bench/rack_focus"))
AIF = Path(os.environ.get("RACK_AIF", "/ssd1/chingheng/Defocus/refocus_bench/3creal_lvcorr/aif"))
FFMPEG = "/ssd1/chingheng/Defocus/refocus_bench/pylibs_video/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"  # static build with libx264
TAU_FOCUS, TAU_SAME = 0.04, 0.02
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except OSError:
    FONT = ImageFont.load_default()


def load(p, size=None):
    """size (w, h): GenRefocus resizes a 1024x682 input to multiples of 16 (1024x688) before generation, so its output
    is resized back to the AiF size for pixel-aligned comparison."""
    im = Image.open(p).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def mpsnr(a, b, m):
    if m.sum() < 100:
        return float("nan")
    mse = ((a - b) ** 2)[m].mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def mark(img, x, y, r=10):
    im = Image.fromarray((img * 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 0), width=3)
    return np.asarray(im).astype(np.float32) / 255.0


def label(img, text):
    im = Image.fromarray((img * 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, d.textlength(text, font=FONT) + 16, 34], fill=(0, 0, 0))
    d.text((8, 5), text, fill=(255, 255, 255), font=FONT)
    return np.asarray(im).astype(np.float32) / 255.0


def save(img, p):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(p)


def video(frames_glob_dir, pattern, dst, fps):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_glob_dir / pattern),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(dst.with_suffix(".mp4"))], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(dst.with_suffix(".mp4")), "-vf",
                    "fps=%d,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" % min(fps, 15),
                    str(dst.with_suffix(".gif"))], check=True)


def main():
    dev = "cpu"   # keeps the GPUs free for timing runs
    net = lpips.LPIPS(net="alex", spatial=True).to(dev).eval()
    lp = lambda a, b: net(torch.from_numpy(a).permute(2, 0, 1)[None].to(dev) * 2 - 1,
                          torch.from_numpy(b).permute(2, 0, 1)[None].to(dev) * 2 - 1)[0, 0].detach().cpu().numpy()
    meta = json.loads((R / "meta_ours.json").read_text())
    gmeta = json.loads((R / "meta_genrefocus.json").read_text())
    page = R / "project_page"; page.mkdir(exist_ok=True)
    table = {}
    for s, info in meta["scenes"].items():
        o = R / s; pts = info["points"]; d = np.load(o / "depth.npy"); aif = load(AIF / f"{s}.png")
        methods = {"FlashReFocus": o / "ours", "GenRefocus": o / "genrefocus"}
        res = {}
        for name, fdir in methods.items():
            frames = [load(fdir / f"{i:02d}.png", (aif.shape[1], aif.shape[0])) for i in range(len(pts))]
            fid_p, fid_l, con_p, con_d = [], [], [], []
            for i, p in enumerate(pts):
                m = np.abs(d - p["depth"]) < TAU_FOCUS
                fid_p.append(mpsnr(frames[i], aif, m[..., None].repeat(3, -1)))
                fid_l.append(float(lp(frames[i], aif)[m].mean()) if m.sum() >= 100 else float("nan"))
                if i + 1 < len(pts):
                    same = np.abs(np.abs(d - p["depth"]) - np.abs(d - pts[i + 1]["depth"])) < TAU_SAME
                    con_p.append(mpsnr(frames[i], frames[i + 1], same[..., None].repeat(3, -1)))
                    con_d.append(float(np.abs(frames[i] - frames[i + 1]).mean(-1)[same].mean()) if same.sum() >= 100 else float("nan"))
            res[name] = dict(infocus_psnr=float(np.nanmean(fid_p)), infocus_lpips=float(np.nanmean(fid_l)),
                             consistency_psnr=float(np.nanmean(con_p)), consistency_absdiff=float(np.nanmean(con_d)))
            # consecutive-frame difference map for the middle pair
            k = len(pts) // 2 - 1
            diff = np.abs(frames[k] - frames[k + 1]).mean(-1)
            save(np.repeat(np.clip(diff * 4, 0, 1)[..., None], 3, -1), o / f"diffmap_{name}.png")
        table[s] = res
        # comparison strip: AiF | 4 focal settings from far to near, rows FlashReFocus / GenRefocus
        sel = [0, 2, 5, 7]
        rows = []
        for name, fdir in methods.items():
            tiles = [label(aif, "AiF input") if name == "FlashReFocus" else np.ones_like(aif)]
            for j, i in enumerate(sel):
                t = mark(load(fdir / f"{i:02d}.png", (aif.shape[1], aif.shape[0])), pts[i]["x"], pts[i]["y"])
                tiles.append(label(t, f"{name}  focus {j + 1}") if j == 0 else t)
            rows.append(np.concatenate([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=1) for t in tiles], 1))
        save(np.concatenate(rows, 0), o / "strip_far_to_near.png")
        # videos: rack focus, strength sweep, side-by-side 8 focal settings
        video(o / "video", "%03d.png", page / f"{s}_rack_focus", 24)
        video(o / "strength", "%03d.png", page / f"{s}_strength_sweep", 15)
        sbs = o / "sbs"; sbs.mkdir(exist_ok=True)
        t_ours = info["render_s_mean"]          # seconds per focal edit measured at this resolution
        t_gen = float(np.mean(gmeta["seconds_per_frame"][s]))
        for i, p in enumerate(pts):
            a = label(mark(load(o / "ours" / f"{i:02d}.png"), p["x"], p["y"]), f"FlashReFocus ({t_ours:.2f} s / edit)")
            b = label(mark(load(o / "genrefocus" / f"{i:02d}.png", (aif.shape[1], aif.shape[0])), p["x"], p["y"]),
                      f"GenRefocus ({t_gen:.0f} s / edit)")
            save(np.concatenate([a, np.ones((a.shape[0], 12, 3)), b], 1), sbs / f"{i:02d}.png")
        video(sbs, "%02d.png", page / f"{s}_side_by_side", 2)
    (R / "rack_metrics.json").write_text(json.dumps(table, indent=1))
    avg = {m: {k: float(np.nanmean([table[s][m][k] for s in table])) for k in table[next(iter(table))][m]}
           for m in ("FlashReFocus", "GenRefocus")}
    (R / "rack_metrics_mean.json").write_text(json.dumps(avg, indent=1))
    print(json.dumps(avg, indent=1))


if __name__ == "__main__":
    main()
