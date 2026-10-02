#!/usr/bin/env python
"""Media for the project page, produced with the paper model (Stage I v3_s1 40K + refiner, omega 1.25; bf16 renderer).

  restore pairs : assets/restore_<name>_{in,out}.jpg, input and restored all-in-focus image (drag-to-compare sliders)
  comparisons   : assets/sbs_<scene>.mp4, FlashReFocus against GenRefocus on the same eight focal points
  demo video    : assets/demo.mp4, a scripted session per scene: restore once, click focal points, turn the blur
                  strength, set two focal planes. Every frame on the right is an actual output of the pipeline.

Run:  CUDA_VISIBLE_DEVICES=<gpu> ~/miniconda3/envs/pisa/bin/python make_page_media.py [--only restore|demo|sbs]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DEFOCUS = HERE.parent
sys.path.insert(0, str(DEFOCUS / "refocus_bench"))
sys.path.insert(0, str(DEFOCUS / "Stage1_PiSA"))
import fast_refocus as FR  # noqa: E402

DATA = Path("/ssd1/chingheng/Deblurring-Dataset")
STAGE1 = DEFOCUS / "Stage1_PiSA/runs/v3_s1/checkpoints/model_40000.pkl"
REFINER = DEFOCUS / "demo/weights/refiner.pt"
ASSETS, WORK = HERE / "assets", HERE / "work"
FFMPEG = "/ssd1/chingheng/Defocus/refocus_bench/pylibs_video/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"  # static build with libx264
FONT = HERE / "fonts/Figtree.ttf"                 # variable font (OFL), the page's heading face

RESTORE = [("dpdd_bench", DATA / "dd_dp_dataset_png/test_c/source/1P0A0917.png"),
           ("dpdd_ivy", DATA / "dd_dp_dataset_png/test_c/source/1P0A0984.png"),
           ("dpdd_fence", DATA / "dd_dp_dataset_png/test_c/source/1P0A1614.png"),
           ("dpdd_bike", DATA / "dd_dp_dataset_png/test_c/source/1P0A2417.png"),
           ("realdof_shrub", DATA / "RealDOF/source/37.png"),
           ("realdof_truck", DATA / "RealDOF/source/41.png"),
           ("realdof_lot", DATA / "RealDOF/source/45.png"),
           ("real_flower", DATA / "Refocus_samples/real_19.jpg"),
           ("real_sculpture", DATA / "Refocus_samples/real_23.jpg")]
DEMO = [("sculpture", DATA / "Refocus_samples/real_23.jpg"),
        ("flower", DATA / "Refocus_samples/real_19.jpg"),
        ("dock", DATA / "Refocus_samples/10.jpg")]

BG, FG, DIM, ACC = (16, 21, 41), (255, 255, 255), (150, 160, 190), (255, 228, 196)
FPS, PW, PH, M, TOP = 24, 936, 614, 16, 64


def font(size, weight=600):
    f = ImageFont.truetype(str(FONT), size)
    f.set_variation_by_axes([weight])
    return f
CW, CH = 2 * PW + 3 * M, TOP + PH + M


def fit(img, long_side):
    w, h = img.size
    s = min(1.0, long_side / max(w, h))
    return img.convert("RGB").resize((max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)), Image.LANCZOS)


def restore_pairs(s1):
    for name, path in RESTORE:
        src, dst = WORK / f"{name}_in.png", WORK / f"{name}_out.png"
        fit(Image.open(path), 1280).save(src)
        s1(src, dst)
        for k, p in (("in", src), ("out", dst)):
            Image.open(p).save(ASSETS / f"restore_{name}_{k}.jpg", quality=88)
        print("restore", name, Image.open(dst).size, flush=True)


def pick_points(depth, aif, quantiles):
    """One textured pixel per depth quantile, away from the border (depth: HxW tensor, 1 = far)."""
    h, w = depth.shape
    g = aif.mean(1, keepdim=True)
    gx = F.pad((g[..., :, 1:] - g[..., :, :-1]).abs(), (0, 1)); gy = F.pad((g[..., 1:, :] - g[..., :-1, :]).abs(), (0, 0, 0, 1))
    tex = F.avg_pool2d(gx + gy, 41, 1, 20)[0, 0]
    flat = F.avg_pool2d(depth[None, None], 41, 1, 20)[0, 0]            # prefer locally constant depth
    border = torch.zeros_like(depth, dtype=torch.bool)
    by, bx = int(h * .12), int(w * .08)
    border[by:h - by, bx:w - bx] = True
    pts = []
    for q in quantiles:
        t = torch.quantile(depth.flatten()[::7].float(), q)
        score = tex.clone()
        score[~border | ((depth - t).abs() > 0.03) | ((flat - depth).abs() > 0.02)] = -1
        for (px, py) in pts:                                           # keep the points apart on screen
            yy, xx = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij")
            score[((yy - py) ** 2 + (xx - px) ** 2) < (0.12 * w) ** 2] = -1
        i = int(score.argmax())
        pts.append((i % w, i // w))
    return pts


class Canvas:
    def __init__(self, pipe):
        self.pipe = pipe
        self.f_lab = font(27, 700); self.f_sub = font(21, 500); self.f_mk = font(28, 700)
        self.left = None

    def panel(self, im):
        return im.resize((PW, PH), Image.LANCZOS)

    def emit(self, right, label, sub="", cursor=None, ripple=0.0, marks=(), slider=None, n=1):
        c = Image.new("RGB", (CW, CH), BG)
        c.paste(self.left, (M, TOP)); c.paste(right, (2 * M + PW, TOP))
        d = ImageDraw.Draw(c, "RGBA")
        d.text((M + 2, 32), "Input photograph", font=self.f_lab, fill=DIM, anchor="lm")
        d.text((2 * M + PW + 2, 32), label, font=self.f_lab, fill=FG, anchor="lm")
        if sub:                                                          # secondary label as a chip
            x0 = 2 * M + PW + 2 + d.textlength(label, font=self.f_lab) + 16
            w = d.textlength(sub, font=self.f_sub)
            d.rounded_rectangle([x0, 16, x0 + w + 28, 48], 16, fill=(255, 228, 196, 36))
            d.text((x0 + 14, 32), sub, font=self.f_sub, fill=ACC, anchor="lm")
        ox, oy = 2 * M + PW, TOP
        for (x, y, t) in marks:
            r = 15
            d.ellipse([ox + x - r - 2, oy + y - r - 2, ox + x + r + 2, oy + y + r + 2], outline=(0, 0, 0, 170), width=7)
            d.ellipse([ox + x - r, oy + y - r, ox + x + r, oy + y + r], outline=(255, 255, 255, 255), width=3)
            d.ellipse([ox + x - 4, oy + y - 4, ox + x + 4, oy + y + 4], fill=(255, 255, 255, 255))
            if t:
                d.text((ox + x + r + 8, oy + y - r - 16), t, font=self.f_mk, fill=FG, stroke_width=3, stroke_fill=(0, 0, 0, 200))
        if slider is not None:
            x0, x1, y = ox + 28, ox + PW - 28, oy + PH - 34
            d.rounded_rectangle([x0 - 16, y - 22, x1 + 16, y + 22], 12, fill=(16, 21, 41, 200))
            d.line([x0, y, x1, y], fill=(255, 255, 255, 110), width=4)
            xs = x0 + (x1 - x0) * slider
            d.line([x0, y, xs, y], fill=ACC + (255,), width=4)
            d.ellipse([xs - 10, y - 10, xs + 10, y + 10], fill=(255, 255, 255, 255))
        if cursor is not None:
            x, y = ox + cursor[0], oy + cursor[1]
            if ripple > 0:
                r = 12 + 34 * ripple
                d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255, int(230 * (1 - ripple))), width=3)
            d.polygon([(x, y), (x, y + 24), (x + 6, y + 18), (x + 11, y + 28), (x + 15, y + 26), (x + 10, y + 16), (x + 18, y + 16)],
                      fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
        buf = np.asarray(c).tobytes()
        for _ in range(n):
            self.pipe.stdin.write(buf)


def ease(t):
    return t * t * (3 - 2 * t)


def to_pil(pred):
    return Image.fromarray((pred[0].float().clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy())


def demo(s1, rd):
    out = ASSETS / "demo.mp4"
    pipe = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CW}x{CH}", "-r", str(FPS),
                             "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    cv = Canvas(pipe)
    for si, (name, path) in enumerate(DEMO):
        src, aif = WORK / f"demo_{name}_in.png", WORK / f"demo_{name}_aif.png"
        fit(Image.open(path), 1024).save(src)
        s1(src, aif)
        rd.set_image(aif)
        W, H = Image.open(aif).size
        sx, sy = PW / W, PH / H
        p_in, p_aif = cv.panel(Image.open(src)), cv.panel(Image.open(aif))
        cv.left = p_in
        pts = pick_points(rd.depth[0], rd.src, [0.93, 0.08, 0.5, 0.75])
        print("demo", name, (W, H), pts, [round(float(rd.depth[0, y, x]), 3) for x, y in pts], flush=True)
        scr = [(x * sx, y * sy) for x, y in pts]

        cv.emit(p_in, "FlashReFocus", "input", n=int(FPS * 0.9))
        for i in range(int(FPS * 0.8)):                                  # wipe to the restored image
            t = ease((i + 1) / (FPS * 0.8)); cut = int(PW * t)
            fr = p_in.copy(); fr.paste(p_aif.crop((0, 0, cut, PH)), (0, 0))
            ImageDraw.Draw(fr).line([cut, 0, cut, PH], fill=FG, width=2)
            cv.emit(fr, "Restore once", "one-step diffusion")
        cv.emit(p_aif, "Restore once", "all-in-focus image", n=int(FPS * 0.9))

        cur, shown, strength = (PW * 0.5, PH * 0.55), p_aif, 0.8
        for k, (p, s) in enumerate(zip(pts, scr)):
            mv = int(FPS * 0.55)
            for i in range(mv):
                t = ease((i + 1) / mv)
                cv.emit(shown, "Click to refocus", "renderer only", cursor=(cur[0] + (s[0] - cur[0]) * t, cur[1] + (s[1] - cur[1]) * t),
                        marks=() if k == 0 else [(scr[k - 1][0], scr[k - 1][1], "")])
            cur = s
            new = cv.panel(to_pil(rd.render(p[0], p[1], strength, None)))
            for i in range(5):
                t = (i + 1) / 5
                cv.emit(Image.blend(shown, new, t), "Click to refocus", "renderer only", cursor=cur, ripple=t, marks=[(s[0], s[1], "")])
            shown = new
            cv.emit(shown, "Click to refocus", "renderer only", cursor=cur, marks=[(s[0], s[1], "")], n=int(FPS * 0.75))

        p, s = pts[-1], scr[-1]                                           # blur strength at the last focal point
        seq = [0.8 + (0.1 - 0.8) * ease(i / 23) for i in range(24)] + [0.1 + (1.0 - 0.1) * ease(i / 35) for i in range(36)] \
            + [1.0 + (0.6 - 1.0) * ease(i / 17) for i in range(18)]
        for v in seq:
            fr = cv.panel(to_pil(rd.render(p[0], p[1], v, None)))
            cv.emit(fr, "Blur strength", f"{v:.2f}", marks=[(s[0], s[1], "")], slider=v)
        cv.emit(fr, "Blur strength", f"{seq[-1]:.2f}", marks=[(s[0], s[1], "")], slider=seq[-1], n=int(FPS * 0.4))

        a, b = pts[1], pts[2]                                             # two focal planes: near and middle pick
        d0 = rd.depth
        da, db = float(d0[0, a[1], a[0]]), float(d0[0, b[1], b[0]])
        try:
            rd.depth = (da + torch.minimum((d0 - da).abs(), (d0 - db).abs())).clamp(0, 1)
            dual = cv.panel(to_pil(rd.render(a[0], a[1], 0.8, None)))
        finally:
            rd.depth = d0
        only_a = cv.panel(to_pil(rd.render(a[0], a[1], 0.8, None)))
        ma, mb = (scr[1][0], scr[1][1], "A"), (scr[2][0], scr[2][1], "B")
        for i in range(6):
            cv.emit(Image.blend(fr, only_a, (i + 1) / 6), "Two focal planes", "plane A", marks=[ma])
        cv.emit(only_a, "Two focal planes", "plane A", marks=[ma], n=int(FPS * 0.7))
        for i in range(6):
            cv.emit(Image.blend(only_a, dual, (i + 1) / 6), "Two focal planes", "A and B both sharp", marks=[ma, mb])
        cv.emit(dual, "Two focal planes", "A and B both sharp", marks=[ma, mb], n=int(FPS * 1.6))
        dual.save(WORK / f"demo_{name}_dual.jpg"); shown.save(WORK / f"demo_{name}_last.jpg")
        if si == 0:
            c = Image.new("RGB", (CW, CH), BG); c.paste(p_in, (M, TOP)); c.paste(shown, (2 * M + PW, TOP))
            d = ImageDraw.Draw(c); d.text((M + 2, 32), "Input photograph", font=cv.f_lab, fill=DIM, anchor="lm")
            d.text((2 * M + PW + 2, 32), "FlashReFocus", font=cv.f_lab, fill=FG, anchor="lm")
            c.save(ASSETS / "demo_poster.jpg", quality=86)
    pipe.stdin.close(); pipe.wait()
    print("wrote", out, out.stat().st_size, flush=True)


def chip(im, name, extra=""):
    """Method name (and seconds per edit) on a translucent rounded chip, top left."""
    d = ImageDraw.Draw(im, "RGBA"); f1, f2 = font(25, 700), font(21, 500)
    w1 = d.textlength(name, font=f1); w2 = d.textlength(extra, font=f2) if extra else 0
    d.rounded_rectangle([14, 14, 14 + 18 + w1 + (14 + w2 if extra else 0) + 18, 58], 22, fill=(16, 21, 41, 205))
    d.text((32, 36), name, font=f1, fill=FG, anchor="lm")
    if extra:
        d.text((32 + w1 + 14, 37), extra, font=f2, fill=ACC, anchor="lm")


def ring(im, x, y, r=13):
    d = ImageDraw.Draw(im, "RGBA")
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], outline=(0, 0, 0, 170), width=7)
    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255, 255), width=3)
    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255, 255))


def sbs():
    """FlashReFocus against GenRefocus on the eight focal points of a scene (blur-matched frames of the focus-sweep study)."""
    import json
    bench = DEFOCUS / "refocus_bench"
    for root, s, out, timed in ((bench / "rack_focus", "04", "04", True), (bench / "rack_focus", "59", "59", True),
                                (bench / "rack_focus_realbokeh", "133", "rb133", False)):
        info = json.loads((root / "meta_ours.json").read_text())["scenes"][s]
        tg = float(np.mean(json.loads((root / "meta_genrefocus.json").read_text())["seconds_per_frame"][s]))
        o = root / s; d = WORK / f"sbs_{out}"; d.mkdir(exist_ok=True)
        for i, p in enumerate(info["points"]):
            a = Image.open(o / "ours_matched" / f"{i:02d}.png").convert("RGB")
            b = Image.open(o / "genrefocus" / f"{i:02d}.png").convert("RGB").resize(a.size, Image.LANCZOS)
            for im, name, t in ((a, "FlashReFocus", f"{info['render_s_mean']:.2f} s / edit"), (b, "GenRefocus", f"{tg:.0f} s / edit")):
                ring(im, p["x"], p["y"]); chip(im, name, t if timed else "")
            c = Image.new("RGB", (a.width * 2 + 12, a.height), BG); c.paste(a, (0, 0)); c.paste(b, (a.width + 12, 0))
            c.save(d / f"{i:02d}.png")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", "2", "-i", str(d / "%02d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(ASSETS / f"sbs_{out}.mp4")], check=True)
        print("sbs", out, c.size, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["restore", "demo", "sbs"])
    a = ap.parse_args()
    ASSETS.mkdir(exist_ok=True); WORK.mkdir(exist_ok=True)
    if a.only in (None, "sbs"):
        sbs()
    if a.only != "sbs":
        FR.setup_backend(False)
        s1 = FR.FastStageI(str(STAGE1), str(REFINER), lambda_sem=1.25)
        if a.only != "demo":
            restore_pairs(s1)
        if a.only != "restore":
            demo(s1, FR.FastRenderer())
