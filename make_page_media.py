#!/usr/bin/env python
"""Media for the project page, produced with the paper model (Stage I v3_s1 40K + refiner, omega 1.25; bf16 renderer).

  restore pairs : assets/restore_<name>_{in,out}.jpg, input and restored all-in-focus image (drag-to-compare sliders)
  comparisons   : assets/sbs_<scene>.mp4, FlashReFocus against GenRefocus on the same eight focal points
  demo video    : assets/demo.mp4, a scripted session per scene played in real time against GenRefocus: first refocus,
                  further focal points, blur strength, two focal planes. Every frame on the right is an output of the
                  pipeline; each result appears after the time measured at 1536x1536 (cost table of the paper).

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
        ("flower", DATA / "Refocus_samples/real_19.jpg")]

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


# Timeline of the demo video, in seconds: the paper's measurements at 1536x1536 on one RTX PRO 6000 (cost table).
T_FIRST, T_EDIT, G_FIRST = 2.50, 0.47, 370.65


def clock_text(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


class Canvas:
    """Two panels played in real time: GenRefocus still working on its first refocus (left) and a FlashReFocus session
    (right). `t` is the time since the first click; it advances by one frame per emitted frame while `running`."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.f_lab = font(27, 700); self.f_sub = font(21, 500); self.f_mk = font(28, 700)
        self.f_clock = font(30, 700); self.f_big = font(24, 600)
        self.left = None
        self.t, self.running, self.edits, self.busy = 0.0, False, 0, None    # busy: (start, duration) of the current job

    def panel(self, im):
        return im.resize((PW, PH), Image.LANCZOS)

    def chip(self, d, x0, text, fill=ACC):
        w = d.textlength(text, font=self.f_sub)
        d.rounded_rectangle([x0, 16, x0 + w + 28, 48], 16, fill=fill[:3] + (36,))
        d.text((x0 + 14, 32), text, font=self.f_sub, fill=fill, anchor="lm")
        return x0 + w + 28

    def mono(self, d, x_right, y, text, f, fill, adv=18):
        """Right-aligned text with a fixed advance per character, so the running clock does not jitter."""
        x = x_right
        for ch in reversed(text):
            w = adv if ch.isdigit() else adv * 0.55
            d.text((x - w / 2, y), ch, font=f, fill=fill, anchor="mm"); x -= w

    def bar(self, d, ox, frac, left, right, color):
        x0, x1, y = ox + 20, ox + PW - 20, TOP + PH - 40
        d.rounded_rectangle([x0, y - 24, x1, y + 24], 14, fill=(16, 21, 41, 215))
        d.text((x0 + 18, y - 1), left, font=self.f_sub, fill=FG, anchor="lm")
        tw = d.textlength(right, font=self.f_sub)
        d.text((x1 - 18, y - 1), right, font=self.f_sub, fill=color, anchor="rm")
        bx0, bx1 = x0 + 18 + d.textlength(left, font=self.f_sub) + 18, x1 - 18 - tw - 18
        d.line([bx0, y, bx1, y], fill=(255, 255, 255, 70), width=6)
        if frac > 0:
            d.line([bx0, y, bx0 + (bx1 - bx0) * min(1.0, frac), y], fill=color + (255,), width=6)

    def emit(self, right, action="", cursor=None, ripple=0.0, marks=(), slider=None, n=1, card=False):
        base = Image.new("RGB", (CW, CH), BG)
        base.paste(self.left, (M, TOP)); base.paste(right, (2 * M + PW, TOP))
        ox, oy = 2 * M + PW, TOP
        for _ in range(n):
            c = base.copy(); d = ImageDraw.Draw(c, "RGBA"); t = self.t
            d.text((M + 2, 32), "GenRefocus", font=self.f_lab, fill=FG, anchor="lm")
            self.chip(d, M + 2 + d.textlength("GenRefocus", font=self.f_lab) + 16, "same photo, same start", DIM)
            d.text((ox + 2, 32), "FlashReFocus", font=self.f_lab, fill=FG, anchor="lm")
            if action:
                self.chip(d, ox + 2 + d.textlength("FlashReFocus", font=self.f_lab) + 16, action)
            self.mono(d, CW - M - 4, 32, clock_text(t), self.f_clock, FG)
            self.bar(d, M, t / G_FIRST, "First refocus", f"{clock_text(t)} / {clock_text(G_FIRST)}", DIM)
            for (x, y, tx) in marks:
                r = 15
                d.ellipse([ox + x - r - 2, oy + y - r - 2, ox + x + r + 2, oy + y + r + 2], outline=(0, 0, 0, 170), width=7)
                d.ellipse([ox + x - r, oy + y - r, ox + x + r, oy + y + r], outline=(255, 255, 255, 255), width=3)
                d.ellipse([ox + x - 4, oy + y - 4, ox + x + 4, oy + y + 4], fill=(255, 255, 255, 255))
                if tx:
                    d.text((ox + x + r + 8, oy + y - r - 16), tx, font=self.f_mk, fill=FG, stroke_width=3, stroke_fill=(0, 0, 0, 200))
            if slider is not None:                                       # blur-strength control, above the status bar
                x0, x1, y = ox + 40, ox + PW - 40, oy + PH - 100
                d.rounded_rectangle([x0 - 20, y - 20, x1 + 20, y + 20], 12, fill=(16, 21, 41, 200))
                d.line([x0, y, x1, y], fill=(255, 255, 255, 110), width=4)
                xs = x0 + (x1 - x0) * slider
                d.line([x0, y, xs, y], fill=ACC + (255,), width=4)
                d.ellipse([xs - 10, y - 10, xs + 10, y + 10], fill=(255, 255, 255, 255))
            if self.busy is not None:
                frac = (t - self.busy[0]) / self.busy[1]
                name = "First refocus" if self.edits == 0 else f"Edit {self.edits + 1}"
                self.bar(d, ox, frac, name, f"{max(0.0, t - self.busy[0]):.2f} s", ACC)
            else:
                done = "Click a focal point" if self.edits == 0 else f"{self.edits} edit{'s' if self.edits > 1 else ''} done"
                self.bar(d, ox, 0, done, f"{T_EDIT:.2f} s per edit" if self.edits else "", ACC)
            if cursor is not None:
                x, y = ox + cursor[0], oy + cursor[1]
                if ripple > 0:
                    r = 12 + 34 * ripple
                    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255, int(230 * (1 - ripple))), width=3)
                d.polygon([(x, y), (x, y + 24), (x + 6, y + 18), (x + 11, y + 28), (x + 15, y + 26), (x + 10, y + 16), (x + 18, y + 16)],
                          fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
            if card:                                                     # summary across both panels
                txt = f"After {t:.0f} s:  FlashReFocus {self.edits} edits   ·   GenRefocus {100 * t / G_FIRST:.0f}% of its first"
                w = d.textlength(txt, font=self.f_big); cx, cy = CW / 2, TOP + PH * 0.5
                d.rounded_rectangle([cx - w / 2 - 30, cy - 34, cx + w / 2 + 30, cy + 34], 20, fill=(16, 21, 41, 235), outline=ACC + (255,), width=2)
                d.text((cx, cy), txt, font=self.f_big, fill=FG, anchor="mm")
            self.last = c
            self.pipe.stdin.write(np.asarray(c).tobytes())
            if self.running:
                self.t += 1.0 / FPS


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
        p_in = cv.panel(Image.open(src))
        cv.left, cv.t, cv.running, cv.edits, cv.busy = p_in, 0.0, False, 0, None
        pts = pick_points(rd.depth[0], rd.src, [0.93, 0.08, 0.5, 0.75, 0.3])
        print("demo", name, (W, H), pts, [round(float(rd.depth[0, y, x]), 3) for x, y in pts], flush=True)
        scr = [(x * sx, y * sy) for x, y in pts]
        st = {"cur": (PW * 0.5, PH * 0.6), "shown": p_in, "marks": [], "slider": None}

        def move(to, action, dur=0.6):
            n = int(FPS * dur)
            for i in range(n):
                t = ease((i + 1) / n)
                cv.emit(st["shown"], action, cursor=(st["cur"][0] + (to[0] - st["cur"][0]) * t, st["cur"][1] + (to[1] - st["cur"][1]) * t),
                        marks=st["marks"], slider=st["slider"])
            st["cur"] = to

        def job(new, action, dur, marks, hold):
            """The request starts now; `new` appears `dur` seconds later, as measured, and counts as one edit."""
            cv.running = True
            cv.busy = (cv.t, dur); st["marks"] = marks
            n = round(dur * FPS)
            for i in range(n):
                cv.emit(st["shown"], action, cursor=st["cur"], ripple=min(1.0, (i + 1) / 6) if i < 6 else 0.0, marks=marks, slider=st["slider"])
            cv.busy = None; cv.edits += 1; st["shown"] = new
            cv.emit(new, action, cursor=st["cur"], marks=marks, slider=st["slider"], n=int(FPS * hold))

        cv.emit(p_in, "click to refocus", cursor=st["cur"], n=int(FPS * 0.5))
        move(scr[0], "click to refocus")
        strength = 0.8                                                    # first refocus: restoration and the first rendering
        job(cv.panel(to_pil(rd.render(pts[0][0], pts[0][1], strength, None))), "restore once, render", T_FIRST, [(scr[0][0], scr[0][1], "")], 0.7)
        for p, s in zip(pts[1:], scr[1:]):                                # further focal points: renderer only
            move(s, "click to refocus")
            job(cv.panel(to_pil(rd.render(p[0], p[1], strength, None))), "click to refocus", T_EDIT, [(s[0], s[1], "")], 0.55)

        p, s = pts[-1], scr[-1]                                           # blur strength at the last focal point, in steps
        for v in (0.3, 0.55, 1.0):
            v0 = strength; n = int(FPS * 0.35)
            for i in range(n):
                st["slider"] = v0 + (v - v0) * ease((i + 1) / n)
                cv.emit(st["shown"], f"blur strength {st['slider']:.2f}", marks=st["marks"], slider=st["slider"])
            strength = v
            job(cv.panel(to_pil(rd.render(p[0], p[1], v, None))), f"blur strength {v:.2f}", T_EDIT, st["marks"], 0.45)
        st["slider"] = None

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
        move(scr[1], "two focal planes")
        job(only_a, "two focal planes", T_EDIT, [ma], 0.5)
        move(scr[2], "two focal planes")
        job(dual, "two focal planes", T_EDIT, [ma, mb], 1.0)
        cv.emit(dual, "two focal planes", marks=[ma, mb], n=int(FPS * 2.2), card=True)
        print("  session", round(cv.t, 1), "s,", cv.edits, "edits", flush=True)
        if si == 0:
            cv.last.save(ASSETS / "demo_poster.jpg", quality=86)
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
