#!/usr/bin/env python
"""Prepare the datasets under the data root (FLASHREFOCUS_DATA, default <repo>/data) after downloading them.

  python tools/prepare_data.py check            report which datasets are in place and which are missing
  python tools/prepare_data.py realbokeh        copy our focal-point annotations into RealBokeh/{train,validation}
                                                and build RealBokeh_val_f2.0/ (f/22 input, f/2.0 target, 219 pairs)
  python tools/prepare_data.py ebb              build EBB_Val176/ from EBB/validation (176 pairs with focal points)
  python tools/prepare_data.py lfdof_refocus    build the LFDOF-Refocus LVCorr benchmark (50 scenes) under the work root
  python tools/prepare_data.py 3creal           copy the 3CReal focal points next to the 3CReal validation pairs
  python tools/prepare_data.py all              everything above that its sources allow

Files are copied, never converted, so every metric is computed on the original bytes.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

ANN = paths.REPO / "annotations"

# folder -> what must exist inside it, and where it comes from (see README, "Data")
EXPECTED = {
    "dd_dp_dataset_png": (["train_c/source", "val_c/source", "test_c/source", "test_c/target"], "DPDD (Canon, PNG)"),
    "RealDOF": (["source", "target"], "RealDOF test set"),
    "LFDOF": (["train_data", "test_data/input", "test_data/ground_truth"], "LFDOF"),
    "div2k/DIV2K_train_HR": ([], "DIV2K train HR"),
    "RealBokeh": (["train/in", "train/gt", "validation/in", "validation/gt"], "RealBokeh_3MP"),
    "EBB": (["train/original", "train/bokeh", "validation/original", "validation/bokeh"], "EBB!"),
    "3CReal": (["validation/original", "validation/bokeh"], "3CReal"),
    "MODEST": ([], "MODEST (aperture ladders, optional)"),
    "Hypersim_processed": ([], "Hypersim with rendered bokeh targets (renderer pre-training only)"),
}


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def check(_):
    print(f"data root: {paths.DATA}")
    for folder, (subs, name) in EXPECTED.items():
        root = paths.DATA / folder
        missing = [s for s in subs if not (root / s).exists()]
        state = "missing" if not root.exists() else ("incomplete: " + ", ".join(missing) if missing else "ok")
        print(f"  {folder:<24} {state:<40} {name}")
    for built in ("EBB_Val176", "RealBokeh_val_f2.0"):
        print(f"  {built:<24} {'ok' if (paths.DATA / built / 'focus_point.json').exists() else 'not built (run prepare)'}")


def realbokeh(_):
    rb = paths.DATA / "RealBokeh"
    for split in ("train", "validation"):
        copy(ANN / "realbokeh" / split / "focus_points_f2.json", rb / split / "focus_points_f2.json")
    ann = json.loads((ANN / "realbokeh/validation/focus_points_f2.json").read_text())["annotations"]
    out = paths.DATA / "RealBokeh_val_f2.0"
    pts = {}
    for k, v in ann.items():
        copy(rb / v["f22_path"], out / "original" / f"{k}.jpg")
        copy(rb / v["f2_path"], out / "bokeh" / f"{k}.jpg")
        pts[k] = {"x": v["x"], "y": v["y"], "width": v["width"], "height": v["height"], "path": f"bokeh/{k}.jpg", "aperture": 2.0}
    (out / "focus_point.json").write_text(json.dumps({"source": "RealBokeh validation: f/22 input, f/2.0 target",
                                                      "annotations": pts}, indent=1))
    print(f"RealBokeh: annotations copied; {len(pts)} pairs -> {out}")


def ebb(_):
    src = paths.DATA / "EBB" / "validation"
    keys = (ANN / "ebb_val176/val176_list.txt").read_text().split()
    ann = json.loads((ANN / "ebb_val176/focus_points_f2.json").read_text())["annotations"]
    out = paths.DATA / "EBB_Val176"
    pts = {}
    for k in keys:
        copy(src / "original" / f"{k}.jpg", out / "original" / f"{k}.jpg")
        copy(src / "bokeh" / f"{k}.jpg", out / "bokeh" / f"{k}.jpg")
        v = ann[k]
        pts[k] = {"x": v["x"], "y": v["y"], "width": v["width"], "height": v["height"], "path": f"bokeh/{k}.jpg"}
    copy(ANN / "ebb_val176/focus_points_f2.json", src / "focus_points_f2.json")
    (out / "focus_point.json").write_text(json.dumps({"source": "EBB! validation, 176 pairs", "annotations": pts}, indent=1))
    print(f"EBB!: {len(pts)} pairs -> {out}")


def lfdof_refocus(_):
    """LFDOF test scenes at the largest aperture: one defocused focal slice as input, another as target, the
    all-in-focus image, and a focal point where the target slice is sharp (pairs.csv lists the choices)."""
    src = paths.DATA / "LFDOF" / "test_data"
    out = paths.WORK / "lfdof_refocus"
    with open(ANN / "lfdof_refocus/pairs.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        k, scene, ap = r["key"], r["scene"], r["aperture"]
        copy(src / "input" / scene / f"{scene}_{ap}_{int(r['source_plane']):02d}.png", out / "input" / f"{k}.png")
        copy(src / "input" / scene / f"{scene}_{ap}_{int(r['target_plane']):02d}.png", out / "target" / f"{k}.png")
        copy(src / "ground_truth" / f"{scene}.png", out / "aif" / f"{k}.png")
    copy(ANN / "lfdof_refocus/focus_points.json", out / "focus_points.json")
    copy(ANN / "lfdof_refocus/pairs.csv", out / "pairs.csv")
    print(f"LFDOF-Refocus: {len(rows)} scenes -> {out}")


def creal(_):
    copy(ANN / "3creal/focus_point.json", paths.DATA / "3CReal" / "validation" / "focus_point.json")
    print("3CReal: focal points copied")


def run_all(a):
    for name, fn, need in (("realbokeh", realbokeh, "RealBokeh/validation/gt"), ("ebb", ebb, "EBB/validation/bokeh"),
                           ("lfdof_refocus", lfdof_refocus, "LFDOF/test_data/input"), ("3creal", creal, "3CReal/validation")):
        if (paths.DATA / need).exists():
            fn(a)
        else:
            print(f"{name}: skipped ({paths.DATA / need} not found)")
    check(a)


if __name__ == "__main__":
    cmds = {"check": check, "realbokeh": realbokeh, "ebb": ebb, "lfdof_refocus": lfdof_refocus, "3creal": creal, "all": run_all}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=list(cmds))
    a = ap.parse_args()
    cmds[a.command](a)
