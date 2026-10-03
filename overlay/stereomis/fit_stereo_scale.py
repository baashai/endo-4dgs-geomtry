"""Stereo-fitted global scale of a monocular depth prior (stereo-assisted protocol).

Median over every `--every`-th frame of the split (training frames: with every=10 none is a test frame,
(i-1) % 8 == 0) of the per-frame median ratio of SGBM stereo depth to prior depth.
Used for P1 (x0.1645, ablations "stereo-fitted") and P2_5 (x0.0572, static camera).
  PYTHONPATH=. python stereomis/fit_stereo_scale.py --root data/stereomis/split_1 --prior moge --out mogesc
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_depth_stereo import stereo_depth  # noqa: E402

ap = ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--prior", default="moge")
ap.add_argument("--every", type=int, default=10)
ap.add_argument("--out", default=None, help="write depth_<out>/ = depth_<prior>/ * scale")
a = ap.parse_args()
stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(a.root, "images")))
cal = [i for i in range(0, len(stems), a.every)]
assert not any((i - 1) % 8 == 0 for i in cal), "calibration frames overlap the test split"
ratios = []
for i in cal:
    z = stereo_depth(stems[i], a.root)
    p = np.load(os.path.join(a.root, f"depth_{a.prior}", stems[i] + ".npy"))[0]
    v = (z > 0) & (p > 0)
    ratios.append(np.median(z[v] / p[v]))
scale = float(np.median(ratios))
print(f"stereo-fitted scale {scale:.4f} from frames {[stems[i] for i in cal]}")
if a.out:
    src, dst = os.path.join(a.root, f"depth_{a.prior}"), os.path.join(a.root, f"depth_{a.out}")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".npy"):
            np.save(os.path.join(dst, f), np.load(os.path.join(src, f)) * scale)
    open(os.path.join(dst, "SCALE.txt"), "w").write(f"{scale}\n")
