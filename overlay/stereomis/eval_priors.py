"""Prior-only evaluation against SGBM stereo on the test frames (review F31).

For each (split, prior folder): raw scale factor median(z_stereo / prior) per frame, then after per-frame
median alignment the mean AbsRel and the median relative error, and the valid-pixel fraction.
  python stereomis/eval_priors.py --out output/stereomis/priors.json
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_depth_stereo import stereo_depth  # noqa: E402

CASES = [("P1", "data/stereomis/split_1", "depth_dam", "Depth-Anything ViT-B"),
         ("P1", "data/stereomis/split_1", "depth_moge", "MoGe-2 ViT-L (metric)"),
         ("P2_5", "data/stereomis/p2_5", "depth_dam", "Depth-Anything ViT-B"),
         ("P2_5", "data/stereomis/p2_5", "depth_moge", "MoGe-2 ViT-L (metric)")]

ap = ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
res = {}
for seq, root, prior, label in CASES:
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "images")))
    test = [s for i, s in enumerate(stems) if (i - 1) % 8 == 0]
    sc, absrel, medrel, valid = [], [], [], []
    for stem in test:
        z = stereo_depth(stem, root)
        p = np.load(os.path.join(root, prior, stem + ".npy"))[0]
        v = (z > 0) & np.isfinite(p) & (p > 0)
        s = np.median(z[v] / p[v]); r = np.abs(p[v] * s - z[v]) / z[v]
        sc.append(s); absrel.append(r.mean()); medrel.append(np.median(r)); valid.append(v.mean())
    res[f"{seq}/{prior}"] = {"sequence": seq, "prior": label, "frames": len(test),
                             "raw_scale_mean": float(np.mean(sc)), "raw_scale_cv": float(np.std(sc, ddof=1) / np.mean(sc)),
                             "aligned_absrel": float(np.mean(absrel)), "median_rel_error": float(np.mean(medrel)),
                             "valid_fraction": float(np.mean(valid))}
    print(seq, label, {k: round(v, 4) for k, v in res[f"{seq}/{prior}"].items() if isinstance(v, float)})
json.dump(res, open(a.out, "w"), indent=1)
