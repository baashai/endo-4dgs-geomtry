"""Save rendered depth of one test frame for several models, plus the stereo reference (for figures).
  PYTHONPATH=. python stereomis/dump_frame.py --stem 000898 --out output/stereomis/frame_000898.npz name=path ...
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from eval_depth_extended import load  # noqa: E402
from eval_depth_stereo import stereo_depth  # noqa: E402
from gaussian_renderer import render  # noqa: E402

ap = ArgumentParser(); ap.add_argument("--stem", required=True); ap.add_argument("--out", required=True)
ap.add_argument("models", nargs="+"); a = ap.parse_args()
out = {}
for item in a.models:
    name, path = item.split("=", 1)
    ds, pp, gs, scene = load(path)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ds.source_path, "images")))
    test = [s for i, s in enumerate(stems) if (i - 1) % 8 == 0]
    cam = scene.getTestCameras()[test.index(a.stem)]
    with torch.no_grad():
        out[name] = render(cam, gs, pp, torch.ones(3, device="cuda"), stage="fine", mode="test")["depth"][0].cpu().numpy()
    out["stereo"] = stereo_depth(a.stem, ds.source_path)
    del gs, scene; torch.cuda.empty_cache()
np.savez_compressed(a.out, **out)
print("saved", sorted(out))
