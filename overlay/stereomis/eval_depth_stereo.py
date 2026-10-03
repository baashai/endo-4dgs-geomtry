"""Geometry check for StereoMIS runs: rendered depth of the test views vs. SGBM stereo depth.

Depth is compared after per-frame median scaling (monocular priors have no reliable scale).
Run from the repo root:
  python stereomis/eval_depth_stereo.py output/stereomis/split_1 output/stereomis/split_1_moge ...
"""
import os
import sys

import cv2
import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.params_utils import merge_hparams

RAW = "data/StereoMIS/P1"
BF_HALF = 4989.29218564127 / 2  # focal * baseline of the rectified pair, at the 640x512 image size
SGBM = cv2.StereoSGBM_create(minDisparity=0, numDisparities=256, blockSize=5, P1=8 * 3 * 25, P2=32 * 3 * 25,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)


def stereo_depth(stem, root=None, lr_check=False):
    """SGBM depth of a rectified 640x512 pair. `root`: a split folder; if it has calib.json
    (stereomis/prepare_sequence.py) its images/, images_right/ and bf are used, else StereoMIS P1.
    lr_check: keep only pixels whose left and right disparities agree within 1 px."""
    if root is not None and os.path.exists(os.path.join(root, "calib.json")):
        import json
        bf = json.load(open(os.path.join(root, "calib.json")))["bf"]
        left = cv2.imread(os.path.join(root, "images", stem + ".png"))
        right = cv2.imread(os.path.join(root, "images_right", stem + ".png"))
    else:
        bf = BF_HALF
        left = cv2.imread(os.path.join(RAW, "images_1", stem + ".png"))
        right = cv2.imread(os.path.join(RAW, "images_right_1", stem + ".png"))
    disp = SGBM.compute(left, right).astype(np.float32) / 16
    valid = disp > 1
    if lr_check:
        # right-view disparity via the mirrored pair; a left pixel x maps to x - d in the right view
        disp_r = np.fliplr(SGBM.compute(np.fliplr(right).copy(), np.fliplr(left).copy()).astype(np.float32) / 16)
        h, w = disp.shape
        xs = np.arange(w)[None].repeat(h, 0)
        xr = np.clip(np.round(xs - disp).astype(int), 0, w - 1)
        dr = disp_r[np.arange(h)[:, None], xr]
        valid &= (dr > 1) & (np.abs(disp - dr) <= 1.0)
    depth = np.zeros_like(disp)
    depth[valid] = bf / disp[valid]
    return depth


def evaluate(model_path):
    import mmcv
    from argparse import ArgumentParser
    parser = ArgumentParser()
    model, pipeline, hyper = ModelParams(parser, sentinel=True), PipelineParams(parser), ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    sys.argv = ["x", "--model_path", model_path]
    args = get_combined_args(parser)
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    ds, hp, pp = model.extract(args), hyper.extract(args), pipeline.extract(args)

    absrel, delta = [], []
    with torch.no_grad():
        gaussians = GaussianModel(ds.sh_degree, hp)
        scene = Scene(ds, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ds.source_path, "images")))
        test_stems = [s for i, s in enumerate(stems) if (i - 1) % 8 == 0]  # EndoNeRF_Dataset test split
        for view, stem in zip(scene.getTestCameras(), test_stems):
            pred = render(view, gaussians, pp, bg, stage="fine")["depth"][0].cpu().numpy()
            gt = stereo_depth(stem, ds.source_path)
            valid = (gt > 0) & (pred > 0)
            p = pred[valid] * np.median(gt[valid] / pred[valid])
            g = gt[valid]
            absrel.append(np.mean(np.abs(p - g) / g))
            delta.append(np.mean(np.maximum(p / g, g / p) < 1.25))
    return np.mean(absrel), np.mean(delta)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        a, d = evaluate(path)
        print(f"{path}: AbsRel {a:.4f}  delta<1.25 {d:.4f}")
