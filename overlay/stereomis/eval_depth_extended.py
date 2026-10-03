"""Extended depth evaluation against SGBM stereo (review findings F02, F21).

For each model, on the test views, reports
  aligned   per-frame median-scaled AbsRel and delta<1.25 (shape agreement; the paper's main metric)
  metric    AbsRel of the raw rendered depth (model and poses are in mm)
  fixed     AbsRel after ONE scale factor fitted on the TRAINING views (median of stereo/rendered)
  scale     per-test-frame median(stereo/rendered): mean, std, coefficient of variation, min, max
Masks: stereo disparity > 1 px (depth > 0), rendered depth finite and > 0.
  PYTHONPATH=. python stereomis/eval_depth_extended.py --out output/stereomis/depth_extended.json name=path ...
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from eval_depth_stereo import stereo_depth  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.params_utils import merge_hparams  # noqa: E402


def load(path):
    import mmcv
    parser = ArgumentParser()
    model, pipeline, hyper = ModelParams(parser, sentinel=True), PipelineParams(parser), ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int); parser.add_argument("--configs", type=str)
    sys.argv = ["x", "--model_path", path]
    args = merge_hparams(get_combined_args(parser), mmcv.Config.fromfile("arguments/stereomis.py"))
    ds, hp, pp = model.extract(args), hyper.extract(args), pipeline.extract(args)
    gs = GaussianModel(ds.sh_degree, hp)
    scene = Scene(ds, gs, load_iteration=-1, shuffle=False)
    return ds, pp, gs, scene


@torch.no_grad()
def depths(cams, stems, gs, pp, src):
    bg = torch.ones(3, device="cuda")
    for cam, stem in zip(cams, stems):
        d = render(cam, gs, pp, bg, stage="fine", mode="test")["depth"][0].cpu().numpy()
        z = stereo_depth(stem, src)
        v = (z > 0) & np.isfinite(d) & (d > 0)
        yield d, z, v


def main():
    ap = ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for item in a.models:
        name, path = item.split("=", 1)
        ds, pp, gs, scene = load(path)
        src = ds.source_path
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(src, "images")))
        train = [s for i, s in enumerate(stems) if (i - 1) % 8 != 0]
        test = [s for i, s in enumerate(stems) if (i - 1) % 8 == 0]
        # one global scale from the training views only
        ratios = [np.median(z[v] / d[v]) for d, z, v in depths(scene.getTrainCameras(), train, gs, pp, src)]
        s_fix = float(np.median(ratios))
        out = {"aligned": [], "delta_aligned": [], "metric": [], "fixed": [], "scale": [], "valid": []}
        for d, z, v in depths(scene.getTestCameras(), test, gs, pp, src):
            dv, zv = d[v], z[v]
            s = float(np.median(zv / dv))
            pa = dv * s
            out["aligned"].append(float(np.mean(np.abs(pa - zv) / zv)))
            out["delta_aligned"].append(float(np.mean(np.maximum(pa / zv, zv / pa) < 1.25)))
            out["metric"].append(float(np.mean(np.abs(dv - zv) / zv)))
            out["fixed"].append(float(np.mean(np.abs(dv * s_fix - zv) / zv)))
            out["scale"].append(s); out["valid"].append(float(v.mean()))
        sc = np.array(out["scale"])
        res[name] = {"path": path, "train_scale": s_fix, "n_train": len(train), "test_frames": test,
                     **{k: float(np.mean(v)) for k, v in out.items() if k != "scale"},
                     "scale_mean": float(sc.mean()), "scale_std": float(sc.std(ddof=1)),
                     "scale_cv": float(sc.std(ddof=1) / sc.mean()), "scale_min": float(sc.min()), "scale_max": float(sc.max()),
                     "per_frame": out}
        print(f"{name}: aligned {res[name]['aligned']:.4f} metric {res[name]['metric']:.4f} fixed {res[name]['fixed']:.4f} "
              f"train_scale {s_fix:.4f} test-scale {sc.mean():.4f}±{sc.std(ddof=1):.4f} (cv {res[name]['scale_cv']:.3f})", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        del gs, scene; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
