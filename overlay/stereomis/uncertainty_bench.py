"""Benchmark per-pixel geometric-uncertainty signals of trained Endo-4DGS models against stereo depth error.

For each model and test view it renders, besides colour and depth,
  acc     accumulated opacity (the rasteriser's weight map, used by Endo-4DGS as confidence)
  zstd    std of Gaussian depth along the ray (blended depth and depth^2 in one extra pass)
  motion  blended deformation magnitude |delta mu| of the Gaussians
and compares uncertainty signals with the per-pixel depth error vs SGBM stereo depth
(after per-frame median scaling, exactly as stereomis/eval_depth_stereo.py):
  u_cov   = 1 - acc                      (low coverage)
  u_ray   = zstd / depth                 (ray depth spread)
  u_prior = |depth - prior| / depth      (disagreement with the monocular prior, median-aligned)
  u_mot   = motion / depth               (deformation magnitude)
  u_ens   = relative std over 3 seeds    (seed ensemble)
  u_comb  = mean percentile rank of u_ray, u_prior, u_ens (training-free combination)
Metrics per frame: AUSE and AURG of the AbsRel sparsification curve, Spearman rho; mean over frames.
Also: AbsRel per image region (specular, blue device, depth edge, other tissue).

Run from the repo root: PYTHONPATH=. python stereomis/uncertainty_bench.py --out output/stereomis/uncertainty
"""
import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from eval_depth_stereo import stereo_depth  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams  # noqa: E402
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.params_utils import merge_hparams  # noqa: E402

SETUPS_P1 = {  # name: (prior folder, [model paths for seeds 0..2])
    "orig": ("depth_dam", [f"output/stereomis/dam_s{s}" for s in range(3)]),
    "fdam": ("depth_dam", [f"output/stereomis/f_dam_s{s}" for s in range(3)]),
    "fmogeps": ("depth_mogeps", ["output/stereomis/f_mogeps_s0", "output/stereomis/f_mogeps_s1", "output/stereomis/f_mogeps_s2"]),
    "finit": ("depth_mogeps", [f"output/stereomis/f_mogeps_init_s{s}" for s in range(3)]),
}
SETUPS_P25 = {
    "orig": ("depth_dam", [f"output/stereomis/p25_orig_s{s}" for s in range(3)]),
    "fdam": ("depth_dam", [f"output/stereomis/p25_fdam_s{s}" for s in range(3)]),
    "finit_raw": ("depth_moge", [f"output/stereomis/p25_finit_raw_s{s}" for s in range(3)]),
    "finit_sc": ("depth_mogesc", [f"output/stereomis/p25_finit_sc_s{s}" for s in range(3)]),
}
FRACS = np.linspace(0, 0.98, 50)
NBINS = 10


def load_model(path):
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
def render_aux(cam, gs):
    """Deformed Gaussians at the camera's time; returns depth, acc, zstd, motion maps (H, W)."""
    import math
    means3D = gs.get_xyz
    t = torch.tensor(cam.time, device="cuda").repeat(means3D.shape[0], 1)
    mu, sc, rot, op, shs = gs._deformation(means3D, gs._scaling, gs._rotation, gs._opacity, gs.get_features, t)
    sc, rot, op = gs.scaling_activation(sc), gs.rotation_activation(rot), gs.opacity_activation(op)
    W = cam.world_view_transform.cuda()
    z = (mu @ W[:3, :3] + W[3, :3])[:, 2:3]
    motion = (mu - means3D).norm(dim=1, keepdim=True)
    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0, viewmatrix=W,
        projmatrix=cam.full_proj_transform.cuda(), sh_degree=gs.active_sh_degree,
        campos=cam.camera_center.cuda(), prefiltered=False, debug=False)
    ras = GaussianRasterizer(raster_settings=settings)
    screen = torch.zeros_like(mu, requires_grad=False)
    out = {}
    c1, _, depth, acc = ras(means3D=mu, means2D=screen, colors_precomp=torch.cat([z, z * z, motion], 1),
                            opacities=op, scales=sc, rotations=rot)
    a = acc.clamp(min=1e-6)
    m1, m2 = c1[0] / a, c1[1] / a
    out["depth"] = depth.cpu().numpy()
    out["acc"] = acc.cpu().numpy()
    out["zstd"] = torch.sqrt((m2 - m1 * m1).clamp(min=0)).cpu().numpy()
    out["motion"] = (c1[2] / a).cpu().numpy()
    return out


def sparsification(u, e):
    """Mean error of the remaining pixels after removing the most uncertain fraction f."""
    order = np.argsort(-u)
    es = e[order]
    n = len(es)
    csum = np.cumsum(es[::-1])[::-1]  # suffix sums: mean of es[k:]
    ks = (FRACS * n).astype(int)
    return np.array([csum[k] / (n - k) for k in ks])


def ause_aurg(u, e):
    model = sparsification(u, e)
    oracle = sparsification(e, e)
    rand = np.full_like(model, e.mean())
    return float(np.trapz(model - oracle, FRACS)), float(np.trapz(rand - model, FRACS)), model, oracle


def region_masks(rgb, zs, valid):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    spec = (v > 0.9) & (s < 0.3)
    spec = cv2.dilate(spec.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    device = (h >= 80) & (h <= 130) & (s > 0.35) & (v > 0.15)
    device = cv2.morphologyEx(device.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    zf = cv2.GaussianBlur(np.where(valid, zs, np.median(zs[valid])).astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(zf, cv2.CV_32F, 1, 0), cv2.Sobel(zf, cv2.CV_32F, 0, 1)
    g = np.sqrt(gx ** 2 + gy ** 2) / np.maximum(zf, 1e-6)
    edge = valid & (g > np.percentile(g[valid], 90))
    # disjoint partition of the valid pixels, precedence device > specular > depth edge > tissue
    device = device & valid
    spec = spec & valid & ~device
    edge = edge & ~device & ~spec
    tissue = valid & ~device & ~spec & ~edge
    return {"specular": spec, "device": device, "depth_edge": edge, "tissue": tissue}


def main():
    ap = ArgumentParser(); ap.add_argument("--out", default="output/stereomis/uncertainty")
    ap.add_argument("--seq", choices=["p1", "p25"], default="p1"); ap.add_argument("--lr_check", action="store_true")
    ap.add_argument("--example", default=None, help="test frame to save maps for (default: middle test frame)")
    a = ap.parse_args()
    SETUPS = SETUPS_P1 if a.seq == "p1" else SETUPS_P25
    os.makedirs(a.out, exist_ok=True)
    results, examples = {}, {}
    for name, (prior_dir, paths) in SETUPS.items():
        per_seed = []
        for si, path in enumerate(paths):
            ds, pp, gs, scene = load_model(path)
            stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ds.source_path, "images")))
            test = [s for i, s in enumerate(stems) if (i - 1) % 8 == 0]
            frames = []
            for cam, stem in zip(scene.getTestCameras(), test):
                o = render_aux(cam, gs)
                o["stem"] = stem
                frames.append(o)
            per_seed.append(frames)
            src = ds.source_path
            del gs, scene; torch.cuda.empty_cache()
        metrics = {k: {"ause": [], "aurg": [], "rho": [], "rnd": [], "gain_pf": []} for k in ("u_cov", "u_ray", "u_prior", "u_mot", "u_ens", "u_comb")}
        regions = {r: [] for r in ("specular", "device", "depth_edge", "tissue")}
        region_frac = {r: [] for r in regions}
        region_err_sum = {r: 0.0 for r in regions}; total_err_sum = 0.0
        curves = {k: [] for k in metrics}; oracle_curves = []
        calib = {k: [] for k in metrics}
        example = a.example or per_seed[0][len(per_seed[0]) // 2]["stem"]
        for fi in range(len(per_seed[0])):
            stem = per_seed[0][fi]["stem"]
            zs = stereo_depth(stem, src, lr_check=a.lr_check)
            rgb = cv2.cvtColor(cv2.imread(os.path.join(src, "images", stem + ".png")), cv2.COLOR_BGR2RGB)
            prior = np.load(os.path.join(src, prior_dir, stem + ".npy"))[0]
            # seed ensemble: per-seed median-normalised depth
            stack = np.stack([s[fi]["depth"] for s in per_seed])
            norm = stack / np.median(stack.reshape(len(stack), -1), axis=1)[:, None, None]
            u_ens = norm.std(0) / np.maximum(norm.mean(0), 1e-6)
            for si, seed_frames in enumerate(per_seed):
                o = seed_frames[fi]
                d = o["depth"]
                # --- uncertainty signals: model and prior outputs only (no stereo mask) ---
                pv = np.isfinite(d) & (d > 0) & np.isfinite(prior) & (prior > 0)
                p = prior * np.median(d[pv] / prior[pv])
                # --- evaluation target: stereo-reference-relative, per-frame scale-aligned error ---
                valid = (zs > 0) & np.isfinite(d) & (d > 0)
                dsc = d * np.median(zs[valid] / d[valid])
                e = np.abs(dsc - zs) / zs
                sig = {
                    "u_cov": 1 - o["acc"],
                    "u_ray": o["zstd"] / np.maximum(d, 1e-6),
                    "u_prior": np.abs(d - p) / np.maximum(d, 1e-6),
                    "u_mot": o["motion"] / np.maximum(d, 1e-6),
                    "u_ens": u_ens,
                }
                m = valid & pv
                ev = e[m]
                ranks = [np.argsort(np.argsort(sig[k][m])) / m.sum() for k in ("u_ray", "u_prior", "u_ens")]
                comb = np.mean(ranks, 0)
                for k in metrics:
                    uv = comb if k == "u_comb" else sig[k][m]
                    au, ag, curve, oc = ause_aurg(uv, ev)
                    metrics[k]["ause"].append(au); metrics[k]["aurg"].append(ag)
                    rnd_area = float(np.trapz(oc[0] - oc, FRACS))  # AUSE of a random ordering on this frame
                    metrics[k]["rnd"].append(rnd_area); metrics[k]["gain_pf"].append(100 * ag / max(rnd_area, 1e-12))
                    metrics[k]["rho"].append(float(spearmanr(uv, ev).correlation))
                    curves[k].append(curve)
                    q = np.argsort(np.argsort(uv)) * NBINS // len(uv)  # uncertainty decile of each pixel
                    calib[k].append([float(ev[q == b].mean()) for b in range(NBINS)])
                oracle_curves.append(oc)
                rm = region_masks(rgb, zs, valid)
                total_err_sum += float(e[valid].sum())
                for r, mk in rm.items():
                    region_err_sum[r] += float(e[mk].sum())
                    if mk.sum() > 50:
                        regions[r].append(float(e[mk].mean())); region_frac[r].append(float(mk.sum() / valid.sum()))
                if si == 0 and stem == example:
                    examples[name] = {"rgb": rgb, "err": np.where(valid, e, np.nan), **{k: np.where(valid, v, np.nan) for k, v in sig.items()},
                                      "regions": {r: mk for r, mk in rm.items()}}
        results[name] = {
            "uncertainty": {k: {mm: [float(np.mean(v)), float(np.std(v))] for mm, v in d.items()} for k, d in metrics.items()},
            "curves": {k: np.mean(v, 0).tolist() for k, v in curves.items()}, "oracle": np.mean(oracle_curves, 0).tolist(),
            "calibration_deciles": {k: np.mean(v, 0).tolist() for k, v in calib.items()},
            "region_absrel": {r: [float(np.mean(v)), float(np.std(v)), len(v)] for r, v in regions.items()},
            "region_fraction": {r: float(np.mean(v)) for r, v in region_frac.items()},
            "region_error_share": {r: region_err_sum[r] / total_err_sum for r in regions},
            "gain_ratio_of_means": {k: 100 * float(np.mean(d["aurg"])) / float(np.mean(d["rnd"])) for k, d in metrics.items()},
            "gain_mean_of_ratios": {k: float(np.mean(d["gain_pf"])) for k, d in metrics.items()},
        }
        print(name, json.dumps({k: round(v["ause"][0], 4) for k, v in results[name]["uncertainty"].items()}),
              json.dumps({r: round(v[0], 3) for r, v in results[name]["region_absrel"].items()}), flush=True)
    tag = f"{a.seq}{'_lr' if a.lr_check else ''}"
    json.dump({"fracs": FRACS.tolist(), "seq": a.seq, "lr_check": a.lr_check, "setups": results},
              open(os.path.join(a.out, f"uncertainty_{tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, f"examples_{tag}.npz"),
                        **{f"{n}_{k}": v for n, ex in examples.items() for k, v in ex.items() if k != "regions"},
                        **{f"{n}_region_{r}": m for n, ex in examples.items() for r, m in ex["regions"].items()})


if __name__ == "__main__":
    main()
