"""MoGe-2 geometry prior (counterpart of pre_dam_dep.py).

Writes, next to <dataset_root>/<rgb_paths>/:
  depth_moge/<name>.npy   (1, H, W) float32 metric depth in `--unit` (default mm, the unit of
                          the StereoMIS poses); 0 where MoGe-2's validity mask is off
  normal_moge/<name>.npy  (H, W, 3) float16 normals, OpenCV camera frame (x right, y down, z fwd)
  points_moge/<name>.npy  (H, W, 3) float16 point map, same frame and unit (only with --save_points)

Runs in the separate `moge` conda env (MoGe needs Python >= 3.10):
  python scripts/pre_moge_dep.py --dataset_root data/stereomis/split_1 --fov_x 58.72
Train with `depth_prior="moge"` (see arguments/stereomis_moge.py).
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from moge.model.v2 import MoGeModel

UNITS = {"m": 1.0, "mm": 1000.0}


def main(args):
    img_dir = os.path.join(args.dataset_root, args.rgb_paths)
    out = {k: img_dir.replace(args.rgb_paths, f"{k}_moge") for k in ("depth", "normal", "points")}
    os.makedirs(out["depth"], exist_ok=True)
    os.makedirs(out["normal"], exist_ok=True)
    if args.save_points:
        os.makedirs(out["points"], exist_ok=True)

    device = torch.device("cuda")
    model = MoGeModel.from_pretrained(args.model).to(device).eval()
    scale = UNITS[args.unit]

    fovs = []
    for name in tqdm(sorted(os.listdir(img_dir))):
        color = np.array(Image.open(os.path.join(img_dir, name)).convert("RGB"))
        image = torch.tensor(color / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            pred = model.infer(image, fov_x=args.fov_x, resolution_level=args.resolution_level)

        valid = pred["mask"].bool() & torch.isfinite(pred["depth"])
        depth = torch.where(valid, pred["depth"] * scale, torch.zeros_like(pred["depth"]))
        stem = os.path.splitext(name)[0]
        np.save(os.path.join(out["depth"], stem + ".npy"), depth[None].cpu().numpy().astype(np.float32))
        if "normal" in pred:
            np.save(os.path.join(out["normal"], stem + ".npy"), pred["normal"].cpu().numpy().astype(np.float16))
        if args.save_points:
            points = torch.where(valid[..., None], pred["points"] * scale, torch.zeros_like(pred["points"]))
            np.save(os.path.join(out["points"], stem + ".npy"), points.cpu().numpy().astype(np.float16))

        fx = pred["intrinsics"][0, 0].item()  # normalized by image width
        fovs.append(np.degrees(2 * np.arctan(0.5 / fx)))
        d = depth[valid]
        tqdm.write(f"{stem}: valid {valid.float().mean().item():.3f}  depth[{args.unit}] "
                   f"min {d.min().item():.1f} med {d.median().item():.1f} max {d.max().item():.1f}")
    print(f"fov_x used/estimated: mean {np.mean(fovs):.2f} deg")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/stereomis/split_1")
    parser.add_argument("--rgb_paths", type=str, default="images")
    parser.add_argument("--model", type=str, default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--fov_x", type=float, default=None,
                        help="known horizontal FoV in degrees; estimated by MoGe-2 if omitted")
    parser.add_argument("--resolution_level", type=int, default=9)
    parser.add_argument("--unit", choices=list(UNITS), default="mm")
    parser.add_argument("--save_points", action="store_true")
    main(parser.parse_args())
