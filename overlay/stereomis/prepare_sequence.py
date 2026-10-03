"""Prepare any StereoMIS sequence/frame range for Endo-4DGS (generalises process_dataset.py + make_split.py).

Writes data/stereomis/<name>/ with
  images/          rectified left frames, 640x512
  images_right/    rectified right frames (evaluation only)
  masks/           3-channel masks (nearest raw mask per frame)
  poses_bounds.npy poses as in pre_pose_stereomis.py; focal = rectified focal at 640x512
  calib.json       fx, cx, cy, bf at 640x512 (used by the stereo evaluation and the prior scale)
With --orig_name it also writes data/stereomis/<orig_name>/ (symlinked images/masks) whose poses
store the unrectified full-resolution focal, reproducing the released Endo-4DGS preprocessing.

  python stereomis/prepare_sequence.py --seq P2_5 --start 5301 --end 5499 --step 2 --name p2_5 --orig_name p2_5_orig
"""
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(__file__))
from rectification import StereoRectifier  # noqa: E402

ap = ArgumentParser()
ap.add_argument("--raw_root", default="data/StereoMIS")
ap.add_argument("--seq", required=True)
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--name", required=True)
ap.add_argument("--orig_name", default=None)
ap.add_argument("--out_root", default="data/stereomis")
a = ap.parse_args()

raw = os.path.join(a.raw_root, a.seq)
calib_file = os.path.join(raw, "StereoCalibration.ini")
rect = StereoRectifier(calib_file, img_size_new=None)
rc = rect.get_rectified_calib()
K = np.asarray(rc["intrinsics"]["left"])
calib = {"fx": float(K[0, 0] / 2), "cx": float(K[0, 2] / 2), "cy": float(K[1, 2] / 2), "bf": float(rc["bf"] / 2),
         "baseline_mm": float(rc["b"]), "seq": a.seq, "frames": [a.start, a.end, a.step]}
fx_unrect = None
for line in open(calib_file):
    if line.strip().startswith("fc_x"):
        fx_unrect = float(line.split("=")[1]); break

out = os.path.join(a.out_root, a.name)
for d in ("images", "images_right", "masks"):
    os.makedirs(os.path.join(out, d), exist_ok=True)
wanted = set(range(a.start, a.end + 1, a.step))
cap = cv2.VideoCapture(glob(os.path.join(raw, "*.mp4"))[0])
idx = 0
while idx <= a.end:
    ok, frame = cap.read()
    if not ok:
        break
    if idx in wanted:
        h, w, _ = frame.shape
        left = torch.from_numpy(cv2.cvtColor(frame[:h // 2], cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        right = torch.from_numpy(cv2.cvtColor(frame[h // 2:], cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        lc, rcimg = rect(left, right)
        for img, sub in ((lc, "images"), (rcimg, "images_right")):
            im = cv2.resize(img.permute(1, 2, 0).numpy(), (w // 2, h // 4))
            cv2.imwrite(os.path.join(out, sub, f"{idx:06d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR).astype(np.uint8))
    idx += 1
n = len(os.listdir(os.path.join(out, "images")))
assert n == len(wanted), f"extracted {n} of {len(wanted)} frames"

raw_masks = {int(m[:6]): m for m in os.listdir(os.path.join(raw, "masks"))}
for i in sorted(wanted):
    m = cv2.imread(os.path.join(raw, "masks", raw_masks[min(raw_masks, key=lambda k: abs(k - i))]))
    m = cv2.resize(m, (640, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out, "masks", f"{i:06d}.png"), m)


def write_poses(folder, focal):
    lines = open(os.path.join(raw, "groundtruth.txt")).read().replace(",", " ").replace("\t", " ").split("\n")
    comp = [[v for v in l.split(" ") if v.strip()] for l in lines if l and l[0] != "#"]
    comp = [c for c in comp if c]
    trans = np.asarray([c[1:4] for c in comp], float) * 1000.0
    quat = np.asarray([c[4:] for c in comp], float)
    poses = []
    for i in sorted(wanted):
        T = np.ones((4, 5))
        T[:3, :3] = R.from_quat(quat[i]).as_matrix(); T[:3, 3] = trans[i]
        T[0, 4], T[1, 4], T[2, 4] = 512, 640, focal
        poses.append(np.append(T[:3].flatten(), [0, 200]))
    np.save(os.path.join(folder, "poses_bounds.npy"), np.array(poses))


write_poses(out, calib["fx"])
json.dump(calib, open(os.path.join(out, "calib.json"), "w"), indent=1)
print(out, n, "frames; calib", calib)
if a.orig_name:
    o = os.path.join(a.out_root, a.orig_name)
    os.makedirs(o, exist_ok=True)
    for d in ("images", "images_right", "masks", "calib.json"):
        dst = os.path.join(o, d)
        if not os.path.lexists(dst):
            os.symlink(os.path.relpath(os.path.join(out, d), o), dst)
    write_poses(o, fx_unrect)
    print(o, "with unrectified focal", fx_unrect)
