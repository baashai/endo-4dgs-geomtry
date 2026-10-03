"""Refine the StereoMIS robot poses of a split with COLMAP bundle adjustment.

The kinematic poses are off by ~0.5 deg (up to 1.5 deg) in rotation, which shows up as several px
of epipolar error between frames. This script
  1. extracts/matches SIFT (exhaustive) with the rectified intrinsics fixed,
  2. triangulates the matches from the robot poses and bundle-adjusts poses + points,
  3. aligns the result back to the robot frame (rotation: chordal mean of the per-camera
     rotation offsets; scale + offset: least squares on the camera centres, so the mm scale of
     the robot poses is kept), and
  4. optionally (--smooth sigma) keeps only the slowly varying part of the correction: the
     per-frame offset robot -> refined is Gaussian-smoothed over time and applied to the robot
     poses. Bundle adjustment treats frames independently and assumes rigid tissue, so its raw
     poses jitter ~10x more than the robot's (0.09 vs 0.01 deg frame to frame), which costs the
     deformable model ~2 dB PSNR.
  5. writes <out_root>/<name>/ = the input split with poses_bounds.npy replaced.
Needs the `colmap` binary (conda env colmap_env). Run from the repo root:
  python stereomis/refine_poses.py --split data/stereomis/split_1f --name split_1fr
"""
import os
import shutil
import sqlite3
import subprocess
from argparse import ArgumentParser

import numpy as np
from scipy.spatial.transform import Rotation as R

parser = ArgumentParser()
parser.add_argument("--split", default="data/stereomis/split_1f")
parser.add_argument("--name", default="split_1fr")
parser.add_argument("--colmap", default="colmap")
parser.add_argument("--work", default=None, help="COLMAP workspace (default <split>/../colmap_<name>)")
parser.add_argument("--smooth", type=float, default=0, help="temporal Gaussian sigma (frames) for the correction; 0 = raw")
args = parser.parse_args()

split = os.path.abspath(args.split)
work = os.path.abspath(args.work or os.path.join(os.path.dirname(split), f"colmap_{args.name}"))
out = os.path.join(os.path.dirname(split), args.name)
images = os.path.join(split, "images")
names = sorted(os.listdir(images))
bounds = np.load(os.path.join(split, "poses_bounds.npy"))
poses = bounds[:, :15].reshape(-1, 3, 5)
H, W, f = poses[0, :, 4]
cx, cy = 654.6777954101562 / 2, 502.6583557128906 / 2  # rectified principal point at 640x512


def colmap(*a):
    subprocess.run([args.colmap, *a], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


os.makedirs(work, exist_ok=True)
db = os.path.join(work, "db.db")
if not os.path.exists(db):
    colmap("feature_extractor", "--database_path", db, "--image_path", images,
           "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "1",
           "--ImageReader.camera_params", f"{f},{f},{cx},{cy}",
           "--FeatureExtraction.use_gpu", "1", "--SiftExtraction.max_num_features", "8000")
    colmap("exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", "1")

# robot poses as a fixed-intrinsics COLMAP model (image ids as assigned by the database)
ids = dict((n, i) for i, n in sqlite3.connect(db).execute("select image_id, name from images"))
gt = os.path.join(work, "robot"); os.makedirs(gt, exist_ok=True)
open(os.path.join(gt, "cameras.txt"), "w").write(f"1 PINHOLE {int(W)} {int(H)} {f} {f} {cx} {cy}\n")
with open(os.path.join(gt, "images.txt"), "w") as fh:
    for n, p in zip(names, poses):
        Rw = p[:, :3].T; t = -Rw @ p[:, 3]; q = R.from_matrix(Rw).as_quat()
        fh.write(f"{ids[n]} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {n}\n\n")
open(os.path.join(gt, "points3D.txt"), "w").write("")
for d in ("tri", "ba", "ba_txt"):
    os.makedirs(os.path.join(work, d), exist_ok=True)
colmap("point_triangulator", "--database_path", db, "--image_path", images,
       "--input_path", gt, "--output_path", os.path.join(work, "tri"))
colmap("bundle_adjuster", "--input_path", os.path.join(work, "tri"), "--output_path", os.path.join(work, "ba"),
       "--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_principal_point", "0",
       "--BundleAdjustment.refine_extra_params", "0")
colmap("model_converter", "--input_path", os.path.join(work, "ba"), "--output_path", os.path.join(work, "ba_txt"),
       "--output_type", "TXT")

ba = {}
for line in open(os.path.join(work, "ba_txt", "images.txt")):
    t = line.split()
    if len(t) >= 10 and t[-1] in ids:
        q = list(map(float, t[1:5])); tv = np.array(list(map(float, t[5:8])))
        Rw = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        ba[t[-1]] = (Rw.T, -Rw.T @ tv)
missing = [n for n in names if n not in ba]
assert not missing, f"COLMAP did not register {len(missing)} images, e.g. {missing[:3]}"

Rb = np.array([ba[n][0] for n in names]); Cb = np.array([ba[n][1] for n in names])
Rg, Cg = poses[:, :, :3], poses[:, :, 3]
Rs = R.from_matrix(np.einsum("nij,nkj->nik", Rg, Rb)).mean().as_matrix()
Cr = Cb @ Rs.T
s = np.sum((Cr - Cr.mean(0)) * (Cg - Cg.mean(0))) / np.sum((Cr - Cr.mean(0)) ** 2)
Rn = np.einsum("ij,njk->nik", Rs, Rb); Cn = s * Cr + (Cg.mean(0) - s * Cr.mean(0))

if args.smooth > 0:
    from scipy.ndimage import gaussian_filter1d
    rot_off = R.from_matrix(np.einsum("nij,njk->nik", Rn, np.transpose(Rg, (0, 2, 1)))).as_rotvec()  # Rn = off @ Rg
    rot_off = gaussian_filter1d(rot_off, args.smooth, axis=0, mode="nearest")
    Rn = np.einsum("nij,njk->nik", R.from_rotvec(rot_off).as_matrix(), Rg)
    Cn = Cg + gaussian_filter1d(Cn - Cg, args.smooth, axis=0, mode="nearest")

dr = [np.degrees(np.linalg.norm(R.from_matrix(Rg[i].T @ Rn[i]).as_rotvec())) for i in range(len(names))]
dc = np.linalg.norm(Cn - Cg, axis=1)
print(f"alignment scale {s:.4f}; change vs robot poses: rotation median {np.median(dr):.2f} max {max(dr):.2f} deg, "
      f"centre median {np.median(dc):.3f} max {dc.max():.3f} mm")

if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(out)
for entry in os.listdir(split):  # share images, masks and depth priors with the input split
    if entry != "poses_bounds.npy":
        os.symlink(os.path.relpath(os.path.realpath(os.path.join(split, entry)), out), os.path.join(out, entry))
new = poses.copy(); new[:, :, :3] = Rn; new[:, :, 3] = Cn
np.save(os.path.join(out, "poses_bounds.npy"), np.concatenate([new.reshape(len(names), 15), bounds[:, 15:]], 1))
print("wrote", out)
