"""Assemble data/stereomis/split_N from the frames written by process_dataset.py.

Same pose conversion as pre_pose_stereomis.py, but the pose range matches the
frames extracted by process_dataset.py (split_1 = frames 801..1000), so that
the number of poses equals the number of images.
Run from the repo root: python stereomis/make_split.py --split 1
"""
import os
import shutil
from argparse import ArgumentParser

import numpy as np
from scipy.spatial.transform import Rotation as R

SPLITS = {1: (801, 1000), 2: (13901, 14100)}

focal_length = 1039.6275634765625
h = 1024
w = 1280

parser = ArgumentParser()
parser.add_argument("--raw", default="data/StereoMIS/P1")
parser.add_argument("--split", type=int, default=1)
parser.add_argument("--out_root", default="data/stereomis")
parser.add_argument("--name", default=None, help="output folder name (default split_<split>)")
parser.add_argument("--rectified_focal", action="store_true",
                    help="store the focal of the rectified, half-size images (569 px) instead of the "
                         "original full-size unrectified 1039.6 px, which gives the renderer a 34 deg "
                         "instead of the true 58.7 deg FoV")
args = parser.parse_args()
if args.rectified_focal:
    focal_length = 1137.9944413564533 / 2

start, end = SPLITS[args.split]
out_dir = os.path.join(args.out_root, args.name or f"split_{args.split}")
os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

src_img = os.path.join(args.raw, f"images_{args.split}")
for f in sorted(os.listdir(src_img)):
    shutil.copy(os.path.join(src_img, f), os.path.join(out_dir, "images", f))

# The loader needs one 3-channel mask per image; raw masks exist only every
# other frame, so each frame takes the nearest raw mask.
os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
raw_masks = {int(m[:6]): m for m in os.listdir(os.path.join(args.raw, "masks"))}
for i in range(start, end + 1):
    nearest = raw_masks[min(raw_masks, key=lambda k: abs(k - i))]
    shutil.copy(os.path.join(args.raw, "masks", nearest), os.path.join(out_dir, "masks", f"{i:06d}.png"))

f = open(os.path.join(args.raw, "groundtruth.txt"), "r")
lines = f.read().replace(",", " ").replace("\t", " ").split("\n")
components = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
              len(line) > 0 and line[0] != "#"]
trans = np.asarray([l[1:4] for l in components if len(l) > 0], dtype=float) * 1000.0  # m to mm
quat = np.asarray([l[4:] for l in components if len(l) > 0], dtype=float)

pose_list = []
for i in range(start, end + 1):
    transformation_matrix = np.ones((4, 5))
    transformation_matrix[:3, :3] = R.from_quat(quat[i]).as_matrix()
    transformation_matrix[:3, 3] = trans[i]
    transformation_matrix[0, 4] = h / 2
    transformation_matrix[1, 4] = w / 2
    transformation_matrix[2, 4] = focal_length
    pose_list.append(np.append(transformation_matrix[0:3, :].flatten(), [0, 200]))

np.save(os.path.join(out_dir, "poses_bounds.npy"), np.array(pose_list))
print(out_dir, len(os.listdir(os.path.join(out_dir, "images"))), "images,", len(pose_list), "poses")
