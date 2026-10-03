"""Estimate the global scale of a monocular depth prior from the known camera poses (no stereo).

SIFT matches between frame pairs with a large enough baseline are triangulated with the known
poses (c2w, OpenCV frame, mm, from poses_bounds.npy); the scale is the median ratio
triangulated depth / prior depth over all matches. (Solving the scale from the prior's own
back-projected points fails here: ~10 mm of prior noise swamps baselines of a few mm.)

  python stereomis/estimate_prior_scale.py --root data/stereomis/split_1 --prior moge --out mogeps
writes depth_<out>/ = depth_<prior>/ * s  (and SCALE.txt)
"""
import os
from argparse import ArgumentParser

import cv2
import numpy as np

parser = ArgumentParser()
parser.add_argument("--root", default="data/stereomis/split_1")
parser.add_argument("--prior", default="moge")
parser.add_argument("--out", default=None, help="write depth_<out>/ with the scaled prior")
# rectified intrinsics of the 640x512 StereoMIS images (StereoRectifier, halved)
parser.add_argument("--fx", type=float, default=1137.9944413564533 / 2)
parser.add_argument("--cx", type=float, default=654.6777954101562 / 2)
parser.add_argument("--cy", type=float, default=502.6583557128906 / 2)
parser.add_argument("--min_baseline", type=float, default=4.0, help="mm")
parser.add_argument("--step", type=int, default=5)
# review F06/F07: frame pool and geometric filters (defaults reproduce the original estimate)
parser.add_argument("--train_only", action="store_true", help="exclude the loader's test frames, (i-1) % 8 == 0")
parser.add_argument("--max_epi", type=float, default=None, help="max Sampson distance (px) to the known-pose F")
parser.add_argument("--cheirality", action="store_true", help="require positive depth in both cameras")
parser.add_argument("--max_reproj", type=float, default=None, help="max reprojection error (px) in both views")
parser.add_argument("--min_angle", type=float, default=None, help="min triangulation angle (deg)")
parser.add_argument("--quiet_pool", action="store_true")
args = parser.parse_args()

stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.root, "images")))
poses = np.load(os.path.join(args.root, "poses_bounds.npy"))[:, :15].reshape(-1, 3, 5)[:, :, :4]
sift = cv2.SIFT_create(4000)
matcher = cv2.BFMatcher(cv2.NORM_L2)


def load(i):
    img = cv2.imread(os.path.join(args.root, "images", stems[i] + ".png"), cv2.IMREAD_GRAYSCALE)
    depth = np.load(os.path.join(args.root, f"depth_{args.prior}", stems[i] + ".npy"))[0]
    kp, des = sift.detectAndCompute(img, None)
    return np.array([k.pt for k in kp], np.float32), des, depth


K = np.array([[args.fx, 0, args.cx], [0, args.fx, args.cy], [0, 0, 1]])


def projection(pose):
    R, t = pose[:, :3], pose[:, 3]  # c2w
    return K @ np.concatenate([R.T, -R.T @ t[:, None]], 1)


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def known_F(pi_, pj_):
    """Fundamental matrix from camera i to j implied by the known c2w poses."""
    Ri, ci, Rj, cj = pi_[:, :3], pi_[:, 3], pj_[:, :3], pj_[:, 3]
    R = Rj.T @ Ri; t = Rj.T @ (ci - cj)  # x_j = R x_i + t
    Kinv = np.linalg.inv(K)
    return Kinv.T @ skew(t) @ R @ Kinv


pool = [i for i in range(0, len(stems), args.step) if not (args.train_only and (i - 1) % 8 == 0)]
if not args.quiet_pool:
    print("frame pool:", len(pool), "frames, e.g.", [stems[i] for i in pool[:6]], "...")
cache, estimates, pairs, kept = {}, [], 0, []
for ii, i in enumerate(pool):
    for j in pool[ii + 1:]:
        if np.linalg.norm(poses[j][:, 3] - poses[i][:, 3]) < args.min_baseline:
            continue
        for k in (i, j):
            if k not in cache:
                cache[k] = load(k)
        (ui, di, Di), (uj, dj, _) = cache[i], cache[j]
        good = [m for m, n in matcher.knnMatch(di, dj, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            continue
        pi, pj = ui[[m.queryIdx for m in good]], uj[[m.trainIdx for m in good]]
        _, inl = cv2.findFundamentalMat(pi, pj, cv2.FM_RANSAC, 1.0, 0.999)
        keep = inl.ravel().astype(bool)
        pi, pj = pi[keep], pj[keep]
        if args.max_epi is not None:
            Fk = known_F(poses[i], poses[j])
            x1 = np.c_[pi, np.ones(len(pi))]; x2 = np.c_[pj, np.ones(len(pj))]
            a1, a2 = x1 @ Fk.T, x2 @ Fk
            samp = np.abs(np.sum(x2 * a1, 1)) / np.sqrt(a1[:, 0] ** 2 + a1[:, 1] ** 2 + a2[:, 0] ** 2 + a2[:, 1] ** 2)
            pi, pj = pi[samp <= args.max_epi], pj[samp <= args.max_epi]
            if len(pi) < 20:
                continue
        Pi_, Pj_ = projection(poses[i]), projection(poses[j])
        X = cv2.triangulatePoints(Pi_, Pj_, pi.T, pj.T)
        X = (X[:3] / X[3]).T
        Ri, ti = poses[i][:, :3], poses[i][:, 3]
        Rj, tj = poses[j][:, :3], poses[j][:, 3]
        z_tri = ((X - ti) @ Ri)[:, 2]  # depth in camera i
        good_pt = np.ones(len(X), bool)
        if args.cheirality:
            good_pt &= ((X - tj) @ Rj)[:, 2] > 0
        if args.max_reproj is not None:
            Xh = np.c_[X, np.ones(len(X))]
            for P, uv in ((Pi_, pi), (Pj_, pj)):
                pr = Xh @ P.T
                good_pt &= np.linalg.norm(pr[:, :2] / pr[:, 2:3] - uv, axis=1) <= args.max_reproj
        if args.min_angle is not None:
            ri, rj = X - ti, X - tj
            cosang = np.sum(ri * rj, 1) / (np.linalg.norm(ri, axis=1) * np.linalg.norm(rj, axis=1))
            good_pt &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) >= args.min_angle
        z_prior = Di[pi[:, 1].round().astype(int).clip(0, Di.shape[0] - 1),
                     pi[:, 0].round().astype(int).clip(0, Di.shape[1] - 1)]
        ok = (z_tri > 0) & (z_prior > 0) & good_pt
        if ok.sum() >= 20:
            estimates.append(np.median(z_tri[ok] / z_prior[ok]))
            pairs += 1; kept.append(int(ok.sum()))

scale = float(np.median(estimates))
print(f"{pairs} frame pairs, per-pair scale median {scale:.4f} (IQR {np.percentile(estimates, 25):.4f}"
      f"-{np.percentile(estimates, 75):.4f}), median matches/pair {int(np.median(kept))}")
if args.out:
    src, dst = os.path.join(args.root, f"depth_{args.prior}"), os.path.join(args.root, f"depth_{args.out}")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".npy"):
            np.save(os.path.join(dst, f), np.load(os.path.join(src, f)) * scale)
    open(os.path.join(dst, "SCALE.txt"), "w").write(f"{scale}\n")
    print("wrote", dst)
