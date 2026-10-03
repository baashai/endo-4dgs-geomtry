"""Main-result figures without hard-coded reference numbers (review F37).

  python make_main_figs.py   ->  fig/qualitative.pdf, fig/perframe.pdf
Inputs: ../Endo-4DGS/output/stereomis/frame_000898.npz (stereomis/dump_frame.py)
        ../Endo-4DGS/output/stereomis/depth_extended.json (stereomis/eval_depth_extended.py)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

OUT = "../Endo-4DGS/output/stereomis"
IMG = "../Endo-4DGS/data/stereomis/split_1f/images/000898.png"
C_OURS, C_FDAM, C_ORIG, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#12161b", "#5b636e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7.2, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
blue = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
orange = LinearSegmentedColormap.from_list("o", ["#fbe3d6", "#f4b393", "#eb6834", "#c24e1d", "#8a3410"])

# ---------- qualitative ----------
Z = np.load(f"{OUT}/frame_000898.npz")
rgb = np.array(Image.open(IMG).convert("RGB"))
zs = Z["stereo"]; valid = zs > 0
lo, hi = np.percentile(zs[valid], [2, 98])
cols = [("Input frame", None), ("Stereo reference", "stereo"), ("Original (Endo-4DGS)", "orig"),
        ("Corrected FoV + DA", "fdam"), ("Ours: FoV + MoGe-2 init", "finit")]
fig, ax = plt.subplots(2, 5, figsize=(7.16, 2.6), gridspec_kw={"wspace": 0.03, "hspace": 0.06})
for c, (title, k) in enumerate(cols):
    ax[0, c].set_title(title, fontsize=7.5, pad=2, fontweight="bold" if k == "finit" else "normal", color=C_OURS if k == "finit" else INK)
    if k is None:
        ax[0, c].imshow(rgb); ax[1, c].axis("off")
    else:
        d = zs if k == "stereo" else Z[k]
        if k != "stereo":
            m = valid & (d > 0)
            d = d * np.median(zs[m] / d[m])  # per-frame scale alignment for display and error
        dm = blue(np.clip((d - lo) / (hi - lo), 0, 1))[..., :3]
        if k == "stereo":
            dm[~valid] = 0.78
            ax[1, c].axis("off")
            ax[1, c].text(0.5, 0.5, "grey: no stereo depth\n(disparity range,\nocclusion, no texture)", ha="center", va="center",
                          fontsize=6.5, color=INK2, transform=ax[1, c].transAxes)
        else:
            e = np.abs(d - zs) / np.maximum(zs, 1e-6)
            em = orange(np.clip(e / 0.5, 0, 1))[..., :3]; em[~valid] = 0.78
            ax[1, c].imshow(em)
            ax[1, c].text(0.03, 0.05, f"AbsRel {e[valid & (Z[k] > 0)].mean():.3f}", transform=ax[1, c].transAxes, fontsize=7,
                          color="white", bbox=dict(boxstyle="round,pad=0.2", fc="#12161bcc", ec="none"))
        ax[0, c].imshow(dm)
    for r in range(2):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        for sp in ax[r, c].spines.values():
            sp.set_visible(False)
ax[0, 0].set_ylabel("Depth", fontsize=7.5); ax[1, 0].axis("on"); ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])
[sp.set_visible(False) for sp in ax[1, 0].spines.values()]; ax[1, 0].set_ylabel("Rel. error", fontsize=7.5)
for x0, cm, labs, ttl in ((0.205, blue, [f"{lo:.0f} mm", f"{hi:.0f} mm"], "stereo-reference depth scale"),
                          (0.46, orange, ["0%", "≥50%"], "|z − z_stereo| / z_stereo")):
    cb = fig.add_axes([x0, 0.05, 0.2, 0.016]); cb.imshow(np.linspace(0, 1, 256)[None], cmap=cm, aspect="auto")
    cb.set_yticks([]); cb.set_xticks([0, 255]); cb.set_xticklabels(labs); cb.set_title(ttl, fontsize=6.5, pad=1.5)
    [sp.set_visible(False) for sp in cb.spines.values()]; cb.tick_params(length=0, pad=1)
fig.subplots_adjust(left=0.03, right=0.995, top=0.92, bottom=0.15)
fig.savefig("fig/qualitative.pdf", dpi=300); fig.savefig("fig/qualitative.png", dpi=150); plt.close(fig)

# ---------- per-frame scale-aligned error, mean of 3 seeds ----------
E = json.load(open(f"{OUT}/depth_extended.json"))
frames = [int(f) for f in E["orig_s0"]["test_frames"]]
fig, ax = plt.subplots(figsize=(3.45, 1.3))
for key, c, lab in (("orig", C_ORIG, "Original"), ("fdam", C_FDAM, "FoV + DA"), ("finit", C_OURS, "Ours")):
    if not all(f"{key}_s{s}" in E for s in range(3)):
        continue
    y = np.mean([E[f"{key}_s{s}"]["per_frame"]["aligned"] for s in range(3)], 0)
    ax.plot(frames, y, color=c, lw=1.4, label=lab, solid_capstyle="round")
    ax.plot(frames[-1], y[-1], "o", color=c, ms=3.2, mec="white", mew=0.6)
ax.set_xlabel("video frame (P1)"); ax.set_ylabel("aligned AbsRel")
ax.grid(axis="y", color="#e3e6ea", lw=0.5); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_ylim(0.10, 0.315)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04), handlelength=1.4, columnspacing=1.2, borderaxespad=0)
fig.tight_layout(pad=0.3); fig.savefig("fig/perframe.pdf"); fig.savefig("fig/perframe.png", dpi=200); plt.close(fig)
print("figures written")

# ---------- image quality vs. scale-aligned depth error (3-seed means +- s.d., from results.tsv) ----------
import csv
import statistics as st
rows = list(csv.DictReader(open(f"{OUT}/results.tsv"), delimiter="\t"))
seed0 = {"dam": (31.8747, 0.2384)}  # seed 0 of the original setup was trained before results.tsv existed (output/stereomis/split_1)
pts = [("dam", "Original", C_ORIG), ("f_dam", "FoV + DA", C_FDAM), ("f_dam_init", "FoV + DA, init", INK2),
       ("mogesc", "MoGe-2 stereo-sc.", INK2), ("f_mogeps", "FoV + MoGe-2", INK2), ("f_mogeps_init", "Ours", C_OURS)]
off = {"dam": (4, -9), "f_dam": (4, 3), "f_dam_init": (-4, 4), "mogesc": (4, -4), "f_mogeps": (4, -8), "f_mogeps_init": (-4, 5)}
ha = {"f_dam_init": "right", "f_mogeps_init": "right"}
fig, ax = plt.subplots(figsize=(3.45, 1.6))
for key, lab, col in pts:
    rr = [r for r in rows if r["name"] == key]
    p = [float(r["PSNR"]) for r in rr]; a = [float(r["AbsRel"]) for r in rr]
    if key in seed0 and not any(r["seed"] == "0" for r in rr):
        p.append(seed0[key][0]); a.append(seed0[key][1])
    if key == "mogesc" and not any(r["seed"] == "0" for r in rr):
        p.append(31.1900); a.append(0.1608)  # seed 0 of the stereo-scaled run (output/stereomis/split_1_mogesc)
    ax.errorbar(st.mean(a), st.mean(p), xerr=st.stdev(a), yerr=st.stdev(p), fmt="o", color=col,
                ms=4.2 if key != "f_mogeps_init" else 5.5, mec="white", mew=0.6, elinewidth=0.7, capsize=1.5,
                zorder=3 if col != INK2 else 2)
    ax.annotate(lab, (st.mean(a), st.mean(p)), xytext=off[key], textcoords="offset points", fontsize=6.6,
                ha=ha.get(key, "left"), color=INK if col != INK2 else INK2, fontweight="bold" if key == "f_mogeps_init" else "normal")
ax.set_xlabel("scale-aligned AbsRel vs. stereo (lower is better)"); ax.set_ylabel("PSNR (dB)")
ax.grid(color="#e3e6ea", lw=0.5); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_xlim(0.12, 0.43); ax.set_ylim(30.85, 32.95)
fig.tight_layout(pad=0.3); fig.savefig("fig/tradeoff.pdf"); fig.savefig("fig/tradeoff.png", dpi=200); plt.close(fig)
print("tradeoff written")
