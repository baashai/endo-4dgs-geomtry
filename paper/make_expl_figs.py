"""Explainability figures from stereomis/uncertainty_bench.py outputs.

  python make_expl_figs.py [--tag p1] [--tag_lr p1_lr]
writes fig/expl_maps_<tag>.pdf, fig/expl_sparsification_<tag>.pdf, fig/expl_calibration_<tag>.pdf,
fig/expl_regions_<tag>.pdf
"""
import json
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

ap = ArgumentParser()
ap.add_argument("--src", default="../Endo-4DGS/output/stereomis/uncertainty")
ap.add_argument("--tag", default="p1")
ap.add_argument("--tag_lr", default="p1_lr")
a = ap.parse_args()
os.makedirs("fig", exist_ok=True)

INK, INK2, GRID = "#12161b", "#5b636e", "#e3e6ea"
SIG = {  # signal: (label, colour) - fixed palette slots
    "u_prior": ("prior disagreement", "#2a78d6"),
    "u_comb": ("combined (rank mean)", "#4a3aa7"),
    "u_ens": ("seed ensemble", "#1baf7a"),
    "u_ray": ("ray depth spread", "#eb6834"),
    "u_cov": ("accumulated opacity", "#e87ba4"),
    "u_mot": ("deformation magnitude", "#eda100"),
}
SETUP = {"orig": ("Original", "#1baf7a"), "fdam": ("FoV + DA", "#eb6834"), "fmogeps": ("FoV + MoGe-2, depth loss", "#8a929c"),
         "finit": ("Ours: FoV + MoGe-2 init", "#2a78d6"), "finit_raw": ("MoGe-2 init, unscaled", "#8a929c"),
         "finit_sc": ("Ours: MoGe-2 init, scaled", "#2a78d6")}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7.5,
                     "axes.labelsize": 7.5, "legend.fontsize": 6.3, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
orange = LinearSegmentedColormap.from_list("o", ["#fbe3d6", "#f4b393", "#eb6834", "#c24e1d", "#8a3410"])
purple = LinearSegmentedColormap.from_list("p", ["#f1eefa", "#c9bff0", "#9085e9", "#5b4bc4", "#2d2178"])

R = json.load(open(f"{a.src}/uncertainty_{a.tag}.json"))
F = np.array(R["fracs"])
S = R["setups"]
seq_label = {"p1": "StereoMIS P1", "p25": "StereoMIS P2_5"}.get(R["seq"], R["seq"])


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


# ---------- maps: error and uncertainty signals for two setups ----------
Z = np.load(f"{a.src}/examples_{a.tag}.npz")
rows = [k for k in ("fdam", "finit", "finit_sc") if f"{k}_err" in Z.files][:2]
cols = ["rgb", "err", "u_cov", "u_ray", "u_prior", "u_ens"]
titles = ["input frame", "aligned error vs stereo", "accumulated opacity", "ray depth spread", "prior disagreement", "seed ensemble"]
fig, ax = plt.subplots(len(rows), len(cols), figsize=(7.16, 1.32 * len(rows) + 0.35), gridspec_kw={"wspace": 0.03, "hspace": 0.12})
for r, name in enumerate(rows):
    err = Z[f"{name}_err"]
    for c, key in enumerate(cols):
        x = ax[r, c]
        x.set_xticks([]); x.set_yticks([]); [s.set_visible(False) for s in x.spines.values()]
        if key == "rgb":
            x.imshow(Z[f"{name}_rgb"])
        elif key == "err":
            m = np.ma.masked_invalid(err)
            x.imshow(m, cmap=orange, vmin=0, vmax=0.5); x.set_facecolor("#c4c8ce")
        else:
            u = Z[f"{name}_{key}"]
            ok = np.isfinite(u) & np.isfinite(err)
            q = np.full(u.shape, np.nan); q[ok] = np.argsort(np.argsort(u[ok])) / ok.sum()  # percentile rank
            x.imshow(np.ma.masked_invalid(q), cmap=purple, vmin=0, vmax=1); x.set_facecolor("#c4c8ce")
            rho = spearmanr(u[ok], err[ok]).correlation
            x.text(0.03, 0.05, f"ρ = {rho:+.2f}", transform=x.transAxes, fontsize=6.3, color="white",
                   bbox=dict(boxstyle="round,pad=0.2", fc="#12161bcc", ec="none"))
        if r == 0:
            x.set_title(titles[c], fontsize=7, pad=2)
    ax[r, 0].set_ylabel(SETUP[name][0], fontsize=6.8, color=INK)
fig.subplots_adjust(left=0.035, right=0.995, top=0.9, bottom=0.03)
fig.savefig(f"fig/expl_maps_{a.tag}.pdf", dpi=300); fig.savefig(f"fig/expl_maps_{a.tag}.png", dpi=150); plt.close(fig)

# ---------- sparsification curves ----------
pick = [k for k in ("fdam", "finit", "finit_sc") if k in S][:2]
fig, ax = plt.subplots(1, len(pick), figsize=(3.45, 1.55), sharey=False)
for i, name in enumerate(pick):
    x = ax[i]; r = S[name]
    oracle = np.array(r["oracle"])
    x.plot(F, oracle, color=INK, lw=1.1, ls="--", label="oracle")
    x.axhline(oracle[0], color="#9aa1aa", lw=0.9, ls=":", label="random")
    for k, (lab, col) in SIG.items():
        x.plot(F, np.array(r["curves"][k]), color=col, lw=1.3, label=lab)
    x.set_title(SETUP[name][0], fontsize=7, pad=2); x.set_xlabel("fraction of pixels removed")
    clean(x)
ax[0].set_ylabel("AbsRel of remaining")
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(pad=0.3, rect=(0, 0.2, 1, 1))
fig.legend(h, l, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), handlelength=1.4, columnspacing=0.8)
fig.savefig(f"fig/expl_sparsification_{a.tag}.pdf", bbox_inches="tight"); fig.savefig(f"fig/expl_sparsification_{a.tag}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- calibration (mean error per uncertainty decile) ----------
fig, ax = plt.subplots(1, len(pick), figsize=(3.45, 1.75), sharey=True)
for i, name in enumerate(pick):
    x = ax[i]; r = S[name]
    for k, (lab, col) in SIG.items():
        x.plot(np.arange(1, 11), r["calibration_deciles"][k], color=col, lw=1.3, marker="o", ms=2.4, label=lab)
    x.set_title(SETUP[name][0], fontsize=7, pad=2); x.set_xlabel("uncertainty decile"); x.set_xticks([1, 5, 10])
    clean(x)
ax[0].set_ylabel("mean aligned AbsRel")
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(pad=0.3, rect=(0, 0.2, 1, 1))
fig.legend(h, l, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), handlelength=1.4, columnspacing=0.8)
fig.savefig(f"fig/expl_calibration_{a.tag}.pdf", bbox_inches="tight"); fig.savefig(f"fig/expl_calibration_{a.tag}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- region attribution ----------
names = [k for k in ("orig", "fdam", "fmogeps", "finit", "finit_raw", "finit_sc") if k in S]
_all = [("tissue", "tissue"), ("device", "device (ring)"), ("specular", "specular"), ("depth_edge", "depth edges")]
_ok = [(rg, lb) for rg, lb in _all if all(np.isfinite(S[n]["region_absrel"][rg][0]) for n in names)]  # P2_5 has no device
regions = [rg for rg, _ in _ok]; rlab = [lb for _, lb in _ok]
fig, ax = plt.subplots(figsize=(3.45, 1.45))
w = 0.8 / len(names)
for j, name in enumerate(names):
    vals = [S[name]["region_absrel"][rg][0] for rg in regions]
    sds = [S[name]["region_absrel"][rg][1] for rg in regions]
    ax.bar(np.arange(len(regions)) + (j - (len(names) - 1) / 2) * w, vals, w * 0.9, yerr=sds, color=SETUP[name][1], label=SETUP[name][0],
           error_kw=dict(lw=0.5, capsize=1, ecolor=INK2))
fr = S[names[0]]["region_fraction"]
ax.set_xticks(np.arange(len(regions))); ax.set_xticklabels([f"{l}\n({100 * fr[rg]:.0f}% px)" for l, rg in zip(rlab, regions)])
ax.set_ylabel("AbsRel"); clean(ax); ax.grid(axis="x", visible=False)
ax.set_ylim(0, 1.3 * max(S[n]["region_absrel"][rg][0] + S[n]["region_absrel"][rg][1] for n in names for rg in regions))
ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.0, columnspacing=0.8)
fig.tight_layout(pad=0.3)
fig.savefig(f"fig/expl_regions_{a.tag}.pdf"); fig.savefig(f"fig/expl_regions_{a.tag}.png", dpi=200); plt.close(fig)
print("written for", a.tag, seq_label)
