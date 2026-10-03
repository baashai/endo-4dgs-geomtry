"""LaTeX table of the uncertainty benchmark, generated from the stereomis/uncertainty_bench.py outputs.

Cells: Spearman rho and share of the attainable sparsification gain, AURG / (AUSE of random order), in %.
  python make_unc_table.py  ->  fig/tab_uncertainty.tex
"""
import json

import numpy as np

SRC = "../Endo-4DGS/output/stereomis/uncertainty_v2"
COND = [("p1", "fdam", "P1, FoV + DA"), ("p1", "finit", "P1, ours"), ("p25", "fdam", "P2\\_5, FoV + DA"),
        ("p25", "finit_sc", "P2\\_5, ours (stereo-scaled)")]
SIG = [("u_cov", "accumulated opacity (Endo-4DGS confidence)"), ("u_ray", "ray depth spread"), ("u_mot", "deformation magnitude"),
       ("u_ens", "seed ensemble (3 seeds)"), ("u_prior", "prior disagreement"), ("u_comb", "combined (rank mean)")]


def load(tag, name):
    d = json.load(open(f"{SRC}/uncertainty_{tag}.json"))
    F = np.array(d["fracs"]); r = d["setups"][name]
    o = np.array(r["oracle"]); rnd = np.trapz(o[0] - o, F)
    return {k: (v["rho"][0], 100 * v["aurg"][0] / rnd) for k, v in r["uncertainty"].items()}


cells = {}
for seq, name, _ in COND:
    cells[(seq, name, "std")] = load(seq, name)
    cells[(seq, name, "lr")] = load(seq + "_lr", name)

lines = [r"\begin{table*}[t]",
         r"\caption{Which signal ranks the stereo-reference-relative depth error? Spearman $\rho$ between each per-pixel signal (computed from the model and its monocular prior only) and the per-frame scale-aligned AbsRel against SGBM stereo, and the share of the attainable ranking gain $G$ (Eq.~\ref{eq:gain}; ratio of the mean AURG to the mean random-order AUSE over 25 (P1) or 13 (P2\_5) test frames and three seeds; 100\,\% = oracle, 0\,\% = random). Strict: stereo pixels that pass a left--right consistency check. Best per column in bold.}",
         r"\label{tab:unc}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
         r"\begin{tabular}{@{}l" + "cc" * 8 + r"@{}}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{4}{c}{%s}" % lab for _, _, lab in COND) + r" \\",
         "".join(r"\cmidrule(lr){%d-%d}" % (2 + 4 * i, 5 + 4 * i) for i in range(4)),
         " & " + " & ".join([r"\multicolumn{2}{c}{SGBM}", r"\multicolumn{2}{c}{strict}"] * 4) + r" \\",
         "Signal & " + " & ".join([r"$\rho$", "gain"] * 8) + r" \\", r"\midrule"]
best = {}
for seq, name, _ in COND:
    for ref in ("std", "lr"):
        c = cells[(seq, name, ref)]
        best[(seq, name, ref)] = (max(c, key=lambda k: c[k][0]), max(c, key=lambda k: c[k][1]))
for k, lab in SIG:
    row = [lab]
    for seq, name, _ in COND:
        for ref in ("std", "lr"):
            rho, gain = cells[(seq, name, ref)][k]
            br, bg = best[(seq, name, ref)]
            row.append((r"\textbf{%.2f}" if k == br else "%.2f") % rho)
            row.append((r"\textbf{%d}" if k == bg else "%d") % round(gain))
    lines.append(" & ".join(row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open("fig/tab_uncertainty.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[-10:]))
