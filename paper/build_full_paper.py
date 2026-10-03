"""Assemble full_paper.tex from front_matter.tex, methodology.tex and main.tex.

Sections are copied verbatim from their sources; only the joins (setup, results framing, conclusion)
are new. Citation keys of main.tex are mapped to refs.bib. Run: python build_full_paper.py
"""
import re
import subprocess
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "full_paper_revised"

fm = open("front_matter.tex").read()
me = open("methodology.tex").read()
mn = open("main.tex").read()


def between(s, start, end, include_start=True):
    i = s.index(start)
    j = s.index(end, i + len(start))
    return s[i if include_start else i + len(start):j]


def block(s, label):
    """The float (figure/table/figure*/table*) that contains \\label{label}."""
    k = s.index(r"\label{%s}" % label)
    starts = [m.start() for m in re.finditer(r"\\begin\{(figure\*?|table\*?)\}", s[:k])]
    i = starts[-1]
    env = re.match(r"\\begin\{(figure\*?|table\*?)\}", s[i:]).group(1)
    j = s.index(r"\end{%s}" % env, k) + len(r"\end{%s}" % env)
    return s[i:j]


KEYMAP = {"endo4dgs": "huang2024endo4dgs", "moge2": "wang2025moge2", "depthanything": "yang2024depthanything",
          "stereomis": "hayoz2023stereomis", "kerbl3dgs": "kerbl2023gaussian", "wu4dgs": "wu2024_4dgs",
          "colmap": "schonberger2016sfm", "sgm": "hirschmuller2008sgm"}


def remap(s):
    def fix(m):
        return r"\cite{" + ",".join(KEYMAP.get(k.strip(), k.strip()) for k in m.group(1).split(",")) + "}"
    return re.sub(r"\\cite\{([^}]*)\}", fix, s)


# ---------------- pieces from the sources ----------------
title = between(fm, r"\title{", r"\maketitle", include_start=False)
abstract = between(fm, r"\begin{abstract}", r"\section{Introduction}", include_start=False)
intro = between(fm, r"\section{Introduction}", r"\section{Related Work}")
related = between(fm, r"\section{Related Work}", r"\subsection{Positioning and novelty of this work}")
positioning = between(fm, r"\subsection{Positioning and novelty of this work}", r"\bibliographystyle")
tikz_preamble = between(me, r"% palette", r"\begin{document}")
method = between(me, r"\section{Methodology}", r"\bibliographystyle")
method = method.replace(r"Section~IV shows that each of them lowers PSNR without improving geometry.",
                        r"Section~\ref{sec:results} shows that each of them lowers PSNR without improving geometry.")

tab_hw = remap(block(mn, "tab:hw"))
tab_main = remap(block(mn, "tab:main")).replace(r"\textit{32.70}", r"\textit{32.69}")
tab_main = tab_main.replace("\\input{fig/tab_main_rows.tex}\n", open("fig/tab_main_rows.tex").read())  # \input breaks \bottomrule
tab_abl = remap(block(mn, "tab:abl"))
fig_qual = remap(block(mn, "fig:qual"))
fig_perframe = remap(block(mn, "fig:perframe"))
fig_trade = remap(block(mn, "fig:trade"))
res_quant = remap(between(mn, r"\textbf{Quantitative.}", r"\textbf{Ablations}"))
res_abl = remap(between(mn, r"\textbf{Ablations}", r"\begin{table}"))

positioning = positioning.replace(r"\subsection{Positioning and novelty of this work}",
                                  r"\section{Comparison with the State of the Art and Novelty}" + "\n" + r"\label{sec:novelty}")
positioning = positioning.replace(r"Table~\ref{tab:design} and Table~\ref{tab:sameclip} place our work against the literature.",
                                  r"Table~\ref{tab:design} (Section~\ref{sec:related}) and Table~\ref{tab:sameclip} place our work against the literature.")

def part(name):
    return "\n" + open(f"parts/{name}.tex").read()


# ---------------- new joining text ----------------
setup = r"""
\section{Experimental Setup}
\label{sec:setup}
\textbf{Data and protocol.} The main experiments use StereoMIS~\cite{hayoz2023stereomis} sequence P1, frames 801--1000, with the split, metrics and geometric evaluation of Section~\ref{sec:evalproto}, under the monocular input protocol (left images and known poses). Section~\ref{sec:p25} repeats the FoV and initialisation comparisons on a second sequence, P2\_5, where a separate stereo-assisted scale protocol is also reported, and Section~\ref{sec:explain} evaluates uncertainty on both. Each main configuration is trained with seeds 0, 1 and 2; we report the mean and the sample standard deviation over seeds ($n{-}1$). Ablations use one seed unless marked otherwise; three runs on two short, temporally correlated clips do not support significance claims, and we make none.

\textbf{Reproduction.} We use the released Endo-4DGS code (commit 39c6ce2) with its StereoMIS configuration. It differs from the description in the paper~\cite{huang2024endo4dgs}: it uses Depth-Anything ViT-B/14 (ONNX), whereas the paper describes Depth-Anything Small, and the loss weights are those of the released configuration. Our local modifications are the depth-prior selector, the optional normal loss, the seed option and the evaluation scripts; with the default options the training code path is unchanged. We therefore call our baseline a reproduction of the released code, not of the paper.

\textbf{Compared configurations.} (1)~\emph{Original}: released Endo-4DGS with Depth-Anything (DA) depth and the 34.2$^\circ$ field of view. (2)~\emph{+ FoV fix}: the same with corrected intrinsics (Section~\ref{sec:intrinsics}). (3)~\emph{+ FoV, DA init only}: corrected intrinsics, DA initialisation and the depth-related terms disabled. (4)~\emph{+ FoV, MoGe-2 + depth loss}: pose-scaled MoGe-2 ($\times$0.093) for initialisation and as the depth target of the full objective. (5)~\emph{Ours}: corrected intrinsics, pose-scaled MoGe-2 initialisation ($\times$0.093) and the depth-related terms disabled. (6)~\emph{Ours, held-out-clean scale}: as (5) with $\times$0.104, estimated from training frames with the known-pose filters (Section~\ref{sec:scale}).

\textbf{Hardware and software.} Table~\ref{tab:hw} lists the platform and the cost of each component. Endo-4DGS and the evaluation run in one environment; MoGe-2 runs in a separate one because it requires a newer Python.
"""

results = (r"""
\section{Results}
\label{sec:results}
""" + fig_qual + "\n" + tab_main + "\n" + tab_abl + "\n" + fig_perframe + "\n" + fig_trade + r"""

\subsection{Quantitative comparison}
""" + res_quant.replace(r"\textbf{Quantitative.} ", "") + r"""
\subsection{Ablations}
""" + res_abl.replace(r"\textbf{Ablations} (Table~\ref{tab:abl}). ", "") + r"""
\subsection{Qualitative results}
Fig.~\ref{fig:qual} shows rendered depth and its error against the stereo reference on a test frame, using the median-error seed of each setup. Both Depth-Anything models render a sharp depth step below the instrument ring that the stereo depth does not support; it appears as a dark band in their error maps. Correcting the field of view alone does not remove it. The MoGe-2 model renders a smoother surface whose scale-aligned error is lower over most of the tissue (AbsRel 0.139 vs.\ 0.231 and 0.248 on this frame). The maps are aligned per frame, so they show shape agreement, not metric depth.

\subsection{Computational cost}
We separate offline preprocessing, training, rendering and the cost of the uncertainty signals. \emph{Preprocessing:} MoGe-2 is larger than Depth-Anything (331\,M vs.\ 97\,M parameters) and runs on the GPU in 0.22\,s per frame (3.2\,GB peak); in our setup Depth-Anything ran on the CPU at 0.75\,s per frame, so the two times are not directly comparable. \emph{Training and rendering:} training time (15.9 vs.\ 16.0\,min), the final number of Gaussians (327\,k vs.\ 330\,k) and rendering speed (38\,FPS) are comparable to the baseline (Table~\ref{tab:hw}); we did not measure training memory. \emph{Uncertainty:} prior disagreement needs no additional model training, but at a new frame it needs the monocular prior (0.22\,s for MoGe-2, i.e.\ about 4.5\,FPS if run serially) plus a median alignment and per-pixel arithmetic; it is inexpensive only when the prior is already computed, as it is for training frames. The seed ensemble needs three trained models, and ray spread and deformation magnitude need one extra rasterisation pass each.
""")

conclusion = r"""
\section{Conclusion}
\label{sec:conclusion}
We revisited monocular 4D Gaussian splatting for endoscopy and measured not only how well it re-renders the video but how well its depth agrees with a stereo reference and where it disagrees. On StereoMIS P1, corrected intrinsics and pose-scaled MoGe-2 initialisation improve PSNR from 31.98 to 32.40\,dB and reduce the per-frame scale-aligned depth AbsRel against SGBM stereo from 0.234 to 0.148, averaged over three seeds, while LPIPS increases from 0.178 to 0.189; with the scale factor estimated from training frames only and filtered for consistency with the known poses, the result is 32.29\,dB and 0.147. On the static P2\_5 clip, MoGe-2 initialisation reduces the aligned AbsRel from 0.064 to 0.046 with monocular inputs, and a separate stereo-calibrated protocol improves PSNR from 26.70 to 28.66\,dB. These measurements assess reference-relative depth shape; our P1 reconstructions are not metric (raw depth about 1.9$\times$ too shallow relative to stereo), and the scale of a state-of-the-art point-map model does not transfer to endoscopic images (6--18$\times$ off). Image-metric rankings need not match depth-accuracy rankings on these clips. Among the tested per-pixel signals, accumulated opacity ranks depth errors weakly, whereas disagreement with the monocular prior is the strongest individual signal on both sequences and under a strict left--right-consistent reference. Depth edges have the highest per-pixel error, but most of the total error lies on ordinary tissue.

\textbf{Limitations.} The study covers two short StereoMIS clips and evaluates only pixels with a usable stereo reference, which remains an estimate even after left--right filtering. Per-frame alignment hides global scale errors, which we report separately. The known-pose scale needs camera motion and quasi-static correspondences and could not be used on P2\_5. The objective, normal-loss and pose ablations are on P1 only, and the jointly disabled depth terms are not separated. EndoNeRF data were not available to us, and comparisons with other methods rely on published numbers under possibly different protocols. \textbf{Future work} will test additional sequences and independently measured geometry, retrain the strongest open-source methods under the same protocol, separate the individual loss terms, and test whether prior disagreement can guide adaptive refinement.
"""

preamble = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs,array}
\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,fit,backgrounds,calc,shapes.misc}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=black}
\definecolor{ours}{HTML}{2A78D6}
\newcommand{\best}[1]{\textbf{#1}}
\newcommand{\pmsd}[1]{{\scriptsize$\pm$#1}}
""" + tikz_preamble

doc = (preamble + r"\begin{document}" + "\n\n" + r"\title{" + title + r"\maketitle" + "\n\n" + r"\begin{abstract}" + abstract
       + intro + related + part("related_uncertainty") + method + part("method_uncertainty") + setup + tab_hw + "\n"
       + results + part("results_metric") + part("results_p25") + part("explainability") + positioning + conclusion
       + "\n" + r"\bibliographystyle{IEEEtran}" + "\n" + r"\bibliography{refs}" + "\n\n" + r"\end{document}" + "\n")
doc = doc.replace(r"\label{sec:related}", r"\label{sec:related}", 1)
import os
if os.environ.get("DRAFT"):
    doc = re.sub(r"@@[A-Z_]+@@", r"\\textbf{[TBD]}", doc)
assert "@@" not in doc, "unfilled placeholders: " + ", ".join(sorted(set(re.findall(r"@@[A-Z_]+@@", doc))))
open(f"{OUT}.tex", "w").write(doc)

for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{OUT}.tex"], ["bibtex", OUT],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{OUT}.tex"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{OUT}.tex"]):
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        sys.exit(f"build step failed ({r.returncode}): {' '.join(cmd)}; see {OUT}.log / {OUT}.blg")
log = open(f"{OUT}.log", errors="ignore").read()
if re.search(r"(Citation|Reference) `[^']+' on page \d+ undefined", log):
    sys.exit("undefined citations or references; see " + OUT + ".log")
print(re.findall(r"Output written on .*", log))
print("undefined:", sorted(set(re.findall(r"(?:Citation|Reference) `([^']+)' on page \d+ undefined", log))))
print("errors:", re.findall(r"^! .*", log, re.M)[:5])
