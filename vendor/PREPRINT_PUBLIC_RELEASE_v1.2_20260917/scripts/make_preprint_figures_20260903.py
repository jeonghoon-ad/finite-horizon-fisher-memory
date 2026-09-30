"""Figures for the preprint (public build v1.x). Plot-only; reads released result CSVs.

Not decision-bearing: no Fisher quantity is computed here beyond the closed-form
2D elliptic curves. Run from the project root:
    python scripts/make_preprint_figures_20260903.py
Outputs: figures/fig{1..4}_*.{png,svg}
Palette: cream #F2EDE4, navy #1A2744, one muted accent #9C8F7A.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
CENSUS = ROOT / "results" / "preprint_replication_and_factorial_20260903" / "DIRECTIONAL_CENSUS_SUMMARY.csv"
ANGLE = ROOT / "results" / "preprint_reruns_angle_and_f5sf_20260903" / "ANGLE_C4_THETA_1P1_SUMMARY.csv"
CELLS = ROOT / "results" / "isolation_factorial_v2_bounded_20260903" / "CELL_READOUTS.csv"
CREAM, NAVY, MUTED = "#F2EDE4", "#1A2744", "#9C8F7A"
plt.rcParams.update({"figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
                     "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "text.color": NAVY, "font.family": "serif", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def fig1_concept():
    fig, ax = plt.subplots(figsize=(10.5, 4.6)); ax.set_xlim(0, 13.5); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, lines, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=CREAM, ec=NAVY, lw=1.8))
        for i, l in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.38 - 0.46 * i, l, ha="center", va="center", fontsize=fs)
    box(2.8, 1.7, 4.4, 1.85, ["write block, $x_w\\in\\mathbb{R}^{32}$", "$W_w=Q$  (normal)", "$W_w=SQS^{-1}$  (conditioned)"])
    box(9.6, 1.7, 3.4, 1.85, ["store, $x_s\\in\\mathbb{R}^{16}$", "$U$ isometric", "no noise enters"])
    ax.add_patch(FancyArrowPatch((0.5, 3.0), (2.8, 3.0), arrowstyle="-|>", mutation_scale=16, color=NAVY, lw=1.6)); ax.text(1.65, 3.2, "input $v\\,s_t$", ha="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((0.5, 2.2), (2.8, 2.2), arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1.6)); ax.text(1.65, 1.8, "noise $z_t$, every step", ha="center", fontsize=9.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((7.2, 2.62), (9.6, 2.62), arrowstyle="-|>", mutation_scale=18, color=NAVY, lw=2.2)); ax.text(8.4, 2.9, "coupling $K_t$", ha="center", fontsize=10)
    ax.text(8.4, 1.25, "closed at $t=24$: isolated\nlive throughout: open", ha="center", va="top", fontsize=9.5, color=MUTED)
    ax.text(5.0, 4.55, "write axis", ha="center", fontsize=11, weight="bold"); ax.text(5.0, 4.12, "only the conditioning of $S$ changes", ha="center", fontsize=10, style="italic")
    ax.text(11.3, 4.55, "storage axis", ha="center", fontsize=11, weight="bold"); ax.text(11.3, 4.12, "only the closure of $K$ changes", ha="center", fontsize=10, style="italic")
    ax.text(6.75, 0.35, "readout: store-only Fisher form  $J^{(s)}=p_s^\\top\\,\\Sigma_{ss}^{-1}\\,p_s$", ha="center", fontsize=10.5)
    save(fig, "fig1_concept")


def fig2_elliptic():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    c = 4.0; t = np.linspace(0, 2 * np.pi, 400)
    a1.plot(np.cos(t), np.sin(t), color=MUTED, lw=1.4, ls="--", label="orthogonal part $Q$: circle")
    a1.plot(np.cos(t), np.sqrt(c) * np.sin(t), color=NAVY, lw=2, label="lens $D=\\mathrm{diag}(1,\\sqrt{c})$, $c=4$")
    lam_max, lam_min = 2 * c / (1 + c), 2 / (1 + c)
    a1.annotate("", xy=(0, lam_max), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2)); a1.text(-2.35, 1.3, "$\\lambda_{max}=\\frac{2c}{1+c}$", fontsize=11)
    a1.annotate("", xy=(lam_min, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2)); a1.text(1.15, 0.25, "$\\lambda_{min}=\\frac{2}{1+c}$", fontsize=11, color=MUTED)
    a1.set_aspect("equal"); a1.set_xlim(-2.5, 2.5); a1.set_ylim(-2.5, 2.5); a1.set_title("2D elliptic carrier $W=D^{-1}R(\\theta)D$", fontsize=11)
    a1.legend(loc="upper center", fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    cs = np.geomspace(1, 100, 200)
    a2.plot(cs, 2 * cs / (1 + cs), color=NAVY, lw=2, label="$\\lambda_{max}(M_\\infty)=2c/(1+c)$"); a2.plot(cs, 2 / (1 + cs), color=MUTED, lw=2, label="$\\lambda_{min}(M_\\infty)=2/(1+c)$")
    a2.axhline(2, color=NAVY, lw=0.8, ls=":"); a2.text(1.1, 2.04, "block trace budget $=2$", fontsize=9)
    a2.axhline(1, color=MUTED, lw=0.8, ls=":"); a2.text(3.2, 1.04, "normal carrier: 1 in every direction", fontsize=9, color=MUTED)
    for cc, (mx, mn) in {4: (1.6, 0.4), 100: (1.980198, 0.019802)}.items():  # measured medians, Appendix B
        a2.plot([cc], [mx], "o", color=NAVY, ms=6); a2.plot([cc], [mn], "o", color=MUTED, ms=6)
    a2.text(40, 1.55, "dots: measured at $n=2048$\n(8 draws × 3 angles)", fontsize=8.5, ha="center")
    a2.set_xscale("log"); a2.set_xlabel("conditioning $c$"); a2.set_ylabel("eigenvalues of $M_\\infty$"); a2.set_ylim(0, 2.2)
    a2.legend(fontsize=8.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55))
    fig.tight_layout(); save(fig, "fig2_elliptic")


def fig3_census():
    rows = list(csv.DictReader(open(CENSUS))) + list(csv.DictReader(open(ANGLE)))

    def get(cfg, metric, k="median"):
        return float([r for r in rows if r["configuration"] == cfg and r["metric"] == metric][0][k])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fam = {"SQS": ("o", NAVY, "similarity $SQS^{-1}$"), "ELL": ("s", MUTED, "2D elliptic"), "H8": ("^", NAVY, "Hamiltonian $e^{JH}$ (8×8 blocks)"), "A0": ("D", MUTED, "orthogonal (normal)")}
    seen = set()
    for cfg in sorted({r["configuration"] for r in rows}):
        key = next((k for k in ("A0", "SQS", "ELL", "H8") if cfg.startswith(k)), None)
        if key is None:
            continue
        m, col, lab = fam[key]; x, y = get(cfg, "K_plus"), get(cfg, "lambda_max")
        xlo, xhi, ylo, yhi = get(cfg, "K_plus", "min"), get(cfg, "K_plus", "max"), get(cfg, "lambda_max", "min"), get(cfg, "lambda_max", "max")
        ax.errorbar(x, y, xerr=[[max(x - xlo, 0)], [max(xhi - x, 0)]], yerr=[[max(y - ylo, 0)], [max(yhi - y, 0)]], fmt=m, color=col,
                    mfc=col if key in ("SQS", "ELL") else CREAM, ms=7, capsize=2, lw=0.9, label=lab if key not in seen else None); seen.add(key)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":"); ax.text(12, 1.12, "trace budget: direction average = 1 for every carrier", fontsize=9, color=MUTED, ha="center")
    ax.set_xscale("log"); ax.set_xlabel("finite-window gain $\\sigma_{4096}$ (largest $||W^{\\pm j}||_2$ for $j \\leq 4096$; median, min–max over 8 draws)")
    ax.set_ylabel("$\\lambda_{max}(M_{2048})$: best-direction Fisher total"); ax.set_title("Directional concentration grows slowly with transient gain", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left"); fig.tight_layout(); save(fig, "fig3_census")


def fig4_factorial():
    cr = list(csv.DictReader(open(CELLS))); H = [64, 256, 1024, 4096]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style = {("nonnormal", "isolated"): (NAVY, "-", "o", "conditioned $SQS^{-1}$, isolated"), ("normal", "isolated"): (NAVY, "-", "s", "normal $Q$, isolated"),
             ("nonnormal", "open"): (MUTED, "--", "o", "conditioned $SQS^{-1}$, open"), ("normal", "open"): (MUTED, "--", "s", "normal $Q$, open")}
    for (w, st), (col, ls, mk, lab) in style.items():
        med, lo, hi = [], [], []
        for h in H:
            v = [float(r["store_j_oldest_t0"]) for r in cr if r["write_path"] == w and r["storage"] == st and r["direction"] == "oracle" and r["horizon"] == str(h)]
            med.append(np.median(v)); lo.append(min(v)); hi.append(max(v))
        ax.plot(H, med, color=col, ls=ls, marker=mk, ms=6, lw=2, mfc=col if w == "nonnormal" else CREAM, label=lab); ax.fill_between(H, lo, hi, color=col, alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("horizon $n$"); ax.set_ylabel("store-only Fisher information, oldest input  $J^{(s)}_n(t=0)$")
    ax.set_title("Isolation sets persistence; write conditioning sets the level", fontsize=11); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.text(4096, 0.19 * 1.3, "constant", fontsize=9, ha="right", color=NAVY); ax.text(4096, 0.00056 * 0.45, "$\\sim 1/n$", fontsize=9, ha="right", color=MUTED)
    fig.tight_layout(); save(fig, "fig4_factorial")


if __name__ == "__main__":
    fig1_concept(); fig2_elliptic(); fig3_census(); fig4_factorial(); print(f"figures written to {OUT}")
