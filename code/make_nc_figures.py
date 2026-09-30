#!/usr/bin/env python3
"""Figures 5-8 for the NC trained-memory validation lane.  Plot-only.

These figures continue the manuscript's existing figures 1-4 and are rendered
from the frozen confirmatory CSVs.  Nothing here is decision-bearing: no Fisher
quantity, efficiency, gate, statistic or verdict is computed in this file.  J,
R_J, the alignments, the accuracies and the retention bounds are READ from the
CSVs exactly as `run_stage.py confirmatory` wrote them.  The only closed forms
evaluated here are the Bayes overlay Phi(a sqrt(J)) -- with the amplitude `a`
read from RUN_RECEIPT.json, never hardcoded -- and the plotting summaries
(median, interquartile range, mean absolute error, least-squares slope) that
exist purely to draw a line or print an annotation on an axis.  A figure whose
input CSV is absent is skipped with a printed note; no value is ever invented.

Palette (house rule, copied from
vendor/PREPRINT_PUBLIC_RELEASE_v1.2_20260917/scripts/make_preprint_figures_20260903.py):
cream #F2EDE4, navy #1A2744, one muted accent #9C8F7A.  No third colour.
Greyscale fills are alpha blends of navy or muted, as in the reference fig4.

Usage
-----
    python make_nc_figures.py --results <lane>/results/confirmatory --out <dir>
    python make_nc_figures.py --selftest      # synthetic data, temp dirs only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

LANE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = LANE / "results" / "confirmatory"
DEFAULT_OUT = LANE / "figures"

CREAM, NAVY, MUTED = "#F2EDE4", "#1A2744", "#9C8F7A"
plt.rcParams.update({"figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
                     "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "text.color": NAVY, "font.family": "serif", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

# Preregistered reference levels (nc_verdict.GATES), drawn as lines only.
GATE_MEDIAN = 0.90
GATE_FLOOR = 0.80
CALIBRATION_BAND = 0.02

WRITERS = ("nonnormal", "normal")
# Two colours only: navy carries the primary series, muted the secondary one.
# Fill (solid vs cream-faced) carries the second factor, exactly as fig4 does.
WRITER_LABEL = {"nonnormal": "conditioned $SQS^{-1}$", "normal": "normal $Q$"}
NAMED_SHAPE = {"v_old": "o", "v_store": "s", "v_write": "^", "v_bottom": "D",
               "v_worst_positive": "v", "v_learn": "*"}
NAMED_ORDER = ("v_old", "v_store", "v_write", "v_bottom", "v_worst_positive", "v_learn")
ALPHA_SHAPE = {0.05: "o", 0.25: "s", 1.0: "^"}


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict[str, str]] | None:
    """Read a CSV into dicts, or return None when the file is absent."""
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def skip(figure: str, path: Path) -> None:
    print(f"skipping {figure}: {path} not found")


def fnum(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def inum(row: dict[str, str], key: str) -> int:
    return int(round(fnum(row, key)))


def group(rows: Sequence[dict[str, str]], keyfn: Callable[[dict[str, str]], Any]) -> dict[Any, list[dict[str, str]]]:
    out: dict[Any, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(keyfn(row), []).append(row)
    return out


def bayes_curve(amplitude: float, J: np.ndarray) -> np.ndarray:
    """Phi(a sqrt(J)) -- the closed-form overlay, evaluated for drawing only."""
    z = amplitude * np.sqrt(np.clip(np.asarray(J, dtype=float), 0.0, None)) / math.sqrt(2.0)
    vals = np.array([math.erf(float(x)) for x in np.ravel(z)])
    return (0.5 * (1.0 + vals)).reshape(np.shape(z))


def save(fig: Figure, name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.{{png,svg}}")


def ladder(n: int, width: float = 0.30) -> np.ndarray:
    """Deterministic in-column offsets so overlapping points stay readable."""
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width / 2.0, width / 2.0, n)


# --------------------------------------------------------------------------
# figure 5 -- learned alignment by carrier type
# --------------------------------------------------------------------------
def fig5_alignment(results: Path, out: Path) -> bool:
    path = results / "CELL_LEVEL.csv"
    rows = load_rows(path)
    if rows is None:
        skip("fig5", path)
        return False

    panels = [("rayleigh_efficiency",
               "Rayleigh efficiency  $R_J = v^{\\top}M_{old}v\\,/\\,\\lambda_{max}$"),
              ("subspace_alignment",
               "leading-subspace alignment  $\\|\\Pi_{0.99}\\,v\\|^{2}$")]
    tick_label = {"nonnormal": "non-normal\nconditioned $SQS^{-1}$", "normal": "normal\n$Q$"}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.6))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.90, bottom=0.34, wspace=0.26)

    by_writer = group(rows, lambda r: r["writer_type"])
    for ax, (metric, ylabel) in zip(axes, panels):
        plotted: list[float] = []
        cloud: list[float] = []
        for slot, writer in enumerate(WRITERS):
            cells = by_writer.get(writer, [])
            if not cells:
                continue
            per_draw = group(cells, lambda r: inum(r, "draw_index"))
            draws = sorted(per_draw)
            medians = [float(np.nanmedian([fnum(r, metric) for r in per_draw[d]])) for d in draws]
            gaps = [float(np.nanmedian([fnum(r, "eigengap_ratio") for r in per_draw[d]])) for d in draws]
            order = np.argsort(np.asarray(medians, dtype=float), kind="stable")
            offsets = np.zeros(len(draws))
            offsets[order] = ladder(len(draws))
            filled = writer == "nonnormal"
            ax.plot(slot + offsets, medians, linestyle="none", marker="o", ms=6.5,
                    color=NAVY, mfc=NAVY if filled else CREAM, mew=1.3,
                    label=f"{WRITER_LABEL[writer]}: per-draw median over optimizer seeds"
                    if ax is axes[0] else None)
            gap_order = np.argsort(np.asarray(gaps, dtype=float), kind="stable")
            gap_offsets = np.zeros(len(draws))
            gap_offsets[gap_order] = ladder(len(draws))
            ax.plot(slot + gap_offsets, gaps, linestyle="none", marker="_", ms=8, mew=1.5,
                    color=MUTED,
                    label="$\\lambda_2/\\lambda_1$ per draw (score of the 2nd eigenvector)"
                    if (ax is axes[0] and slot == 0) else None)
            plotted.extend(medians)
            plotted.extend(gaps)
            cloud.extend(medians)

        ax.axhline(GATE_MEDIAN, color=NAVY, lw=0.9, ls=":")
        ax.axhline(GATE_FLOOR, color=MUTED, lw=0.9, ls=":")
        ax.set_xticks(range(len(WRITERS)))
        ax.set_xticklabels([tick_label[w] for w in WRITERS], fontsize=9.5)
        ax.set_xlim(-0.58, len(WRITERS) - 0.42)
        ax.set_ylabel(ylabel, fontsize=10)

        finite = [v for v in plotted if np.isfinite(v)]
        low = min([GATE_FLOOR - 0.06] + finite) if finite else 0.0
        high = max([1.0] + finite) if finite else 1.0
        ax.set_ylim(min(low - 0.02, GATE_FLOOR - 0.06), high + 0.03)
        ax.text(-0.55, GATE_MEDIAN, f"{GATE_MEDIAN:.2f} median gate",
                fontsize=8.5, va="bottom", ha="left", color=NAVY)
        ax.text(-0.55, GATE_FLOOR, f"{GATE_FLOOR:.2f} per-draw floor",
                fontsize=8.5, va="bottom", ha="left", color=MUTED)

        # The gate lives in the last few percent, so resolve the top explicitly.
        tight = [v for v in cloud if np.isfinite(v)]
        if tight and (max(tight) - min(tight)) < 0.08:
            pad = max(0.004, (max(tight) - min(tight)) * 0.25)
            axin = ax.inset_axes((0.64, 0.07, 0.34, 0.30))
            for slot, writer in enumerate(WRITERS):
                cells = by_writer.get(writer, [])
                if not cells:
                    continue
                per_draw = group(cells, lambda r: inum(r, "draw_index"))
                draws = sorted(per_draw)
                medians = [float(np.nanmedian([fnum(r, metric) for r in per_draw[d]])) for d in draws]
                order = np.argsort(np.asarray(medians, dtype=float), kind="stable")
                offsets = np.zeros(len(draws))
                offsets[order] = ladder(len(draws))
                axin.plot(slot + offsets, medians, linestyle="none", marker="o", ms=4.2,
                          color=NAVY, mfc=NAVY if writer == "nonnormal" else CREAM, mew=1.0)
            axin.set_xlim(-0.55, len(WRITERS) - 0.45)
            axin.set_ylim(min(tight) - pad, max(tight) + pad)
            axin.set_xticks(range(len(WRITERS)))
            axin.set_xticklabels(["non-norm.", "normal"], fontsize=7)
            axin.tick_params(labelsize=7)
            axin.set_title("zoom on the point cloud", fontsize=7.5)

    axes[0].set_title("Where the learned direction lands", fontsize=11)
    axes[1].set_title("How much of the leading subspace it uses", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.115))
    fig.text(0.5, 0.075,
             "Each marker is one carrier draw; in-column position is ordered by value and carries no data.",
             ha="center", fontsize=8.5, color=NAVY)
    fig.text(0.5, 0.030,
             "The muted dashes are $\\lambda_2/\\lambda_1$ for that draw: the score a direction sitting exactly on the "
             "second eigenvector would earn, i.e. the headroom the gate actually has.",
             ha="center", fontsize=8.5, color=NAVY)
    save(fig, "fig5_alignment", out)
    return True


# --------------------------------------------------------------------------
# figure 6 -- predicted versus empirical accuracy
# --------------------------------------------------------------------------
def fig6_calibration(results: Path, out: Path) -> bool:
    path = results / "DIRECTION_ROWS.csv"
    rows = load_rows(path)
    if rows is None:
        skip("fig6", path)
        return False

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.4, 6.8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.3, 1.0]})
    style = {"isolated": (NAVY, NAVY, "o", "isolated"),
             "open": (MUTED, CREAM, "o", "open")}
    all_x: list[float] = []
    all_y: list[float] = []
    per_condition: dict[str, float] = {}
    for condition, (edge, face, marker, label) in style.items():
        subset = [r for r in rows if r["condition"] == condition]
        if not subset:
            continue
        x = np.array([fnum(r, "predicted_accuracy") for r in subset])
        y = np.array([fnum(r, "recalibrated_accuracy") for r in subset])
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        ax.scatter(x, y, s=7, marker=marker, facecolors=face, edgecolors=edge,
                   linewidths=0.5, alpha=0.55, rasterized=True,
                   label=f"{label}  (n={x.size})")
        axr.scatter(x, y - x, s=7, marker=marker, facecolors=face, edgecolors=edge,
                    linewidths=0.5, alpha=0.55, rasterized=True)
        per_condition[condition] = float(np.mean(np.abs(y - x))) if x.size else float("nan")
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    if all_x:
        ax_arr = np.asarray(all_x)
        ay_arr = np.asarray(all_y)
        lo = float(min(ax_arr.min(), ay_arr.min()))
        hi = float(max(ax_arr.max(), ay_arr.max()))
        pad = 0.02 * max(hi - lo, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=NAVY, lw=0.9, ls=":",
                label="identity")
        mae = float(np.mean(np.abs(ay_arr - ax_arr)))
        slope, intercept = np.polyfit(ax_arr, ay_arr, 1)
        grid = np.linspace(lo - pad, hi + pad, 64)
        ax.plot(grid, slope * grid + intercept, color=MUTED, lw=1.4,
                label=f"least-squares fit, slope {slope:.4f}")
        note = [f"mean absolute error  {mae:.4f}",
                f"fitted slope  {slope:.4f}   intercept  {intercept:+.4f}"]
        for condition in ("isolated", "open"):
            if condition in per_condition:
                note.append(f"MAE, {condition}:  {per_condition[condition]:.4f}")
        ax.text(0.03, 0.97, "\n".join(note), transform=ax.transAxes, va="top", ha="left",
                fontsize=9)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_ylabel("empirical accuracy (recalibrated readout)", fontsize=10)
    ax.set_title("Every direction, every condition, against the closed form", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    axr.axhspan(-CALIBRATION_BAND, CALIBRATION_BAND, color=NAVY, alpha=0.10, lw=0)
    axr.axhline(0.0, color=NAVY, lw=0.9, ls=":")
    axr.set_xlabel("predicted accuracy  $\\Phi(a\\sqrt{J})$")
    axr.set_ylabel("empirical $-$ predicted", fontsize=10)
    axr.text(0.03, 0.06, f"shaded: preregistered calibration band $\\pm{CALIBRATION_BAND:.2f}$",
             transform=axr.transAxes, fontsize=8.5, color=NAVY)
    fig.tight_layout()
    save(fig, "fig6_calibration", out)
    return True


# --------------------------------------------------------------------------
# figure 7 -- causal direction swap, ordered by J
# --------------------------------------------------------------------------
def _shape_for(direction: str) -> str | None:
    if direction.startswith("v_learn"):
        return NAMED_SHAPE["v_learn"]
    return NAMED_SHAPE.get(direction)


def _plot_direction_panel(ax: Axes, subset: Sequence[dict[str, str]], ycol: str,
                          amplitude: float, with_legend: bool) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for writer in WRITERS:
        rows = [r for r in subset if r["writer_type"] == writer]
        if not rows:
            continue
        filled = writer == "nonnormal"
        randoms = [r for r in rows if r["direction"].startswith("v_random_")]
        if randoms:
            xr = np.array([fnum(r, "J") for r in randoms])
            yr = np.array([fnum(r, ycol) for r in randoms])
            keep = np.isfinite(xr) & np.isfinite(yr) & (xr > 0)
            ax.scatter(xr[keep], yr[keep], s=6, marker=".", color=MUTED, alpha=0.7,
                       rasterized=True,
                       label="random directions" if (with_legend and filled) else None)
            if keep.any():
                lo = min(lo, float(xr[keep].min()))
                hi = max(hi, float(xr[keep].max()))
        for name in NAMED_ORDER:
            if name == "v_learn":
                picked = [r for r in rows if r["direction"].startswith("v_learn")]
            else:
                picked = [r for r in rows if r["direction"] == name]
            if not picked:
                continue
            x = np.array([fnum(r, "J") for r in picked])
            y = np.array([fnum(r, ycol) for r in picked])
            keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if not keep.any():
                continue
            marker = _shape_for(picked[0]["direction"]) or "P"
            ax.plot(x[keep], y[keep], linestyle="none", marker=marker,
                    ms=10 if marker == "*" else 7, color=NAVY,
                    mfc=NAVY if filled else CREAM, mew=1.2,
                    label=name if (with_legend and filled) else None)
            lo = min(lo, float(x[keep].min()))
            hi = max(hi, float(x[keep].max()))
    if np.isfinite(lo) and np.isfinite(hi) and hi > 0:
        grid = np.geomspace(max(lo * 0.6, 1e-12), hi * 1.6, 400)
        ax.plot(grid, bayes_curve(amplitude, grid), color=NAVY, lw=1.2, ls="-", alpha=0.85,
                label=f"$\\Phi(a\\sqrt{{J}})$, $a={amplitude:g}$" if with_legend else None)
        ax.set_xlim(max(lo * 0.6, 1e-12), hi * 1.6)
    ax.set_xscale("log")
    return lo, hi


def fig7_swap(results: Path, out: Path) -> bool:
    cells_path = results / "CELL_LEVEL.csv"
    rows_path = results / "DIRECTION_ROWS.csv"
    swap_path = results / "SWAP_ROWS.csv"
    receipt_path = results / "RUN_RECEIPT.json"
    for path in (cells_path, rows_path, swap_path, receipt_path):
        if not path.is_file():
            skip("fig7", path)
            return False
    cells = load_rows(cells_path) or []
    rows = load_rows(rows_path) or []
    swaps = load_rows(swap_path) or []
    receipt = load_receipt(receipt_path) or {}
    config = receipt.get("config", {})
    if "amplitude" not in config:
        print(f"skipping fig7: {receipt_path} not found")  # no amplitude, no curve
        return False
    amplitude = float(config["amplitude"])

    per_draw: dict[int, list[float]] = {}
    for row in cells:
        per_draw.setdefault(inum(row, "draw_index"), []).append(fnum(row, "rayleigh_efficiency"))
    draw_median = {d: float(np.nanmedian(v)) for d, v in per_draw.items()}
    if not draw_median:
        skip("fig7", cells_path)
        return False
    across = float(np.nanmedian(list(draw_median.values())))
    rep = min(draw_median, key=lambda d: (abs(draw_median[d] - across), d))

    isolated = [r for r in rows if r["condition"] == "isolated" and inum(r, "draw_index") == rep]
    horizons = sorted({inum(r, "horizon") for r in isolated})
    train_horizon = int(config.get("train_horizon", horizons[0] if horizons else 0))
    if train_horizon not in horizons and horizons:
        train_horizon = horizons[0]
    subset = [r for r in isolated if inum(r, "horizon") == train_horizon]
    swap_subset = [r for r in swaps if inum(r, "draw_index") == rep]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    _plot_direction_panel(ax, subset, "recalibrated_accuracy", amplitude, with_legend=True)
    _plot_direction_panel(axb, swap_subset, "fixed_readout_accuracy", amplitude, with_legend=False)

    ax.set_xlabel("store-only Fisher information of the probed direction  $J$")
    axb.set_xlabel("store-only Fisher information of the probed direction  $J$")
    ax.set_ylabel("accuracy", fontsize=10)
    ax.set_title("Recalibrated readout: accuracy follows $J$", fontsize=11)
    axb.set_title("Learned readout held fixed: the swap cost", fontsize=11)
    ax.text(0.02, 0.97,
            f"representative carrier draw {rep}\n"
            f"(per-draw median $R_J$ = {draw_median[rep]:.4f}, closest to the\n"
            f"across-draw median {across:.4f}); isolated, horizon {train_horizon}\n"
            "filled = conditioned $SQS^{-1}$, open = normal $Q$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5)
    axb.text(0.02, 0.97,
             "same draw and directions, readout frozen at the\n"
             "one the optimizer learned (SWAP_ROWS)",
             transform=axb.transAxes, va="top", ha="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, "fig7_swap", out)
    return True


# --------------------------------------------------------------------------
# figure 8 -- open versus isolated retention
# --------------------------------------------------------------------------
def fig8_retention(results: Path, out: Path) -> bool:
    path = results / "RETENTION_ROWS.csv"
    rows = load_rows(path)
    if rows is None:
        skip("fig8", path)
        return False

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.4, 4.8),
                                  gridspec_kw={"width_ratios": [1.45, 1.0]})
    style = {("nonnormal", "isolated"): (NAVY, "-", "o", True),
             ("nonnormal", "open"): (NAVY, "--", "o", False),
             ("normal", "isolated"): (MUTED, "-", "s", True),
             ("normal", "open"): (MUTED, "--", "s", False)}
    plain = [r for r in rows if r.get("disturbance", "none") == "none"]
    for (writer, condition), (color, ls, marker, filled) in style.items():
        subset = [r for r in plain if r["writer_type"] == writer and r["condition"] == condition]
        if not subset:
            continue
        horizons = sorted({inum(r, "horizon") for r in subset})
        med, q25, q75, pred = [], [], [], []
        for horizon in horizons:
            values = np.array([fnum(r, "recalibrated_accuracy") for r in subset
                               if inum(r, "horizon") == horizon])
            predicted = np.array([fnum(r, "predicted_accuracy") for r in subset
                                  if inum(r, "horizon") == horizon])
            med.append(float(np.nanmedian(values)))
            q25.append(float(np.nanpercentile(values, 25)))
            q75.append(float(np.nanpercentile(values, 75)))
            pred.append(float(np.nanmedian(predicted)))
        ax.plot(horizons, med, color=color, ls=ls, marker=marker, ms=6, lw=2,
                mfc=color if filled else CREAM, mew=1.3,
                label=f"{WRITER_LABEL[writer]}, {condition}")
        ax.fill_between(horizons, q25, q75, color=color, alpha=0.12, lw=0)
        ax.plot(horizons, pred, color=color, ls=(0, (1, 1.6)), lw=0.9, alpha=0.75)

    ax.set_xscale("log")
    ax.set_xlabel("horizon $n$")
    ax.set_ylabel("accuracy on the learned direction (recalibrated)", fontsize=10)
    ax.set_title("Isolation sets persistence; the band is the interquartile range", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=NAVY, lw=1.0, ls=(0, (1, 1.6))))
    labels.append("predicted $\\Phi(a\\sqrt{J})$ (thin)")
    ax.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower left")

    approx = [r for r in rows if r.get("condition") == "approximate_isolation"]
    if approx:
        seen: set[tuple[float, str]] = set()
        for row in approx:
            alpha = fnum(row, "alpha")
            floor = fnum(row, "guaranteed_floor")
            retained = fnum(row, "retained_fraction")
            if not (np.isfinite(floor) and np.isfinite(retained)):
                continue
            marker = ALPHA_SHAPE.get(round(alpha, 6), "P")
            disturbance = row.get("disturbance", "aligned")
            color = NAVY if disturbance == "aligned" else MUTED
            filled = row["writer_type"] == "nonnormal"
            key = (round(alpha, 6), disturbance)
            axb.plot([floor], [retained], linestyle="none", marker=marker, ms=6.5,
                     color=color, mfc=color if filled else CREAM, mew=1.1, alpha=0.85,
                     label=f"$\\alpha={alpha:g}$, {disturbance}" if key not in seen else None)
            seen.add(key)
        floors = [fnum(r, "guaranteed_floor") for r in approx]
        floors = [f for f in floors if np.isfinite(f)]
        if floors:
            span = np.linspace(min(floors) * 0.92, max(floors) * 1.04, 32)
            axb.plot(span, span, color=NAVY, lw=0.9, ls=":", label="$y=x$: guaranteed floor")
        axb.set_xlabel("guaranteed floor  $1/(1+\\alpha)$")
        axb.set_ylabel("retained fraction  $J/J_0$", fontsize=10)
        axb.set_title("Approximate isolation against its bound", fontsize=11)
        axb.text(0.97, 0.06, "points on or above the dotted line satisfy the bound;\n"
                             "filled = conditioned $SQS^{-1}$, open = normal $Q$",
                 transform=axb.transAxes, fontsize=8.5, ha="right")
        axb.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        axb.axis("off")
        axb.text(0.5, 0.5, "no approximate-isolation rows in this file",
                 ha="center", va="center", fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, "fig8_retention", out)
    return True


# --------------------------------------------------------------------------
# selftest -- synthetic data in a temp directory, never in `results`
# --------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_synthetic(results: Path) -> None:
    """Fabricate an obviously synthetic CSV set that matches the frozen schema."""
    rng = np.random.default_rng(20260918)
    draws = [0, 1, 2, 3]
    seeds = [0, 1]
    test_horizons = [64, 256]
    train_horizon = 128
    amplitude = 2.0
    alphas = [0.05, 0.25, 1.0]
    randoms = [f"v_random_{i:03d}" for i in range(8)]

    results.mkdir(parents=True, exist_ok=True)
    (results / "SYNTHETIC_DATA_DO_NOT_USE.txt").write_text(
        "Fabricated by make_nc_figures.py --selftest. Not a measurement. "
        "Never copy into results/confirmatory.\n")
    (results / "RUN_RECEIPT.json").write_text(json.dumps({
        "stage": "SELFTEST_SYNTHETIC_NOT_A_MEASUREMENT",
        "config": {"amplitude": amplitude, "train_horizon": train_horizon,
                   "test_horizons": test_horizons, "optimizer_seeds": len(seeds)},
    }, indent=2))

    cells: list[dict[str, Any]] = []
    directions: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []
    swaps: list[dict[str, Any]] = []
    for draw in draws:
        for writer in WRITERS:
            lam = 0.26 if writer == "nonnormal" else 0.15
            gap = 0.87 if writer == "nonnormal" else 0.89
            base_J = lam * (1.0 + 0.1 * draw)
            for seed in seeds:
                r_j = float(np.clip(0.985 + 0.01 * rng.standard_normal(), 0.90, 1.0))
                cells.append({
                    "draw_index": draw, "writer_type": writer,
                    "lambda_max_M_old": lam, "eigengap_ratio": gap,
                    "leading_subspace_dim": 1,
                    "oracle_accuracy": 0.78, "random_accuracy_median": 0.63,
                    "seed_index": seed, "rayleigh_efficiency": r_j,
                    "subspace_alignment": float(np.clip(r_j - 0.01, 0.0, 1.0)),
                    "cosine_to_v_old": float(np.sqrt(r_j)),
                    "accuracy": 0.77, "behavioral_efficiency": 0.93,
                    "final_loss": 0.51, "clip_events": 3, "clip_fraction": 0.002,
                    "grad_norm_median": 0.21, "grad_norm_max": 1.4,
                    "best_validation_step": 9000, "best_validation_accuracy": 0.77,
                    "best_validation_rayleigh": r_j,
                })

            named = {"v_old": base_J, "v_store": 0.62 * base_J, "v_write": 0.41 * base_J,
                     "v_bottom": 0.02 * base_J, "v_worst_positive": 0.05 * base_J}
            for seed in seeds:
                named[f"v_learn_s{seed}"] = 0.97 * base_J
            for index, name in enumerate(randoms):
                named[name] = base_J * float(0.02 + 0.5 * rng.random())

            for condition in ("isolated", "open"):
                for horizon in sorted(set(test_horizons) | {train_horizon}):
                    decay = 1.0 if condition == "isolated" else 64.0 / horizon
                    for name, J0 in named.items():
                        J = float(J0 * decay)
                        predicted = float(bayes_curve(amplitude, np.array([J]))[0])
                        directions.append({
                            "draw_index": draw, "writer_type": writer, "condition": condition,
                            "horizon": horizon, "direction": name, "J": J,
                            "predicted_accuracy": predicted,
                            "recalibrated_accuracy": float(np.clip(
                                predicted + 0.004 * rng.standard_normal(), 0.0, 1.0)),
                            "empirical_J": J * float(1.0 + 0.01 * rng.standard_normal()),
                            "rayleigh_efficiency_M_old": float(np.clip(J0 / base_J, 0.0, 1.0)),
                            "subspace_alignment": float(np.clip(J0 / base_J, 0.0, 1.0)),
                            "cosine_to_v_old": float(np.sqrt(np.clip(J0 / base_J, 0.0, 1.0))),
                        })

            for seed in seeds:
                for name, J0 in named.items():
                    if name.startswith("v_learn_s") and name != f"v_learn_s{seed}":
                        continue
                    predicted = float(bayes_curve(amplitude, np.array([J0]))[0])
                    matched = name == f"v_learn_s{seed}"
                    fixed = predicted if matched else 0.5 + 0.35 * (predicted - 0.5)
                    swaps.append({
                        "draw_index": draw, "writer_type": writer, "seed_index": seed,
                        "direction": name, "J": J0,
                        "fixed_readout_accuracy": fixed,
                        "recalibrated_accuracy": predicted,
                        "readout_gap": predicted - fixed,
                    })

                learned_J = named[f"v_learn_s{seed}"]
                for horizon in test_horizons:
                    for condition in ("isolated", "open"):
                        decay = 1.0 if condition == "isolated" else 64.0 / horizon
                        J = float(learned_J * decay)
                        predicted = float(bayes_curve(amplitude, np.array([J]))[0])
                        retention.append({
                            "draw_index": draw, "writer_type": writer, "seed_index": seed,
                            "horizon": horizon, "condition": condition, "disturbance": "none",
                            "alpha": 0.0, "J": J, "predicted_accuracy": predicted,
                            "recalibrated_accuracy": float(np.clip(
                                predicted + 0.003 * rng.standard_normal(), 0.0, 1.0)),
                            "fixed_readout_accuracy": predicted - 0.01,
                        })
                    for alpha in alphas:
                        for disturbance in ("aligned", "random_psd"):
                            floor = 1.0 / (1.0 + alpha)
                            retained = floor if disturbance == "aligned" else float(
                                min(1.0, floor + 0.05 + 0.1 * rng.random()))
                            J0 = float(learned_J)
                            J = J0 * retained
                            predicted = float(bayes_curve(amplitude, np.array([J]))[0])
                            retention.append({
                                "draw_index": draw, "writer_type": writer, "seed_index": seed,
                                "horizon": horizon, "condition": "approximate_isolation",
                                "disturbance": disturbance, "alpha": alpha, "J": J,
                                "predicted_accuracy": predicted,
                                "recalibrated_accuracy": float(np.clip(
                                    predicted + 0.003 * rng.standard_normal(), 0.0, 1.0)),
                                "fixed_readout_accuracy": predicted - 0.01,
                                "J0": J0, "retained_fraction": retained,
                                "guaranteed_floor": floor,
                                "bound_margin": retained - floor,
                                "empirical_J": J * 1.001,
                                "empirical_J0": J0 * 1.001,
                                "empirical_retained_fraction": retained * 1.0005,
                            })

    _write_csv(results / "CELL_LEVEL.csv", cells)
    _write_csv(results / "DIRECTION_ROWS.csv", directions)
    _write_csv(results / "RETENTION_ROWS.csv", retention)
    _write_csv(results / "SWAP_ROWS.csv", swaps)


def selftest() -> int:
    root = Path(tempfile.mkdtemp(prefix="nc_selftest_synthetic_"))
    results = root / "synthetic_results"
    out = root / "synthetic_figures"
    print(f"selftest: SYNTHETIC data only, written to {results}")
    try:
        write_synthetic(results)
        rendered = render_all(results, out)
        assert all(rendered.values()), f"a figure was skipped on complete synthetic input: {rendered}"

        names = ["fig5_alignment", "fig6_calibration", "fig7_swap", "fig8_retention"]
        for name in names:
            for extension in ("png", "svg"):
                path = out / f"{name}.{extension}"
                assert path.is_file(), f"missing {path}"
                assert path.stat().st_size > 0, f"empty {path}"
        pngs = sorted(out.glob("*.png"))
        svgs = sorted(out.glob("*.svg"))
        assert len(pngs) == 4, f"expected 4 png, found {[p.name for p in pngs]}"
        assert len(svgs) == 4, f"expected 4 svg, found {[p.name for p in svgs]}"

        # A missing CSV must skip, not fabricate.
        partial = root / "partial_results"
        partial.mkdir()
        shutil.copy(results / "CELL_LEVEL.csv", partial / "CELL_LEVEL.csv")
        partial_out = root / "partial_figures"
        rendered_partial = render_all(partial, partial_out)
        assert rendered_partial == {"fig5": True, "fig6": False, "fig7": False, "fig8": False}, \
            f"missing-file handling changed: {rendered_partial}"

        for path in pngs + svgs:
            print(f"  ok  {path.name}  {path.stat().st_size} bytes")
        print("selftest: PASS (synthetic input; this says nothing about any measurement)")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------
def render_all(results: Path, out: Path) -> dict[str, bool]:
    return {
        "fig5": fig5_alignment(results, out),
        "fig6": fig6_calibration(results, out),
        "fig7": fig7_swap(results, out),
        "fig8": fig8_retention(results, out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="directory holding the frozen confirmatory CSVs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="directory to write fig5..fig8 into")
    parser.add_argument("--selftest", action="store_true",
                        help="render all four figures from synthetic data in a temp directory")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    rendered = render_all(args.results.resolve(), args.out.resolve())
    drawn = [name for name, ok in rendered.items() if ok]
    print(f"figures drawn: {', '.join(drawn) if drawn else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
