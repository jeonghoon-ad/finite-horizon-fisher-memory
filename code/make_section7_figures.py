#!/usr/bin/env python3
"""Supplementary figures S7a-S7c for the trained-memory manuscript section. Plot-only.

The release renderer applies a white background and readable label placement.
Numerical selections and plotted coordinates are unchanged.

These figures are NOT decision-bearing. No Fisher quantity, efficiency, retention
bound, gate, statistic or verdict is computed in this file. J, the Rayleigh
efficiencies, the objectives, the accuracies, the retained fractions and the
guaranteed floors are READ from the frozen CSVs exactly as the confirmatory and
nc3c stages wrote them; `predicted_accuracy` is likewise read, never recomputed,
and the amplitude `a` and the training horizon used in the axis labels come from
confirmatory/RUN_RECEIPT.json and are never hardcoded. The only closed form
evaluated anywhere below is the identity line y = x. The remaining derived
numbers -- median, interquartile range, mean absolute error, least-squares slope
and the own-versus-cross tallies -- exist solely to draw a line or print an
annotation on an axis, and carry no verdict.

v_bottom is EXCLUDED from the figS7a calibration population. Under isolation those
rows probe the exact 16-dimensional kernel of the task operator, where the reported J
is a floating-point round-off residual; under open coupling the same direction carries
small leaked information (corrected 2026-09-23: an earlier version of this note, and of
the on-axis annotation, called every v_bottom row round-off). Including them would let
the null-direction control set the calibration range. The exclusion and
the number of dropped rows are printed on the axes, in the empty lower-right
corner, so the omission is visible to a reader of the figure alone.

A figure whose input CSV is absent is skipped with a printed note and the remaining
figures still render. No value is ever invented to fill a gap.

Usage
-----
    python make_section7_figures.py --results <lane>/results --out <dir>
    python make_section7_figures.py --selftest    # synthetic data, temp dir only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

LANE = Path(__file__).resolve().parents[1]

CREAM, NAVY, MUTED = "#F2EDE4", "#1A2744", "#9C8F7A"
plt.rcParams.update({"figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
                     "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "text.color": NAVY, "font.family": "serif", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

Row = dict[str, str]


# --------------------------------------------------------------------------- io

def save(fig: plt.Figure, out: Path, name: str) -> None:
    """Write the white-background manuscript pair via the presentation layer."""
    from make_public_figures import save_figure
    save_figure(fig, out, name)


def load_rows(path: Path, figure: str) -> list[Row] | None:
    """Read a frozen CSV, or announce the skip and return None."""
    if not path.is_file():
        print(f"skipping {figure}: {path} not found")
        return None
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_config(results: Path) -> dict[str, Any]:
    """Read config from the confirmatory run receipt; absent keys stay absent."""
    path = results / "confirmatory" / "RUN_RECEIPT.json"
    if not path.is_file():
        print(f"note: {path} not found; amplitude and training horizon omitted from labels")
        return {}
    try:
        return dict(json.loads(path.read_text()).get("config", {}))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"note: {path} unreadable ({exc}); amplitude and training horizon omitted from labels")
        return {}


def floats(rows: Sequence[Row], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=float)


def where(rows: Sequence[Row], **equals: Any) -> list[Row]:
    return [r for r in rows if all(r[k] == str(v) for k, v in equals.items())]


def ls_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope, for an on-axis annotation only."""
    if x.size < 2 or float(np.ptp(x)) == 0.0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


# ------------------------------------------------------------------ figure S7a

def fig_s7a_calibration(results: Path, out: Path) -> bool:
    """Predicted versus empirical accuracy over all directions except v_bottom."""
    path = results / "confirmatory" / "DIRECTION_ROWS.csv"
    rows = load_rows(path, "figS7a_calibration")
    if rows is None:
        return False

    kept = [r for r in rows if r["direction"] != "v_bottom"]
    dropped = len(rows) - len(kept)
    if not kept:
        print("skipping figS7a_calibration: no rows left after the v_bottom exclusion")
        return False

    predicted = floats(kept, "predicted_accuracy")
    empirical = floats(kept, "recalibrated_accuracy")
    isolated = np.asarray([r["condition"] == "isolated" for r in kept])
    residual = empirical - predicted
    mae = float(np.abs(residual).mean())
    slope = ls_slope(predicted, empirical)

    amplitude = load_config(results).get("amplitude")
    xlabel = "predicted accuracy  $\\Phi(a\\sqrt{J})$"
    if amplitude is not None:
        xlabel += f",  $a = {float(amplitude):g}$"

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.09})

    n_open, n_iso = int((~isolated).sum()), int(isolated.sum())
    ax.scatter(predicted[~isolated], empirical[~isolated], s=11, facecolors="none", edgecolors=MUTED,
               linewidths=0.45, alpha=0.40, rasterized=True, zorder=2, label=f"open  ($n={n_open:,}$)")
    ax.scatter(predicted[isolated], empirical[isolated], s=7, color=NAVY, linewidths=0,
               alpha=0.28, rasterized=True, zorder=3, label=f"isolated  ($n={n_iso:,}$)")

    lo = float(min(predicted.min(), empirical.min()))
    hi = float(max(predicted.max(), empirical.max()))
    pad = 0.03 * (hi - lo) if hi > lo else 0.01
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=NAVY, lw=1.3, zorder=4, label="identity")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("empirical accuracy, recalibrated readout")
    ax.set_title("Fisher information predicts delayed accuracy across every probed direction", fontsize=11)

    # The cloud runs along the diagonal, so the two empty corners carry the text: the
    # one-line statistic upper left with the legend under it, the exclusion note lower
    # right. Nothing is allowed to cross the identity line.
    ax.text(0.025, 0.975,
            f"plotted $n = {len(kept):,}$    MAE $= {mae:.4f}$    least-squares slope $= {slope:.3f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.945),
              markerscale=1.8)
    exclusion = [
        f"v_bottom excluded ({dropped:,} rows): under isolation its $J$",
        "is a round-off residual in the exact kernel; with open",
        "coupling it carries small leaked information. Reported",
        "separately in the text; these rows do not set the axis range.",
    ]
    ax.text(0.975, 0.03, "\n".join(exclusion), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8.6)

    axr.axhspan(-0.02, 0.02, color=MUTED, alpha=0.20, lw=0, zorder=1)
    axr.axhline(0.0, color=NAVY, lw=1.0, zorder=3)
    axr.scatter(predicted[~isolated], residual[~isolated], s=11, facecolors="none", edgecolors=MUTED,
                linewidths=0.45, alpha=0.40, rasterized=True, zorder=2)
    axr.scatter(predicted[isolated], residual[isolated], s=7, color=NAVY, linewidths=0,
                alpha=0.28, rasterized=True, zorder=2)
    span = max(0.03, 1.08 * float(np.abs(residual).max()))
    axr.set_ylim(-span, span)
    axr.set_xlabel(xlabel)
    axr.set_ylabel("empirical $-$ predicted")
    axr.text(0.025, 0.06, "shaded band: $\\pm 0.02$", transform=axr.transAxes, fontsize=8.6, color=MUTED)

    save(fig, out, "figS7a_calibration")
    print(f"figS7a_calibration: n={len(kept)} plotted, {dropped} v_bottom rows excluded, "
          f"MAE={mae:.5f}, slope={slope:.4f}")
    return True


# ------------------------------------------------------------------ figure S7b

def fig_s7b_objective_crossing(results: Path, out: Path) -> bool:
    """Own objective against the objective of the readout trained at the other time."""
    path = results / "nc3c" / "CELL_LEVEL.csv"
    rows = load_rows(path, "figS7b_objective_crossing")
    if rows is None:
        return False

    arms = (("readout trained at $t = 0$", "R_own_t0", "R_cross_t0_from_t12"),
            ("readout trained at $t = 12$", "R_own_t12", "R_cross_t12_from_t0"))
    style = {"nonnormal": ("o", NAVY, "conditioned $SQS^{-1}$"),
             "normal": ("s", CREAM, "normal $Q$")}

    values: list[tuple[np.ndarray, np.ndarray]] = []
    for _, own_key, cross_key in arms:
        values.append((floats(rows, own_key), floats(rows, cross_key)))
    flat = np.concatenate([np.concatenate(v) for v in values])
    lo, hi = float(flat.min()), float(flat.max())
    pad = 0.06 * (hi - lo) if hi > lo else 0.05
    lim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.0), sharex=True, sharey=True)
    for ax, (title, own_key, cross_key) in zip(axes, arms):
        ax.plot(lim, lim, color=MUTED, ls="--", lw=1.1, zorder=2, label="identity, own $=$ cross")
        own_all, cross_all = floats(rows, own_key), floats(rows, cross_key)
        for o, c in zip(own_all, cross_all):
            ax.plot([c, c], [c, o], color=MUTED, lw=0.7, alpha=0.40, zorder=3)

        tallies = []
        for writer, (marker, face, label) in style.items():
            sub = where(rows, writer_type=writer)
            if not sub:
                continue
            own, cross = floats(sub, own_key), floats(sub, cross_key)
            ax.plot(cross, own, marker, mfc=face, mec=NAVY, mew=1.1, ms=6.5, ls="none",
                    zorder=4, label=label)
            tallies.append(f"{label}: {int((own > cross).sum())}/{len(sub)}")

        # The own values sit against the top of the square, so the tally goes low and right.
        ax.text(0.97, 0.05, "own $>$ cross\n" + "\n".join(tallies), transform=ax.transAxes,
                va="bottom", ha="right", fontsize=9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("objective of the other readout, same carrier")

    axes[0].set_ylabel("own objective  $R_{\\mathrm{own}}$")
    axes[0].legend(frameon=False, fontsize=8.6, ncol=3, loc="upper left",
                   bbox_to_anchor=(0.0, -0.14))
    fig.suptitle("Each trained readout dominates on its own target time", fontsize=11.5, y=0.99)
    save(fig, out, "figS7b_objective_crossing")

    report = []
    for title, own_key, cross_key in arms:
        own, cross = floats(rows, own_key), floats(rows, cross_key)
        report.append(f"{own_key}: {int((own > cross).sum())}/{len(rows)}")
    print("figS7b_objective_crossing: own>cross  " + "   ".join(report))
    return True


# ------------------------------------------------------------------ figure S7c

def fig_s7c_isolation_transport(results: Path, out: Path) -> bool:
    """Accuracy against horizon under isolation, and the approximate-isolation floor."""
    path = results / "confirmatory" / "RETENTION_ROWS.csv"
    rows = load_rows(path, "figS7c_isolation_transport")
    if rows is None:
        return False

    config = load_config(results)
    train_horizon = config.get("train_horizon")

    # Left panel. isolated_paired_across_horizons is a diagnostic that reuses a single
    # noise realisation across horizons and is excluded here.
    panel = [r for r in rows if r["condition"] in ("isolated", "open")]
    horizons = sorted({int(r["horizon"]) for r in panel})
    style = {
        ("nonnormal", "isolated"): (NAVY, "-", "o", NAVY, "conditioned $SQS^{-1}$, isolated"),
        ("nonnormal", "open"): (NAVY, "--", "o", CREAM, "conditioned $SQS^{-1}$, open"),
        ("normal", "isolated"): (MUTED, "-", "s", MUTED, "normal $Q$, isolated"),
        ("normal", "open"): (MUTED, "--", "s", CREAM, "normal $Q$, open"),
    }

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.0, 5.0),
                                   gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.24})

    series_handles: list[Line2D] = []
    fixed_handles: list[Line2D] = []
    if horizons:
        for (writer, condition), (colour, dash, marker, face, label) in style.items():
            med, q25, q75, pred, fixed = [], [], [], [], []
            for h in horizons:
                cell = [r for r in panel
                        if r["writer_type"] == writer and r["condition"] == condition
                        and int(r["horizon"]) == h]
                if not cell:
                    med.append(np.nan); q25.append(np.nan); q75.append(np.nan)
                    pred.append(np.nan); fixed.append(np.nan)
                    continue
                rec = floats(cell, "recalibrated_accuracy")
                med.append(float(np.median(rec)))
                q25.append(float(np.percentile(rec, 25)))
                q75.append(float(np.percentile(rec, 75)))
                pred.append(float(np.median(floats(cell, "predicted_accuracy"))))
                fixed.append(float(np.median(floats(cell, "fixed_readout_accuracy"))))
            axl.fill_between(horizons, q25, q75, color=colour, alpha=0.12, lw=0, zorder=2)
            axl.plot(horizons, med, color=colour, ls=dash, marker=marker, ms=6, lw=2,
                     mfc=face, mec=colour, zorder=4)
            axl.plot(horizons, pred, color=colour, ls=":", lw=1.1, zorder=3)
            series_handles.append(Line2D([], [], color=colour, ls=dash, marker=marker, ms=6, lw=2,
                                         mfc=face, mec=colour, label=label))
            if condition == "isolated":
                # One decoder per (draw, writer, seed): nc_experiment passes the trained
                # (w, b) to every horizon, so these markers are the SAME readout evaluated
                # at five horizons. They are left unjoined so the near-vertical excursion
                # at the training horizon is not read as a trajectory.
                axl.plot(horizons, fixed, color=colour, ls="none", marker="v", ms=7,
                         mfc=CREAM, mec=colour, mew=1.3, zorder=5)
                fixed_handles.append(Line2D([], [], color=colour, ls="none", marker="v", ms=7,
                                            mfc=CREAM, mec=colour, mew=1.3,
                                            label=f"fixed readout, isolated ({writer})"))
        fixed_handles.append(Line2D([], [], color=NAVY, ls=":", lw=1.1,
                                    label="predicted $\\Phi(a\\sqrt{J})$, same colour"))

        if train_horizon is not None and horizons[0] <= float(train_horizon) <= horizons[-1]:
            axl.axvline(float(train_horizon), color=MUTED, lw=0.9, ls=":", zorder=1)
            axl.text(float(train_horizon), 0.012, " readout trained here", transform=
                     axl.get_xaxis_transform(), fontsize=8.6, color=MUTED, ha="left", va="bottom")

    axl.set_xscale("log")
    axl.set_xlabel("horizon $n$")
    axl.set_ylabel("accuracy on the learned direction")
    axl.set_title("Isolation preserves the trained direction; the frozen decoder ages", fontsize=11)
    handles = series_handles + fixed_handles
    if handles:
        axl.legend(handles=handles, frameon=False, fontsize=8.0,
                   ncol=2 if len(handles) > 4 else 1,
                   loc="upper left", bbox_to_anchor=(0.0, -0.13))

    # Right panel. Approximate isolation: retained fraction against its guaranteed floor.
    approx = where(rows, condition="approximate_isolation")
    if not approx:
        axr.text(0.5, 0.5, "no approximate_isolation rows", transform=axr.transAxes,
                 ha="center", va="center", fontsize=10, color=MUTED)
        axr.set_xticks([]); axr.set_yticks([])
    else:
        retained = floats(approx, "retained_fraction")
        floor = floats(approx, "guaranteed_floor")
        margin = floats(approx, "bound_margin")
        both = np.concatenate([retained, floor])
        lo, hi = float(both.min()), float(both.max())
        pad = 0.05 * (hi - lo) if hi > lo else 0.02
        lim = (lo - pad, hi + pad)
        axr.plot(lim, lim, color=NAVY, lw=1.2, zorder=4, label="floor, $y = x$")
        groups = {"aligned": (NAVY, "o", NAVY), "random_psd": (MUTED, "^", "none")}
        for disturbance, (colour, marker, face) in groups.items():
            sub = where(approx, disturbance=disturbance)
            if not sub:
                continue
            axr.plot(floats(sub, "guaranteed_floor"), floats(sub, "retained_fraction"), marker,
                     mfc=face, mec=colour, mew=0.5, ms=4, ls="none", alpha=0.45,
                     rasterized=True, zorder=3, label=f"{disturbance}  ($n={len(sub):,}$)")
        axr.set_xlim(lim)
        axr.set_ylim(lim)
        axr.set_aspect("equal")
        axr.set_xlabel("guaranteed floor")
        axr.set_ylabel("retained fraction")
        axr.set_title("Approximate isolation stays on or above its floor", fontsize=11)
        axr.text(0.035, 0.965,
                 f"$n = {len(approx):,}$ rows\nsmallest bound_margin in the file:\n"
                 f"{float(margin.min()):.1e}  (floating-point scale)",
                 transform=axr.transAxes, va="top", ha="left", fontsize=8.6)
        axr.legend(frameon=False, fontsize=8.6, loc="lower right")

    save(fig, out, "figS7c_isolation_transport")
    print(f"figS7c_isolation_transport: {len(panel)} isolated/open rows over horizons {horizons}, "
          f"{len(approx)} approximate_isolation rows")
    return True


# --------------------------------------------------------------------- selftest

_DIRECTIONS = ("v_old", "v_store", "v_write", "v_bottom", "v_worst_positive",
               "v_random_000", "v_random_001", "v_learn_s0", "v_learn_s1")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _synthetic_lane(root: Path) -> None:
    """Build a small synthetic result tree matching the frozen schema. Fabricated
    numbers, shaped only to exercise the plotting code; no scientific meaning."""
    rng = np.random.default_rng(20260918)
    horizons = (64, 128, 256, 1024)
    writers = ("normal", "nonnormal")

    direction_rows: list[dict[str, Any]] = []
    for draw in range(2):
        for writer in writers:
            for condition in ("isolated", "open"):
                for horizon in horizons:
                    for direction in _DIRECTIONS:
                        if direction == "v_bottom":
                            j = float(rng.uniform(1e-33, 1e-30))
                        else:
                            j = float(rng.uniform(1e-4, 3e-1))
                        # arbitrary monotone stand-in for the released predicted column
                        predicted = 0.5 + 0.30 * j / (0.25 + j)
                        direction_rows.append({
                            "draw_index": draw, "writer_type": writer, "condition": condition,
                            "horizon": horizon, "direction": direction, "J": j,
                            "predicted_accuracy": predicted,
                            "recalibrated_accuracy": predicted + float(rng.normal(0, 0.004)),
                            "empirical_J": j * float(rng.uniform(0.98, 1.02)),
                            "rayleigh_efficiency_M_old": float(rng.uniform(0.5, 1.0)),
                            "subspace_alignment": float(rng.uniform(0.5, 1.0)),
                            "cosine_to_v_old": float(rng.uniform(0.5, 1.0)),
                        })
    _write_csv(root / "confirmatory" / "DIRECTION_ROWS.csv",
               ["draw_index", "writer_type", "condition", "horizon", "direction", "J",
                "predicted_accuracy", "recalibrated_accuracy", "empirical_J",
                "rayleigh_efficiency_M_old", "subspace_alignment", "cosine_to_v_old"],
               direction_rows)

    retention_rows: list[dict[str, Any]] = []
    blanks = {"J0": "", "retained_fraction": "", "guaranteed_floor": "", "bound_margin": ""}
    for draw in range(2):
        for writer in writers:
            base = 0.78 if writer == "nonnormal" else 0.71
            for seed in range(2):
                for horizon in horizons:
                    for condition in ("isolated", "open", "isolated_paired_across_horizons"):
                        decay = 1.0 if condition != "open" else 64.0 / horizon
                        acc = 0.5 + (base - 0.5) * decay
                        fixed = acc if horizon == 128 else 0.5 + float(rng.normal(0, 0.02))
                        retention_rows.append({
                            "draw_index": draw, "writer_type": writer, "seed_index": seed,
                            "horizon": horizon, "condition": condition, "disturbance": "none",
                            "alpha": 0.0, "J": float(rng.uniform(0.01, 0.3)),
                            "predicted_accuracy": acc,
                            "recalibrated_accuracy": acc + float(rng.normal(0, 0.003)),
                            "fixed_readout_accuracy": fixed, **blanks,
                        })
                    for disturbance in ("aligned", "random_psd"):
                        for alpha in (0.05, 0.25, 1.0):
                            floor = 1.0 / (1.0 + alpha)
                            slack = 0.0 if disturbance == "aligned" else float(rng.uniform(0.0, 0.12))
                            retention_rows.append({
                                "draw_index": draw, "writer_type": writer, "seed_index": seed,
                                "horizon": horizon, "condition": "approximate_isolation",
                                "disturbance": disturbance, "alpha": alpha,
                                "J": float(rng.uniform(0.01, 0.3)),
                                "predicted_accuracy": 0.6, "recalibrated_accuracy": 0.6,
                                "fixed_readout_accuracy": 0.5,
                                "J0": float(rng.uniform(0.05, 0.2)),
                                "retained_fraction": floor + slack,
                                "guaranteed_floor": floor, "bound_margin": slack,
                            })
    _write_csv(root / "confirmatory" / "RETENTION_ROWS.csv",
               ["draw_index", "writer_type", "seed_index", "horizon", "condition", "disturbance",
                "alpha", "J", "predicted_accuracy", "recalibrated_accuracy",
                "fixed_readout_accuracy", "J0", "retained_fraction", "guaranteed_floor",
                "bound_margin"],
               retention_rows)

    cell_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for draw in range(6):
        for writer in writers:
            own0, own12 = 0.999 - draw * 1e-4, 0.998 - draw * 1e-4
            cross0 = float(rng.uniform(0.30, 0.80))
            cross12 = float(rng.uniform(0.40, 0.80))
            cell_rows.append({
                "draw_index": draw, "writer_type": writer,
                "oracle_overlap": float(rng.uniform(0.1, 0.4)),
                "R_own_t0": own0, "R_own_t12": own12,
                "R_cross_t0_from_t12": cross0, "R_cross_t12_from_t0": cross12,
                "G_star_t0": 0.55, "G_star_t12": 0.25, "G_learn_t0": 0.54, "G_learn_t12": 0.25,
                "C_t0": 0.98, "C_t12": 1.00, "A_sub_own_t0": 0.99, "A_sub_own_t12": 0.99,
                "accuracy_own_t0": 0.72, "accuracy_own_t12": 0.66,
                "lambda_max_t0": 0.16, "lambda_max_t12": 0.08,
            })
            for trained in (0, 12):
                for evaluated in (0, 12):
                    cross_rows.append({
                        "draw_index": draw, "writer_type": writer, "trained_on": trained,
                        "evaluated_on": evaluated, "seed_index": 0,
                        "own_objective": trained == evaluated,
                        "rayleigh_efficiency": float(rng.uniform(0.7, 1.0)),
                        "subspace_alignment": float(rng.uniform(0.02, 1.0)),
                        "J": float(rng.uniform(0.05, 0.17)), "predicted_accuracy": 0.7,
                        "recalibrated_accuracy": 0.7, "fixed_readout_accuracy": 0.6,
                    })
    _write_csv(root / "nc3c" / "CELL_LEVEL.csv",
               ["draw_index", "writer_type", "oracle_overlap", "R_own_t0", "R_own_t12",
                "R_cross_t0_from_t12", "R_cross_t12_from_t0", "G_star_t0", "G_star_t12",
                "G_learn_t0", "G_learn_t12", "C_t0", "C_t12", "A_sub_own_t0", "A_sub_own_t12",
                "accuracy_own_t0", "accuracy_own_t12", "lambda_max_t0", "lambda_max_t12"],
               cell_rows)
    _write_csv(root / "nc3c" / "CROSS_EVALUATION_ROWS.csv",
               ["draw_index", "writer_type", "trained_on", "evaluated_on", "seed_index",
                "own_objective", "rayleigh_efficiency", "subspace_alignment", "J",
                "predicted_accuracy", "recalibrated_accuracy", "fixed_readout_accuracy"],
               cross_rows)

    (root / "confirmatory" / "RUN_RECEIPT.json").write_text(
        json.dumps({"stage": "synthetic_selftest",
                    "config": {"amplitude": 1.5, "train_horizon": 128}}, indent=2))


def selftest() -> int:
    """Render all three figures from synthetic CSVs in a temp dir, then clean up."""
    tmp = Path(tempfile.mkdtemp(prefix="section7_selftest_"))
    try:
        results, out = tmp / "results", tmp / "figures"
        _synthetic_lane(results)
        built = [fig_s7a_calibration(results, out),
                 fig_s7b_objective_crossing(results, out),
                 fig_s7c_isolation_transport(results, out)]
        assert all(built), f"a figure did not render: {built}"
        names = ("figS7a_calibration", "figS7b_objective_crossing", "figS7c_isolation_transport")
        for name in names:
            for suffix in (".png", ".pdf"):
                target = out / f"{name}{suffix}"
                assert target.is_file(), f"missing output {target}"
                size = target.stat().st_size
                assert size > 0, f"empty output {target}"
                print(f"  ok {name}{suffix}  {size:,} bytes")
        missing = results / "confirmatory" / "DIRECTION_ROWS.csv"
        missing.unlink()
        assert fig_s7a_calibration(results, out) is False, "missing-input skip did not return False"
        print("selftest PASS: 3 png + 3 pdf written and non-empty; missing-input skip works")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"selftest temp dir removed: {tmp}")


# ------------------------------------------------------------------------- main

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=LANE / "results",
                        help="frozen results base directory (default: <lane>/results)")
    parser.add_argument("--out", type=Path, default=None,
                        help="figure output directory (default: <results>/figures)")
    parser.add_argument("--selftest", action="store_true",
                        help="render all three figures from synthetic CSVs in a temp dir")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    results = args.results.resolve()
    out = (args.out or results / "figures").resolve()
    built = sum([fig_s7a_calibration(results, out),
                 fig_s7b_objective_crossing(results, out),
                 fig_s7c_isolation_transport(results, out)])
    print(f"{built}/3 figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
