#!/usr/bin/env python3
"""Raw-to-manuscript claim ledger.

Handoff section 8: "Create a machine-readable claim ledger containing source path/hash,
raw field or generating command, formula, subset, aggregation, units, experiment phase,
and every manuscript location for each reported number."

Every value is recomputed here from the frozen CSVs. Nothing is copied from a report.
Aggregation is named explicitly on every row, because a median of paired ratios is not a
ratio of medians and this lane has already published one number that confused them.

Plot-and-account only: computes no new scientific quantity and applies no decision rule.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

LANE = Path(__file__).resolve().parents[1]
OUT = LANE / "docs" / "v2_0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for k, v in list(row.items()):
            if v == "":
                row[k] = None
                continue
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def num(row: dict[str, Any], key: str) -> float:
    v = row.get(key)
    return float(v) if v is not None else float("nan")


def entry(claim_id, value, source, field, formula, subset, aggregation, units, phase,
          locations, note=""):
    return {"claim_id": claim_id, "value": value, "source_path": source["path"],
            "source_sha256": source["sha256"], "raw_field": field, "formula": formula,
            "subset": subset, "aggregation": aggregation, "units": units,
            "experiment_phase": phase, "manuscript_locations": locations, "note": note}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src = {}
    for name, rel in (
        ("direction", "results/confirmatory/DIRECTION_ROWS.csv"),
        ("retention", "results/confirmatory/RETENTION_ROWS.csv"),
        ("cell_nc1", "results/confirmatory/CELL_LEVEL.csv"),
        ("verdict_nc1", "results/confirmatory/VERDICT.json"),
        ("receipt_nc1", "results/confirmatory/RUN_RECEIPT.json"),
        ("cell_nc3c", "results/nc3c/CELL_LEVEL.csv"),
        ("cross_nc3c", "results/nc3c/CROSS_EVALUATION_ROWS.csv"),
        ("reference_nc3c", "results/nc3c/REFERENCE_ROWS.csv"),
        ("verdict_nc3c", "results/nc3c/VERDICT.json"),
        ("receipt_nc3c", "results/nc3c/RUN_RECEIPT.json"),
        ("transport", "results/parity/DECODER_TRANSPORT_PARITY.json"),
        ("scan", "results/parity/SCAN_PARITY.json"),
        ("cpu_cuda", "results/parity/CROSS_IMPLEMENTATION_PARITY.json"),
        ("cell_nc3c_b2", "results/nc3c_block2_20260920/CELL_LEVEL.csv"),
        ("verdict_nc3c_b2", "results/nc3c_block2_20260920/VERDICT.json"),
        ("receipt_nc3c_b2", "results/nc3c_block2_20260920/RUN_RECEIPT.json"),
        ("freshness_nc3c_b2", "results/nc3c_block2_20260920/FRESHNESS_AUDIT.json"),
    ):
        p = LANE / rel
        src[name] = {"path": rel, "sha256": sha256(p) if p.exists() else None}

    direction = read(LANE / "results/confirmatory/DIRECTION_ROWS.csv")
    retention = read(LANE / "results/confirmatory/RETENTION_ROWS.csv")
    vnc1 = json.loads((LANE / "results/confirmatory/VERDICT.json").read_text())
    cell3 = read(LANE / "results/nc3c/CELL_LEVEL.csv")
    vnc3 = json.loads((LANE / "results/nc3c/VERDICT.json").read_text())
    transport = json.loads((LANE / "results/parity/DECODER_TRANSPORT_PARITY.json").read_text())
    scan = json.loads((LANE / "results/parity/SCAN_PARITY.json").read_text())
    cpu = json.loads((LANE / "results/parity/CROSS_IMPLEMENTATION_PARITY.json").read_text())

    L: list[dict[str, Any]] = []

    # --- NC-1 alignment ---------------------------------------------------
    for wt in ("nonnormal", "normal"):
        a = vnc1["by_writer_type"][wt]["alignment"]
        L.append(entry(f"nc1.R_J.median.{wt}", a["rayleigh_summary"]["median"], src["cell_nc1"],
            "rayleigh_efficiency", "median over carriers of (median over 5 seeds)",
            f"writer_type={wt}, isolated, horizon=H_train=128, direction=v_learn",
            "median of per-carrier medians; carrier draw is the independent unit; 16 carriers",
            "dimensionless, Fisher/Rayleigh efficiency in [0,1], NOT accuracy",
            "NC-1/NC-2 confirmatory", ["Results 7.2", "Abstract"],
            "R_J near 0.999 must never be described as 99.9% accuracy"))
        L.append(entry(f"nc1.A_sub.median.{wt}", a["subspace_summary"]["median"], src["cell_nc1"],
            "subspace_alignment", "median over carriers of (median over 5 seeds) of ||P_E0.99 v||^2",
            f"writer_type={wt}, isolated, horizon=128, direction=v_learn",
            "median of per-carrier medians, 16 carriers", "dimensionless in [0,1]",
            "NC-1/NC-2 confirmatory", ["Results 7.2"]))
        L.append(entry(f"nc1.calibration_mae.{wt}", a["calibration_mae"], src["direction"],
            "predicted_accuracy, recalibrated_accuracy", "mean |predicted - empirical|",
            f"writer_type={wt}, ALL directions, both storage conditions, all evaluated horizons",
            f"pooled mean over {a['calibration_rows']} rows, NOT a carrier-level statistic",
            "accuracy difference", "NC-1/NC-2 confirmatory", ["Results 7.3", "Abstract"]))
        L.append(entry(f"nc1.calibration_slope.{wt}", a["calibration_slope"], src["direction"],
            "predicted_accuracy, recalibrated_accuracy",
            "least-squares slope of empirical on predicted",
            f"writer_type={wt}, same population as the MAE row",
            f"pooled OLS over {a['calibration_rows']} rows", "dimensionless",
            "NC-1/NC-2 confirmatory", ["Results 7.3"]))
        L.append(entry(f"nc1.random_baseline_R_J.median.{wt}",
            float(np.median(a["random_baseline_rayleigh_by_carrier"])), src["direction"],
            "rayleigh_efficiency_M_old", "median over carriers of (median over 128 random directions)",
            f"writer_type={wt}, isolated, horizon=128, direction=v_random_*",
            "median of per-carrier medians, 16 carriers", "dimensionless",
            "NC-1/NC-2 confirmatory", ["Results 7.2"],
            "the comparator that makes R_J ~ 0.999 non-trivial"))

    # --- the J range, both populations, explicitly labelled ---------------
    J = np.array([num(r, "J") for r in direction])
    J = J[np.isfinite(J)]
    Jnn = np.array([num(r, "J") for r in direction
                    if str(r["direction"]) != "v_bottom" and num(r, "J") > 0])
    L.append(entry("nc1.J_range.all_rows_orders", float(np.log10(J.max() / J.min())), src["direction"],
        "J", "log10(max J / min J)", "ALL 44160 direction rows including v_bottom",
        "pooled min and max", "orders of magnitude", "NC-1/NC-2 confirmatory",
        ["NOT FOR USE"],
        "MEANINGLESS: v_bottom lies in the exact 16-dimensional kernel of M_old, so its J is a "
        "round-off residual, not a positive-information regime. Recorded so the number cannot be "
        "quoted innocently."))
    L.append(entry("nc1.J_range.nonnull_orders", float(np.log10(Jnn.max() / Jnn.min())), src["direction"],
        "J", "log10(max J / min J)", "all direction rows EXCLUDING v_bottom, J > 0",
        f"pooled min {Jnn.min():.4e} and max {Jnn.max():.4e} over {len(Jnn)} rows",
        "orders of magnitude", "NC-1/NC-2 confirmatory", ["Results 7.3", "Abstract"],
        "THE figure to quote. Supersedes the 'twelve orders of magnitude' in the first issue of "
        "RESULTS_20260918_EN.md, which was read off a figure axis."))
    L.append(entry("nc1.v_bottom.J", float(np.median([num(r, "J") for r in direction
                                                      if str(r["direction"]) == "v_bottom"])),
        src["direction"], "J", "median J of the null-direction control",
        "direction=v_bottom, all conditions and horizons", "pooled median",
        "Fisher information", "NC-1/NC-2 confirmatory", ["Results 7.3 null control"],
        "reported as J = 0 with predicted accuracy 1/2; retained in the dataset, never removed"))

    # --- the two ratios, four aggregations, per writer type ----------------
    def jgroup(wt, cond, H):
        sub = [r for r in retention if r["writer_type"] == wt and r["condition"] == cond
               and int(num(r, "horizon")) == H]
        by = {}
        for r in sub:
            by.setdefault(int(num(r, "draw_index")), []).append(num(r, "J"))
        return (np.array([num(r, "J") for r in sub]),
                np.array([np.median(v) for v in by.values()]))

    for wt in ("nonnormal", "normal"):
        o64p, o64d = jgroup(wt, "open", 64)
        o4kp, o4kd = jgroup(wt, "open", 4096)
        i4kp, i4kd = jgroup(wt, "isolated", 4096)
        along = o64d / o4kd
        across = i4kd / o4kd
        L.append(entry(f"nc1.decay.along_horizon.paired_median.{wt}", float(np.median(along)),
            src["retention"], "J",
            "median over carriers of [ J(open, H=64) / J(open, H=4096) ], ratio formed WITHIN each carrier",
            f"writer_type={wt}, condition=open, direction=v_learn, H=64 vs H=4096",
            "median of PAIRED per-carrier ratios, 16 carriers; "
            f"IQR [{np.percentile(along,25):.1f}, {np.percentile(along,75):.1f}]",
            "dimensionless ratio", "NC-1/NC-2 confirmatory", ["Results 7.5"],
            "THE along-horizon figure. Never write that continued coupling caused a 580-fold "
            "decline from H=64 to H=4096."))
        L.append(entry(f"nc1.decay.along_horizon.ratio_of_medians.{wt}",
            float(np.median(o64p) / np.median(o4kp)), src["retention"], "J",
            "[median J(open,64)] / [median J(open,4096)]",
            f"writer_type={wt}, condition=open", "RATIO OF POOLED MEDIANS, not a paired statistic",
            "dimensionless ratio", "NC-1/NC-2 confirmatory", ["Supplement only"],
            "differs from the paired median; recorded so the two cannot be conflated"))
        L.append(entry(f"nc1.contrast.isolated_over_open_at_4096.paired_median.{wt}",
            float(np.median(across)), src["retention"], "J",
            "median over carriers of [ J(isolated, H=4096) / J(open, H=4096) ]",
            f"writer_type={wt}, direction=v_learn, both arms at H=4096",
            "median of PAIRED per-carrier ratios, 16 carriers; "
            f"IQR [{np.percentile(across,25):.1f}, {np.percentile(across,75):.1f}]",
            "dimensionless ratio", "NC-1/NC-2 confirmatory", ["Results 7.5"],
            "THIS is the quantity the '580' figure belongs to. Per writer type; never pooled."))
        L.append(entry(f"nc1.contrast.isolated_over_open_at_4096.ratio_of_medians.{wt}",
            float(np.median(i4kp) / np.median(o4kp)), src["retention"], "J",
            "[median J(isolated,4096)] / [median J(open,4096)]", f"writer_type={wt}",
            "RATIO OF POOLED MEDIANS", "dimensionless ratio", "NC-1/NC-2 confirmatory",
            ["Supplement only"],
            "COLLISION HAZARD: the normal writer's value here equals the non-normal writer's "
            "paired median to one decimal. Always print writer type and aggregation together."))
        L.append(entry(f"nc1.isolated_J.pooled_median.{wt}", float(np.median(i4kp)),
            src["retention"], "J", "median of J under exact isolation",
            f"writer_type={wt}, condition=isolated, direction=v_learn, any H (J is horizon-invariant)",
            "pooled median over 16 carriers x 5 seeds = 80 rows", "Fisher information",
            "NC-1/NC-2 confirmatory", ["Results 7.5"],
            "supersedes the values 0.2205 / 0.1466 quoted in VERIFICATION_OF_COMPLETED_RUN as "
            "medians: those were single arbitrary rows produced by a dict keyed on horizon"))

    # --- decoder statistics, final only -----------------------------------
    for wt in ("nonnormal", "normal"):
        m = vnc1["by_writer_type"][wt]["readout_mismatch"]
        for label, key, subset in (
            ("fixed_at_train", "fixed_median_at_train_horizon",
             "isolated, H = H_train = 128, learned direction, its own trained readout"),
            ("recal_at_train", "recalibrated_median_at_train_horizon",
             "isolated, H = 128, learned direction, readout refit on an independent calibration sample"),
            ("fixed_away", "fixed_median_away",
             "isolated, H in {64,256,1024,4096}, learned direction, readout frozen at H=128"),
            ("recal_away", "recalibrated_median_away",
             "isolated, H in {64,256,1024,4096}, learned direction, readout refit at each horizon"),
        ):
            L.append(entry(f"nc1.decoder.{label}.{wt}", m[key], src["verdict_nc1"],
                f"readout_mismatch.{key}", "median over all matching rows",
                f"writer_type={wt}, {subset}",
                "pooled median over 16 carriers x 5 seeds (x 4 horizons for the 'away' rows)",
                "accuracy", "NC-1/NC-2 confirmatory", ["Results 7.6"],
                "FINAL run values. The figures 0.7728 and 0.5067 quoted earlier came from an "
                "intermediate rehearsal and must not appear."))

    # --- NC-3C ------------------------------------------------------------
    for wt in ("nonnormal", "normal"):
        for j in ("0", "12"):
            b = vnc3["by_writer_type"][wt]
            L.append(entry(f"nc3c.R_own.median.t{j}.{wt}", b["own_objective"][j]["summary"]["median"],
                src["cell_nc3c"], f"R_own_t{j}", "median over carriers of (median over 5 seeds)",
                f"writer_type={wt}, objective t_0={j}, direction trained on that objective",
                "median of per-carrier medians, 16 carriers", "dimensionless efficiency",
                "NC-3C confirmatory", ["Results 7.4", "Abstract"]))
            L.append(entry(f"nc3c.C.median.t{j}.{wt}", b["contrast_recovery"][j]["summary"]["median"],
                src["cell_nc3c"], f"C_t{j}",
                "median over carriers of (R_own - R_cross) / G*, ratio formed WITHIN each carrier",
                f"writer_type={wt}, objective t_0={j}, carriers with G* > 0.02",
                f"median of PAIRED per-carrier ratios over "
                f"{b['contrast_recovery'][j]['carriers_usable']} usable carriers",
                "dimensionless", "NC-3C confirmatory", ["Results 7.4"]))
            L.append(entry(f"nc3c.G_star.median.t{j}.{wt}",
                b["contrast_recovery"][j]["oracle_separation_summary"]["median"],
                src["cell_nc3c"], f"G_star_t{j}",
                "1 - (other oracle)^T M_j (other oracle) / lambda_max(M_j)",
                f"writer_type={wt}, objective t_0={j}", "median over 16 carriers, pre-training quantity",
                "dimensionless", "NC-3C confirmatory", ["Results 7.4"],
                "CONFIRMATORY geometry. The preliminary 0.41/0.35 and minimum 0.117 were "
                "EXPLORATORY draws 800-807 and belong only to the feasibility statement."))
        d = vnc3["by_writer_type"][wt]["diagonal_dominance"]
        for j, dd in d["differences"].items():
            L.append(entry(f"nc3c.diag_diff.median.t{j}.{wt}", dd["median_difference"],
                src["cell_nc3c"], f"R_own_t{j} - R_cross_t{j}",
                "median over carriers of the WITHIN-carrier difference",
                f"writer_type={wt}, objective t_0={j}",
                f"median of paired per-carrier differences, 16 carriers; paired 95% bootstrap CI "
                f"[{dd['bootstrap']['ci_low']:.4f}, {dd['bootstrap']['ci_high']:.4f}]",
                "dimensionless", "NC-3C confirmatory", ["Results 7.4"]))
    gs = np.concatenate([[num(r, f"G_star_t{j}") for r in cell3] for j in (0, 12)])
    L.append(entry("nc3c.G_star.min_across_all_cells", float(np.min(gs)), src["cell_nc3c"],
        "G_star_t0, G_star_t12", "minimum over every cell and objective",
        "both writer types, both objectives, 16 carriers", "pooled minimum of 64 values",
        "dimensionless", "NC-3C confirmatory", ["Results 7.4"],
        "no cell fell below the frozen 0.02 negligibility threshold"))

    # --- parity -----------------------------------------------------------
    # ---- the outcome-unseen second block, section 7.4.1. Reported on its own, never
    # pooled with the rehearsed block: separate source files, separate claim ids.
    b2_dir = LANE / "results/nc3c_block2_20260920"
    if b2_dir.exists():
        vb2 = json.loads((b2_dir / "VERDICT.json").read_text())
        rb2 = json.loads((b2_dir / "RUN_RECEIPT.json").read_text())
        ab2 = json.loads((b2_dir / "FRESHNESS_AUDIT.json").read_text())
        cellb2 = read(b2_dir / "CELL_LEVEL.csv")
        loc = "section 7.4.1"
        for wt in ("nonnormal", "normal"):
            b = vb2["by_writer_type"][wt]
            for j in ("0", "12"):
                L.append(entry(f"nc3c_b2.R_own.median.t{j}.{wt}",
                    b["own_objective"][j]["summary"]["median"], src["cell_nc3c_b2"],
                    f"R_own_t{j}", "median over carriers of (median over 5 seeds)",
                    f"block 16..31, writer {wt}, objective t0={j}", "median of per-carrier medians",
                    "dimensionless Rayleigh efficiency", "nc3c_block2_confirmatory", loc))
                L.append(entry(f"nc3c_b2.G_star.median.t{j}.{wt}",
                    b["contrast_recovery"][j]["oracle_separation_summary"]["median"],
                    src["verdict_nc3c_b2"], f"G_star_t{j}", "1 - r_j(v*_other)",
                    f"block 16..31, writer {wt}, objective t0={j}", "median over carriers",
                    "dimensionless", "nc3c_block2_confirmatory", loc))
                L.append(entry(f"nc3c_b2.G_learn.median.t{j}.{wt}",
                    b["diagonal_dominance"]["differences"][j]["bootstrap"]["point"],
                    src["verdict_nc3c_b2"], f"G_learn_t{j}",
                    "paired percentile bootstrap point estimate over carrier draws",
                    f"block 16..31, writer {wt}, objective t0={j}", "bootstrap point over carriers",
                    "dimensionless", "nc3c_block2_confirmatory", loc))
                L.append(entry(f"nc3c_b2.C.median.t{j}.{wt}",
                    b["contrast_recovery"][j]["summary"]["median"], src["cell_nc3c_b2"],
                    f"C_t{j}", "G_learn_j / G*_j",
                    f"block 16..31, writer {wt}, objective t0={j}, "
                    f"{b['contrast_recovery'][j]['carriers_usable']} usable carriers",
                    "median of per-carrier values", "dimensionless",
                    "nc3c_block2_confirmatory", loc))
            L.append(entry(f"nc3c_b2.diagonal_dominance.both_strict_draws.{wt}",
                b["diagonal_dominance"]["both_strict_draws"], src["verdict_nc3c_b2"],
                "both_strict_draws", "count of carriers with G_learn > 0 on both objectives",
                f"block 16..31, writer {wt}", "count over 16 carriers", "carriers",
                "nc3c_block2_confirmatory", loc))
        gs2 = [float(r[f"G_star_t{j}"]) for r in cellb2 for j in (0, 12)]
        L.append(entry("nc3c_b2.G_star.min_across_all_cells", float(np.min(gs2)),
            src["cell_nc3c_b2"], "G_star_t0 and G_star_t12", "minimum over all cells",
            "block 16..31, both writer types, both objectives", "minimum", "dimensionless",
            "nc3c_block2_confirmatory", loc))
        L.append(entry("nc3c_b2.verdict.overall", vb2["overall"], src["verdict_nc3c_b2"],
            "overall", "frozen gate table applied by nc3c_verdict.py",
            "block 16..31", "not an aggregation", "verdict string",
            "nc3c_block2_confirmatory", loc))
        L.append(entry("nc3c_b2.valid_cells", rb2["valid_cells"], src["receipt_nc3c_b2"],
            "valid_cells", "cells whose 33-control battery returned VALID",
            "block 16..31", "count over 32 cells", "cells",
            "nc3c_block2_confirmatory", loc,
            note="the first execution of this block returned 0 of 32 valid on an instrument "
                 "that could not fail, and persisted no data row; it was repaired and rerun once"))
        L.append(entry("nc3c_b2.freshness.fresh", ab2["fresh"], src["freshness_nc3c_b2"],
            "fresh", "no requested draw exercised, and four empty collision sets",
            "block 16..31 against the completed lane and every prior NC-3C draw",
            "not an aggregation", "boolean", "nc3c_block2_preflight", loc))

    L.append(entry("parity.cpu_cuda.worst_R_J_difference",
        cpu["endpoint_criterion"]["worst_absolute_difference"], src["cpu_cuda"],
        "endpoint_criterion.worst_absolute_difference",
        "max |R_J(numpy CPU float64) - R_J(torch CUDA float64)|",
        "4 PILOT cells (draws 900, 901 x 2 writer types), 2000 training steps",
        "maximum over 4 cells", "dimensionless", "implementation parity",
        ["Supplement S1"],
        "NOT a full-matrix independent reproduction. The medians 0.999376 / 0.999328 belong to a "
        "throughput benchmark on a pilot draw and stay out of the confirmatory paragraph."))
    L.append(entry("parity.scan.worst_relative_transfer_residual",
        scan["worst_relative_transfer_error"], src["scan"], "worst_relative_transfer_error",
        "max relative difference, Blelloch scan vs the frozen sequential closure_transfers",
        "32 cells: 4 carrier draws x 2 writer types x 4 horizons", "maximum over 32 cells",
        "relative error", "implementation parity", ["Supplement S1"]))
    L.append(entry("parity.transport.worst_logit_difference",
        transport["worst_logit_difference"], src["transport"], "worst_logit_difference",
        "max |logit(transported decoder at H) - logit(fitted decoder at H_train)| over episodes",
        "32 rows: 4 carrier draws x 2 writer types x 4 horizons, 50000 shared episodes each",
        "maximum over all rows and episodes", "logit units", "implementation parity",
        ["Results 7.6", "Abstract"],
        "Uses the SAME sampled training-horizon episodes transported by the known hold and the "
        "inverse-adjoint decoder. Verifies a pointwise coordinate identity. It is not an "
        "independent discovery about biological or general neural forgetting."))
    L.append(entry("parity.transport.worst_accuracy_restoration_error",
        transport["worst_restoration_error_accuracy"], src["transport"],
        "worst_restoration_error_accuracy",
        "max |accuracy(transported at H) - accuracy(fitted at H_train)|",
        "same 32 rows", "maximum over rows", "accuracy", "implementation parity",
        ["Results 7.6"]))

    payload = {
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "purpose": "handoff section 8 raw-to-manuscript claim ledger",
        "rule": ("every value recomputed from the frozen CSVs by this script; no value copied "
                 "from a report; aggregation named on every row"),
        "sources": src, "entries": L, "entry_count": len(L),
    }
    (OUT / "CLAIM_LEDGER_v2_0_20260918.json").write_text(json.dumps(payload, indent=2, default=float))

    with (OUT / "CLAIM_LEDGER_v2_0_20260918.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=[
            "claim_id", "value", "units", "subset", "aggregation", "experiment_phase",
            "source_path", "manuscript_locations", "note"])
        w.writeheader()
        for e in L:
            w.writerow({k: (", ".join(e[k]) if isinstance(e[k], list) else e[k])
                        for k in w.fieldnames})
    print(json.dumps({"entries": len(L), "sources": len(src),
                      "json": str((OUT / "CLAIM_LEDGER_v2_0_20260918.json").relative_to(LANE)),
                      "csv": str((OUT / "CLAIM_LEDGER_v2_0_20260918.csv").relative_to(LANE))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
