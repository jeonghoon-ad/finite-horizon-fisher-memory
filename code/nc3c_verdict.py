#!/usr/bin/env python3
"""Apply the frozen NC-3C decision rules to the confirmatory artifacts.

Authority: ../authority/nc3c/NC3C_TASK_SPECIFIC_GEOMETRY_PROTOCOL_v1_20260918_EN.md section 8
           ../authority/nc3c/NC3C_TASK_SPECIFIC_GEOMETRY_PREREG_v1_20260918.yaml decision_rules

Reads only the CSVs and the run receipt. Takes no threshold and no horizon from the
command line. Refuses to run while the invalid-run ledger is non-empty, and refuses a
denominator that is not the preregistered one.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import nc3c_experiment as X
import nc_stats as S

BOOTSTRAP_SEED = 20260918
RESAMPLES = 10000

GATES = {
    "own_objective": {
        "median_min": 0.90,
        "bootstrap_lower_bound_min": 0.80,
        "carrier_draws_at_or_above_0.80_min": 14,
    },
    "diagonal_dominance": {
        "carrier_draws_with_both_min": 14,
        "paired_interval_excludes_zero": True,
    },
    "contrast_recovery": {
        "median_min": 0.75,
        "bootstrap_lower_bound_min": 0.50,
        "negligible_separation_threshold": X.NEGLIGIBLE_SEPARATION,
        # Without a floor, a run in which 15 of 16 carriers were certified
        # geometrically indistinguishable would compute the median and the 10,000
        # resample bootstrap on ONE value, which the bound then satisfies by
        # construction, and report contrast_recovery_pass. The floor is the same
        # 14-of-16 denominator every other count gate uses.
        "minimum_usable_carriers": 14,
        # Protocol section 9 kill rule 4: "if both task operators are nearly identical
        # on MOST draws, the objectives do not constitute a discriminating
        # confirmation". Most of 16 is more than 8.
        "not_discriminating_negligible_carriers": 9,
    },
    "calibration": {
        "mae_max": 0.02,
        "slope_interval": [0.95, 1.05],
        "ordered_pair_agreement_min": 0.90,
        "ordered_pair_relative_gap": 0.05,
    },
    "kill": {
        "own_objective_median_below": 0.75,
    },
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def _per_carrier(cell_rows: list[dict[str, Any]], writer_type: str,
                 key: str) -> tuple[np.ndarray, list[int]]:
    by_draw: dict[int, float] = {}
    for row in cell_rows:
        if row["writer_type"] != writer_type:
            continue
        by_draw[int(row["draw_index"])] = float(row[key])
    draws = sorted(by_draw)
    return np.array([by_draw[d] for d in draws]), draws


def own_objective_family(cell_rows: list[dict[str, Any]], writer_type: str,
                         objective: int) -> dict[str, Any]:
    values, draws = _per_carrier(cell_rows, writer_type, f"R_own_t{objective}")
    lower = S.bootstrap_lower_bound(values, seed=BOOTSTRAP_SEED, resamples=RESAMPLES)
    boot = S.paired_bootstrap(values, seed=BOOTSTRAP_SEED, resamples=RESAMPLES)
    checks = [
        S.evaluate_gate(f"median_R_own_t{objective}", float(np.median(values)), ">=",
                        GATES["own_objective"]["median_min"]),
        S.evaluate_gate(f"bootstrap_lower_bound_t{objective}", lower, ">=",
                        GATES["own_objective"]["bootstrap_lower_bound_min"]),
        S.evaluate_gate(f"carriers_at_or_above_0.80_t{objective}",
                        float(np.sum(values >= 0.80)), ">=",
                        GATES["own_objective"]["carrier_draws_at_or_above_0.80_min"]),
    ]
    return {
        "objective": objective, "carrier_draws": draws,
        "by_carrier": values.tolist(), "summary": S.summarize(values),
        "bootstrap": boot, "bootstrap_one_sided_lower": lower,
        "checks": checks, "pass": all(c["pass"] for c in checks),
        "kill_triggered": bool(
            float(np.median(values)) < GATES["kill"]["own_objective_median_below"]),
    }


def diagonal_dominance_family(cell_rows: list[dict[str, Any]],
                              writer_type: str) -> dict[str, Any]:
    objectives = list(X.OBJECTIVES)
    own: dict[int, np.ndarray] = {}
    cross: dict[int, np.ndarray] = {}
    draws: list[int] = []
    for j in objectives:
        other = [t for t in objectives if t != j][0]
        own[j], draws = _per_carrier(cell_rows, writer_type, f"R_own_t{j}")
        cross[j], _ = _per_carrier(cell_rows, writer_type, f"R_cross_t{j}_from_t{other}")

    both = np.ones(len(draws), dtype=bool)
    differences: dict[int, dict[str, Any]] = {}
    for j in objectives:
        strict = own[j] > cross[j]
        both &= strict
        diff = own[j] - cross[j]
        interval = S.paired_bootstrap_difference(
            own[j], cross[j], seed=BOOTSTRAP_SEED, resamples=RESAMPLES)
        differences[j] = {
            "median_difference": float(np.median(diff)),
            "draws_strict": int(np.sum(strict)),
            "bootstrap": interval,
            "interval_excludes_zero": bool(interval["ci_low"] > 0.0 or interval["ci_high"] < 0.0),
            "by_carrier": diff.tolist(),
        }

    checks = [S.evaluate_gate("carriers_with_both_strict_inequalities",
                              float(np.sum(both)), ">=",
                              GATES["diagonal_dominance"]["carrier_draws_with_both_min"])]
    for j in objectives:
        checks.append(S.evaluate_gate(
            f"paired_interval_excludes_zero_t{j}",
            1.0 if differences[j]["interval_excludes_zero"] else 0.0, ">=", 1.0))
    return {"carrier_draws": draws, "both_strict_draws": int(np.sum(both)),
            "differences": {str(k): v for k, v in differences.items()},
            "checks": checks, "pass": all(c["pass"] for c in checks)}


def contrast_family(cell_rows: list[dict[str, Any]], writer_type: str,
                    objective: int) -> dict[str, Any]:
    rows = [r for r in cell_rows if r["writer_type"] == writer_type]
    negligible = [int(r["draw_index"]) for r in rows
                  if float(r[f"G_star_t{objective}"]) <= X.NEGLIGIBLE_SEPARATION]
    usable = [r for r in rows if int(r["draw_index"]) not in negligible]
    values = np.array([float(r[f"C_t{objective}"]) for r in usable])
    draws = [int(r["draw_index"]) for r in usable]
    total = len({int(r["draw_index"]) for r in rows})
    not_discriminating = bool(
        len(negligible) >= GATES["contrast_recovery"]["not_discriminating_negligible_carriers"])
    if len(values) and not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"{writer_type} t{objective}: non-finite contrast recovery on a usable "
            "carrier; that is an instrument failure, not a result")
    if not_discriminating or len(values) < GATES["contrast_recovery"]["minimum_usable_carriers"]:
        # Kill rule 4 is a RECORDED VERDICT, not a traceback. The experiment ran; what
        # it found is that these two objectives are not separable in this geometry.
        return {
            "objective": objective,
            "usable_carrier_draws": draws,
            "negligible_separation_carriers": negligible,
            "negligible_label": "OBJECTIVES_GEOMETRICALLY_INDISTINGUISHABLE",
            "carriers_total": total, "carriers_usable": len(values),
            "minimum_usable_carriers": GATES["contrast_recovery"]["minimum_usable_carriers"],
            "kill_rule_4_triggered": not_discriminating,
            "verdict": "OBJECTIVES_NOT_DISCRIMINATING",
            "checks": [S.evaluate_gate(f"usable_carriers_t{objective}", float(len(values)),
                                       ">=", float(GATES["contrast_recovery"]["minimum_usable_carriers"]))],
            "pass": False,
            "reading": ("the contrast statistic was not evaluated: too few carriers have a "
                        "separation the geometry can express. Protocol section 9 kill rule 4."),
        }
    lower = S.bootstrap_lower_bound(values, seed=BOOTSTRAP_SEED, resamples=RESAMPLES)
    checks = [
        S.evaluate_gate(f"median_contrast_recovery_t{objective}",
                        float(np.median(values)), ">=",
                        GATES["contrast_recovery"]["median_min"]),
        S.evaluate_gate(f"contrast_bootstrap_lower_bound_t{objective}", lower, ">=",
                        GATES["contrast_recovery"]["bootstrap_lower_bound_min"]),
    ]
    star, _ = _per_carrier(cell_rows, writer_type, f"G_star_t{objective}")
    return {
        "objective": objective,
        "usable_carrier_draws": draws,
        "negligible_separation_carriers": negligible,
        "negligible_label": "OBJECTIVES_GEOMETRICALLY_INDISTINGUISHABLE",
        "carriers_total": total, "carriers_usable": len(values),
        "kill_rule_4_triggered": False, "verdict": "EVALUATED",
        "by_carrier": values.tolist(),
        "summary": S.summarize(values),
        "oracle_separation_summary": S.summarize(star),
        "bootstrap_one_sided_lower": lower,
        "checks": checks, "pass": all(c["pass"] for c in checks),
    }


def calibration_family(reference_rows: list[dict[str, Any]],
                       cross_rows: list[dict[str, Any]],
                       writer_type: str, objective: int) -> dict[str, Any]:
    rows = [r for r in reference_rows
            if r["writer_type"] == writer_type and int(float(r["evaluated_on"])) == objective]
    rows += [r for r in cross_rows
             if r["writer_type"] == writer_type and int(float(r["evaluated_on"])) == objective]
    predicted = np.array([float(r["predicted_accuracy"]) for r in rows])
    observed = np.array([float(r["recalibrated_accuracy"]) for r in rows])
    mae = float(np.mean(np.abs(predicted - observed)))
    design = np.column_stack([predicted, np.ones_like(predicted)])
    slope, intercept = np.linalg.lstsq(design, observed, rcond=None)[0]

    per_draw = defaultdict(list)
    for r in rows:
        per_draw[int(r["draw_index"])].append(r)
    agreements = []
    for draw, subset in sorted(per_draw.items()):
        J = np.array([float(r["J"]) for r in subset])
        acc = np.array([float(r["recalibrated_accuracy"]) for r in subset])
        agreements.append(S.ordered_pair_agreement(
            J, acc, relative_gap=GATES["calibration"]["ordered_pair_relative_gap"])["fraction"])
    low, high = GATES["calibration"]["slope_interval"]
    checks = [
        S.evaluate_gate(f"calibration_mae_t{objective}", mae, "<=",
                        GATES["calibration"]["mae_max"]),
        S.evaluate_gate(f"calibration_slope_lower_t{objective}", float(slope), ">=", low),
        S.evaluate_gate(f"calibration_slope_upper_t{objective}", float(slope), "<=", high),
        S.evaluate_gate(f"ordered_pair_agreement_t{objective}",
                        float(np.median(agreements)), ">=",
                        GATES["calibration"]["ordered_pair_agreement_min"]),
    ]
    return {"objective": objective, "rows": len(rows), "mae": mae,
            "slope": float(slope), "intercept": float(intercept),
            "ordered_pair_summary": S.summarize(np.array(agreements)),
            "checks": checks, "pass": all(c["pass"] for c in checks)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.results) if args.results else Path(__file__).resolve().parents[1] / "results" / "nc3c"

    receipt = json.loads((root / "RUN_RECEIPT.json").read_text())
    # The budget is pinned HERE, not trusted from the file that claims it. A reduced
    # budget run otherwise writes a VERDICT.json indistinguishable from a confirmatory
    # one, which the lane already demonstrated by producing exactly such a file.
    import run_nc3c
    drift = {k: (receipt["config"].get(k), v) for k, v in run_nc3c.FROZEN.items()
             if receipt["config"].get(k) != v}
    if drift:
        raise RuntimeError(
            "the run configuration differs from the frozen preregistration and no verdict "
            f"may be computed from it; observed vs frozen: {drift}")
    if receipt.get("stage") != "nc3c_confirmatory" or not receipt.get("budget_is_frozen"):
        raise RuntimeError(
            f"receipt stage is {receipt.get('stage')!r} with budget_is_frozen="
            f"{receipt.get('budget_is_frozen')!r}; only a frozen-budget confirmatory run "
            "may produce a verdict")
    ledger = root / "INVALID_RUN_LEDGER.json"
    if ledger.exists() and json.loads(ledger.read_text()).get("count"):
        raise RuntimeError(f"the invalid-run ledger is non-empty: {ledger}")
    if receipt.get("invalid_runs"):
        raise RuntimeError(
            f"the run receipt lists invalidated cells; no verdict: {receipt['invalid_runs']}")
    expected = len(X.CARRIER_DRAWS)
    if len(receipt["carrier_draws"]) != expected:
        raise RuntimeError(
            f"the run used {len(receipt['carrier_draws'])} carrier draws; the "
            f"preregistration fixes {expected}")
    # Amendment 2026-09-20. The draw identity was compared against the hard-coded first
    # block, so this tool could not certify the Founder-approved second block 16..31 at
    # all. It is now compared against the block THIS run's receipt records, which keeps
    # the guard's purpose (rows must not come from some other block) and strengthens it:
    # the CSVs must agree with their own receipt, whichever approved block was run. The
    # denominator is still pinned to the preregistered count checked immediately above.
    block = sorted(int(d) for d in receipt["carrier_draws"])

    cell_rows = read_csv(root / "CELL_LEVEL.csv")
    cross_rows = read_csv(root / "CROSS_EVALUATION_ROWS.csv")
    reference_rows = read_csv(root / "REFERENCE_ROWS.csv")

    writers = sorted({r["writer_type"] for r in cell_rows})
    if writers != sorted(getattr(__import__("nc_geometry"), "WRITER_TYPES")):
        raise RuntimeError(f"writer types {writers}; both arms are required")
    for name, rows in (("CELL_LEVEL", cell_rows), ("CROSS_EVALUATION", cross_rows),
                       ("REFERENCE", reference_rows)):
        for writer in writers:
            found = {int(r["draw_index"]) for r in rows if r["writer_type"] == writer}
            if found != set(block):
                raise RuntimeError(
                    f"{name} carries draws {sorted(found)} for {writer}; the run receipt "
                    f"records exactly {block}")

    verdict: dict[str, Any] = {
        "protocol": "NC3C_TASK_SPECIFIC_GEOMETRY_PROTOCOL_v1_20260918_EN.md",
        "seed_namespace": X.SEED_NAMESPACE,
        "carrier_block": block,
        "objectives": list(X.OBJECTIVES),
        "gates": GATES,
        "amplitude": receipt["config"]["amplitude"],
        "train_horizon": receipt["config"]["train_horizon"],
        "by_writer_type": {},
    }
    pvalues: dict[str, float] = {}
    for writer in writers:
        block: dict[str, Any] = {"own_objective": {}, "contrast_recovery": {}, "calibration": {}}
        for j in X.OBJECTIVES:
            block["own_objective"][str(j)] = own_objective_family(cell_rows, writer, j)
            block["contrast_recovery"][str(j)] = contrast_family(cell_rows, writer, j)
            block["calibration"][str(j)] = calibration_family(
                reference_rows, cross_rows, writer, j)
            own = np.array(block["own_objective"][str(j)]["by_carrier"])
            pvalues[f"own_t{j}|{writer}"] = S.paired_permutation_test(
                own, np.full_like(own, GATES["own_objective"]["median_min"]),
                seed=BOOTSTRAP_SEED, resamples=RESAMPLES)["p_value"]
            contrast_block = block["contrast_recovery"][str(j)]
            if contrast_block["verdict"] == "EVALUATED":
                contrast = np.array(contrast_block["by_carrier"])
                pvalues[f"contrast_t{j}|{writer}"] = S.paired_permutation_test(
                    contrast, np.full_like(contrast, GATES["contrast_recovery"]["median_min"]),
                    seed=BOOTSTRAP_SEED, resamples=RESAMPLES)["p_value"]
        block["diagonal_dominance"] = diagonal_dominance_family(cell_rows, writer)
        block["pass"] = all([
            all(block["own_objective"][str(j)]["pass"] for j in X.OBJECTIVES),
            all(block["contrast_recovery"][str(j)]["pass"] for j in X.OBJECTIVES),
            all(block["calibration"][str(j)]["pass"] for j in X.OBJECTIVES),
            block["diagonal_dominance"]["pass"],
        ])
        block["kill_triggered"] = any(
            block["own_objective"][str(j)]["kill_triggered"] for j in X.OBJECTIVES)
        verdict["by_writer_type"][writer] = block

    verdict["holm"] = S.holm_correction(pvalues)
    verdict["own_objective_pass"] = all(
        verdict["by_writer_type"][w]["own_objective"][str(j)]["pass"]
        for w in writers for j in X.OBJECTIVES)
    verdict["diagonal_dominance_pass"] = all(
        verdict["by_writer_type"][w]["diagonal_dominance"]["pass"] for w in writers)
    verdict["contrast_recovery_pass"] = all(
        verdict["by_writer_type"][w]["contrast_recovery"][str(j)]["pass"]
        for w in writers for j in X.OBJECTIVES)
    verdict["calibration_pass"] = all(
        verdict["by_writer_type"][w]["calibration"][str(j)]["pass"]
        for w in writers for j in X.OBJECTIVES)
    verdict["kill_triggered"] = any(
        verdict["by_writer_type"][w]["kill_triggered"] for w in writers)
    verdict["objectives_not_discriminating"] = any(
        verdict["by_writer_type"][w]["contrast_recovery"][str(j)].get("kill_rule_4_triggered", False)
        for w in writers for j in X.OBJECTIVES)
    verdict["overall"] = (
        "OBJECTIVES_NOT_DISCRIMINATING" if verdict["objectives_not_discriminating"] else
        "KILL" if verdict["kill_triggered"] else
        "FULL_PASS" if all(verdict[k] for k in (
            "own_objective_pass", "diagonal_dominance_pass",
            "contrast_recovery_pass", "calibration_pass")) else
        "PARTIAL")

    out = Path(args.out) if args.out else root / "VERDICT.json"
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=float))
    print(json.dumps({k: verdict[k] for k in (
        "overall", "own_objective_pass", "diagonal_dominance_pass",
        "contrast_recovery_pass", "calibration_pass", "kill_triggered",
        "objectives_not_discriminating")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
