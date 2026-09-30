#!/usr/bin/env python3
"""Apply the frozen decision rules of protocol section 10 to confirmatory artifacts.

Reads only the CSV files written by run_stage.py confirmatory.  Computes nothing
that is not preregistered, and never reads a raw cell object, so that an
independent verifier can run it against the same CSVs.

Family structure (declared amendment A5, see docs/PROTOCOL_AMENDMENTS_v1.md):
every gate is evaluated SEPARATELY for the normal and the non-normal writer.
The overall verdict requires both, which is the conservative reading of
"at least 14 of 16 carrier draws".
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import nc_geometry as G
import nc_stats as S

BOOTSTRAP_SEED = 20260918
RESAMPLES = 10000

# Kill rule 10.4 item 4 gives no numeric threshold.  Declared in amendment A29:
# a carrier-level median paired gap of more than this counts as a fixed-decoder
# failure.  Applied to a difference of accuracies, the same scale as the
# calibration gate, but a separate and separately declared quantity.
READOUT_MISMATCH_THRESHOLD = 0.02

# Two-sided tolerance on the empirical retained fraction (amendment A28, revised
# A30).  Fixed before the confirmatory run; the realised confirmatory spread is
# recomputed and reported afterwards rather than used to move it.
EMPIRICAL_ISOLATION_TOLERANCE = 0.02

GATES = {
    "alignment": {
        "median_rayleigh_efficiency_min": 0.90,
        "bootstrap_lower_bound_min": 0.80,
        "carrier_draws_at_or_above_0.80_min": 14,
        "median_behavioral_efficiency_min": 0.80,
        "accuracy_calibration_mae_max": 0.02,
    },
    "causal_geometry": {
        "spearman_J_accuracy_min": 0.95,
        "ordered_pair_agreement_min": 0.90,
        "bottom_below_random_draws_min": 14,
    },
    "retention": {
        "isolated_accuracy_range_max": 0.01,
        "accuracy_calibration_mae_max": 0.02,
        "open_isolated_order_agreement_draws_min": 15,
    },
    "kill": {"rayleigh_efficiency_below": 0.75, "behavioral_efficiency_below": 0.60},
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


def _by_carrier(values: dict[int, list[float]], draws: list[int]) -> np.ndarray:
    return np.array([float(np.median(values[d])) for d in draws])


def alignment_family(cell_rows: list[dict[str, Any]], direction_rows: list[dict[str, Any]],
                     writer_type: str, train_horizon: int) -> dict[str, Any]:
    cells = [r for r in cell_rows if r["writer_type"] == writer_type]
    draws = sorted({int(r["draw_index"]) for r in cells})

    rayleigh: dict[int, list[float]] = defaultdict(list)
    behavioral: dict[int, list[float]] = defaultdict(list)
    subspace: dict[int, list[float]] = defaultdict(list)
    for row in cells:
        rayleigh[int(row["draw_index"])].append(float(row["rayleigh_efficiency"]))
        behavioral[int(row["draw_index"])].append(float(row["behavioral_efficiency"]))
        subspace[int(row["draw_index"])].append(float(row["subspace_alignment"]))

    rj = _by_carrier(rayleigh, draws)
    rb = _by_carrier(behavioral, draws)
    asub = _by_carrier(subspace, draws)

    calibration = [r for r in direction_rows if r["writer_type"] == writer_type]
    mae = float(np.mean([abs(float(r["predicted_accuracy"]) - float(r["recalibrated_accuracy"]))
                         for r in calibration]))
    # Protocol 4.7 E4 requires the calibration slope to be REPORTED; section 10
    # gates only the MAE.  Reported here, deliberately not gated.
    predicted = np.array([float(r["predicted_accuracy"]) for r in calibration])
    observed = np.array([float(r["recalibrated_accuracy"]) for r in calibration])
    design = np.column_stack([predicted, np.ones_like(predicted)])
    slope, intercept = np.linalg.lstsq(design, observed, rcond=None)[0]
    residual = observed - design @ np.array([slope, intercept])
    slope_se = float(np.sqrt(
        float(residual @ residual) / max(len(observed) - 2, 1)
        * np.linalg.inv(design.T @ design)[0, 0]))

    boot = S.paired_bootstrap(rj, seed=BOOTSTRAP_SEED, resamples=RESAMPLES)
    lower = S.bootstrap_lower_bound(rj, seed=BOOTSTRAP_SEED, resamples=RESAMPLES)

    # Preregistered secondary test: the learned direction against the random
    # baseline, paired within carrier draw.
    random_rj: dict[int, list[float]] = defaultdict(list)
    for row in direction_rows:
        if (row["writer_type"] == writer_type and row["condition"] == "isolated"
                and int(float(row["horizon"])) == train_horizon
                and str(row["direction"]).startswith("v_random_")):
            random_rj[int(row["draw_index"])].append(float(row["rayleigh_efficiency_M_old"]))
    baseline = _by_carrier(random_rj, draws)
    permutation = S.paired_permutation_test(rj, baseline, seed=BOOTSTRAP_SEED, resamples=RESAMPLES)

    checks = [
        S.evaluate_gate("median_rayleigh_efficiency", float(np.median(rj)), ">=",
                        GATES["alignment"]["median_rayleigh_efficiency_min"]),
        S.evaluate_gate("bootstrap_lower_bound_rayleigh", lower, ">=",
                        GATES["alignment"]["bootstrap_lower_bound_min"]),
        S.evaluate_gate("carriers_at_or_above_0.80", float(np.sum(rj >= 0.80)), ">=",
                        GATES["alignment"]["carrier_draws_at_or_above_0.80_min"]),
        S.evaluate_gate("median_behavioral_efficiency", float(np.median(rb)), ">=",
                        GATES["alignment"]["median_behavioral_efficiency_min"]),
        S.evaluate_gate("accuracy_calibration_mae", mae, "<=",
                        GATES["alignment"]["accuracy_calibration_mae_max"]),
    ]
    return {
        "writer_type": writer_type,
        "carrier_draws": draws,
        "rayleigh_by_carrier": rj.tolist(),
        "behavioral_by_carrier": rb.tolist(),
        "subspace_by_carrier": asub.tolist(),
        "random_baseline_rayleigh_by_carrier": baseline.tolist(),
        "rayleigh_summary": S.summarize(rj),
        "behavioral_summary": S.summarize(rb),
        "subspace_summary": S.summarize(asub),
        "bootstrap": boot,
        "bootstrap_one_sided_lower": lower,
        "calibration_mae": mae,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_slope_se": slope_se,
        "calibration_slope_ci95": [float(slope - 1.96 * slope_se), float(slope + 1.96 * slope_se)],
        "calibration_rows": len(calibration),
        "permutation_vs_random": permutation,
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
        "kill_triggered": bool(
            float(np.median(rj)) < GATES["kill"]["rayleigh_efficiency_below"]
            or float(np.median(rb)) < GATES["kill"]["behavioral_efficiency_below"]
        ),
    }


def causal_geometry_family(direction_rows: list[dict[str, Any]], writer_type: str,
                           train_horizon: int) -> dict[str, Any]:
    rows = [r for r in direction_rows
            if r["writer_type"] == writer_type and r["condition"] == "isolated"
            and int(float(r["horizon"])) == train_horizon]
    draws = sorted({int(r["draw_index"]) for r in rows})

    spearmans: list[float] = []
    agreements: list[float] = []
    bottom_below_random = 0
    per_draw: list[dict[str, Any]] = []
    for draw in draws:
        subset = [r for r in rows if int(r["draw_index"]) == draw]
        J = np.array([float(r["J"]) for r in subset])
        acc = np.array([float(r["recalibrated_accuracy"]) for r in subset])
        rho = S.spearman(J, acc)
        agreement = S.ordered_pair_agreement(J, acc, relative_gap=0.05)
        bottom = next(float(r["recalibrated_accuracy"]) for r in subset
                      if r["direction"] == "v_bottom")
        random_median = float(np.median([float(r["recalibrated_accuracy"]) for r in subset
                                         if str(r["direction"]).startswith("v_random_")]))
        below = bool(bottom < random_median)
        bottom_below_random += int(below)
        spearmans.append(rho)
        agreements.append(agreement["fraction"])
        per_draw.append({"draw_index": draw, "spearman": rho,
                         "ordered_pair_fraction": agreement["fraction"],
                         "pairs_considered": agreement["n_pairs_considered"],
                         "bottom_accuracy": bottom, "random_median_accuracy": random_median,
                         "bottom_below_random": below, "directions": len(subset)})

    checks = [
        S.evaluate_gate("median_spearman_J_accuracy", float(np.median(spearmans)), ">=",
                        GATES["causal_geometry"]["spearman_J_accuracy_min"]),
        S.evaluate_gate("median_ordered_pair_agreement", float(np.median(agreements)), ">=",
                        GATES["causal_geometry"]["ordered_pair_agreement_min"]),
        S.evaluate_gate("bottom_below_random_draws", float(bottom_below_random), ">=",
                        GATES["causal_geometry"]["bottom_below_random_draws_min"]),
    ]
    return {
        "writer_type": writer_type,
        "per_draw": per_draw,
        "spearman_summary": S.summarize(np.array(spearmans)),
        "ordered_pair_summary": S.summarize(np.array(agreements)),
        "bottom_below_random_draws": bottom_below_random,
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def retention_family(retention_rows: list[dict[str, Any]], writer_type: str) -> dict[str, Any]:
    rows = [r for r in retention_rows if r["writer_type"] == writer_type]
    draws = sorted({int(r["draw_index"]) for r in rows})

    isolated_ranges: list[float] = []
    approximate_rows = 0
    order_agreement = 0
    bound_violations: list[dict[str, Any]] = []
    per_draw: list[dict[str, Any]] = []
    for draw in draws:
        subset = [r for r in rows if int(r["draw_index"]) == draw]
        seeds = sorted({int(r["seed_index"]) for r in subset})
        ranges = []
        paired_ranges: list[float] = []
        agree_seed = []
        for seed in seeds:
            iso = [r for r in subset if int(r["seed_index"]) == seed and r["condition"] == "isolated"]
            paired = [r for r in subset if int(r["seed_index"]) == seed
                      and r["condition"] == "isolated_paired_across_horizons"]
            opn = [r for r in subset if int(r["seed_index"]) == seed and r["condition"] == "open"]
            iso_acc = [float(r["recalibrated_accuracy"]) for r in iso]
            ranges.append(max(iso_acc) - min(iso_acc))
            if paired:
                paired_acc = [float(r["recalibrated_accuracy"]) for r in paired]
                paired_ranges.append(max(paired_acc) - min(paired_acc))
            by_horizon_iso = {int(r["horizon"]): r for r in iso}
            by_horizon_open = {int(r["horizon"]): r for r in opn}
            ordered = []
            for horizon in sorted(by_horizon_iso):
                a = by_horizon_iso[horizon]
                b = by_horizon_open[horizon]
                predicted = float(a["J"]) - float(b["J"])
                observed = float(a["recalibrated_accuracy"]) - float(b["recalibrated_accuracy"])
                ordered.append(np.sign(predicted) == np.sign(observed) or abs(predicted) < 1e-12)
            agree_seed.append(all(ordered))
        draw_range = float(np.median(ranges))
        draw_agree = bool(np.median([1.0 if a else 0.0 for a in agree_seed]) >= 0.5)
        isolated_ranges.append(draw_range)
        order_agreement += int(draw_agree)
        for row in subset:
            if row["condition"] != "approximate_isolation" or row.get("bound_margin") is None:
                continue
            approximate_rows += 1
            # The ANALYTIC margin of the aligned family is zero by construction:
            # R = alpha*Sigma attains the bound exactly, so that half of the
            # check cannot fail.  The EMPIRICAL retained fraction, estimated
            # from the states the sampler actually produced, can.  Both are
            # recorded; the empirical one is what is gated.
            analytic_margin = float(row["bound_margin"])
            analytic_ratio = float(row["retained_fraction"])
            empirical = row.get("empirical_retained_fraction")
            floor = float(row["guaranteed_floor"])
            record = {k: row.get(k) for k in
                      ("draw_index", "seed_index", "horizon", "alpha", "disturbance",
                       "retained_fraction", "empirical_retained_fraction",
                       "guaranteed_floor", "bound_margin")}
            record["aligned_family_is_equality_case"] = bool(
                str(row.get("disturbance")) == "aligned")
            # The gate is TWO-SIDED.  A one-sided check on the floor cannot catch
            # a sampler that applied no disturbance at all: that produces a
            # retained fraction of 1.0, which is above every floor, and is
            # precisely the failure this gate exists to catch.  So the empirical
            # ratio must also track its analytic twin from above as well as below.
            if empirical is None or not np.isfinite(float(empirical)):
                record["reason"] = "empirical_retained_fraction missing or non-finite"
                bound_violations.append(record)
                continue
            empirical = float(empirical)
            empirical_margin = empirical - floor
            deviation = empirical - analytic_ratio
            record["empirical_bound_margin"] = empirical_margin
            record["empirical_minus_analytic"] = deviation
            if analytic_margin < -1e-9:
                record["reason"] = "analytic retained fraction below its guaranteed floor"
                bound_violations.append(record)
            elif empirical_margin < -EMPIRICAL_ISOLATION_TOLERANCE:
                record["reason"] = "empirical retained fraction below its guaranteed floor"
                bound_violations.append(record)
            elif abs(deviation) > EMPIRICAL_ISOLATION_TOLERANCE:
                record["reason"] = (
                    "empirical retained fraction departs from its analytic value; the "
                    "declared disturbance may not have been realised by the sampler")
                bound_violations.append(record)
        per_draw.append({"draw_index": draw, "isolated_accuracy_range": draw_range,
                         "isolated_accuracy_range_paired_diagnostic": (
                             float(np.median(paired_ranges)) if paired_ranges else float("nan")),
                         "open_isolated_order_agrees": draw_agree})

    mae = float(np.mean([abs(float(r["predicted_accuracy"]) - float(r["recalibrated_accuracy"]))
                         for r in rows if r.get("recalibrated_accuracy") is not None
                         and r["condition"] != "isolated_paired_across_horizons"]))
    paired_all = [d["isolated_accuracy_range_paired_diagnostic"] for d in per_draw]
    paired_all = [x for x in paired_all if np.isfinite(x)]
    checks = [
        S.evaluate_gate("max_isolated_accuracy_range", float(np.max(isolated_ranges)), "<=",
                        GATES["retention"]["isolated_accuracy_range_max"]),
        S.evaluate_gate("retention_calibration_mae", mae, "<=",
                        GATES["retention"]["accuracy_calibration_mae_max"]),
        S.evaluate_gate("open_isolated_order_agreement_draws", float(order_agreement), ">=",
                        GATES["retention"]["open_isolated_order_agreement_draws_min"]),
        S.evaluate_gate("approximate_isolation_bound_violations",
                        float(len(bound_violations)), "<=", 0.0),
        S.evaluate_gate("approximate_isolation_rows_checked",
                        float(approximate_rows), ">=", 1.0),
    ]
    return {
        "writer_type": writer_type,
        "per_draw": per_draw,
        "isolated_range_summary": S.summarize(np.array(isolated_ranges)),
        "isolated_range_paired_diagnostic": {
            "max": float(np.max(paired_all)) if paired_all else float("nan"),
            "median": float(np.median(paired_all)) if paired_all else float("nan"),
            "draws": len(paired_all),
            "meaning": (
                "same standard normals reused at every horizon; under exact isolation "
                "this range is an algebraic identity and should be 0. Not a gate. If "
                "the gated unpaired range exceeds 0.01 while this is 0, the gate tripped "
                "on Monte Carlo noise, not on a propagation error."),
        },
        "open_isolated_order_agreement_draws": order_agreement,
        "retention_calibration_mae": mae,
        "approximate_isolation_bound_violations": bound_violations,
        "approximate_isolation_rows_checked": approximate_rows,
        "empirical_isolation_tolerance": EMPIRICAL_ISOLATION_TOLERANCE,
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }



def readout_mismatch(swap_rows: list[dict[str, Any]], retention_rows: list[dict[str, Any]],
                     writer_type: str, train_horizon: int,
                     covariance_aware_passed: bool,
                     threshold: float = READOUT_MISMATCH_THRESHOLD) -> dict[str, Any]:
    """Kill rule 10.4 item 4, measured as HORIZON staleness on the learned direction.

    "If only a fixed decoder fails while covariance-aware readout passes,
    classify the result as READOUT_MISMATCH, not memory loss."

    Two earlier readings of this rule were wrong, and both are recorded in
    amendment A29/A33 rather than quietly replaced.

    1. Reading it on the learned direction at the TRAINING horizon measures
       nothing: that is the one combination where the readout was fitted for
       exactly this direction at exactly this horizon.

    2. Reading it on DIRECTION SWAPS also measures nothing, because a linear
       readout evaluated on a direction nearly orthogonal to its own must score
       chance whatever its state of repair.  Measured here: the fixed readout's
       |accuracy - 0.5| is 0.2301 on `v_old` against 0.2305 on the learned
       direction, i.e. it retains the SAME information; and of 160 `v_old`
       seed-cells, 83 have fixed = 1 - recalibrated and 74 have fixed =
       recalibrated, which is the eigenvector SIGN convention, not decay.  A
       swap "gap" is therefore Phi(a sqrt J) - 0.5, the direction's information
       content, and a classifier built on it reports a mismatch on any healthy
       system and NO_MISMATCH only when nothing is stored.

    What the rule is actually about is the retention experiment of section 5.4:
    the decoder is fitted at `H_train` and the store then rotates by
    `U^(H - T_w)`.  Under exact isolation the Fisher information is invariant,
    so the covariance-aware readout is flat across horizons while a fixed
    decoder drifts out of the coordinates it was fitted in.  That is a genuine
    readout failure with no memory loss behind it, and it is what is classified
    here: the learned direction, its own readout, isolated, at horizons away
    from `H_train`.

    The swap table is still summarised, sign-corrected, as a descriptive
    diagnostic, and is explicitly not used for the classification.
    """
    isolated = [r for r in retention_rows
                if r["writer_type"] == writer_type
                and r["condition"] == "isolated"
                and r.get("fixed_readout_accuracy") is not None]
    away = [r for r in isolated if int(float(r["horizon"])) != train_horizon]
    at_train = [r for r in isolated if int(float(r["horizon"])) == train_horizon]
    if not away:
        raise RuntimeError(
            f"{writer_type}: no isolated retention rows away from the training horizon; "
            "kill rule 10.4 item 4 cannot be evaluated and must not default to NO_MISMATCH")

    by_draw: dict[int, list[float]] = defaultdict(list)
    recal_by_draw: dict[int, list[float]] = defaultdict(list)
    for row in away:
        draw = int(row["draw_index"])
        by_draw[draw].append(
            float(row["recalibrated_accuracy"]) - float(row["fixed_readout_accuracy"]))
        recal_by_draw[draw].append(float(row["recalibrated_accuracy"]))
    draws = sorted(by_draw)
    gaps = np.array([float(np.median(by_draw[d])) for d in draws])

    fixed_away = np.array([float(r["fixed_readout_accuracy"]) for r in away])
    recal_away = np.array([float(r["recalibrated_accuracy"]) for r in away])
    if not at_train:
        raise RuntimeError(
            f"{writer_type}: no isolated retention row AT the training horizon; the "
            "positive control for kill rule 10.4 item 4 is missing, so a fixed-decoder "
            "failure away from H_train cannot be distinguished from a decoder that "
            "never worked")
    fixed_at_train = np.array([float(r["fixed_readout_accuracy"]) for r in at_train])
    recal_at_train = np.array([float(r["recalibrated_accuracy"]) for r in at_train])
    # Sign-corrected: a readout that recovers the class with the opposite sign
    # has not lost the information, it has lost the label convention.
    fixed_away_signed = np.maximum(fixed_away, 1.0 - fixed_away)

    median_gap = float(np.median(gaps))
    if not np.isfinite(median_gap) or not np.isfinite(float(np.median(fixed_at_train))):
        raise RuntimeError(
            f"{writer_type}: the readout-mismatch statistic is non-finite; a lost "
            "instrument must not be reported as NO_MISMATCH")
    fixed_degrades = bool(median_gap > threshold)
    # The classification CONSUMES its own positive control.  Kill rule 10.4
    # item 4 is about a decoder that WENT stale; a decoder that never worked at
    # the horizon it was fitted for does not satisfy the rule's antecedent, and
    # labelling that READOUT_MISMATCH would attribute a training failure to
    # coordinate drift.
    positive_control = bool(
        float(np.median(recal_at_train)) - float(np.median(fixed_at_train)) <= threshold)
    if not covariance_aware_passed:
        classification = "COVARIANCE_AWARE_ALSO_FAILS_NOT_A_READOUT_QUESTION"
    elif not positive_control:
        classification = "FIXED_DECODER_NEVER_WORKED"
    elif fixed_degrades:
        classification = "READOUT_MISMATCH"
    else:
        classification = "NO_MISMATCH"

    swap = [r for r in swap_rows if r["writer_type"] == writer_type]
    swap_fixed = np.array([float(r["fixed_readout_accuracy"]) for r in swap]) if swap else np.array([])
    swap_recal = np.array([float(r["recalibrated_accuracy"]) for r in swap]) if swap else np.array([])

    return {
        "writer_type": writer_type,
        "rule": "protocol 10.4 item 4",
        "measured_on": (
            "learned direction, its own readout, isolated, horizons away from H_train"),
        "classification": classification,
        "threshold_on_carrier_median_gap": threshold,
        "threshold_declared_in": "docs/PROTOCOL_AMENDMENTS_v1.md amendments A29 and A33",
        "carrier_draws": len(draws),
        "rows_away_from_train_horizon": len(away),
        "carrier_median_gap": median_gap,
        "carrier_gap_min": float(np.min(gaps)),
        "carrier_gap_max": float(np.max(gaps)),
        "fixed_median_away": float(np.median(fixed_away)),
        "fixed_median_away_sign_corrected": float(np.median(fixed_away_signed)),
        "recalibrated_median_away": float(np.median(recal_away)),
        "fixed_median_at_train_horizon": float(np.median(fixed_at_train)),
        "recalibrated_median_at_train_horizon": float(np.median(recal_at_train)),
        "positive_control_fixed_works_where_fitted": positive_control,
        "fixed_rows_below_chance_away": int(np.sum(fixed_away < 0.5)),
        "covariance_aware_alignment_passed": bool(covariance_aware_passed),
        "swap_table_descriptive_only": {
            "rows": len(swap),
            "fixed_median": float(np.median(swap_fixed)) if len(swap) else float("nan"),
            "fixed_median_sign_corrected": (
                float(np.median(np.maximum(swap_fixed, 1.0 - swap_fixed)))
                if len(swap) else float("nan")),
            "recalibrated_median": float(np.median(swap_recal)) if len(swap) else float("nan"),
            "why_not_used": (
                "a linear readout on a direction nearly orthogonal to its own must score "
                "chance regardless of its state of repair, so a swap gap measures the "
                "direction's information content and the eigenvector sign convention, "
                "not decoder staleness"),
        },
    }


def objective_specificity(direction_rows: list[dict[str, Any]], writer_type: str,
                          train_horizon: int) -> dict[str, Any]:
    """NC-3: the expected ordering is set by v^T M_old v, not by the other operators."""
    rows = [r for r in direction_rows
            if r["writer_type"] == writer_type and r["condition"] == "isolated"
            and int(float(r["horizon"])) == train_horizon]
    draws = sorted({int(r["draw_index"]) for r in rows})
    records = []
    ordered = 0
    for draw in draws:
        subset = {r["direction"]: r for r in rows if int(r["draw_index"]) == draw}
        values = {name: float(subset[name]["J"]) for name in ("v_old", "v_store", "v_write")}
        accs = {name: float(subset[name]["recalibrated_accuracy"])
                for name in ("v_old", "v_store", "v_write")}
        correct = values["v_old"] >= values["v_store"] and values["v_old"] >= values["v_write"]
        ordered += int(correct)
        records.append({"draw_index": draw, **{f"J_{k}": v for k, v in values.items()},
                        **{f"acc_{k}": v for k, v in accs.items()},
                        "v_old_dominates": bool(correct)})
    return {"writer_type": writer_type, "per_draw": records,
            "draws_with_v_old_dominant": ordered, "draws": len(draws)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.results) if args.results else Path(__file__).resolve().parents[1] / "results" / "confirmatory"
    # The analysis horizon is READ FROM THE RUN, never supplied on the command
    # line.  A mistyped horizon silently produces a complete but different
    # verdict from the same CSVs, which is the quietest way to publish a wrong
    # number in this whole pipeline.
    receipt = json.loads((root / "RUN_RECEIPT.json").read_text())
    train_horizon = int(receipt["config"]["train_horizon"])
    amplitude = float(receipt["config"]["amplitude"])
    # The denominator is the PREREGISTERED count, not whatever the receipt says.
    # A receipt declaring 20 draws would otherwise let "at least 14 of 16" pass
    # as 14 of 20, i.e. 70 percent against a stated 87.5 percent.
    expected_draws = len(G.CONFIRMATORY_DRAWS)
    receipt_draws = len(receipt["carrier_draws"])
    if receipt_draws != expected_draws:
        raise RuntimeError(
            f"the run used {receipt_draws} carrier draws but the preregistration fixes "
            f"{expected_draws}; the count gates of protocol section 10 are not "
            "meaningful on a different denominator")
    if int(receipt["parameter_source"]["proposal"]["train_horizon"]) != train_horizon:
        raise RuntimeError(
            "the receipt's config.train_horizon does not match the pilot proposal it "
            "cites; the artifact has been edited")
    if abs(float(receipt["parameter_source"]["proposal"]["amplitude"]) - amplitude) > 1e-12:
        raise RuntimeError(
            "the receipt's config.amplitude does not match the pilot proposal it cites; "
            "the artifact has been edited")

    def _draw_count(rows: list[dict[str, Any]], name: str) -> None:
        writers = sorted({r["writer_type"] for r in rows})
        if writers != sorted(G.WRITER_TYPES):
            raise RuntimeError(
                f"{name} carries writer types {writers}, expected "
                f"{sorted(G.WRITER_TYPES)}; a verdict on one arm must not be reported "
                "as if both were measured")
        expected_set = set(G.CONFIRMATORY_DRAWS)
        for writer in writers:
            found = {int(r["draw_index"]) for r in rows if r["writer_type"] == writer}
            if found != expected_set:
                raise RuntimeError(
                    f"{name} carries draws {sorted(found)} for {writer}, expected exactly "
                    f"{sorted(expected_set)}; the identity of the carriers is checked, not "
                    "only how many there are")
    if receipt.get("invalid_runs"):
        raise RuntimeError(
            "the run receipt lists invalidated cells; a verdict may not be computed "
            f"until they are resolved: {receipt['invalid_runs']}")
    cell_rows = read_csv(root / "CELL_LEVEL.csv")
    direction_rows = read_csv(root / "DIRECTION_ROWS.csv")
    retention_rows = read_csv(root / "RETENTION_ROWS.csv")
    swap_rows = read_csv(root / "SWAP_ROWS.csv")

    _draw_count(cell_rows, "CELL_LEVEL.csv")
    _draw_count(direction_rows, "DIRECTION_ROWS.csv")
    _draw_count(retention_rows, "RETENTION_ROWS.csv")
    _draw_count(swap_rows, "SWAP_ROWS.csv")

    verdict: dict[str, Any] = {
        "train_horizon": train_horizon, "amplitude": amplitude,
        "train_horizon_source": "results/confirmatory/RUN_RECEIPT.json config.train_horizon",
        "expected_carrier_draws": expected_draws,
        "gates": GATES, "by_writer_type": {}}
    pvalues: dict[str, float] = {}
    for writer_type in sorted({r["writer_type"] for r in cell_rows}):
        alignment = alignment_family(cell_rows, direction_rows, writer_type, train_horizon)
        causal = causal_geometry_family(direction_rows, writer_type, train_horizon)
        retention = retention_family(retention_rows, writer_type)
        specificity = objective_specificity(direction_rows, writer_type, train_horizon)
        mismatch = readout_mismatch(swap_rows, retention_rows, writer_type, train_horizon,
                                    alignment["pass"])
        # Protocol 9.4 forbids seed-only p-values.  The behaviour test runs on
        # the per-carrier medians already computed by alignment_family, so the
        # independent unit is the carrier draw, not the optimizer seed.
        behavioural = np.array(alignment["behavioral_by_carrier"])
        # Tested against the PREREGISTERED threshold, not against zero.  A test
        # against zero only asks "better than a random direction", which is not
        # the gate; the gate is 0.80.
        behaviour_p = S.paired_permutation_test(
            behavioural,
            np.full_like(behavioural, GATES["alignment"]["median_behavioral_efficiency_min"]),
            seed=BOOTSTRAP_SEED, resamples=RESAMPLES,
        )
        behaviour_p["null_value"] = GATES["alignment"]["median_behavioral_efficiency_min"]
        pvalues[f"alignment|{writer_type}"] = alignment["permutation_vs_random"]["p_value"]
        pvalues[f"behavior|{writer_type}"] = behaviour_p["p_value"]
        verdict["by_writer_type"][writer_type] = {
            "alignment": alignment, "causal_geometry": causal,
            "retention": retention, "objective_specificity": specificity,
            "readout_mismatch": mismatch,
            "behavior_permutation": behaviour_p,
        }

    verdict["holm"] = S.holm_correction(pvalues)
    types = list(verdict["by_writer_type"])
    verdict["nc1_alignment_pass"] = all(
        verdict["by_writer_type"][t]["alignment"]["pass"] for t in types)
    verdict["causal_geometry_pass"] = all(
        verdict["by_writer_type"][t]["causal_geometry"]["pass"] for t in types)
    verdict["retention_pass"] = all(
        verdict["by_writer_type"][t]["retention"]["pass"] for t in types)
    verdict["kill_triggered"] = any(
        verdict["by_writer_type"][t]["alignment"]["kill_triggered"] for t in types)
    verdict["readout_mismatch_classified"] = {
        t: verdict["by_writer_type"][t]["readout_mismatch"]["classification"] for t in types}
    unmeasured = {
        t: [name for name, value in (
            ("median_rayleigh_efficiency",
             float(np.median(verdict["by_writer_type"][t]["alignment"]["rayleigh_by_carrier"]))),
            ("median_behavioral_efficiency",
             float(np.median(verdict["by_writer_type"][t]["alignment"]["behavioral_by_carrier"]))),
        ) if not np.isfinite(value)]
        for t in types
    }
    verdict["unmeasured_endpoints"] = {t: v for t, v in unmeasured.items() if v}
    # A lost endpoint is not a non-kill.  `nan < 0.60` is False, so a total loss
    # of the behavioural measurement would otherwise read as "kill not
    # triggered" and pass silently into the report.
    verdict["instrument_failure"] = bool(verdict["unmeasured_endpoints"])
    verdict["overall"] = (
        "INVALID_UNMEASURED_ENDPOINT" if verdict["instrument_failure"] else
        "KILL" if verdict["kill_triggered"] else
        "FULL_PASS" if (verdict["nc1_alignment_pass"] and verdict["causal_geometry_pass"]
                        and verdict["retention_pass"]) else
        "PARTIAL"
    )

    out = Path(args.out) if args.out else root / "VERDICT.json"
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=float))
    print(json.dumps({
        "overall": verdict["overall"],
        "nc1_alignment_pass": verdict["nc1_alignment_pass"],
        "causal_geometry_pass": verdict["causal_geometry_pass"],
        "retention_pass": verdict["retention_pass"],
        "kill_triggered": verdict["kill_triggered"],
        "instrument_failure": verdict["instrument_failure"],
        "readout_mismatch": verdict["readout_mismatch_classified"],
        "train_horizon_from_receipt": train_horizon,
        "written": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
