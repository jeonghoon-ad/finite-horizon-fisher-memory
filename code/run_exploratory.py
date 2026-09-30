#!/usr/bin/env python3
"""Exploratory arms — NOT preregistered, NOT decision-bearing.

The preregistered decision rules of protocol section 10 contain four items that
cannot fail for any scientific reason (see docs/IMPLEMENTATION_AUDIT.md, findings
A1, A3, A5).  Those gates are left exactly as frozen.  These three arms are added
alongside so that the lane contains at least one measurement whose outcome is not
fixed by algebra.  They carry no gate, license no manuscript claim, and are
reported in a separate file.

X1  Objective specificity at a second target time.
    Train the write direction on an input injected at t_0 = T_w/2 instead of
    t_0 = 0.  Falsifiable prediction: the trained direction aligns with
    eigmax(M_{t_0}) and NOT with eigmax(M_0), and scores ABOVE v_old on
    M_{t_0} while scoring BELOW v_old on M_0.  A learner that always returns
    the same "best memory direction" fails this.

X2  Non-invertible disposal control (protocol 5.2 item 4, "optional").
    Replace the invertible hold by a rank-deficient projection of the store.
    Prediction: information is genuinely destroyed, unlike the invertible hold
    of section 6.1, so J falls by roughly the fraction of the sensitivity that
    lies outside the retained subspace.

X3  High-resolution open-arm replicate.
    At H = 1024 and 4096 the open arm's predicted departure from chance is a
    few Monte Carlo standard errors at the preregistered 50,000 episodes.  This
    arm repeats those cells at 500,000 episodes so that "correct decay" can be
    distinguished from "signal identically zero".  The preregistered 50,000
    remains the primary reading.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nc_geometry as G        # noqa: E402
import nc_model as M           # noqa: E402
import nc_experiment as E      # noqa: E402
import run_stage as R          # noqa: E402

SECOND_TARGET_TIME = G.WRITE_WINDOW // 2      # 12
DISPOSAL_RANK = G.STORE_DIM // 2              # 8
HIGH_RESOLUTION_EPISODES = 500000


def arm_x1(draw_index: int, writer_type: str, amplitude: float, train_horizon: int,
           steps: int, batch_size: int, seeds: int) -> dict[str, Any]:
    draw = G.build_carrier_draw(draw_index)
    W, K, U = draw["W"][writer_type], draw["K"], draw["U"]
    ops = G.system_operators(W, K, U)
    M_zero = ops["M_old"]
    M_second = G.target_operator(ops, SECOND_TARGET_TIME)
    lam_zero = float(G.eig_descending(M_zero)[0][0])
    values_second, vectors_second = G.eig_descending(M_second)
    lam_second = float(values_second[0])
    v_second_oracle = vectors_second[:, 0]
    v_zero_oracle = G.eig_descending(M_zero)[1][:, 0]

    L_second = ops["transfers"][G.WRITE_WINDOW - 1 - SECOND_TARGET_TIME]
    A = np.linalg.matrix_power(U, train_horizon - G.WRITE_WINDOW)
    P = A @ L_second
    Sigma = A @ ops["C_ss"] @ A.T
    R_factor = M.noise_factor(Sigma)

    rows = []
    for seed_index in range(seeds):
        seed = G.nc_seed(draw_index, f"x1|{writer_type}|{seed_index}")
        result = M.train_direction(P, R_factor, amplitude, seed, steps, batch_size,
                                   loss_every=200)
        v = result["v"]
        rows.append({
            "draw_index": draw_index, "writer_type": writer_type,
            "seed_index": seed_index, "target_time": SECOND_TARGET_TIME,
            "R_J_on_M_second": float(v @ M_second @ v) / lam_second,
            "R_J_on_M_zero": float(v @ M_zero @ v) / lam_zero,
            "cosine_to_second_oracle": float(abs(v @ v_second_oracle)),
            "cosine_to_zero_oracle": float(abs(v @ v_zero_oracle)),
            "v_old_R_J_on_M_second": float(v_zero_oracle @ M_second @ v_zero_oracle) / lam_second,
            "second_oracle_R_J_on_M_zero": float(v_second_oracle @ M_zero @ v_second_oracle) / lam_zero,
            "oracle_overlap": float(abs(v_second_oracle @ v_zero_oracle)),
            "lambda_max_M_second": lam_second, "lambda_max_M_zero": lam_zero,
        })
    for row in rows:
        row["beats_v_old_on_its_own_objective"] = bool(
            row["R_J_on_M_second"] > row["v_old_R_J_on_M_second"])
        row["is_not_just_v_old"] = bool(row["R_J_on_M_zero"] < 0.99)
    return {"arm": "X1_second_target_time", "rows": rows}


def arm_x2(draw_index: int, writer_type: str, amplitude: float,
           horizon: int = 256) -> dict[str, Any]:
    draw = G.build_carrier_draw(draw_index)
    W, K, U = draw["W"][writer_type], draw["K"], draw["U"]
    ops = G.system_operators(W, K, U)
    controls = G.direction_controls(ops, draw_index, writer_type, random_count=2)
    v = controls["v_old"]
    P, Sigma, _ = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], horizon)
    p = P @ v
    J_invertible = float(p @ np.linalg.solve(Sigma, p))

    rng = np.random.default_rng(G.nc_seed(draw_index, f"disposal|{writer_type}"))
    basis, _ = np.linalg.qr(rng.standard_normal((G.STORE_DIM, G.STORE_DIM)))
    keep = basis[:, :DISPOSAL_RANK]
    projection = keep @ keep.T
    p_disposed = projection @ p
    Sigma_disposed = projection @ Sigma @ projection.T
    J_disposed = float(p_disposed @ np.linalg.pinv(Sigma_disposed, hermitian=True) @ p_disposed)
    retained_geometry = float(np.linalg.norm(p_disposed) ** 2 / np.linalg.norm(p) ** 2)
    return {
        "arm": "X2_noninvertible_disposal",
        "draw_index": draw_index, "writer_type": writer_type, "horizon": horizon,
        "retained_rank": DISPOSAL_RANK, "store_dim": G.STORE_DIM,
        "J_invertible_hold": J_invertible,
        "J_after_disposal": J_disposed,
        "retained_fraction": J_disposed / J_invertible if J_invertible > 0 else float("nan"),
        "sensitivity_norm_retained_fraction": retained_geometry,
        "accuracy_invertible": G.bayes_accuracy(amplitude, J_invertible),
        "accuracy_after_disposal": G.bayes_accuracy(amplitude, J_disposed),
        "information_actually_destroyed": bool(J_disposed < 0.999 * J_invertible),
    }


def arm_x3(draw_index: int, writer_type: str, amplitude: float,
           horizons: tuple[int, ...] = (1024, 4096)) -> dict[str, Any]:
    draw = G.build_carrier_draw(draw_index)
    W, K, U = draw["W"][writer_type], draw["K"], draw["U"]
    ops = G.system_operators(W, K, U)
    controls = G.direction_controls(ops, draw_index, writer_type, random_count=2)
    v = controls["v_old"]
    rows = []
    for horizon in horizons:
        P, Sigma = G.open_endpoint(W, K, U, horizon)
        evaluator = E.ConditionEvaluator(
            P, Sigma, amplitude,
            test_seed=G.nc_seed(draw_index, f"x3test|{writer_type}|{horizon}"),
            calibration_seed=G.nc_seed(draw_index, f"x3cal|{writer_type}|{horizon}"),
            episodes=HIGH_RESOLUTION_EPISODES,
            calibration_episodes=HIGH_RESOLUTION_EPISODES,
        )
        accuracy, _, _ = evaluator.recalibrated(v)
        J = evaluator.fisher(v)
        predicted = G.bayes_accuracy(amplitude, J)
        se = float(np.sqrt(0.25 / HIGH_RESOLUTION_EPISODES))
        rows.append({
            "draw_index": draw_index, "writer_type": writer_type,
            "condition": "open", "horizon": horizon,
            "episodes": HIGH_RESOLUTION_EPISODES,
            "J": J, "empirical_J": evaluator.empirical_fisher(v),
            "predicted_accuracy": predicted, "recalibrated_accuracy": accuracy,
            "absolute_error": abs(predicted - accuracy),
            "binomial_se_at_chance": se,
            "predicted_departure_from_chance_in_se": (predicted - 0.5) / se,
            "observed_departure_from_chance_in_se": (accuracy - 0.5) / se,
            "resolvable_above_chance": bool((accuracy - 0.5) / se >= 5.0),
        })
    return {"arm": "X3_high_resolution_open_arm", "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, required=True)
    parser.add_argument("--train-horizon", type=int, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--draws", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    outdir = R.RESULTS / "exploratory"
    # Deliberately NOT the confirmatory draws: running an exploratory arm must
    # never put a confirmatory carrier in front of anyone before the
    # confirmatory stage has run.
    draws = list(G.EXPLORATORY_DRAWS[:args.draws])
    overlap = set(draws) & (set(G.CONFIRMATORY_DRAWS) | set(G.PILOT_DRAWS))
    if overlap:
        raise RuntimeError(f"exploratory draws collide with reserved draws: {sorted(overlap)}")
    x1_rows, x2_rows, x3_rows = [], [], []
    for draw_index in draws:
        for writer_type in G.WRITER_TYPES:
            x1_rows.extend(arm_x1(draw_index, writer_type, args.amplitude,
                                  args.train_horizon, args.steps, args.batch_size,
                                  args.seeds)["rows"])
            x2_rows.append(arm_x2(draw_index, writer_type, args.amplitude))
            x3_rows.extend(arm_x3(draw_index, writer_type, args.amplitude)["rows"])

    R.write_csv(outdir / "X1_SECOND_TARGET_TIME.csv", x1_rows)
    R.write_csv(outdir / "X2_DISPOSAL.csv", x2_rows)
    R.write_csv(outdir / "X3_HIGH_RESOLUTION_OPEN.csv", x3_rows)
    summary = {
        "status": "EXPLORATORY_NOT_PREREGISTERED_NOT_DECISION_BEARING",
        "draws": draws,
        "draw_block": "EXPLORATORY_DRAWS, disjoint from confirmatory and pilot",
        "X1": {
            "rows": len(x1_rows),
            "beats_v_old_on_its_own_objective":
                int(sum(r["beats_v_old_on_its_own_objective"] for r in x1_rows)),
            "is_not_just_v_old": int(sum(r["is_not_just_v_old"] for r in x1_rows)),
            "median_R_J_on_M_second": float(np.median([r["R_J_on_M_second"] for r in x1_rows])),
            "median_R_J_on_M_zero": float(np.median([r["R_J_on_M_zero"] for r in x1_rows])),
            "median_v_old_R_J_on_M_second":
                float(np.median([r["v_old_R_J_on_M_second"] for r in x1_rows])),
            "median_oracle_overlap": float(np.median([r["oracle_overlap"] for r in x1_rows])),
        },
        "X2": {
            "rows": len(x2_rows),
            "information_destroyed_cells":
                int(sum(r["information_actually_destroyed"] for r in x2_rows)),
            "median_retained_fraction":
                float(np.median([r["retained_fraction"] for r in x2_rows])),
        },
        "X3": {
            "rows": len(x3_rows),
            "resolvable_cells": int(sum(r["resolvable_above_chance"] for r in x3_rows)),
            "max_absolute_error": float(np.max([r["absolute_error"] for r in x3_rows])),
        },
        "elapsed_seconds": time.time() - started,
        "peak_rss_bytes": R.peak_rss_bytes(),
        "code_manifest": R.code_manifest(),
        "environment": R.environment(),
    }
    R.write_json(outdir / "EXPLORATORY_RECEIPT.json", summary)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("code_manifest", "environment")}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
