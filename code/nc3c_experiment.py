"""NC-3C cell driver: two objectives, two writer types, cross-evaluated.

Authority: ../authority/nc3c/NC3C_TASK_SPECIFIC_GEOMETRY_PROTOCOL_v1_20260918_EN.md
           ../authority/nc3c/NC3C_TASK_SPECIFIC_GEOMETRY_PREREG_v1_20260918.yaml

A cell is one carrier draw x one writer type. Within a cell, a direction is trained
independently for each objective and then evaluated against BOTH objectives with the
carrier, coupling and store map held fixed. Everything below the objective axis is the
completed lane's machinery, unchanged and imported, so the two experiments share one
implementation of the geometry, the sampler, the trainer and the readouts.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

import nc_geometry as G
import nc_model as M
import nc_experiment as E

# Frozen by the preregistration.
SEED_NAMESPACE = "NC-3C-V1"
CARRIER_DRAWS = tuple(range(16))
OBJECTIVES = (0, 12)                    # target input times t_0
NEGLIGIBLE_SEPARATION = 0.02            # frozen before the run


def nc3c_seed(draw_index: int, stream: str) -> int:
    """Namespaced seed. A namespace that has never been used, so no NC-3C seed can
    collide with a seen one by construction rather than by inspection."""
    source = G.load_vendor()["source"]
    return int(source.seed_for(SEED_NAMESPACE, draw_index, stream))


def build_carrier_draw(draw_index: int) -> dict[str, Any]:
    """The completed lane's carrier construction, rebound to the NC-3C namespace."""
    original = G.nc_seed
    original_ns = G.SEED_NAMESPACE
    try:
        G.SEED_NAMESPACE = SEED_NAMESPACE
        G.nc_seed = nc3c_seed
        draw = G.build_carrier_draw(draw_index)
    finally:
        G.nc_seed = original
        G.SEED_NAMESPACE = original_ns
    draw["seed_namespace"] = SEED_NAMESPACE
    return draw


def objective_operators(ops: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """M_j, its spectrum and its oracle, for every declared objective."""
    out: dict[int, dict[str, Any]] = {}
    for t0 in OBJECTIVES:
        M_j = ops["M_old"] if t0 == 0 else G.target_operator(ops, t0)
        values, vectors = G.eig_descending(M_j)
        out[t0] = {
            "M": M_j,
            "lambda_max": float(values[0]),
            "eigenvalues": values,
            "oracle": vectors[:, 0].copy(),
            "L": ops["transfers"][G.WRITE_WINDOW - 1 - t0],
            "subspace": G.leading_subspace(M_j, 0.99),
        }
    return out


def run_cell(draw_index: int, writer_type: str, config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    draw = build_carrier_draw(draw_index)
    W, K, U = draw["W"][writer_type], draw["K"], draw["U"]
    ops = G.system_operators(W, K, U)
    objectives = objective_operators(ops)
    amplitude = config["amplitude"]
    train_horizon = config["train_horizon"]

    # Oracle separation available in THIS carrier's geometry, computed before any
    # training, so the contrast denominator is a property of the system and not of the
    # optimizer.
    separation: dict[int, float] = {}
    for t0 in OBJECTIVES:
        other = [t for t in OBJECTIVES if t != t0][0]
        v_other = objectives[other]["oracle"]
        separation[t0] = 1.0 - float(v_other @ objectives[t0]["M"] @ v_other) / objectives[t0]["lambda_max"]

    # ---- train one direction per objective per seed -----------------------
    trained: dict[int, list[dict[str, Any]]] = {t0: [] for t0 in OBJECTIVES}
    evaluators: dict[int, E.ConditionEvaluator] = {}
    for t0 in OBJECTIVES:
        P, covariance, _ = G.isolated_endpoint_from(
            U, objectives[t0]["L"], ops["C_ss"], train_horizon
        )
        R = M.noise_factor(covariance)
        evaluators[t0] = E.ConditionEvaluator(
            P, covariance, amplitude,
            test_seed=nc3c_seed(draw_index, f"test|{writer_type}|t{t0}"),
            calibration_seed=nc3c_seed(draw_index, f"cal|{writer_type}|t{t0}"),
            episodes=config["test_episodes"],
            calibration_episodes=config["calibration_episodes"],
        )
        for seed_index in range(config["optimizer_seeds"]):
            seed = nc3c_seed(draw_index, f"optimizer|{writer_type}|t{t0}|{seed_index}")
            result = M.train_direction(
                P, R, amplitude, seed,
                steps=config["training_steps"], batch_size=config["batch_size"],
                learning_rate=config["learning_rate"], grad_clip=config["gradient_clip"],
                loss_every=config["loss_every"],
            )
            result["seed_index"] = seed_index
            result["trained_on"] = t0
            trained[t0].append(result)

    # ---- cross-evaluation: every direction against every objective --------
    rows: list[dict[str, Any]] = []
    for trained_on in OBJECTIVES:
        for result in trained[trained_on]:
            v = result["v"]
            for evaluated_on in OBJECTIVES:
                obj = objectives[evaluated_on]
                evaluator = evaluators[evaluated_on]
                J = evaluator.fisher(v)
                accuracy, _, _ = evaluator.recalibrated(v)
                rows.append({
                    "draw_index": draw_index,
                    "writer_type": writer_type,
                    "trained_on": trained_on,
                    "evaluated_on": evaluated_on,
                    "seed_index": result["seed_index"],
                    "own_objective": bool(trained_on == evaluated_on),
                    "rayleigh_efficiency": float(v @ obj["M"] @ v) / obj["lambda_max"],
                    "subspace_alignment": float(np.sum((obj["subspace"].T @ v) ** 2)),
                    "cosine_to_oracle": float(abs(v @ obj["oracle"])),
                    "J": J,
                    "empirical_J": evaluator.empirical_fisher(v),
                    "predicted_accuracy": G.bayes_accuracy(amplitude, J),
                    "recalibrated_accuracy": accuracy,
                    "fixed_readout_accuracy": evaluator.with_readout(
                        v, result["w"], result["b"]),
                    "lambda_max": obj["lambda_max"],
                    "oracle_separation": separation[evaluated_on],
                    "clip_fraction": result["clip_fraction"],
                    "final_loss": result["final_loss"],
                })

    # ---- reference directions, for the calibration gate -------------------
    rng = np.random.default_rng(nc3c_seed(draw_index, f"random_directions|{writer_type}"))
    randoms = rng.standard_normal((G.WRITE_DIM, E.RANDOM_DIRECTION_COUNT))
    randoms /= np.linalg.norm(randoms, axis=0, keepdims=True)
    reference_rows: list[dict[str, Any]] = []
    for evaluated_on in OBJECTIVES:
        obj = objectives[evaluated_on]
        evaluator = evaluators[evaluated_on]
        named = {f"oracle_t{t0}": objectives[t0]["oracle"] for t0 in OBJECTIVES}
        named["bottom"] = G.eig_descending(obj["M"])[1][:, -1].copy()
        directions = list(named.items()) + [
            (f"random_{i:03d}", randoms[:, i]) for i in range(E.RANDOM_DIRECTION_COUNT)]
        for name, v in directions:
            J = evaluator.fisher(v)
            accuracy, _, _ = evaluator.recalibrated(v)
            reference_rows.append({
                "draw_index": draw_index, "writer_type": writer_type,
                "evaluated_on": evaluated_on, "direction": name,
                "J": J, "predicted_accuracy": G.bayes_accuracy(amplitude, J),
                "recalibrated_accuracy": accuracy,
                "rayleigh_efficiency": float(v @ obj["M"] @ v) / obj["lambda_max"],
            })

    # ---- arena-information guard, both objectives -------------------------
    # The completed lane refuses a cell whose oracle direction fails to beat the median
    # random direction, because that is a broken instrument and not a result. The same
    # guard applies here, and it applies to O_12 in particular: the protocol declares
    # that objective the harder one at the frozen amplitude.
    for evaluated_on in OBJECTIVES:
        subset = [r for r in reference_rows if r["evaluated_on"] == evaluated_on]
        oracle = next(r["recalibrated_accuracy"] for r in subset
                      if r["direction"] == f"oracle_t{evaluated_on}")
        random_median = float(np.median([r["recalibrated_accuracy"] for r in subset
                                         if r["direction"].startswith("random_")]))
        if not (oracle - random_median > 0.0):
            raise RuntimeError(
                f"draw {draw_index} {writer_type} t{evaluated_on}: oracle accuracy "
                f"{oracle:.6f} does not exceed the random-direction median "
                f"{random_median:.6f}; the arena carries no information and this cell "
                "is invalid")

    # ---- cell-level endpoints --------------------------------------------
    def median_over_seeds(trained_on: int, evaluated_on: int, key: str) -> float:
        values = [r[key] for r in rows
                  if r["trained_on"] == trained_on and r["evaluated_on"] == evaluated_on]
        return float(np.median(values))

    endpoints: dict[str, Any] = {}
    for j in OBJECTIVES:
        other = [t for t in OBJECTIVES if t != j][0]
        own = median_over_seeds(j, j, "rayleigh_efficiency")
        cross = median_over_seeds(other, j, "rayleigh_efficiency")
        star = separation[j]
        endpoints[f"R_own_t{j}"] = own
        endpoints[f"R_cross_t{j}_from_t{other}"] = cross
        endpoints[f"G_star_t{j}"] = star
        endpoints[f"G_learn_t{j}"] = own - cross
        endpoints[f"C_t{j}"] = (own - cross) / star if star > NEGLIGIBLE_SEPARATION else float("nan")
        endpoints[f"separation_negligible_t{j}"] = bool(star <= NEGLIGIBLE_SEPARATION)
        endpoints[f"diagonal_dominant_t{j}"] = bool(own > cross)
        endpoints[f"A_sub_own_t{j}"] = median_over_seeds(j, j, "subspace_alignment")
        endpoints[f"accuracy_own_t{j}"] = median_over_seeds(j, j, "recalibrated_accuracy")
        endpoints[f"lambda_max_t{j}"] = objectives[j]["lambda_max"]
        subset = [r for r in reference_rows if r["evaluated_on"] == j]
        endpoints[f"oracle_minus_random_t{j}"] = (
            next(r["recalibrated_accuracy"] for r in subset if r["direction"] == f"oracle_t{j}")
            - float(np.median([r["recalibrated_accuracy"] for r in subset
                               if r["direction"].startswith("random_")])))

    return {
        "draw_index": draw_index,
        "writer_type": writer_type,
        "seed_namespace": SEED_NAMESPACE,
        "config": config,
        "seeds": draw["seeds"],
        "hashes": draw["hashes"],
        "oracle_overlap": float(abs(
            objectives[OBJECTIVES[0]]["oracle"] @ objectives[OBJECTIVES[1]]["oracle"])),
        "endpoints": endpoints,
        "rows": rows,
        "reference_rows": reference_rows,
        "learned_vectors": {
            f"t{r['trained_on']}_s{r['seed_index']}": r["v"].tolist()
            for t0 in OBJECTIVES for r in trained[t0]},
        "elapsed_seconds": time.time() - started,
    }
