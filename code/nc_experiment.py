"""Cell driver: train the write direction, then run every evaluation-only arm.

Authority: ../authority/NEURAL_COMPUTATION_TRAINED_MEMORY_VALIDATION_PROTOCOL_v1_20260918_EN.md
           ../authority/NEURAL_COMPUTATION_TRAINED_MEMORY_VALIDATION_PREREG_v1_20260918.yaml

A "cell" is one carrier draw crossed with one writer type.  The confirmatory
matrix is 16 draws x 2 writer types x 5 optimizer seeds = 160 trained models.

Shared-noise discipline (protocol section 4.8, validity control 7): within one
(cell, storage condition, horizon) every direction is scored on the SAME noise
realizations and the SAME class labels.  Because the endpoint is
x = y a P v + R eps, the noise term R eps does not depend on the direction, so
it is drawn and propagated once and every direction is a rank-one shift of it.
The pooled within-class scatter used by the recalibrated readout is likewise
direction-independent and is computed once per condition.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

import nc_geometry as G
import nc_model as M

TEST_HORIZONS = (64, 256, 1024, 4096)
ALPHAS = (0.05, 0.25, 1.0)
RANDOM_DIRECTION_COUNT = 128


def _class_split_means(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x[y > 0].mean(axis=0), x[y < 0].mean(axis=0)


class ConditionEvaluator:
    """Everything that is shared by all directions at one (condition, horizon)."""

    def __init__(self, P: np.ndarray, covariance: np.ndarray, amplitude: float,
                 test_seed: int, calibration_seed: int, episodes: int,
                 calibration_episodes: int,
                 noise_factor_override: np.ndarray | None = None) -> None:
        self.P = P
        self.covariance = covariance
        self.amplitude = amplitude
        # Any R with R R^T = covariance is a valid sampler.  The default is the
        # Cholesky factor.  The override exists because Cholesky is NOT
        # equivariant under orthogonal conjugation: chol(A C A^T) is not
        # A chol(C), so feeding the same standard normals through the Cholesky
        # factors of different horizons produces DIFFERENT realisations, not the
        # same closure noise rotated.  The paired-across-horizons diagnostic
        # needs the latter, and gets it by passing A chol(C_ss).
        R = M.noise_factor(covariance) if noise_factor_override is None else noise_factor_override
        test_rng = np.random.default_rng(test_seed)
        cal_rng = np.random.default_rng(calibration_seed)
        eps_test, self.y_test = M.sample_shared_noise(test_rng, R.shape[1], episodes)
        eps_cal, self.y_cal = M.sample_shared_noise(cal_rng, R.shape[1], calibration_episodes)
        self.noise_test = eps_test @ R.T
        self.noise_cal = eps_cal @ R.T
        self.sign_test = np.sign(self.y_test)
        # Direction-independent pooled within-class scatter of the calibration set.
        pos = self.noise_cal[self.y_cal > 0]
        neg = self.noise_cal[self.y_cal < 0]
        centered = np.concatenate([pos - pos.mean(axis=0), neg - neg.mean(axis=0)], axis=0)
        self.pooled = centered.T @ centered / (len(centered) - 2)
        self.noise_mean_pos = pos.mean(axis=0)
        self.noise_mean_neg = neg.mean(axis=0)
        self.episodes = episodes
        self.calibration_episodes = calibration_episodes

    def fisher(self, v: np.ndarray) -> float:
        p = self.P @ v
        return float(p @ np.linalg.solve(self.covariance, p))

    def empirical_fisher(self, v: np.ndarray) -> float:
        """J estimated from SAMPLED calibration states only.

        The analytic J is a property of the covariance the experimenter built.
        This one is a property of the states the sampler actually produced, so
        a declared disturbance that the sampler failed to realise shows up here
        and nowhere else.
        """
        signal = self.amplitude * (self.P @ v)
        delta = ((self.noise_mean_pos + signal) - (self.noise_mean_neg - signal)) / (2.0 * self.amplitude)
        return float(delta @ np.linalg.solve(self.pooled, delta))

    def predicted_accuracy(self, v: np.ndarray) -> float:
        return G.bayes_accuracy(self.amplitude, self.fisher(v))

    def recalibrated(self, v: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Refit the linear readout for this direction from calibration data only."""
        signal = self.amplitude * (self.P @ v)
        mu_pos = self.noise_mean_pos + signal
        mu_neg = self.noise_mean_neg - signal
        w = np.linalg.solve(self.pooled, mu_pos - mu_neg)
        b = -float(w @ (mu_pos + mu_neg)) / 2.0
        logit = (self.noise_test + self.y_test[:, None] * signal[None, :]) @ w + b
        acc = float(np.mean(np.sign(logit) == self.sign_test))
        return acc, w, b

    def with_readout(self, v: np.ndarray, w: np.ndarray, b: float) -> float:
        """Score a direction with a readout that is NOT refit (deployment diagnostic)."""
        signal = self.amplitude * (self.P @ v)
        logit = (self.noise_test + self.y_test[:, None] * signal[None, :]) @ w + b
        return float(np.mean(np.sign(logit) == self.sign_test))


def build_conditions(draw: dict[str, Any], ops: dict[str, Any], writer_type: str,
                     config: dict[str, Any]) -> dict[tuple[str, int], ConditionEvaluator]:
    """Isolated and open evaluators at every preregistered test horizon."""
    draw_index = draw["draw_index"]
    W, K, U = draw["W"][writer_type], draw["K"], draw["U"]
    out: dict[tuple[str, int], ConditionEvaluator] = {}
    # The training horizon is always evaluated, so that the primary endpoints
    # E3/E1 are read in exactly the condition the optimizer saw.
    horizons = sorted(set(config["test_horizons"]) | {config["train_horizon"]})
    for horizon in horizons:
        for condition in ("isolated", "open"):
            if condition == "isolated":
                P, covariance, _ = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], horizon)
            else:
                P, covariance = G.open_endpoint(W, K, U, horizon)
            # Protocol 5.2 requires IDENTICAL noise seeds across storage
            # conditions, so the tag deliberately excludes `condition`: the
            # isolated and open arms at one horizon see the same standard
            # normal draws and the same labels, and differ only in the
            # covariance factor they are pushed through.  Seeds DO vary with
            # horizon, which keeps the across-horizon range in decision rule
            # 10.3 item 1 a real Monte Carlo check instead of a bitwise
            # identity.
            tag = f"{writer_type}|{horizon}"
            out[(condition, horizon)] = ConditionEvaluator(
                P, covariance, config["amplitude"],
                test_seed=G.nc_seed(draw_index, f"test|{tag}"),
                calibration_seed=G.nc_seed(draw_index, f"cal|{tag}"),
                episodes=config["test_episodes"],
                calibration_episodes=config["calibration_episodes"],
            )
    # A second isolated evaluator whose seed does NOT depend on the horizon, so
    # the same standard normals are reused at every horizon.  Under exact
    # isolation with an invertible hold, J is horizon-invariant and the
    # covariance-aware discriminant is GL(d)-equivariant, so this reading is an
    # algebraic identity and its across-horizon range is exactly zero.  It is a
    # DIAGNOSTIC, not a gate: it separates "the preregistered unpaired range gate
    # tripped on Monte Carlo noise" from "the horizon propagation is wrong",
    # which the gate alone cannot distinguish.
    closure_factor = M.noise_factor(ops["C_ss"])
    for horizon in horizons:
        P, covariance, A = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], horizon)
        out[("isolated_paired_across_horizons", horizon)] = ConditionEvaluator(
            P, covariance, config["amplitude"],
            test_seed=G.nc_seed(draw_index, f"test|{writer_type}|paired"),
            calibration_seed=G.nc_seed(draw_index, f"cal|{writer_type}|paired"),
            episodes=config["test_episodes"],
            calibration_episodes=config["calibration_episodes"],
            noise_factor_override=A @ closure_factor,
        )
    return out


def approximate_isolation_covariances(covariance: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """R_h constructions bounded by alpha * Sigma, matching the release's
    approximate_isolation_checks: an aligned equality case and a random PSD case."""
    rng = np.random.default_rng(seed)
    root = G.load_vendor()["strengthening"].sqrt_psd(covariance)
    q, _ = np.linalg.qr(rng.standard_normal(covariance.shape))
    eigenvalues = rng.uniform(0.0, 1.0, size=covariance.shape[0])
    H = q @ np.diag(eigenvalues) @ q.T
    out: dict[str, np.ndarray] = {}
    for alpha in ALPHAS:
        out[f"aligned_alpha{alpha:g}"] = covariance + alpha * covariance
        out[f"random_psd_alpha{alpha:g}"] = covariance + alpha * (root @ H @ root.T)
    return out


def run_cell(draw_index: int, writer_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Train every optimizer seed for one cell, then run all evaluation-only arms."""
    started = time.time()
    draw = G.build_carrier_draw(draw_index)
    W, K, U = draw["W"][writer_type], draw["K"], draw["U"]
    ops = G.system_operators(W, K, U)
    write_oracle, write_oracle_meta = G.write_block_oracle(draw)
    controls = G.direction_controls(ops, draw_index, writer_type,
                                    RANDOM_DIRECTION_COUNT, write_oracle=write_oracle)
    M_old = ops["M_old"]
    eigenvalues = controls["M_old_eigenvalues"]
    lambda_max = float(eigenvalues[0])
    subspace = G.leading_subspace(M_old, 0.99)

    train_horizon = config["train_horizon"]
    amplitude = config["amplitude"]
    P_train, S_train, _ = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], train_horizon)
    R_train = M.noise_factor(S_train)

    # ---- training -------------------------------------------------------
    # Secondary best-validation checkpoint (protocol 4.6: secondary only).
    validation_rng = np.random.default_rng(
        G.nc_seed(draw_index, f"validation|{writer_type}")
    )
    validation_eps, validation_labels = M.sample_shared_noise(
        validation_rng, G.STORE_DIM, config["validation_episodes"]
    )
    validation = {"eps": validation_eps, "labels": validation_labels}

    trained: list[dict[str, Any]] = []
    for seed_index in range(config["optimizer_seeds"]):
        seed = G.nc_seed(draw_index, f"optimizer|{writer_type}|{seed_index}")
        result = M.train_direction(
            P_train, R_train, amplitude, seed,
            steps=config["training_steps"], batch_size=config["batch_size"],
            learning_rate=config["learning_rate"], grad_clip=config["gradient_clip"],
            loss_every=config["loss_every"],
            validation=validation, validate_every=config["validate_every"],
        )
        result["seed_index"] = seed_index
        trained.append(result)

    conditions = build_conditions(draw, ops, writer_type, config)
    primary = conditions[("isolated", train_horizon)]

    # ---- direction table -------------------------------------------------
    named = {
        "v_old": controls["v_old"],
        "v_store": controls["v_store"],
        "v_write": controls["v_write"],
        "v_bottom": controls["v_bottom"],
        "v_worst_positive": controls["v_worst_positive"],
    }
    directions: list[tuple[str, np.ndarray]] = list(named.items())
    for index in range(RANDOM_DIRECTION_COUNT):
        directions.append((f"v_random_{index:03d}", controls["v_random"][:, index]))
    for result in trained:
        directions.append((f"v_learn_s{result['seed_index']}", result["v"]))

    rows: list[dict[str, Any]] = []
    for (condition, horizon), evaluator in sorted(conditions.items()):
        if condition == "isolated_paired_across_horizons":
            # Retention diagnostic only.  It reuses ONE noise realisation at
            # every horizon, so letting it into DIRECTION_ROWS would put a
            # five-fold duplication of a single realisation into the gated
            # calibration population.
            continue
        for name, v in directions:
            J = evaluator.fisher(v)
            acc_recal, w_recal, b_recal = evaluator.recalibrated(v)
            row = {
                "draw_index": draw_index,
                "writer_type": writer_type,
                "condition": condition,
                "horizon": horizon,
                "direction": name,
                "J": J,
                "predicted_accuracy": G.bayes_accuracy(amplitude, J),
                "recalibrated_accuracy": acc_recal,
                "empirical_J": evaluator.empirical_fisher(v),
                "rayleigh_efficiency_M_old": float(v @ M_old @ v) / lambda_max,
                "subspace_alignment": float(np.sum((subspace.T @ v) ** 2)),
                "cosine_to_v_old": float(abs(v @ controls["v_old"])),
            }
            rows.append(row)

    # ---- causal direction swap, fixed learned readout ---------------------
    swap_rows: list[dict[str, Any]] = []
    for result in trained:
        for name, v in directions:
            if name.startswith("v_learn_") and name != f"v_learn_s{result['seed_index']}":
                continue
            fixed = primary.with_readout(v, result["w"], result["b"])
            recal, _, _ = primary.recalibrated(v)
            swap_rows.append({
                "draw_index": draw_index,
                "writer_type": writer_type,
                "seed_index": result["seed_index"],
                "direction": name,
                "J": primary.fisher(v),
                "fixed_readout_accuracy": fixed,
                "recalibrated_accuracy": recal,
                "readout_gap": recal - fixed,
            })

    # ---- NC-2 retention on the learned direction --------------------------
    retention_rows: list[dict[str, Any]] = []
    # The training horizon is included, because the fixed decoder's accuracy AT
    # the horizon it was fitted for is the positive control for kill rule 10.4
    # item 4: without it, "the fixed decoder fails away from H_train" could just
    # mean the decoder never worked.
    retention_horizons = sorted(set(config["test_horizons"]) | {train_horizon})
    for result in trained:
        v = result["v"]
        for horizon in retention_horizons:
            for condition in ("isolated", "open", "isolated_paired_across_horizons"):
                evaluator = conditions[(condition, horizon)]
                acc, _, _ = evaluator.recalibrated(v)
                retention_rows.append({
                    "draw_index": draw_index, "writer_type": writer_type,
                    "seed_index": result["seed_index"], "horizon": horizon,
                    "condition": condition, "disturbance": "none", "alpha": 0.0,
                    "J": evaluator.fisher(v),
                    "empirical_J": evaluator.empirical_fisher(v),
                    "predicted_accuracy": G.bayes_accuracy(amplitude, evaluator.fisher(v)),
                    "recalibrated_accuracy": acc,
                    "fixed_readout_accuracy": evaluator.with_readout(v, result["w"], result["b"]),
                })
            # approximate isolation is a perturbation of the ISOLATED covariance
            isolated = conditions[("isolated", horizon)]
            perturbed = approximate_isolation_covariances(
                isolated.covariance,
                G.nc_seed(draw_index, f"approx|{writer_type}|{horizon}"),
            )
            for label, covariance in perturbed.items():
                alpha = float(label.split("alpha")[1])
                # Protocol 5.2: identical writer, direction, input and NOISE
                # SEEDS across storage conditions.  The same standard-normal
                # draws are reused and only the covariance factor changes.
                evaluator = ConditionEvaluator(
                    isolated.P, covariance, amplitude,
                    test_seed=G.nc_seed(draw_index, f"test|{writer_type}|{horizon}"),
                    calibration_seed=G.nc_seed(draw_index, f"cal|{writer_type}|{horizon}"),
                    episodes=config["test_episodes"],
                    calibration_episodes=config["calibration_episodes"],
                )
                acc, _, _ = evaluator.recalibrated(v)
                J0 = isolated.fisher(v)
                J = evaluator.fisher(v)
                retention_rows.append({
                    "draw_index": draw_index, "writer_type": writer_type,
                    "seed_index": result["seed_index"], "horizon": horizon,
                    "condition": "approximate_isolation",
                    "disturbance": label.split("_alpha")[0], "alpha": alpha,
                    "J": J, "J0": J0, "retained_fraction": J / J0 if J0 > 0 else float("nan"),
                    "empirical_J": evaluator.empirical_fisher(v),
                    "empirical_J0": isolated.empirical_fisher(v),
                    "empirical_retained_fraction": (
                        evaluator.empirical_fisher(v) / isolated.empirical_fisher(v)
                        if isolated.empirical_fisher(v) > 0 else float("nan")),
                    "guaranteed_floor": 1.0 / (1.0 + alpha),
                    "bound_margin": (J / J0 - 1.0 / (1.0 + alpha)) if J0 > 0 else float("nan"),
                    "predicted_accuracy": G.bayes_accuracy(amplitude, J),
                    "recalibrated_accuracy": acc,
                    "fixed_readout_accuracy": evaluator.with_readout(v, result["w"], result["b"]),
                })

    # ---- cell-level endpoints --------------------------------------------
    primary_rows = [r for r in rows
                    if r["condition"] == "isolated" and r["horizon"] == train_horizon]
    random_rows = [r for r in primary_rows if r["direction"].startswith("v_random_")]
    if len(random_rows) != RANDOM_DIRECTION_COUNT:
        raise RuntimeError(
            f"expected {RANDOM_DIRECTION_COUNT} random directions, found {len(random_rows)}")
    random_acc = np.median([r["recalibrated_accuracy"] for r in random_rows])
    oracle_row = next(r for r in primary_rows if r["direction"] == "v_old")
    oracle_acc = oracle_row["recalibrated_accuracy"]
    # Analytic twins of the two selection quantities.  Noise-free, and exactly
    # horizon-invariant under isolation, so the pilot's tie-break is decidable.
    analytic_oracle = oracle_row["predicted_accuracy"]
    analytic_random = float(np.median([r["predicted_accuracy"] for r in random_rows]))
    learned = []
    for result in trained:
        name = f"v_learn_s{result['seed_index']}"
        row = next(r for r in primary_rows if r["direction"] == name)
        denominator = oracle_acc - random_acc
        if not (denominator > 0.0):
            # The oracle direction failing to beat the median random direction is
            # not a result, it is a broken instrument.  Silently emitting a NaN
            # behavioural efficiency here is how a lost measurement reaches a
            # verdict as "kill not triggered".
            raise RuntimeError(
                f"draw {draw_index} {writer_type}: oracle accuracy {oracle_acc:.6f} does "
                f"not exceed the random-direction median {random_acc:.6f}; the "
                "behavioural endpoint is unmeasurable and this cell is invalid")
        learned.append({
            "seed_index": result["seed_index"],
            "rayleigh_efficiency": row["rayleigh_efficiency_M_old"],
            "subspace_alignment": row["subspace_alignment"],
            "cosine_to_v_old": row["cosine_to_v_old"],
            "accuracy": row["recalibrated_accuracy"],
            "behavioral_efficiency": (
                (row["recalibrated_accuracy"] - random_acc) / denominator
                if denominator > 0 else float("nan")
            ),
            "final_loss": result["final_loss"],
            "clip_events": result["clip_events"],
            "clip_fraction": result["clip_fraction"],
            "grad_norm_median": result["grad_norm_median"],
            "grad_norm_max": result["grad_norm_max"],
            "best_validation_step": (
                result["best_validation"]["step"] if result["best_validation"] else None
            ),
            "best_validation_accuracy": (
                result["best_validation"]["accuracy"] if result["best_validation"] else None
            ),
            "best_validation_rayleigh": (
                float(result["best_validation"]["v"] @ M_old @ result["best_validation"]["v"])
                / lambda_max if result["best_validation"] else None
            ),
        })

    return {
        "draw_index": draw_index,
        "writer_type": writer_type,
        "config": config,
        "seeds": draw["seeds"],
        "hashes": draw["hashes"],
        "lambda_max_M_old": lambda_max,
        "M_old_rank": controls["M_old_rank"],
        "write_block_oracle": write_oracle_meta,
        "M_old_eigenvalues": [float(value) for value in eigenvalues],
        "eigengap_ratio": float(eigenvalues[1] / eigenvalues[0]),
        "leading_subspace_dim": int(subspace.shape[1]),
        "oracle_accuracy": float(oracle_acc),
        "random_accuracy_median": float(random_acc),
        "analytic_oracle_accuracy": float(analytic_oracle),
        "analytic_random_accuracy_median": float(analytic_random),
        "learned": learned,
        "rows": rows,
        "swap_rows": swap_rows,
        "retention_rows": retention_rows,
        "loss_curves": {f"s{r['seed_index']}": r["loss_curve"] for r in trained},
        "learned_vectors": {f"s{r['seed_index']}": r["v"].tolist() for r in trained},
        "learned_readouts": {
            f"s{r['seed_index']}": {"w": r["w"].tolist(), "b": float(r["b"])} for r in trained
        },
        "elapsed_seconds": time.time() - started,
    }
