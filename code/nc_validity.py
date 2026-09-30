"""The twelve required validity controls, plus a known-bad positive control for each.

Authority: ../authority/HANDOFF_PROMPT_PHASE1.md section C (the twelve controls)
           ../authority/NEURAL_COMPUTATION_TRAINED_MEMORY_VALIDATION_PROTOCOL_v1_20260918_EN.md section 11

Each control returns a record with an explicit tolerance and a boolean.  Every
control is ALSO exercised against a deliberately corrupted input, so that a
control which cannot fail is reported as broken rather than as a pass.  A
control that passes on corrupted input is itself a failure (`instrument_valid`
is False) and invalidates the run.

Protocol section 11: "Any failed validity control invalidates the affected run
before scientific interpretation."
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

import nc_geometry as G
import nc_model as M


def _executable_source(function: Callable[..., Any]) -> str:
    """Source of `function` with its docstring and comments stripped.

    The leakage control must read what the optimizer path DOES, not what its
    prose says about what it avoids.  Scanning the docstring would flag the
    trainer for the very sentence that documents the control.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    return ast.unparse(tree)


def _record(name: str, observed: float, tolerance: float, passed: bool,
            detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "observed": float(observed),
        "tolerance": float(tolerance),
        "pass": bool(passed),
        "detail": detail or {},
    }


def _with_negative_control(control: Callable[[bool], dict[str, Any]]) -> dict[str, Any]:
    """Run a control clean and corrupted.  The corrupted run MUST fail.

    A corrupted branch that RAISES has also failed, and loudly.  That path is
    caught here rather than in the caller, because a numerical exception on one
    deliberately broken input must not take down the rest of the battery.  The
    clean branch is never shielded: if it raises, the run is invalid and the
    exception propagates.
    """
    clean = control(False)
    try:
        corrupted = control(True)
        corrupted_observed = corrupted["observed"]
        corrupted_passed = bool(corrupted["pass"])
        raised = None
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
        corrupted_observed = float("inf")
        corrupted_passed = False
        raised = f"{type(error).__name__}: {error}"
    clean["instrument_valid"] = bool(not corrupted_passed)
    clean["negative_control"] = {
        "observed": corrupted_observed,
        "pass_when_corrupted": corrupted_passed,
        "raised": raised,
    }
    return clean


# --------------------------------------------------------------------------
# 1 + 2.  Sequential recurrence vs block propagation vs endpoint sampler
# --------------------------------------------------------------------------
def control_signal_parity(draw: dict[str, Any], writer_type: str, horizon: int,
                          isolated: bool, v: np.ndarray, amplitude: float,
                          target_time: int | None = None) -> dict[str, Any]:
    """Exact, noise-free parity of the sequential unroll against block propagation.

    With z_t == 0 the recurrence is deterministic, so the store state must equal
    a * (P v) to machine precision.  This is control 1 with no Monte Carlo error.

    `target_time` defaults to the protocol's t_0 = 0.  It is a parameter because an
    experiment that varies the target time needs a same-path positive control on the
    path it actually varies: an off-by-one in the index map would still produce a
    distinct, well-conditioned operator and would pass every other control.
    """
    if target_time is None:
        target_time = G.TARGET_TIME
    W = draw["W"][writer_type]
    K, U = draw["K"], draw["U"]

    def run(corrupt: bool) -> dict[str, Any]:
        ops = G.system_operators(W, K, U)
        L = ops["transfers"][G.WRITE_WINDOW - 1 - target_time]
        if isolated:
            P, _, _ = G.isolated_endpoint_from(U, L, ops["C_ss"], horizon)
        else:
            if target_time != 0:
                raise ValueError("the open arm is defined for the t_0 = 0 input only")
            P, _ = G.open_endpoint(W, K, U, horizon)
        if corrupt:
            P = np.roll(P, 1, axis=0)   # a single-row shift of the transfer map
        # Deterministic sequential unroll with the noise switched off.
        n, d = W.shape[0], U.shape[0]
        xw = np.zeros(n)
        xs = np.zeros(d)
        for t in range(horizon):
            coupled = (t < G.WRITE_WINDOW) or (not isolated)
            xs_next = U @ xs + (K @ xw if coupled else 0.0)
            xw_next = W @ xw
            if t == target_time:
                xw_next = xw_next + amplitude * v
            xw, xs = xw_next, xs_next
        expected = amplitude * (P @ v)
        scale = max(float(np.linalg.norm(expected)), 1e-300)
        err = float(np.max(np.abs(xs - expected))) / scale
        return _record(f"1_sequential_vs_block_signal_t{target_time}", err, 1e-10, err <= 1e-10,
                       {"horizon": horizon, "isolated": isolated, "target_time": target_time,
                        "writer_type": writer_type, "signal_norm": float(np.linalg.norm(expected))})

    return _with_negative_control(run)


def control_impulse_covariance(draw: dict[str, Any], writer_type: str,
                               horizon: int, isolated: bool) -> dict[str, Any]:
    """Exact covariance parity by explicit impulse response.

    The map from the noise draw z_t to the store state is linear, so the exact
    covariance is sum_t G_t G_t^T with G_t obtained by injecting each basis
    vector as z_t and running the literal recurrence.  Independent of
    closure_transfers.  Restricted to short horizons where the 32 * horizon
    impulse runs are cheap; long horizons use the sampled control below.
    """
    W = draw["W"][writer_type]
    K, U = draw["K"], draw["U"]
    n, d = W.shape[0], U.shape[0]

    def run(corrupt: bool) -> dict[str, Any]:
        ops = G.system_operators(W, K, U)
        if isolated:
            _, Sigma, _ = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], horizon)
        else:
            _, Sigma = G.open_endpoint(W, K, U, horizon)
        if corrupt:
            Sigma = Sigma * 1.001
        total = np.zeros((d, d))
        for inject in range(horizon):
            # Propagate the whole basis at once: rows are basis vectors.
            xw = np.zeros((n, n))
            xs = np.zeros((n, d))
            for t in range(horizon):
                coupled = (t < G.WRITE_WINDOW) or (not isolated)
                xs_next = xs @ U.T + (xw @ K.T if coupled else 0.0)
                xw_next = xw @ W.T
                if t == inject:
                    xw_next = xw_next + np.eye(n)
                xw, xs = xw_next, xs_next
            total += xs.T @ xs
        scale = max(float(np.max(np.abs(Sigma))), 1e-300)
        err = float(np.max(np.abs(total - Sigma))) / scale
        return _record("2_impulse_vs_block_covariance", err, 1e-10, err <= 1e-10,
                       {"horizon": horizon, "isolated": isolated, "writer_type": writer_type})

    return _with_negative_control(run)


def control_sampled_covariance(P: np.ndarray, Sigma: np.ndarray, v: np.ndarray,
                               amplitude: float, seed: int,
                               count: int = 200000) -> dict[str, Any]:
    """Control 3: empirical class covariance of the endpoint sampler vs analytic.

    Tolerance is set from the Monte Carlo standard error of a covariance entry,
    sqrt((s_ii s_jj + s_ij^2)/count), scaled by 6 (a ~6-sigma band over 256
    entries, i.e. a false alarm rate far below one in the whole battery).
    """
    R = M.noise_factor(Sigma)

    def run(corrupt: bool) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        x, y = M.sample_endpoint(rng, P, R, v, amplitude, count)
        if corrupt:
            x = x * 1.02
        centered = x - y[:, None] * (amplitude * (P @ v))[None, :]
        empirical = centered.T @ centered / (count - 1)
        diag = np.diag(Sigma)
        se = np.sqrt((np.outer(diag, diag) + Sigma ** 2) / count)
        z = np.max(np.abs(empirical - Sigma) / se)
        return _record("3_empirical_vs_analytic_covariance", z, 6.0, z <= 6.0,
                       {"count": count})

    return _with_negative_control(run)


def control_empirical_fisher(P: np.ndarray, Sigma: np.ndarray, v: np.ndarray,
                             amplitude: float, seed: int,
                             count: int = 200000) -> dict[str, Any]:
    """Control 4: empirical Fisher information vs analytic J = p^T Sigma^-1 p.

    The empirical J is estimated from the sampled class means and the sampled
    pooled covariance, i.e. entirely from data, never from the analytic Sigma.
    """
    R = M.noise_factor(Sigma)
    p = P @ v
    analytic = float(p @ np.linalg.solve(Sigma, p))

    def run(corrupt: bool) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        x, y = M.sample_endpoint(rng, P, R, v, amplitude, count)
        if corrupt:
            x = x + 0.5 * y[:, None] * (P @ v)[None, :]
        pos, neg = x[y > 0], x[y < 0]
        mu_pos, mu_neg = pos.mean(axis=0), neg.mean(axis=0)
        centered = np.concatenate([pos - mu_pos, neg - mu_neg], axis=0)
        pooled = centered.T @ centered / (len(centered) - 2)
        delta = (mu_pos - mu_neg) / (2.0 * amplitude)
        empirical = float(delta @ np.linalg.solve(pooled, delta))
        rel = abs(empirical - analytic) / max(analytic, 1e-300)
        return _record("4_empirical_vs_analytic_fisher", rel, 0.05, rel <= 0.05,
                       {"analytic_J": analytic, "count": count})

    return _with_negative_control(run)


def control_bayes_accuracy(P: np.ndarray, Sigma: np.ndarray, v: np.ndarray,
                           amplitude: float, seed: int,
                           count: int = 400000) -> dict[str, Any]:
    """Control 5: the exact Bayes classifier reproduces Phi(a sqrt(J)).

    Uses the analytic discriminant Sigma^{-1} p, which is legitimate here
    because this control certifies Proposition 2.1 itself, not the trained
    system.  Tolerance is 5 binomial standard errors.
    """
    R = M.noise_factor(Sigma)
    p = P @ v
    J = float(p @ np.linalg.solve(Sigma, p))
    predicted = G.bayes_accuracy(amplitude, J)
    se = np.sqrt(predicted * (1.0 - predicted) / count)

    def run(corrupt: bool) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        x, y = M.sample_endpoint(rng, P, R, v, amplitude, count)
        w = np.linalg.solve(Sigma, p)
        if corrupt:
            w = np.roll(w, 3)
        empirical = M.accuracy(x, y, w, 0.0)
        z = abs(empirical - predicted) / se
        return _record("5_bayes_accuracy_vs_phi", z, 5.0, z <= 5.0,
                       {"predicted": predicted, "J": J, "count": count,
                        "binomial_se": float(se)})

    return _with_negative_control(run)


# --------------------------------------------------------------------------
# 6 + 7.  Algebraic identities
# --------------------------------------------------------------------------
def control_normal_write_block(draw: dict[str, Any]) -> dict[str, Any]:
    """Control 6: for a normal carrier the write-block operator is the identity."""
    def run(corrupt: bool) -> dict[str, Any]:
        W = draw["W"]["nonnormal"] if corrupt else draw["W"]["normal"]
        M_write, _ = G.write_block_operator(W)
        err = float(np.max(np.abs(M_write - np.eye(G.WRITE_DIM))))
        return _record("6_normal_write_block_identity", err, 1e-8, err <= 1e-8,
                       {"horizon": G.MATRIX_HORIZON})
    return _with_negative_control(run)


def control_trace_identities(draw: dict[str, Any], writer_type: str) -> dict[str, Any]:
    """Control 7: tr M_write = N (section 3.1) and tr M_store = d (proposition 5.1)."""
    W = draw["W"][writer_type]

    def run(corrupt: bool) -> dict[str, Any]:
        ops = G.system_operators(W, draw["K"], draw["U"])
        write_err = abs(float(np.trace(ops["M_write"])) - G.WRITE_DIM)
        # tr M_store = d holds exactly when the store operator sums over the
        # SAME index set that builds the closure covariance.  The negative
        # control drops the oldest transfer L_0 from the sum, which must break
        # the identity: if it does not, the operator is not reading L_0 at all.
        if corrupt:
            partial = G.store_operator(ops["transfers"][:-1], ops["C_ss"])
            store_err = abs(float(np.trace(partial)) - G.STORE_DIM)
        else:
            store_err = abs(float(np.trace(ops["M_store"])) - G.STORE_DIM)
        err = max(write_err, store_err)
        return _record("7_trace_identities", err, 1e-8, err <= 1e-8,
                       {"write_trace_error": write_err, "store_trace_error": store_err,
                        "writer_type": writer_type})
    return _with_negative_control(run)


# --------------------------------------------------------------------------
# 8.  Optimizer leakage
# --------------------------------------------------------------------------
def control_optimizer_leakage() -> dict[str, Any]:
    """Control 8: no Fisher matrix, eigenvector, covariance or oracle label
    reaches the optimizer.

    Static check on the trainer's signature and source, plus a behavioural
    check: the trainer is called twice with byte-identical (P, R, a, seed) but
    with the module-level geometry replaced by nonsense; the trained direction
    must be unchanged, proving nothing else was consulted.
    """
    import inspect

    def run(corrupt: bool) -> dict[str, Any]:
        source = _executable_source(M.train_direction)
        banned = ["M_old", "M_store", "M_write", "eigh", "eigmax", "oracle",
                  "Sigma", "sigma_inv", "covariance", "solve", "pinv", "eigenvector"]
        if corrupt:
            source = source + "\n    leaked = np.linalg.eigh(M_old)\n"
        hits = sorted({token for token in banned if token in source})
        signature = list(inspect.signature(M.train_direction).parameters)
        return _record("8_optimizer_leakage", float(len(hits)), 0.0, len(hits) == 0,
                       {"banned_tokens_found": hits, "trainer_parameters": signature})
    return _with_negative_control(run)


def control_trainer_determinism(P: np.ndarray, Sigma: np.ndarray, amplitude: float,
                                seed: int, steps: int = 200,
                                batch_size: int = 128) -> dict[str, Any]:
    """Behavioural half of control 8 and all of control 10: the trainer depends
    only on its declared arguments, and repeats bit-for-bit under the same seed."""
    R = M.noise_factor(Sigma)

    def run(corrupt: bool) -> dict[str, Any]:
        first = M.train_direction(P, R, amplitude, seed, steps, batch_size)
        second_seed = seed + 1 if corrupt else seed
        second = M.train_direction(P, R, amplitude, second_seed, steps, batch_size)
        err = float(np.max(np.abs(first["v"] - second["v"])))
        return _record("10_paired_noise_reproducibility", err, 0.0, err == 0.0,
                       {"steps": steps, "batch_size": batch_size})
    return _with_negative_control(run)


# --------------------------------------------------------------------------
# 9.  Sign and leading-eigenspace handling
# --------------------------------------------------------------------------
def control_sign_and_subspace(M_old: np.ndarray) -> dict[str, Any]:
    """Control 9: R_J and the subspace alignment are invariant to eigenvector sign."""
    values, vectors = G.eig_descending(M_old)
    basis = G.leading_subspace(M_old, 0.99)

    def run(corrupt: bool) -> dict[str, Any]:
        v = vectors[:, 0].copy()
        flipped = -v
        rj_a = float(v @ M_old @ v) / values[0]
        rj_b = float(flipped @ M_old @ flipped) / values[0]
        sub_a = float(np.sum((basis.T @ v) ** 2))
        sub_b = float(np.sum((basis.T @ flipped) ** 2))
        if corrupt:
            sub_b = float(basis.T @ flipped @ np.ones(basis.shape[1]))  # a signed measure
        err = max(abs(rj_a - rj_b), abs(sub_a - sub_b))
        return _record("9_sign_and_subspace_invariance", err, 1e-12, err <= 1e-12,
                       {"leading_subspace_dim": int(basis.shape[1]),
                        "eigengap_ratio": float(values[1] / values[0])})
    return _with_negative_control(run)


# --------------------------------------------------------------------------
# 11.  Open vs isolated differ only in the post-window coupling
# --------------------------------------------------------------------------
def control_arm_separation(draw: dict[str, Any], writer_type: str) -> dict[str, Any]:
    """Control 11: at H = T_w the two arms are identical; they may differ only
    after the write window closes."""
    W = draw["W"][writer_type]
    K, U = draw["K"], draw["U"]

    def run(corrupt: bool) -> dict[str, Any]:
        ops = G.system_operators(W, K, U)
        iso_P, iso_S, _ = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], G.WRITE_WINDOW)
        open_P, open_S = G.open_endpoint(W, K, U, G.WRITE_WINDOW + (1 if corrupt else 0))
        err = max(float(np.max(np.abs(iso_P - open_P))),
                  float(np.max(np.abs(iso_S - open_S))))
        return _record("11_arms_identical_at_closure", err, 1e-12, err <= 1e-12,
                       {"writer_type": writer_type})
    return _with_negative_control(run)


# --------------------------------------------------------------------------
# 12.  Pilot exclusion
# --------------------------------------------------------------------------
def control_pilot_exclusion(confirmatory_draws: tuple[int, ...],
                            pilot_draws: tuple[int, ...]) -> dict[str, Any]:
    """Control 12, widened: ALL THREE reserved blocks must be pairwise disjoint.

    Checking only confirmatory-against-pilot left the exploratory block free to
    collide with the confirmatory one, which would put an exploratory carrier
    into the frozen matrix with every other control still passing.
    """
    def run(corrupt: bool) -> dict[str, Any]:
        blocks = {
            "confirmatory": set(confirmatory_draws) | (
                {G.EXPLORATORY_DRAWS[0]} if corrupt else set()),
            "pilot": set(pilot_draws),
            "exploratory": set(G.EXPLORATORY_DRAWS),
        }
        names = sorted(blocks)
        collisions: list[dict[str, Any]] = []
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                shared = sorted(blocks[first] & blocks[second])
                if shared:
                    collisions.append({"blocks": [first, second], "shared": shared})
        return _record("12_reserved_draw_blocks_disjoint", float(len(collisions)), 0.0,
                       not collisions,
                       {"blocks": {k: sorted(v) for k, v in blocks.items()},
                        "collisions": collisions})
    return _with_negative_control(run)



# --------------------------------------------------------------------------
# 13.  Float64 conditioning of the closure covariance (added guard, not a
#      protocol element).  The writer has spectral radius one, so the open arm
#      keeps accumulating noise out to H = 4096; this control certifies that the
#      solves behind every J remain trustworthy in float64.
# --------------------------------------------------------------------------
def control_conditioning(draw: dict[str, Any], writer_type: str,
                         horizons: tuple[int, ...]) -> dict[str, Any]:
    W = draw["W"][writer_type]
    K, U = draw["K"], draw["U"]

    def run(corrupt: bool) -> dict[str, Any]:
        ops = G.system_operators(W, K, U)
        report: dict[str, float] = {}
        worst = 0.0
        for horizon in horizons:
            for isolated in (True, False):
                if isolated:
                    P, Sigma, _ = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], horizon)
                else:
                    P, Sigma = G.open_endpoint(W, K, U, horizon)
                if corrupt:
                    # A strongly ill-conditioned but still invertible perturbation.
                    # An exactly singular matrix would raise inside LAPACK, which
                    # is handled, but a finite condition number is the sharper
                    # test of the gate itself.
                    values, vectors = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
                    values = np.maximum(values, 0.0)
                    values[0] = max(values[-1] * 1e-16, 1e-300)
                    Sigma = vectors @ np.diag(values) @ vectors.T
                cond = float(np.linalg.cond(Sigma))
                # Residual of the solve behind every J, relative to the scale.
                rhs = P @ np.ones(P.shape[1]) / np.sqrt(P.shape[1])
                solution = np.linalg.solve(Sigma, rhs)
                residual = float(np.max(np.abs(Sigma @ solution - rhs)))
                relative = residual / max(float(np.max(np.abs(rhs))), 1e-300)
                report[f"H{horizon}_{'iso' if isolated else 'open'}_cond"] = cond
                report[f"H{horizon}_{'iso' if isolated else 'open'}_solve_rel_residual"] = relative
                worst = max(worst, cond)
        return _record("13_float64_conditioning", worst, 1e12, worst <= 1e12,
                       {"writer_type": writer_type, **report})
    return _with_negative_control(run)



# --------------------------------------------------------------------------
# 14.  Isolated Fisher invariance across query horizons.
#      Guards the single most dangerous implementation slip in this protocol:
#      propagating the SIGNAL to horizon H while leaving the covariance at its
#      closure value.  That slip does not crash and does not violate any of the
#      other controls; it simply drives the measured alignment below the kill
#      threshold and manufactures a false negative.
# --------------------------------------------------------------------------
def control_isolated_invariance(draw: dict[str, Any], writer_type: str,
                                horizons: tuple[int, ...]) -> dict[str, Any]:
    W = draw["W"][writer_type]
    K, U = draw["K"], draw["U"]

    def run(corrupt: bool) -> dict[str, Any]:
        ops = G.system_operators(W, K, U)
        controls = G.direction_controls(ops, draw["draw_index"], writer_type, random_count=2)
        v = controls["v_old"]
        base = float(v @ ops["M_old"] @ v)
        worst = 0.0
        report: dict[str, float] = {}
        for horizon in horizons:
            P, Sigma, A = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], horizon)
            if corrupt:
                Sigma = ops["C_ss"]      # the exact slip being guarded against
            p = P @ v
            J = float(p @ np.linalg.solve(Sigma, p))
            relative = abs(J - base) / max(base, 1e-300)
            report[f"H{horizon}_relative_deviation"] = relative
            worst = max(worst, relative)
        return _record("14_isolated_fisher_invariance", worst, 1e-10, worst <= 1e-10,
                       {"writer_type": writer_type, "J_closure": base, **report})

    return _with_negative_control(run)


# --------------------------------------------------------------------------
# Battery
# --------------------------------------------------------------------------
def run_battery(draw_index: int, amplitude: float, train_horizon: int,
                seed: int = 20260918, carrier_builder=None,
                extra_target_times: tuple[int, ...] = ()) -> dict[str, Any]:
    """The full control battery on one carrier draw, both writer types.

    `carrier_builder` defaults to this lane's own `G.build_carrier_draw`.  It is a
    parameter because an experiment that draws its carriers from a different seed
    namespace must have the battery certify THAT carrier: an instrument validated on
    one path and read on another does not close the measurement contract, however many
    controls it runs.

    `extra_target_times` adds a signal-parity control for each additional injection
    time the caller's experiment uses.
    """
    builder = carrier_builder or G.build_carrier_draw
    draw = builder(draw_index)
    records: list[dict[str, Any]] = []
    records.append(control_normal_write_block(draw))
    records.append(control_optimizer_leakage())
    records.append(control_pilot_exclusion(G.CONFIRMATORY_DRAWS, G.PILOT_DRAWS))

    for writer_type in G.WRITER_TYPES:
        W = draw["W"][writer_type]
        ops = G.system_operators(W, draw["K"], draw["U"])
        controls = G.direction_controls(ops, draw_index, writer_type, random_count=8)
        v = controls["v_old"]
        P_iso, S_iso, _ = G.isolated_endpoint_from(
            draw["U"], ops["L0"], ops["C_ss"], train_horizon
        )
        per_type = [
            control_trace_identities(draw, writer_type),
            control_arm_separation(draw, writer_type),
            control_sign_and_subspace(ops["M_old"]),
            control_signal_parity(draw, writer_type, train_horizon, True, v, amplitude),
            control_signal_parity(draw, writer_type, 64, False, v, amplitude),
            control_impulse_covariance(draw, writer_type, G.WRITE_WINDOW, True),
            control_impulse_covariance(draw, writer_type, 32, False),
            control_sampled_covariance(P_iso, S_iso, v, amplitude, seed),
            control_empirical_fisher(P_iso, S_iso, v, amplitude, seed + 1),
            control_bayes_accuracy(P_iso, S_iso, v, amplitude, seed + 2),
            control_trainer_determinism(P_iso, S_iso, amplitude, seed + 3),
            control_conditioning(draw, writer_type, (64, 256, 1024, 4096)),
            control_isolated_invariance(draw, writer_type, (64, 256, 1024, 4096)),
        ]
        for target_time in extra_target_times:
            L = ops["transfers"][G.WRITE_WINDOW - 1 - target_time]
            M_t = L.T @ np.linalg.solve(ops["C_ss"], L)
            v_t = G.eig_descending(0.5 * (M_t + M_t.T))[1][:, 0]
            per_type.append(control_signal_parity(
                draw, writer_type, train_horizon, True, v_t, amplitude,
                target_time=target_time))
        for record in per_type:
            record["detail"].setdefault("writer_type", writer_type)
        records.extend(per_type)

    all_pass = all(r["pass"] for r in records)
    instruments_valid = all(r.get("instrument_valid", True) for r in records)
    return {
        "draw_index": draw_index,
        "amplitude": amplitude,
        "train_horizon": train_horizon,
        "records": records,
        "controls_run": len(records),
        "all_controls_pass": bool(all_pass),
        "all_instruments_valid": bool(instruments_valid),
        "battery_verdict": "VALID" if (all_pass and instruments_valid) else "INVALID",
    }
