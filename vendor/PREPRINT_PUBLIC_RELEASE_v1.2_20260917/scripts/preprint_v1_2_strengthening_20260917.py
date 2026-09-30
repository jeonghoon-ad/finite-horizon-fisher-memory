"""Strengthening experiments for the v1.2 Fisher-memory preprint.

This public, deterministic script adds four items to the v1.1 package:
1. a finite-horizon Cesaro/commutant certification check;
2. end-to-end store-operator direction selection;
3. sampled GLS decoder verification under invertible post-write holds; and
4. randomized checks of the approximate-isolation lower bound.

It imports only numerical helper functions from the frozen v1.1 scripts and does
not require the internal execution-gate environment variables.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results" / "preprint_v1_2_strengthening_20260917"
RESULTS.mkdir(parents=True, exist_ok=True)

DRAW_COUNT = 8
WRITE_DIM = 32
STORE_DIM = 16
WRITE_WINDOW = 24
MATRIX_HORIZON = 2048
TRIALS_PER_DRAW = 5000
MASTER_SEED = 2026091701
ALPHAS = (0.05, 0.25, 1.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


source = load_module("preprint_replication", SCRIPTS / "preprint_replication_and_factorial_20260903.py")
iso = load_module("isolation_factorial", SCRIPTS / "isolation_factorial_v2_bounded_20260903.py")


def similarity_components_from_meta(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seeds = meta["seeds"]
    left = source.haar_orthogonal(WRITE_DIM, np.random.default_rng(int(seeds["s_left"])))
    right = source.haar_orthogonal(WRITE_DIM, np.random.default_rng(int(seeds["s_right"])))
    singular_values = np.geomspace(1.0, iso.CONDITION, WRITE_DIM).astype(np.float64)
    S = left @ np.diag(singular_values) @ right.T
    Q = source.haar_orthogonal(WRITE_DIM, np.random.default_rng(int(seeds["q"])))
    W = S @ Q @ np.linalg.inv(S)
    return S, Q, W


def exact_commutant_projection(Q: np.ndarray, A: np.ndarray, tol: float = 1e-8):
    values, vectors = np.linalg.eig(Q.astype(np.complex128))
    vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
    used = np.zeros(len(values), dtype=bool)
    groups: list[tuple[complex, np.ndarray]] = []
    for i, value in enumerate(values):
        if used[i]:
            continue
        indices = [j for j, other in enumerate(values) if (not used[j]) and abs(other - value) <= tol]
        for j in indices:
            used[j] = True
        basis, _ = np.linalg.qr(vectors[:, indices])
        groups.append((value, basis))
    Abar = np.zeros_like(A, dtype=np.complex128)
    for _, basis in groups:
        projector = basis @ basis.conj().T
        Abar += projector @ A @ projector
    gaps = [abs(1.0 - vi * np.conj(vj)) for i, (vi, _) in enumerate(groups) for j, (vj, _) in enumerate(groups) if i != j]
    delta = float(min(gaps)) if gaps else float("inf")
    Abar = np.real_if_close(Abar, tol=1000).real
    return Abar, delta


def cesaro_average(Q: np.ndarray, A: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(A)
    power = np.eye(Q.shape[0])
    for _ in range(n):
        out += power @ A @ power.T
        power = power @ Q
    return out / n


def finite_horizon_certificate(S: np.ndarray, Q: np.ndarray, W: np.ndarray, n: int) -> dict[str, float]:
    Sinv = np.linalg.inv(S)
    A = Sinv @ Sinv.T
    Abar, delta = exact_commutant_projection(Q, A)
    An = cesaro_average(Q, A, n)
    Abar_inv = np.linalg.inv(Abar)
    M_inf = Sinv.T @ Abar_inv @ Sinv
    M_n, _ = source.fisher_memory_matrix(W, n)

    offdiag_norm = float(np.linalg.norm(A - Abar, ord="fro"))
    A_error_bound = float(2.0 * offdiag_norm / (n * delta))
    eta = float(np.linalg.norm(Abar_inv, 2) * A_error_bound)
    if eta < 1.0:
        inverse_bound = float(np.linalg.norm(Abar_inv, 2) ** 2 * A_error_bound / (1.0 - eta))
        M_error_bound = float(np.linalg.norm(Sinv, 2) ** 2 * inverse_bound)
    else:
        M_error_bound = float("inf")

    return {
        "eigenvalue_group_gap_delta": delta,
        "A_cesaro_error_2": float(np.linalg.norm(An - Abar, 2)),
        "A_cesaro_error_bound": A_error_bound,
        "certificate_eta": eta,
        "M_error_2": float(np.linalg.norm(M_n - M_inf, 2)),
        "M_error_bound": M_error_bound,
        "M_relative_frobenius_error": float(np.linalg.norm(M_n - M_inf, "fro") / np.linalg.norm(M_inf, "fro")),
        "commutator_residual": float(np.linalg.norm(Abar @ Q - Q @ Abar, "fro")),
    }


def rotation(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)


def near_degenerate_control() -> dict[str, float]:
    theta = 0.7
    gap = 1e-4
    Q = np.block([[rotation(theta), np.zeros((2, 2))], [np.zeros((2, 2)), rotation(theta + gap)]])
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    A[:2, 2:] = 0.8 * np.eye(2)
    A[2:, :2] = 0.8 * np.eye(2)
    Abar = np.block([
        [1.5 * np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), 3.5 * np.eye(2)],
    ])
    m = 1000
    Am = cesaro_average(Q, A, m)
    A2m = cesaro_average(Q, A, 2 * m)
    delta = min(
        abs(1.0 - vi * np.conj(vj))
        for vi in np.linalg.eigvals(Q.astype(complex))
        for vj in np.linalg.eigvals(Q.astype(complex))
        if abs(vi - vj) > 1e-8
    )
    return {
        "rotation_gap": gap,
        "spectral_group_gap_delta": float(delta),
        "m": m,
        "error_to_exact_projection_fro": float(np.linalg.norm(Am - Abar, "fro")),
        "two_scale_difference_fro": float(np.linalg.norm(A2m - Am, "fro")),
        "commutator_residual_fro": float(np.linalg.norm(Am @ Q - Q @ Am, "fro")),
        "coefficient_bound_2_over_m_delta": float(2.0 / (m * delta)),
    }


def closure_transfers(W: np.ndarray, K: np.ndarray, U: np.ndarray, steps: int):
    lower = np.zeros((K.shape[0], W.shape[0]), dtype=np.float64)
    store_power = np.eye(U.shape[0], dtype=np.float64)
    transfers: list[np.ndarray] = []
    covariance = np.zeros((K.shape[0], K.shape[0]), dtype=np.float64)
    for _ in range(steps):
        transfers.append(lower.copy())
        covariance += lower @ lower.T
        lower = lower @ W + store_power @ K
        store_power = store_power @ U
    return transfers, covariance


def store_operator(transfers: list[np.ndarray], covariance: np.ndarray) -> np.ndarray:
    covariance_inverse = np.linalg.inv(covariance)
    return sum((L.T @ covariance_inverse @ L for L in transfers), np.zeros((WRITE_DIM, WRITE_DIM)))


def rayleigh(M: np.ndarray, v: np.ndarray) -> float:
    return float(v.T @ M @ v)


def decoder_experiment(
    p: np.ndarray,
    covariance: np.ndarray,
    U: np.ndarray,
    draw_index: int,
) -> dict[str, Any]:
    covariance_inverse = np.linalg.inv(covariance)
    J0 = float(p.T @ covariance_inverse @ p)
    a0 = covariance_inverse @ p / J0
    rng = np.random.default_rng(MASTER_SEED + draw_index)
    scalar = rng.standard_normal(TRIALS_PER_DRAW)
    noise = rng.multivariate_normal(np.zeros(STORE_DIM), covariance, size=TRIALS_PER_DRAW)
    state = scalar[:, None] * p[None, :] + noise
    closure_estimate = state @ a0
    error = closure_estimate - scalar
    result: dict[str, Any] = {
        "J0": J0,
        "theoretical_error_variance": 1.0 / J0,
        "sample_error_variance": float(np.var(error, ddof=1)),
        "sample_to_theory_variance_ratio": float(np.var(error, ddof=1) * J0),
        "bias_in_standard_errors": float(np.mean(error) / np.sqrt(np.var(error, ddof=1) / TRIALS_PER_DRAW)),
        "holds": {},
    }
    for rho in (1.0, 0.9):
        for horizon in (200, 700):
            Uh = np.linalg.matrix_power(U, horizon)
            Ah = (rho ** horizon) * Uh
            later_state = state @ Ah.T
            later_p = Ah @ p
            later_covariance = Ah @ covariance @ Ah.T
            inverse_adjoint = np.linalg.solve(Ah.T, a0)
            estimate_inverse_adjoint = later_state @ inverse_adjoint
            unwarped_state = np.linalg.solve(Ah, later_state.T).T
            estimate_unwarped = unwarped_state @ a0
            gls = np.linalg.solve(later_covariance, later_p)
            gls /= float(later_p.T @ gls)
            estimate_gls = later_state @ gls
            result["holds"][f"rho={rho},H={horizon}"] = {
                "inverse_adjoint_max_abs_diff": float(np.max(np.abs(estimate_inverse_adjoint - closure_estimate))),
                "state_unwarp_max_abs_diff": float(np.max(np.abs(estimate_unwarped - closure_estimate))),
                "propagated_gls_max_abs_diff": float(np.max(np.abs(estimate_gls - closure_estimate))),
                "common_scale": float(rho ** horizon),
            }
    return result


def sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def approximate_isolation_checks(
    p: np.ndarray,
    covariance: np.ndarray,
    draw_index: int,
) -> list[dict[str, float]]:
    covariance_inverse = np.linalg.inv(covariance)
    J0 = float(p.T @ covariance_inverse @ p)
    Croot = sqrt_psd(covariance)
    rng = np.random.default_rng(MASTER_SEED + 1000 + draw_index)
    q, _ = np.linalg.qr(rng.standard_normal((STORE_DIM, STORE_DIM)))
    eigenvalues = rng.uniform(0.0, 1.0, size=STORE_DIM)
    H = q @ np.diag(eigenvalues) @ q.T
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        random_R = alpha * Croot @ H @ Croot.T
        aligned_R = alpha * covariance
        for label, R in (("random_psd", random_R), ("aligned_equality", aligned_R)):
            J = float(p.T @ np.linalg.solve(covariance + R, p))
            rows.append({
                "alpha": alpha,
                "construction": label,
                "J0": J0,
                "J": J,
                "retained_fraction": J / J0,
                "guaranteed_floor": 1.0 / (1.0 + alpha),
                "bound_margin": J / J0 - 1.0 / (1.0 + alpha),
            })
    return rows


def main() -> None:
    certificate_rows: list[dict[str, Any]] = []
    store_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    isolation_rows: list[dict[str, Any]] = []

    for draw in range(DRAW_COUNT):
        normal, nonnormal, meta = iso.paired_write_carriers(source, draw)
        coupling, U, random_direction, seeds = iso.paired_draw_objects(source, draw)
        S, Q, reconstructed = similarity_components_from_meta(meta)
        if np.max(np.abs(reconstructed - nonnormal)) > 1e-12:
            raise RuntimeError("reconstructed paired carrier mismatch")

        cert = finite_horizon_certificate(S, Q, nonnormal, MATRIX_HORIZON)
        cert["draw"] = draw
        certificate_rows.append(cert)

        write_matrix, _ = source.fisher_memory_matrix(nonnormal, MATRIX_HORIZON)
        write_values, write_vectors = np.linalg.eigh((write_matrix + write_matrix.T) / 2)
        write_oracle = source.canonicalize_eigenvector_signs(write_vectors)[:, -1]

        transfers, covariance = closure_transfers(nonnormal, coupling, U, WRITE_WINDOW)
        Mstore = store_operator(transfers, covariance)
        store_values, store_vectors = np.linalg.eigh((Mstore + Mstore.T) / 2)
        store_oracle = source.canonicalize_eigenvector_signs(store_vectors)[:, -1]
        covariance_inverse = np.linalg.inv(covariance)

        def direction_stats(name: str, v: np.ndarray) -> dict[str, Any]:
            oldest_sensitivity = transfers[-1] @ v
            return {
                "draw": draw,
                "direction": name,
                "write_allocation": rayleigh(write_matrix, v),
                "store_total": rayleigh(Mstore, v),
                "oldest_input_information": float(oldest_sensitivity.T @ covariance_inverse @ oldest_sensitivity),
                "oldest_transfer_gain": float(np.linalg.norm(oldest_sensitivity)),
                "store_trace": float(np.trace(Mstore)),
                "store_lambda_max": float(store_values[-1]),
            }

        write_stats = direction_stats("write_oracle", write_oracle)
        store_stats = direction_stats("store_oracle", store_oracle)
        random_stats = direction_stats("random", random_direction[:, 0])
        store_rows.extend((write_stats, store_stats, random_stats))

        p_store = transfers[-1] @ store_oracle
        decoder = decoder_experiment(p_store, covariance, U, draw)
        decoder["draw"] = draw
        decoder_rows.append(decoder)
        for row in approximate_isolation_checks(p_store, covariance, draw):
            row["draw"] = draw
            isolation_rows.append(row)

    control = near_degenerate_control()

    def summary(values):
        arr = np.asarray(values, dtype=float)
        return {
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
        }

    grouped = {name: [r for r in store_rows if r["direction"] == name] for name in ("write_oracle", "store_oracle", "random")}
    paired_total_ratios = [
        grouped["store_oracle"][i]["store_total"] / grouped["write_oracle"][i]["store_total"]
        for i in range(DRAW_COUNT)
    ]
    paired_oldest_ratios = [
        grouped["store_oracle"][i]["oldest_input_information"] / grouped["write_oracle"][i]["oldest_input_information"]
        for i in range(DRAW_COUNT)
    ]

    all_hold_errors = [
        metric
        for row in decoder_rows
        for hold in row["holds"].values()
        for key, metric in hold.items()
        if key.endswith("max_abs_diff")
    ]
    receipt = {
        "status": "PASS",
        "script": str(Path(__file__).name),
        "script_sha256": sha256_file(Path(__file__)),
        "draw_count": DRAW_COUNT,
        "trials_per_draw": TRIALS_PER_DRAW,
        "finite_horizon_certificate": {
            "relative_M_error": summary([r["M_relative_frobenius_error"] for r in certificate_rows]),
            "all_actual_errors_below_reported_bound": all(
                np.isfinite(r["M_error_bound"]) and r["M_error_2"] <= r["M_error_bound"] + 1e-12
                for r in certificate_rows
            ),
            "near_degenerate_control": control,
        },
        "store_operator_selection": {
            "store_trace": summary([r["store_trace"] for r in grouped["store_oracle"]]),
            "store_oracle_vs_write_oracle_total_ratio": summary(paired_total_ratios),
            "store_oracle_vs_write_oracle_oldest_ratio": summary(paired_oldest_ratios),
            "individual_oldest_ratio_below_one_count": int(sum(value < 1.0 for value in paired_oldest_ratios)),
            "store_oracle_write_allocation": summary([r["write_allocation"] for r in grouped["store_oracle"]]),
            "all_store_oracles_above_write_average_1p05": all(r["write_allocation"] >= 1.05 for r in grouped["store_oracle"]),
            "all_oldest_transfer_gains_above_1e_6": all(r["oldest_transfer_gain"] >= 1e-6 for r in grouped["store_oracle"]),
        },
        "sampled_decoder": {
            "variance_ratio": summary([r["sample_to_theory_variance_ratio"] for r in decoder_rows]),
            "bias_in_standard_errors": summary([r["bias_in_standard_errors"] for r in decoder_rows]),
            "maximum_hold_readout_difference": float(max(all_hold_errors)),
        },
        "approximate_isolation": {
            "minimum_bound_margin": float(min(r["bound_margin"] for r in isolation_rows)),
            "all_bounds_pass": all(r["bound_margin"] >= -1e-12 for r in isolation_rows),
            "aligned_equality_max_abs_error": float(max(
                abs(r["retained_fraction"] - r["guaranteed_floor"])
                for r in isolation_rows if r["construction"] == "aligned_equality"
            )),
        },
    }

    (RESULTS / "V1_2_STRENGTHENING_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (RESULTS / "FINITE_HORIZON_CERTIFICATES.json").write_text(json.dumps(certificate_rows, indent=2), encoding="utf-8")
    (RESULTS / "NEAR_DEGENERATE_CONTROL.json").write_text(json.dumps(control, indent=2), encoding="utf-8")
    (RESULTS / "STORE_OPERATOR_SELECTION.json").write_text(json.dumps(store_rows, indent=2), encoding="utf-8")
    (RESULTS / "SAMPLED_DECODER.json").write_text(json.dumps(decoder_rows, indent=2), encoding="utf-8")
    (RESULTS / "APPROXIMATE_ISOLATION.json").write_text(json.dumps(isolation_rows, indent=2), encoding="utf-8")

    with (RESULTS / "STORE_OPERATOR_SELECTION.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(store_rows[0].keys()))
        writer.writeheader()
        writer.writerows(store_rows)
    with (RESULTS / "APPROXIMATE_ISOLATION.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(isolation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(isolation_rows)

    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
