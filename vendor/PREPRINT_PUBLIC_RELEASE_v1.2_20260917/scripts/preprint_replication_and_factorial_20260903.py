"""Replicated directional census and write-path x storage-isolation factorial.

DECISION-BEARING SYNTHETIC MEASUREMENT. Run only through the REPRODUCTION wrapper
after the named independent source review binds this exact script SHA-256 and
contains exactly one ``REPRODUCTION DECISION: RUN`` line.

Part A mirrors the reviewed finite-horizon Fisher-memory instrument from
``fourth_cell_cond_sweep_20260902.py``. Part B mirrors the 2026-08-23
Ganguli pocket-capacity time-varying instrument. No checkpoint, trained weight,
GPU, network service, controller, queue, or roadmap is read or modified.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time

import numpy as np


MASTER_SEED = 20260903
EPS = 1.0
N = 32
DRAW_COUNT = 8
MATRIX_HORIZON = 2048
HORIZONS = (32, 128, 512, 2048, 4096)
FACTORIAL_HORIZONS = (64, 256, 1024, 4096)
POWER_BOUND_HORIZON = 4096
WRITE_WINDOW = 24

SIMILARITY_CONDITIONS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
HELDOUT_SIMILARITY_CONDITION = 35.0
ELLIPTIC_CONDITIONS = (4.0, 100.0)
HAMILTONIAN_CONDITIONS = (4.0, 100.0)

TRACE_ATOL = 5e-9
SELF_CONSISTENCY_ATOL = 5e-9
THEOREM_NUMERIC_ATOL = 5e-10
FLATNESS_REL_TOL = 0.05
SIX_DECIMAL_ATOL = 5e-7
FACTORIAL_CONSTANT_ATOL = 5e-7

COND_SWEEP_SHA256 = "c6c4d8d4be53df512f60d0d62a2180969d5767dd6e721b278f9e364fff6f04e9"
GANGULI_SHA256 = "34ff3513574f28a887b1f915d2eee9f4be1a9065bd5c4cbc357db9a892068903"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COND_SWEEP_PATH = PROJECT_ROOT / "scripts" / "fourth_cell_cond_sweep_20260902.py"
GANGULI_PATH = PROJECT_ROOT / "scripts" / "verify_ganguli_pocket_capacity_20260823.py"
REVIEW_PATH = PROJECT_ROOT / "prior_art" / "REVIEW_PREPRINT_REPLICATION_AND_ISOLATION_FACTORIAL_20260903.md"
OUTPUT_DIR = PROJECT_ROOT / "results" / "preprint_replication_and_factorial_20260903"


# BEGIN VERBATIM MIRROR: fourth_cell_cond_sweep_20260902.py
def haar_orthogonal(dim: int, gen: np.random.Generator) -> np.ndarray:
    a = gen.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def fisher_memory(W: np.ndarray, V: np.ndarray, horizon: int, noise_cov: np.ndarray | None = None):
    """Return (Jk, Jtot) with Jk[k] = trace of the k-th Fisher memory block, normalized by 1/eps.

    V is N x m (m input channels, orthonormal columns). noise_cov is the per-step
    process-noise covariance; default eps*I.
    """
    n_dim = W.shape[0]
    q_step = np.eye(n_dim) * EPS if noise_cov is None else noise_cov
    cov = np.zeros((n_dim, n_dim))
    power = np.eye(n_dim)
    prop = []  # W^k V
    for _ in range(horizon):
        cov += power @ q_step @ power.T
        prop.append(power @ V)
        power = W @ power
    cov_inv = np.linalg.pinv(cov, rcond=1e-14, hermitian=True)
    jk = np.array([float(np.trace(p.T @ cov_inv @ p)) for p in prop])
    return jk * EPS, float(jk.sum() * EPS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def canonical_symplectic(dim: int) -> np.ndarray:
    if dim % 2:
        raise ValueError("symplectic dimension must be even")
    j2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    return np.kron(np.eye(dim // 2, dtype=np.float64), j2)


def haar_orthogonal_symplectic(dim: int, seed: int) -> np.ndarray:
    """Haar draw from O(dim) intersect Sp(dim), in interleaved (q_i,p_i) order."""
    if dim % 2:
        raise ValueError("symplectic dimension must be even")
    half = dim // 2
    gen = np.random.default_rng(seed)
    z = (gen.standard_normal((half, half)) + 1j * gen.standard_normal((half, half))) / np.sqrt(2.0)
    unitary, triangular = np.linalg.qr(z)
    diagonal = np.diag(triangular)
    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0.0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    unitary = unitary * phases

    mixer = np.zeros((dim, dim), dtype=np.float64)
    for row in range(half):
        for col in range(half):
            real = float(unitary[row, col].real)
            imag = float(unitary[row, col].imag)
            mixer[2 * row : 2 * row + 2, 2 * col : 2 * col + 2] = (
                (real, -imag),
                (imag, real),
            )
    return mixer


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonicalize_eigenvector_signs(vectors: np.ndarray) -> np.ndarray:
    canonical = vectors.copy()
    for column in range(canonical.shape[1]):
        pivot = int(np.argmax(np.abs(canonical[:, column])))
        if canonical[pivot, column] < 0.0:
            canonical[:, column] *= -1.0
    return canonical


def fisher_memory_matrix(W: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the symmetrized M_n and C_n using the mirrored covariance path."""
    covariance = np.zeros((N, N), dtype=np.float64)
    power = np.eye(N, dtype=np.float64)
    powers: list[np.ndarray] = []
    for _ in range(horizon):
        covariance += power @ power.T
        powers.append(power)
        power = W @ power
    covariance_inverse = np.linalg.pinv(covariance, rcond=1e-14, hermitian=True)
    matrix = np.zeros((N, N), dtype=np.float64)
    for propagated in powers:
        matrix += propagated.T @ covariance_inverse @ propagated
    return 0.5 * (matrix + matrix.T), covariance


def hamiltonian_exponential(H: np.ndarray) -> np.ndarray:
    """Compute exp(JH) through the real skew-symmetric similarity transform."""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("H must be symmetric positive definite")
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_square_root = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    skew = square_root @ canonical_symplectic(H.shape[0]) @ square_root
    skew = 0.5 * (skew - skew.T)
    hermitian = 1j * skew
    frequencies, modes = np.linalg.eigh(hermitian)
    orthogonal_exponential = (modes * np.exp(-1j * frequencies)) @ modes.conj().T
    imaginary_residual = float(np.max(np.abs(orthogonal_exponential.imag)))
    if imaginary_residual > 1e-10:
        raise RuntimeError(f"skew exponential imaginary residual too large: {imaginary_residual}")
    return inverse_square_root @ orthogonal_exponential.real @ square_root
# END VERBATIM MIRROR: fourth_cell_cond_sweep_20260902.py


# BEGIN VERBATIM MIRROR: verify_ganguli_pocket_capacity_20260823.py
def build_chain(dim: int, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Hidden feedforward chain with super-linear gain G_j = exp(c j^2); v = e_0."""
    W = np.zeros((dim, dim))
    for j in range(dim - 1):
        # G_{j+1}/G_j = exp(c((j+1)^2 - j^2)) = exp(c(2j+1))
        W[j + 1, j] = np.exp(c * (2 * j + 1))
    v = np.zeros((dim, 1))
    v[0, 0] = 1.0
    return W, v


def build_hybrid(d_chain: int, d_pocket: int, c: float, kappa: float, gen):
    """[[S, 0], [K, U]] on (chain, pocket). Input enters the chain head."""
    S, v_chain = build_chain(d_chain, c)
    U = haar_orthogonal(d_pocket, gen)
    K = kappa * gen.standard_normal((d_pocket, d_chain)) / np.sqrt(d_chain)
    n_dim = d_chain + d_pocket
    W = np.zeros((n_dim, n_dim))
    W[:d_chain, :d_chain] = S
    W[d_chain:, :d_chain] = K
    W[d_chain:, d_chain:] = U
    v = np.zeros((n_dim, 1))
    v[:d_chain] = v_chain
    return W, v


def fisher_memory_tv(Ws: list[np.ndarray], V: np.ndarray, Qs: list[np.ndarray]):
    """Time-varying: x_{t+1} = W_t x_t + V s_t + z_t, z_t ~ N(0, Q_t). Returns (Jk, Jtot)."""
    horizon = len(Ws)
    n_dim = Ws[0].shape[0]
    prop = np.eye(n_dim)
    cov = np.zeros((n_dim, n_dim))
    props = []
    for k in range(horizon):  # k = delay; input entered at t = horizon-1-k
        props.append(prop @ V)
        cov += prop @ Qs[horizon - 1 - k] @ prop.T
        prop = prop @ Ws[horizon - 1 - k]
    cov_inv = np.linalg.pinv(cov, rcond=1e-14, hermitian=True)
    jk = np.array([float(np.trace(p.T @ cov_inv @ p)) for p in props])
    return jk * EPS, float(jk.sum() * EPS)


W_WRITE = 24  # write window length, in steps


def gated_run(sink: np.ndarray, horizon: int):
    """Chain -> sink, coupling K live only for the first W_WRITE steps; no direct sink noise."""
    W_open = Wh.copy()
    W_open[D_C:, D_C:] = sink
    W_shut = W_open.copy()
    W_shut[D_C:, :D_C] = 0.0  # door closed
    Ws = [W_open if t < W_WRITE else W_shut for t in range(horizon)]
    Qs = [Qiso for _ in range(horizon)]
    return fisher_memory_tv(Ws, vh, Qs)
# END VERBATIM MIRROR: verify_ganguli_pocket_capacity_20260823.py


# These globals are initialized by replay_20260823_hybrid() before gated_run.
D_C = 16
D_L = 16
NH = D_C + D_L
Wh = np.empty((0, 0), dtype=np.float64)
vh = np.empty((0, 1), dtype=np.float64)
Qiso = np.empty((0, 0), dtype=np.float64)


def seed_for(configuration: str, draw_index: int, stream: str) -> int:
    material = f"{MASTER_SEED}|{configuration}|{draw_index}|{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def make_similarity(condition: float, draw_index: int) -> tuple[np.ndarray, dict[str, object]]:
    label = f"SQS-c{condition:g}"
    seeds = {
        name: seed_for(label, draw_index, name)
        for name in ("s_left", "s_right", "q")
    }
    left = haar_orthogonal(N, np.random.default_rng(seeds["s_left"]))
    right = haar_orthogonal(N, np.random.default_rng(seeds["s_right"]))
    singular_values = np.geomspace(1.0, condition, N).astype(np.float64)
    similarity = left @ np.diag(singular_values) @ right.T
    q = haar_orthogonal(N, np.random.default_rng(seeds["q"]))
    carrier = similarity @ q @ np.linalg.inv(similarity)
    return carrier, {
        "family": "SQS^-1",
        "condition": condition,
        "seeds": seeds,
        "s_condition_observed": float(np.linalg.cond(similarity)),
        "s_sha256_float64_c_order": sha256_array(similarity),
        "q_sha256_float64_c_order": sha256_array(q),
    }


def make_a0(draw_index: int) -> tuple[np.ndarray, dict[str, object]]:
    seed = seed_for("A0", draw_index, "q")
    carrier = haar_orthogonal(N, np.random.default_rng(seed))
    return carrier, {"family": "A0 Haar orthogonal", "condition": 1.0, "seeds": {"q": seed}}


def elliptic_theta_values(condition: float) -> tuple[tuple[str, float], ...]:
    return (("0.7", 0.7), ("2.0", 2.0), ("sqrt_c", float(np.sqrt(condition))))


def make_elliptic(condition: float, theta_name: str, theta: float, draw_index: int) -> tuple[np.ndarray, dict[str, object]]:
    label = f"ELL-c{condition:g}-theta-{theta_name}"
    mixer_seed = seed_for(label, draw_index, "mixer")
    mixer = haar_orthogonal(N, np.random.default_rng(mixer_seed))
    root = np.sqrt(condition)
    rotation = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    scale = np.diag([1.0, root]).astype(np.float64)
    block = np.linalg.inv(scale) @ rotation @ scale
    repeated = np.kron(np.eye(N // 2, dtype=np.float64), block)
    carrier = mixer @ repeated @ mixer.T
    return carrier, {
        "family": "2D elliptic block-repeated and orthogonally mixed",
        "condition": condition,
        "theta_name": theta_name,
        "theta": theta,
        "formula": "W=O blockdiag(D^-1 R(theta) D) O^T; D=diag(1,sqrt(c))",
        "seeds": {"mixer": mixer_seed},
        "mixer_sha256_float64_c_order": sha256_array(mixer),
    }


def make_hamiltonian8(condition: float, draw_index: int) -> tuple[np.ndarray, dict[str, object]]:
    label = f"H8-c{condition:g}"
    eigenbasis_seed = seed_for(label, draw_index, "h_eigenbasis")
    mixer_seed = seed_for(label, draw_index, "symplectic_mixer")
    basis = haar_orthogonal(8, np.random.default_rng(eigenbasis_seed))
    eigenvalues = np.geomspace(1.0, condition, 8).astype(np.float64)
    h_block = basis @ np.diag(eigenvalues) @ basis.T
    w_block = hamiltonian_exponential(h_block)
    block_h = np.kron(np.eye(N // 8, dtype=np.float64), h_block)
    block_w = np.kron(np.eye(N // 8, dtype=np.float64), w_block)
    mixer = haar_orthogonal_symplectic(N, mixer_seed)
    carrier = mixer @ block_w @ mixer.T
    h_full = mixer @ block_h @ mixer.T
    return carrier, {
        "family": "positive-definite Hamiltonian exp(JH), repeated 8x8 blocks",
        "condition": condition,
        "seeds": {"h_eigenbasis": eigenbasis_seed, "symplectic_mixer": mixer_seed},
        "h_eigenvalues": [float(value) for value in eigenvalues],
        "h_condition_observed": float(np.linalg.cond(h_full)),
        "h_sha256_float64_c_order": sha256_array(h_full),
        "mixer_sha256_float64_c_order": sha256_array(mixer),
    }


def configuration_specs(include_heldout: bool) -> list[tuple[str, object]]:
    specs: list[tuple[str, object]] = [("A0", make_a0)]
    for condition in SIMILARITY_CONDITIONS:
        specs.append((f"SQS-c{condition:g}", lambda draw, c=condition: make_similarity(c, draw)))
    if include_heldout:
        specs.append((
            f"SQS-c{HELDOUT_SIMILARITY_CONDITION:g}-heldout",
            lambda draw: make_similarity(HELDOUT_SIMILARITY_CONDITION, draw),
        ))
    for condition in ELLIPTIC_CONDITIONS:
        for theta_name, theta in elliptic_theta_values(condition):
            specs.append((
                f"ELL-c{condition:g}-theta-{theta_name}",
                lambda draw, c=condition, tn=theta_name, t=theta: make_elliptic(c, tn, t, draw),
            ))
    for condition in HAMILTONIAN_CONDITIONS:
        specs.append((f"H8-c{condition:g}", lambda draw, c=condition: make_hamiltonian8(c, draw)))
    return specs


def matrix_readout(carrier: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    matrix, covariance = fisher_memory_matrix(carrier, MATRIX_HORIZON)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvectors = canonicalize_eigenvector_signs(eigenvectors)
    oracle = eigenvectors[:, -1:].copy()
    quadratic = float((oracle.T @ matrix @ oracle).item())
    _, scalar_total = fisher_memory(carrier, oracle, MATRIX_HORIZON)
    return {
        "horizon": MATRIX_HORIZON,
        "lambda_max": float(eigenvalues[-1]),
        "lambda_min": float(eigenvalues[0]),
        "trace_over_N": float(np.trace(matrix) / N),
        "oracle_direction": [float(value) for value in oracle[:, 0]],
        "oracle_direction_sha256_float64_c_order": sha256_array(oracle),
        "oracle_quadratic": quadratic,
        "oracle_scalar_fisher_total": scalar_total,
        "self_consistency_abs_error": max(abs(quadratic - eigenvalues[-1]), abs(scalar_total - eigenvalues[-1])),
        "matrix_sha256_float64_c_order": sha256_array(matrix),
        "covariance_sha256_float64_c_order": sha256_array(covariance),
    }, oracle


def finite_power_certificate(carrier: np.ndarray) -> dict[str, object]:
    inverse = np.linalg.inv(carrier)
    forward_power = np.eye(N, dtype=np.float64)
    inverse_power = np.eye(N, dtype=np.float64)
    k_plus = 1.0
    k_minus = 1.0
    k_plus_argmax = 0
    k_minus_argmax = 0
    all_finite = True
    for k in range(1, POWER_BOUND_HORIZON + 1):
        forward_power = carrier @ forward_power
        inverse_power = inverse @ inverse_power
        forward_norm = float(np.linalg.svd(forward_power, compute_uv=False)[0])
        inverse_norm = float(np.linalg.svd(inverse_power, compute_uv=False)[0])
        if not (np.isfinite(forward_norm) and np.isfinite(inverse_norm)):
            all_finite = False
            break
        if forward_norm > k_plus:
            k_plus = forward_norm
            k_plus_argmax = k
        if inverse_norm > k_minus:
            k_minus = inverse_norm
            k_minus_argmax = k
    return {
        "k_plus_max_k_le_4096": k_plus,
        "k_plus_argmax": k_plus_argmax,
        "k_minus_max_k_le_4096": k_minus,
        "k_minus_argmax": k_minus_argmax,
        "includes_k0_identity": True,
        "all_finite": all_finite,
    }


def oracle_curve_and_theorem(
    carrier: np.ndarray,
    oracle: np.ndarray,
    lambda_max: float,
    certificate: dict[str, object],
) -> dict[str, object]:
    k_plus = float(certificate["k_plus_max_k_le_4096"])
    k_minus = float(certificate["k_minus_max_k_le_4096"])
    by_horizon: dict[str, object] = {}
    total_violations = 0
    flatness_failures = 0
    for horizon in HORIZONS:
        jk, total = fisher_memory(carrier, oracle, horizon)
        lower = 1.0 / (horizon * k_plus * k_plus * k_minus * k_minus)
        upper = (k_plus * k_plus * k_minus * k_minus) / horizon
        tolerance = THEOREM_NUMERIC_ATOL * max(1.0, lower, upper)
        low_bad = jk < lower - tolerance
        high_bad = jk > upper + tolerance
        violations = int(np.count_nonzero(low_bad | high_bad))
        total_violations += violations
        n_j_oldest = float(horizon * jk[-1])
        flatness_relative_error = abs(n_j_oldest - lambda_max) / lambda_max
        flatness_pass = horizon < 512 or flatness_relative_error < FLATNESS_REL_TOL
        if not flatness_pass:
            flatness_failures += 1
        by_horizon[str(horizon)] = {
            "j_total_oracle": float(total),
            "j_oldest": float(jk[-1]),
            "n_times_j_oldest": n_j_oldest,
            "theorem_lower": lower,
            "theorem_upper": upper,
            "theorem_violation_count": violations,
            "minimum_lower_slack": float(np.min(jk - lower)),
            "minimum_upper_slack": float(np.min(upper - jk)),
            "flatness_relative_error_vs_lambda_max_n2048": float(flatness_relative_error),
            "flatness_pass_if_n_ge_512": bool(flatness_pass),
        }
    return {
        "by_horizon": by_horizon,
        "theorem_total_violation_count": total_violations,
        "theorem_numeric_tolerance": THEOREM_NUMERIC_ATOL,
        "flatness_failure_count_n_ge_512": flatness_failures,
        "flatness_threshold_relative": FLATNESS_REL_TOL,
    }


def measure_draw(configuration: str, draw_index: int, maker) -> dict[str, object]:
    carrier, construction = maker(draw_index)
    readout, oracle = matrix_readout(carrier)
    certificate = finite_power_certificate(carrier)
    curve = oracle_curve_and_theorem(carrier, oracle, float(readout["lambda_max"]), certificate)
    j_form = canonical_symplectic(N)
    return {
        "configuration": configuration,
        "draw_index": draw_index,
        "construction": construction,
        "carrier_sha256_float64_c_order": sha256_array(carrier),
        "normality_defect_fro": float(np.linalg.norm(carrier @ carrier.T - carrier.T @ carrier, ord="fro")),
        "symplectic_defect_fro": float(np.linalg.norm(carrier.T @ j_form @ carrier - j_form, ord="fro")),
        "matrix_readout_n2048": readout,
        "power_certificate": certificate,
        "oracle_fmc": curve,
    }


def summary_stats(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25.0)),
        "q3": float(np.percentile(array, 75.0)),
        "iqr": float(np.percentile(array, 75.0) - np.percentile(array, 25.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "draw_count": int(array.size),
    }


def summarize_configuration(configuration: str, draws: list[dict[str, object]]) -> dict[str, object]:
    metrics: dict[str, list[float]] = {
        "lambda_max": [],
        "lambda_min": [],
        "trace_over_N": [],
        "K_plus": [],
        "K_minus": [],
    }
    for horizon in HORIZONS:
        metrics[f"J_oldest_n{horizon}"] = []
        metrics[f"nJ_oldest_n{horizon}"] = []
    for draw in draws:
        readout = draw["matrix_readout_n2048"]
        certificate = draw["power_certificate"]
        metrics["lambda_max"].append(float(readout["lambda_max"]))
        metrics["lambda_min"].append(float(readout["lambda_min"]))
        metrics["trace_over_N"].append(float(readout["trace_over_N"]))
        metrics["K_plus"].append(float(certificate["k_plus_max_k_le_4096"]))
        metrics["K_minus"].append(float(certificate["k_minus_max_k_le_4096"]))
        for horizon in HORIZONS:
            row = draw["oracle_fmc"]["by_horizon"][str(horizon)]
            metrics[f"J_oldest_n{horizon}"].append(float(row["j_oldest"]))
            metrics[f"nJ_oldest_n{horizon}"].append(float(row["n_times_j_oldest"]))
    return {
        "configuration": configuration,
        "draw_count": len(draws),
        "statistics": {name: summary_stats(values) for name, values in metrics.items()},
        "theorem_total_violation_count": sum(
            int(draw["oracle_fmc"]["theorem_total_violation_count"]) for draw in draws
        ),
        "flatness_total_failure_count_n_ge_512": sum(
            int(draw["oracle_fmc"]["flatness_failure_count_n_ge_512"]) for draw in draws
        ),
    }


def exploratory_log_gain_fit(summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    train_labels = [f"SQS-c{condition:g}" for condition in SIMILARITY_CONDITIONS]
    x = np.array([
        np.log(summaries[label]["statistics"]["K_plus"]["median"])
        for label in train_labels
    ], dtype=np.float64)
    y = np.array([
        summaries[label]["statistics"]["lambda_max"]["median"]
        for label in train_labels
    ], dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    heldout_label = f"SQS-c{HELDOUT_SIMILARITY_CONDITION:g}-heldout"
    heldout_gain = float(summaries[heldout_label]["statistics"]["K_plus"]["median"])
    heldout_observed = float(summaries[heldout_label]["statistics"]["lambda_max"]["median"])
    heldout_predicted = float(coefficients[0] + coefficients[1] * np.log(heldout_gain))
    return {
        "status": "EXPLORATORY_NOT_A_PREREGISTERED_READING",
        "fit_target": "configuration-median lambda_max_n2048",
        "fit_predictor": "log(configuration-median K_plus_max_k_le_4096)",
        "training_configurations": train_labels,
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "training_rmse": float(np.sqrt(np.mean((y - fitted) ** 2))),
        "heldout_configuration": heldout_label,
        "heldout_condition": HELDOUT_SIMILARITY_CONDITION,
        "heldout_median_K_plus": heldout_gain,
        "heldout_observed_median_lambda_max": heldout_observed,
        "heldout_predicted_median_lambda_max": heldout_predicted,
        "heldout_absolute_error": abs(heldout_observed - heldout_predicted),
        "no_pass_fail_threshold": True,
    }


def replay_20260823_hybrid() -> np.ndarray:
    global Wh, vh, Qiso
    gen = np.random.default_rng(20260823)
    _q = haar_orthogonal(32, gen)
    _v1 = gen.standard_normal((32, 1))
    _v1 /= np.linalg.norm(_v1)
    for channels in (1, 4, 8):
        _vm = haar_orthogonal(32, gen)[:, :channels]
    Wh, vh = build_hybrid(D_C, D_L, 0.05, kappa=1.0, gen=gen)
    Qiso = np.zeros((NH, NH))
    Qiso[:D_C, :D_C] = EPS * np.eye(D_C)
    return Wh[D_C:, D_C:].copy()


def fisher_memory_tv_varying(
    carriers: list[np.ndarray],
    write_vectors: list[np.ndarray],
    noise_covariances: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Generalization of the mirrored time-varying route to explicit B_t and G_t."""
    horizon = len(carriers)
    if not (len(write_vectors) == len(noise_covariances) == horizon):
        raise ValueError("time-varying sequence lengths differ")
    n_dim = carriers[0].shape[0]
    propagation = np.eye(n_dim, dtype=np.float64)
    covariance = np.zeros((n_dim, n_dim), dtype=np.float64)
    propagated_writes: list[np.ndarray] = []
    for delay in range(horizon):
        time_index = horizon - 1 - delay
        propagated_writes.append(propagation @ write_vectors[time_index])
        covariance += propagation @ noise_covariances[time_index] @ propagation.T
        propagation = propagation @ carriers[time_index]
    covariance_inverse = np.linalg.pinv(covariance, rcond=1e-14, hermitian=True)
    jk = np.array([
        float(np.trace(vector.T @ covariance_inverse @ vector))
        for vector in propagated_writes
    ])
    return jk * EPS, float(jk.sum() * EPS)


def normal_direct_run(sink: np.ndarray, horizon: int, isolated: bool) -> tuple[np.ndarray, float]:
    dimension = sink.shape[0]
    direct = np.zeros((dimension, 1), dtype=np.float64)
    direct[0, 0] = 1.0
    zero_write = np.zeros_like(direct)
    full_noise = EPS * np.eye(dimension, dtype=np.float64)
    zero_noise = np.zeros_like(full_noise)
    carriers = [sink for _ in range(horizon)]
    if isolated:
        writes = [direct if time_index < WRITE_WINDOW else zero_write for time_index in range(horizon)]
        noises = [full_noise if time_index < WRITE_WINDOW else zero_noise for time_index in range(horizon)]
    else:
        writes = [direct for _ in range(horizon)]
        noises = [full_noise for _ in range(horizon)]
    return fisher_memory_tv_varying(carriers, writes, noises)


LEGACY_ANCHORS = {
    "F3": {
        64: (10.607109625121435, 0.09088060263547652),
        256: (10.712709915366927, 0.019741713775965154),
        1024: (10.735837163388616, 0.004763850829566483),
        4096: (10.741385925362444, 0.0011848652403966733),
    },
    "F4": {
        64: (8.81530538455138, 0.3027069703690194),
        256: (8.81530538455135, 0.30270697036900157),
        1024: (8.815305384551351, 0.30270697036899596),
        4096: (8.815305384551252, 0.30270697036900646),
    },
    "F5": {
        64: (9.091822739186345, 0.3026766783633771),
        256: (5.898106994535585, 0.0),
        1024: (5.898106994535585, 0.0),
        4096: (5.898106994535585, 0.0),
    },
}


def factorial_measurement() -> tuple[dict[str, object], list[dict[str, object]]]:
    sink = replay_20260823_hybrid()
    arms: dict[str, object] = {}
    controls: list[dict[str, object]] = []
    for arm in ("F1", "F2", "F3", "F4", "F5"):
        rows: dict[str, object] = {}
        for horizon in FACTORIAL_HORIZONS:
            if arm == "F1":
                jk, total = normal_direct_run(sink, horizon, isolated=False)
            elif arm == "F2":
                jk, total = normal_direct_run(sink, horizon, isolated=True)
            elif arm == "F3":
                jk, total = fisher_memory(Wh, vh, horizon, noise_cov=Qiso)
            elif arm == "F4":
                jk, total = gated_run(sink, horizon)
            else:
                jk, total = gated_run(0.9 * sink, horizon)
            rows[str(horizon)] = {
                "j_total": float(total),
                "j_oldest": float(jk[-1]),
                "n_times_j_oldest": float(horizon * jk[-1]),
                "nonzero_j_count": int(np.count_nonzero(jk)),
            }
            if arm in LEGACY_ANCHORS:
                expected_total, expected_oldest = LEGACY_ANCHORS[arm][horizon]
                error = max(abs(total - expected_total), abs(float(jk[-1]) - expected_oldest))
                controls.append({
                    "name": f"{arm}.reproduce_20260823.n{horizon}",
                    "observed": {"j_total": float(total), "j_oldest": float(jk[-1])},
                    "expected": {"j_total": expected_total, "j_oldest": expected_oldest},
                    "absolute_error_max": float(error),
                    "tolerance": SIX_DECIMAL_ATOL,
                    "pass": bool(error <= SIX_DECIMAL_ATOL),
                })
            if arm == "F1":
                error = max(abs(total - 1.0), abs(float(jk[-1]) - 1.0 / horizon))
                controls.append({
                    "name": f"F1.normal_open_analytic.n{horizon}",
                    "observed": {"j_total": float(total), "j_oldest": float(jk[-1])},
                    "expected": {"j_total": 1.0, "j_oldest": 1.0 / horizon},
                    "absolute_error_max": float(error),
                    "tolerance": SIX_DECIMAL_ATOL,
                    "pass": bool(error <= SIX_DECIMAL_ATOL),
                })
        arms[arm] = {
            "definition": {
                "F1": "normal direct write; sink/noise/write stay open",
                "F2": "normal direct write; write and sink noise both close after 24 steps; isometric storage",
                "F3": "2026-08-23 non-normal chain; write/noise path stays open; isometric sink",
                "F4": "2026-08-23 non-normal chain; coupling closes after 24 steps; isolated isometric sink",
                "F5": "2026-08-23 non-normal chain; coupling closes after 24 steps; isolated 0.9-contracting sink",
            }[arm],
            "readouts": rows,
        }

    f2_totals = [float(arms["F2"]["readouts"][str(h)]["j_total"]) for h in FACTORIAL_HORIZONS]
    f2_oldest = [float(arms["F2"]["readouts"][str(h)]["j_oldest"]) for h in FACTORIAL_HORIZONS]
    total_one = max(abs(value - 1.0) for value in f2_totals) <= FACTORIAL_CONSTANT_ATOL
    oldest_constant = max(f2_oldest) - min(f2_oldest) <= FACTORIAL_CONSTANT_ATOL and min(f2_oldest) > 0.0
    if total_one and oldest_constant:
        reading = "F2_CONSTANT_AT_JTOT_1_RETENTION_INDEPENDENT_OF_WRITE_PATH_NORMALITY"
    elif f2_oldest[-1] < 0.5 * f2_oldest[0]:
        reading = "F2_DECAYS_D1_IS_ALSO_A_RETENTION_CONDITION_REWRITE_V0_2_SECTION_5"
    else:
        reading = "F2_BETWEEN_PREREGISTERED_BRANCHES"
    return {
        "schema": "preprint_write_path_storage_factorial_v1",
        "model": "x_(t+1)=W_t x_t+B_t s_t+G_t z_t",
        "write_window_steps": WRITE_WINDOW,
        "timing_contract": (
            "Open arms F1/F3 remain open for the full horizon. Isolated arms F2/F4/F5 accept writes/noise "
            "through t=0..23 and close the sink path after step 24. F4/F5 retain chain noise after closure, "
            "but it cannot enter the sink."
        ),
        "sink_dimension": 16,
        "horizons": list(FACTORIAL_HORIZONS),
        "seed_replay": {
            "seed": 20260823,
            "draw_order": "Q, v1, three G1b Haar draws (m=1,4,8), then build_hybrid",
        },
        "arms": arms,
        "pre_registered_f2_reading": reading,
        "f2_constant_rule": {
            "max_abs_jtot_minus_1": FACTORIAL_CONSTANT_ATOL,
            "max_minus_min_j_oldest": FACTORIAL_CONSTANT_ATOL,
            "requires_positive_oldest": True,
        },
    }, controls


def control(name: str, observed: object, expected: object, passed: bool, tolerance: object) -> dict[str, object]:
    return {"name": name, "observed": observed, "expected": expected, "tolerance": tolerance, "pass": bool(passed)}


def census_controls(draws_by_configuration: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for configuration, draws in draws_by_configuration.items():
        checks.append(control(
            f"{configuration}.draw_count",
            len(draws),
            DRAW_COUNT,
            len(draws) == DRAW_COUNT,
            "exact",
        ))
        for draw in draws:
            index = int(draw["draw_index"])
            readout = draw["matrix_readout_n2048"]
            trace_error = abs(float(readout["trace_over_N"]) - 1.0)
            checks.append(control(
                f"{configuration}.draw{index}.trace_over_N",
                readout["trace_over_N"],
                1.0,
                trace_error <= TRACE_ATOL,
                TRACE_ATOL,
            ))
            checks.append(control(
                f"{configuration}.draw{index}.oracle_self_consistency",
                readout["self_consistency_abs_error"],
                0.0,
                float(readout["self_consistency_abs_error"]) <= SELF_CONSISTENCY_ATOL,
                SELF_CONSISTENCY_ATOL,
            ))
            violations = int(draw["oracle_fmc"]["theorem_total_violation_count"])
            checks.append(control(
                f"{configuration}.draw{index}.finite_window_theorem",
                violations,
                0,
                violations == 0,
                f"{THEOREM_NUMERIC_ATOL} relative numeric slack",
            ))
            checks.append(control(
                f"{configuration}.draw{index}.power_bounds_finite",
                draw["power_certificate"]["all_finite"],
                True,
                bool(draw["power_certificate"]["all_finite"]),
                "exact boolean",
            ))
            if configuration == "A0":
                spectrum_error = max(
                    abs(float(readout["lambda_max"]) - 1.0),
                    abs(float(readout["lambda_min"]) - 1.0),
                )
                checks.append(control(
                    f"A0.draw{index}.identity_spectrum",
                    spectrum_error,
                    0.0,
                    spectrum_error <= TRACE_ATOL,
                    TRACE_ATOL,
                ))
    return checks


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_plot_ready_tables(summaries: dict[str, dict[str, object]]) -> None:
    summary_rows: list[dict[str, object]] = []
    nj_rows: list[dict[str, object]] = []
    for configuration, summary in summaries.items():
        for metric in ("lambda_max", "lambda_min", "trace_over_N", "K_plus", "K_minus"):
            stats = summary["statistics"][metric]
            summary_rows.append({"configuration": configuration, "metric": metric, **stats})
        for horizon in HORIZONS:
            stats = summary["statistics"][f"nJ_oldest_n{horizon}"]
            nj_rows.append({"configuration": configuration, "horizon": horizon, **stats})
    fields = ["configuration", "metric", "median", "q1", "q3", "iqr", "min", "max", "draw_count"]
    atomic_csv(OUTPUT_DIR / "DIRECTIONAL_CENSUS_SUMMARY.csv", fields, summary_rows)
    nj_fields = ["configuration", "horizon", "median", "q1", "q3", "iqr", "min", "max", "draw_count"]
    atomic_csv(OUTPUT_DIR / "DIRECTIONAL_NJ_TABLE.csv", nj_fields, nj_rows)


def validate_execution_gate() -> tuple[bool, str]:
    if os.environ.get("PREPRINT_REPRODUCTION_WRAPPER_PASSED") != "1":
        return False, "run through scripts/reproduce_public.py (REPRODUCTION)"
    selected = os.environ.get("PREPRINT_REPRODUCTION_REVIEW_FILE", "")
    try:
        if Path(selected).resolve() != REVIEW_PATH.resolve():
            return False, f"wrapper selected wrong review: {selected!r}"
    except OSError:
        return False, f"wrapper selected unreadable review: {selected!r}"
    if not REVIEW_PATH.is_file():
        return False, f"named review absent: {REVIEW_PATH}"
    script_hash = sha256_file(Path(__file__).resolve())
    review_text = REVIEW_PATH.read_text(encoding="utf-8")
    if script_hash not in review_text:
        return False, "named review does not bind this exact script SHA-256"
    decisions = re.findall(r"(?m)^\s*(?:\*\*)?REPRODUCTION DECISION:\s*(RUN|BLOCK)(?:\*\*)?\s*$", review_text)
    if decisions != ["RUN"]:
        return False, f"named review must contain exactly one RUN decision; found {decisions}"
    return True, "named review, exact script SHA-256, and RUN decision bound"


def provenance(started: float) -> dict[str, object]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "cond_sweep_source_path": str(COND_SWEEP_PATH),
        "cond_sweep_source_sha256_expected": COND_SWEEP_SHA256,
        "cond_sweep_source_sha256_observed": sha256_file(COND_SWEEP_PATH),
        "ganguli_source_path": str(GANGULI_PATH),
        "ganguli_source_sha256_expected": GANGULI_SHA256,
        "ganguli_source_sha256_observed": sha256_file(GANGULI_PATH),
        "review_file": os.environ.get("PREPRINT_REPRODUCTION_REVIEW_FILE", "NOT_ESTABLISHED"),
        "numpy_version": np.__version__,
        "float_dtype": "float64",
        "master_seed": MASTER_SEED,
        "elapsed_seconds": float(time.perf_counter() - started),
        "resource_scope": "local CPU only; no GPU, checkpoint, trained weight, network, controller, queue, or roadmap",
    }


def main() -> int:
    started = time.perf_counter()
    gate_ok, gate_detail = validate_execution_gate()
    if not gate_ok:
        print(f"REFUSED: {gate_detail}.")
        return 2
    if sha256_file(COND_SWEEP_PATH) != COND_SWEEP_SHA256:
        print("REFUSED: reviewed fourth-cell conditioning source SHA-256 drifted.")
        return 2
    if sha256_file(GANGULI_PATH) != GANGULI_SHA256:
        print("REFUSED: reviewed 2026-08-23 Ganguli source SHA-256 drifted.")
        return 2

    draws_by_configuration: dict[str, list[dict[str, object]]] = {}
    for configuration, maker in configuration_specs(include_heldout=True):
        print(f"MEASURE {configuration}: {DRAW_COUNT} draws", flush=True)
        draws_by_configuration[configuration] = [
            measure_draw(configuration, draw_index, maker)
            for draw_index in range(DRAW_COUNT)
        ]

    summaries = {
        configuration: summarize_configuration(configuration, draws)
        for configuration, draws in draws_by_configuration.items()
    }
    fit = exploratory_log_gain_fit(summaries)
    factorial, factorial_checks = factorial_measurement()
    checks = census_controls(draws_by_configuration) + factorial_checks
    valid = all(bool(item["pass"]) for item in checks)
    status = "VALID_MEASUREMENT" if valid else "INSTRUMENT_INVALID"

    flatness_failures = sum(
        int(summary["flatness_total_failure_count_n_ge_512"])
        for summary in summaries.values()
    )
    theorem_violations = sum(
        int(summary["theorem_total_violation_count"])
        for summary in summaries.values()
    )
    census_reading = {
        "theorem_check": "ZERO_VIOLATIONS" if theorem_violations == 0 else "INSTRUMENT_INVALID",
        "theorem_violation_count": theorem_violations,
        "oracle_flatness": (
            "FLAT_FMC_OBSERVATION_RETAINED"
            if flatness_failures == 0
            else "FLAT_FMC_OBSERVATION_WITHDRAWN"
        ),
        "oracle_flatness_failure_count_n_ge_512": flatness_failures,
    } if valid else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    census_payload: dict[str, object] = {
        "schema": "preprint_replicated_directional_census_v1",
        "run_status": status,
        "scope": "synthetic N=32 linear carriers; 8 draws/configuration; isotropic in-loop noise; no trained weights or task metric",
        "matrix_horizon": MATRIX_HORIZON,
        "oracle_direction_definition": "top eigenvector of M_2048, fixed and evaluated at every listed horizon",
        "horizons": list(HORIZONS),
        "draw_count_per_configuration": DRAW_COUNT,
        "draws": draws_by_configuration,
        "summaries": summaries,
        "pre_registered_readings": census_reading,
        "exploratory_log_gain_fit": fit,
        "provenance": provenance(started),
    }
    controls_payload: dict[str, object] = {
        "schema": "preprint_replication_and_factorial_positive_controls_v1",
        "run_status": "PASS" if valid else "INSTRUMENT_INVALID",
        "failure_count": sum(1 for item in checks if not item["pass"]),
        "checks": checks,
        "invalid_run_rule": "any trace, self-consistency, finite-bound, theorem, draw-count, legacy-reproduction, or F1 analytic control failure invalidates the full run",
        "provenance": provenance(started),
    }
    factorial["run_status"] = status
    factorial["positive_controls_reference"] = "POSITIVE_CONTROLS.json"
    factorial["provenance"] = provenance(started)
    receipt: dict[str, object] = {
        "schema": "preprint_replication_and_factorial_receipt_v1",
        "run_status": status,
        "execution_gate": gate_detail,
        "census_configuration_count_including_heldout": len(draws_by_configuration),
        "census_draw_count_total": sum(len(draws) for draws in draws_by_configuration.values()),
        "factorial_arms": ["F1", "F2", "F3", "F4", "F5"],
        "factorial_model": factorial["model"],
        "factorial_timing_contract": factorial["timing_contract"],
        "provenance": provenance(started),
    }

    atomic_json(OUTPUT_DIR / "DIRECTIONAL_CENSUS.json", census_payload)
    atomic_json(OUTPUT_DIR / "ISOLATION_FACTORIAL.json", factorial)
    atomic_json(OUTPUT_DIR / "POSITIVE_CONTROLS.json", controls_payload)
    atomic_json(OUTPUT_DIR / "RUN_RECEIPT.json", receipt)
    write_plot_ready_tables(summaries)

    if not valid:
        print("INSTRUMENT_INVALID: one or more preregistered controls failed.")
        return 3
    print(f"PASS: wrote reviewed measurement artifacts under {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
