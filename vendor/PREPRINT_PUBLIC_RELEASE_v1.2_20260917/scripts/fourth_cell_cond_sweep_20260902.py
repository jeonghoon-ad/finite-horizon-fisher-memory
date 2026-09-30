"""Conditioning sweep for bounded reversible fourth-cell carriers.

DECISION-BEARING SYNTHETIC MEASUREMENT. Run only through the REPRODUCTION review
wrapper after the required independent pre-review exists. This script reads no
checkpoint and writes only the preregistered JSON artifacts under
results/fourth_cell_cond_sweep_20260902/.

The established instrument and provenance helpers are copied verbatim from the
reviewed scripts/fourth_cell_capacity_20260902.py. The new Fisher-matrix route
wraps, and does not alter, the mirrored scalar Fisher-memory route.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


SEED = 20260903
EPS = 1.0
N = 32
HORIZONS = (32, 128, 512, 2048)
FLOOR_HORIZONS = (32, 128, 512, 2048, 4096)
MATRIX_HORIZONS = (128, 2048)
BOUND_HORIZON = 4096
TRACE_ATOL = 5e-10
SELF_CONSISTENCY_ATOL = 1e-9
CONTROL_SIX_DECIMAL_ATOL = 5e-7
SYMPLECTIC_ATOL = 1e-9
SOURCE_SHA256 = "8f2bb0a0bed1597e142114920ea9d655e0b3892651df87854c4caeefcec5fc0e"
S10_MATRIX_SHA256 = "d62b8bbcdc051c2de33a6a11c0f35583830c77742efb475a081190062effe1b2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "scripts" / "fourth_cell_capacity_20260902.py"
OUTPUT_DIR = PROJECT_ROOT / "results" / "fourth_cell_cond_sweep_20260902"

SIMILARITY_CONDITIONS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
SYMPLECTIC_CONDITIONS = (4.0, 100.0)

# The inherited streams reproduce A0, the base write vector, S-10, and P2.
# The new streams are separately seeded from the follow-up's master seed.
SEEDS: dict[str, int] = {
    "master": SEED,
    "a0_haar_orthogonal": 2026090201,
    "base_write_vector": 2026090202,
    "symplectic_haar_mixer": 2026090203,
    "a3_s_left": 2026090204,
    "a3_s_right": 2026090205,
    "a3_q": 2026090206,
    "p4_h_eigenbasis": 2026090301,
    "p8_h_eigenbasis": 2026090302,
}


# BEGIN VERBATIM MIRROR: fourth_cell_capacity_20260902.py
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


def shape_stats(jk: np.ndarray) -> dict[str, float]:
    total = jk.sum()
    if total <= 0:
        return {"tau": 0.0, "persist_half": 0.0, "oldest": 0.0}
    k = np.arange(len(jk))
    return {
        "tau": float((k * jk).sum() / total),
        "persist_half": float(jk[len(jk) // 2 :].sum() / total),
        "oldest": float(jk[-1]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def unit_vector(seed: int) -> np.ndarray:
    gen = np.random.default_rng(seed)
    vector = gen.standard_normal((N, 1)).astype(np.float64)
    return vector / np.linalg.norm(vector)


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


def symplectic_carrier(condition_number: float, mixer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return W=exp(JH) and H, with H SPD and cond(H)=condition_number."""
    root = np.sqrt(condition_number)
    cosine = np.cos(root)
    sine = np.sin(root)
    block = np.array(
        [[cosine, root * sine], [-sine / root, cosine]],
        dtype=np.float64,
    )
    h_block = np.diag([1.0, condition_number]).astype(np.float64)
    block_w = np.kron(np.eye(N // 2, dtype=np.float64), block)
    block_h = np.kron(np.eye(N // 2, dtype=np.float64), h_block)
    return mixer @ block_w @ mixer.T, mixer @ block_h @ mixer.T


def similarity_carrier(condition_number: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = haar_orthogonal(N, np.random.default_rng(SEEDS["a3_s_left"]))
    right = haar_orthogonal(N, np.random.default_rng(SEEDS["a3_s_right"]))
    singular_values = np.geomspace(1.0, condition_number, N).astype(np.float64)
    similarity = left @ np.diag(singular_values) @ right.T
    q = haar_orthogonal(N, np.random.default_rng(SEEDS["a3_q"]))
    carrier = similarity @ q @ np.linalg.inv(similarity)
    return carrier, similarity, q


def carrier_certificate(W: np.ndarray, J: np.ndarray) -> dict[str, object]:
    normality_defect = float(np.linalg.norm(W @ W.T - W.T @ W, ord="fro"))
    symplectic_defect = float(np.linalg.norm(W.T @ J @ W - J, ord="fro"))
    power = np.eye(W.shape[0], dtype=np.float64)
    maximum = 0.0
    maximum_k = 0
    all_finite = True
    for k in range(1, BOUND_HORIZON + 1):
        power = W @ power
        sigma_max = float(np.linalg.svd(power, compute_uv=False)[0])
        if not np.isfinite(sigma_max):
            all_finite = False
            maximum = sigma_max
            maximum_k = k
            break
        if sigma_max > maximum:
            maximum = sigma_max
            maximum_k = k
    return {
        "is_normal": normality_defect,
        "normality_defect_fro": normality_defect,
        "is_symplectic": symplectic_defect,
        "symplectic_defect_fro": symplectic_defect,
        "bounded": maximum,
        "bounded_max_sigma_wk": maximum,
        "bounded_argmax_k": maximum_k,
        "bounded_all_finite": all_finite,
        "bounded_k_range": [1, BOUND_HORIZON],
    }


def floor_behavior(floors: dict[int, float]) -> dict[str, object]:
    """Operationalize the 08-23 note's four qualitative floor labels.

    The note distinguishes exact zero, constant, 1/n, and exponential behavior.
    This pre-run implementation chooses the nonzero label with the smallest RMS
    residual in log space; it does not tune a post-result threshold.
    """
    horizons = np.array(sorted(floors), dtype=np.float64)
    values = np.array([floors[int(h)] for h in horizons], dtype=np.float64)
    if values[-1] == 0.0:
        return {
            "label": "zero",
            "rule": "J(n-1) at n=4096 is exactly zero",
            "log_rms_residuals": None,
        }
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        return {
            "label": "NOT_ESTABLISHED",
            "rule": "positive finite floor values are required for log-scale classification",
            "log_rms_residuals": None,
        }

    log_values = np.log(values)

    constant_fit = np.full_like(log_values, log_values.mean())
    reciprocal_level = (log_values + np.log(horizons)).mean()
    reciprocal_fit = reciprocal_level - np.log(horizons)
    design = np.column_stack((np.ones_like(horizons), horizons))
    exp_coefficients, *_ = np.linalg.lstsq(design, log_values, rcond=None)
    exponential_fit = design @ exp_coefficients

    residuals = {
        "constant": float(np.sqrt(np.mean((log_values - constant_fit) ** 2))),
        "1/n": float(np.sqrt(np.mean((log_values - reciprocal_fit) ** 2))),
        "exponential": float(np.sqrt(np.mean((log_values - exponential_fit) ** 2))),
    }
    label = min(residuals, key=residuals.get)
    return {
        "label": label,
        "rule": "minimum log-space RMS residual across constant, c/n, and c*exp(b*n); exact zero handled first",
        "log_rms_residuals": residuals,
        "exponential_log_slope": float(exp_coefficients[1]),
    }


def memory_readouts(W: np.ndarray, V: np.ndarray, include_curve: bool) -> dict[str, object]:
    totals: dict[str, float] = {}
    floors: dict[int, float] = {}
    full_curve: list[float] | None = None
    full_stats: dict[str, float] | None = None
    for horizon in FLOOR_HORIZONS:
        jk, jtot = fisher_memory(W, V, horizon)
        floors[horizon] = float(jk[-1])
        if horizon in HORIZONS:
            totals[str(horizon)] = float(jtot)
        if horizon == BOUND_HORIZON:
            full_curve = [float(value) for value in jk]
            full_stats = shape_stats(jk)
    result: dict[str, object] = {
        "j_tot": totals,
        "floor_n4096": floors[BOUND_HORIZON],
        "floor_by_horizon": {str(k): v for k, v in floors.items()},
        "floor_behavior": floor_behavior(floors),
        "shape_n4096": full_stats,
    }
    if include_curve:
        result["fmc_jk_k_le_4095"] = full_curve
    return result


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def provenance(started: float) -> dict[str, object]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_path": str(SOURCE_PATH),
        "source_sha256_expected": SOURCE_SHA256,
        "source_sha256_observed": sha256_file(SOURCE_PATH),
        "numpy_version": np.__version__,
        "float_dtype": "float64",
        "elapsed_seconds": float(time.perf_counter() - started),
        "seeds": SEEDS,
        "review_file": os.environ.get("PREPRINT_REPRODUCTION_REVIEW_FILE", "NOT_ESTABLISHED"),
    }


# END VERBATIM MIRROR


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


def fisher_matrix_readout(W: np.ndarray, horizon: int) -> tuple[dict[str, object], np.ndarray]:
    matrix, covariance = fisher_memory_matrix(W, horizon)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvectors = canonicalize_eigenvector_signs(eigenvectors)
    best = eigenvectors[:, -1:].copy()
    lambda_max = float(eigenvalues[-1])
    quadratic = float((best.T @ matrix @ best).item())
    _, mirrored_total = fisher_memory(W, best, horizon)
    payload: dict[str, object] = {
        "horizon": horizon,
        "lambda_max": lambda_max,
        "lambda_min": float(eigenvalues[0]),
        "trace_over_N": float(np.trace(matrix) / N),
        "eigenvalues_ascending": [float(value) for value in eigenvalues],
        "eigenvectors_columns": [[float(value) for value in row] for row in eigenvectors],
        "v_best": [float(value) for value in best[:, 0]],
        "v_best_norm": float(np.linalg.norm(best)),
        "v_best_quadratic": quadratic,
        "v_best_mirrored_fisher_total": float(mirrored_total),
        "self_consistency_abs_error_matrix": abs(quadratic - lambda_max),
        "self_consistency_abs_error_mirrored": abs(mirrored_total - lambda_max),
        "matrix_sha256_float64_c_order": sha256_array(matrix),
        "covariance_sha256_float64_c_order": sha256_array(covariance),
    }
    return payload, best


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


def higher_block_symplectic_carrier(
    block_size: int,
    condition_number: float,
    eigenbasis_seed: int,
    mixer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    basis = haar_orthogonal(block_size, np.random.default_rng(eigenbasis_seed))
    h_eigenvalues = np.geomspace(1.0, condition_number, block_size).astype(np.float64)
    h_block = basis @ np.diag(h_eigenvalues) @ basis.T
    w_block = hamiltonian_exponential(h_block)
    repetitions = N // block_size
    block_h = np.kron(np.eye(repetitions, dtype=np.float64), h_block)
    block_w = np.kron(np.eye(repetitions, dtype=np.float64), w_block)
    return mixer @ block_w @ mixer.T, mixer @ block_h @ mixer.T


def base_vector_totals(W: np.ndarray, base_write: np.ndarray) -> dict[str, float]:
    totals: dict[str, float] = {}
    for horizon in MATRIX_HORIZONS:
        _, total = fisher_memory(W, base_write, horizon)
        totals[str(horizon)] = float(total)
    return totals


def measure_arm(
    name: str,
    W: np.ndarray,
    construction: dict[str, object],
    base_write: np.ndarray,
    j_form: np.ndarray,
) -> dict[str, object]:
    matrix_readouts: dict[str, object] = {}
    best_by_horizon: dict[int, np.ndarray] = {}
    for horizon in MATRIX_HORIZONS:
        payload, best = fisher_matrix_readout(W, horizon)
        matrix_readouts[str(horizon)] = payload
        best_by_horizon[horizon] = best
    best_reference = best_by_horizon[2048]
    return {
        "name": name,
        "construction": construction,
        "matrix_sha256_float64_c_order": sha256_array(W),
        "certificates": carrier_certificate(W, j_form),
        "matrix_readouts": matrix_readouts,
        "base_vector_j_tot": base_vector_totals(W, base_write),
        "base_write_vector_sha256_float64_c_order": sha256_array(base_write),
        "v_best_reference_horizon": 2048,
        "v_best_fmc": memory_readouts(W, best_reference, include_curve=True),
    }


def control_record(name: str, observed: object, expected: object, ok: bool, tolerance: object) -> dict[str, object]:
    return {
        "name": name,
        "observed": observed,
        "expected": expected,
        "tolerance": tolerance,
        "pass": bool(ok),
    }


def positive_control_checks(arms: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for horizon in MATRIX_HORIZONS:
        a0_readout = arms["A0"]["matrix_readouts"][str(horizon)]
        spectrum_error = max(abs(value - 1.0) for value in a0_readout["eigenvalues_ascending"])
        checks.append(control_record(
            f"A0.spectrum_all_ones.n{horizon}",
            spectrum_error,
            0.0,
            spectrum_error <= TRACE_ATOL,
            TRACE_ATOL,
        ))

    for arm_name, arm in arms.items():
        for horizon in MATRIX_HORIZONS:
            readout = arm["matrix_readouts"][str(horizon)]
            trace_error = abs(readout["trace_over_N"] - 1.0)
            self_error = max(
                readout["self_consistency_abs_error_matrix"],
                readout["self_consistency_abs_error_mirrored"],
            )
            checks.append(control_record(
                f"{arm_name}.trace_over_N.n{horizon}",
                readout["trace_over_N"],
                1.0,
                trace_error <= TRACE_ATOL,
                TRACE_ATOL,
            ))
            checks.append(control_record(
                f"{arm_name}.v_best_self_consistency.n{horizon}",
                self_error,
                0.0,
                self_error <= SELF_CONSISTENCY_ATOL,
                SELF_CONSISTENCY_ATOL,
            ))

    s10 = arms["S-10"]
    s10_lambda = s10["matrix_readouts"]["2048"]["lambda_max"]
    checks.append(control_record(
        "S-10.matrix_reproduces_reviewed_A3",
        s10["matrix_sha256_float64_c_order"],
        S10_MATRIX_SHA256,
        s10["matrix_sha256_float64_c_order"] == S10_MATRIX_SHA256,
        "exact SHA-256",
    ))
    checks.append(control_record(
        "S-10.lambda_max.n2048",
        s10_lambda,
        4.212278,
        abs(s10_lambda - 4.212278) <= CONTROL_SIX_DECIMAL_ATOL,
        CONTROL_SIX_DECIMAL_ATOL,
    ))

    for condition in SYMPLECTIC_CONDITIONS:
        arm_name = f"P2-{int(condition)}"
        observed = arms[arm_name]["matrix_readouts"]["2048"]["lambda_max"]
        expected = 2.0 * condition / (condition + 1.0)
        checks.append(control_record(
            f"{arm_name}.closed_form_lambda_max.n2048",
            observed,
            expected,
            abs(observed - expected) <= CONTROL_SIX_DECIMAL_ATOL,
            CONTROL_SIX_DECIMAL_ATOL,
        ))

    for arm_name, arm in arms.items():
        certificates = arm["certificates"]
        checks.append(control_record(
            f"{arm_name}.bounded_all_finite",
            certificates["bounded_all_finite"],
            True,
            bool(certificates["bounded_all_finite"]),
            "exact boolean",
        ))
        if arm_name.startswith("P"):
            defect = certificates["symplectic_defect_fro"]
            checks.append(control_record(
                f"{arm_name}.symplectic_defect",
                defect,
                "<1e-9",
                defect < SYMPLECTIC_ATOL,
                SYMPLECTIC_ATOL,
            ))
    return checks


def preregistered_readings(arms: dict[str, dict[str, object]]) -> dict[str, object]:
    similarity_lambdas = [
        float(arms[f"S-{int(condition)}"]["matrix_readouts"]["2048"]["lambda_max"])
        for condition in SIMILARITY_CONDITIONS
    ]
    monotonic = all(
        later >= earlier - SELF_CONSISTENCY_ATOL
        for earlier, later in zip(similarity_lambdas, similarity_lambdas[1:])
    )
    exceeds_eight = similarity_lambdas[-1] > 8.0
    saturates_below_five = max(similarity_lambdas) < 5.0
    if monotonic and exceeds_eight:
        q1_growth = "KNOB_REAL_CONTROLLED_BY_CONDITIONING"
    elif saturates_below_five:
        q1_growth = "KNOB_WEAK_SATURATES_BELOW_5"
    else:
        q1_growth = "BETWEEN_PREREGISTERED_BRANCHES"

    similarity_floor_labels = {
        f"S-{int(condition)}": arms[f"S-{int(condition)}"]["v_best_fmc"]["floor_behavior"]["label"]
        for condition in SIMILARITY_CONDITIONS
    }
    if any(label == "constant" for label in similarity_floor_labels.values()):
        q1_floor = "CONSTANT_FLOOR_FOUND_REPORT_FIRST"
    elif all(label == "1/n" for label in similarity_floor_labels.values()):
        q1_floor = "CAPACITY_WITHOUT_PERSISTENCE_D3_STILL_NEEDED"
    else:
        q1_floor = "BETWEEN_PREREGISTERED_BRANCHES"

    larger_block_lambdas = {
        f"P{block_size}-{int(condition)}": float(
            arms[f"P{block_size}-{int(condition)}"]["matrix_readouts"]["2048"]["lambda_max"]
        )
        for block_size in (4, 8)
        for condition in SYMPLECTIC_CONDITIONS
    }
    if all(value <= 2.0 + 1e-6 for value in larger_block_lambdas.values()):
        q2 = "CAP_AT_2_SURVIVES_BLOCK_SIZE_SWEEP"
    elif any(larger_block_lambdas[f"P{block_size}-100"] > 2.0 for block_size in (4, 8)):
        q2 = "CAP_BELONGS_TO_2X2_BLOCKS_ONLY"
    else:
        q2 = "BETWEEN_PREREGISTERED_BRANCHES"

    return {
        "Q1a_growth": {
            "reading": q1_growth,
            "conditions": [float(value) for value in SIMILARITY_CONDITIONS],
            "lambda_max_n2048": similarity_lambdas,
            "monotonic_non_decreasing": monotonic,
            "c100_exceeds_8": exceeds_eight,
            "all_below_5": saturates_below_five,
        },
        "Q1b_floor": {
            "reading": q1_floor,
            "labels": similarity_floor_labels,
        },
        "Q2_block_size": {
            "reading": q2,
            "lambda_max_n2048": larger_block_lambdas,
        },
    }


def main() -> int:
    started = time.perf_counter()
    if os.environ.get("PREPRINT_REPRODUCTION_WRAPPER_PASSED") != "1":
        print("REFUSED: run through scripts/reproduce_public.py (REPRODUCTION).")
        return 2
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        print("REFUSED: reviewed fourth-cell source SHA-256 has drifted.")
        return 2

    j_form = canonical_symplectic(N)
    base_write = unit_vector(SEEDS["base_write_vector"])
    mixer = haar_orthogonal_symplectic(N, SEEDS["symplectic_haar_mixer"])
    arm_specs: list[tuple[str, np.ndarray, dict[str, object]]] = []

    a0 = haar_orthogonal(N, np.random.default_rng(SEEDS["a0_haar_orthogonal"]))
    arm_specs.append(("A0", a0, {"family": "Haar orthogonal", "purpose": "positive control"}))

    for condition in SIMILARITY_CONDITIONS:
        W, similarity, q = similarity_carrier(condition)
        arm_specs.append((
            f"S-{int(condition)}",
            W,
            {
                "family": "similarity-to-orthogonal",
                "formula": "W=SQS^-1",
                "condition_number_target": condition,
                "condition_number_observed": float(np.linalg.cond(similarity)),
                "s_sha256_float64_c_order": sha256_array(similarity),
                "q_sha256_float64_c_order": sha256_array(q),
                "shared_seeds": ["a3_s_left", "a3_s_right", "a3_q"],
            },
        ))

    for condition in SYMPLECTIC_CONDITIONS:
        W, H = symplectic_carrier(condition, mixer)
        arm_specs.append((
            f"P2-{int(condition)}",
            W,
            {
                "family": "symplectic repeated 2x2 blocks",
                "formula": "W=exp(JH)",
                "block_size": 2,
                "block_count": 16,
                "h_condition_number": float(np.linalg.cond(H)),
                "h_sha256_float64_c_order": sha256_array(H),
            },
        ))

    for block_size, seed_key in ((4, "p4_h_eigenbasis"), (8, "p8_h_eigenbasis")):
        for condition in SYMPLECTIC_CONDITIONS:
            W, H = higher_block_symplectic_carrier(block_size, condition, SEEDS[seed_key], mixer)
            arm_specs.append((
                f"P{block_size}-{int(condition)}",
                W,
                {
                    "family": f"symplectic repeated {block_size}x{block_size} blocks",
                    "formula": "W=exp(JH)",
                    "block_size": block_size,
                    "block_count": N // block_size,
                    "h_eigenvalues": [float(value) for value in np.geomspace(1.0, condition, block_size)],
                    "h_condition_number": float(np.linalg.cond(H)),
                    "h_eigenbasis_seed": SEEDS[seed_key],
                    "h_sha256_float64_c_order": sha256_array(H),
                },
            ))

    if len(arm_specs) != 13:
        raise RuntimeError(f"preregistered arm count mismatch: {len(arm_specs)}")

    arms: dict[str, dict[str, object]] = {}
    for name, W, construction in arm_specs:
        arms[name] = measure_arm(name, W, construction, base_write, j_form)

    checks = positive_control_checks(arms)
    valid = all(check["pass"] for check in checks)
    run_status = "VALID_MEASUREMENT" if valid else "INSTRUMENT_INVALID"
    readings = preregistered_readings(arms) if valid else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    controls_payload: dict[str, object] = {
        "schema": "fourth_cell_cond_sweep_positive_controls_v1",
        "run_status": "PASS" if valid else "INSTRUMENT_INVALID",
        "invalid_run_rule": {
            "criterion": "A0 spectrum, trace/N, S-10, P2 closed form, self-consistency, boundedness, and symplectic certificates must all pass",
            "maximum_fix_and_rerun": 1,
            "second_failure": "REDESIGN_REQUIRED",
        },
        "checks": checks,
        "failure_count": sum(1 for check in checks if not check["pass"]),
        "provenance": provenance(started),
    }
    result_payload: dict[str, object] = {
        "schema": "fourth_cell_cond_sweep_v1",
        "run_status": run_status,
        "scope": "synthetic N=32 linear carriers; isotropic in-loop noise; Fisher memory matrix and scalar FMC; no trained weights, task metric, or track verdict",
        "questions": {
            "Q1": "best-direction Fisher capacity versus cond(S), and its long-delay floor",
            "Q2": "whether 4x4 or 8x8 symplectic blocks exceed the 2x2 cap of 2",
        },
        "best_direction_reference_horizon": 2048,
        "arms": arms,
        "positive_controls_reference": "POSITIVE_CONTROLS.json",
        "pre_registered_readings": readings,
        "provenance": provenance(started),
    }
    atomic_json(OUTPUT_DIR / "POSITIVE_CONTROLS.json", controls_payload)
    atomic_json(OUTPUT_DIR / "FOURTH_CELL_COND_SWEEP.json", result_payload)

    controls_payload["provenance"] = provenance(started)
    result_payload["provenance"] = provenance(started)
    atomic_json(OUTPUT_DIR / "POSITIVE_CONTROLS.json", controls_payload)
    atomic_json(OUTPUT_DIR / "FOURTH_CELL_COND_SWEEP.json", result_payload)
    if not valid:
        print("INSTRUMENT_INVALID: one or more preregistered positive controls failed.")
        return 3
    print(f"PASS: wrote {OUTPUT_DIR / 'POSITIVE_CONTROLS.json'}")
    print(f"PASS: wrote {OUTPUT_DIR / 'FOURTH_CELL_COND_SWEEP.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
