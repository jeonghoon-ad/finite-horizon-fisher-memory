"""Fourth-cell Fisher-capacity measurement for bounded reversible carriers.

DECISION-BEARING SYNTHETIC MEASUREMENT. Run only through the REPRODUCTION review
wrapper after the required independent pre-review exists. This script reads no
checkpoint and uses no trained artifact. It writes only the two preregistered
JSON files under results/fourth_cell_capacity_20260902/.

The Fisher-memory instrument functions ``haar_orthogonal``, ``fisher_memory``,
and ``shape_stats`` are copied verbatim from
scripts/verify_ganguli_pocket_capacity_20260823.py. Their source relationship
and source SHA-256 are recorded in scripts/MIRROR_OF.json.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


SEED = 20260902
EPS = 1.0
N = 32
HORIZONS = (32, 128, 512, 2048)
FLOOR_HORIZONS = (32, 128, 512, 2048, 4096)
BOUND_HORIZON = 4096
CONTROL_ATOL = 5e-10  # equality to nine digits
SOURCE_SHA256 = "34ff3513574f28a887b1f915d2eee9f4be1a9065bd5c4cbc357db9a892068903"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "scripts" / "verify_ganguli_pocket_capacity_20260823.py"
OUTPUT_DIR = PROJECT_ROOT / "results" / "fourth_cell_capacity_20260902"

# Every random stream is independently seeded and recorded in both outputs.
SEEDS: dict[str, int] = {
    "master": SEED,
    "a0_haar_orthogonal": 2026090201,
    "base_write_vector": 2026090202,
    "symplectic_haar_mixer": 2026090203,
    "a3_s_left": 2026090204,
    "a3_s_right": 2026090205,
    "a3_q": 2026090206,
}
for _arm_index, _arm_name in enumerate(("A1", "A2", "A3"), start=1):
    for _draw in range(5):
        SEEDS[f"{_arm_name.lower()}_null_{_draw + 1}"] = 2026090300 + 10 * _arm_index + _draw


# BEGIN VERBATIM MIRROR: verify_ganguli_pocket_capacity_20260823.py
def haar_orthogonal(dim: int, gen: np.random.Generator) -> np.ndarray:
    a = gen.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


# ----------------------------------------------------------------------------- core


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


# END VERBATIM MIRROR


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


def summarize_nulls(draws: list[dict[str, object]]) -> dict[str, object]:
    by_horizon: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        values = np.array([draw["readouts"]["j_tot"][str(horizon)] for draw in draws], dtype=np.float64)
        by_horizon[str(horizon)] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    floors = np.array([draw["readouts"]["floor_n4096"] for draw in draws], dtype=np.float64)
    labels: dict[str, int] = {}
    for draw in draws:
        label = str(draw["readouts"]["floor_behavior"]["label"])
        labels[label] = labels.get(label, 0) + 1
    return {
        "draw_count": len(draws),
        "j_tot_by_horizon": by_horizon,
        "floor_n4096": {
            "mean": float(floors.mean()),
            "min": float(floors.min()),
            "max": float(floors.max()),
        },
        "floor_behavior_label_counts": labels,
    }


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


def main() -> int:
    started = time.perf_counter()
    if os.environ.get("PREPRINT_REPRODUCTION_WRAPPER_PASSED") != "1":
        print("REFUSED: run through scripts/reproduce_public.py (REPRODUCTION).")
        return 2
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        print("REFUSED: mirrored Fisher instrument source SHA-256 has drifted.")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    j_form = canonical_symplectic(N)
    base_write = unit_vector(SEEDS["base_write_vector"])
    a0 = haar_orthogonal(N, np.random.default_rng(SEEDS["a0_haar_orthogonal"]))
    a0_readouts = memory_readouts(a0, base_write, include_curve=True)
    control_errors = {
        horizon: abs(value - 1.0)
        for horizon, value in a0_readouts["j_tot"].items()
    }
    control_valid = all(error <= CONTROL_ATOL for error in control_errors.values())

    positive_controls: dict[str, object] = {
        "schema": "fourth_cell_positive_controls_v1",
        "run_status": "PASS" if control_valid else "INSTRUMENT_INVALID",
        "invalid_run_rule": {
            "criterion": "A0 J_tot must equal 1.000000000 to nine digits at n=32,128,512,2048",
            "atol": CONTROL_ATOL,
            "maximum_fix_and_rerun": 1,
            "second_failure": "REDESIGN_REQUIRED",
        },
        "A0": {
            "construction": "fixed-seed Haar orthogonal Q",
            "matrix_sha256_float64_c_order": sha256_array(a0),
            "write_vector_sha256_float64_c_order": sha256_array(base_write),
            "certificates": carrier_certificate(a0, j_form),
            "readouts": a0_readouts,
            "absolute_j_tot_errors": control_errors,
            "pass": control_valid,
        },
        "provenance": provenance(started),
    }
    atomic_json(OUTPUT_DIR / "POSITIVE_CONTROLS.json", positive_controls)

    if not control_valid:
        invalid_payload: dict[str, object] = {
            "schema": "fourth_cell_capacity_v1",
            "run_status": "INSTRUMENT_INVALID",
            "measurement_contract": "input(v) -> transform(W, certificates) -> carrier(linear recurrence, isotropic in-loop noise) -> observation(state) -> readout(FMC) -> statistic(J_tot,floor) -> null(write direction) -> decision(capacity>1?,floor constant?)",
            "experimental_arms": {},
            "provenance": provenance(started),
        }
        atomic_json(OUTPUT_DIR / "FOURTH_CELL.json", invalid_payload)
        print("INSTRUMENT_INVALID: A0 failed the preregistered nine-digit sum-rule check.")
        return 3

    mixer = haar_orthogonal_symplectic(N, SEEDS["symplectic_haar_mixer"])
    a1, h1 = symplectic_carrier(4.0, mixer)
    a2, h2 = symplectic_carrier(100.0, mixer)
    a3, similarity, a3_q = similarity_carrier(10.0)
    arm_matrices = {"A1": a1, "A2": a2, "A3": a3}
    construction_metadata: dict[str, dict[str, object]] = {
        "A1": {
            "transport": "W=exp(JH), cond(H)=4",
            "h_condition_number": float(np.linalg.cond(h1)),
            "h_sha256_float64_c_order": sha256_array(h1),
            "mixing": "one shared fixed Haar draw from O(32) intersect Sp(32), preserving canonical J while mixing all 16 blocks",
        },
        "A2": {
            "transport": "W=exp(JH), cond(H)=100",
            "h_condition_number": float(np.linalg.cond(h2)),
            "h_sha256_float64_c_order": sha256_array(h2),
            "mixing": "one shared fixed Haar draw from O(32) intersect Sp(32), preserving canonical J while mixing all 16 blocks",
        },
        "A3": {
            "transport": "W=SQS^-1, cond(S)=10",
            "s_condition_number": float(np.linalg.cond(similarity)),
            "s_sha256_float64_c_order": sha256_array(similarity),
            "q_sha256_float64_c_order": sha256_array(a3_q),
        },
    }

    arms: dict[str, object] = {}
    for arm_name, matrix in arm_matrices.items():
        readouts = memory_readouts(matrix, base_write, include_curve=True)
        null_draws: list[dict[str, object]] = []
        for draw in range(1, 6):
            seed_name = f"{arm_name.lower()}_null_{draw}"
            write = unit_vector(SEEDS[seed_name])
            null_draws.append(
                {
                    "draw": draw,
                    "seed": SEEDS[seed_name],
                    "write_vector_sha256_float64_c_order": sha256_array(write),
                    "readouts": memory_readouts(matrix, write, include_curve=False),
                }
            )
        arms[arm_name] = {
            "construction": construction_metadata[arm_name],
            "matrix_sha256_float64_c_order": sha256_array(matrix),
            "write_vector_sha256_float64_c_order": sha256_array(base_write),
            "certificates": carrier_certificate(matrix, j_form),
            "readouts": readouts,
            "decision_readout": {
                "capacity_minus_one_by_horizon": {
                    horizon: float(value - 1.0)
                    for horizon, value in readouts["j_tot"].items()
                },
                "capacity_gt_one_raw_by_horizon": {
                    horizon: bool(value > 1.0)
                    for horizon, value in readouts["j_tot"].items()
                },
                "capacity_gt_one_to_nine_digits_by_horizon": {
                    horizon: bool(round(value, 9) > 1.0)
                    for horizon, value in readouts["j_tot"].items()
                },
                "floor_constant": readouts["floor_behavior"]["label"] == "constant",
                "floor_behavior": readouts["floor_behavior"]["label"],
                "scope": "measurement readout only; no track verdict",
            },
            "null_write_direction_draws": null_draws,
            "null_summary_mean_and_range": summarize_nulls(null_draws),
        }

    capacity_payload: dict[str, object] = {
        "schema": "fourth_cell_capacity_v1",
        "run_status": "VALID_MEASUREMENT",
        "scope": "synthetic N=32 linear carriers; one scalar unit-norm write channel; isotropic in-loop noise; Fisher memory only; no track verdict",
        "measurement_contract": "input(v) -> transform(W, certified normal/non-normal/symplectic/bounded) -> carrier(linear recurrence with isotropic in-loop noise, G1 path) -> observation(state) -> readout(FMC) -> statistic(J_tot,floor) -> null(write-direction draws) -> decision(capacity>1?,floor constant?)",
        "decision_axis": ["capacity greater than 1", "floor behavior"],
        "pre_registered_outcome_readings": {
            "all_equal_1": "fourth cell empty for these linear carriers; any new general sentence requires proof",
            "greater_than_1_constant_floor": "fourth cell real",
            "greater_than_1_one_over_n_floor": "capacity without persistence; D3 gate still needed",
            "a3_differs_in_kind_from_a1_a2": "symplecticity and boundedness separated; direction must follow measured contrast",
        },
        "A0_reference": {
            "file": "POSITIVE_CONTROLS.json",
            "pass": True,
            "j_tot": a0_readouts["j_tot"],
        },
        "experimental_arms": arms,
        "provenance": provenance(started),
    }
    atomic_json(OUTPUT_DIR / "FOURTH_CELL.json", capacity_payload)

    # Refresh elapsed time after the larger result has been serialized.
    positive_controls["provenance"] = provenance(started)
    capacity_payload["provenance"] = provenance(started)
    atomic_json(OUTPUT_DIR / "POSITIVE_CONTROLS.json", positive_controls)
    atomic_json(OUTPUT_DIR / "FOURTH_CELL.json", capacity_payload)
    print(f"PASS: wrote {OUTPUT_DIR / 'POSITIVE_CONTROLS.json'}")
    print(f"PASS: wrote {OUTPUT_DIR / 'FOURTH_CELL.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
