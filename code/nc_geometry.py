"""Geometry core for the Neural Computation trained-memory validation lane (NC-TM-V1).

Authority (read-only, under ../authority/):
  NEURAL_COMPUTATION_TRAINED_MEMORY_VALIDATION_PREREG_v1_20260918.yaml   (machine authority)
  NEURAL_COMPUTATION_TRAINED_MEMORY_VALIDATION_PROTOCOL_v1_20260918_EN.md (scientific authority)
  PREPRINT_PUBLIC_BUILD_v1.2_20260917_EN.md                               (theorem context)

Every numerical helper that already exists in the frozen v1.2 public release is
IMPORTED from it, never reimplemented.  See MIRROR_OF.json for the exact source
files, their SHA-256 digests, and the list of symbols consumed.

Index convention, established against the release and asserted at runtime:
  closure_transfers(W, K, U, steps)[j]  <->  input injected at t = steps - 1 - j
  therefore transfers[0]        == 0        (t = T_w - 1 is never read)
            transfers[T_w - 1]  == L_0      (t = 0, the oldest input, the NC target)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Frozen protocol constants.  These come from the YAML preregistration and may
# not be changed by any downstream module.
# --------------------------------------------------------------------------
WRITE_DIM = 32           # system.writer_dim
STORE_DIM = 16           # system.store_dim
WRITE_WINDOW = 24        # system.write_interval  (T_w)
TARGET_TIME = 0          # system.target_time     (t_0)
CONDITION = 10.0         # kappa(S) for the bi-power-bounded non-normal writer
MATRIX_HORIZON = 2048    # the n in M_write = M_2048 (write-block operator)

SEED_NAMESPACE = "NC-TM-V1"

# Confirmatory carrier draws are 0..15.  Pilot draws live in a disjoint,
# deliberately far-away block so that no pilot draw can ever be mistaken for a
# confirmatory one (preregistration item pilot.excluded_from_confirmation).
CONFIRMATORY_DRAWS = tuple(range(16))
PILOT_DRAWS = (900, 901)
# The exploratory arms get their own block too, so that running them can never
# inspect a confirmatory carrier before the confirmatory stage has run.
EXPLORATORY_DRAWS = tuple(range(800, 808))

WRITER_TYPES = ("normal", "nonnormal")

LANE_ROOT = Path(__file__).resolve().parents[1]
VENDOR = LANE_ROOT / "vendor" / "PREPRINT_PUBLIC_RELEASE_v1.2_20260917"
VENDOR_SCRIPTS = VENDOR / "scripts"

# SHA-256 of the frozen release scripts we import from.  Taken from the release
# MANIFEST.json shipped inside the handoff package; verified on every load.
VENDOR_EXPECTED_SHA256 = {
    "preprint_replication_and_factorial_20260903.py":
        "f8c0884d41cf7a112468013dee6ad70702a7a7be4b6ee700d2100d128aaa2ad4",
    "isolation_factorial_v2_bounded_20260903.py":
        "5aa11fa435d85883f9b9f16521abc1ec58fc2514e8022ffbc6d3294c435a2ccb",
    "preprint_v1_2_strengthening_20260917.py":
        "c1fd328db9cea8677f74611d96d1cce9a09b673a096da7098aaba249ab7fa7bc",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VENDOR_CACHE: dict[str, ModuleType] | None = None


def load_vendor(verify: bool = True) -> dict[str, ModuleType]:
    """Import the frozen release modules, verifying their bytes first."""
    global _VENDOR_CACHE
    if _VENDOR_CACHE is not None:
        return _VENDOR_CACHE
    if verify:
        for filename, expected in VENDOR_EXPECTED_SHA256.items():
            observed = sha256_file(VENDOR_SCRIPTS / filename)
            if observed != expected:
                raise RuntimeError(
                    f"frozen release integrity failure for {filename}: "
                    f"expected {expected}, observed {observed}"
                )
    modules = {
        "source": _load_module(
            "nc_vendor_source", VENDOR_SCRIPTS / "preprint_replication_and_factorial_20260903.py"
        ),
        "iso": _load_module(
            "nc_vendor_iso", VENDOR_SCRIPTS / "isolation_factorial_v2_bounded_20260903.py"
        ),
        "strengthening": _load_module(
            "nc_vendor_strengthening", VENDOR_SCRIPTS / "preprint_v1_2_strengthening_20260917.py"
        ),
    }
    _VENDOR_CACHE = modules
    return modules


def vendor_digests() -> dict[str, str]:
    return {name: sha256_file(VENDOR_SCRIPTS / name) for name in VENDOR_EXPECTED_SHA256}


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------
DRAW_BLOCKS = {
    "confirmatory": CONFIRMATORY_DRAWS,
    "pilot": PILOT_DRAWS,
    "exploratory": EXPLORATORY_DRAWS,
}


def draw_block_collisions() -> list[tuple[str, str, list[int]]]:
    """Every pairwise intersection of the three reserved draw blocks."""
    out: list[tuple[str, str, list[int]]] = []
    names = sorted(DRAW_BLOCKS)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = sorted(set(DRAW_BLOCKS[first]) & set(DRAW_BLOCKS[second]))
            if shared:
                out.append((first, second, shared))
    return out


def assert_draw_blocks_disjoint() -> None:
    collisions = draw_block_collisions()
    if collisions:
        raise RuntimeError(f"reserved draw blocks intersect: {collisions}")


def nc_seed(draw_index: int, stream: str) -> int:
    """Namespaced seed.  Uses the release's own seed_for, rebound to NC-TM-V1."""
    source = load_vendor()["source"]
    return int(source.seed_for(SEED_NAMESPACE, draw_index, stream))


def _make_similarity_nc(draw_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the reviewed make_similarity code object with seed_for rebound to NC-TM-V1.

    This mirrors the technique the release itself uses in
    isolation_factorial_v2_bounded_20260903.paired_write_carriers: the code
    object and every numerical helper stay byte-identical; only the seed
    namespace changes, so the NC draws are disjoint from every published draw.
    """
    source = load_vendor()["source"]

    def seed_adapter(_configuration: str, draw: int, stream: str) -> int:
        return nc_seed(draw, stream)

    rebound = dict(source.make_similarity.__globals__)
    rebound["seed_for"] = seed_adapter
    make_similarity_nc = FunctionType(
        source.make_similarity.__code__,
        rebound,
        name=source.make_similarity.__name__,
        argdefs=source.make_similarity.__defaults__,
        closure=source.make_similarity.__closure__,
    )
    if make_similarity_nc.__code__ is not source.make_similarity.__code__:
        raise RuntimeError("make_similarity code object was not preserved")
    return make_similarity_nc(CONDITION, draw_index)


def build_carrier_draw(draw_index: int) -> dict[str, Any]:
    """One paired carrier draw: shared Q, K, U; normal and non-normal writers.

    The pair shares Q, the coupling K and the store map U, exactly as required
    by protocol section 3.2.
    """
    source = load_vendor()["source"]
    nonnormal, construction = _make_similarity_nc(draw_index)
    q_seed = int(construction["seeds"]["q"])
    normal = source.haar_orthogonal(WRITE_DIM, np.random.default_rng(q_seed))
    if sha256_array(normal) != construction["q_sha256_float64_c_order"]:
        raise RuntimeError("paired Q reconstruction does not match make_similarity Q")

    coupling_seed = nc_seed(draw_index, "coupling_rows")
    store_seed = nc_seed(draw_index, "store_u")
    coupling_full = source.haar_orthogonal(WRITE_DIM, np.random.default_rng(coupling_seed))
    coupling = coupling_full[:STORE_DIM, :].copy()
    coupling /= float(np.linalg.svd(coupling, compute_uv=False)[0])
    store_map = source.haar_orthogonal(STORE_DIM, np.random.default_rng(store_seed))

    return {
        "draw_index": int(draw_index),
        "seed_namespace": SEED_NAMESPACE,
        "seeds": {
            "make_similarity": {k: int(v) for k, v in construction["seeds"].items()},
            "coupling_rows": int(coupling_seed),
            "store_u": int(store_seed),
        },
        "W": {"normal": normal, "nonnormal": nonnormal},
        "K": coupling,
        "U": store_map,
        "similarity_condition_observed": float(construction["s_condition_observed"]),
        "hashes": {
            "W_normal": sha256_array(normal),
            "W_nonnormal": sha256_array(nonnormal),
            "K": sha256_array(coupling),
            "U": sha256_array(store_map),
            "similarity": construction["s_sha256_float64_c_order"],
        },
    }


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
def closure_transfers(W: np.ndarray, K: np.ndarray, U: np.ndarray, steps: int):
    """Canonical import: preprint_v1_2_strengthening_20260917.closure_transfers."""
    return load_vendor()["strengthening"].closure_transfers(W, K, U, steps)


def store_operator(transfers: list[np.ndarray], covariance: np.ndarray) -> np.ndarray:
    """Canonical import: preprint_v1_2_strengthening_20260917.store_operator."""
    return load_vendor()["strengthening"].store_operator(transfers, covariance)


def write_block_operator(W: np.ndarray, horizon: int = MATRIX_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Canonical import: preprint_replication_and_factorial_20260903.fisher_memory_matrix.

    This is M_n of manuscript section 2 on the 32x32 writer matrix, i.e. the
    protocol's M_write = M_2048.  It is a write-block quantity, not a store one.
    """
    return load_vendor()["source"].fisher_memory_matrix(W, horizon)


def system_operators(W: np.ndarray, K: np.ndarray, U: np.ndarray) -> dict[str, Any]:
    """All closure-time operators for one (writer, coupling, store) system.

    Returns L_0, C_ss, M_old, M_store, M_write and their spectra.
    """
    transfers, covariance = closure_transfers(W, K, U, WRITE_WINDOW)
    if not np.allclose(transfers[0], 0.0, atol=0.0):
        raise RuntimeError("index convention violated: transfers[0] must be exactly zero")
    L0 = transfers[WRITE_WINDOW - 1]
    covariance_inverse = np.linalg.inv(covariance)
    M_store = store_operator(transfers, covariance)
    M_old = L0.T @ covariance_inverse @ L0
    M_old = 0.5 * (M_old + M_old.T)
    M_write, write_covariance = write_block_operator(W)
    return {
        "transfers": transfers,
        "C_ss": covariance,
        "C_ss_inv": covariance_inverse,
        "L0": L0,
        "M_old": M_old,
        "M_store": 0.5 * (M_store + M_store.T),
        "M_write": M_write,
        "C_write": write_covariance,
    }


def target_operator(ops: dict[str, Any], target_time: int) -> np.ndarray:
    """M_{t_0} = L_{t_0}^T C_ss^-1 L_{t_0} for an input injected at step t_0.

    Index map: transfers[j] corresponds to injection time t = T_w - 1 - j, so
    L_{t_0} = transfers[T_w - 1 - t_0].
    """
    if not 0 <= target_time <= WRITE_WINDOW - 2:
        raise ValueError(f"target time {target_time} has no store transfer")
    L = ops["transfers"][WRITE_WINDOW - 1 - target_time]
    M = L.T @ np.linalg.solve(ops["C_ss"], L)
    return 0.5 * (M + M.T)


def carrier_fingerprint(draw: dict[str, Any]) -> dict[str, Any]:
    """Byte hashes plus BLAS-PORTABLE numerical invariants of one carrier draw.

    Measured fact, not a defect: `haar_orthogonal` builds Q from `np.linalg.qr`,
    and QR is not bit-reproducible across LAPACK implementations.  The same seed
    on macOS/Accelerate and on Linux/OpenBLAS produced carriers agreeing to
    about 1e-14 absolute and lambda_max(M_old) to about 1e-15 relative, but with
    different SHA-256 digests.  A journal artifact therefore cannot define
    reproduction as byte equality of the carriers.  The invariants below are the
    portable identity; the digests are recorded as machine-specific provenance.
    """
    invariants: dict[str, float] = {}
    for writer_type in WRITER_TYPES:
        ops = system_operators(draw["W"][writer_type], draw["K"], draw["U"])
        values, _ = eig_descending(ops["M_old"])
        store_values, _ = eig_descending(ops["M_store"])
        invariants[f"{writer_type}_lambda_max_M_old"] = float(values[0])
        invariants[f"{writer_type}_lambda2_over_lambda1_M_old"] = float(values[1] / values[0])
        invariants[f"{writer_type}_trace_M_old"] = float(np.sum(values))
        invariants[f"{writer_type}_trace_M_store"] = float(np.sum(store_values))
        invariants[f"{writer_type}_lambda_max_M_store"] = float(store_values[0])
        invariants[f"{writer_type}_cond_C_ss"] = float(np.linalg.cond(ops["C_ss"]))
        invariants[f"{writer_type}_W_frobenius"] = float(np.linalg.norm(draw["W"][writer_type]))
    invariants["K_top_singular_value"] = float(
        np.linalg.svd(draw["K"], compute_uv=False)[0])
    invariants["U_orthogonality_defect"] = float(
        np.max(np.abs(draw["U"] @ draw["U"].T - np.eye(STORE_DIM))))
    invariants["similarity_condition"] = float(draw["similarity_condition_observed"])
    return {
        "draw_index": draw["draw_index"],
        "byte_digests_machine_specific": draw["hashes"],
        "portable_invariants": invariants,
        "portable_invariants_rounded_12": {k: float(f"{v:.12e}") for k, v in invariants.items()},
        "portability_note": (
            "SHA-256 of the carrier arrays is LAPACK-implementation dependent because "
            "haar_orthogonal uses np.linalg.qr. Reproduction is defined on the rounded "
            "invariants, not on the digests."
        ),
    }


def eig_descending(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition, eigenvalues descending, canonical signs."""
    source = load_vendor()["source"]
    values, vectors = np.linalg.eigh(0.5 * (M + M.T))
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = source.canonicalize_eigenvector_signs(vectors[:, order])
    return values, vectors


def leading_subspace(M: np.ndarray, ratio: float = 0.99) -> np.ndarray:
    """Columns spanning E_{0.99} = span{e_i : lambda_i >= ratio * lambda_max}."""
    values, vectors = eig_descending(M)
    keep = values >= ratio * values[0]
    return vectors[:, keep]


def write_block_oracle(draw: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """The write-block oracle, taken from the NON-NORMAL writer of the pair.

    For a normal carrier the write-block operator is the identity exactly
    (manuscript section 3.2; measured residual about 2e-14), so `eigmax(M_2048)`
    on a normal writer returns the LAPACK ordering of numerical noise: it is not
    a direction, it is not reproducible across BLAS builds, and it carries no
    information.  The manuscript's own section 5.2 convention is used instead:
    the oracle is selected from the conditioned writer's `M_2048` and SHARED
    with the paired normal cell, as a matched directional intervention.
    """
    M_write_nn, _ = write_block_operator(draw["W"]["nonnormal"])
    M_write_n, _ = write_block_operator(draw["W"]["normal"])
    identity_residual = float(np.max(np.abs(M_write_n - np.eye(WRITE_DIM))))
    values, vectors = eig_descending(M_write_nn)
    return vectors[:, 0].copy(), {
        "source": "nonnormal_writer_M_2048_shared_with_paired_normal_cell",
        "normal_identity_residual": identity_residual,
        "nonnormal_lambda_max": float(values[0]),
        "nonnormal_lambda_min": float(values[-1]),
    }


def direction_controls(ops: dict[str, Any], draw_index: int, writer_type: str,
                       random_count: int = 128,
                       write_oracle: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Protocol section 4.8 control directions (learned direction added later)."""
    old_values, old_vectors = eig_descending(ops["M_old"])
    _, store_vectors = eig_descending(ops["M_store"])
    if write_oracle is None:
        _, write_vectors = eig_descending(ops["M_write"])
        write_oracle = write_vectors[:, 0].copy()
    # M_old = L_0^T C_ss^-1 L_0 has rank at most the store dimension, so its
    # bottom eigenvector lies in a (N - d)-dimensional kernel and is an
    # arbitrary member of it.  It is kept because decision rule 10.2 item 3
    # names it, and the smallest NONZERO eigendirection is reported alongside
    # as the direction that rule was presumably reaching for.
    positive = np.flatnonzero(old_values > 1e-12 * max(old_values[0], 1e-300))
    worst_positive = old_vectors[:, positive[-1]].copy() if len(positive) else old_vectors[:, -1].copy()
    stream = f"random_directions|{writer_type}"
    rng = np.random.default_rng(nc_seed(draw_index, stream))
    randoms = rng.standard_normal((WRITE_DIM, random_count))
    randoms /= np.linalg.norm(randoms, axis=0, keepdims=True)
    return {
        "v_old": old_vectors[:, 0].copy(),
        "v_store": store_vectors[:, 0].copy(),
        "v_write": write_oracle,
        "v_bottom": old_vectors[:, -1].copy(),
        "v_worst_positive": worst_positive,
        "v_random": randoms,
        "M_old_eigenvalues": old_values,
        "M_old_rank": int(len(positive)),
    }


# --------------------------------------------------------------------------
# Storage conditions
# --------------------------------------------------------------------------
def isolated_endpoint_from(U: np.ndarray, L0: np.ndarray, C_ss: np.ndarray,
                           horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if horizon < WRITE_WINDOW:
        raise ValueError(f"isolated horizon {horizon} is inside the write window {WRITE_WINDOW}")
    A = np.linalg.matrix_power(U, horizon - WRITE_WINDOW)
    return A @ L0, A @ C_ss @ A.T, A


def open_endpoint(W: np.ndarray, K: np.ndarray, U: np.ndarray,
                  horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(sensitivity map, covariance) at query horizon H under open coupling.

    The coupling never closes, so the store keeps admitting write-path noise.
    Full propagation: run the same closure recursion out to `horizon` steps and
    read the transfer of the t_0 = 0 input, which is transfers[horizon - 1].
    """
    if horizon < 1:
        raise ValueError("open horizon must be at least 1")
    transfers, covariance = closure_transfers(W, K, U, horizon)
    return transfers[horizon - 1], covariance


def fisher_information(P: np.ndarray, covariance: np.ndarray, v: np.ndarray) -> float:
    """J = p^T Sigma^{-1} p with p = P v."""
    p = P @ v
    return float(p @ np.linalg.solve(covariance, p))


def fisher_operator(P: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """M = P^T Sigma^{-1} P, so that J(v) = v^T M v."""
    M = P.T @ np.linalg.solve(covariance, P)
    return 0.5 * (M + M.T)


def bayes_accuracy(amplitude: float, J: float) -> float:
    """Proposition 2.1:  Acc* = Phi(a sqrt(J))."""
    from math import erf, sqrt
    z = amplitude * np.sqrt(max(J, 0.0))
    return float(0.5 * (1.0 + erf(z / sqrt(2.0))))
