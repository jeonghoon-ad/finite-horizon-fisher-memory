"""Isolation factorial v2 on paired bi-power-bounded write carriers.

DECISION-BEARING SYNTHETIC MEASUREMENT. Run only through the REPRODUCTION wrapper
after the named independent source review binds this exact script SHA-256 and
contains exactly one ``REPRODUCTION DECISION: RUN`` line.

The reviewed preprint replication module is imported read-only by exact
SHA-256. Its ``make_similarity`` code object is executed unchanged under the
frozen ISO-V2 seed namespace, and its time-varying Fisher route is used as a
parity oracle for the block-specialized full/store readout. No checkpoint,
trained weight, GPU, network, Controller, queue, roadmap, or manuscript is
read or modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
from types import FunctionType, ModuleType

import numpy as np


SOURCE_SHA256 = "f8c0884d41cf7a112468013dee6ad70702a7a7be4b6ee700d2100d128aaa2ad4"
WRITE_DIM = 32
STORE_DIM = 16
TOTAL_DIM = WRITE_DIM + STORE_DIM
CONDITION = 10.0
DRAW_COUNT = 8
WRITE_WINDOW = 24
HORIZONS = (64, 256, 1024, 4096)
TRACE_ATOL = 1e-9
NORMAL_IDENTITY_ATOL = 1e-12
ROUTE_PARITY_ATOL = 1e-9
INVARIANCE_ATOL = 1e-9
DIFFERENCE_FLOOR = 1e-9
CONSTANT_REL_SPREAD = 1e-6
OPEN_DECAY_FACTOR = 10.0
CONCENTRATION_THRESHOLD = 1.05
REACH_THRESHOLD = 1e-6

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "scripts" / "preprint_replication_and_factorial_20260903.py"
REVIEW_PATH = PROJECT_ROOT / "prior_art" / "REVIEW_ISOLATION_FACTORIAL_V2_BOUNDED_20260903.md"
OUTPUT_DIR = PROJECT_ROOT / "results" / "isolation_factorial_v2_bounded_20260903"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def load_reviewed_source() -> ModuleType:
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("reviewed preprint replication source SHA-256 drifted")
    spec = importlib.util.spec_from_file_location("reviewed_preprint_replication_20260903", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct read-only source import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def control(
    name: str,
    observed: object,
    expected: object,
    passed: bool,
    tolerance: object,
) -> dict[str, object]:
    return {
        "name": name,
        "observed": observed,
        "expected": expected,
        "tolerance": tolerance,
        "pass": bool(passed),
    }


def iso_seed(source: ModuleType, draw_index: int, stream: str) -> int:
    return int(source.seed_for("ISO-V2", draw_index, stream))


def paired_write_carriers(
    source: ModuleType,
    draw_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Run the reviewed make_similarity code with its seed dependency rebound.

    The code object and every numerical helper stay unchanged. Only the
    configuration label passed to the reviewed seed function is frozen to
    ISO-V2, as required by this handoff.
    """

    def seed_adapter(_configuration: str, draw: int, stream: str) -> int:
        return iso_seed(source, draw, stream)

    rebound_globals = dict(source.make_similarity.__globals__)
    rebound_globals["seed_for"] = seed_adapter
    make_similarity_iso = FunctionType(
        source.make_similarity.__code__,
        rebound_globals,
        name=source.make_similarity.__name__,
        argdefs=source.make_similarity.__defaults__,
        closure=source.make_similarity.__closure__,
    )
    nonnormal, construction = make_similarity_iso(CONDITION, draw_index)
    q_seed = int(construction["seeds"]["q"])
    normal = source.haar_orthogonal(WRITE_DIM, np.random.default_rng(q_seed))
    q_hash = sha256_array(normal)
    if q_hash != construction["q_sha256_float64_c_order"]:
        raise RuntimeError("paired Q reconstruction does not match reviewed make_similarity Q")
    return normal, nonnormal, {
        "seed_namespace": "seed_for('ISO-V2', draw, stream)",
        "seeds": construction["seeds"],
        "normal_q_sha256_float64_c_order": q_hash,
        "nonnormal_w_sha256_float64_c_order": sha256_array(nonnormal),
        "similarity_sha256_float64_c_order": construction["s_sha256_float64_c_order"],
        "similarity_condition_observed": construction["s_condition_observed"],
        "make_similarity_code_object_identity": bool(make_similarity_iso.__code__ is source.make_similarity.__code__),
    }


def paired_draw_objects(
    source: ModuleType,
    draw_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    seeds = {
        "coupling_rows": iso_seed(source, draw_index, "coupling_rows"),
        "store_u": iso_seed(source, draw_index, "store_u"),
        "random_direction": iso_seed(source, draw_index, "random_direction"),
    }
    coupling_orthogonal = source.haar_orthogonal(
        WRITE_DIM,
        np.random.default_rng(seeds["coupling_rows"]),
    )
    coupling = coupling_orthogonal[:STORE_DIM, :].copy()
    coupling /= float(np.linalg.svd(coupling, compute_uv=False)[0])
    store = source.haar_orthogonal(STORE_DIM, np.random.default_rng(seeds["store_u"]))
    random_direction = np.random.default_rng(seeds["random_direction"]).standard_normal((WRITE_DIM, 1))
    random_direction /= np.linalg.norm(random_direction)
    return coupling, store, random_direction, seeds


def top_oracle(source: ModuleType, carrier: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    matrix, covariance = source.fisher_memory_matrix(carrier, source.MATRIX_HORIZON)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvectors = source.canonicalize_eigenvector_signs(eigenvectors)
    oracle = eigenvectors[:, -1:].copy()
    return oracle, {
        "horizon": source.MATRIX_HORIZON,
        "lambda_max": float(eigenvalues[-1]),
        "lambda_min": float(eigenvalues[0]),
        "trace_over_N": float(np.trace(matrix) / WRITE_DIM),
        "matrix_sha256_float64_c_order": sha256_array(matrix),
        "covariance_sha256_float64_c_order": sha256_array(covariance),
        "oracle_sha256_float64_c_order": sha256_array(oracle),
        "oracle_direction": [float(value) for value in oracle[:, 0]],
    }


def empty_state(direction_count: int) -> dict[str, object]:
    return {
        "pw": np.eye(WRITE_DIM, dtype=np.float64),
        "lower": np.zeros((STORE_DIM, WRITE_DIM), dtype=np.float64),
        "ps": np.eye(STORE_DIM, dtype=np.float64),
        "cww": np.zeros((WRITE_DIM, WRITE_DIM), dtype=np.float64),
        "cws": np.zeros((WRITE_DIM, STORE_DIM), dtype=np.float64),
        "css": np.zeros((STORE_DIM, STORE_DIM), dtype=np.float64),
        "prop_w": [[] for _ in range(direction_count)],
        "prop_s": [[] for _ in range(direction_count)],
    }


def clone_state(state: dict[str, object]) -> dict[str, object]:
    return {
        "pw": state["pw"].copy(),
        "lower": state["lower"].copy(),
        "ps": state["ps"].copy(),
        "cww": state["cww"].copy(),
        "cws": state["cws"].copy(),
        "css": state["css"].copy(),
        "prop_w": [list(rows) for rows in state["prop_w"]],
        "prop_s": [list(rows) for rows in state["prop_s"]],
    }


def reverse_step(
    state: dict[str, object],
    write_carrier: np.ndarray,
    coupling: np.ndarray,
    store_map: np.ndarray,
    directions: tuple[np.ndarray, ...],
) -> None:
    pw = state["pw"]
    lower = state["lower"]
    ps = state["ps"]
    state["cww"] += pw @ pw.T
    state["cws"] += pw @ lower.T
    state["css"] += lower @ lower.T
    for index, direction in enumerate(directions):
        state["prop_w"][index].append((pw @ direction)[:, 0].copy())
        state["prop_s"][index].append((lower @ direction)[:, 0].copy())
    state["pw"] = pw @ write_carrier
    state["lower"] = lower @ write_carrier + ps @ coupling
    state["ps"] = ps @ store_map


def read_state(state: dict[str, object]) -> list[dict[str, object]]:
    covariance = np.block([
        [state["cww"], state["cws"]],
        [state["cws"].T, state["css"]],
    ])
    covariance_inverse = np.linalg.pinv(covariance, rcond=1e-14, hermitian=True)
    store_inverse = np.linalg.pinv(state["css"], rcond=1e-14, hermitian=True)
    outputs: list[dict[str, object]] = []
    for prop_w_rows, prop_s_rows in zip(state["prop_w"], state["prop_s"]):
        prop_w = np.asarray(prop_w_rows, dtype=np.float64)
        prop_s = np.asarray(prop_s_rows, dtype=np.float64)
        propagated = np.concatenate((prop_w, prop_s), axis=1)
        full_j = np.einsum("ki,ij,kj->k", propagated, covariance_inverse, propagated)
        store_j = np.einsum("ki,ij,kj->k", prop_s, store_inverse, prop_s)
        outputs.append({
            "full_j": full_j,
            "full_j_total": float(np.sum(full_j)),
            "store_j": store_j,
            "store_j_total": float(np.sum(store_j)),
            "covariance_sha256_float64_c_order": sha256_array(covariance),
            "store_covariance_sha256_float64_c_order": sha256_array(state["css"]),
        })
    return outputs


def route_snapshots(
    write_carrier: np.ndarray,
    coupling: np.ndarray,
    store_open: np.ndarray,
    store_closed: np.ndarray,
    isolated: bool,
    directions: tuple[np.ndarray, ...],
) -> dict[int, list[dict[str, object]]]:
    snapshots: dict[int, list[dict[str, object]]] = {}
    if not isolated:
        state = empty_state(len(directions))
        for delay_count in range(1, max(HORIZONS) + 1):
            reverse_step(state, write_carrier, coupling, store_open, directions)
            if delay_count in HORIZONS:
                snapshots[delay_count] = read_state(state)
        return snapshots

    closed_targets = {horizon - WRITE_WINDOW: horizon for horizon in HORIZONS}
    closed_state = empty_state(len(directions))
    captured: dict[int, dict[str, object]] = {}
    for closed_count in range(1, max(closed_targets) + 1):
        reverse_step(
            closed_state,
            write_carrier,
            np.zeros_like(coupling),
            store_closed,
            directions,
        )
        if closed_count in closed_targets:
            captured[closed_targets[closed_count]] = clone_state(closed_state)
    for horizon in HORIZONS:
        state = captured[horizon]
        for _ in range(WRITE_WINDOW):
            reverse_step(state, write_carrier, coupling, store_open, directions)
        snapshots[horizon] = read_state(state)
    return snapshots


def build_full_carriers(
    write_carrier: np.ndarray,
    coupling: np.ndarray,
    store_open: np.ndarray,
    store_closed: np.ndarray,
    isolated: bool,
    horizon: int,
) -> list[np.ndarray]:
    open_carrier = np.block([
        [write_carrier, np.zeros((WRITE_DIM, STORE_DIM), dtype=np.float64)],
        [coupling, store_open],
    ])
    if not isolated:
        return [open_carrier for _ in range(horizon)]
    closed_carrier = np.block([
        [write_carrier, np.zeros((WRITE_DIM, STORE_DIM), dtype=np.float64)],
        [np.zeros_like(coupling), store_closed],
    ])
    return [open_carrier if time_index < WRITE_WINDOW else closed_carrier for time_index in range(horizon)]


def source_route_parity(
    source: ModuleType,
    write_carrier: np.ndarray,
    coupling: np.ndarray,
    store_open: np.ndarray,
    store_closed: np.ndarray,
    isolated: bool,
    directions: tuple[np.ndarray, ...],
    horizon: int,
    local_rows: list[dict[str, object]],
) -> tuple[float, float]:
    carriers = build_full_carriers(
        write_carrier,
        coupling,
        store_open,
        store_closed,
        isolated,
        horizon,
    )
    writes = np.zeros((TOTAL_DIM, len(directions)), dtype=np.float64)
    writes[:WRITE_DIM, :] = np.concatenate(directions, axis=1)
    noise = np.zeros((TOTAL_DIM, TOTAL_DIM), dtype=np.float64)
    noise[:WRITE_DIM, :WRITE_DIM] = np.eye(WRITE_DIM, dtype=np.float64)
    source_j, source_total = source.fisher_memory_tv_varying(
        carriers,
        [writes for _ in range(horizon)],
        [noise for _ in range(horizon)],
    )
    local_sum = sum((row["full_j"] for row in local_rows), np.zeros(horizon, dtype=np.float64))
    curve_error = float(np.max(np.abs(source_j - local_sum)))
    total_error = abs(float(source_total) - sum(float(row["full_j_total"]) for row in local_rows))
    return curve_error, total_error


def json_readout(row: dict[str, object], horizon: int) -> dict[str, object]:
    full_j = row["full_j"]
    store_j = row["store_j"]
    reach = {
        str(time_index): float(store_j[horizon - 1 - time_index])
        for time_index in range(WRITE_WINDOW)
    }
    return {
        "full_state": {
            "j_by_delay": [float(value) for value in full_j],
            "j_total": float(row["full_j_total"]),
            "j_oldest": float(full_j[-1]),
            "nonzero_lag_count": int(np.count_nonzero(full_j)),
        },
        "store_only": {
            "j_by_delay": [float(value) for value in store_j],
            "j_total": float(row["store_j_total"]),
            "j_oldest_stored_lag_t0": float(store_j[-1]),
            "nonzero_lag_count": int(np.count_nonzero(store_j)),
            "write_window_reach_by_input_time": reach,
            "reach_count_above_1e-6": sum(value > REACH_THRESHOLD for value in reach.values()),
        },
        "covariance_sha256_float64_c_order": row["covariance_sha256_float64_c_order"],
        "store_covariance_sha256_float64_c_order": row["store_covariance_sha256_float64_c_order"],
    }


def maximum_readout_difference(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
) -> tuple[float, float]:
    full = max(float(np.max(np.abs(a["full_j"] - b["full_j"]))) for a, b in zip(left, right))
    store = max(float(np.max(np.abs(a["store_j"] - b["store_j"]))) for a, b in zip(left, right))
    return full, store


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


def measurement(source: ModuleType) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    checks: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    draws: list[dict[str, object]] = []
    for draw_index in range(DRAW_COUNT):
        normal, nonnormal, carrier_meta = paired_write_carriers(source, draw_index)
        coupling, store, random_direction, object_seeds = paired_draw_objects(source, draw_index)
        oracle, oracle_meta = top_oracle(source, nonnormal)
        directions = (oracle, random_direction)
        checks.append(control(
            f"draw{draw_index}.make_similarity_code_identity",
            carrier_meta["make_similarity_code_object_identity"],
            True,
            bool(carrier_meta["make_similarity_code_object_identity"]),
            "exact boolean",
        ))
        checks.append(control(
            f"draw{draw_index}.coupling_operator_norm",
            float(np.linalg.svd(coupling, compute_uv=False)[0]),
            1.0,
            abs(float(np.linalg.svd(coupling, compute_uv=False)[0]) - 1.0) <= 1e-12,
            1e-12,
        ))
        checks.append(control(
            f"draw{draw_index}.coupling_rows_orthonormal",
            float(np.max(np.abs(coupling @ coupling.T - np.eye(STORE_DIM)))),
            0.0,
            float(np.max(np.abs(coupling @ coupling.T - np.eye(STORE_DIM)))) <= 1e-12,
            1e-12,
        ))

        write_carriers = {"normal": normal, "nonnormal": nonnormal}
        trace_meta: dict[str, object] = {}
        for write_name, write_carrier in write_carriers.items():
            trace_meta[write_name] = {}
            for horizon in HORIZONS:
                matrix, covariance = source.fisher_memory_matrix(write_carrier, horizon)
                trace_over_n = float(np.trace(matrix) / WRITE_DIM)
                checks.append(control(
                    f"draw{draw_index}.{write_name}.trace_identity.n{horizon}",
                    trace_over_n,
                    1.0,
                    abs(trace_over_n - 1.0) <= TRACE_ATOL,
                    TRACE_ATOL,
                ))
                identity_error = None
                if write_name == "normal":
                    identity_error = float(np.max(np.abs(matrix - np.eye(WRITE_DIM))))
                    checks.append(control(
                        f"draw{draw_index}.normal.matrix_identity.n{horizon}",
                        identity_error,
                        0.0,
                        identity_error <= NORMAL_IDENTITY_ATOL,
                        NORMAL_IDENTITY_ATOL,
                    ))
                trace_meta[write_name][str(horizon)] = {
                    "trace": float(np.trace(matrix)),
                    "trace_over_N": trace_over_n,
                    "identity_max_abs_error_if_normal": identity_error,
                    "matrix_sha256_float64_c_order": sha256_array(matrix),
                    "covariance_sha256_float64_c_order": sha256_array(covariance),
                }

        cell_rows: dict[str, object] = {}
        control_variants: dict[str, object] = {}
        for write_name, write_carrier in write_carriers.items():
            baseline_open = route_snapshots(
                write_carrier, coupling, store, store, False, directions,
            )
            baseline_isolated = route_snapshots(
                write_carrier, coupling, store, store, True, directions,
            )
            # REVIEWER PATCH (independent source review, 2026-09-03 13:19): physical post-closure
            # contraction. The earlier call was byte-identical to baseline_isolated and
            # could not fail. Store-only gate at n in {64,256,1024}; full-state gate at
            # n=64 only; larger horizons recorded (float64 underflow / pinv threshold).
            postclose_contract_physical = route_snapshots(
                write_carrier, coupling, store, 0.9 * store, True, directions,
            )
            write_contract_factored = route_snapshots(
                write_carrier, coupling, 0.9 * store, store, True, directions,
            )
            identity_postclose = route_snapshots(
                write_carrier, coupling, store, np.eye(STORE_DIM), True, directions,
            )
            control_variants[write_name] = {
                "postclosure_0.9U_physical": {},
                "write_and_postclosure_0.9U_scale_factored": {},
                "postclosure_identity_store": {},
            }
            for storage_name, snapshots, isolated in (
                ("open", baseline_open, False),
                ("isolated", baseline_isolated, True),
            ):
                cell_name = f"{write_name}_{storage_name}"
                cell_rows[cell_name] = {"oracle": {}, "random": {}}
                for horizon in HORIZONS:
                    for direction_index, direction_name in enumerate(("oracle", "random")):
                        rendered = json_readout(snapshots[horizon][direction_index], horizon)
                        cell_rows[cell_name][direction_name][str(horizon)] = rendered
                        csv_rows.append({
                            "draw": draw_index,
                            "write_path": write_name,
                            "storage": storage_name,
                            "direction": direction_name,
                            "horizon": horizon,
                            "full_j_total": rendered["full_state"]["j_total"],
                            "full_j_oldest": rendered["full_state"]["j_oldest"],
                            "full_nonzero_lag_count": rendered["full_state"]["nonzero_lag_count"],
                            "store_j_total": rendered["store_only"]["j_total"],
                            "store_j_oldest_t0": rendered["store_only"]["j_oldest_stored_lag_t0"],
                            "store_nonzero_lag_count": rendered["store_only"]["nonzero_lag_count"],
                            "reach_count_above_1e-6": rendered["store_only"]["reach_count_above_1e-6"],
                        })
                parity_horizons = HORIZONS if draw_index == 0 else (64,)
                for horizon in parity_horizons:
                    curve_error, total_error = source_route_parity(
                        source,
                        write_carrier,
                        coupling,
                        store,
                        store,
                        isolated,
                        directions,
                        horizon,
                        snapshots[horizon],
                    )
                    checks.append(control(
                        f"draw{draw_index}.{cell_name}.source_route_parity.n{horizon}",
                        {"curve_max_abs_error": curve_error, "total_abs_error": total_error},
                        {"curve_max_abs_error": 0.0, "total_abs_error": 0.0},
                        max(curve_error, total_error) <= ROUTE_PARITY_ATOL,
                        ROUTE_PARITY_ATOL,
                    ))

            write_phase_differences: list[float] = []
            for horizon in HORIZONS:
                post_full, post_store = maximum_readout_difference(
                    baseline_isolated[horizon],
                    postclose_contract_physical[horizon],
                )
                gate_store = horizon in (64, 256, 1024)
                gate_full = horizon == 64
                if gate_store or gate_full:
                    gated_error = max(
                        post_store if gate_store else 0.0,
                        post_full if gate_full else 0.0,
                    )
                    checks.append(control(
                        f"draw{draw_index}.{write_name}.postclosure_contraction_invariance.n{horizon}",
                        {"full_curve_max_abs_error": post_full, "store_curve_max_abs_error": post_store,
                         "gated": {"store_only": gate_store, "full_state": gate_full}},
                        {"gated_max_abs_error": 0.0},
                        gated_error <= INVARIANCE_ATOL,
                        INVARIANCE_ATOL,
                    ))
                identity_full, identity_store = maximum_readout_difference(
                    baseline_isolated[horizon],
                    identity_postclose[horizon],
                )
                checks.append(control(
                    f"draw{draw_index}.{write_name}.identity_store_invariance.n{horizon}",
                    identity_store,
                    0.0,
                    identity_store <= INVARIANCE_ATOL,
                    INVARIANCE_ATOL,
                ))
                write_full, write_store = maximum_readout_difference(
                    baseline_isolated[horizon],
                    write_contract_factored[horizon],
                )
                write_phase_differences.extend((write_full, write_store))
                control_variants[write_name]["postclosure_0.9U_physical"][str(horizon)] = {
                    "full_curve_max_abs_difference_from_isometric": post_full,
                    "store_curve_max_abs_difference_from_isometric": post_store,
                    "physical_closed_map": "0.9*U (run physically)",
                    "gated": {"store_only": horizon in (64, 256, 1024), "full_state": horizon == 64},
                    "note": "n=4096 store scale 0.9^(2*4072) underflows float64 (UNDERFLOW_EXPECTED); full-state at n>=256 falls below the 1e-14 relative pseudoinverse threshold; recorded, not gated",
                }
                control_variants[write_name]["write_and_postclosure_0.9U_scale_factored"][str(horizon)] = {
                    "full_curve_max_abs_difference_from_isometric": write_full,
                    "store_curve_max_abs_difference_from_isometric": write_store,
                }
                control_variants[write_name]["postclosure_identity_store"][str(horizon)] = {
                    "full_curve_max_abs_difference_from_U": identity_full,
                    "store_curve_max_abs_difference_from_U": identity_store,
                }
            checks.append(control(
                f"draw{draw_index}.{write_name}.write_phase_contraction_must_differ",
                max(write_phase_differences),
                f"> {DIFFERENCE_FLOOR}",
                max(write_phase_differences) > DIFFERENCE_FLOOR,
                "strict threshold",
            ))

        draws.append({
            "draw_index": draw_index,
            "carrier_pair": carrier_meta,
            "v2_object_seeds": object_seeds,
            "coupling_sha256_float64_c_order": sha256_array(coupling),
            "coupling_operator_norm": float(np.linalg.svd(coupling, compute_uv=False)[0]),
            "store_u_sha256_float64_c_order": sha256_array(store),
            "random_direction_sha256_float64_c_order": sha256_array(random_direction),
            "nonnormal_oracle": oracle_meta,
            "write_carrier_trace_controls": trace_meta,
            "cells": cell_rows,
            "invariance_control_variants": control_variants,
        })

    for cell_name in ("normal_open", "normal_isolated", "nonnormal_open", "nonnormal_isolated"):
        checks.append(control(
            f"{cell_name}.draw_count",
            sum(cell_name in draw["cells"] for draw in draws),
            DRAW_COUNT,
            sum(cell_name in draw["cells"] for draw in draws) == DRAW_COUNT,
            "exact",
        ))
    payload: dict[str, object] = {
        "schema": "isolation_factorial_v2_bounded_v1",
        "run_status": "PENDING_CONTROLS",
        "model": "x_(t+1)=W_t x_t+B_t s_t+G_t z_t; x=(x_w[32],x_s[16]); x_0=0; epsilon=1",
        "write_axis": {
            "normal": "Q",
            "nonnormal": "SQS^-1 at condition(S)=10",
            "pairing": "same Q and same draw; only S differs",
        },
        "storage_axis": {
            "open": "K live at every t; no direct store noise",
            "isolated": "K live for t<24 and zero for t>=24; no direct store noise",
            "shared": "write input and write-block noise continue for the entire horizon; A=U isometric",
        },
        "direction_subarms": ["paired nonnormal M_2048 oracle", "paired seeded random unit vector"],
        "horizons": list(HORIZONS),
        "draw_count": DRAW_COUNT,
        "draws": draws,
        "pre_registered_readings": None,
    }
    return payload, checks, csv_rows


def derive_readings(payload: dict[str, object]) -> dict[str, object]:
    draws = payload["draws"]
    concentration_ratios = []
    for draw in draws:
        normal = draw["cells"]["normal_isolated"]["oracle"]["4096"]["store_only"]["j_total"]
        nonnormal = draw["cells"]["nonnormal_isolated"]["oracle"]["4096"]["store_only"]["j_total"]
        concentration_ratios.append(float(nonnormal) / float(normal))
    concentration_median = float(np.median(concentration_ratios))
    readings: dict[str, object] = {
        "NONNORMAL_ISOLATED_STORE_ONLY_CONCENTRATION": {
            "label": (
                "STORE_ONLY_CONCENTRATION_ABOVE_ONE_yes"
                if concentration_median > CONCENTRATION_THRESHOLD
                else "STORE_ONLY_CONCENTRATION_ABOVE_ONE_no"
            ),
            "threshold": "> 1.05",
            "paired_ratio_per_draw": concentration_ratios,
            "ratio_summary": summary_stats(concentration_ratios),
        }
    }
    for write_name in ("normal", "nonnormal"):
        isolated_curve = []
        open_curve = []
        for horizon in HORIZONS:
            isolated_curve.append(float(np.median([
                draw["cells"][f"{write_name}_isolated"]["oracle"][str(horizon)]["store_only"]["j_oldest_stored_lag_t0"]
                for draw in draws
            ])))
            open_curve.append(float(np.median([
                draw["cells"][f"{write_name}_open"]["oracle"][str(horizon)]["store_only"]["j_oldest_stored_lag_t0"]
                for draw in draws
            ])))
        isolated_relative_spread = (
            (max(isolated_curve) - min(isolated_curve))
            / max(abs(float(np.mean(isolated_curve))), np.finfo(np.float64).tiny)
        )
        open_decay = open_curve[0] / max(open_curve[-1], np.finfo(np.float64).tiny)
        reach_counts = [
            int(draw["cells"][f"{write_name}_isolated"]["oracle"]["4096"]["store_only"]["reach_count_above_1e-6"])
            for draw in draws
        ]
        readings[f"{write_name.upper()}_ISOLATED_OLDEST_LAG"] = {
            "label": (
                "STORE_ONLY_OLDEST_LAG_CONSTANT_yes"
                if isolated_relative_spread < CONSTANT_REL_SPREAD
                else "STORE_ONLY_OLDEST_LAG_CONSTANT_no"
            ),
            "median_by_horizon": dict(zip((str(h) for h in HORIZONS), isolated_curve)),
            "relative_spread_formula": "(max-min)/abs(mean) on the four draw medians",
            "relative_spread": float(isolated_relative_spread),
            "threshold": "< 1e-6",
        }
        readings[f"{write_name.upper()}_OPEN_OLDEST_LAG"] = {
            "label": (
                "OPEN_CELL_OLDEST_LAG_DECAYS_yes"
                if open_decay >= OPEN_DECAY_FACTOR
                else "OPEN_CELL_OLDEST_LAG_DECAYS_no"
            ),
            "median_by_horizon": dict(zip((str(h) for h in HORIZONS), open_curve)),
            "n64_over_n4096": float(open_decay),
            "threshold": ">= 10",
        }
        readings[f"{write_name.upper()}_ISOLATED_REACH_COUNT"] = {
            "label": "REACH_COUNT",
            "definition": "oracle, n=4096, number of t=0..23 inputs with store-only J > 1e-6",
            "per_draw": reach_counts,
            "summary": summary_stats([float(value) for value in reach_counts]),
        }
    return readings


def provenance(started: float, source: ModuleType) -> dict[str, object]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "reviewed_source_path": str(SOURCE_PATH),
        "reviewed_source_sha256_expected": SOURCE_SHA256,
        "reviewed_source_sha256_observed": sha256_file(SOURCE_PATH),
        "review_file": os.environ.get("PREPRINT_REPRODUCTION_REVIEW_FILE", "NOT_ESTABLISHED"),
        "numpy_version": np.__version__,
        "float_dtype": "float64",
        "elapsed_s": float(time.perf_counter() - started),
        "resource_scope": "local CPU only; no GPU, checkpoint, trained weight, network, Controller, queue, roadmap, or manuscript",
    }


def main() -> int:
    started = time.perf_counter()
    gate_ok, gate_detail = validate_execution_gate()
    if not gate_ok:
        print(f"REFUSED: {gate_detail}.")
        return 2
    try:
        source = load_reviewed_source()
    except RuntimeError as error:
        print(f"REFUSED: {error}.")
        return 2

    print("MEASURE isolation factorial v2: 8 paired draws x 4 cells x 2 directions", flush=True)
    payload, checks, csv_rows = measurement(source)
    valid = all(bool(item["pass"]) for item in checks)
    status = "VALID_MEASUREMENT" if valid else "INSTRUMENT_INVALID"
    payload["run_status"] = status
    payload["pre_registered_readings"] = derive_readings(payload) if valid else None
    prov = provenance(started, source)
    payload["provenance"] = prov
    controls_payload: dict[str, object] = {
        "schema": "isolation_factorial_v2_bounded_positive_controls_v1",
        "run_status": "PASS" if valid else "INSTRUMENT_INVALID",
        "failure_count": sum(1 for item in checks if not item["pass"]),
        "checks": checks,
        "invalid_run_rule": "any frozen control failure suppresses every preregistered reading",
        "rerun_rule": "one corrected repeat run maximum; a second invalid run is REDESIGN_REQUIRED",
        "provenance": prov,
    }
    receipt: dict[str, object] = {
        "schema": "isolation_factorial_v2_bounded_receipt_v1",
        "run_status": status,
        "execution_gate": gate_detail,
        "authority_scope": "Frozen Isolation Factorial v2 reproduction only",
        "attempt_index": 1,
        "controls_total": len(checks),
        "controls_failed": controls_payload["failure_count"],
        "artifacts": [
            "ISOLATION_FACTORIAL_V2.json",
            "CELL_READOUTS.csv",
            "PREREGISTERED_READINGS.json",
            "POSITIVE_CONTROLS.json",
            "RUN_RECEIPT.json",
        ],
        "gate_11": "instrument result only; no track or manuscript verdict",
        "provenance": prov,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source.atomic_json(OUTPUT_DIR / "ISOLATION_FACTORIAL_V2.json", payload)
    source.atomic_json(
        OUTPUT_DIR / "PREREGISTERED_READINGS.json",
        {
            "schema": "isolation_factorial_v2_bounded_preregistered_readings_v1",
            "run_status": status,
            "readings": payload["pre_registered_readings"],
            "provenance": prov,
        },
    )
    source.atomic_json(OUTPUT_DIR / "POSITIVE_CONTROLS.json", controls_payload)
    source.atomic_json(OUTPUT_DIR / "RUN_RECEIPT.json", receipt)
    source.atomic_csv(
        OUTPUT_DIR / "CELL_READOUTS.csv",
        [
            "draw", "write_path", "storage", "direction", "horizon",
            "full_j_total", "full_j_oldest", "full_nonzero_lag_count",
            "store_j_total", "store_j_oldest_t0", "store_nonzero_lag_count",
            "reach_count_above_1e-6",
        ],
        csv_rows,
    )
    if not valid:
        print("INSTRUMENT_INVALID: one or more frozen controls failed; readings suppressed.")
        return 3
    print(f"PASS: wrote reviewed Isolation Factorial v2 artifacts under {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
