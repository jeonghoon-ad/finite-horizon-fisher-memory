"""Addendum reruns: distinct elliptic angle and scale-factored F5.

DECISION-BEARING SYNTHETIC MEASUREMENT. Run only through the REPRODUCTION wrapper
after the named independent source review binds this exact script SHA-256 and
contains exactly one ``REPRODUCTION DECISION: RUN`` line.

The reviewed replication instrument is imported read-only by exact SHA-256.
This addendum creates no new scientific family: A2 adds theta=1.1 at c=4,
and F5-sf evaluates the reviewed contracting isolated sink after removing the
common post-closure contraction scale while retaining the naive route beside
it. No checkpoint, trained weight, GPU, network, controller, queue, or roadmap
is read or modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
from types import ModuleType

import numpy as np


SOURCE_SHA256 = "f8c0884d41cf7a112468013dee6ad70702a7a7be4b6ee700d2100d128aaa2ad4"
ANGLE_CONDITION = 4.0
ANGLE_NAME = "1.1"
ANGLE_THETA = 1.1
DRAW_COUNT = 8
RHO = 0.9
HORIZONS = (64, 256, 1024, 4096)
SIX_DECIMAL_ATOL = 5e-7
ELLIPTIC_ANALYTIC_ATOL = 5e-6

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "scripts" / "preprint_replication_and_factorial_20260903.py"
REVIEW_PATH = PROJECT_ROOT / "prior_art" / "REVIEW_PREPRINT_RERUNS_ANGLE_AND_F5SF_20260903.md"
OUTPUT_DIR = PROJECT_ROOT / "results" / "preprint_reruns_angle_and_f5sf_20260903"


NAIVE_F5_ANCHORS = {
    64: (9.091822739186345, 0.3026766783633771),
    256: (5.898106994535585, 0.0),
    1024: (5.898106994535585, 0.0),
    4096: (5.898106994535585, 0.0),
}
SCALE_FACTORED_ANCHOR = (9.091822739186345, 0.3026766783633771)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reviewed_source() -> ModuleType:
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("reviewed replication source SHA-256 drifted")
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


def measure_a2(source: ModuleType) -> tuple[dict[str, object], list[dict[str, object]]]:
    configuration = "ELL-c4-theta-1.1"

    def maker(draw_index: int):
        return source.make_elliptic(ANGLE_CONDITION, ANGLE_NAME, ANGLE_THETA, draw_index)

    draws = [source.measure_draw(configuration, draw_index, maker) for draw_index in range(DRAW_COUNT)]
    summary = source.summarize_configuration(configuration, draws)
    checks = source.census_controls({configuration: draws})
    distinct_from = (0.7, 2.0, float(np.sqrt(ANGLE_CONDITION)))
    checks.append(control(
        "A2.theta_is_distinct_at_c4",
        ANGLE_THETA,
        f"not in {distinct_from}",
        all(abs(ANGLE_THETA - value) > 1e-12 for value in distinct_from),
        1e-12,
    ))
    expected_max = 2.0 * ANGLE_CONDITION / (1.0 + ANGLE_CONDITION)
    expected_min = 2.0 / (1.0 + ANGLE_CONDITION)
    for draw in draws:
        index = int(draw["draw_index"])
        readout = draw["matrix_readout_n2048"]
        maximum_error = abs(float(readout["lambda_max"]) - expected_max)
        minimum_error = abs(float(readout["lambda_min"]) - expected_min)
        checks.append(control(
            f"A2.draw{index}.elliptic_closed_form_n2048",
            {
                "lambda_max": readout["lambda_max"],
                "lambda_min": readout["lambda_min"],
            },
            {"lambda_max": expected_max, "lambda_min": expected_min},
            max(maximum_error, minimum_error) <= ELLIPTIC_ANALYTIC_ATOL,
            ELLIPTIC_ANALYTIC_ATOL,
        ))
    payload: dict[str, object] = {
        "schema": "preprint_addendum_a2_distinct_angle_v1",
        "configuration": configuration,
        "condition": ANGLE_CONDITION,
        "theta": ANGLE_THETA,
        "theta_distinct_from_original_c4_values": list(distinct_from),
        "draw_count": DRAW_COUNT,
        "matrix_horizon": source.MATRIX_HORIZON,
        "horizons": list(source.HORIZONS),
        "oracle_direction_definition": "top eigenvector of M_2048, fixed across every horizon",
        "draws": draws,
        "summary": summary,
        "flatness_reading": (
            "FLAT_FMC_OBSERVATION_RETAINED"
            if int(summary["flatness_total_failure_count_n_ge_512"]) == 0
            else "FLAT_FMC_OBSERVATION_WITHDRAWN"
        ),
    }
    return payload, checks


def write_a2_plot_ready_tables(source: ModuleType, summary: dict[str, object]) -> None:
    configuration = str(summary["configuration"])
    summary_rows: list[dict[str, object]] = []
    for metric in ("lambda_max", "lambda_min", "trace_over_N", "K_plus", "K_minus"):
        summary_rows.append({
            "configuration": configuration,
            "metric": metric,
            **summary["statistics"][metric],
        })
    source.atomic_csv(
        OUTPUT_DIR / "ANGLE_C4_THETA_1P1_SUMMARY.csv",
        ["configuration", "metric", "median", "q1", "q3", "iqr", "min", "max", "draw_count"],
        summary_rows,
    )
    nj_rows: list[dict[str, object]] = []
    for horizon in source.HORIZONS:
        nj_rows.append({
            "configuration": configuration,
            "horizon": horizon,
            **summary["statistics"][f"nJ_oldest_n{horizon}"],
        })
    source.atomic_csv(
        OUTPUT_DIR / "ANGLE_C4_THETA_1P1_NJ.csv",
        ["configuration", "horizon", "median", "q1", "q3", "iqr", "min", "max", "draw_count"],
        nj_rows,
    )


def scale_factored_f5(source: ModuleType, sink: np.ndarray, horizon: int) -> tuple[np.ndarray, float]:
    """Evaluate F5 in a gauge with the common post-closure rho scale removed.

    During the write window this is the reviewed contracting sink rho*U. After
    closure, dividing the final sink sensitivity by rho**(n-WRITE_WINDOW) and
    its covariance/cross-covariance by the matching powers is equivalent to
    replacing each closed-step rho*U by U. The latter avoids underflow while
    preserving the exact Fisher quadratic form under an invertible congruence.
    """
    open_carrier = source.Wh.copy()
    open_carrier[source.D_C :, source.D_C :] = RHO * sink
    closed_factored = open_carrier.copy()
    closed_factored[source.D_C :, : source.D_C] = 0.0
    closed_factored[source.D_C :, source.D_C :] = sink
    carriers = [
        open_carrier if time_index < source.W_WRITE else closed_factored
        for time_index in range(horizon)
    ]
    noise_covariances = [source.Qiso for _ in range(horizon)]
    return source.fisher_memory_tv(carriers, source.vh, noise_covariances)


def measure_f5sf(source: ModuleType) -> tuple[dict[str, object], list[dict[str, object]]]:
    sink = source.replay_20260823_hybrid()
    rows: dict[str, object] = {}
    checks: list[dict[str, object]] = []
    expected_sf_total, expected_sf_oldest = SCALE_FACTORED_ANCHOR
    for horizon in HORIZONS:
        naive_jk, naive_total = source.gated_run(RHO * sink, horizon)
        factored_jk, factored_total = scale_factored_f5(source, sink, horizon)
        naive_expected_total, naive_expected_oldest = NAIVE_F5_ANCHORS[horizon]
        naive_error = max(
            abs(float(naive_total) - naive_expected_total),
            abs(float(naive_jk[-1]) - naive_expected_oldest),
        )
        factored_error = max(
            abs(float(factored_total) - expected_sf_total),
            abs(float(factored_jk[-1]) - expected_sf_oldest),
        )
        checks.append(control(
            f"F5-naive.reproduce_reviewed_anchor.n{horizon}",
            {"j_total": float(naive_total), "j_oldest": float(naive_jk[-1])},
            {"j_total": naive_expected_total, "j_oldest": naive_expected_oldest},
            naive_error <= SIX_DECIMAL_ATOL,
            SIX_DECIMAL_ATOL,
        ))
        checks.append(control(
            f"F5-sf.scale_factored_constant.n{horizon}",
            {"j_total": float(factored_total), "j_oldest": float(factored_jk[-1])},
            {"j_total": expected_sf_total, "j_oldest": expected_sf_oldest},
            factored_error <= SIX_DECIMAL_ATOL,
            SIX_DECIMAL_ATOL,
        ))
        rows[str(horizon)] = {
            "post_closure_common_scale_exponent": horizon - source.W_WRITE,
            "naive_pseudoinverse": {
                "j_total": float(naive_total),
                "j_oldest": float(naive_jk[-1]),
                "nonzero_j_count": int(np.count_nonzero(naive_jk)),
            },
            "scale_factored": {
                "j_total": float(factored_total),
                "j_oldest": float(factored_jk[-1]),
                "nonzero_j_count": int(np.count_nonzero(factored_jk)),
            },
        }
    sf_totals = [float(rows[str(h)]["scale_factored"]["j_total"]) for h in HORIZONS]
    sf_oldest = [float(rows[str(h)]["scale_factored"]["j_oldest"]) for h in HORIZONS]
    constant = (
        max(sf_totals) - min(sf_totals) <= SIX_DECIMAL_ATOL
        and max(sf_oldest) - min(sf_oldest) <= SIX_DECIMAL_ATOL
        and min(sf_oldest) > 0.0
    )
    checks.append(control(
        "F5-sf.constant_across_horizons",
        {
            "j_total_range": max(sf_totals) - min(sf_totals),
            "j_oldest_range": max(sf_oldest) - min(sf_oldest),
            "minimum_j_oldest": min(sf_oldest),
        },
        "both ranges <= 5e-7 and oldest positive",
        constant,
        SIX_DECIMAL_ATOL,
    ))
    payload: dict[str, object] = {
        "schema": "preprint_addendum_f5_scale_factored_v1",
        "model": "reviewed F5: non-normal chain, isolated rho=0.9 sink, write coupling closes after 24 steps",
        "rho": RHO,
        "write_window_steps": source.W_WRITE,
        "factorization": (
            "keep rho*U during t=0..23; after closure replace rho*U by U, which is the numerically stable "
            "gauge obtained by dividing final sink sensitivity by rho^(n-24), sink covariance by "
            "rho^(2(n-24)), and cross-covariance by rho^(n-24)"
        ),
        "horizons": list(HORIZONS),
        "rows": rows,
        "pre_registered_reading": (
            "CONTRACTING_ISOLATED_SINK_FISHER_RETENTION_CONSTANT_NAIVE_ZERO_IS_PINV_ARTIFACT"
            if constant
            else "BETWEEN_PREREGISTERED_BRANCHES"
        ),
    }
    return payload, checks


def provenance(started: float, source: ModuleType) -> dict[str, object]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "reviewed_source_path": str(SOURCE_PATH),
        "reviewed_source_sha256_expected": SOURCE_SHA256,
        "reviewed_source_sha256_observed": sha256_file(SOURCE_PATH),
        "source_of_source_hashes": {
            "fourth_cell_cond_sweep_20260902.py": source.COND_SWEEP_SHA256,
            "verify_ganguli_pocket_capacity_20260823.py": source.GANGULI_SHA256,
        },
        "review_file": os.environ.get("PREPRINT_REPRODUCTION_REVIEW_FILE", "NOT_ESTABLISHED"),
        "numpy_version": np.__version__,
        "float_dtype": "float64",
        "elapsed_seconds": float(time.perf_counter() - started),
        "resource_scope": "local CPU only; no GPU, checkpoint, trained weight, network, controller, queue, or roadmap",
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

    print("MEASURE A2: c=4, theta=1.1, 8 draws", flush=True)
    a2, a2_checks = measure_a2(source)
    print("MEASURE F5-sf: naive and scale-factored routes", flush=True)
    f5sf, f5sf_checks = measure_f5sf(source)
    checks = a2_checks + f5sf_checks
    valid = all(bool(item["pass"]) for item in checks)
    status = "VALID_MEASUREMENT" if valid else "INSTRUMENT_INVALID"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prov = provenance(started, source)
    a2["run_status"] = status
    a2["provenance"] = prov
    f5sf["run_status"] = status
    f5sf["provenance"] = prov
    controls_payload: dict[str, object] = {
        "schema": "preprint_addendum_angle_and_f5sf_positive_controls_v1",
        "run_status": "PASS" if valid else "INSTRUMENT_INVALID",
        "failure_count": sum(1 for item in checks if not item["pass"]),
        "checks": checks,
        "invalid_run_rule": (
            "any draw-count, trace, oracle-consistency, theorem, finite-power, distinct-angle, elliptic closed-form, "
            "naive-anchor, scale-factored-anchor, or cross-horizon-constancy failure invalidates the full run"
        ),
        "provenance": prov,
    }
    receipt: dict[str, object] = {
        "schema": "preprint_addendum_angle_and_f5sf_receipt_v1",
        "run_status": status,
        "execution_gate": gate_detail,
        "authority_scope": "Addendum A2 and F5-sf reproduction only",
        "a2_configuration": "ELL-c4-theta-1.1, 8 draws",
        "f5sf_routes": ["reviewed naive pseudoinverse", "scale-factored post-closure gauge"],
        "artifacts": [
            "ANGLE_C4_THETA_1P1.json",
            "ANGLE_C4_THETA_1P1_SUMMARY.csv",
            "ANGLE_C4_THETA_1P1_NJ.csv",
            "F5_SCALE_FACTORED.json",
            "POSITIVE_CONTROLS.json",
            "RUN_RECEIPT.json",
        ],
        "provenance": prov,
    }
    source.atomic_json(OUTPUT_DIR / "ANGLE_C4_THETA_1P1.json", a2)
    source.atomic_json(OUTPUT_DIR / "F5_SCALE_FACTORED.json", f5sf)
    source.atomic_json(OUTPUT_DIR / "POSITIVE_CONTROLS.json", controls_payload)
    source.atomic_json(OUTPUT_DIR / "RUN_RECEIPT.json", receipt)
    write_a2_plot_ready_tables(source, a2["summary"])

    if not valid:
        print("INSTRUMENT_INVALID: one or more preregistered controls failed.")
        return 3
    print(f"PASS: wrote reviewed addendum artifacts under {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
