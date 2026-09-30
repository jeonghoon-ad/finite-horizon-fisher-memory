#!/usr/bin/env python3
"""Stage runner for the NC trained-memory validation lane.

Stages
------
smoke         one carrier draw, one optimizer seed, one writer type, shortened
              training, plus the full validity battery and an open/isolated
              parity check.  Produces SMOKE_CONFIG.json and SMOKE_RECEIPT.json.
pilot         the preregistered pilot grid over amplitude x training horizon on
              the two excluded pilot carrier draws.  Produces PILOT_GRID.csv and
              a frozen-parameter proposal.
confirmatory  the frozen 16 x 2 x 5 matrix.  Produces the per-cell records, the
              carrier-level CSVs and a RUN_RECEIPT.json.

Nothing in this file may change an endpoint, a threshold, a sample count or a
decision rule; those live in the authority documents and in nc_verdict.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nc_geometry as G           # noqa: E402
import nc_experiment as E         # noqa: E402
import nc_model as M              # noqa: E402
import nc_validity as V           # noqa: E402

LANE = Path(__file__).resolve().parents[1]
RESULTS = LANE / "results"

# Frozen preregistered training budget (YAML: confirmatory.*).
FROZEN = {
    "training_steps": 12000,
    "batch_size": 512,
    "learning_rate": 0.003,
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
    "optimizer": "Adam",
    "optimizer_seeds": 5,
    "test_horizons": list(E.TEST_HORIZONS),
    "test_episodes": 50000,
    "calibration_episodes": 50000,
    "validation_episodes": 10000,
    "validate_every": 1000,
    "loss_every": 50,
    "random_direction_count": E.RANDOM_DIRECTION_COUNT,
}

PILOT_AMPLITUDES = (1.0, 1.5, 2.0, 2.5)
PILOT_HORIZONS = (128, 256)

# Decimal places the pilot selection statistic is rounded to before sorting.
# Chosen to sit far above float64 round-off (~1e-16) and far below any
# difference that could matter scientifically.
SELECTION_PRECISION = 12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_manifest() -> dict[str, str]:
    code = Path(__file__).resolve().parent
    return {p.name: sha256_file(p) for p in sorted(code.glob("*.py"))}


def blas_identity() -> dict[str, Any]:
    """Which LAPACK actually built the carriers.  See G.carrier_fingerprint."""
    try:
        config = np.show_config(mode="dicts")
    except Exception:
        return {"available": False}
    build = config.get("Build Dependencies", {}) if isinstance(config, dict) else {}
    return {"available": True, "blas": build.get("blas"), "lapack": build.get("lapack")}


def carrier_fingerprints(draws) -> list[dict[str, Any]]:
    return [G.carrier_fingerprint(G.build_carrier_draw(d)) for d in draws]


def environment() -> dict[str, Any]:
    return {
        "blas": blas_identity(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "numpy_threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes, Linux reports kilobytes.
    return int(usage) if sys.platform == "darwin" else int(usage) * 1024


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_config(amplitude: float, train_horizon: int, **overrides: Any) -> dict[str, Any]:
    config = dict(FROZEN)
    config["amplitude"] = float(amplitude)
    config["train_horizon"] = int(train_horizon)
    config.update(overrides)
    config["test_horizons"] = list(config["test_horizons"])
    return config


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------
def open_isolated_parity(draw_index: int, writer_type: str, amplitude: float,
                         episodes: int = 40000) -> dict[str, Any]:
    """Certify the two arms end to end by sampling, not only by algebra.

    At the closure horizon the arms must agree; past it they must not.  Both
    halves are asserted, so an implementation that silently collapsed the open
    arm into the isolated one would fail here.
    """
    draw = G.build_carrier_draw(draw_index)
    W, K, U = draw["W"][writer_type], draw["K"], draw["U"]
    ops = G.system_operators(W, K, U)
    controls = G.direction_controls(ops, draw_index, writer_type, random_count=4)
    v = controls["v_old"]
    out: dict[str, Any] = {}
    for horizon in (G.WRITE_WINDOW, 256):
        P_iso, S_iso, _ = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], horizon)
        P_open, S_open = G.open_endpoint(W, K, U, horizon)
        J_iso = float((P_iso @ v) @ np.linalg.solve(S_iso, P_iso @ v))
        J_open = float((P_open @ v) @ np.linalg.solve(S_open, P_open @ v))
        rng_iso = np.random.default_rng(G.nc_seed(draw_index, f"parity|iso|{horizon}"))
        rng_open = np.random.default_rng(G.nc_seed(draw_index, f"parity|iso|{horizon}"))
        x_iso, y_iso = M.sample_endpoint(rng_iso, P_iso, M.noise_factor(S_iso), v, amplitude, episodes)
        x_open, y_open = M.sample_endpoint(rng_open, P_open, M.noise_factor(S_open), v, amplitude, episodes)
        acc_iso = M.accuracy(x_iso, y_iso, np.linalg.solve(S_iso, P_iso @ v), 0.0)
        acc_open = M.accuracy(x_open, y_open, np.linalg.solve(S_open, P_open @ v), 0.0)
        out[f"H{horizon}"] = {
            "J_isolated": J_iso, "J_open": J_open,
            "accuracy_isolated": acc_iso, "accuracy_open": acc_open,
            "predicted_isolated": G.bayes_accuracy(amplitude, J_iso),
            "predicted_open": G.bayes_accuracy(amplitude, J_open),
            "labels_identical": bool(np.array_equal(y_iso, y_open)),
        }
    closure = out[f"H{G.WRITE_WINDOW}"]
    later = out["H256"]
    out["arms_agree_at_closure"] = bool(abs(closure["J_isolated"] - closure["J_open"]) < 1e-12)
    out["arms_separate_after_closure"] = bool(later["J_open"] < 0.5 * later["J_isolated"])
    out["parity_pass"] = bool(out["arms_agree_at_closure"] and out["arms_separate_after_closure"])
    return out


def stage_smoke(args: argparse.Namespace) -> int:
    G.assert_draw_blocks_disjoint()
    started = time.time()
    outdir = RESULTS / "smoke"
    config = make_config(
        args.amplitude, args.train_horizon,
        optimizer_seeds=1,
        training_steps=args.steps,
        test_episodes=args.episodes,
        calibration_episodes=args.episodes,
        validation_episodes=min(args.episodes, 10000),
        validate_every=max(args.steps // 4, 1),
    )
    write_json(outdir / "SMOKE_CONFIG.json", {
        "stage": "smoke", "config": config,
        "carrier_draw": args.draw, "writer_type": args.writer_type,
        "code_manifest": code_manifest(),
        "vendor_digests": G.vendor_digests(),
        "environment": environment(),
    })

    battery = V.run_battery(args.draw, config["amplitude"], config["train_horizon"])
    parity = open_isolated_parity(args.draw, args.writer_type, config["amplitude"])

    cell_started = time.time()
    cell = E.run_cell(args.draw, args.writer_type, config)
    cell_seconds = time.time() - cell_started

    examples = config["training_steps"] * config["batch_size"] * config["optimizer_seeds"]
    learned = cell["learned"][0]
    receipt = {
        "stage": "smoke",
        "carrier_draw": args.draw,
        "writer_type": args.writer_type,
        "config": config,
        "validity_battery": {
            "verdict": battery["battery_verdict"],
            "controls_run": battery["controls_run"],
            "all_controls_pass": battery["all_controls_pass"],
            "all_instruments_valid": battery["all_instruments_valid"],
            "records": battery["records"],
        },
        "open_isolated_parity": parity,
        "cell_summary": {
            "lambda_max_M_old": cell["lambda_max_M_old"],
            "eigengap_ratio": cell["eigengap_ratio"],
            "leading_subspace_dim": cell["leading_subspace_dim"],
            "oracle_accuracy": cell["oracle_accuracy"],
            "random_accuracy_median": cell["random_accuracy_median"],
            "learned": learned,
            "loss_curve_head": cell["loss_curves"]["s0"][:5],
            "loss_curve_tail": cell["loss_curves"]["s0"][-5:],
        },
        "throughput": {
            "cell_seconds": cell_seconds,
            "training_examples": examples,
            "examples_per_second": examples / max(cell_seconds, 1e-9),
            "seconds_per_trained_model": cell_seconds / config["optimizer_seeds"],
        },
        "peak_rss_bytes": peak_rss_bytes(),
        "elapsed_seconds": time.time() - started,
        "code_manifest": code_manifest(),
        "vendor_digests": G.vendor_digests(),
        "carrier_fingerprints": carrier_fingerprints([args.draw]),
        "environment": environment(),
    }
    receipt["smoke_verdict"] = (
        "PASS" if (battery["battery_verdict"] == "VALID" and parity["parity_pass"]) else "FAIL"
    )
    write_json(outdir / "SMOKE_RECEIPT.json", receipt)
    write_csv(outdir / "SMOKE_ROWS.csv", cell["rows"])
    print(json.dumps({
        "smoke_verdict": receipt["smoke_verdict"],
        "battery": battery["battery_verdict"],
        "controls_run": battery["controls_run"],
        "parity_pass": parity["parity_pass"],
        "cell_seconds": round(cell_seconds, 3),
        "examples_per_second": round(receipt["throughput"]["examples_per_second"], 1),
        "peak_rss_mb": round(receipt["peak_rss_bytes"] / 1e6, 1),
        "R_J": round(learned["rayleigh_efficiency"], 5),
        "oracle_accuracy": round(cell["oracle_accuracy"], 4),
        "random_accuracy": round(cell["random_accuracy_median"], 4),
    }, indent=2))
    return 0 if receipt["smoke_verdict"] == "PASS" else 1


# --------------------------------------------------------------------------
# pilot
# --------------------------------------------------------------------------
def _pilot_point(payload: tuple[int, str, float, int, dict[str, Any]]) -> dict[str, Any]:
    draw_index, writer_type, amplitude, horizon, overrides = payload
    config = make_config(amplitude, horizon, **overrides)
    cell = E.run_cell(draw_index, writer_type, config)
    return {
        "draw_index": draw_index,
        "writer_type": writer_type,
        "amplitude": amplitude,
        "train_horizon": horizon,
        "lambda_max_M_old": cell["lambda_max_M_old"],
        "eigengap_ratio": cell["eigengap_ratio"],
        "oracle_accuracy": cell["oracle_accuracy"],
        "random_accuracy_median": cell["random_accuracy_median"],
        "analytic_oracle_accuracy": cell["analytic_oracle_accuracy"],
        "analytic_random_accuracy_median": cell["analytic_random_accuracy_median"],
        "median_rayleigh_efficiency": float(np.median(
            [x["rayleigh_efficiency"] for x in cell["learned"]])),
        "median_behavioral_efficiency": float(np.median(
            [x["behavioral_efficiency"] for x in cell["learned"]])),
        "median_subspace_alignment": float(np.median(
            [x["subspace_alignment"] for x in cell["learned"]])),
        "calibration_mae": float(np.mean(
            [abs(r["predicted_accuracy"] - r["recalibrated_accuracy"]) for r in cell["rows"]])),
        "max_clip_fraction": float(np.max([x["clip_fraction"] for x in cell["learned"]])),
        "elapsed_seconds": cell["elapsed_seconds"],
    }


def stage_pilot(args: argparse.Namespace) -> int:
    G.assert_draw_blocks_disjoint()
    started = time.time()
    outdir = RESULTS / "pilot"
    overrides = {"optimizer_seeds": 2}
    if args.steps:
        overrides["training_steps"] = args.steps
    if args.episodes:
        overrides["test_episodes"] = args.episodes
        overrides["calibration_episodes"] = args.episodes
    work = [
        (draw, writer_type, amplitude, horizon, overrides)
        for draw in G.PILOT_DRAWS
        for writer_type in G.WRITER_TYPES
        for amplitude in PILOT_AMPLITUDES
        for horizon in PILOT_HORIZONS
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_pilot_point, work))
    write_csv(outdir / "PILOT_GRID.csv", rows)

    # The selection criterion is evaluated on the ANALYTIC accuracies
    # Phi(a sqrt(J)), not on the sampled ones.  Both quantities estimate the
    # same number, but the sampled ones carry Monte Carlo noise, and a noisy
    # criterion makes the tie-break between two algebraically identical
    # training horizons a coin flip (measured P = 0.500) while amendment A3
    # asserts it is deterministic.  On analytic accuracies the two horizons tie
    # EXACTLY, so the declared "then smaller horizon" fallback actually fires.
    window = {"oracle": (0.70, 0.85), "random": (0.55, 0.75)}
    selection: list[dict[str, Any]] = []
    for amplitude in PILOT_AMPLITUDES:
        for horizon in PILOT_HORIZONS:
            cells = [r for r in rows if r["amplitude"] == amplitude and r["train_horizon"] == horizon]
            oracle = [r["analytic_oracle_accuracy"] for r in cells]
            random_acc = [r["analytic_random_accuracy_median"] for r in cells]
            in_window = all(window["oracle"][0] <= o <= window["oracle"][1] for o in oracle) and \
                        all(window["random"][0] <= r <= window["random"][1] for r in random_acc)
            selection.append({
                "amplitude": amplitude, "train_horizon": horizon,
                "cells": len(cells),
                "oracle_min": min(oracle), "oracle_max": max(oracle),
                "random_min": min(random_acc), "random_max": max(random_acc),
                "all_cells_in_window": bool(in_window),
                # Rounded to SELECTION_PRECISION so that options which are
                # algebraically identical actually tie.  Measured: the analytic
                # oracle accuracy at H=128 and H=256 differs by at most 1.1e-16,
                # one ULP, so an unrounded sort lets the last bit of a float
                # choose the training horizon while the amendment claims the
                # rule is deterministic.
                "oracle_window_centre_distance": round(
                    float(np.mean([abs(o - 0.775) for o in oracle])), SELECTION_PRECISION),
            })
    # Declared amendment A2: window violation is measured, not waved through.
    for entry in selection:
        entry["oracle_violation"] = float(
            max(0.0, window["oracle"][0] - entry["oracle_min"])
            + max(0.0, entry["oracle_max"] - window["oracle"][1]))
        entry["random_violation"] = float(
            max(0.0, window["random"][0] - entry["random_min"])
            + max(0.0, entry["random_max"] - window["random"][1]))
        entry["total_violation"] = round(
            entry["oracle_violation"] + entry["random_violation"], SELECTION_PRECISION)
    write_csv(outdir / "PILOT_SELECTION.csv", selection)
    eligible = [s for s in selection if s["all_cells_in_window"]]
    eligible.sort(key=lambda s: (s["oracle_window_centre_distance"], s["amplitude"], s["train_horizon"]))
    # If the preregistered window is infeasible for every grid point, the run
    # does not silently widen it.  It selects by the smallest measured
    # violation, labels the outcome AMENDED_SELECTION, and records the exact
    # violation so the Founder can see what was conceded.
    selection_mode = "PREREGISTERED_WINDOW" if eligible else "BLOCKED_NO_GRID_POINT_IN_WINDOW"
    if not eligible:
        # Amendment A2: the lane does NOT auto-proceed outside the preregistered
        # window.  It names the least-violating point as a proposal for the
        # Founder and marks the stage BLOCKED.  stage_confirmatory refuses to
        # run on a blocked pilot unless the Founder decision is passed
        # explicitly on the command line.
        fallback = sorted(selection, key=lambda s: (
            s["total_violation"], s["oracle_window_centre_distance"],
            s["amplitude"], s["train_horizon"]))
        chosen = fallback[0]
    else:
        chosen = eligible[0]
    receipt = {
        "stage": "pilot",
        "pilot_draws": list(G.PILOT_DRAWS),
        "grid": {"amplitudes": list(PILOT_AMPLITUDES), "train_horizons": list(PILOT_HORIZONS)},
        "selection_window": window,
        "points_run": len(rows),
        "eligible": eligible,
        "selection_mode": selection_mode,
        "proposed": chosen,
        "all_grid_points": selection,
        "overrides": overrides,
        "peak_rss_bytes": peak_rss_bytes(),
        "elapsed_seconds": time.time() - started,
        "code_manifest": code_manifest(),
        "vendor_digests": G.vendor_digests(),
        "environment": environment(),
    }
    write_json(outdir / "PILOT_RECEIPT.json", receipt)
    print(json.dumps({"points": len(rows), "eligible": len(eligible),
                      "selection_mode": selection_mode,
                      "proposed": receipt["proposed"],
                      "elapsed_seconds": round(receipt["elapsed_seconds"], 1)}, indent=2))
    return 0


# --------------------------------------------------------------------------
# confirmatory
# --------------------------------------------------------------------------
def _confirmatory_cell(payload: tuple[int, str, dict[str, Any]]) -> dict[str, Any]:
    """One confirmatory cell, with its OWN validity battery.

    Protocol section 11: "Any failed validity control invalidates the affected
    run before scientific interpretation."  That is per run, not once for the
    whole matrix, so the battery is attached to every cell and its verdict
    travels with the cell's data.
    """
    draw_index, writer_type, config = payload
    battery = V.run_battery(draw_index, config["amplitude"], config["train_horizon"])
    try:
        cell = E.run_cell(draw_index, writer_type, config)
    except Exception as error:
        # Protocol section 11 says a failed cell is INVALIDATED, not that the
        # matrix is destroyed.  Inside a process pool an uncaught raise takes
        # down all 32 cells and writes nothing, so the failure is captured and
        # ledgered here instead.
        return {
            "draw_index": draw_index, "writer_type": writer_type,
            "rows": [], "swap_rows": [], "retention_rows": [], "learned": [],
            "loss_curves": {}, "learned_vectors": {}, "learned_readouts": {},
            "elapsed_seconds": 0.0,
            "validity": {
                "verdict": "INVALID",
                "controls_run": battery["controls_run"],
                "all_controls_pass": battery["all_controls_pass"],
                "all_instruments_valid": battery["all_instruments_valid"],
                "failed": [f"cell_raised:{type(error).__name__}"],
                "instruments_that_cannot_fail": [],
                "records": battery["records"],
                "exception": f"{type(error).__name__}: {error}",
            },
        }
    cell["validity"] = {
        "verdict": battery["battery_verdict"],
        "controls_run": battery["controls_run"],
        "all_controls_pass": battery["all_controls_pass"],
        "all_instruments_valid": battery["all_instruments_valid"],
        "failed": [r["name"] for r in battery["records"] if not r["pass"]],
        "instruments_that_cannot_fail": [
            r["name"] for r in battery["records"] if not r.get("instrument_valid", True)],
        "records": battery["records"],
    }
    return cell


def stage_confirmatory(args: argparse.Namespace) -> int:
    """The frozen matrix.  Its amplitude and training horizon come from the
    pilot receipt, never from a typed-in number."""
    started = time.time()
    outdir = RESULTS / "confirmatory"
    pilot_receipt_path = RESULTS / "pilot" / "PILOT_RECEIPT.json"
    if not pilot_receipt_path.exists():
        raise RuntimeError(
            "no pilot receipt: the confirmatory stage may not choose its own "
            f"amplitude or training horizon (expected {pilot_receipt_path})")
    pilot = json.loads(pilot_receipt_path.read_text())
    proposal = pilot["proposed"]
    if pilot["selection_mode"] != "PREREGISTERED_WINDOW" and not args.accept_amended_selection:
        raise RuntimeError(
            f"pilot selection mode is {pilot['selection_mode']}: no grid point placed "
            "all pilot cells inside the preregistered window. This is a Founder "
            "decision, not a runner default. Re-run with --accept-amended-selection "
            "only after that decision is recorded. Least-violating proposal was "
            f"{proposal}")
    amplitude = float(proposal["amplitude"])
    train_horizon = int(proposal["train_horizon"])
    if args.amplitude and abs(args.amplitude - amplitude) > 1e-12:
        raise RuntimeError(
            f"--amplitude {args.amplitude} contradicts the pilot proposal {amplitude}")
    if args.train_horizon and args.train_horizon != train_horizon:
        raise RuntimeError(
            f"--train-horizon {args.train_horizon} contradicts the pilot proposal {train_horizon}")
    config = make_config(amplitude, train_horizon)
    if args.steps:
        config["training_steps"] = args.steps
    if args.episodes:
        config["test_episodes"] = args.episodes
        config["calibration_episodes"] = args.episodes

    G.assert_draw_blocks_disjoint()
    work = [(draw, writer_type, config)
            for draw in G.CONFIRMATORY_DRAWS for writer_type in G.WRITER_TYPES]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cells = list(pool.map(_confirmatory_cell, work))

    invalid: list[dict[str, Any]] = []
    for stale in ("DIRECTION_ROWS.csv", "SWAP_ROWS.csv", "RETENTION_ROWS.csv",
                  "CELL_LEVEL.csv", "VERDICT.json", "RUN_RECEIPT.json",
                  "INVALID_RUN_LEDGER.json"):
        # A run that fails wholesale must not leave the previous run's artifacts
        # sitting next to a new receipt.
        (outdir / stale).unlink(missing_ok=True)

    all_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    per_cell_dir = outdir / "cells"
    per_cell_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        if cell["validity"]["verdict"] != "VALID":
            invalid.append({
                "draw_index": cell["draw_index"], "writer_type": cell["writer_type"],
                "verdict": cell["validity"]["verdict"],
                "failed": cell["validity"]["failed"],
                "instruments_that_cannot_fail": cell["validity"]["instruments_that_cannot_fail"],
                "exception": cell["validity"].get("exception"),
            })
            continue      # an invalidated cell contributes no scientific row
        all_rows.extend(cell["rows"])
        swap_rows.extend(cell["swap_rows"])
        retention_rows.extend(cell["retention_rows"])
        for learned in cell["learned"]:
            cell_rows.append({
                "draw_index": cell["draw_index"], "writer_type": cell["writer_type"],
                "lambda_max_M_old": cell["lambda_max_M_old"],
                "eigengap_ratio": cell["eigengap_ratio"],
                "leading_subspace_dim": cell["leading_subspace_dim"],
                "oracle_accuracy": cell["oracle_accuracy"],
                "random_accuracy_median": cell["random_accuracy_median"],
                **learned,
            })
        write_json(
            per_cell_dir / f"CELL_draw{cell['draw_index']:03d}_{cell['writer_type']}.json",
            {k: v for k, v in cell.items() if k not in ("rows", "swap_rows", "retention_rows")},
        )

    write_csv(outdir / "DIRECTION_ROWS.csv", all_rows)
    write_csv(outdir / "SWAP_ROWS.csv", swap_rows)
    write_csv(outdir / "RETENTION_ROWS.csv", retention_rows)
    write_csv(outdir / "CELL_LEVEL.csv", cell_rows)

    receipt = {
        "stage": "confirmatory",
        "config": config,
        "parameter_source": {
            "file": str(pilot_receipt_path),
            "selection_mode": pilot["selection_mode"],
            "proposal": proposal,
            "amended_selection_accepted": bool(args.accept_amended_selection),
        },
        "carrier_draws": list(G.CONFIRMATORY_DRAWS),
        "writer_types": list(G.WRITER_TYPES),
        "trained_models": len(G.CONFIRMATORY_DRAWS) * len(G.WRITER_TYPES) * config["optimizer_seeds"],
        "cells": len(cells),
        "direction_rows": len(all_rows),
        "swap_rows": len(swap_rows),
        "retention_rows": len(retention_rows),
        "pilot_draws_excluded": list(G.PILOT_DRAWS),
        "reserved_draw_blocks": {k: list(v) for k, v in G.DRAW_BLOCKS.items()},
        "draw_block_collisions": G.draw_block_collisions(),
        "carrier_fingerprints": carrier_fingerprints(G.CONFIRMATORY_DRAWS),
        "invalid_runs": invalid,
        "valid_cells": len(cells) - len(invalid),
        "validity_controls_per_cell": (
            cells[0]["validity"]["controls_run"] if cells else 0),
        "run_valid": bool(not invalid),
        "peak_rss_bytes": peak_rss_bytes(),
        "elapsed_seconds": time.time() - started,
        "cell_seconds": {f"{c['draw_index']}_{c['writer_type']}": c["elapsed_seconds"] for c in cells},
        "code_manifest": code_manifest(),
        "vendor_digests": G.vendor_digests(),
        "environment": environment(),
    }
    write_json(outdir / "RUN_RECEIPT.json", receipt)
    write_json(outdir / "INVALID_RUN_LEDGER.json",
               {"invalid_runs": invalid, "count": len(invalid),
                "rule": "protocol section 11: a failed validity control invalidates the "
                        "affected run before scientific interpretation"})
    print(json.dumps({"cells": len(cells), "valid_cells": receipt["valid_cells"],
                      "invalid_runs": len(invalid),
                      "trained_models": receipt["trained_models"],
                      "direction_rows": len(all_rows),
                      "elapsed_seconds": round(receipt["elapsed_seconds"], 1),
                      "peak_rss_mb": round(receipt["peak_rss_bytes"] / 1e6, 1)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "pilot", "confirmatory"))
    parser.add_argument("--draw", type=int, default=G.PILOT_DRAWS[0])
    parser.add_argument("--writer-type", default="nonnormal", choices=list(G.WRITER_TYPES))
    parser.add_argument("--amplitude", type=float, default=0.0,
                        help="smoke/pilot only; the confirmatory stage reads the pilot receipt")
    parser.add_argument("--train-horizon", type=int, default=0,
                        help="smoke/pilot only; the confirmatory stage reads the pilot receipt")
    parser.add_argument("--accept-amended-selection", action="store_true",
                        help="requires a recorded Founder decision; see docs/RUN_AUTHORITY.md")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = parser.parse_args()
    if args.stage == "smoke":
        args.amplitude = args.amplitude or 2.0
        args.train_horizon = args.train_horizon or 128
        args.steps = args.steps or 1500
        args.episodes = args.episodes or 20000
        return stage_smoke(args)
    if args.stage == "pilot":
        return stage_pilot(args)
    return stage_confirmatory(args)


if __name__ == "__main__":
    raise SystemExit(main())
