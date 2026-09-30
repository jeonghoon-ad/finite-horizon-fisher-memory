#!/usr/bin/env python3
"""NC-3C stage runner: freshness audit, per-cell validity, the frozen 320-model matrix.

Authority: ../authority/nc3c/NC3C_TASK_SPECIFIC_GEOMETRY_PROTOCOL_v1_20260918_EN.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nc_geometry as G           # noqa: E402
import nc_validity as V           # noqa: E402
import nc3c_experiment as X       # noqa: E402
import run_stage as R             # noqa: E402

RESULTS = R.LANE / "results" / "nc3c"

# Launcher amendment, 2026-09-20 (AGENT_HANDOFF_APPROVED_FINAL_RUN_AND_AD023_FILING §4.3).
# The Founder approved one further confirmation on NC-3C-V1 draws 16..31. The block and the
# output directory were hard-coded here; they are now selectable, and the freshness audit
# additionally compares the requested block against every NC-3C draw this lane has already
# instantiated. No scientific module is touched: the carrier construction, the seeds, the
# validity battery, the budget and the decision rules are the frozen ones.
PRIOR_NC3C_DRAWS = tuple(X.CARRIER_DRAWS)          # the 2026-09-18 full-budget block
BURNED_NC3C_DRAWS = (9900,)                         # timing probe, docs/nc3c/BURNED_DRAWS_20260919.md

FROZEN = {
    "amplitude": 1.5,
    "train_horizon": 128,
    "training_steps": 12000,
    "batch_size": 512,
    "learning_rate": 0.003,
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
    "optimizer": "Adam",
    "optimizer_seeds": 5,
    "test_episodes": 50000,
    "calibration_episodes": 50000,
    "loss_every": 50,
    "objectives": list(X.OBJECTIVES),
    "negligible_separation_threshold": X.NEGLIGIBLE_SEPARATION,
}


# --------------------------------------------------------------------------
# Added validity controls 30 and 31
# --------------------------------------------------------------------------
def control_objective_distinctness(draw_index: int, writer_type: str) -> dict[str, Any]:
    """Control 30. The two objectives must be genuinely different OPERATORS.

    This is an IMPLEMENTATION control: did the run actually build two operators from two
    different injection times, or is it comparing a matrix with itself? The negative
    control sets both target times to 0, which makes the difference exactly zero.

    It deliberately does NOT use the 0.02 negligibility threshold. An earlier version of
    this control failed a cell whose oracle separation `G*` fell below that threshold,
    which contradicted protocol section 8.3 and handoff section 6.7: those say such a
    carrier is excluded from the CONTRAST RATIO ONLY and is retained in the own-objective
    and calibration analyses. Invalidating the cell instead would make that clause
    unreachable and would let one degenerate carrier out of sixteen void a 320-model run.
    The handoff is the authority; the control was wrong and is now a record.

    `G*_0`, `G*_12` and the oracle overlap are recorded here for every cell, and the 0.02
    rule is enforced downstream, at the contrast statistic, where section 8.3 puts it.
    """
    def run(corrupt: bool) -> dict[str, Any]:
        draw = X.build_carrier_draw(draw_index)
        ops = G.system_operators(draw["W"][writer_type], draw["K"], draw["U"])
        times = (0, 0) if corrupt else X.OBJECTIVES
        built = []
        for t0 in times:
            M = ops["M_old"] if t0 == 0 else G.target_operator(ops, t0)
            values, vectors = G.eig_descending(M)
            built.append((M, float(values[0]), vectors[:, 0]))
        (M_a, lam_a, v_a), (M_b, lam_b, v_b) = built
        scale = max(float(np.linalg.norm(M_a, "fro")), 1e-300)
        difference = float(np.linalg.norm(M_a - M_b, "fro")) / scale
        return V._record("30_objective_operators_are_distinct", difference, 1e-9,
                         difference > 1e-9,
                         {"writer_type": writer_type, "draw_index": draw_index,
                          "target_times": list(times),
                          "oracle_overlap": float(abs(v_a @ v_b)),
                          "G_star_first": 1.0 - float(v_b @ M_a @ v_b) / lam_a,
                          "G_star_second": 1.0 - float(v_a @ M_b @ v_a) / lam_b,
                          "negligibility_threshold_enforced_downstream_at":
                              "contrast statistic, protocol section 8.3",
                          "lambda_max_first": lam_a, "lambda_max_second": lam_b})
    return V._with_negative_control(run)


def control_seed_freshness(draw_index: int) -> dict[str, Any]:
    """Control 31. No NC-3C seed may equal a seed the completed lane used.

    Checked against the completed lane's namespace over every stream name this lane
    uses and every draw index it reserved. The negative control asks the same question
    of the completed lane's own namespace, where the answer must be a collision.
    """
    streams = ["s_left", "s_right", "q", "coupling_rows", "store_u",
               "random_directions|normal", "random_directions|nonnormal"]
    reserved = list(G.CONFIRMATORY_DRAWS) + list(G.PILOT_DRAWS) + list(G.EXPLORATORY_DRAWS)

    # Repair, 2026-09-20. The negative control used to seed the completed lane's namespace
    # at THIS draw index, which collides only when the index is one the completed lane
    # actually reserved. That holds for draws 0..15 and fails for any later block, so on
    # draws 16..31 the instrument could not fail and every cell was marked invalid. The
    # corrupt branch is now mapped onto a reserved index, so the collision it must show is
    # guaranteed by construction for any block. The positive branch is unchanged.
    probe = G.CONFIRMATORY_DRAWS[draw_index % len(G.CONFIRMATORY_DRAWS)]

    def run(corrupt: bool) -> dict[str, Any]:
        if corrupt:
            mine = {G.nc_seed(probe, s) for s in streams}
        else:
            mine = {X.nc3c_seed(draw_index, s) for s in streams}
        theirs = {G.nc_seed(d, s) for d in reserved for s in streams}
        overlap = mine & theirs
        return V._record("31_seed_freshness", float(len(overlap)), 0.0, not overlap,
                         {"draw_index": draw_index, "streams": len(streams),
                          "completed_lane_seeds_compared": len(theirs),
                          "negative_control_probe_draw": probe if corrupt else None,
                          "namespace": G.SEED_NAMESPACE if corrupt else X.SEED_NAMESPACE})
    return V._with_negative_control(run)


def _cell(payload: tuple[int, str, dict[str, Any]]) -> dict[str, Any]:
    draw_index, writer_type, config = payload
    # The battery certifies the carrier this cell MEASURES, not the completed lane's
    # carrier of the same index, and it covers the t_0 axis this experiment varies.
    battery = V.run_battery(
        draw_index, config["amplitude"], config["train_horizon"],
        carrier_builder=X.build_carrier_draw,
        extra_target_times=tuple(t for t in X.OBJECTIVES if t != 0))
    extra = [control_objective_distinctness(draw_index, writer_type),
             control_seed_freshness(draw_index)]
    battery["records"].extend(extra)
    battery["controls_run"] += len(extra)
    battery["all_controls_pass"] = all(r["pass"] for r in battery["records"])
    battery["all_instruments_valid"] = all(
        r.get("instrument_valid", True) for r in battery["records"])
    battery["battery_verdict"] = (
        "VALID" if battery["all_controls_pass"] and battery["all_instruments_valid"] else "INVALID")
    try:
        cell = X.run_cell(draw_index, writer_type, config)
    except Exception as error:
        return {"draw_index": draw_index, "writer_type": writer_type,
                "rows": [], "reference_rows": [], "endpoints": {},
                "learned_vectors": {}, "elapsed_seconds": 0.0,
                "validity": {"verdict": "INVALID", "controls_run": battery["controls_run"],
                             "failed": [f"cell_raised:{type(error).__name__}"],
                             "instruments_that_cannot_fail": [],
                             "records": battery["records"],
                             "exception": f"{type(error).__name__}: {error}"}}
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


def run_stream_names(config: dict[str, Any]) -> list[str]:
    """Every seed stream the run actually draws, enumerated from the code paths.

    Enumerated rather than listed by hand so the audit's coverage cannot drift below
    the run's as streams are added.
    """
    names = ["s_left", "s_right", "q", "coupling_rows", "store_u"]
    for writer_type in G.WRITER_TYPES:
        names.append(f"random_directions|{writer_type}")
        for t0 in X.OBJECTIVES:
            names.append(f"test|{writer_type}|t{t0}")
            names.append(f"cal|{writer_type}|t{t0}")
            for seed_index in range(config["optimizer_seeds"]):
                names.append(f"optimizer|{writer_type}|t{t0}|{seed_index}")
    return names


def frozen_seed_list(config: dict[str, Any],
                     draws: tuple[int, ...] | None = None) -> dict[str, Any]:
    """The complete seed list the protocol says is frozen before any outcome is seen."""
    draws = tuple(X.CARRIER_DRAWS) if draws is None else tuple(draws)
    names = run_stream_names(config)
    seeds = {str(d): {n: X.nc3c_seed(d, n) for n in names} for d in draws}
    flat = [v for row in seeds.values() for v in row.values()]
    return {"namespace": X.SEED_NAMESPACE, "carrier_draws": list(draws),
            "streams_per_draw": len(names), "total_seeds": len(flat),
            "distinct_seeds": len(set(flat)), "stream_names": names, "seeds": seeds}


def freshness_audit(config: dict[str, Any],
                    draws: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Prove, before training, that no seed or carrier of `draws` was seen before.

    Three questions, not one. Is any requested draw already on the exercised list? Does any
    of its seeds collide with the completed NC-TM-V1 lane, or with an NC-3C-V1 draw this lane
    already instantiated? Does any carrier digest collide with either population? The
    exercised list is the 2026-09-18 full-budget block and the draw burned by the 2026-09-19
    timing probe. Nothing here trains, classifies or scores a carrier.

    Re-auditing the 2026-09-18 block therefore reports `fresh: false`, which is the true
    state of the world after that block was run; it is not a regression.
    """
    draws = tuple(X.CARRIER_DRAWS) if draws is None else tuple(draws)
    streams = run_stream_names(config)
    reserved = list(G.CONFIRMATORY_DRAWS) + list(G.PILOT_DRAWS) + list(G.EXPLORATORY_DRAWS)
    prior_nc3c = [d for d in tuple(PRIOR_NC3C_DRAWS) + tuple(BURNED_NC3C_DRAWS)
                  if d not in draws]

    theirs = {G.nc_seed(d, s) for d in reserved for s in streams}
    prior_own = {X.nc3c_seed(d, s) for d in prior_nc3c for s in streams}
    mine = {X.nc3c_seed(d, s) for d in draws for s in streams}

    their_carriers = {G.build_carrier_draw(d)["hashes"]["W_nonnormal"] for d in reserved}
    prior_own_carriers = {X.build_carrier_draw(d)["hashes"]["W_nonnormal"] for d in prior_nc3c}
    my_carriers = {d: X.build_carrier_draw(d)["hashes"]["W_nonnormal"] for d in draws}
    mine_digests = set(my_carriers.values())

    already_exercised = sorted(set(draws) & (set(PRIOR_NC3C_DRAWS) | set(BURNED_NC3C_DRAWS)))
    collisions = {
        "requested_draws_already_exercised": already_exercised,
        "seed_collisions": sorted(mine & theirs),
        "seed_collisions_with_prior_nc3c": sorted(mine & prior_own),
        "carrier_digest_collisions": sorted(mine_digests & their_carriers),
        "carrier_digest_collisions_with_prior_nc3c": sorted(mine_digests & prior_own_carriers),
    }
    return {
        "nc3c_namespace": X.SEED_NAMESPACE,
        "audited_draws": list(draws),
        "streams_enumerated": len(streams),
        "completed_lane_namespace": G.SEED_NAMESPACE,
        "prior_nc3c_draws_compared": prior_nc3c,
        "prior_nc3c_draws_on_record": sorted(set(PRIOR_NC3C_DRAWS) | set(BURNED_NC3C_DRAWS)),
        "nc3c_seeds": len(mine), "completed_lane_seeds": len(theirs),
        "prior_nc3c_seeds": len(prior_own),
        **collisions,
        "nc3c_carrier_digests": my_carriers,
        "distinct_carrier_digests": len(mine_digests),
        "fresh": not any(collisions.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--draws", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--block", default="",
                        help="inclusive carrier-draw range, e.g. 16-31. Default: the frozen block.")
    parser.add_argument("--results-dir", default="",
                        help="output directory. Required when --block selects a different block, "
                             "so a new run cannot overwrite a frozen one.")
    args = parser.parse_args()

    if args.block:
        lo, _, hi = args.block.partition("-")
        block = tuple(range(int(lo), int(hi) + 1)) if hi else (int(lo),)
    else:
        block = tuple(X.CARRIER_DRAWS)

    results = Path(args.results_dir).expanduser().resolve() if args.results_dir else RESULTS
    if block != tuple(X.CARRIER_DRAWS) and results == RESULTS:
        raise SystemExit("a non-default --block must be given its own --results-dir; refusing to "
                         "write a new block into the frozen output directory")

    started = time.time()
    results.mkdir(parents=True, exist_ok=True)
    audit = freshness_audit(dict(FROZEN), block)
    R.write_json(results / "FRESHNESS_AUDIT.json", audit)
    R.write_json(results / "FROZEN_SEED_LIST.json", frozen_seed_list(dict(FROZEN), block))
    print(json.dumps({k: audit[k] for k in (
        "nc3c_namespace", "audited_draws", "requested_draws_already_exercised",
        "nc3c_seeds", "completed_lane_seeds",
        "prior_nc3c_seeds", "seed_collisions", "seed_collisions_with_prior_nc3c",
        "carrier_digest_collisions", "carrier_digest_collisions_with_prior_nc3c",
        "fresh")}, indent=2))
    if not audit["fresh"]:
        raise RuntimeError("NC-3C seeds or carriers are not fresh; the protocol forbids the run")
    if args.audit_only:
        return 0

    overridden = bool(args.steps or args.episodes or args.draws)
    config = dict(FROZEN)
    if args.steps:
        config["training_steps"] = args.steps
    if args.episodes:
        config["test_episodes"] = args.episodes
        config["calibration_episodes"] = args.episodes
    draws = block[:args.draws] if args.draws else block

    for stale in ("CELL_LEVEL.csv", "CROSS_EVALUATION_ROWS.csv", "REFERENCE_ROWS.csv",
                  "VERDICT.json", "RUN_RECEIPT.json", "INVALID_RUN_LEDGER.json"):
        (results / stale).unlink(missing_ok=True)

    work = [(d, w, config) for d in draws for w in G.WRITER_TYPES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cells = list(pool.map(_cell, work))

    invalid, cross_rows, reference_rows, cell_rows = [], [], [], []
    per_cell = results / "cells"
    per_cell.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        if cell["validity"]["verdict"] != "VALID":
            invalid.append({k: cell["validity"].get(k) for k in
                            ("verdict", "failed", "instruments_that_cannot_fail", "exception")}
                           | {"draw_index": cell["draw_index"], "writer_type": cell["writer_type"]})
            continue
        cross_rows.extend(cell["rows"])
        reference_rows.extend(cell["reference_rows"])
        cell_rows.append({"draw_index": cell["draw_index"], "writer_type": cell["writer_type"],
                          "oracle_overlap": cell["oracle_overlap"], **cell["endpoints"]})
        R.write_json(per_cell / f"CELL_draw{cell['draw_index']:03d}_{cell['writer_type']}.json",
                     {k: v for k, v in cell.items() if k not in ("rows", "reference_rows")})

    R.write_csv(results / "CELL_LEVEL.csv", cell_rows)
    R.write_csv(results / "CROSS_EVALUATION_ROWS.csv", cross_rows)
    R.write_csv(results / "REFERENCE_ROWS.csv", reference_rows)
    receipt = {
        "stage": "nc3c_rehearsal" if overridden else "nc3c_confirmatory",
        "carrier_block": list(block), "results_dir": str(results),
        "budget_is_frozen": not overridden,
        "overrides_passed": {"steps": args.steps, "episodes": args.episodes, "draws": args.draws},
        "config": config,
        "carrier_draws": list(draws), "writer_types": list(G.WRITER_TYPES),
        "objectives": list(X.OBJECTIVES),
        "trained_models": len(draws) * len(G.WRITER_TYPES) * config["optimizer_seeds"] * len(X.OBJECTIVES),
        "cells": len(cells), "valid_cells": len(cells) - len(invalid),
        "invalid_runs": invalid, "run_valid": bool(not invalid),
        "validity_controls_per_cell": cells[0]["validity"]["controls_run"] if cells else 0,
        "cross_evaluation_rows": len(cross_rows), "reference_rows": len(reference_rows),
        "freshness_audit": audit,
        "peak_rss_bytes": R.peak_rss_bytes(),
        "elapsed_seconds": time.time() - started,
        "code_manifest": R.code_manifest(), "vendor_digests": G.vendor_digests(),
        "environment": R.environment(),
    }
    R.write_json(results / "RUN_RECEIPT.json", receipt)
    R.write_json(results / "INVALID_RUN_LEDGER.json",
                 {"invalid_runs": invalid, "count": len(invalid)})
    print(json.dumps({"cells": len(cells), "valid_cells": receipt["valid_cells"],
                      "invalid_runs": len(invalid), "trained_models": receipt["trained_models"],
                      "cross_rows": len(cross_rows),
                      "elapsed_seconds": round(receipt["elapsed_seconds"], 1),
                      "peak_rss_mb": round(receipt["peak_rss_bytes"] / 1e6, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
