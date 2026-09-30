#!/usr/bin/env python3
"""Produce the two parity artifacts that existed only as prose.

A number quoted in a manuscript whose producing computation has no artifact is not a
receipt. Two such numbers were found by the canonical-source inventory:

  * the Blelloch scan parity (9.7e-14 / 6.2e-15), which lived only in a note; and
  * the decoder-transport restoration (3.4e-14), which lived only inside a reviewer's
    report, and which the abstract's last clause rests on.

Both are recomputed here from the frozen carriers and written to
`results/parity/`. Evaluation-only: no model is trained, no gate is touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nc_geometry as G       # noqa: E402
import nc_model as M          # noqa: E402
import nc_scan as SC          # noqa: E402
import run_stage as R         # noqa: E402

OUT = R.LANE / "results" / "parity"
HORIZONS = (64, 256, 1024, 4096)
TRAIN_HORIZON = 128
AMPLITUDE = 1.5


def scan_parity() -> dict:
    """Blelloch scan against the canonical sequential closure_transfers."""
    rows = []
    for draw_index in G.CONFIRMATORY_DRAWS[:4]:
        draw = G.build_carrier_draw(draw_index)
        for writer_type in G.WRITER_TYPES:
            W, K, U = draw["W"][writer_type], draw["K"], draw["U"]
            for steps in (G.WRITE_WINDOW, 256, 1024, 4096):
                check = SC.parity_check(W, K, U, steps, G.closure_transfers)
                check.update({"draw_index": draw_index, "writer_type": writer_type})
                rows.append(check)
    worst_t = max(r["max_relative_transfer_error"] for r in rows)
    worst_c = max(r["max_relative_covariance_error"] for r in rows)
    return {
        "artifact": "blelloch_scan_parity",
        "what": ("every prefix of the closure recurrence computed by a work-efficient "
                 "parallel scan, against the frozen release's sequential implementation"),
        "cells": rows, "cells_checked": len(rows),
        "worst_relative_transfer_error": worst_t,
        "worst_relative_covariance_error": worst_c,
        "tolerance": 1e-11,
        "verdict": "PASS" if (worst_t <= 1e-11 and worst_c <= 1e-11) else "FAIL",
        "note": ("the canonical sequential implementation remains the reference; the scan "
                 "is a third independent implementation kept for correctness, not speed"),
    }


def decoder_transport() -> dict:
    """Is the fixed decoder's deficit away from H_train ENTIRELY coordinate drift?

    The store applies the known invertible hold `A = U^(H - H_train)` between the
    training horizon and the query horizon. A decoder `w` fitted at `H_train` is tied to
    that frame. Transporting it by the inverse adjoint, `w_H = A^{-T} w`, should restore
    exactly the training-horizon decision on every episode, because

        w_H . x_H = (A^{-T} w) . (A x) = w . x .

    If the restoration is exact, the deficit contains no lost information. This is the
    computation the abstract's last clause rests on, and it had no artifact.
    """
    rows = []
    for draw_index in G.CONFIRMATORY_DRAWS[:4]:
        draw = G.build_carrier_draw(draw_index)
        for writer_type in G.WRITER_TYPES:
            W, K, U = draw["W"][writer_type], draw["K"], draw["U"]
            ops = G.system_operators(W, K, U)
            controls = G.direction_controls(ops, draw_index, writer_type, random_count=2)
            v = controls["v_old"]

            P0, S0, _ = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], TRAIN_HORIZON)
            R0 = M.noise_factor(S0)
            # A decoder genuinely fitted at the training horizon, from data.
            rng = np.random.default_rng(G.nc_seed(draw_index, f"transport|fit|{writer_type}"))
            x_fit, y_fit = M.sample_endpoint(rng, P0, R0, v, AMPLITUDE, 50000)
            w0, b0 = M.fit_lda(x_fit, y_fit)

            eps_rng = np.random.default_rng(G.nc_seed(draw_index, f"transport|test|{writer_type}"))
            eps, labels = M.sample_shared_noise(eps_rng, G.STORE_DIM, 50000)
            x0 = labels[:, None] * (AMPLITUDE * (P0 @ v))[None, :] + eps @ R0.T
            base = M.accuracy(x0, labels, w0, b0)

            for horizon in HORIZONS:
                A = np.linalg.matrix_power(U, horizon - TRAIN_HORIZON)
                # The SAME episodes, carried through the known hold.
                xh = x0 @ A.T
                fixed = M.accuracy(xh, labels, w0, b0)
                transported = np.linalg.solve(A.T, w0)
                restored = M.accuracy(xh, labels, transported, b0)
                unwarped = M.accuracy(np.linalg.solve(A, xh.T).T, labels, w0, b0)
                Ph, Sh, _ = G.isolated_endpoint_from(U, ops["L0"], ops["C_ss"], horizon)
                ph = Ph @ v
                rows.append({
                    "draw_index": draw_index, "writer_type": writer_type,
                    "horizon": horizon,
                    "accuracy_at_train_horizon": base,
                    "accuracy_fixed_decoder": fixed,
                    "accuracy_transported_decoder": restored,
                    "accuracy_state_unwarped": unwarped,
                    "restoration_error_transport": abs(restored - base),
                    "restoration_error_unwarp": abs(unwarped - base),
                    "fixed_decoder_deficit": base - fixed,
                    "J_at_train_horizon": float((P0 @ v) @ np.linalg.solve(S0, P0 @ v)),
                    "J_at_this_horizon": float(ph @ np.linalg.solve(Sh, ph)),
                    "max_abs_logit_difference_transported": float(np.max(np.abs(
                        (xh @ transported + b0) - (x0 @ w0 + b0)))),
                })
    worst_restore = max(r["restoration_error_transport"] for r in rows)
    worst_logit = max(r["max_abs_logit_difference_transported"] for r in rows)
    worst_J = max(abs(r["J_at_this_horizon"] - r["J_at_train_horizon"]) /
                  max(r["J_at_train_horizon"], 1e-300) for r in rows)
    return {
        "artifact": "decoder_transport_parity",
        "what": ("whether the fixed decoder's deficit away from the training horizon is "
                 "entirely coordinate drift, tested by transporting the same decoder "
                 "through the known invertible hold"),
        "rows": rows, "rows_checked": len(rows),
        "worst_restoration_error_accuracy": worst_restore,
        "worst_logit_difference": worst_logit,
        "worst_relative_J_deviation": worst_J,
        "median_fixed_decoder_deficit": float(np.median([r["fixed_decoder_deficit"] for r in rows])),
        "tolerance_logit": 1e-9,
        "verdict": "PASS" if worst_logit <= 1e-9 else "FAIL",
        "reading": ("the transported decoder reproduces the training-horizon logit on "
                    "every episode, so the deficit carries no lost information: it is "
                    "decoder coordinate obsolescence, not memory loss"),
    }


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    scan = scan_parity()
    transport = decoder_transport()
    for payload, name in ((scan, "SCAN_PARITY.json"), (transport, "DECODER_TRANSPORT_PARITY.json")):
        payload["environment"] = R.environment()
        payload["code_manifest"] = R.code_manifest()
        payload["elapsed_seconds"] = time.time() - started
        R.write_json(OUT / name, payload)
    print(json.dumps({
        "scan_parity": {k: scan[k] for k in ("cells_checked", "worst_relative_transfer_error",
                                             "worst_relative_covariance_error", "verdict")},
        "decoder_transport": {k: transport[k] for k in (
            "rows_checked", "worst_restoration_error_accuracy", "worst_logit_difference",
            "worst_relative_J_deviation", "median_fixed_decoder_deficit", "verdict")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
