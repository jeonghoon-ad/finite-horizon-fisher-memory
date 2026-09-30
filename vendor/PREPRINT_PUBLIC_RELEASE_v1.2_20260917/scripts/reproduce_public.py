#!/usr/bin/env python3
"""Public reproduction entry point for the v1.2 candidate package.

The original decision-bearing scripts retain their internal execution checks.
This wrapper binds the included review records to the exact scripts for the
three 2026-09-03 gated runs, and sets the historical gate flag for older runs.
Run inside a disposable copy of the package because original scripts may
rewrite their result directories.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

TASKS = {
    "instrument": ("scripts/verify_ganguli_pocket_capacity_20260823.py", None),
    "first-census": ("scripts/fourth_cell_capacity_20260902.py", None),
    "condition-sweep": ("scripts/fourth_cell_cond_sweep_20260902.py", None),
    "replicated-census": (
        "scripts/preprint_replication_and_factorial_20260903.py",
        "prior_art/REVIEW_PREPRINT_REPLICATION_AND_ISOLATION_FACTORIAL_20260903.md",
    ),
    "angle-control": (
        "scripts/preprint_reruns_angle_and_f5sf_20260903.py",
        "prior_art/REVIEW_PREPRINT_RERUNS_ANGLE_AND_F5SF_20260903.md",
    ),
    "factorial": (
        "scripts/isolation_factorial_v2_bounded_20260903.py",
        "prior_art/REVIEW_ISOLATION_FACTORIAL_V2_BOUNDED_20260903.md",
    ),
    "strengthening": ("scripts/preprint_v1_2_strengthening_20260917.py", None),
    "figures": ("scripts/make_preprint_figures_20260903.py", None),
}


def run_task(name: str) -> None:
    script, review = TASKS[name]
    env = os.environ.copy()
    if name not in {"instrument", "strengthening", "figures"}:
        env["PREPRINT_REPRODUCTION_WRAPPER_PASSED"] = "1"
    if review is not None:
        env["PREPRINT_REPRODUCTION_REVIEW_FILE"] = str((ROOT / review).resolve())
    print(f"\n=== {name}: {script} ===", flush=True)
    subprocess.run([PYTHON, str(ROOT / script)], cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["strengthening", "figures", "core", "full"],
        default="strengthening",
    )
    args = parser.parse_args()
    if args.stage == "strengthening":
        names = ["strengthening"]
    elif args.stage == "figures":
        names = ["figures"]
    elif args.stage == "core":
        names = ["replicated-census", "angle-control", "factorial", "strengthening", "figures"]
    else:
        names = list(TASKS)
    for name in names:
        run_task(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
