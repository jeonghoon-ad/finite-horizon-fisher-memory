#!/usr/bin/env python3
"""Numerical check of the identification in manuscript section 3.3.0.

Claim:  M_inf = A_C(W^T)^{-1}, where A_C(T) = lim_n (1/n) sum_{j<n} (T^T)^j T^j.
Equivalently, with G_n = C_n / n and G = lim G_n,

    || M_n - G_n^{-1} ||_2  ->  0     and     W G W^T = G .

This is a verification of a classical identity on the carriers this paper uses, not a new
result. It computes no gate, no statistic and no verdict, and it writes nothing: it prints a
residual table so a reader can see the convergence rather than take the identification on
trust. The rate is O(1/n): each doubling of n roughly halves the residual.

    python code/check_cesaro_identification.py [--draw 0] [--horizons 2500,5000,10000,20000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nc_geometry as G  # noqa: E402


def residuals(W: np.ndarray, n: int) -> tuple[float, float]:
    """(|| M_n - G_n^{-1} ||_2, || W G_n W^T - G_n ||_2) for one horizon."""
    N = W.shape[0]
    C = np.zeros((N, N))
    P = np.eye(N)
    for _ in range(n):
        C += P @ P.T
        P = P @ W
    Gn = C / n
    C_inv = np.linalg.inv(C)
    M = np.zeros((N, N))
    P = np.eye(N)
    for _ in range(n):
        M += P.T @ C_inv @ P
        P = P @ W
    return (float(np.linalg.norm(M - np.linalg.inv(Gn), 2)),
            float(np.linalg.norm(W @ Gn @ W.T - Gn, 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--horizons", default="2500,5000,10000,20000")
    args = parser.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]

    draw = G.build_carrier_draw(args.draw)
    print(f"carrier draw {args.draw}, N = {G.WRITE_DIM}")
    for writer_type in G.WRITER_TYPES:
        W = draw["W"][writer_type]
        print(f"\n  {writer_type}")
        print(f"  {'n':>8s} {'||M_n - G_n^-1||_2':>22s} {'||W G_n W^T - G_n||_2':>24s} {'ratio':>8s}")
        previous = None
        for n in horizons:
            fisher, fixed_point = residuals(W, n)
            ratio = "" if previous is None else f"{previous / fisher:8.2f}"
            print(f"  {n:8d} {fisher:22.3e} {fixed_point:24.3e} {ratio:>8s}")
            previous = fisher
    print("\nA ratio near 2 for each doubling of n is the expected O(1/n) convergence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
