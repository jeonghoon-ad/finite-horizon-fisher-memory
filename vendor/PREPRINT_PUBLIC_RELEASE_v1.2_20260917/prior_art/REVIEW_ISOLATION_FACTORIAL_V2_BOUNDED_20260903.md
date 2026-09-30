# independent source review PRE-REVIEW — `scripts/isolation_factorial_v2_bounded_20260903.py` (Isolation Factorial v2 on the bounded class)

**Written:** [2026/09/03/14:00] (revised; first version 13:19 was BLOCK on the defect in §2) (fresh `TZ=Asia/Seoul date`). **Reviewer:** independent of the implementer.
**Authority:** author trigger of the frozen handoff `research_notes/handoffs/the implementer_HANDOFF_ISOLATION_FACTORIAL_V2_BOUNDED_CLASS_20260903_EN.md` (the implementer console, 2026-09-03 ~12:5x), reported by the implementer at 13:10.
**Script under review:** SHA-256 `69acea303c4c14e1ce094a3b5c0acbf78fdab62d06cefcca5f1f2cbb07a7887e`, 815 lines (the implementer's `5b2fd952…b94fa48`, 802 lines, read in full, plus the reviewer patch below). **Manifest:** `scripts/MIRROR_OF.json` entry present, primary source pinned to the reviewed `preprint_replication_and_factorial_20260903.py` (`03775a6e…f8be`, SHA re-verified by me). **Mirror diff:** the wrapper will generate `prior_art/MIRROR_DIFF_isolation_factorial_v2_bounded_20260903_20260903.txt` at run time; the script copies no block, it imports the reviewed module read-only and executes the reviewed `make_similarity` code object with only its `seed_for` global rebound (code-object identity asserted at runtime).
**review scope:** admits or blocks an instrument; no track or manuscript verdict.

## 1. What is correct (checked line by line)
- **Design fidelity to the handoff.** 32-dim write block, paired `Q` vs `SQS^{-1}` (c = 10) with the same `Q` per draw (reconstructed from the recorded q seed and hash-checked against the reviewed constructor's Q hash), 16-dim Haar-isometric store, row-orthonormal coupling with operator norm 1, input `[v;0]`, noise `diag(I_32, 0)` at every step in every cell, coupling live for `t<24` then zero (isolated) or always live (open). Exactly one thing changes per axis. Oracle = top eigenvector of `M_2048` of the non-normal carrier (sign-canonicalized), shared with the paired normal cell; seeded random direction shared across the four cells of a draw.
- **Block reverse recurrence** (`reverse_step`): for `P=[[P_w,0],[L,P_s]]` and `W_t=[[W_w,0],[K_t,A_t]]`, `PW_t=[[P_wW_w,0],[LW_w+P_sK_t,P_sA_t]]`; covariance accumulates `P·diag(I,0)·P^T` = `[[P_wP_w^T, P_wL^T],[LP_w^T, LL^T]]`; propagated write `(P_wv, Lv)`. All three match the code. Delay ordering (closed steps first, then 24 open steps, captured per horizon) is the correct reverse of `fisher_memory_tv_varying`; reach maps input time `t` to delay `n-1-t`. Full-state parity against the reviewed route is enforced at n = 64 on all draws and at all horizons on draw 0 (fail-closed).
- **Tolerances attainable.** I evaluated the reviewed `fisher_memory_matrix` on two paired draws: `|M_n(Q)-I|_max` = 2.8e-14 at n = 4096 (gate 1e-12); `tr/N-1` ≤ 4e-15 for both carriers (gate 1e-9). 16 evaluations in 0.14 s; the whole run is well under a CPU-minute.
- **Readings** implement the handoff's four pre-registered readings on the oracle sub-arm; aggregation choices (median of paired ratios; (max−min)/|mean| over four horizon medians; n64/n4096 median ratio; per-draw reach count at n = 4096) are acceptable resolutions of the prose and are recorded in the artifact. Readings suppressed on any control failure; no auto-rerun; atomic writes; provenance complete.

## 2. Blocking defect (one) — RESOLVED by reviewer patch on author "do" (13:2x)
**The Theorem 5.1 control cannot fail as written.** Lines 482–484 build `postclose_contract_factored` with the call `route_snapshots(write_carrier, coupling, store, store, True, directions)`, which is byte-identical to `baseline_isolated` (lines 479–481). The "post-closure 0.9U evaluated in the scale-factored gauge" is therefore the baseline compared with itself, and the control `postclosure_contraction_invariance` (lines 543–553) is a tautology. The handoff's control 3 asks for the *physical* post-closure map `0.9U` to reproduce the isometric cell; the theorem is the claim that it does, and the instrument must be able to observe it failing. (The scale-factored gauge is legitimate as a *presentation* of F5 in the earlier script, where the physical naive route was run beside it and anchored; here nothing physical is run.)

**Reviewer patch applied (independent reviewer, on author instruction; the implementer had not yet edited; precedent: the one-token `.item()` patch on the cond-sweep script, 2026-09-02):** lines 482–488 now build `postclose_contract_physical` with closed map `0.9 * store`; the control gates the store-only curves at n ∈ {64, 256, 1024} and the full-state curve at n = 64, and records the rest; the variant key is `postclosure_0.9U_physical` with an explanatory note. `py_compile` PASS. Diff confined to those three regions; nothing else changed. The fix as specified before the patch was:
1. Replace lines 482–484 with a physical arm: `postclose_contract_physical = route_snapshots(write_carrier, coupling, store, 0.9 * store, True, directions)` (open map `U`, closed map `0.9U`).
2. Control: **store-only** curves must match `baseline_isolated` to 1e-9 at n ∈ {64, 256, 1024} (the store block is uniformly scaled by `0.9^{n-24}`, so the relative pseudoinverse is unaffected; at n = 4096 the scale `0.9^{2·4072}` underflows float64 to 0, which is the finite-precision clause: record n = 4096 as `UNDERFLOW_EXPECTED`, not pass/fail). **Full-state** curves must match to 1e-9 at n = 64 only (store/write scale 0.9^80 ≈ 2e-4, above rcond); at n ≥ 256 the store block falls below the 1e-14 relative pseudoinverse threshold (0.9^232 ≈ 2e-11) and must be recorded, not gated.
3. Keep the existing factored variant if desired, relabelled `gauge_identity_by_construction` and excluded from pass/fail.
4. Everything else unchanged. Report the new SHA; I will re-review the diff only.

## 3. Non-blocking notes (record in the receipt, no code change required)
- `identity_postclose` (closed map `I`): by Appendix A.7 the **full-state** curves should also be invariant, not only store-only; the script records the full difference (`full_curve_max_abs_difference_from_U`). Expect ≤ 1e-9 there too; if not, that is a finding to report, not a control failure.
- `write_contract_factored` (0.9U during the write window, U after) is a valid "must differ" control; its post-closure part is gauge and its write-window part is the physical difference.
- `nonzero_lag_count` counts exact zeros only; fine for the block recurrence, where unreached lags are exactly zero.

## 4. measurement protocol chain (for the re-review)
input (oracle / random `v`) → transform (paired `Q` / `SQS^{-1}`; K; U) → carrier (block recurrence, parity-checked against the reviewed route) → activation/intervention (coupling closure at 24; store map variants) → observation (full-state and store-only Fisher forms) → readout (curves, totals, oldest, reach) → probe (physical 0.9U post-closure; identity store; write-window contraction) → null (normal carrier: `M_n=I`; trace identity both carriers) → statistic (four pre-registered readings) → decision (labels only on 0 control failures). The probe link is the one broken above.

## 5. Post-run acceptance checks (recorded before execution)
1. `POSITIVE_CONTROLS.json` PASS, `failure_count = 0`, including all `postclosure_contraction_invariance` gates and every parity control.
2. Normal cells: `M_n = I` to 1e-12 at all horizons; trace identity both carriers.
3. Isolated cells: store-only oldest-lag constant across horizons (spread < 1e-6); open cells decay ≥ 10×.
4. Recorded, not gated: identity-store full-state difference (expect ≤ 1e-9); physical-contraction full-state difference at n ≥ 256 and store-only at n = 4096 (expect large, by underflow).
5. Provenance: source SHA observed = expected; review = this file; `elapsed_s` under 60.

**REPRODUCTION DECISION: RUN**

## 6. Post-run outcome [2026/09/03/14:02]
All five acceptance checks of §5 met: VALID_MEASUREMENT, 296/296 controls, 0 failures; normal $M_n=I$ to 1e-12 and trace identity at all horizons; isolated store-only oldest-lag spread 0.0, open decay 88–119×; identity-store full-state difference ≤ 1.7e-15 (recorded); physical-contraction differences large exactly at the predicted underflow/threshold horizons (recorded); source SHA observed = expected; elapsed 7.8 s. Result note: `research_notes/RESULT_ISOLATION_FACTORIAL_V2_BOUNDED_20260903_EN.md`.

## Public derivative identity

The review above describes the historical source it names. The public copy changes only packaging metadata, reproduction labels and dependency pins. It is not a new independent review. The original source and its corresponding packaged derivative are bound below; MANIFEST.json records the full file identities.

- Historical source `69acea303c4c14e1ce094a3b5c0acbf78fdab62d06cefcca5f1f2cbb07a7887e`; packaged derivative `5aa11fa435d85883f9b9f16521abc1ec58fc2514e8022ffbc6d3294c435a2ccb`.
