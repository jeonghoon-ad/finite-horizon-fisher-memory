# independent source review PRE-REVIEW — `scripts/preprint_replication_and_factorial_20260903.py` (replicated directional census + write-path × storage factorial)

**Written:** [2026/09/03/08:03] (fresh `TZ=Asia/Seoul date`). **Reviewer:** independent of the implementer.
**Authority chain, stated exactly:** the handoff `the implementer_HANDOFF_PREPRINT_REPLICATION_AND_ISOLATION_FACTORIAL_20260903_EN.md` was written at 06:59 as a draft with no authority. the implementer reports (07:41, forwarded to me by the author) that the author instructed it in its console: "please read this and execute." No separate GO was typed to me in chat. I treat the author's instruction to the implementer, forwarded by the author, as the GO for this CPU-only run; if that reading is wrong, the artifacts are deletable and no decision has been taken on them.
**Script:** SHA-256 `03775a6ee34e241446b2e2d03587b65bde3a90cd3c6814d19a0f7cda9030f8be`, 964 lines, read in full. **Manifest:** `scripts/MIRROR_OF.json` SHA-256 `f59bf9819162dce1ce50d8594f206a5d50a56fa5900daec736f5fc8e48c63c6d`.
**review scope:** admits or blocks an instrument; no track verdict.

## 1. Instrument identity (L-066)
Parsed function bodies compared: **10/10 identical** to the reviewed `fourth_cell_cond_sweep_20260902.py` (`6c0abe5e…3814e`) and **4/4 identical** to `verify_ganguli_pocket_capacity_20260823.py` (`34ff3513…8903`): `build_chain`, `build_hybrid`, `fisher_memory_tv`, `gated_run`. Constants `EPS, N, W_WRITE=24, D_C=D_L=16, NH` match the sources. Both source SHAs are pinned and checked at runtime. The wrapper's **MIRROR_DIFF** file (`prior_art/MIRROR_DIFF_preprint_replication_and_factorial_20260903_<date>.txt`) is expected to show hunks only outside the two mirrored blocks.

## 2. New code, checked
- **Draw seeding:** `seed_for(MASTER_SEED, configuration, draw, stream)` via SHA-256; every realized seed recorded. Independent per-draw draws of $Q$, $S$ (left/right Haar), elliptic mixer, H8 eigenbasis and O∩Sp mixer. Matches handoff Part A.
- **Configurations:** A0; SQS $c\in\{2,5,10,20,50,100\}$ + held-out $c=35$ (exploratory only); elliptic $D^{-1}R(\theta)D$ at $c\in\{4,100\}\times\theta\in\{0.7,2.0,\sqrt c\}$ (decoupled, as the review asked); H8 $e^{JH}$ at $c\in\{4,100\}$. 16 configurations × 8 draws = 128 carriers.
- **Oracle direction:** top eigenvector of $M_{2048}$, fixed, evaluated at $n\in\{32,128,512,2048,4096\}$ through the mirrored scalar `fisher_memory`; self-consistency (matrix route vs scalar route) gated at 5e-9.
- **Finite-window $K_\pm$:** $\max_{k\le4096}\|W^{\pm k}\|_2$ including $k=0$. These are exactly the constants the theorem's proof needs for horizons $n\le4096$ (only powers $j<n$ enter $C_n$ and $W^kv$), so the every-lag check $1/(nK_+^2K_-^2)\le J_n(k)\le K_+^2K_-^2/n$ is a legitimate finite-window test, labelled as such. Any violation ⇒ INSTRUMENT_INVALID. Correct.
- **Flatness rule** $|nJ_n(n-1)-\lambda_{max}|/\lambda_{max}<5\%$ at $n\ge512$: failure withdraws the observation, does not invalidate the instrument. Correct separation.
- **Exploratory fit:** trains on six SQS medians against $\log K_+$, predicts held-out $c=35$; no threshold; labelled non-preregistered. Correct.
- **Factorial:** time-varying route `fisher_memory_tv_varying(carriers, writes, noises)` — propagator ordering checked ($P_t = W_{n-1}\cdots W_{t+1}$, accumulated by right-multiplying the earlier carrier): correct. F1 = normal direct write $e_1$ into the 16-dim isometric sink with full isotropic noise every step (the §2 model on the sink; analytic control $J_{tot}=1$, $J(n-1)=1/n$). F2 = same, but writes **and** sink noise stop after step 24 (isolation); prediction if the reviewer is right: $J_{tot}=1$ (24 lags at $1/24$) and $J(n-1)=1/24$ constant. F3/F4/F5 = the 2026-08-23 arms through the mirrored functions with the RNG preamble replayed (seed 20260823: $Q$, $v_1$, three Haar draws for $m=1,4,8$, then `build_hybrid(16,16,0.05,κ=1)$); **hard anchors** from the reviewed 08-23 run at all four horizons, tolerance 5e-7 — I checked the anchors against the 08-23 note §4.3–4.4 table (10.60711…10.74091 / 9.0881e-2…1.1849e-3; 8.81531 / 0.30271; 9.09182→5.89811 / 0.30268→0): they match. If the replay order were wrong, these controls fail closed.
- **F2 reading rule** implemented exactly as handoff Part B (constant at 1 vs decay >2× vs between). F2 is not used as a control. Correct.
- **Execution gate:** refuses unless the wrapper selected *this* review file, the file contains the exact script SHA, and exactly one `REPRODUCTION DECISION: RUN`. Correct and stricter than before.
- **Outputs:** `DIRECTIONAL_CENSUS.json`, `ISOLATION_FACTORIAL.json`, `POSITIVE_CONTROLS.json`, `RUN_RECEIPT.json`, two plot-ready CSVs; atomic writes; provenance with both source hashes.
- **Runtime estimate:** 128 draws × (2048-step $M$ + 2×4096 SVDs + five FMC loops) ≈ 2–5 CPU minutes. No GPU, no network, no checkpoint.

## 3. measurement protocol contract — closed
input (oracle $v$ from $M_{2048}$; $e_1$ or the 08-23 chain write) → transform (carrier with normality/symplectic/$K_\pm$ certificates; time-varying $W_t,B_t,G_t$ stated in the receipt) → carrier (mirrored routes) → readout ($M_n$ eigensystem; scalar FMC; time-varying FMC) → statistic ($\lambda_{max/min}$, $nJ_n(n-1)$, $J_{tot}$) → controls (trace/N=1 every draw; A0 identity spectrum; theorem zero violations; F1 analytic; F3/F4/F5 08-23 anchors) → decision (pre-registered readings, suppressed on any control failure).

## 4. Post-run acceptance checks (recorded before execution)
1. `POSITIVE_CONTROLS.json` PASS, `failure_count = 0` (this includes zero theorem violations across all 128 draws and the 08-23 anchors to 5e-7).
2. Census: A0 medians exactly 1; SQS-c10 median $\lambda_{max}$ within the draw spread of the single-carrier value 4.21; flatness reading reported.
3. Factorial: F1 analytic PASS; F2 reading reported as labelled; F3–F5 anchors PASS.
4. Provenance: both source SHAs observed = expected; review file = this file.
5. Any failure → instrument-scope report only.

## 5. Decision
Script SHA bound to this decision: `03775a6ee34e241446b2e2d03587b65bde3a90cd3c6814d19a0f7cda9030f8be`.

**REPRODUCTION DECISION: RUN**

Execution: `PYTHON=projects/so21-bounded-obs/.venv/bin/python bash internal-administration/scripts/run_with_review.sh projects/NMH-RNN/scripts/preprint_replication_and_factorial_20260903.py`. Results to `results/preprint_replication_and_factorial_20260903/`; then `RESULT_PREPRINT_REPLICATION_AND_ISOLATION_FACTORIAL_20260903_EN.md` with scope. The author rules.

## Public derivative identity

The review above describes the historical source it names. The public copy changes only packaging metadata, reproduction labels and dependency pins. It is not a new independent review. The original source and its corresponding packaged derivative are bound below; MANIFEST.json records the full file identities.

- Historical source `03775a6ee34e241446b2e2d03587b65bde3a90cd3c6814d19a0f7cda9030f8be`; packaged derivative `f8c0884d41cf7a112468013dee6ad70702a7a7be4b6ee700d2100d128aaa2ad4`.
