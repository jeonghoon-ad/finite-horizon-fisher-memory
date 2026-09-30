# independent source review PRE-REVIEW — `scripts/preprint_reruns_angle_and_f5sf_20260903.py` (addendum reruns: third distinct angle at c = 4; F5 scale-factored)

**Written:** [2026/09/03/09:49] (fresh `TZ=Asia/Seoul date`). **Reviewer:** independent of the implementer.
**Authority:** author GO "go" 2026-09-03 08:52 (chat, in answer to the two-item addendum), reconfirmed to the implementer in its console per the implementer's 09:43 self-review. Scope: A2 and F5-sf only.
**Script under review:** SHA-256 `76fab286d9defe93ce718e0e18151efb589d02146b880e4065247603619f4ce3`, 385 lines, read in full. **Manifest:** `scripts/MIRROR_OF.json` entry present, primary source pinned to the reviewed `preprint_replication_and_factorial_20260903.py` (`03775a6e…f8be`).
**review scope:** admits or blocks an instrument; no track verdict.

## 1. Instrument identity (L-066)
This script does **not** copy functions; it imports the reviewed replication script as a read-only module after checking its SHA-256 at runtime (`load_reviewed_source`), and calls its `make_elliptic`, `measure_draw`, `summarize_configuration`, `census_controls`, `replay_20260823_hybrid`, `gated_run`, `fisher_memory_tv`, `atomic_json`, and constants unchanged. That is a stronger form of mirroring than a verbatim copy: the mirrored code cannot drift because it is the reviewed file itself. The wrapper's generated **MIRROR_DIFF** file (`prior_art/MIRROR_DIFF_preprint_reruns_angle_and_f5sf_20260903_<date>.txt`) will therefore show the whole new file as a diff against the primary source; that is expected and carries no meaning beyond "no copied block exists".

## 2. New code, checked
- **A2:** configuration `ELL-c4-theta-1.1`; $\theta=1.1$ checked distinct from $\{0.7, 2.0, \sqrt4=2.0\}$; the same 8-draw protocol, oracle definition (top eigenvector of $M_{2048}$), horizons, $K_\pm$ certificate, every-lag theorem check and 5% flatness reading, all through the reviewed functions. Added per-draw closed-form controls $\lambda_{max}=1.6$, $\lambda_{min}=0.4$ at 5e-6 (the reviewed c = 4 draws deviate < 2e-7 from the limit). Correct.
- **F5 scale-factored:** during the 24-step write window the carrier is the reviewed one with sink $0.9U$; after closure the coupling is zero and no noise enters the sink (`Qiso` is chain-only), so the post-closure carrier is block-diagonal $\mathrm{diag}(W_{chain}, 0.9U)$. Replacing it by $\mathrm{diag}(W_{chain}, U)$ multiplies every post-closure propagator by $T_k=\mathrm{diag}(I, 0.9^{-k}I)$ on the sink rows; every term of the final sensitivity and of the final covariance (write-window noise propagated through closure, and post-closure chain noise, which $T$ leaves untouched) is transformed by the same $T_{n-24}$, i.e. $p\to Tp$, $C\to TCT^\top$, an invertible congruence under which $J=p^\top C^{-1}p$ is exactly invariant (Theorem 5.1 of the draft). **The factorization is exact, not approximate.** The naive route (`gated_run(0.9\cdot sink)`) is kept beside it and must reproduce the reviewed anchors, so the artifact stays documented.
- **Anchors:** naive anchors are the reviewed F5 values at four horizons; the scale-factored anchor is the reviewed $n=64$ naive value (before underflow), pre-registered as the constant by the invariance theorem; a separate cross-horizon constancy control (ranges ≤ 5e-7, oldest > 0). These are instrument controls with pre-declared expectations, not outcome selection.
- **Gate:** refuses unless the wrapper selected this review, the file contains the exact SHA, and exactly one RUN line. Six artifacts, atomic writes, provenance with the source-of-source hashes.
- **Runtime:** 8 elliptic draws + 8 time-varying FMC evaluations; seconds.

## 3. measurement protocol — closed
input (oracle $v$; 08-23 write vector) → transform (elliptic carrier with certificates; F5 time-varying carrier with the stated post-closure gauge) → carrier (reviewed routes) → readout → statistic ($\lambda_{max/min}$, $nJ_n(n-1)$; $J_{tot}$, $J(n-1)$ naive and factored) → controls (trace/N, A0-style identities, theorem, closed form, naive anchors, factored anchor, constancy) → decision (pre-registered readings; suppressed on failure).

## 4. Post-run acceptance checks (recorded before execution)
1. `POSITIVE_CONTROLS.json` PASS, `failure_count = 0`.
2. A2: $\lambda_{max}$ median 1.600 on 8 draws; flatness retained; theorem violations 0.
3. F5: naive reproduces (9.09182, 0.302677) at $n=64$ and (5.89811, 0) at $n\ge256$; scale-factored reproduces (9.09182, 0.302677) at all four horizons.
4. Provenance: source SHA observed = expected; review = this file.

## 5. Decision
Script SHA bound: `76fab286d9defe93ce718e0e18151efb589d02146b880e4065247603619f4ce3`.

**REPRODUCTION DECISION: RUN**

## Public derivative identity

The review above describes the historical source it names. The public copy changes only packaging metadata, reproduction labels and dependency pins. It is not a new independent review. The original source and its corresponding packaged derivative are bound below; MANIFEST.json records the full file identities.

- Historical source `76fab286d9defe93ce718e0e18151efb589d02146b880e4065247603619f4ce3`; packaged derivative `b634bf2ef7fa47eb5fa8262c0ae560a84ecf30e219805f65d55984b12938a364`.
