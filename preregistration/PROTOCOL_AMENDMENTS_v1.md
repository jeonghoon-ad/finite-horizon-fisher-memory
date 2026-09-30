# Protocol Amendments v1 — declared BEFORE the confirmatory run

**Lane:** NC-TM-V1
**Declared:** 2026-09-18, before any confirmatory carrier draw was trained or inspected.

Each item below is a place where the preregistration is silent, ambiguous, or
self-contradictory. None of them changes an endpoint, a threshold, a sample count or
a kill rule. Each states the reading this lane adopts and why the alternative was
rejected. The pilot draws (900, 901) are excluded from confirmation, so any number
quoted here from a pilot draw cannot contaminate the confirmatory result.

---

## A1 — Pilot cell structure

**Silence.** Protocol 4.4 says the pilot uses "two carrier draws and two optimizer
seeds". It does not say whether the writer type is crossed.

**Adopted.** A carrier draw is a *pair*: one draw yields both the normal writer `Q`
and the non-normal writer `S Q S^-1` sharing `Q`, `K` and `U`. This is forced by the
confirmatory matrix `16 x 5 x 2 = 160`, which counts writer type as a crossing of
the 16 draws, not as extra draws. The pilot therefore runs
`2 draws x 2 writer types x 2 seeds = 8 models` per grid point, and the selection
window must hold in **all four** `(draw, writer type)` cells. This is the strictest
available reading.

## A2 — One amplitude, not one per writer type

**Risk.** The manuscript's section 5.2 table reports oldest-stored-input information
of `0.03375` (normal) against `0.18823` (conditioned) for a *shared* oracle
direction, a factor of 5.6 in `J`. A single frozen amplitude cannot put both writer
types inside `Acc in [0.70, 0.85]` if that factor also holds for each type's own
`lambda_max(M_old)`.

**Adopted.** One amplitude, shared, as protocol 4.4's singular "`a` and `H_train`
are frozen" requires. Feasibility is a pilot measurement, not an assumption. If no
grid point satisfies all four pilot cells, the stage reports `BLOCKED` and the
decision returns to the author; the lane does **not** silently switch to a
per-writer-type amplitude.

## A3 — `H_train` is algebraically inert under exact isolation

**Fact, not a choice.** Under the primary training condition the coupling is exactly
zero for `t >= T_w` and the store map `U` is orthogonal, so
`A_h = U^(H - T_w)` acts on signal and covariance alike and

```
J_H = (A_h L_0 v)^T (A_h C_ss A_h^T)^-1 (A_h L_0 v) = v^T L_0^T C_ss^-1 L_0 v = J_{T_w}
```

exactly, for every `H >= T_w`. Therefore:

1. The pilot grid over `H_train in {128, 256}` cannot change any Fisher or accuracy
   quantity. It is effectively one-dimensional over `a`. The tie is broken by the
   deterministic rule already in `run_stage.stage_pilot`: smallest mean distance of
   the oracle accuracy from the window centre `0.775`, then smaller amplitude, then
   smaller horizon.
2. **Decision rule 10.3 item 1** ("isolated covariance-aware accuracy varies by at
   most 0.01 across query horizons") is an *algebraic identity checked at Monte
   Carlo resolution*, not an empirical discovery. It certifies the implementation.
   It is reported as such and must not be written up as evidence for the retention
   theorem.

## A4 — Statistic of the paired permutation test

**Silence.** Protocol 9.3 names "paired permutation tests as secondary" without
naming the statistic. Sections 9.2 and 10 are median-centric.

**Adopted.** The mean of the paired differences, the canonical sign-flip statistic.
Under sign flips the *median* of the paired differences takes only the values
`{-d, 0, +d}` and has essentially no resolution: by exact enumeration over 8 carrier
draws with a unanimous constant difference, the median-based p-value is `0.7266`
while the mean-based p-value is `0.0078`. A secondary test that cannot reject an
effect present in every carrier draw is unusable. This touches no primary gate.

## A5 — Family structure and Holm scope

**Ambiguity.** Protocol 9.3 orders Holm correction "over the two primary families:
alignment and behavior". The decision rules are stated per carrier draw without
saying how the two writer types combine.

**Adopted.** Every gate is evaluated **separately for each writer type**, and the
overall reading requires both. Holm therefore runs over four p-values
(`alignment|normal`, `alignment|nonnormal`, `behavior|normal`,
`behavior|nonnormal`). Requiring both types is the conservative direction: a lane
that passed on one type only would be reported as `PARTIAL`, never as a pass.

## A6 — What "recalibrated readout" means

**Ambiguity.** Protocol 4.9.1 says "refit only the linear readout for each
direction"; 5.4.1 says "covariance-aware recalibrated linear discriminant"; 4.7 E4
says "using a separately calibrated linear readout".

**Adopted.** A pooled-within-class linear discriminant fit on an **independent
calibration sample** of 50,000 episodes drawn from a dedicated seed stream, never on
the test sample and never from the analytic covariance. For equal-covariance
Gaussians with equal priors this is exactly the Bayes-optimal linear rule, so it is
"covariance-aware" in the protocol's sense while remaining a fit from data. Handing
the readout `Sigma^-1 p` directly would make the "empirical" accuracy a restatement
of the analytic prediction and would void endpoint E4.

## A7 — Where E3 and E4 are read

**Silence.** E3 does not name a condition or horizon; E4 says "every evaluated
direction and horizon".

**Adopted.** `R_beh` is read in the training condition (isolated, at `H_train`) with
the recalibrated readout, because that is the only condition the optimizer saw. The
E4 calibration MAE is computed over **all** evaluated rows: every direction, both
storage conditions, every test horizon plus `H_train`.

## A8 — `v_bottom` is exactly uninformative by rank, not by geometry

**Fact.** `M_old = L_0^T C_ss^-1 L_0` has rank at most `d = 16` in a
`N = 32`-dimensional write space, so its bottom 16 eigenvectors carry `J = 0`
exactly and accuracy `0.5`. **Decision rule 10.2 item 3** ("`v_bottom` performs below
the random-direction median in at least 14 of 16 draws") is therefore close to
automatic. It is reported as a rank fact and must not be written up as evidence that
the geometry is causal.

## A9 — Gradient clipping is active on essentially every step

**Measured on a pilot draw.** With the preregistered clip of `1.0`, the raw global
gradient norm has median about `2.4` and maximum about `9.1`, so the clip binds on
effectively 100% of steps and the optimizer is in practice Adam on a
norm-normalised gradient. The preregistered value is implemented unchanged; the clip
fraction and the raw-norm quantiles are recorded in every receipt so that a reader
can see this.

## A10 — Scope limit on what a PASS can mean

**Adversarial note, carried into the report.** The trained system is linear with
Gaussian noise and a linear readout, and the training objective is a convex
surrogate for the same discrimination problem whose optimum is the leading
eigenvector of `M_old`. A PASS therefore certifies that the *implemented* end-to-end
training recovers the *analytically predicted* direction. It is a bridge and
consistency result. It is not evidence that trained recurrent networks in general
find Fisher-optimal directions, and protocol section 15's forbidden-conclusion list
applies in full.

## A11 — Initial writer state and the process-noise window

**Silence with a 287-fold consequence.** Protocol 3.1 never states `x_w(0)`, and 4.3
says only "draw independent writer process noise at every step".

**Adopted.** `x_w(0) = x_s(0) = 0` deterministically; `z_t ~ N(0, I_N)` is drawn for
`t = 0, ..., T_w - 1` only, so `z_0` enters at the same step as the signal and no
noise precedes `t_0`. `C_ss` is the closure covariance over exactly that transfer
family. This is the canonical implementation's convention and the manuscript's
section 2 hypothesis ("the initial state is known, `x_0 = 0`").

**Why it matters.** Measured on one draw with `v_old`: `J = 0.28700` under
`x_w(0) = 0`, `0.06413` under a 24-step burn-in, `0.00100` under a 2048-step burn-in
matched to `M_2048`. The third choice puts the entire experiment at chance, and it
is the one an implementer "matching the write-block horizon" would reach for.

## A12 — The calibration MAE gate applies to the recalibrated readout only

**Contradiction.** Protocol 4.7 E4 says "for every evaluated direction and horizon",
and 10.1 item 5 / 10.3 item 2 gate the MAE at 0.02. But `Phi(a sqrt J)` is the Bayes
accuracy, reachable only by a covariance-aware readout, while 4.9 mode 2 and 5.4
mode 2 mandate a **fixed** readout that the paper's own theory predicts will drift.

**Adopted.** The MAE gate is computed on the recalibrated readout only. Fixed-readout
error is reported in full under the protocol's own `READOUT_MISMATCH` label (kill
rule 4), extended from NC-2 to the NC-1 direction-swap table. **No threshold moved.**

**Why the alternative was rejected.** Measured fixed-decoder absolute error across
horizons reached `0.2824`, fourteen times the gate, with accuracy `0.4445` below
chance because the rotated sensitivity flips sign against a stale decoder. Read
literally, the gate converts a predicted and understood phenomenon into a mandated
kill of the Fisher-to-behaviour claim.

## A13 — The selection window is a pilot criterion, not a per-carrier admission gate

**Apparent trap.** Measured `lambda_max(M_old)` over 32 cells has normal median
`0.14344` and non-normal median `0.24521`, so `a = 1.5` leaves a minority of normal
cells just below `0.70` and `a = 2.0` pushes about a quarter of non-normal cells
above `0.85`. Protocol 10.4 forbids a new amplitude grid after inspection, which
appears to leave a confirmatory draw outside the window with no legal exit.

**Adopted.** There is no trap: protocol 4.4 states the window as the *selection
criterion for `a`*, and nothing in section 10 or 11 invalidates a confirmatory draw
that lands outside it. Out-of-window confirmatory cells are valid data and are
reported with their measured oracle and random accuracies. If the pilot itself finds
no grid point placing all four pilot cells inside both windows, `run_stage` selects
the smallest measured total violation, labels the result `AMENDED_SELECTION`, and
records the exact violation for the author. The window is never widened.

## A14 — `gate_schedule`

**Undefined.** The YAML lists `gate_schedule` under `frozen_parameters`; the term
appears nowhere in the protocol or the manuscript.

**Adopted.** `gate_schedule` is read as the coupling schedule `K_t`: `K_t = K` for
`t < T_w`, and thereafter `0` (isolated) or `K` (open). It is frozen and is the only
thing distinguishing the two storage arms, which validity control 11 asserts by
showing the arms are identical at closure and separate afterwards.

## A15 — Equivalence margin for kill rule 3

**Undefined.** Kill rule 3 turns on whether the paired normal system "matches" the
non-normal system, with no definition of "matches".

**Adopted.** Equivalence means the paired within-draw accuracy difference has a 95%
bootstrap interval contained in `+/- 0.01`. Recorded in advance: the analytic gap is
known before any training, since `lambda_max(M_old)` differs by roughly `0.145`
against `0.245` across draws, so this rule is a consistency check on the
implementation rather than a discovery. **No non-normal task-advantage claim is made
either way** — protocol section 15 forbids it.

## A16 — Reproduction is defined on invariants, not on carrier bytes

**Measured limit.** `haar_orthogonal` builds `Q` from `np.linalg.qr`, which is not
bit-reproducible across LAPACK implementations. Identical seeds on macOS/Accelerate
and Linux/OpenBLAS gave carriers agreeing to `9.5e-15` absolute and
`lambda_max(M_old)` to `1.8e-15` relative, with **different SHA-256 digests**.

**Adopted.** Reproduction is defined on the portable invariants recorded by
`nc_geometry.carrier_fingerprint` (`lambda_max(M_old)`, `lambda_2/lambda_1`,
`tr M_store`, `cond C_ss`, the coupling's top singular value, the similarity
condition number), rounded to 12 significant figures. Array digests are recorded as
machine-specific provenance, and every receipt carries the BLAS identity of the
machine that produced it. The canonical construction is not changed.

## A17 — Bootstrap specification

**Under-specified.** Protocol 10.1 item 2 says "95% bootstrap lower bound `R_J >=
0.80`" without naming the statistic or the sidedness.

**Adopted.** The one-sided 5th percentile of a 10,000-resample percentile bootstrap
of the **median across carrier draws**, resampling carrier draws with replacement
with optimizer seeds held inside their draw. One-sided, not the low end of a
two-sided 95% interval: conflating them would silently loosen the gate. Reported with
the full per-draw scatter, because with 16 units the percentile bound is granular.

## A18 — Declared exploratory arms

**Not preregistered, not decision-bearing, reported separately.** Added because four
preregistered decision-rule items cannot fail (see `IMPLEMENTATION_AUDIT.md` section
2), and a lane containing only unfalsifiable gates is not worth running.

- **X1, second target time.** Train on an input injected at `t_0 = T_w/2`. Falsifiable
  prediction: the trained direction aligns with `eigmax(M_{t_0})` and not with
  `eigmax(M_0)`, and beats `v_old` on `M_{t_0}`. The two oracles are nearly
  orthogonal (measured overlap `0.0502` on a pilot draw), so a learner that always
  returns the same "best memory direction" fails.
- **X2, non-invertible disposal.** Protocol 5.2 item 4's "optional" arm, given a fixed
  specification: a rank-8 projection of the 16-dimensional store. Prediction:
  information is genuinely destroyed, unlike the invertible hold of manuscript 6.1.
- **X3, high-resolution open arm.** The open arm at `H = 1024` and `4096` is only a
  few Monte Carlo standard errors from chance at the preregistered 50,000 episodes.
  X3 repeats those cells at 500,000 episodes. **The preregistered 50,000 remains the
  primary reading**; X3 only answers whether "correct decay" is distinguishable from
  "signal identically zero".

## A19 — Rejected reviewer suggestion, recorded

An auditor proposed implementing the covariance-aware recalibrated readout as the
analytic `Sigma_h^{-1} p_h`. **Rejected.** That would make the "empirical" accuracy
an algebraic restatement of `Phi(a sqrt J)` and would void endpoint E4, whose entire
content is the comparison of a prediction against a measurement. The pooled-covariance
discriminant fit on a disjoint 50,000-episode calibration sample satisfies both the
"refit the linear readout" and the "covariance-aware discriminant" readings while
remaining an estimate from data (amendment A6).

## A20 — The pilot selection criterion is evaluated on ANALYTIC accuracies

**Defect in this lane's first implementation, found in review.** The criterion was
evaluated on sampled accuracies. Both training horizons are algebraically identical
under isolation, so the choice between them was made by Monte Carlo noise: an
independent reviewer measured `P(H_train = 256 beats 128) = 0.500` and
`P(the amplitude flips off 1.5) = 0.073`, while amendment A3 asserted the rule was
deterministic.

**Adopted.** The window and the tie-break are evaluated on the analytic accuracies
`Phi(a sqrt(J))`, which are noise-free and exactly horizon-invariant, and the
selection statistic is rounded to **12 decimal places** before sorting. Measured
justification: the analytic oracle accuracy at `H = 128` and `H = 256` differs by at
most `1.1e-16`, one ULP, so without rounding the last bit of a float chooses the
training horizon. With rounding the two horizons tie exactly and the declared "then
smaller horizon" fallback actually fires. Sampled accuracies are still reported for
every grid point.

## A21 — Spearman and pair-agreement are read as medians over carrier draws

**Silence.** Protocol 10.2 items 1 and 2 give thresholds without saying whether the
statistic is computed per carrier draw or pooled over all of them.

**Adopted.** Computed within each carrier draw and reported as the median across
draws, consistent with section 9.1's "the independent unit is the carrier draw" and
9.4's prohibition on treating repeated measurements as independent samples. The
per-draw values are all reported. Pooling the 8,646 direction pairs would also
overstate the evidence: the directions share one noise pool, which an independent
auditor measured as a `9.5x` variance inflation, i.e. `n_eff = 850`, not 8,070.

## A22 — Two additions to the direction set, both declared

- **`v_write` for a normal cell** is taken from the paired **non-normal** writer's
  `M_2048`, the manuscript's own section 5.2 matched-intervention convention, because
  a normal carrier has `M_2048 = I` exactly and `eigmax` of that matrix is LAPACK
  round-off (measured residual `1.75e-14`). The preregistered control is still
  "the write-block oracle"; only its construction for the degenerate cell is pinned.
- **`v_worst_positive`**, the eigenvector of the smallest **nonzero** eigenvalue of
  `M_old`, is **added** to the reported direction set. It does not replace `v_bottom`,
  which decision rule 10.2 item 3 names and which is kept unchanged. It is reported
  because `v_bottom` lies in a 16-dimensional kernel and carries exactly zero
  information, so it is not the "worst direction" the rule was reaching for.

## A23 — Validity battery runs per confirmatory cell, and an invalid cell is dropped

**Defect in this lane's first implementation, found in review.** The battery ran only
in the smoke stage and the confirmatory receipt hard-coded `"invalid_runs": []`, so
the 160 models would have shipped with no validity evidence of their own.

**Adopted.** The full battery runs inside every confirmatory cell, its verdict
travels with that cell's data, and a cell whose battery is not `VALID` contributes no
scientific row and is written to `INVALID_RUN_LEDGER.json`. `nc_verdict` refuses to
compute a verdict while the ledger is non-empty. Cost measured at about 1 s per cell.

## A24 — Kill rule 10.4 item 4 is implemented, not assumed

**Defect in this lane's first implementation, found in review.** The
`READOUT_MISMATCH` classification was documented but never computed, and
`SWAP_ROWS.csv` was written and never read.

**Adopted.** `nc_verdict.readout_mismatch` reads the swap table and classifies each
writer type as `READOUT_MISMATCH`, `NO_MISMATCH`, or
`COVARIANCE_AWARE_ALSO_FAILS_NOT_A_READOUT_QUESTION`. This is the rule that stops a
predicted, understood decoder-staleness effect from being written up as lost memory.

## A25 — The approximate-isolation gate reads the EMPIRICAL retained fraction

**Defect in this lane's first implementation, found in review.** The check read only
the analytic margin, and the aligned family is `Sigma + alpha*Sigma`, whose margin is
zero by construction (measured `+/-1.1e-16`). Half of the checks could not fail.

**Adopted.** The analytic margin is still recorded and must not be negative, but the
gate is on the **empirical** retained fraction estimated from the states the sampler
actually produced. A declared disturbance that the sampler failed to realise shows up
there and nowhere else. The aligned family is explicitly flagged as the equality case.

## A26 — The analysis horizon is read from the run, never typed in

**Defect in this lane's first implementation, found in review.** `nc_verdict` took
`--train-horizon` from the command line and never checked it. A reviewer ran it at
128, 256 and 4096 against the same CSVs and got three complete, silently different
verdicts.

**Adopted.** `nc_verdict` reads the horizon and amplitude from
`results/confirmatory/RUN_RECEIPT.json`, refuses to run while the invalid-run ledger
is non-empty, and raises if the number of carrier draws differs from the run's own
count, so that the "at least 14 of 16" gates cannot be evaluated on a different
denominator. Symmetrically, `stage_confirmatory` reads its amplitude and training
horizon from the pilot receipt and refuses to proceed on a `BLOCKED` pilot without an
explicitly recorded author decision.

## A27 — Exploratory arms use their own carrier block

**Ordering hazard, found in review.** `run_exploratory.py` defaulted to confirmatory
draws 0 to 7, so running an exploratory arm would have put confirmatory carriers in
front of the experimenter before the confirmatory stage ran.

**Adopted.** The exploratory arms draw from `EXPLORATORY_DRAWS = 800..807`, disjoint
from both the confirmatory block (`0..15`) and the pilot block (`900, 901`), and
`run_exploratory` raises if the sets ever intersect.

## A28 — Tolerance on the empirical retained fraction, fixed from pilot measurement

**Declared before the confirmatory run, measured on the excluded pilot draws only.**
Amendment A25 gates the empirical retained fraction, so that gate needs a tolerance
that noise cannot trip.

Measured across all 48 pilot cells (2 pilot draws x 2 writer types x 2 horizons x
3 alphas x 2 disturbance families) at the preregistered 50,000 episodes, the
deviation of the empirical retained fraction from its analytic value was:

```
mean  +0.000873      sd 0.001488
min   -0.002957      max +0.004661      max |deviation| 0.004661
```

The deviation is slightly positive on average, consistent with the known upward bias
of a plug-in Fisher estimator in finite samples.

**Adopted.** The tolerance stays at **0.02**, which is about 13 standard deviations
and about four times the largest deviation observed. It therefore cannot manufacture
a failure from Monte Carlo noise at the preregistered sample size, and a violation
beyond it indicates a sampler that failed to realise its declared disturbance, which
is exactly what the gate is for. The tolerance is fixed here and is not revisited
after the confirmatory data are inspected.

## A29 — Where kill rule 10.4 item 4 is evaluated, and its threshold

**Defect in this lane's second implementation, found in delta review.** The
`READOUT_MISMATCH` classifier read only the learned directions at the training
condition. That is the one combination in the entire artifact where a fixed decoder
cannot be stale: each seed's readout was fitted for exactly that direction at exactly
that horizon. On a real run the classifier therefore reported `NO_MISMATCH` while
`v_old`, carrying the same Fisher information as the learned direction (`0.1751`
against `0.1749`), scored **`0.3030` on the fixed decoder against `0.7349`
recalibrated**, with 84 of 160 rows below chance. The artifact asserted the opposite
of the measurement, in the field a write-up would quote.

**Adopted.** The rule is evaluated on the rows where it has content:

- `SWAP_ROWS`, the **non-learned** directions at the training condition (stale
  direction, current horizon), and
- `RETENTION_ROWS` **off** the training horizon (stale horizon, current direction).

Aggregated per carrier draw as the median of the paired per-row gap, so the
independent unit is the carrier draw. The classifier **raises** on an empty subset
rather than defaulting to `NO_MISMATCH`.

**Threshold, declared here because the protocol gives none:** a carrier-level median
paired gap above **0.02** counts as a fixed-decoder failure. It is the same numeric
scale as the calibration gate but a different quantity, and it is now a named
constant rather than a bare literal.

## A30 — The empirical-isolation gate is two-sided

**Residual found in delta review.** The gate fired only when the empirical retained
fraction fell *below* the floor. A sampler that applied no disturbance at all produces
a ratio of `1.0`, which is above every floor: forcing all 1,920 rows of one writer
type to `1.0` produced zero violations. That is exactly the failure the gate was added
to catch. A missing column was likewise a silent pass.

**Adopted.** The gate is two-sided. A row is a violation when the analytic retained
fraction falls below its floor, **or** the empirical fraction falls below the floor by
more than the tolerance, **or** the empirical fraction departs from its analytic twin
by more than the tolerance in either direction, **or** the column is missing or
non-finite. The tolerance stays at `0.02` and is a named constant.

**Correction to A28's stated headroom.** A28 justified `0.02` as "about 13 standard
deviations and about four times the largest deviation observed", measured on 48 pilot
cells. The confirmatory run produces about 3,840 approximate-isolation rows, eighty
times more, and the deviation scales with alpha. Rescaled to 50,000 episodes the worst
cell (aligned, alpha = 1) has a standard deviation near `0.0030`, so the real headroom
is about **6.7 standard deviations**, not 13. Still safe, and the tolerance does not
move; the claim about its size is corrected here rather than repeated.

## A31 — An unmeasured endpoint is INVALID, never a quiet non-kill

**Defect found in delta review.** `behavioral_efficiency` was emitted as `NaN` when the
oracle accuracy failed to exceed the random-direction median. Downstream,
`nan < 0.60` is False, so kill rule 10.4 item 1 could not fire on a lost behavioural
instrument; and the paired permutation test returned the smallest attainable p-value
on NaN input, because every comparison against NaN is False, so an instrument failure
was reported as the strongest evidence the design can produce
(`p = 9.999e-05`, Holm-adjusted `0.0004`).

**Adopted, three places.**

1. `nc_experiment.run_cell` **raises** when the behavioural denominator is not
   positive. An oracle that cannot beat a random direction is a broken instrument, not
   a result.
2. `nc_stats.paired_permutation_test` **raises** on any non-finite input, naming how
   many values were bad.
3. `nc_verdict` computes `instrument_failure` and reports
   `overall = INVALID_UNMEASURED_ENDPOINT`, which is distinct from `KILL` and from
   `PARTIAL`. Per review scope the agent does not declare a kill; this label says only
   that the measurement was not obtained.

## A32 — All three reserved draw blocks are checked, everywhere

**Residual found in delta review.** Validity control 12 and `stage_confirmatory`
checked only confirmatory-against-pilot. Setting
`CONFIRMATORY_DRAWS = (0, 1, 800, 801)` ran to completion with `run_valid: True`,
control 12 passing, and exploratory carriers written into the confirmatory receipt.

**Adopted.** `nc_geometry.assert_draw_blocks_disjoint` checks all three pairwise
intersections and is called at the top of every stage; validity control 12 is widened
to the same check with a negative control that collides the blocks; and every receipt
records the block membership and the collision list.

## A33 — Kill rule 10.4 item 4, corrected a second time

**Amendment A29 was also wrong, and this records why rather than replacing it.**
A29 moved the classifier off the learned direction at the training horizon and onto
the direction swaps. Measured on a rehearsal artifact, that reading does not measure
staleness either:

```
class              n     fixed med  recal med  |fixed-0.5| med
v_learn          160        0.7305     0.7345           0.2305
v_old            160        0.3030     0.7349           0.2301
v_random       20480        0.4994     0.6123           0.0308
v_bottom         160        0.4996     0.5005           0.0024
```

The fixed readout retains **the same** information about `v_old` as about the learned
direction (`0.2301` against `0.2305`); of 160 `v_old` seed-cells, **83 have
fixed = 1 - recalibrated and 74 have fixed = recalibrated**, which is the eigenvector
sign convention, not decay. And a linear readout evaluated on a direction nearly
orthogonal to its own must score chance whatever its state of repair, so the "gap" on
random directions is `Phi(a sqrt J) - 0.5`, the direction's information content. A
classifier built on that reports a mismatch on every healthy system and `NO_MISMATCH`
only when nothing is stored: the field would read affirmatively wrong.

**Adopted.** The rule is measured where it has physical content: the **retention**
experiment of protocol 5.4. The decoder is fitted at `H_train`; the store then rotates
by `U^(H - T_w)`. Under exact isolation the Fisher information is invariant, so the
covariance-aware readout is flat across horizons while a fixed decoder drifts out of
the coordinates it was fitted in. That is a readout failure with no memory loss behind
it, which is exactly what the rule names.

Classification therefore reads: **the learned direction, its own readout, isolated, at
horizons away from `H_train`**, aggregated as the median paired gap per carrier draw,
threshold `0.02`. The swap table is still summarised, including a sign-corrected
`max(acc, 1-acc)` column, as a descriptive diagnostic that is explicitly not used for
the classification.

## A34 — Three containment fixes found in final review

- **Writer types are pinned, not only draw counts.** Dropping every non-normal row
  previously produced a complete verdict with `nc1_alignment_pass: true` on one arm
  while the receipt declared both. Every family now asserts both writer types are
  present before any statistic is computed.
- **The paired-across-horizons diagnostic is confined to the retention rows.** It
  reuses one noise realisation at every horizon, so leaving it in `DIRECTION_ROWS`
  put a five-fold duplication of a single realisation into the gated calibration
  population. The measured shift was `6e-5` to `9e-5` against a `0.02` gate, so no
  gate flipped, but one third of the gated population was duplicated data. It is now
  excluded from `DIRECTION_ROWS` entirely.
- **A raised cell is ledgered, not fatal.** Amendment A31's new raise sat inside
  `ProcessPoolExecutor.map`, so a single cell that protocol section 11 says should be
  invalidated would instead have destroyed all 32 cells and written nothing. The
  exception is now captured per cell and written to `INVALID_RUN_LEDGER.json`.

## A35 — `nc_scan.py`: a parallel scan for the recurrence, and a batched trainer

**Added on author instruction** ("when you use the 3090 you should parallelise, with
Blelloch"). The closure recursion `lower[j+1] = lower[j] W + U^j K` is a first-order
linear recurrence, so each step is an affine map and composition
`(A1,C1).(A2,C2) = (A1 A2, C1 A2 + C2)` is associative with identity `(I, 0)`. A
Blelloch exclusive scan returns every prefix in O(n) work and O(log n) depth.

The frozen sequential `closure_transfers` remains the reference and the primary path.
`nc_scan.parity_check` asserts agreement; measured max relative error `9.7e-14` on
transfers and `6.2e-15` on the covariance at `n = 4096`.

One defect is recorded because it is the kind that hides: the first implementation
composed the down-sweep in the wrong order (left subtree before the parent prefix).
With a commutative operator that is invisible; with matrix composition it produced
relative errors of order 1. The parity check caught it immediately, which is why the
canonical implementation is kept as the reference rather than replaced.

## A36 — Residuals accepted before the run, recorded rather than fixed

After five review rounds the package was cleared to run. Three residuals were judged
non-blocking by the reviewer and are recorded here so they are not discovered later as
if they had been hidden.

1. **`results/confirmatory/cells/` is not cleared before a run.** After a wholesale
   failure, stale per-cell JSONs would survive beside a new receipt. `nc_verdict` never
   reads that directory and refuses to run at all while the invalid-run ledger is
   non-empty. Moot for the confirmatory run itself, which executes on a freshly
   extracted lane with no `results/` present.
2. **The non-finite guard in `readout_mismatch` omits `median(recal_at_train)`.** It is
   unreachable through the recalibration path, and the failure mode it would produce is
   the loud `FIXED_DECODER_NEVER_WORKED`, not the silent `NO_MISMATCH` that made the
   original finding.
3. **`peak_rss_bytes()` uses `RUSAGE_SELF`**, so the receipt's memory figure excludes the
   pool workers. Independently measured: 372 MB per cell at the frozen configuration,
   about 12 GB across 32 workers against 251 GB available. Bookkeeping, not capacity.

`run_all.sh` and the lane's copy of `run_with_review.sh` are now listed in the
freeze. Neither is globbed by `code_manifest()`, and the lane's copy of the gate is
byte-identical to the portfolio original.


## A37 — NC-3C residuals accepted at CLEAR_TO_RUN

Recorded rather than fixed, on the reviewer's own judgement that none is worth holding the
run and that each is inert today.

1. **Six NC-TM-V1 seed draws remain inside the battery.** `G.direction_controls`, called
   from `run_battery` and from `control_isolated_invariance`, draws its random directions
   from the completed lane's namespace even when the battery is pointed at an NC-3C
   carrier. Both call sites use only `controls["v_old"]` and discard the randoms, so no
   reported number depends on them. It is a latent trap the day a battery control uses
   `v_random`, and it is the residue of the defect the carrier-builder parameter fixed.
2. **The new freeze needs two working directories to verify**: 18 paths resolve against
   `code/`, 2 against the lane root, so no single `shasum -c` invocation returns clean.
3. **Control 31 still enumerates seven streams by hand** while the freshness audit now
   enumerates all 35 from the config.
4. **The receipt is self-reported.** The budget pin stops accidental confusion between a
   rehearsal and the confirmatory run; it does not stop deliberate forgery, and is not
   meant to.
5. **The contrast block reads `OBJECTIVES_NOT_DISCRIMINATING` in the 3-to-8 negligible
   band** where the overall verdict is `PARTIAL` and kill rule 4 did not fire. The label is
   reused for "not evaluated" and for "kill rule 4 fired"; `kill_rule_4_triggered`
   distinguishes them in the same object.

## A38 — NC-3C contributes one new confirmatory finding, not four

Accepted before the run, from the independent reviewer's quantitative analysis and not
revised afterwards. Of the four gate families, one is a measurable event: whether
behavioural training reaches the top eigenvector of the task operator at `t_0 = 12`, as it
demonstrably does at `t_0 = 0`. Diagonal dominance reduces to `G*_j > 0`, certified per
cell before training. Contrast recovery is the same identity divided through. Behavioural
calibration is a sampler-versus-formula check whose observed MAE equals its binomial noise
floor to three digits.

No gate was added, removed or re-scoped after learning this. Adding gates once you know
which ones cannot fail manufactures falsifiability rather than measuring it. The result
report and the manuscript's evidence-taxonomy table will state the one-finding scope.
