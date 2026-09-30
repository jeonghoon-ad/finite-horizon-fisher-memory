# NC-3C — Task-Specific Geometry, Fresh Confirmation

**Version:** v1
**Date:** 2026-09-18
**Status:** PREREGISTRATION — frozen before any training outcome is inspected
**Parent authority:** `AGENT_HANDOFF_NC_MANUSCRIPT_AND_TASK_SPECIFIC_CONFIRMATION_20260918_EN.md`
**Supersedes nothing.** Exploratory arm X1 is retained as exploratory evidence and is
**not** relabelled.

---

## 1. Why this experiment exists

The completed NC-1/NC-2 confirmation passed all 26 of its gates, but an adversarial
audit established that four of its ten decision-rule items are algebraic identities of
the model and cannot fail. The one measurement in that lane whose outcome was not fixed
in advance, exploratory arm X1, was introduced **after** the confirmatory protocol was
written, so it is exploratory by construction and cannot carry a preregistered claim.

This protocol converts that question into a fresh, preregistered, confirmatory
experiment on data that has never been seen.

## 2. Primary question

> Does changing the target input time change the task-specific end-to-end Fisher
> operator, and do independently trained write directions follow the corresponding
> geometry rather than converging to one universal memory direction?

## 3. System

Identical to NC-1/NC-2 and unchanged: `N = 32`, `d = 16`, `T_w = 24`,
`x_w(0) = x_s(0) = 0`, process noise `z_t ~ N(0, I_N)` for `t = 0 .. T_w-1`, paired
writers `W = Q` and `W = S Q S^-1` with `kappa(S) = 10` sharing `Q`, `K` and `U`,
coupling `K_t = K` for `t < T_w` and `0` thereafter (isolated), store map `U` orthogonal
and frozen.

Frozen training configuration, carried over unchanged from the completed run:
Adam, learning rate `3e-3`, weight decay `0`, batch `512`, **12,000 steps**, global
gradient-norm clip `1.0`, amplitude **`a = 1.5`**, training horizon **`H_train = 128`**,
`u ~ N(0, I)` normalised, `w = 0`, `b = 0`, final checkpoint primary.

## 4. Freshness

All carrier draws, optimizer seeds, labels and noise seeds are drawn from a **new seed
namespace `NC-3C-V1`**, which has never been used. Nothing in NC-1/NC-2, exploratory X1,
the pilot, or any smoke test used that namespace, so no seed can collide with a seen one
by construction rather than by inspection. The complete seed list is generated and frozen
in the run manifest before any training outcome is inspected, and a mechanical check
asserts the NC-3C seed set is disjoint from every seed the completed lane recorded.

Carrier draws: `0 .. 15` **within the `NC-3C-V1` namespace**.

## 5. Objectives and operators

Two objectives, trained independently:

```
O_0  : target input time t_0 = 0
O_12 : target input time t_0 = 12 = T_w / 2
```

For objective `j`, the task-specific end-to-end operator is

```
M_j = L_j^T C_ss^-1 L_j ,     L_j = transfers[T_w - 1 - j]
```

which is the same construction the completed lane used for `M_old = M_0`, evaluated at a
different injection time. `M_0` and `M_12` share `C_ss`, the carrier, the coupling and
the store map; they differ only in which input the task asks about.

**No Fisher object enters the optimizer.** The trainer receives the sensitivity map, the
noise factor, the amplitude and the labels, exactly as in the completed run.

## 6. Run matrix

```
16 fresh paired carrier draws
x 5 optimizer seeds
x 2 writer types
x 2 objectives
= 320 trained models
```

No post-result rescue grid. No new amplitude, optimizer, task or carrier grid may be
introduced after confirmatory outcomes are inspected.

## 7. Primary quantities

For a direction trained on objective `i`, evaluated against objective `j`:

```
R_{j<-i} = v_{learn,i}^T M_j v_{learn,i} / lambda_max(M_j)
```

Own-objective: `R_{0<-0}`, `R_{12<-12}`. Cross-objective: `R_{0<-12}`, `R_{12<-0}`.

**Oracle-normalised contrast recovery.** Let `v_j*` be objective `j`'s own oracle and
`v_jbar*` the other objective's oracle. The separation actually available in the geometry
of that carrier is

```
G*_j = 1 - (v_jbar*)^T M_j (v_jbar*) / lambda_max(M_j)
```

the learned contrast is

```
G_j^learn = R_{j<-j} - R_{j<-jbar}
```

and, when `G*_j > 0`, contrast recovery is

```
C_j = G_j^learn / G*_j
```

This normalisation is used so that the practical margin is set by the geometry of each
carrier and **not** by the effect size already seen in exploratory X1.

## 8. Confirmatory gates

Every gate is evaluated **separately for each writer type**, and the overall reading
requires both. The independent unit is the carrier draw; optimizer seeds are repeated
measurements reduced by the within-draw median before any test.

### 8.1 Own-objective efficiency, both objectives

- median `R_{j<-j} >= 0.90`
- 95% carrier-bootstrap one-sided lower bound `>= 0.80`
- at least **14 of 16** carrier draws with `R_{j<-j} >= 0.80`

### 8.2 Diagonal dominance

At least **14 of 16** carrier draws satisfy **both**

```
R_{0<-0}  >  R_{0<-12}        and        R_{12<-12}  >  R_{12<-0}
```

and the paired 95% bootstrap interval for each median diagonal difference excludes zero.

### 8.3 Contrast recovery, both objectives

- median `C_j >= 0.75`
- 95% bootstrap one-sided lower bound `>= 0.50`

A carrier whose `G*_j < 0.02` is reported as `OBJECTIVES_GEOMETRICALLY_INDISTINGUISHABLE`
for the contrast ratio only, and is retained in the own-objective and calibration
analyses. **The 0.02 threshold is frozen here, before the run.**

### 8.4 Behavioural calibration, both objectives

- empirical versus `Phi(a sqrt J)` mean absolute error `<= 0.02`
- calibration slope within `[0.95, 1.05]`
- Fisher ordering and recalibrated behavioural ordering agree for at least **90%** of
  direction pairs separated by more than 5% in `J`, the gap taken relative to the larger
  of the pair

## 9. Kill rules

- If either objective has median own-objective `R < 0.75`, no task-specific learning
  claim is permitted.
- If diagonal dominance fails in either objective, no claim that training follows
  objective-specific geometry is permitted.
- If `J` does not predict behaviour, no Fisher-to-behaviour extension for NC-3C.
- If both task operators are nearly identical on most draws, the conclusion is that the
  chosen objectives do not constitute a discriminating confirmation. **Target times are
  not changed after training outcomes are seen.**

## 10. Required causal cross-evaluation

This is the claim-bearing test. For every trained direction, with the carrier, coupling
and store map held fixed:

1. evaluate on its **own** objective with a recalibrated readout;
2. evaluate on the **other** objective with a recalibrated readout;
3. compare the empirical results against `R_{j<-i}` and `Phi(a sqrt J)`.

Recalibration means a pooled-within-class linear discriminant fit on an independent
50,000-episode calibration sample, as in the completed run. Fixed-readout results are
recorded as a deployment diagnostic and are excluded from the calibration gate.

## 11. Validity

The 29-control battery of the completed lane runs per cell, each control clean and
against a corrupted input. A cell whose battery is not `VALID` contributes no scientific
row and is written to the invalid-run ledger. The verdict tool refuses to run while that
ledger is non-empty.

Two controls are added for this protocol:

- **30. Objective distinctness.** `M_0` and `M_12` must differ: the check records
  `|cos(v_0*, v_12*)|` and `G*_j` for every cell, and a negative control that sets
  `t_0 = 0` for both objectives must fail it.
- **31. Seed freshness.** The NC-3C seed set must be disjoint from every seed recorded
  in the completed lane's receipts.

## 12. Declared before the run

- **The `O_12` task is harder at the frozen amplitude.** Measured on the exploratory
  draws, `lambda_max(M_12)` is roughly half `lambda_max(M_0)`, so at `a = 1.5` the
  oracle accuracy is about 0.76 for `O_0` and 0.71 for `O_12` (non-normal), and about
  0.72 and 0.67 (normal). This is **not** a blocker and the amplitude is not changed:
  every gate in section 8 is a normalised quantity (`R`, `C`, a calibration residual),
  so a difficulty difference between objectives does not bias any of them. It is
  recorded here so it cannot be presented later as a finding.
- **The separation is not degenerate.** On the exploratory draws `G*_0` has median
  0.4113 (range 0.1165 to 0.7529) and `G*_12` median 0.3478 (range 0.1280 to 0.5828),
  with `0 of 16` cells below the 0.02 negligibility threshold. The contrast-recovery
  gate is therefore measurable rather than vacuous. Fresh draws may differ; this is a
  feasibility statement, not a prediction.
- **What a PASS will and will not mean.** A PASS licenses: *changing the target memory
  time changed the task-specific end-to-end Fisher operator, and independently trained
  write directions followed the corresponding geometry rather than converging to a
  universal memory direction.* It does not license any statement about nonlinear
  systems, language models, continual learning, catastrophic forgetting, architecture
  superiority, or product readiness.

## 13. Stop condition

Complete when: the seed list and thresholds are frozen; an independent review returns
`CLEAR_TO_RUN`; all 320 runs complete or are invalidated under the frozen rule; the
cross-evaluation completes; an independent verifier recomputes the primary endpoints
from raw artifacts; and the verdict is written from the frozen rule rather than from
selected examples.
