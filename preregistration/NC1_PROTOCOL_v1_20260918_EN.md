# Neural Computation Submission Experiment Protocol

## Task-Trained Alignment with Directional Fisher Memory

**Version:** v1  
**Date:** 2026-09-18  
**Status:** PREREGISTRATION DRAFT — NO RUN AUTHORITY  
**Target manuscript:** *Non-Normality Allocates Fisher Capacity; Isolation Preserves It*  
**Target journal:** *Neural Computation*  
**Purpose:** Add one claim-bearing trained-system experiment that connects the paper's analytic Fisher geometry to learned recurrent behavior without broadening the paper into a general architecture benchmark.

---

## 0. Executive decision

The journal-facing experiment should test one question:

> **Can end-to-end task training, without access to the Fisher operator, discover the write direction predicted by the end-to-end memory geometry, and does that geometry quantitatively predict delayed behavioral performance?**

The experiment does **not** attempt to prove general sequence-model superiority, continual learning, language-model value, or nonlinear universality.

The claim-bearing path is:

\[
\text{analytic Fisher direction}
\rightarrow
\text{task-trained write mask}
\rightarrow
\text{behavioral delayed recall}
\rightarrow
\text{causal direction swap}
\rightarrow
\text{open-versus-isolated retention intervention}.
\]

A secondary experiment may additionally train the write-to-store coupling. It is not required for the primary paper claim.

---

## 1. Scientific gap addressed

The current paper establishes:

1. a fixed directional trace budget;
2. normal-carrier isotropy;
3. a uniform \(1/n\) lag-wise Fisher law for bi-power-bounded carriers under continuing process noise;
4. a Cesàro–commutant large-horizon operator;
5. finite-horizon certification;
6. an end-to-end store operator;
7. exact and approximate post-write retention guarantees; and
8. covariance-aware decoding.

The remaining journal-level empirical gap is:

> The high-information direction is currently selected analytically. A trained recurrent system has not yet been shown to discover and use it from behavioral loss.

This protocol closes that gap with a delayed binary-estimation task whose Bayes accuracy is an exact function of the Fisher information.

---

## 2. Behavioral bridge theorem

### Proposition 2.1 — Fisher information determines optimal delayed binary accuracy

Let the final observed store state satisfy

\[
x\mid y
\sim
\mathcal N(y\,a\,p,\Sigma),
\qquad
y\in\{-1,+1\}
\]

with equal class priors, input amplitude \(a>0\), class sensitivity \(p\), and class-independent covariance \(\Sigma\succ0\). Define

\[
J=p^\top\Sigma^{-1}p.
\]

The Bayes-optimal linear discriminant is proportional to \(\Sigma^{-1}p\), and its accuracy is

\[
\boxed{
\operatorname{Acc}^{\star}
=
\Phi\!\left(a\sqrt{J}\right),
}
\]

where \(\Phi\) is the standard normal cumulative distribution function.

#### Proof

Under \(y=+1\),

\[
p^\top\Sigma^{-1}x
\sim
\mathcal N(aJ,J).
\]

The equal-prior decision threshold is zero, hence

\[
\Pr(\hat y=y)
=
\Pr\big[\mathcal N(aJ,J)>0\big]
=
\Phi(a\sqrt J).
\]

This proposition gives an exact behavioral prediction for every direction, carrier, storage condition, and delay in the primary experiment.

---

## 3. Recurrent memory system

### 3.1 State equations

Writer dimension:

\[
N=32.
\]

Store dimension:

\[
d=16.
\]

Write interval:

\[
T_w=24.
\]

Writer:

\[
x_{w,t+1}
=
W x_{w,t}
+
a\,y\,v\,\mathbf 1[t=t_0]
+
z_t,
\qquad
z_t\sim\mathcal N(0,I_N).
\]

Store:

\[
x_{s,t+1}
=
K_t x_{w,t}
+
U x_{s,t}.
\]

The store receives no direct process noise.

Primary target time:

\[
t_0=0.
\]

Storage conditions:

- **Isolated:** \(K_t=K\) for \(t<T_w\), and \(K_t=0\) for \(t\ge T_w\).
- **Open:** \(K_t=K\) for all \(t\).

The store map \(U\) is orthogonal and frozen.

### 3.2 Paired writer types

For each independent carrier draw:

- **Normal writer**
  \[
  W_N=Q,\qquad Q^\top Q=I.
  \]

- **Bi-power-bounded non-normal writer**
  \[
  W_{NN}=SQS^{-1},
  \qquad
  \kappa(S)=10.
  \]

The paired writers share the same \(Q\), coupling \(K\), and store map \(U\).

### 3.3 Directional operators

For each writer/storage instance compute:

1. **Write-block operator**
   \[
   M_{\mathrm{write}}=M_{2048}.
   \]

2. **Store-total operator**
   \[
   M_{\mathrm{store}}
   =
   \sum_{t=0}^{T_w-2}
   L_t^\top C_{ss}^{-1}L_t.
   \]

3. **Oldest-input operator**
   \[
   M_{\mathrm{old}}
   =
   L_0^\top C_{ss}^{-1}L_0.
   \]

The claim-bearing behavioral task targets \(t_0=0\), so its exact oracle is

\[
v_{\mathrm{old}}
=
\operatorname{eigmax}(M_{\mathrm{old}}).
\]

The paper's existing directions remain explicit comparators:

\[
v_{\mathrm{write}}
=
\operatorname{eigmax}(M_{\mathrm{write}}),
\]

\[
v_{\mathrm{store}}
=
\operatorname{eigmax}(M_{\mathrm{store}}).
\]

---

## 4. Experiment NC-1 — Task-trained directional alignment

### 4.1 Question

Can gradient training on delayed classification, with no Fisher or eigensystem supervision, discover the high-information direction \(v_{\mathrm{old}}\)?

### 4.2 Trainable parameters

The recurrent matrices \(W,K,U\) remain frozen.

Train only:

1. an unconstrained vector \(u\in\mathbb R^N\), converted to a unit write direction
   \[
   v=\frac{u}{\|u\|_2};
   \]

2. a linear logistic readout
   \[
   \ell(x_s)=w^\top x_s+b.
   \]

No Fisher matrix, eigenvector, covariance, or analytic decoder is exposed to the optimizer.

### 4.3 Behavioral task

For each episode:

1. sample \(y\in\{-1,+1\}\) uniformly;
2. inject \(a y v\) at \(t_0=0\);
3. draw independent writer process noise at every step;
4. evolve to query horizon \(H_{\mathrm{train}}\);
5. predict \(y\) from the store state with binary cross-entropy loss.

The primary training condition is the isolated store.

### 4.4 Pilot-only calibration

Use two carrier draws and two optimizer seeds that are permanently excluded from the confirmatory analysis.

The pilot may choose only:

- amplitude \(a\in\{1,1.5,2,2.5\}\);
- training horizon \(H_{\mathrm{train}}\in\{128,256\}\).

Selection criterion:

- oracle test accuracy between 0.70 and 0.85;
- random-direction test accuracy between 0.55 and 0.75;
- no numerical clipping or covariance singularity.

After selection, \(a\) and \(H_{\mathrm{train}}\) are frozen. No other protocol element may be changed from pilot outcomes.

### 4.5 Confirmatory sample

Independent carrier draws:

\[
D=16.
\]

Optimizer seeds per carrier:

\[
S=5.
\]

The carrier draw, not the optimizer seed, is the independent statistical unit.

### 4.6 Fixed training budget

Unless the pilot reveals a runtime defect rather than a scientific result:

- optimizer: Adam;
- learning rate: \(3\times10^{-3}\);
- weight decay: 0;
- batch size: 512;
- gradient steps: 12,000;
- gradient-norm clipping: 1.0;
- final checkpoint is primary;
- best-validation checkpoint is secondary only;
- no adaptive rescue grid after confirmatory results are inspected.

Initializations:

- \(u\sim\mathcal N(0,I)\), then normalized;
- \(w=0\), \(b=0\).

### 4.7 Primary endpoints

#### Endpoint E1 — Fisher efficiency of the learned direction

\[
R_J
=
\frac{v_{\mathrm{learn}}^\top
M_{\mathrm{old}}
v_{\mathrm{learn}}}
{\lambda_{\max}(M_{\mathrm{old}})}.
\]

This endpoint is invariant to eigenvector sign and remains valid under a nearly degenerate leading eigenspace.

#### Endpoint E2 — leading-subspace alignment

Define

\[
\mathcal E_{0.99}
=
\operatorname{span}
\left\{
e_i:
\lambda_i(M_{\mathrm{old}})
\ge0.99\lambda_{\max}(M_{\mathrm{old}})
\right\}.
\]

Then

\[
A_{\mathrm{sub}}
=
\left\|
P_{\mathcal E_{0.99}}v_{\mathrm{learn}}
\right\|_2^2.
\]

If the leading eigengap is below \(0.01\lambda_{\max}\), \(R_J\), not single-vector cosine, is the primary alignment measure.

#### Endpoint E3 — relative behavioral efficiency

With empirical learned accuracy \(A_L\), analytic-oracle accuracy \(A_O\), and median random-direction accuracy \(A_R\),

\[
R_{\mathrm{beh}}
=
\frac{A_L-A_R}
{A_O-A_R}.
\]

#### Endpoint E4 — Fisher-to-behavior calibration

For every evaluated direction and horizon,

\[
A_{\mathrm{pred}}
=
\Phi\!\left(
a\sqrt{J}
\right).
\]

Report mean absolute error and calibration slope between \(A_{\mathrm{pred}}\) and empirical accuracy using a separately calibrated linear readout.

### 4.8 Direction controls

For every carrier draw, evaluate without additional write-direction training:

1. \(v_{\mathrm{old}}\): lag-specific oracle;
2. \(v_{\mathrm{store}}\): store-total oracle;
3. \(v_{\mathrm{write}}\): write-block oracle;
4. \(v_{\mathrm{bottom}}\): bottom eigenvector of \(M_{\mathrm{old}}\);
5. 128 isotropic random directions;
6. \(v_{\mathrm{learn}}\): task-trained direction.

For each direction, use the same test noise and class labels within a carrier draw.

### 4.9 Causal direction-swap test

At the trained checkpoint, keep \(W,K,U\) fixed and replace only \(v\) by each control direction.

Evaluate two readout modes:

1. **Recalibrated readout:** refit only the linear readout for each direction.
2. **Fixed learned readout:** retain the trained readout to measure practical coordinate mismatch.

The recalibrated test is the claim-bearing causal test of directional information. The fixed-readout test is a secondary deployment diagnostic.

---

## 5. Experiment NC-2 — Retention intervention on a learned write direction

### 5.1 Question

Does the same learned direction exhibit the retention behavior predicted by the post-write channel?

### 5.2 Intervention

For each trained \(v_{\mathrm{learn}}\), evaluate the identical writer, direction, input, noise seeds, and write-window coupling under:

1. exact isolation;
2. open coupling;
3. exact isolation plus additive store disturbance satisfying
   \[
   0\preceq R_h
   \preceq
   \alpha A_h\Sigma_0A_h^\top,
   \quad
   \alpha\in\{0.05,0.25,1\};
   \]
4. optional noninvertible disposal control.

No retraining occurs between storage conditions.

### 5.3 Query horizons

\[
H\in\{64,256,1024,4096\}.
\]

### 5.4 Readout modes

1. covariance-aware recalibrated linear discriminant;
2. fixed decoder trained at \(H_{\mathrm{train}}\).

The first tests the information theory. The second measures practical decoder robustness.

### 5.5 Predictions

#### Exact isolation

\[
J_H=J_{T_w}
\]

under an invertible uncontaminated hold, hence

\[
A^\star_H
=
\Phi(a\sqrt{J_{T_w}})
\]

is constant across horizons.

#### Open coupling

Continuing write-path noise contaminates the store. The predicted empirical accuracy is

\[
A^\star_H
=
\Phi(a\sqrt{J_H}),
\]

with \(J_H\) obtained from full propagation.

#### Approximate isolation

\[
J_H
\ge
\frac{J_0}{1+\alpha},
\]

and therefore

\[
A^\star_H
\ge
\Phi\!\left(
a\sqrt{\frac{J_0}{1+\alpha}}
\right).
\]

### 5.6 Primary endpoints

1. prediction-versus-observation accuracy error across horizons;
2. retained Fisher ratio \(J_H/J_0\);
3. fixed-decoder degradation relative to covariance-aware readout;
4. paired isolated-versus-open accuracy margin.

---

## 6. Experiment NC-3 — Objective specificity

### 6.1 Question

Does training select the direction appropriate to the behavioral objective, rather than a generic "best memory" direction?

### 6.2 Comparisons

For the oldest-item task compare:

\[
v_{\mathrm{learn}}
\quad\text{to}\quad
v_{\mathrm{old}},
v_{\mathrm{store}},
v_{\mathrm{write}}.
\]

The expected ordering is determined by

\[
v^\top M_{\mathrm{old}}v,
\]

not by the store-total or write-block Rayleigh quotient.

### 6.3 Claim

A PASS supports:

> A trained recurrent input mask can discover the direction predicted by the task-specific end-to-end Fisher operator.

It does **not** support:

> One oracle direction is optimal for every lag, objective, or coupling geometry.

---

## 7. Optional secondary experiment — Learned admission coupling

This experiment is run only if NC-1 passes its alignment gate.

### 7.1 Trainable parameters

Train:

- unit write direction \(v\);
- a row-orthonormal coupling \(K\), parameterized by a differentiable QR or Stiefel retraction;
- the linear decoder.

Keep \(W\) and \(U\) fixed.

### 7.2 Arms

1. \(v\)-only training, fixed \(K\);
2. \(K\)-only training, fixed \(v_{\mathrm{old}}\);
3. joint \(v+K\) training;
4. frozen random \(v,K\).

### 7.3 Measurements

At each checkpoint compute, without feeding it to the optimizer,

\[
M_{\mathrm{old}}(K)
=
L_0(K)^\top
C_{ss}(K)^{-1}
L_0(K).
\]

Test whether:

- task accuracy tracks \(v^\top M_{\mathrm{old}}(K)v\);
- joint training improves the Rayleigh quotient;
- causal shuffling of \(K\) or \(v\) destroys the gain.

### 7.4 Secondary advancement rule

Joint training advances only if, across carrier draws:

- median information ratio over \(v\)-only is at least 1.10;
- empirical accuracy improves by at least 0.02;
- the directionality and calibration checks still pass.

A null result leaves NC-1 as the paper's learned-system result.

---

## 8. Non-decision-bearing task-solvability references

The following may be included only as context:

- a 32-unit GRU;
- a 32-state orthogonal RNN;
- a freely trained stable linear RNN.

They are not used to validate the Fisher theorem and do not enter the primary decision rule. The manuscript makes no general architecture-superiority claim.

---

## 9. Statistical analysis

### 9.1 Independent unit

The independent unit is the carrier draw.

Optimizer seeds are repeated measurements nested within carrier draw.

### 9.2 Aggregation

For each carrier draw:

1. compute the median endpoint over optimizer seeds;
2. compare paired normal/non-normal or open/isolated conditions within that draw.

### 9.3 Uncertainty

Use:

- 10,000-draw paired bootstrap over carrier instances;
- 95% confidence intervals;
- paired permutation tests as secondary;
- Holm correction over the two primary families: alignment and behavior.

### 9.4 Reported distributions

Report:

- all carrier-level points;
- median and interquartile range;
- paired differences;
- sign consistency;
- no seed-only \(p\)-values treating optimizer seeds as independent samples.

---

## 10. Frozen decision rules

### 10.1 NC-1 alignment PASS

All conditions must hold:

1. median
   \[
   R_J\ge0.90;
   \]
2. 95% bootstrap lower bound
   \[
   R_J\ge0.80;
   \]
3. at least 14 of 16 carrier draws have
   \[
   R_J\ge0.80;
   \]
4. median
   \[
   R_{\mathrm{beh}}\ge0.80;
   \]
5. empirical-versus-predicted accuracy mean absolute error
   \[
   \le0.02.
   \]

### 10.2 Causal geometry PASS

Across all evaluated control directions:

- Spearman correlation between \(J\) and recalibrated empirical accuracy is at least 0.95;
- at least 90% of direction pairs ordered by a Fisher gap exceeding 5% are ordered the same way behaviorally;
- \(v_{\mathrm{bottom}}\) performs below the random-direction median in at least 14 of 16 carrier draws.

### 10.3 Retention PASS

- isolated covariance-aware accuracy varies by at most 0.01 across query horizons;
- predicted-versus-empirical accuracy mean absolute error is at most 0.02;
- every approximate-isolation condition satisfies its declared lower bound within Monte Carlo confidence;
- open-versus-isolated ordering agrees with full propagated \(J_H\) in at least 15 of 16 draws.

### 10.4 Kill rules

- If \(R_J<0.75\) or \(R_{\mathrm{beh}}<0.60\), no learned-alignment claim is permitted.
- If behavioral accuracy is not calibrated by \(\Phi(a\sqrt J)\), no Fisher-to-behavior claim is permitted.
- If the paired normal system matches the non-normal system after each is allowed its own task-specific end-to-end oracle, no non-normal task-advantage claim is permitted.
- If only a fixed decoder fails while covariance-aware readout passes, classify the result as `READOUT_MISMATCH`, not memory loss.
- No new optimizer, amplitude, task, or carrier grid may be introduced after confirmatory outcomes are inspected.

---

## 11. Validity controls

1. Recurrence unroll equals block-matrix propagation.
2. Empirical class covariance matches analytic covariance.
3. Analytic and empirical \(J\) agree within tolerance.
4. The Bayes classifier reproduces \(\Phi(a\sqrt J)\).
5. Normal write-block control satisfies \(M_n=I\).
6. Trace identities pass.
7. Same noise and labels are used for paired directional interventions.
8. The optimizer receives no Fisher matrix, eigenvector, covariance, or oracle label.
9. Direction-sign equivalence is handled by squared alignment.
10. Degenerate leading eigenspaces use subspace alignment.
11. Open/isolated arms differ only in post-window coupling.
12. Pilot carrier draws never enter confirmatory analysis.

Any failed validity control invalidates the affected run before scientific interpretation.

---

## 12. Compute plan

This experiment is small relative to language-model training.

Primary confirmatory matrix:

\[
16\text{ carrier draws}
\times
5\text{ optimizer seeds}
\times
2\text{ writer types}
=
160\text{ trained models}.
\]

The storage interventions and direction swaps are evaluation-only and require no retraining.

Before freezing the runtime estimate:

1. run one carrier × one seed smoke;
2. record examples/second and peak memory;
3. extrapolate the full budget;
4. freeze the budget and prohibit rescue expansions.

Expected hardware class: one modern consumer GPU or CPU-vectorized execution. The actual budget must be computed from the smoke receipt rather than estimated in the paper.

---

## 13. Required artifacts

Each trained run must emit:

- frozen configuration JSON;
- carrier/coupling/store hashes;
- optimizer seed;
- final \(v,w,b\);
- complete loss curve;
- \(M_{\mathrm{write}},M_{\mathrm{store}},M_{\mathrm{old}}\) eigenvalues;
- \(R_J,A_{\mathrm{sub}},R_{\mathrm{beh}}\);
- predicted and empirical accuracy by horizon;
- fixed and recalibrated readout results;
- causal direction-swap table;
- validity-control results;
- runtime and peak memory;
- SHA-256 manifest.

Aggregate outputs:

- carrier-level CSV;
- paired statistical report;
- figures generated from frozen CSVs only;
- machine-readable verdict;
- invalid-run ledger.

---

## 14. Manuscript integration

The main paper should add one section:

# Learned alignment and behavioral validation

Recommended subsections:

1. delayed binary-estimation theorem;
2. task and training protocol;
3. learned-direction alignment;
4. causal direction swaps;
5. Fisher-predicted versus empirical accuracy;
6. learned-direction retention under open and isolated storage;
7. limitations.

Recommended figures:

- **Figure 5:** learned \(R_J\) and subspace alignment by carrier type;
- **Figure 6:** predicted versus empirical accuracy for all directions and horizons;
- **Figure 7:** causal direction-swap accuracy ordered by \(J\);
- **Figure 8:** open versus isolated accuracy curves for the same learned direction.

The abstract may then state:

> Without Fisher supervision, gradient training on delayed binary estimation recovered directions near the leading eigenspace of the task-specific end-to-end Fisher operator, and the resulting behavioral accuracy followed the analytic prediction \(\Phi(a\sqrt J)\). Closing the store coupling preserved this learned information across query horizons, whereas continued coupling produced the predicted decay.

This sentence is licensed only if all primary gates pass.

---

## 15. Permitted and forbidden conclusions

### Permitted after full PASS

> In the tested linear recurrent memories, task training without Fisher supervision learned write masks close to the leading eigenspace of the task-specific end-to-end Fisher operator. The operator quantitatively predicted delayed binary behavior, and a causal storage intervention separated directional allocation from post-write retention.

### Forbidden

- trained recurrent networks generally learn Fisher-optimal directions;
- non-normal memories universally outperform normal memories;
- the result establishes language-model or continual-learning value;
- the result extends automatically to nonlinear or non-Gaussian systems;
- Fisher concentration guarantees high absolute task accuracy;
- an isolated store solves catastrophic forgetting.

---

## 16. Stop condition

This protocol is complete when:

1. pilot-only parameters are frozen;
2. all carrier and optimizer seeds are fixed;
3. run authority names the frozen hashes;
4. all 160 primary runs complete or are invalidated under the frozen rule;
5. evaluation-only interventions complete;
6. an independent verifier recomputes the primary endpoints from raw artifacts;
7. the manuscript claim is written from the frozen verdict rather than from selected examples.
