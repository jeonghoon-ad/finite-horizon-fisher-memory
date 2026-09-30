# Change log — v1.2 (2026-09-17) to v2.0 (2026-09-18)

## Added

- **Section 7, Task-Trained Realization and Objective-Specific Validation.** The 160-run
  training validation and the separate 320-run prospectively specified confirmation of
  objective-specific geometry, with the trainable/frozen partition stated explicitly, the
  Fisher-to-behaviour calibration, the allocation/retention intervention, the decoder
  transport result, and an evidence taxonomy.
- One paragraph to the Abstract.
- **Section 9.1 to 9.3, Limitations:** scope of the training studies; what endpoint
  sampling does and does not represent; the provenance of the confirmation's carriers.
- **Section 11.1, Reproducibility:** what reproduction means when `numpy.linalg.qr` is not
  bit-reproducible across LAPACK, and the wrapper-versus-scientific-code distinction.

## Renumbered

v1.2 sections 7 to 10 become v2.0 sections 8 to 11. No content in them changed.

## Unchanged

The entire analytic chain of v1.2: sections 1 to 6, Appendices A to D, the references, and
every previously published figure and number.

## Corrected, relative to the internal reports that preceded this build

None of these appeared in v1.2. They are corrections to interim reports, listed so the
record is complete.

| correction | was | is |
|---|---|---|
| open-arm decline along the horizon | "approximately 580-fold" | **118.3** (non-normal), **119.3** (normal), medians of paired per-carrier ratios |
| the 580 figure | attached to the along-horizon decline | it is the **isolated-versus-open ratio at a fixed horizon**: 520.4 and 486.9, paired medians |
| calibration range | "twelve orders of magnitude in J" | **4.23 orders**, `1.881e-05` to `0.3171`, over non-null directions |
| isolated `J` summary | 0.2205 / 0.1466, labelled a median | those were single arbitrary rows; the pooled medians are **0.2553 / 0.1406** |
| fixed decoder at the training horizon | 0.7728 recalibrated | **0.7760**; and away, 0.5067 becomes **0.5029**. The earlier pair came from a rehearsal |
| CPU/CUDA agreement | medians 0.999376 / 0.999328 quoted as confirmatory | those are a throughput benchmark on a pilot draw; the endpoint parity figure is **6.962e-10** on four pilot cells |
| decoder transport | 3.4e-14, from a reviewer's own computation | **7.994e-15** worst logit difference and **0.0** accuracy restoration error, now a released artifact over 32 rows |
| scan parity | 9.7e-14, one draw at one horizon | **4.570e-13**, worst over 32 cells, now a released artifact |
| diagonal dominance | "reduces to `G* > 0`, certified before training" | `G* > 0` permits separation; it does not certify what a finite optimizer returns. The exact accounting identity is given in 7.4 and verified to residual 0.0 |
| the training objective | called "a convex surrogate for the same Rayleigh quotient" | **withdrawn.** At a fixed write mask the logistic loss is convex in the readout; the joint problem over a unit-norm mask and a readout is not convex, and no convexity claim is made |
| NC-3C oracle separations | 0.41 / 0.35, minimum 0.117 | those are **exploratory** draws. The confirmatory medians are **0.2393 / 0.2090** (non-normal) and **0.5095 / 0.4455** (normal), minimum **0.0603** |

## Lineage

This build continues the 2026-09-17 v1.2 line, not the packaged release labelled v1.4,
which carries scientifically earlier content despite the higher version number. The lineage record is an internal working record and is not included. All predecessors are preserved unchanged.

## 2026-09-21 — attribution and positioning revision

Triggered by a prior-art disposition over four novelty audits. **No experiment re-run, no number
changed, no study pooled.** Title and subtitle changed to name the finite-horizon object; the
abstract and the §1 contribution paragraph now state the limit as an **identification** of a
classical object rather than a discovery; a *Classical ingredients* subsection credits the
Sz.-Nagy-tradition similarity result and Gehér's matrix theorems and gives the identification
\(M_\infty=\mathcal A_C(W^\top)^{-1}\) with its bound; Ganguli is credited at the first
spatial-operator definition and Kang at the trained mask; the horizon-two identity is promoted to
Proposition 3.2b; the temporal Fisher blocks are defined before the spatial aggregation, with a
warning against the observability-Gramian label; the sentence claiming the invariance statement as
this paper's contribution is replaced. Three references added. Full account in
`PRIOR_ART_DISPOSITION_20260921_EN.md`.

