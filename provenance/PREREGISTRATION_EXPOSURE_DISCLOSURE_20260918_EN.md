# Preregistration exposure — the NC-3C carriers were rehearsed before they were confirmed

**Raised by:** handoff section 8, *"Confirm from existing logs which rehearsal carriers
overlap with the final matrix; do not equate collision-free seeds with guaranteed absence
of outcome-informed selection. If overlaps occurred, disclose them and seek an editorial
disposition rather than silently claiming complete novelty of the data."*

**Answer: they overlap completely, and this is a departure from the handoff's own
section 6.2.** Disclosed here before the manuscript is written, with the evidence, the
exposure it does and does not create, and the remedies.

---

## 1. What happened

The NC-3C confirmatory matrix ran on carrier draws `0..15` of the `NC-3C-V1` namespace.
**Two reduced-budget rehearsals of the full chain ran on those same sixteen draws before
it**, and their summary statistics were displayed and read.

| | rehearsal (retained) | confirmatory |
|---|---|---|
| receipt stage | `nc3c_rehearsal` | `nc3c_confirmatory` |
| `budget_is_frozen` | `false` | `true` |
| training steps | 1,200 | 12,000 |
| test episodes | 12,000 | 50,000 |
| carrier draws | `0..15` | `0..15` |

The carrier SHA-256 digests differ between the two, but that is not independence: the
rehearsals ran on macOS/Accelerate and the confirmatory on Linux/OpenBLAS, and
`np.linalg.qr` is not bit-reproducible across LAPACK builds. Same namespace, same draw
indices, same seeds, therefore **the same carriers up to float64 round-off**.

A first rehearsal at 1,500 steps and 15,000 episodes preceded the retained one; its
artifacts were overwritten when the second ran. Its summary statistics were also read.

## 2. Why this is a departure, stated plainly

Handoff section 6.2 required fresh carrier draws "that did not appear in: NC-1/NC-2;
exploratory X1; pilot runs; **implementation smoke tests**." A reduced-budget rehearsal of
the full chain on the confirmatory carriers is an implementation smoke test on those
carriers. The freshness audit that was built and passed answers a narrower question than
the one section 6.2 asked: it proves disjointness from NC-1/NC-2, the pilot and X1, and it
says nothing about rehearsals inside the NC-3C namespace itself.

This is the coordinator's error. The audit was written to check the boundary it was
easiest to check.

## 3. What the record shows about outcome-informed selection

**No preregistered threshold moved.** The protocol and preregistration were frozen at
14:13 and 14:14; the retained rehearsal receipt is 14:36. Every gate value in the applied
verdict is identical to the frozen preregistration:

| gate | preregistered | applied |
|---|---:|---:|
| own-objective median | 0.90 | 0.90 |
| own-objective bootstrap lower bound | 0.80 | 0.80 |
| own-objective carriers at or above 0.80 | 14 | 14 |
| diagonal-dominance carrier count | 14 | 14 |
| contrast-recovery median | 0.75 | 0.75 |
| contrast-recovery bootstrap lower bound | 0.50 | 0.50 |
| negligibility threshold | 0.02 | 0.02 |
| calibration MAE | 0.02 | 0.02 |
| calibration slope interval | [0.95, 1.05] | [0.95, 1.05] |
| ordered-pair agreement | 0.90 | 0.90 |
| kill, own-objective median below | 0.75 | 0.75 |

**Changed after the rehearsals: 0 of 11.**

**Two gates were added after the first rehearsal**, both from an independent reviewer's
BLOCKER-2 and both recorded in a review file written before the change:
`minimum_usable_carriers = 14` and `not_discriminating_negligible_carriers = 9`. Both are
strictly tightening: each can only cause a failure, never a pass. Both were inert in the
confirmatory run, which had zero negligible carriers.

The two objectives (`t_0 = 0` and `t_0 = 12`) were fixed at 14:13, before any NC-3C
rehearsal. The protocol's kill rule forbidding a change of target times after seeing
outcomes was present from the start and was never exercised.

## 4. What this does and does not cost

**It does not support** a claim that a threshold, an objective, a sample count or a
decision rule was chosen to fit an observed result. The record excludes that.

**It does cost** the strongest available form of the claim. The confirmatory matrix was
not run on carriers whose behaviour was unseen: it was run on carriers whose
reduced-budget behaviour had been displayed twice. What the record cannot exclude is the
unfalsifiable counterfactual — whether the decision to proceed to the full budget would
have been the same had the rehearsals looked bad. That is a rehearsal-informed *decision
to proceed*, not a rehearsal-informed *change of rules*, and the manuscript must say so
rather than describe the carriers as unseen.

## 5. Remedies, for the owner's editorial disposition

1. **Disclose and proceed.** The methods section states that the confirmatory carriers had
   been exercised at reduced budget before the frozen run, that no threshold changed, and
   that the two added gates were tightening and inert. Costs nothing, keeps a qualified
   claim. This document is that disclosure.
2. **Re-run the confirmatory matrix on a carrier block that has never been rehearsed**, for
   example `NC-3C-V1` draws `16..31`, with the code and thresholds unchanged and with the
   rehearsals confined to the existing `0..15`. The frozen matrix took 47.7 s on a rented
   box; the whole rental was 0.22 h and about `$0.08`. This removes the exposure instead of
   qualifying it, and it leaves the existing run in the archive as a pre-specified
   replication on the same rules.
   **This requires new paid compute, which handoff section 6 does not authorize and
   section 10 restricts. It is the owner's call.**
3. Do nothing and claim complete novelty. **Not available.** The handoff forbids it and the
   record contradicts it.

**Recommendation: option 2, then option 1 for whichever run is reported second.** The
exposure is real, the cost of removing it is about eight cents and one minute of compute,
and a submission candidate should not carry a qualification that cheap to retire. If the
owner prefers not to authorize further compute, option 1 is honest and sufficient, and the
manuscript wording is drafted for it either way.
