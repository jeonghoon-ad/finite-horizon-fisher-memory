# Prior-art disposition and what changed in the manuscript

**Date:** 2026-09-21. **Input:** `FISHER_PRIOR_ART_DISPOSITION_AND_REVISION_SCOPE_20260921_EN.md`,
itself a disposition over four novelty audits.
**Status:** attribution and positioning revision. **No experiment was re-run, no carrier block
added, no number changed.**

## The finding, stated plainly

Two things the manuscript leaned on are established, and the manuscript now says so at the point
of use rather than only in Related Work.

1. **The long-horizon limit object is classical.** $M_\infty$ is the inverse of the classical
   Cesàro asymptotic limit of $W^\top$. Gehér's matrix results are the precise antecedent, closer
   than the mean-ergodic theorem the manuscript previously gestured at. "We discover a new
   Cesàro/commutant limit" would not have been defensible; it is not claimed.
2. **Fisher-optimal input masks, and their behavioural evaluation, are not new.** Kang, Shirasaka
   and Suzuki (2021) optimize a reservoir input mask by the leading eigenvector of a Fisher memory
   matrix and then evaluate it on memory tasks. What remains distinctive here is narrow: the mask
   is trained from the behavioural loss **with the Fisher objective withheld**, and compared
   against a task-specific finite-window and downstream operator.

**This is not a finding that anything in the paper is false.** Correctness, replication,
originality, journal importance and eligibility for intellectual-property protection are four separate questions, and only the third
moved.

## Verified here before adopting

The identification $M_\infty=\mathcal A_C(W^\top)^{-1}$ was checked numerically on this paper's own
carrier family before it was written into §3.3.0, rather than copied from the review. On draw 0 the
residual $\lVert M_n-G_n^{-1}\rVert_2$ falls 4.77e-3, 2.54e-3, 1.20e-3, 5.91e-4, 3.26e-4 across
$n=2500$ to $40000$, ratios 1.87, 2.12, 2.03, 1.81 per doubling, which is the claimed $O(1/n)$. The
normal carrier sits at 1e-13. The check ships as `code/check_cesaro_identification.py` with its own
review, so a reader can repeat it.

The review's warning about the audit reports' own errors was also checked: this manuscript writes
the similarity as $W=SQS^{-1}$ throughout, not the erroneous $SQS^{-\top}$ form, and it does not
make Fisher invariance conditional on decoder transport.

## Changes made, once

| # | change | where |
|---|---|---|
| 1 | title and subtitle now name the finite-horizon object rather than two established principles | front matter |
| 2 | abstract says the limit is **identified** with a classical object, and that the ingredients are classical | abstract |
| 3 | contribution paragraph restated against its sources, listing what is established and what is specialization | §1 |
| 4 | Ganguli credited at the first spatial-operator definition; Kang credited there and again at the trained mask | §2, §7.1 |
| 5 | new §3.3.0 records the classical ingredients, cites Sz.-Nagy-tradition similarity and Gehér, and gives the identification with its bound | §3.3.0 |
| 6 | horizon-two identity promoted from a census remark to **Proposition 3.2b** with proof, explicitly elementary and with no priority claim | §3.2 |
| 7 | temporal Fisher blocks $\mathcal I_{tu}=L_t^\top C_{ss}^{-1}L_u$ defined **before** the spatial aggregation, with a warning against calling $M_{\text{store}}$ an observability Gramian | §5.1 |
| 8 | the sentence claiming the invariance statement as the contribution replaced: the identity is standard, the placement and the measurement are not | §10 |
| 9 | three references added: Boyacıoğlu and van Breugel [29], Gehér's matrices article [30] and dissertation [31] | bibliography |

## What did not change, deliberately

The experimental record is untouched: the 160-run validation, the 320-run objective-specific study
with its rehearsal disclosure, the separately reported outcome-unseen block 16..31 including its
first attempt with an invalid measurement, the storage interventions and the decoder transport. No study was
pooled and no claim was manufactured from gate counts.

**Existing experiments do not become unnecessary or false because their target has prior art**, and
more repetitions would not have supplied novelty. None were run.

## Still open, and not for this agent

Send the Gehér source and the overlap table to the authorized legal representative and compare
actual claim combinations for [internal record]. A claim-level audit of a theory brief establishes
neither invalidity nor safety. Filing, software terms and external release remain owner decisions.
