# Code review — `run_nc3c.py` launcher amendment for the approved second block

**Reviewed:** `code/run_nc3c.py`, 2026-09-20.
**Prior frozen digest:** `61ae4b99df252860df1093b37d287694d13786300281bd39c65478a9e44cd856`
(`docs/nc3c/CODE_FREEZE_NC3C_20260918.txt`).
**Authority for the change:** `AGENT_HANDOFF_APPROVED_FINAL_RUN_AND_AD023_FILING_20260920_EN.md`
§4.1 and §4.3, which approve one confirmation on `NC-3C-V1` draws 16..31 and require a
"narrowly identified launcher/configuration amendment" when the runner hard-codes the block.
**Exact diff:** `docs/nc3c_block2/LAUNCHER_AMENDMENT_20260920.diff`.

## What changed

1. `--block LO-HI` selects the carrier-draw range. Default is the frozen block, so the
   command that produced the 2026-09-18 results is unchanged in behaviour.
2. `--results-dir` selects the output directory, and a **non-default block without its own
   results directory is refused**, so a new block cannot overwrite a frozen one.
3. `frozen_seed_list` and `freshness_audit` take the block as an argument, defaulting to the
   frozen block.
4. `freshness_audit` now also compares the requested block against **every NC-3C-V1 draw this
   lane has already instantiated**: the 2026-09-18 full-budget block 0..15, and draw 9900
   burned by the 2026-09-19 timing probe. Previously it compared only against the completed
   NC-TM-V1 lane, which would not have caught a collision with our own earlier NC-3C work.
5. The receipt records `carrier_block` and `results_dir`.

## What did not change

Verified by hash against the freeze records: **16 of 16 scientific modules unchanged**, and
both frozen authority documents unchanged. The carrier construction, seed derivation, the
validity battery, the optimizer, precision, amplitude, training horizon, target times, sample
counts, step count, aggregation, thresholds and invalid-run rules are the frozen ones. No
decision rule was touched. `FROZEN` is untouched.

## independent reviewer pass: what could this break?

- **Silent overwrite of the frozen results.** Refused by construction: a non-default block
  with the default directory raises before any work. Checked by running it.
- **Audit covering fewer draws than the run.** `freshness_audit` and `frozen_seed_list` are
  both passed the same `block` the run uses, and `run_stream_names` is still enumerated from
  the code paths rather than hand-listed, so stream coverage cannot silently fall behind.
- **The new prior-use comparison excluding the block itself.** `prior_nc3c` filters out any
  draw that is in `draws`, so re-auditing the frozen block cannot report a self-collision and
  fail a legitimate re-audit. Checked by re-running the default audit.
- **`--draws N` truncation.** Now slices the selected block rather than the frozen tuple,
  which is what a partial rehearsal of a chosen block should mean. Rehearsal remains marked
  `nc3c_rehearsal` with `budget_is_frozen: false`.
- **A block that is not fresh.** Still raises `RuntimeError` before any training, now on four
  collision sets instead of two.
- **Scientific reach.** The amendment cannot change what is measured: it only decides which
  draw indices are built and where the outputs land.

## Test evidence required before the claim-bearing run

Per §4.3 the administrative path is exercised **only on already burned data**: the default
audit re-run, and a `--block 9900` audit into a scratch directory. The new block 16..31 is
audited but not trained until the Linux binding is in place.

**Verdict: the amendment is administrative and is cleared for use.** It does not touch the
scientific modules, and it strictly strengthens the pre-run freshness check.

---

## Addendum, 2026-09-20 — control 31's negative control, found by running it

The first execution of block 16..31 returned **32 of 32 cells INVALID**, with `failed: []`
and `instruments_that_cannot_fail: ['31_seed_freshness']` in every cell. No control failed;
one control could not fail.

**Cause.** `control_seed_freshness`'s negative branch seeded the *completed lane's* namespace
at the cell's own draw index and asked whether it collided with the completed lane's reserved
seeds. That collides only when the index is itself reserved there, which is true of draws
0..15 and false of 16..31. On the new block the corrupt branch found no collision, so the
instrument was correctly reported as unable to fail, and the frozen rule marked every cell
invalid. **The carriers were not the problem, and the freshness fact was never in doubt**: the
independent preflight had already shown 0 collisions on all four axes.

**Repair.** The corrupt branch is mapped onto a reserved index,
`G.CONFIRMATORY_DRAWS[draw_index % 16]`, so the collision it must exhibit is guaranteed by
construction for any block. **The positive branch is unchanged**, and still asks exactly the
protocol's question: does any `NC-3C-V1` seed of this draw equal a seed the completed lane
used?

**Tested on already-exercised indices only**, never on an unexercised draw's outcome:

| draw | status | pass | instrument_valid |
|---:|---|---|---|
| 0 | old block, exercised 2026-09-18 | True | True |
| 15 | old block, exercised 2026-09-18 | True | True |
| 9900 | burned by the timing probe | True | True |
| 16 | new block, seeds already enumerated by the preflight | True | True |

Draws 0 and 15 behaved identically before the repair, so there is no regression on the block
the frozen results came from.

**This is the one corrected repeat run allowed by the measurement protocol.** A second run with an invalid measurement would
be `REDESIGN_REQUIRED / NO SCIENTIFIC VERDICT`, not a third attempt.
