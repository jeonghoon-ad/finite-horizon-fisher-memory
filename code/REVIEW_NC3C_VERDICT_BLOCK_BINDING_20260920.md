# Code review — `nc3c_verdict.py` block-identity binding

**Script reviewed:** `code/nc3c_verdict.py`
**SHA-256 of the reviewed bytes:** `338aec69a33288eb858a38d9cb71f902af7014144ad88a5116c57a8667639737`
**Previous frozen digest:** `b86ee71f5e8f5b6ec6edd3cef4d378a2a80958109ac9a12fea7a30d0211b1067`
**Diff:** `docs/nc3c_block2/VERDICT_TOOL_AMENDMENT_20260920.diff` (35 lines)
**Reviewed:** 2026-09-20, before the tool was applied to the second block.

The digest above is recorded here deliberately: the execution wrapper warns that a review
which does not name a script's SHA proves only that some file is newer than the script.

## The problem it fixes

The tool refused to run on the author-approved block:

```
RuntimeError: CELL_LEVEL carries draws [16..31] for nonnormal; expected exactly [0..15]
```

The row-identity guard compared the CSV's draw indices against the hard-coded first block,
so the tool could not certify **any** block but the one it shipped with.

## The change

One constant is rebound. `block` is now read from `RUN_RECEIPT.json`'s `carrier_draws`, and
the three CSVs must carry exactly that set for each writer type.

## What is preserved, and what is strengthened

- **The denominator is untouched.** The check immediately above still requires
  `len(receipt["carrier_draws"]) == len(X.CARRIER_DRAWS)`, so a run with fewer or more than
  the preregistered 16 draws is still refused. A rehearsal with `--draws 2` still cannot
  produce a verdict.
- **The invalid-cell refusal is untouched.** A receipt listing invalidated cells still
  aborts before any statistic.
- **The guard is strictly stronger.** It previously answered "are these rows from block
  0..15?". It now answers "do these rows agree with the receipt that claims to describe
  them?", which catches CSV/receipt mismatch for any block, including the original one.
- **No threshold, gate, statistic, aggregation, pairing or decision rule is touched.** The
  `GATES` table, the bootstrap, the permutation test and the Holm correction are unchanged.

## independent reviewer pass

- *Could a run now self-certify with an arbitrary block?* Only within the preregistered
  count, and the block that a run may execute is fixed by the freeze record and the
  use-history preflight before any training. The verdict tool is not the gate that decides
  which block is legitimate; it checks internal consistency.
- *Could a receipt be edited to match doctored CSVs?* That was equally true before for the
  original block, and it is outside this guard's scope. Package hashes cover it.
- *Does it change the first block's verdict?* No: for that run `receipt["carrier_draws"]` is
  `[0..15]`, so `block` equals the old constant and the comparison is identical. This was
  re-verified by recomputing the frozen verdict after the change.

**Verdict: cleared.** The change is a binding correction, not a method change.
