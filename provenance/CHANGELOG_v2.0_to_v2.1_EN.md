# v2.0 to v2.1, 2026-09-23

**No technical content changed.** Six wording changes and typography only, listed exhaustively.

| Where | Change |
|---|---|
| Title block | date `Public build v2.0 --- HELD FROM POSTING` became `23 September 2026` |
| First page | the posting-hold notice removed, its condition being met (receipt for [office identifier omitted] [application identifier omitted]) |
| First page | the build note rewritten without internal lineage vocabulary; the ledger pointer kept |
| §2 overview | an em-dash parenthesis became a comma parenthesis |
| §3 sampled-decoder check | the three-readings sentence recast with a colon; same three readings, same maximum discrepancy 8.0e-15 |
| §7 table, two rows | em dashes became commas |
| Release contents | "Superseded one-pagers, internal handoffs, governance records and unrelated mirror metadata" became "Superseded drafts, internal working records and unrelated metadata" |
| Typography | bold emphasis removed from 74 spans in running text and tables; bold kept for 38 run-in headings and 9 named terms and labels |

Verified: once `**` markup is ignored, a line diff of the v2.0 and v2.1 sources shows exactly the six
wording hunks above. The v2.1 PDF has no em dash. Two builds give byte-identical PDFs; 31 pages, no
overfull boxes. Every file under `results/`, `claim_ledger/`, `preregistration/` and `vendor/` is
byte-identical to v2.0.
