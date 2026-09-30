# v2.1 to v2.2, 2026-09-23

**One scientific statement corrected, two internal inconsistencies fixed, one figure annotation
corrected, and the claim ledger corrected to match.** Found by a comparison of this manuscript with the
filed patent specification, which had already corrected the same statement.

| Where | Change |
|---|---|
| §7.3 null-direction paragraph | v2.0 and v2.1 said every null-direction row has J = 0 and that the positive values are round-off. That holds only under isolation. Now: under isolation, round-off residuals, median 1.86e-32, max 1.53e-30 over 160 rows; with the coupling open, leaked information from 8.3e-6 to 7.1e-3 over the corresponding 160 rows. The two populations are not pooled. Recomputed from `results/nc1_nc2_confirmatory/DIRECTION_ROWS.csv` |
| §7.3 range sentence | "over the directions carrying non-zero information" became "over every direction except the null direction", since the open null rows carry non-zero information and are excluded |
| §2 hypotheses | "tr M_n = N/ε" contradicted the paper's own ε-free M_n with trace N. Now: the Fisher matrix is M_n/ε, of trace N/ε, while M_n and tr M_n = N do not depend on ε |
| §4 table | the c = 4 elliptic row said 3 angles; Appendix B has four (0.7, 1.1, 2.0, sqrt c) |
| Figure 5 annotation | said all 320 excluded rows are round-off; now states the isolated and open cases. `code/make_section7_figures.py` annotation and docstring changed; Figures 6 and 7 regenerate unchanged |
| Claim ledger | `nc1.v_bottom.J`, a median pooled over both populations (4.17e-6), marked NOT FOR USE; four per-population entries added; the note on `nc1.J_range.all_rows_orders` corrected. 74 to 78 entries |
| Licence | `code/LICENSE` added: CC BY-NC 4.0, author decision 2026-09-23, official legal code text; `LICENSE_DECISION_REQUIRED.md` removed as resolved; `LICENSE_SCOPE.md` and `RELEASE_STATUS_20260923.md` updated. The manuscript PDF is unchanged by this |
| REPRODUCE.md | documents the lane directory names the scripts expect and the two links that make the figure command work; without them it silently writes no figure. This defect was present in v2.0 |

Verified: v2.1 to v2.2 is four manuscript hunks; two builds give byte-identical PDFs, 31 pages, no
overfull boxes, no em dash. Every file under `results/`, `preregistration/` and `vendor/` is byte-identical
to v2.0.
