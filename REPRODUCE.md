# Reproduction

Run from the root of a freshly extracted copy or a fresh clone.

## Verify first

```bash
shasum -a 256 -c SHA256SUMS
export PYTHONDONTWRITEBYTECODE=1
```

Python 3.11 or 3.12 with the packages in requirements.txt is required. BLAS thread caps may be set for local resource management without changing experiment parameters.

## Reanalyse released data

In one extraction, create the two legacy input aliases and run:

```bash
ln -s nc1_nc2_confirmatory results/confirmatory
ln -s nc3c_confirmatory results/nc3c
python code/nc_verdict.py --out NC1_VERDICT_RECOMPUTED.json
python code/nc3c_verdict.py --out NC3C_VERDICT_RECOMPUTED.json
python code/make_claim_ledger.py
python code/check_cesaro_identification.py
python code/make_public_figures.py --out reproduced/public_figures
python code/make_nc_figures.py --out reproduced/nc_figures
python code/make_section7_figures.py --out reproduced/section7_figures
shasum -a 256 -c SHA256SUMS
```

The ledger wrapper runs the original generator, applies the published null-population correction, and compares all 78 entries with the shipped ledger. It writes CSV and JSON files to reproduced/claim_ledger. Runtime timestamps and source identities are recorded separately from numerical comparisons. It computes the values rather than copying the reference ledger.

## Retrain NC-1/NC-2 and exploratory controls

Use a separate disposable extraction. Verify its checksums first, set PYTHONDONTWRITEBYTECODE=1, and do not create the two input aliases. In that copy:

```bash
python code/run_stage.py smoke
python code/run_stage.py pilot
python code/run_stage.py confirmatory
python code/run_exploratory.py --amplitude 1.5 --train-horizon 128
shasum -a 256 -c SHA256SUMS
```

These commands create results/smoke, results/pilot, results/confirmatory and results/exploratory; none is a shipped output directory. The confirmatory parameters are read from the new pilot receipt.

## Retrain the second NC-3C block

Use another disposable extraction, verify checksums first, set PYTHONDONTWRITEBYTECODE=1, and do not create the input aliases:

```bash
python code/run_nc3c.py --block 16-31 --results-dir results/nc3c_block2_20260920
python code/nc3c_verdict.py --results results/nc3c_block2_20260920 --out results/nc3c_block2_20260920/VERDICT.json
shasum -a 256 -c SHA256SUMS
```

The original first-block fresh-launch command intentionally rejects its already-exercised block. Its verdict can be recomputed from the shipped data above. Replaying the second block is a numerical reproduction, not a new outcome-unseen confirmation. The invalid first execution and the development-exposure disclosures remain in the release.

## Sources and document build

The five upstream public derivatives retain the numerical implementations. Their metadata, reproduction labels and source-hash pins differ from the originals. MANIFEST.json records both hashes, and the import checks and upstream reproduction wrapper verify the packaged files. The originals are archived outside this package. The review records distinguish original sources from packaged derivatives; they do not claim a new independent review.

The original NC scientific code remains byte-identical except for the declared dependency-hash constants in nc_geometry.py. The retained source comments and licence terms are unchanged. To reproduce the upstream checks, use vendor/PREPRINT_PUBLIC_RELEASE_v1.2_20260917/scripts/reproduce_public.py.

Build the manuscript with code/build_preprint_v2_2.sh, pandoc 3.11 and Tectonic 0.16.9. The existing basename is retained, and the visible date is 30 September 2026. Rebuild only in a disposable source copy: the build rewrites the document files and is separate from the reproduction commands above.

NumPy QR can produce different matrix bits across LAPACK implementations. Compare the recorded numerical invariants and labels. Source hashes identify exact code bytes; they do not guarantee identical matrix bits across platforms.

## Manuscript figure presentation

After the result links above, render all seven manuscript figures into a new directory:

```bash
python code/make_public_figures.py --out reproduced/public_figures
```

This presentation layer uses the original vendor plotting functions and frozen CSVs, with white backgrounds and revised label placement. The vendor source and numerical inputs are unchanged. Section 7 figure reproduction uses the same presentation layer. The census axis label denotes forward gain, as in its source column and the manuscript caption.


## Appendix C.2 run record

The run record for the Appendix C.2 checks is `vendor/PREPRINT_PUBLIC_RELEASE_v1.2_20260917/results/preprint_v1_2_strengthening_20260917/V1_2_STRENGTHENING_RECEIPT.json`. This is a path within the extracted Supplementary Material S1 package.
