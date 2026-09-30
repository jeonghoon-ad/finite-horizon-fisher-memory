# Clean-build verification, and the exact scope of what was verified

Handoff section 11 item 4. The handoff also sets the boundary this report must respect:
*"Your release verification must distinguish what was recomputed from raw artifacts from
what is merely stated in a report."*

Method: the public release candidate was copied into an empty directory outside the lane
and exercised there. Nothing in the lane was read during the checks below.

---

## 1. What was recomputed from raw artifacts

| check | result |
|---|---|
| package self-verification, `shasum -c SHA256SUMS` | **199 of 199 OK**, 0 failed |
| manuscript rebuilt inside the clean copy by its own shipped `code/build_preprint_v2_0.sh` | PDF reproduced **byte-identical**, `d2afed2de8517186d85bb0f1b202f7655e6f7ed3cc23319fa923486ec8ad13a7` |
| typeset output | all **28** pages rendered and inspected, and again after every later change; **0** overfull boxes, and the build now fails on any |
| NC-1/NC-2 verdict recomputed from the released CSVs by the released code | **`FULL_PASS`**, matching the shipped verdict |
| NC-1/NC-2, leaf-by-leaf comparison against the shipped `VERDICT.json` | 1,201 leaves compared, **9 differ**, worst relative difference `1.135e-13` |
| **all 26 NC-1/NC-2 gate observables and their pass flags** | **bit-identical** |
| NC-3C verdict recomputed from the released CSVs | **`FULL_PASS`**, matching |
| NC-3C, leaf-by-leaf | 891 leaves compared, 16 differ, worst relative difference `1.610e-13` |
| claim ledger | all 52 entries regenerated from the frozen CSVs by `make_claim_ledger.py`; no value copied from a report |
| parity artifacts | recomputed: scan parity 32 cells, decoder transport 32 rows |

**The 25 differing leaves are all in one family**: the calibration slope, its intercept,
its standard error and its interval. Those come from `numpy.linalg.lstsq`, which is not
bit-reproducible across BLAS implementations; the original verdicts were computed on
Linux/OpenBLAS and this verification on macOS/Accelerate. **No quantity that a decision
rule reads differs at all.** The calibration slope is reported in the manuscript, not
gated, and it is stable to twelve significant figures.

## 2. What was NOT recomputed, and is therefore only asserted

Stated plainly, because the handoff requires the distinction:

- **The training runs themselves were not re-executed.** The 160 and 320 optimization runs
  are taken from their frozen receipts and CSVs. A full re-execution would need paid
  compute, which this handoff does not authorize.
- **The per-cell validity batteries were not re-run.** Their records are read from the
  per-cell JSONs.
- **Figures 1 to 4 were not re-rendered.** They come from the vendored v1.2 release
  unchanged, and their generator is released with them. **Figures 5 to 7 were re-rendered**
  from the frozen CSVs by `code/make_section7_figures.py` on 2026-09-19 and inspected as
  images; that script computes no Fisher quantity, gate or statistic, and reads every
  plotted value, the amplitude and the training horizon from the frozen outputs.
- **The instance-side backup verification cannot be repeated.** The instance was destroyed
  on 2026-09-18 at 15:12 under a separate explicit owner instruction. See section 4.
- **Cross-platform carrier agreement** is quoted from the earlier measurement
  (`9.5e-15` absolute, `1.8e-15` relative on `lambda_max`); it was not re-measured here,
  because it needs two machines.

## 3. Missing sources, marked rather than synthesized

The build script marks any absent source `MISSING` and never reconstructs it from a
summary table. At build time one entry, an internal lineage record that is not part of this release, was missing and has since been supplied. The package manifest records the state of every
entry.

## 4. Instance and backup boundary — a correction to the handoff's premise

Handoff sections 1 and 10 instruct that instance `51391833` be kept stopped and not
destroyed without a separate explicit owner instruction.

**That instruction had already been given and executed before this handoff arrived.** The
owner directed destruction verbatim ("instance 는 파기 해라. 백업 했지?"), and the instance
was destroyed on 2026-09-18 at 15:12. Fleet is zero for this lane. The action was
authorized; the handoff was written without knowledge of it.

Consequences, stated rather than papered over:

- The **retained per-file path/size/SHA manifest** that section 10 asks for as a
  precondition for recommending destruction **cannot now be produced for the box side**.
  What exists is the verification performed while the instance was running: a
  file-by-file SHA-256 comparison of `results/nc3c` between box and local, 40 files each
  side, 0 mismatches, no box-only files. The per-file listing from that comparison was not
  retained as an artifact.
- Section 10 says a summary reading "40/40" is not itself the retained listing. **Agreed,
  and it is not presented as one.**
- The older 57-file backup summary for the first instance is likewise a summary, and its
  missing per-file manifest has **not** been recovered. It is not presented as recovered.
- All backed-up material is present locally and verifies: the lane manifest covers every
  file, and the release candidate self-verifies at 190 of 190.

## 5. Reproduction command surface

`REPRODUCE.md` in the candidate lists the commands. Reproduction is defined on the
portable invariants in each run receipt, not on byte equality of the carrier arrays,
because `numpy.linalg.qr` is not bit-reproducible across LAPACK implementations. The
original runs were launched through an internal review-gate wrapper that is not part of
the release; the scientific code is released unchanged and only the launch path differs. A
wrapper hash is not the hash of the original scientific run.
