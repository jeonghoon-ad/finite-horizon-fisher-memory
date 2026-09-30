# Reproduction Guide

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Recorded build environment:

```text
Python 3.13.x
NumPy 2.3.5
SciPy 1.17.0
Matplotlib 3.10.8
float64
```

## 2. Strengthening results only

This is the claim-bearing v1.2 addition and normally finishes quickly:

```bash
python scripts/reproduce_public.py --stage strengthening
```

Expected receipt:

```text
results/preprint_v1_2_strengthening_20260917/V1_2_STRENGTHENING_RECEIPT.json
```

The script regenerates:

- finite-horizon certification checks;
- the near-degenerate negative control;
- end-to-end store-operator selection;
- sampled GLS decoder verification;
- approximate-isolation tests.

## 3. Figures

```bash
python scripts/reproduce_public.py --stage figures
```

Outputs are written to `figures/`.

## 4. Core historical measurements

```bash
python scripts/reproduce_public.py --stage core
```

This runs the replicated census, angle control, paired factorial, v1.2 strengthening and figure build. The 128-instance census can take several minutes depending on BLAS and CPU configuration.

## 5. Full historical reproduction

```bash
python scripts/reproduce_public.py --stage full
```

This also runs the original instrument and single-instance sweeps. Run in a disposable package copy because the historical scripts write their canonical result directories.

## 6. Validation

```bash
python scripts/validate_release.py
sha256sum -c SHA256SUMS.txt
```

`validate_release.py` checks JSON readability, required files, figure references, Python compilation, and the v1.2 PASS receipt. It does not re-prove the written theorems.
