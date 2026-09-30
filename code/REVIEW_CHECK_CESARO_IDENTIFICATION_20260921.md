# Code review — `check_cesaro_identification.py`

**Script reviewed:** `code/check_cesaro_identification.py`
**SHA-256 of the reviewed bytes:** `5d7628890cf697eb5bb8c8b02bbb01e43e65a149b670b441cd318619898db918`
**Reviewed:** 2026-09-21, before it was cited in the manuscript.

## Purpose

Manuscript §3.3.0 states that $M_\infty$ is the inverse of the classical Cesàro asymptotic limit
of $W^\top$, and says the numerical check ships with the code. This is that check.

## What it does and does not do

- It computes two residuals per horizon and prints them. **It writes no file, computes no gate,
  no statistic and no verdict**, and it is not on any decision path.
- It uses `nc_geometry.build_carrier_draw`, so it checks the identity on the same carrier family
  the paper measures, not on a convenient synthetic example.
- It is not evidence of novelty in either direction. It shows the identification is arithmetically
  right, which is precisely why the manuscript attributes the limit object to the classical
  literature instead of claiming it.

## independent reviewer pass

- *Could it appear to converge without converging?* The printed ratio makes that checkable: the
  claim is $O(1/n)$, so a ratio near 2 per doubling is the expected behaviour, and a ratio near 1
  would show a plateau. Measured on draw 0: 1.87, 2.12, 2.03, 1.81 for the conditioned writer.
- *Is the normal case trivially passing?* It is: the residual sits at 1e-13 to 1e-12, the correct
  behaviour since $M_n=I$ exactly there, and it is reported rather than hidden. **Its printed ratio is
  about 0.5, not 2, and that is not a failure**: at machine precision the residual is accumulated
  rounding, which grows slowly with the number of terms rather than decaying like $1/n$. The ratio
  column is only meaningful where the residual is above float noise.
- *Cost.* Two dense loops per horizon at $N=32$; the default horizons run in under a minute on
  one core. No rental resource is implied.

**Verdict: cleared.** It supports a sentence the manuscript now makes, and it makes that sentence
checkable by a reader.
