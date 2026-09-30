# Preprint Public Release Candidate v1.2

**Manuscript:** `PREPRINT_PUBLIC_BUILD_v1.2_20260917_EN.md`  
**Title:** *Non-Normality Allocates Fisher Capacity; Isolation Preserves It*  
**Author:** Jeonghoon Lee, Attractor Dynamics Inc.

## Status

This package is a strengthened **public-release candidate**, not evidence that a journal has accepted the work and not a substitute for the author's final publication or legal review.

The v1.2 revision adds four claim-bearing elements to v1.1:

1. a finite-horizon Cesàro/commutant certification theorem with an explicit spectral-gap condition;
2. an end-to-end store-information operator and store-optimized direction experiment;
3. an exact and approximate post-write isolation theorem;
4. sampled generalized-least-squares decoding under invertible post-write holds.

It also corrects the Kerg et al. comparison, narrows the interpretation of the paired factorial, discloses that the original oracle is a matched write-block intervention rather than an independently optimized store direction, and removes the superseded one-page summary and internal governance archive from the public candidate.

## Main new receipt

`results/preprint_v1_2_strengthening_20260917/V1_2_STRENGTHENING_RECEIPT.json`

Headline results:

- finite-horizon relative operator error at `n=2048`, median `8.44e-6` over 8 paired instances;
- store-oracle / write-oracle store-total ratio, median `1.224`, range `1.112–1.582`;
- sampled decoder variance / theoretical variance, median `1.017`, range `0.969–1.032`;
- exact compensated-readout maximum discrepancy `7.99e-15`;
- approximate-isolation lower bound passed in all randomized and equality controls.

## Reproduction

See `REPRODUCE.md`. The fastest claim-bearing command is:

```bash
python scripts/reproduce_public.py --stage strengthening
```

The legacy census and factorial scripts retain their original review-gate checks for provenance. The wrapper supplies the included review records. Run full reproduction in a disposable copy because legacy scripts may rewrite their result directories.

## Scope

All experiments are synthetic, linear and Gaussian unless otherwise stated. The package does not establish that a trained neural network will learn the selected directions, that Fisher concentration yields useful task performance, or that the proposed decomposition dominates modern recurrent baselines.
