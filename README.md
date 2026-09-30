# Finite-Horizon Fisher Memory in Two-Sided Power-Bounded Recurrent Systems

Reproduction materials for the preprint by Jeonghoon Lee (Attractor Dynamics Inc.).

We study how noisy linear-Gaussian recurrent systems allocate Fisher information during writing, transfer it to a downstream store, and preserve it after writing ends. The training checks keep the recurrent carriers fixed. The results concern this model class; they do not establish performance gains in a language model.

The repository contains the manuscript, code, result tables, run records and preregistration files. Its scripts reproduce the paper's calculations, figures and training checks.

## Read the paper

- [PDF](manuscript/PREPRINT_PUBLIC_BUILD_v2.2_20260923_EN.pdf)
- [Markdown](manuscript/PREPRINT_PUBLIC_BUILD_v2.2_20260923_EN.md)
- [LaTeX](manuscript/PREPRINT_PUBLIC_BUILD_v2.2_20260923_EN.tex)

## Reproduce the checks

From the root of a fresh clone or extracted release, verify the files first:

```bash
shasum -a 256 -c SHA256SUMS
```

Then follow [REPRODUCE.md](REPRODUCE.md). It gives the commands in order, including the two input links needed to reanalyse the released data. Each training row uses a separate disposable copy. Python 3.11 or 3.12 and the packages in [requirements.txt](requirements.txt) are required.

The [claim ledger](claim_ledger/) lists every reported number with its source. Numerical reproduction and new outcome-unseen evidence are distinct; the manuscript and run records state the development exposure and execution history.

## Contents

- `manuscript/`: the paper as Markdown, LaTeX and PDF, with its figures.
- `claim_ledger/`: reported numbers and their sources.
- `code/`: analysis, training and figure scripts.
- `results/`: recorded outputs and figure inputs.
- `preregistration/`: protocols, machine-readable preregistrations and amendments.
- `provenance/`: development-exposure disclosures, build verification, prior-art review, changelogs and source-derivative records.
- `vendor/`: the v1.2 upstream release used by this work.

## Licence

The code is released under CC BY-NC 4.0 ([code/LICENSE](code/LICENSE)). Commercial use requires a separate written licence. [LICENSE_SCOPE.md](LICENSE_SCOPE.md) states the terms for each part of the repository.
