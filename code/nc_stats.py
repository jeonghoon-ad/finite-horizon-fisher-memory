"""Frozen statistical analysis for the Neural Computation trained-memory lane (NC-TM-V1).

Authority (read-only, under ../authority/):
  NEURAL_COMPUTATION_TRAINED_MEMORY_VALIDATION_PREREG_v1_20260918.yaml    (machine authority)
  NEURAL_COMPUTATION_TRAINED_MEMORY_VALIDATION_PROTOCOL_v1_20260918_EN.md (scientific authority)
  PREPRINT_PUBLIC_BUILD_v1.2_20260917_EN.md                               (theorem context)

Protocol section 9 fixes the machinery implemented here:

  9.1 independent unit    the carrier draw.  Optimizer seeds are repeated
                          measurements NESTED inside a carrier draw and never
                          enter a test as independent samples (9.4).
  9.2 aggregation         per carrier draw, take the median endpoint over
                          optimizer seeds, then compare paired conditions
                          (normal / non-normal, open / isolated) WITHIN a draw.
  9.3 uncertainty         10,000-draw paired bootstrap over carrier instances,
                          95% confidence intervals, paired permutation tests as
                          the secondary test, and a Holm correction over the two
                          primary families (alignment, behavior).  The
                          preregistration does not name a permutation statistic;
                          see `paired_permutation_test` for the choice made here
                          and the measured reason for it.
  9.4 distributions       report all carrier-level points, median and IQR,
                          paired differences, and sign consistency.

Protocol section 10 fixes the decision rules these statistics feed.  The
comparator forms are reproduced here only so that `evaluate_gate` has a single
vocabulary; the thresholds themselves live in the YAML and are supplied by the
caller, never hard-coded in this module.

  10.1 alignment PASS        median R_J >= 0.90; 95% bootstrap lower bound on
                             R_J >= 0.80; at least 14 of 16 carrier draws with
                             R_J >= 0.80; median R_beh >= 0.80; accuracy
                             calibration MAE <= 0.02.
  10.2 causal geometry PASS  Spearman(J, recalibrated accuracy) >= 0.95; at
                             least 90% of direction pairs whose Fisher gap
                             exceeds 5% ordered the same way behaviorally;
                             v_bottom below the random-direction median in at
                             least 14 of 16 draws.
  10.3 retention PASS        isolated accuracy range <= 0.01; calibration
                             MAE <= 0.02; open-versus-isolated ordering agreeing
                             with propagated J_H in at least 15 of 16 draws.
  10.4 kill rules            R_J < 0.75 or R_beh < 0.60.

Purity contract.  Every public function is pure: no file, network, or console
I/O, no module-level mutable state, and no randomness except through an explicit
`seed` argument routed into `numpy.random.default_rng`.  numpy is the only hard
dependency.  scipy is optional and is touched in exactly one place: the
`__main__` self-test cross-checks `spearman` against `scipy.stats.spearmanr`
when scipy is importable, and skips that cross-check otherwise.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Statistic = Callable[..., Any]

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "CONFIDENCE_LEVEL",
    "COMPARATORS",
    "median_within_carrier",
    "paired_bootstrap",
    "paired_bootstrap_difference",
    "paired_permutation_test",
    "holm_correction",
    "spearman",
    "ordered_pair_agreement",
    "bootstrap_lower_bound",
    "summarize",
    "evaluate_gate",
]

# --------------------------------------------------------------------------
# Frozen constants, taken verbatim from the YAML preregistration (`analysis:`).
# They are defaults only; a caller may not widen them without amending the
# preregistration.
# --------------------------------------------------------------------------
BOOTSTRAP_RESAMPLES: int = 10_000        # analysis.bootstrap_resamples
CONFIDENCE_LEVEL: float = 0.95           # analysis.confidence_level

#: Comparator vocabulary accepted by :func:`evaluate_gate`.
COMPARATORS: tuple[str, ...] = (">=", "<=", ">", "<")


# --------------------------------------------------------------------------
# Internal helpers (private; not part of the frozen public surface).
# --------------------------------------------------------------------------
def _as_1d(values: npt.ArrayLike, name: str) -> FloatArray:
    """Coerce `values` to a non-empty 1-D float array, or raise."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _as_paired(a: npt.ArrayLike, b: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Coerce two arms to non-empty 1-D float arrays of equal length, or raise.

    Equal length is what makes the comparison paired: entry i of both arms must
    be the same carrier draw (protocol 9.2).
    """
    left = _as_1d(a, "a")
    right = _as_1d(b, "b")
    if left.shape != right.shape:
        raise ValueError(
            "paired arms must have one entry per carrier draw; "
            f"got {left.shape} and {right.shape}"
        )
    return left, right


def _check_resamples(resamples: int) -> int:
    """Validate a resample count."""
    count = int(resamples)
    if count < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    return count


def _apply_statistic(statistic: Statistic, samples: FloatArray) -> FloatArray:
    """Apply `statistic` row-wise to a [resamples, n] matrix.

    Tries the vectorised `axis=1` form first (numpy reducers support it) and
    falls back to a per-row loop for statistics that do not take `axis`.  Both
    paths produce identical values; only speed differs.
    """
    reduced: FloatArray | None = None
    try:
        candidate = np.asarray(statistic(samples, axis=1), dtype=float)
    except TypeError:
        candidate = None
    if candidate is not None and candidate.shape == (samples.shape[0],):
        reduced = candidate
    if reduced is None:
        reduced = np.asarray(
            [float(statistic(row)) for row in samples], dtype=float
        )
    return reduced


def _bootstrap_indices(
    rng: np.random.Generator, n: int, resamples: int
) -> npt.NDArray[np.int64]:
    """Carrier-draw indices, sampled with replacement, shape [resamples, n].

    The SAME index matrix is used for both arms of a paired comparison, which is
    what keeps the bootstrap paired rather than independent (protocol 9.3).
    """
    return rng.integers(0, n, size=(resamples, n), dtype=np.int64)


def _average_ranks(values: FloatArray) -> FloatArray:
    """Ranks of `values`, 1-based, with ties resolved to their average rank."""
    n = values.size
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks_sorted = np.empty(n, dtype=float)

    start = 0
    while start < n:
        stop = start + 1
        while stop < n and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # 1-based positions start+1 .. stop, averaged over the tie block.
        ranks_sorted[start:stop] = 0.5 * ((start + 1) + stop)
        start = stop

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


# --------------------------------------------------------------------------
# 1. Aggregation (protocol 9.2).
# --------------------------------------------------------------------------
def median_within_carrier(values_by_seed: npt.ArrayLike) -> FloatArray:
    """Median endpoint over optimizer seeds, within each carrier draw.

    Protocol 9.1/9.2: the carrier draw is the independent unit and optimizer
    seeds are nested repeated measurements.  Collapsing seeds to their median
    here is what prevents a downstream test from treating seeds as independent
    samples (forbidden by 9.4).

    Parameters
    ----------
    values_by_seed:
        2-D array-like of shape ``[n_carriers, n_seeds]``.

    Returns
    -------
    1-D float array of shape ``[n_carriers]``.
    """
    array = np.asarray(values_by_seed, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            "values_by_seed must be 2-D [n_carriers, n_seeds], "
            f"got shape {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"values_by_seed must have at least one carrier and one seed, "
            f"got shape {array.shape}"
        )
    return np.asarray(np.median(array, axis=1), dtype=float)


# --------------------------------------------------------------------------
# 2-3. Bootstrap over carrier draws (protocol 9.3).
# --------------------------------------------------------------------------
def paired_bootstrap(
    values: npt.ArrayLike,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    statistic: Statistic = np.median,
) -> dict[str, Any]:
    """Percentile bootstrap of `statistic` over carrier draws.

    Carrier draws are resampled with replacement (they are the independent unit,
    protocol 9.1), the statistic is recomputed on each resample, and the 2.5 /
    97.5 percentiles of the resulting distribution give the 95% interval
    (`analysis.confidence_level = 0.95`).

    Deterministic given `seed`.

    Returns
    -------
    dict with keys ``point``, ``ci_low``, ``ci_high``, ``resamples``, ``method``.
    """
    array = _as_1d(values, "values")
    count = _check_resamples(resamples)

    rng = np.random.default_rng(seed)
    indices = _bootstrap_indices(rng, array.size, count)
    distribution = _apply_statistic(statistic, array[indices])

    tail = 0.5 * (1.0 - CONFIDENCE_LEVEL) * 100.0   # 2.5 for a 95% interval
    ci_low, ci_high = np.percentile(distribution, [tail, 100.0 - tail])

    return {
        "point": float(statistic(array)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "resamples": count,
        "method": "percentile_bootstrap_over_carrier_draws",
    }


def paired_bootstrap_difference(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    statistic: Statistic = np.median,
) -> dict[str, Any]:
    """Percentile bootstrap of `statistic` applied to the paired difference a - b.

    The two arms are matched carrier draw by carrier draw (protocol 9.2:
    "compare paired normal/non-normal or open/isolated conditions within that
    draw").  Each bootstrap resample therefore draws ONE set of carrier indices
    and applies it to both arms, so the pairing survives resampling.  Resampling
    the arms independently would inflate the interval and is not what the
    preregistration specifies.

    Deterministic given `seed`.

    Returns
    -------
    dict with keys ``point``, ``ci_low``, ``ci_high``, ``resamples``, ``method``.
    """
    left, right = _as_paired(a, b)
    count = _check_resamples(resamples)

    rng = np.random.default_rng(seed)
    indices = _bootstrap_indices(rng, left.size, count)
    # Same indices on both arms: the paired difference is formed after the
    # matched carrier draws are selected.
    distribution = _apply_statistic(statistic, left[indices] - right[indices])

    tail = 0.5 * (1.0 - CONFIDENCE_LEVEL) * 100.0
    ci_low, ci_high = np.percentile(distribution, [tail, 100.0 - tail])

    return {
        "point": float(statistic(left - right)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "resamples": count,
        "method": "paired_percentile_bootstrap_over_carrier_draws",
    }


# --------------------------------------------------------------------------
# 4. Paired permutation test (protocol 9.3, secondary).
# --------------------------------------------------------------------------
def paired_permutation_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    statistic: Statistic = np.mean,
) -> dict[str, Any]:
    """Sign-flip permutation test on the paired differences a - b.

    The secondary test of protocol 9.3.  Under the null of exchangeable signs
    within a carrier draw, negating an individual draw's difference leaves the
    distribution unchanged.  Each permutation independently negates each
    carrier's difference and recomputes the statistic.

    The two-sided p-value uses the add-one (Monte-Carlo unbiased) form

        p = (1 + #{ |perm| >= |observed| }) / (1 + resamples)

    so it is never zero; its floor is ``1 / (1 + resamples)``.

    Choice of statistic.  The preregistration names the test but not its
    statistic.  This function defaults to the MEAN of the paired differences,
    the canonical sign-flip statistic, even though the rest of the module is
    median-centric because protocol section 10 states its thresholds on medians.
    The reason is resolution, not taste: under sign flips the median of the
    differences takes only a handful of values, so it cannot separate a strong
    effect from a weak one.  For a unanimous constant difference over eight
    carrier draws the exact median-based sign-flip p-value is 0.727 while the
    mean-based value is 0.0078 -- the median test cannot reject an effect that
    every single carrier draw shows.  Both facts are asserted in the self-test.

    Pass ``statistic=np.median`` explicitly to obtain the median variant.

    Deterministic given `seed`.

    Returns
    -------
    dict with keys ``observed``, ``p_value``, ``resamples``.
    """

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(a_arr)) and np.all(np.isfinite(b_arr))):
        # A non-finite input silently becomes the SMALLEST attainable p-value:
        # every comparison against NaN is False, so nothing counts as extreme and
        # p collapses to 1/(1+resamples).  A lost instrument would then be
        # reported as the strongest evidence the design can produce.  Refuse.
        raise ValueError(
            "paired_permutation_test received non-finite values "
            f"(a: {int(np.count_nonzero(~np.isfinite(a_arr)))} bad of {a_arr.size}, "
            f"b: {int(np.count_nonzero(~np.isfinite(b_arr)))} bad of {b_arr.size}). "
            "A non-finite statistic is an instrument failure, not evidence.")
    left, right = _as_paired(a, b)
    count = _check_resamples(resamples)

    differences = left - right
    observed = float(statistic(differences))

    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.array([-1.0, 1.0]), size=(count, differences.size)
    )
    permuted = _apply_statistic(statistic, signs * differences[None, :])

    extreme = int(np.count_nonzero(np.abs(permuted) >= abs(observed)))
    p_value = (1.0 + extreme) / (1.0 + count)

    return {
        "observed": observed,
        "p_value": float(p_value),
        "resamples": count,
    }


# --------------------------------------------------------------------------
# 5. Holm-Bonferroni correction (protocol 9.3).
# --------------------------------------------------------------------------
def holm_correction(pvalues_by_name: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni step-down correction over a family of tests.

    Applied by the preregistration over the two primary families, alignment and
    behavior (`analysis.multiple_testing`).  Call it once per family; do not
    pool families into one call.

    With ``m`` tests and p-values sorted ascending, the adjusted value of the
    ``j``-th smallest (``j`` 1-based) is

        p_adj[j] = max_{i <= j} ( (m - i + 1) * p_raw[i] )

    capped at 1.0.  The running maximum enforces monotonicity, so an adjusted
    value never falls below the adjusted value of a smaller raw p.

    Returns
    -------
    dict mapping each name to ``{"p_raw", "p_adjusted", "rank"}`` where ``rank``
    is the 1-based position in ascending raw-p order (ties broken by the input
    order, deterministically).
    """
    names = list(pvalues_by_name)
    if not names:
        return {}

    raw = np.asarray([float(pvalues_by_name[name]) for name in names], dtype=float)
    if not np.all(np.isfinite(raw)):
        raise ValueError("Holm correction requires finite p-values")
    if np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ValueError("Holm correction requires p-values in [0, 1]")

    m = raw.size
    order = np.argsort(raw, kind="stable")        # stable => deterministic ties

    multipliers = np.arange(m, 0, -1, dtype=float)   # m, m-1, ..., 1
    stepped = multipliers * raw[order]
    adjusted_sorted = np.minimum(np.maximum.accumulate(stepped), 1.0)

    result: dict[str, dict[str, Any]] = {}
    for position, index in enumerate(order):
        result[names[index]] = {
            "p_raw": float(raw[index]),
            "p_adjusted": float(adjusted_sorted[position]),
            "rank": position + 1,
        }
    # Preserve the caller's key order in the returned mapping.
    return {name: result[name] for name in names}


# --------------------------------------------------------------------------
# 6. Spearman rank correlation (protocol 10.2).
# --------------------------------------------------------------------------
def spearman(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Spearman rank correlation, with average ranks for ties.

    Implemented directly on numpy: rank both vectors (ties share their average
    rank), then take the Pearson correlation of the ranks.  This is the estimator
    protocol 10.2 calls for between Fisher information J and recalibrated
    empirical accuracy across direction controls.

    Returns ``nan`` when either rank vector has zero variance, which happens when
    one input is constant or is entirely tied; a correlation is undefined there
    and must not be silently reported as 0.
    """
    left = _as_1d(x, "x")
    right = _as_1d(y, "y")
    if left.shape != right.shape:
        raise ValueError(
            f"spearman requires equal-length inputs, got {left.shape} and {right.shape}"
        )
    if left.size < 2:
        raise ValueError("spearman requires at least two observations")

    rx = _average_ranks(left)
    ry = _average_ranks(right)

    rx_centered = rx - rx.mean()
    ry_centered = ry - ry.mean()
    denominator = math.sqrt(
        float(rx_centered @ rx_centered) * float(ry_centered @ ry_centered)
    )
    if denominator == 0.0:
        return float("nan")
    return float((rx_centered @ ry_centered) / denominator)


# --------------------------------------------------------------------------
# 7. Ordered-pair agreement (protocol 10.2).
# --------------------------------------------------------------------------
def ordered_pair_agreement(
    j_values: npt.ArrayLike,
    accuracies: npt.ArrayLike,
    relative_gap: float = 0.05,
) -> dict[str, Any]:
    """Fraction of well-separated direction pairs that behavior orders as J does.

    Protocol 10.2 requires that "at least 90% of direction pairs ordered by a
    Fisher gap exceeding 5% are ordered the same way behaviorally".

    A pair ``(i, k)``, ``i < k``, is CONSIDERED when its Fisher gap is large
    relative to the larger of the two J values:

        |J_i - J_k| / max(J_i, J_k) > relative_gap

    A considered pair AGREES when ``sign(J_i - J_k) == sign(acc_i - acc_k)``.
    An accuracy tie has sign 0, never matches a non-zero Fisher sign, and so
    counts as a disagreement: the protocol asks for concordant ordering, and a
    tie does not order anything.

    Pairs whose ``max(J_i, J_k) <= 0`` are skipped: no relative gap is defined
    there.  Pairs involving a non-finite value fail the gap comparison and are
    likewise excluded from the denominator.

    Returns
    -------
    dict with ``n_pairs_considered``, ``n_agreeing``, ``fraction``.  ``fraction``
    is ``nan`` when no pair clears the gap, so an empty comparison can never be
    read as a pass.
    """
    j_array = _as_1d(j_values, "j_values")
    acc_array = _as_1d(accuracies, "accuracies")
    if j_array.shape != acc_array.shape:
        raise ValueError(
            "j_values and accuracies must be equal length, got "
            f"{j_array.shape} and {acc_array.shape}"
        )
    gap_threshold = float(relative_gap)
    if gap_threshold < 0.0:
        raise ValueError(f"relative_gap must be non-negative, got {relative_gap}")

    n = j_array.size
    considered = 0
    agreeing = 0

    for i in range(n - 1):
        for k in range(i + 1, n):
            j_i = float(j_array[i])
            j_k = float(j_array[k])
            denominator = max(j_i, j_k)
            if not (denominator > 0.0):          # zero / negative / nan guard
                continue
            gap = abs(j_i - j_k) / denominator
            if not (gap > gap_threshold):        # nan-safe: nan is excluded
                continue
            considered += 1
            j_sign = float(np.sign(j_i - j_k))
            acc_sign = float(np.sign(float(acc_array[i]) - float(acc_array[k])))
            if j_sign == acc_sign and j_sign != 0.0:
                agreeing += 1

    fraction = float(agreeing) / considered if considered > 0 else float("nan")
    return {
        "n_pairs_considered": considered,
        "n_agreeing": agreeing,
        "fraction": fraction,
    }


# --------------------------------------------------------------------------
# 8. One-sided bootstrap lower bound (protocol 10.1, condition 2).
# --------------------------------------------------------------------------
def bootstrap_lower_bound(
    values: npt.ArrayLike,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    statistic: Statistic = np.median,
    level: float = CONFIDENCE_LEVEL,
) -> float:
    """One-sided lower bound on `statistic` at `level`, over carrier draws.

    This is the quantity protocol 10.1 condition 2 calls the "95% bootstrap
    lower bound": the ``(1 - level)`` percentile of the bootstrap distribution,
    i.e. the 5th percentile at ``level = 0.95``.  It is NOT the lower end of the
    two-sided 95% interval returned by :func:`paired_bootstrap`, which is the
    2.5th percentile.

    Deterministic given `seed`.
    """
    array = _as_1d(values, "values")
    count = _check_resamples(resamples)
    confidence = float(level)
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"level must lie strictly in (0, 1), got {level}")

    rng = np.random.default_rng(seed)
    indices = _bootstrap_indices(rng, array.size, count)
    distribution = _apply_statistic(statistic, array[indices])

    return float(np.percentile(distribution, (1.0 - confidence) * 100.0))


# --------------------------------------------------------------------------
# 9. Distribution summary (protocol 9.4).
# --------------------------------------------------------------------------
def summarize(values: npt.ArrayLike) -> dict[str, Any]:
    """Carrier-level distribution summary required by protocol 9.4.

    Reports median and interquartile range alongside the range, the mean, and
    the sign consistency of the paired differences.  ``sign_consistency`` is the
    fraction of STRICTLY positive entries; zeros and negatives both count
    against it.
    """
    array = _as_1d(values, "values")
    q1, median, q3 = np.percentile(array, [25.0, 50.0, 75.0])

    return {
        "n": int(array.size),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "sign_consistency": float(np.count_nonzero(array > 0.0) / array.size),
    }


# --------------------------------------------------------------------------
# 10. Gate evaluation (protocol 10.1-10.4).
# --------------------------------------------------------------------------
def evaluate_gate(
    name: str,
    observed: float,
    comparator: str,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one frozen decision rule from protocol section 10.

    `comparator` must be one of :data:`COMPARATORS`; an unknown comparator
    raises rather than defaulting, so a typo cannot silently become a pass.

    A non-finite `observed` is ALWAYS ``pass = False`` and carries an extra
    ``"nan_observed": True`` key.  A NaN endpoint means the measurement did not
    produce a number; it is an instrument failure, not a satisfied gate, and it
    must never be coerced into one.
    """
    if comparator not in COMPARATORS:
        raise ValueError(
            f"unknown comparator {comparator!r}; expected one of {COMPARATORS}"
        )

    observed_value = float(observed)
    threshold_value = float(threshold)

    result: dict[str, Any] = {
        "name": str(name),
        "observed": observed_value,
        "comparator": comparator,
        "threshold": threshold_value,
    }

    if not math.isfinite(observed_value):
        result["pass"] = False
        result["nan_observed"] = True
        return result

    if comparator == ">=":
        passed = observed_value >= threshold_value
    elif comparator == "<=":
        passed = observed_value <= threshold_value
    elif comparator == ">":
        passed = observed_value > threshold_value
    else:  # "<"
        passed = observed_value < threshold_value

    result["pass"] = bool(passed)
    return result


# --------------------------------------------------------------------------
# Self-test.  Small synthetic arrays with hand-computed answers.
# --------------------------------------------------------------------------
def _self_test() -> None:
    # ---- 1. median_within_carrier -------------------------------------
    per_seed = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [10.0, 0.0, 2.0]])
    per_carrier = median_within_carrier(per_seed)
    assert per_carrier.shape == (3,), per_carrier.shape
    assert np.allclose(per_carrier, [2.0, 5.0, 2.0]), per_carrier
    # Even seed count -> midpoint of the two central values.
    assert np.allclose(median_within_carrier([[1.0, 3.0]]), [2.0])
    for bad in ([1.0, 2.0, 3.0], np.zeros((0, 3)), np.zeros((3, 0))):
        try:
            median_within_carrier(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("median_within_carrier accepted a bad shape")

    # ---- 2. paired_bootstrap ------------------------------------------
    constant = np.full(16, 0.91)
    boot_constant = paired_bootstrap(constant, seed=0, resamples=500)
    assert boot_constant["point"] == 0.91, boot_constant
    # Every resample of a constant array is that constant: zero-width interval.
    assert boot_constant["ci_low"] == 0.91, boot_constant
    assert boot_constant["ci_high"] == 0.91, boot_constant
    assert boot_constant["ci_high"] - boot_constant["ci_low"] == 0.0
    assert boot_constant["resamples"] == 500
    assert boot_constant["method"] == "percentile_bootstrap_over_carrier_draws"

    spread = np.linspace(0.80, 0.95, 16)
    boot_a = paired_bootstrap(spread, seed=7, resamples=2000)
    boot_b = paired_bootstrap(spread, seed=7, resamples=2000)
    assert boot_a == boot_b, "paired_bootstrap is not deterministic given seed"
    assert boot_a["ci_low"] <= boot_a["point"] <= boot_a["ci_high"], boot_a
    # The seed must actually steer the resampling.  Checked on the mean of an
    # irregular array: the median of a regular grid has a coarse, highly tied
    # bootstrap distribution whose percentiles legitimately coincide across
    # seeds, so it cannot detect a seed that is being ignored.
    irregular = np.array([0.8137, 0.9042, 0.8619, 0.7788, 0.9311, 0.8455,
                          0.8903, 0.8021, 0.9187, 0.7954, 0.8762, 0.9068,
                          0.8296, 0.8874, 0.9225, 0.8408])
    assert (
        paired_bootstrap(irregular, seed=7, resamples=2000, statistic=np.mean)
        != paired_bootstrap(irregular, seed=8, resamples=2000, statistic=np.mean)
    ), "seed does not change the bootstrap resampling"
    # A non-default statistic must be honoured.
    boot_mean = paired_bootstrap(spread, seed=7, resamples=500, statistic=np.mean)
    assert math.isclose(boot_mean["point"], float(np.mean(spread))), boot_mean
    # A statistic with no `axis` keyword must take the loop path and agree.
    def no_axis_median(row: npt.ArrayLike) -> float:
        return float(np.median(np.asarray(row, dtype=float)))

    boot_loop = paired_bootstrap(spread, seed=7, resamples=500,
                                 statistic=no_axis_median)
    boot_vec = paired_bootstrap(spread, seed=7, resamples=500, statistic=np.median)
    assert math.isclose(boot_loop["ci_low"], boot_vec["ci_low"]), (boot_loop, boot_vec)
    assert math.isclose(boot_loop["ci_high"], boot_vec["ci_high"]), (boot_loop, boot_vec)

    # ---- 3. paired_bootstrap_difference -------------------------------
    arm_a = np.array([0.90, 0.92, 0.88, 0.94, 0.91, 0.93, 0.89, 0.95])
    arm_b = arm_a - 0.05                                   # constant offset
    diff = paired_bootstrap_difference(arm_a, arm_b, seed=3, resamples=1000)
    assert math.isclose(diff["point"], 0.05, abs_tol=1e-12), diff
    # A constant paired difference has a zero-width paired interval, even though
    # each arm on its own is spread out.  This is the pairing doing its job.
    assert math.isclose(diff["ci_low"], 0.05, abs_tol=1e-12), diff
    assert math.isclose(diff["ci_high"], 0.05, abs_tol=1e-12), diff
    assert diff["method"] == "paired_percentile_bootstrap_over_carrier_draws"
    assert diff == paired_bootstrap_difference(arm_a, arm_b, seed=3, resamples=1000)
    # Independent (unpaired) bootstrap of the same arms would NOT be zero-width.
    unpaired_width = (
        paired_bootstrap(arm_a, seed=3, resamples=1000)["ci_high"]
        - paired_bootstrap(arm_a, seed=3, resamples=1000)["ci_low"]
    )
    assert unpaired_width > 0.0, "spread arm should have a non-degenerate interval"
    # a - b is the sign convention.
    flipped = paired_bootstrap_difference(arm_b, arm_a, seed=3, resamples=1000)
    assert math.isclose(flipped["point"], -0.05, abs_tol=1e-12), flipped
    try:
        paired_bootstrap_difference([1.0, 2.0], [1.0], seed=0, resamples=10)
    except ValueError:
        pass
    else:
        raise AssertionError("paired_bootstrap_difference accepted ragged arms")

    # ---- 4. paired_permutation_test -----------------------------------
    identical = paired_permutation_test(arm_a, arm_a, seed=11, resamples=200)
    # Zero differences: every sign flip reproduces the observed 0, so |perm| >=
    # |observed| always holds and p saturates at exactly 1.0.
    assert identical["observed"] == 0.0, identical
    assert identical["p_value"] == 1.0, identical
    assert identical["resamples"] == 200

    perm = paired_permutation_test(arm_a, arm_b, seed=11, resamples=999)
    assert math.isclose(perm["observed"], 0.05, abs_tol=1e-12), perm
    floor = 1.0 / (1.0 + 999)
    assert floor <= perm["p_value"] <= 1.0, perm
    # Two-sided: swapping the arms negates the observed and leaves p alone.
    swapped = paired_permutation_test(arm_b, arm_a, seed=11, resamples=999)
    assert math.isclose(swapped["observed"], -perm["observed"], abs_tol=1e-12)
    assert math.isclose(swapped["p_value"], perm["p_value"], rel_tol=1e-12)
    assert perm == paired_permutation_test(arm_a, arm_b, seed=11, resamples=999)

    # A unanimous constant offset over the 16 confirmatory carrier draws is the
    # most extreme configuration sign flips can reach: all 16 signs must agree,
    # probability 2 / 2**16 = 3.05e-5, so the Monte-Carlo p lands on its floor.
    sixteen_a = np.linspace(0.86, 0.94, 16)
    sixteen_b = sixteen_a - 0.05
    clean = paired_permutation_test(sixteen_a, sixteen_b, seed=2, resamples=999)
    assert math.isclose(clean["observed"], 0.05, abs_tol=1e-12), clean
    assert clean["p_value"] <= 0.01, clean

    # Why the default statistic is the mean and not the median.  Under sign
    # flips the median of the differences takes only the values {-d, 0, +d}, so
    # it cannot reject an effect every carrier draw shows.  Exact enumeration
    # over eight draws: median p = 0.7266, mean p = 0.0078.  The Monte-Carlo
    # values below must reproduce that gap.
    degenerate = paired_permutation_test(
        arm_a, arm_b, seed=2, resamples=4000, statistic=np.median
    )
    assert degenerate["p_value"] > 0.5, degenerate
    assert perm["p_value"] < 0.1, perm
    assert perm["p_value"] < degenerate["p_value"], (perm, degenerate)

    # Null-ish data must not produce a small p.
    noise_a = np.array([0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.47, 0.50])
    noise_b = np.array([0.51, 0.49, 0.50, 0.48, 0.52, 0.47, 0.51, 0.50])
    null_perm = paired_permutation_test(noise_a, noise_b, seed=5, resamples=2000)
    assert null_perm["p_value"] > 0.2, null_perm

    # ---- 5. holm_correction -------------------------------------------
    # Textbook m = 3 example.  Sorted: 0.01, 0.03, 0.04.
    #   3*0.01 = 0.03 ; 2*0.03 = 0.06 ; 1*0.04 = 0.04
    # Running maximum enforces monotonicity: 0.03, 0.06, 0.06.
    holm = holm_correction({"A": 0.01, "B": 0.04, "C": 0.03})
    assert set(holm) == {"A", "B", "C"}, holm
    assert list(holm) == ["A", "B", "C"], "input key order not preserved"
    assert math.isclose(holm["A"]["p_raw"], 0.01)
    assert math.isclose(holm["A"]["p_adjusted"], 0.03), holm["A"]
    assert holm["A"]["rank"] == 1, holm["A"]
    assert math.isclose(holm["C"]["p_adjusted"], 0.06), holm["C"]
    assert holm["C"]["rank"] == 2, holm["C"]
    # Monotonicity: B's raw 0.04 * 1 = 0.04 is pulled UP to C's 0.06.
    assert math.isclose(holm["B"]["p_adjusted"], 0.06), holm["B"]
    assert holm["B"]["rank"] == 3, holm["B"]
    ordered = sorted(holm.values(), key=lambda entry: entry["rank"])
    adjusted = [entry["p_adjusted"] for entry in ordered]
    assert adjusted == sorted(adjusted), adjusted
    # Cap at 1.0.
    capped = holm_correction({"X": 0.6, "Y": 0.7, "Z": 0.8})
    assert all(entry["p_adjusted"] == 1.0 for entry in capped.values()), capped
    # Single test is unchanged.
    single = holm_correction({"only": 0.02})
    assert math.isclose(single["only"]["p_adjusted"], 0.02), single
    assert single["only"]["rank"] == 1
    assert holm_correction({}) == {}
    for bad_family in ({"A": float("nan")}, {"A": 1.5}, {"A": -0.1}):
        try:
            holm_correction(bad_family)
        except ValueError:
            pass
        else:
            raise AssertionError("holm_correction accepted an invalid p-value")

    # ---- 6. spearman ---------------------------------------------------
    assert math.isclose(spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), 1.0)
    assert math.isclose(spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]), -1.0)
    # Monotone but non-linear: Spearman is 1 where Pearson is not.
    assert math.isclose(spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0)
    # Known tied pair.  Ranks of y = [5, 6, 7, 8, 7] are [1, 2, 3.5, 5, 3.5].
    x_known = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_known = [5.0, 6.0, 7.0, 8.0, 7.0]
    rho = spearman(x_known, y_known)
    assert math.isclose(rho, 0.8207826816681233, rel_tol=1e-12), rho
    assert math.isnan(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    try:
        spearman([1.0, 2.0], [1.0, 2.0, 3.0])
    except ValueError:
        pass
    else:
        raise AssertionError("spearman accepted ragged inputs")

    scipy_checked = False
    try:
        from scipy import stats as _scipy_stats     # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        for x_case, y_case in (
            (x_known, y_known),
            ([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
             [0.61, 0.55, 0.68, 0.57, 0.72, 0.88, 0.58, 0.75]),
            ([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0],
             [9.0, 8.0, 8.0, 5.0, 5.0, 2.0, 1.0]),
        ):
            reference = float(_scipy_stats.spearmanr(x_case, y_case).statistic)
            mine = spearman(x_case, y_case)
            assert math.isclose(mine, reference, rel_tol=1e-12, abs_tol=1e-12), (
                mine, reference, x_case, y_case
            )
        scipy_checked = True

    # ---- 7. ordered_pair_agreement -------------------------------------
    # Hand-built 3-point example, relative_gap = 0.05.
    #   J   = [1.0, 2.0, 4.0]        acc = [0.80, 0.70, 0.80]
    #   (0,1): |1-2|/2 = 0.500 > 0.05 -> considered.
    #          sign(J) = -1, sign(acc) = +1                  -> disagree
    #   (0,2): |1-4|/4 = 0.750 > 0.05 -> considered.
    #          sign(J) = -1, sign(acc) =  0 (tie)            -> disagree
    #   (1,2): |2-4|/4 = 0.500 > 0.05 -> considered.
    #          sign(J) = -1, sign(acc) = -1                  -> agree
    #   => 3 considered, 1 agreeing, fraction = 1/3.
    hand = ordered_pair_agreement([1.0, 2.0, 4.0], [0.80, 0.70, 0.80])
    assert hand["n_pairs_considered"] == 3, hand
    assert hand["n_agreeing"] == 1, hand
    assert math.isclose(hand["fraction"], 1.0 / 3.0), hand

    # Perfectly concordant, all gaps wide.
    concordant = ordered_pair_agreement([1.0, 2.0, 4.0], [0.60, 0.75, 0.90])
    assert concordant == {
        "n_pairs_considered": 3, "n_agreeing": 3, "fraction": 1.0
    }, concordant

    # Gap screening: |1 - 1.02| / 1.02 = 0.0196 < 0.05, so that pair is dropped.
    screened = ordered_pair_agreement([1.0, 1.02, 2.0], [0.90, 0.60, 0.95])
    assert screened["n_pairs_considered"] == 2, screened
    assert screened["n_agreeing"] == 2, screened
    assert math.isclose(screened["fraction"], 1.0), screened

    # All gaps below threshold -> nothing considered, fraction is nan, never 1.0.
    empty = ordered_pair_agreement([1.0, 1.01], [0.5, 0.9])
    assert empty["n_pairs_considered"] == 0, empty
    assert empty["n_agreeing"] == 0, empty
    assert math.isnan(empty["fraction"]), empty
    assert evaluate_gate("empty", empty["fraction"], ">=", 0.90)["pass"] is False

    # Zero-J guard: max(J_i, J_k) == 0 has no relative gap and is skipped.
    zero_guard = ordered_pair_agreement([0.0, 0.0, 3.0], [0.5, 0.9, 0.9])
    assert zero_guard["n_pairs_considered"] == 2, zero_guard

    # relative_gap = 0 keeps every strictly separated pair.
    wide_open = ordered_pair_agreement([1.0, 1.01], [0.5, 0.9], relative_gap=0.0)
    assert wide_open["n_pairs_considered"] == 1, wide_open
    assert wide_open["n_agreeing"] == 1, wide_open

    # ---- 8. bootstrap_lower_bound --------------------------------------
    assert math.isclose(bootstrap_lower_bound(constant, seed=1, resamples=500), 0.91)
    lower = bootstrap_lower_bound(spread, seed=4, resamples=4000)
    two_sided = paired_bootstrap(spread, seed=4, resamples=4000)
    # The one-sided 95% bound is the 5th percentile; the two-sided interval's
    # low end is the 2.5th.  The one-sided bound is therefore never below it.
    assert lower >= two_sided["ci_low"] - 1e-12, (lower, two_sided)
    assert lower <= two_sided["point"] + 1e-12, (lower, two_sided)
    assert lower == bootstrap_lower_bound(spread, seed=4, resamples=4000)
    # Level monotonicity: a stricter level gives a weaker (lower) bound.
    strict = bootstrap_lower_bound(spread, seed=4, resamples=4000, level=0.99)
    loose = bootstrap_lower_bound(spread, seed=4, resamples=4000, level=0.80)
    assert strict <= lower <= loose + 1e-12, (strict, lower, loose)
    for bad_level in (0.0, 1.0, 1.5):
        try:
            bootstrap_lower_bound(spread, seed=0, resamples=10, level=bad_level)
        except ValueError:
            pass
        else:
            raise AssertionError("bootstrap_lower_bound accepted an invalid level")

    # ---- 9. summarize ---------------------------------------------------
    summary = summarize([1.0, 2.0, 3.0, 4.0])
    assert summary["n"] == 4, summary
    assert math.isclose(summary["median"], 2.5), summary
    assert math.isclose(summary["q1"], 1.75), summary
    assert math.isclose(summary["q3"], 3.25), summary
    assert math.isclose(summary["iqr"], 1.50), summary
    assert math.isclose(summary["min"], 1.0), summary
    assert math.isclose(summary["max"], 4.0), summary
    assert math.isclose(summary["mean"], 2.5), summary
    assert math.isclose(summary["sign_consistency"], 1.0), summary
    # Strictly positive: a zero counts against sign consistency.
    mixed = summarize([-1.0, 0.0, 3.0, 4.0])
    assert math.isclose(mixed["sign_consistency"], 0.5), mixed
    assert math.isclose(summarize([-2.0, -1.0])["sign_consistency"], 0.0)
    try:
        summarize([])
    except ValueError:
        pass
    else:
        raise AssertionError("summarize accepted an empty array")

    # ---- 10. evaluate_gate ----------------------------------------------
    # Protocol 10.1 condition 1: median R_J >= 0.90.
    gate = evaluate_gate("alignment_median_R_J", 0.93, ">=", 0.90)
    assert gate == {
        "name": "alignment_median_R_J",
        "observed": 0.93,
        "comparator": ">=",
        "threshold": 0.90,
        "pass": True,
    }, gate
    assert evaluate_gate("g", 0.89, ">=", 0.90)["pass"] is False
    # Boundary: >= passes on equality, > does not.
    assert evaluate_gate("g", 0.90, ">=", 0.90)["pass"] is True
    assert evaluate_gate("g", 0.90, ">", 0.90)["pass"] is False
    assert evaluate_gate("g", 0.90, "<=", 0.90)["pass"] is True
    assert evaluate_gate("g", 0.90, "<", 0.90)["pass"] is False
    # Protocol 10.1 condition 5: calibration MAE <= 0.02.
    assert evaluate_gate("calibration_mae", 0.014, "<=", 0.02)["pass"] is True
    assert evaluate_gate("calibration_mae", 0.031, "<=", 0.02)["pass"] is False
    # NaN and infinities are never a pass, and they are labelled.
    for bad_observed in (float("nan"), float("inf"), float("-inf")):
        nan_gate = evaluate_gate("nan_gate", bad_observed, ">=", 0.90)
        assert nan_gate["pass"] is False, nan_gate
        assert nan_gate["nan_observed"] is True, nan_gate
    assert "nan_observed" not in gate
    try:
        evaluate_gate("bad", 1.0, "=>", 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("evaluate_gate accepted an unknown comparator")

    # ---- End-to-end shape check on a 16-draw / 5-seed confirmatory layout --
    rng = np.random.default_rng(20260918)
    synthetic = 0.93 + 0.01 * rng.standard_normal((16, 5))
    r_j = median_within_carrier(synthetic)
    assert r_j.shape == (16,)
    gates = [
        evaluate_gate("median_R_J", float(np.median(r_j)), ">=", 0.90),
        evaluate_gate(
            "bootstrap_lower_bound_R_J",
            bootstrap_lower_bound(r_j, seed=1, resamples=1000),
            ">=",
            0.80,
        ),
        evaluate_gate(
            "draws_at_or_above_0.80",
            float(np.count_nonzero(r_j >= 0.80)),
            ">=",
            14.0,
        ),
    ]
    assert all(entry["pass"] for entry in gates), gates
    assert summarize(r_j)["n"] == 16

    if not scipy_checked:
        print("note: scipy is not importable; spearman cross-check was skipped")
    print("nc_stats self-test OK")


if __name__ == "__main__":
    _self_test()
