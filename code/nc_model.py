"""Endpoint sampler, sequential-unroll parity control, trainer, and readouts.

Authority: ../authority/NEURAL_COMPUTATION_TRAINED_MEMORY_VALIDATION_PROTOCOL_v1_20260918_EN.md
           ../authority/NEURAL_COMPUTATION_TRAINED_MEMORY_VALIDATION_PREREG_v1_20260918.yaml

The system is linear with additive Gaussian writer noise, so the store state at
the query horizon is exactly Gaussian:

    x_s = y * a * (P v) + R eps,     R R^T = Sigma,     eps ~ N(0, I_d).

Handoff item B requires the exact endpoint sampler to be used wherever it is
mathematically equivalent, with a sequential unroll preserved as an independent
parity control.  Both live here.

Optimizer leakage control (validity control 8): the trainer receives only
`amplitude`, the sensitivity map `P`, the noise factor `R`, and the labels.  It
never receives Sigma, Sigma^{-1}, any Fisher operator, any eigenvector, or any
oracle label.  `P` and `R` are the forward simulator, not the answer: `R` is a
Cholesky factor of the covariance and the trainer performs no solve with it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Forward samplers
# --------------------------------------------------------------------------
def noise_factor(covariance: np.ndarray) -> np.ndarray:
    """Lower-triangular R with R R^T = covariance, with an eigen fallback."""
    symmetric = 0.5 * (covariance + covariance.T)
    try:
        return np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(symmetric)
        values = np.maximum(values, 0.0)
        return vectors @ np.diag(np.sqrt(values))


def sample_endpoint(
    rng: np.random.Generator,
    P: np.ndarray,
    R: np.ndarray,
    v: np.ndarray,
    amplitude: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact endpoint sampler.  Returns (x_s [count, d], y [count] in {-1, +1})."""
    labels = rng.integers(0, 2, size=count).astype(np.float64) * 2.0 - 1.0
    eps = rng.standard_normal((count, R.shape[1]))
    signal = amplitude * (P @ v)
    return labels[:, None] * signal[None, :] + eps @ R.T, labels


def sample_shared_noise(
    rng: np.random.Generator, dim: int, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the labels and standard noise ONCE, to be shared across directions.

    Protocol section 4.8: "For each direction, use the same test noise and class
    labels within a carrier draw."  Validity control 7.
    """
    labels = rng.integers(0, 2, size=count).astype(np.float64) * 2.0 - 1.0
    eps = rng.standard_normal((count, dim))
    return eps, labels


def endpoint_from_shared(
    eps: np.ndarray, labels: np.ndarray, P: np.ndarray, R: np.ndarray,
    v: np.ndarray, amplitude: float
) -> np.ndarray:
    signal = amplitude * (P @ v)
    return labels[:, None] * signal[None, :] + eps @ R.T


def sequential_unroll(
    rng: np.random.Generator,
    W: np.ndarray,
    K: np.ndarray,
    U: np.ndarray,
    v: np.ndarray,
    amplitude: float,
    horizon: int,
    write_window: int,
    isolated: bool,
    count: int,
    target_time: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent parity control: literal step-by-step recurrence.

        x_w[t+1] = W x_w[t] + a y v 1[t == t_0] + z_t,   z_t ~ N(0, I_N)
        x_s[t+1] = K_t x_w[t] + U x_s[t]
        K_t = K for t < T_w;  isolated => K_t = 0 for t >= T_w, else K_t = K.

    Deliberately slow and literal.  Used only to certify the fast paths.
    """
    n = W.shape[0]
    d = U.shape[0]
    labels = rng.integers(0, 2, size=count).astype(np.float64) * 2.0 - 1.0
    xw = np.zeros((count, n))
    xs = np.zeros((count, d))
    for t in range(horizon):
        coupled = (t < write_window) or (not isolated)
        xs_next = xs @ U.T
        if coupled:
            xs_next = xs_next + xw @ K.T
        xw_next = xw @ W.T + rng.standard_normal((count, n))
        if t == target_time:
            xw_next = xw_next + amplitude * labels[:, None] * v[None, :]
        xw, xs = xw_next, xs_next
    return xs, labels


# --------------------------------------------------------------------------
# Readouts
# --------------------------------------------------------------------------
def fit_lda(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Covariance-aware recalibrated linear discriminant, fit from DATA only.

    Equal priors, pooled within-class covariance.  For equal-covariance
    Gaussians this is the Bayes-optimal linear rule, and it is estimated from
    the calibration sample rather than handed the analytic covariance, so no
    oracle quantity enters the readout path.
    """
    pos = x[y > 0]
    neg = x[y < 0]
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError("calibration sample must contain both classes")
    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    centered = np.concatenate([pos - mu_pos, neg - mu_neg], axis=0)
    pooled = centered.T @ centered / (len(centered) - 2)
    delta = mu_pos - mu_neg
    w = np.linalg.solve(pooled, delta)
    b = -float(w @ (mu_pos + mu_neg)) / 2.0
    return w, b


def accuracy(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """Sign agreement.  A logit of exactly zero counts as an error."""
    logit = x @ w + b
    predicted = np.sign(logit)
    return float(np.mean(predicted == np.sign(y)))


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------
def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class AdamState:
    """Adam exactly as preregistered: lr 3e-3, weight decay 0, global clip 1.0."""

    def __init__(self, shapes: list[tuple[int, ...]], lr: float, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros(shape) for shape in shapes]
        self.vv = [np.zeros(shape) for shape in shapes]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.vv[i] = self.beta2 * self.vv[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.vv[i] / (1.0 - self.beta2 ** self.t)
            out.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return out


def global_clip(grads: list[np.ndarray], max_norm: float) -> tuple[list[np.ndarray], float]:
    total = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    if total > max_norm and total > 0.0:
        scale = max_norm / total
        return [g * scale for g in grads], total
    return grads, total


def train_direction(
    P: np.ndarray,
    R: np.ndarray,
    amplitude: float,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float = 3e-3,
    grad_clip: float = 1.0,
    loss_every: int = 1,
    validation: dict[str, np.ndarray] | None = None,
    validate_every: int = 0,
) -> dict[str, Any]:
    """Train the unit write direction v = u/||u|| and the logistic readout (w, b).

    Analytic gradients of  mean softplus(-y * (x.w + b))  with
    x = y a P v + R eps:

        dL/dlogit_n = -y_n sigmoid(-y_n logit_n) / B
        dL/dw       = X^T dL/dlogit
        dL/db       = sum dL/dlogit
        dL/dv       = a (sum_n dL/dlogit_n y_n) P^T w
        dL/du       = (I - v v^T) / ||u|| @ dL/dv

    The optimizer path touches only P, R, a and the labels.  Nothing derived
    from Sigma^{-1}, any Fisher operator, or any eigenvector is visible here.
    """
    write_dim = P.shape[1]
    store_dim = P.shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(write_dim)
    u = u / np.linalg.norm(u)
    w = np.zeros(store_dim)
    b = 0.0

    optimizer = AdamState([u.shape, w.shape, ()], lr=learning_rate)
    losses: list[float] = []
    clip_events = 0
    grad_norms: list[float] = []
    validation_history: list[dict[str, float]] = []
    best = None

    for step in range(steps):
        v = u / np.linalg.norm(u)
        labels = rng.integers(0, 2, size=batch_size).astype(np.float64) * 2.0 - 1.0
        eps = rng.standard_normal((batch_size, store_dim))
        signal = amplitude * (P @ v)
        x = labels[:, None] * signal[None, :] + eps @ R.T
        logit = x @ w + b
        margin = labels * logit
        loss = float(np.mean(_softplus(-margin)))
        g_logit = -(labels * _sigmoid(-margin)) / batch_size
        grad_w = x.T @ g_logit
        grad_b = float(np.sum(g_logit))
        grad_v = amplitude * float(np.sum(g_logit * labels)) * (P.T @ w)
        norm_u = float(np.linalg.norm(u))
        grad_u = (grad_v - v * float(v @ grad_v)) / norm_u

        grads, raw_norm = global_clip([grad_u, grad_w, np.array(grad_b)], grad_clip)
        if raw_norm > grad_clip:
            clip_events += 1
        if loss_every and step % loss_every == 0:
            grad_norms.append(raw_norm)
        u, w, b_array = optimizer.step([u, w, np.array(b)], grads)
        b = float(b_array)

        if loss_every and step % loss_every == 0:
            losses.append(loss)
        if validate_every and validation is not None and (step + 1) % validate_every == 0:
            v_now = u / np.linalg.norm(u)
            x_val = endpoint_from_shared(
                validation["eps"], validation["labels"], P, R, v_now, amplitude
            )
            acc = accuracy(x_val, validation["labels"], w, b)
            validation_history.append({"step": step + 1, "accuracy": acc})
            if best is None or acc > best["accuracy"]:
                best = {"step": step + 1, "accuracy": acc,
                        "v": v_now.copy(), "w": w.copy(), "b": b}

    v_final = u / np.linalg.norm(u)
    return {
        "v": v_final,
        "w": w,
        "b": b,
        "u_norm": float(np.linalg.norm(u)),
        "loss_curve": losses,
        "final_loss": losses[-1] if losses else float("nan"),
        "clip_events": clip_events,
        "clip_fraction": clip_events / max(steps, 1),
        "grad_norm_samples": grad_norms,
        "grad_norm_median": float(np.median(grad_norms)) if grad_norms else float("nan"),
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
        "grad_norm_final": grad_norms[-1] if grad_norms else float("nan"),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": int(seed),
        "validation_history": validation_history,
        "best_validation": best,
    }
