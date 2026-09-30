"""Work-efficient parallel scan for the write-store recurrence, and a batched
multi-model trainer.  Both exist to use the GPU properly rather than to run a
sequential Python loop on it.

The closure recursion is a first-order linear recurrence:

    lower[j+1] = lower[j] W + U^j K ,      lower[0] = 0

Each step is an affine map of the running state, so encode step j as the pair
`(A_j, C_j) = (W, U^j K)` acting on the right, `X -> X A_j + C_j`.  Composition
"apply x first, then y" is

    (x.A, x.C) . (y.A, y.C) = (x.A y.A, x.C y.A + y.C)

which is associative with identity `(I, 0)`.  A Blelloch exclusive scan over the
sequence therefore returns every `lower[j]` at once, in O(n) work and O(log n)
depth, instead of n sequential steps.

The canonical sequential implementation in the frozen release stays the
reference.  `parity_check` asserts agreement; nothing here replaces it.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# numpy backend
# --------------------------------------------------------------------------
def _compose(a_left: np.ndarray, c_left: np.ndarray,
             a_right: np.ndarray, c_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched monoid product.  `left` is applied first."""
    return a_left @ a_right, c_left @ a_right + c_right


def blelloch_transfers(W: np.ndarray, K: np.ndarray, U: np.ndarray,
                       steps: int) -> tuple[np.ndarray, np.ndarray]:
    """All closure transfers and the closure covariance, by parallel scan.

    Returns `(transfers, covariance)` with `transfers` of shape
    `(steps, d, N)`, matching `closure_transfers` element for element.
    """
    n_write = W.shape[0]
    n_store = U.shape[0]
    size = 1
    while size < steps:
        size *= 2

    # Leaves.  Padding beyond `steps` is the identity, which the scan ignores.
    A = np.broadcast_to(np.eye(n_write), (size, n_write, n_write)).copy()
    C = np.zeros((size, n_store, n_write))
    A[:steps] = W
    power = np.eye(n_store)
    store_powers = np.empty((steps, n_store, n_store))
    for j in range(steps):                       # U^j; cheap, d = 16
        store_powers[j] = power
        power = power @ U
    C[:steps] = store_powers @ K

    # Up-sweep: reduce in place over a binary tree.
    stride = 1
    while stride < size:
        left = np.arange(stride - 1, size, 2 * stride)
        right = left + stride
        A[right], C[right] = _compose(A[left], C[left], A[right], C[right])
        stride *= 2

    # Down-sweep: turn the reduction tree into an exclusive scan.
    A[size - 1] = np.eye(n_write)
    C[size - 1] = 0.0
    stride = size // 2
    while stride >= 1:
        left = np.arange(stride - 1, size, 2 * stride)
        right = left + stride
        a_left, c_left = A[left].copy(), C[left].copy()
        A[left], C[left] = A[right], C[right]
        # The parent slot holds the prefix of everything BEFORE this subtree, so
        # it is applied FIRST and the left subtree's reduction second.  With a
        # commutative operator the order would not show; matrix composition is
        # not commutative, and reversing it silently produces wrong transfers.
        A[right], C[right] = _compose(A[right], C[right], a_left, c_left)
        stride //= 2

    transfers = C[:steps]
    covariance = np.einsum("jab,jcb->ac", transfers, transfers)
    return transfers, covariance


def parity_check(W: np.ndarray, K: np.ndarray, U: np.ndarray, steps: int,
                 reference) -> dict[str, Any]:
    """Assert the scan reproduces the canonical sequential implementation."""
    ref_transfers, ref_cov = reference(W, K, U, steps)
    scan_transfers, scan_cov = blelloch_transfers(W, K, U, steps)
    ref_stack = np.stack(ref_transfers)
    scale = max(float(np.max(np.abs(ref_stack))), 1e-300)
    transfer_error = float(np.max(np.abs(ref_stack - scan_transfers))) / scale
    cov_scale = max(float(np.max(np.abs(ref_cov))), 1e-300)
    covariance_error = float(np.max(np.abs(ref_cov - scan_cov))) / cov_scale
    return {
        "steps": steps,
        "max_relative_transfer_error": transfer_error,
        "max_relative_covariance_error": covariance_error,
        "tolerance": 1e-11,
        "pass": bool(transfer_error <= 1e-11 and covariance_error <= 1e-11),
    }


# --------------------------------------------------------------------------
# torch backend: the same scan, plus all models trained at once
# --------------------------------------------------------------------------
def torch_blelloch_transfers(W, K, U, steps: int, device: str = "cuda"):
    import torch
    n_write, n_store = W.shape[0], U.shape[0]
    size = 1
    while size < steps:
        size *= 2
    Wt = torch.as_tensor(W, dtype=torch.float64, device=device)
    Kt = torch.as_tensor(K, dtype=torch.float64, device=device)
    Ut = torch.as_tensor(U, dtype=torch.float64, device=device)
    A = torch.eye(n_write, dtype=torch.float64, device=device).expand(size, -1, -1).clone()
    C = torch.zeros(size, n_store, n_write, dtype=torch.float64, device=device)
    A[:steps] = Wt
    powers = torch.empty(steps, n_store, n_store, dtype=torch.float64, device=device)
    power = torch.eye(n_store, dtype=torch.float64, device=device)
    for j in range(steps):
        powers[j] = power
        power = power @ Ut
    C[:steps] = powers @ Kt

    stride = 1
    while stride < size:
        left = torch.arange(stride - 1, size, 2 * stride, device=device)
        right = left + stride
        a_l, c_l = A[left], C[left]
        A[right], C[right] = a_l @ A[right], c_l @ A[right] + C[right]
        stride *= 2
    A[size - 1] = torch.eye(n_write, dtype=torch.float64, device=device)
    C[size - 1] = 0.0
    stride = size // 2
    while stride >= 1:
        left = torch.arange(stride - 1, size, 2 * stride, device=device)
        right = left + stride
        a_l, c_l = A[left].clone(), C[left].clone()
        parent_a, parent_c = A[right].clone(), C[right].clone()
        A[left], C[left] = parent_a, parent_c
        # Parent prefix first, left-subtree reduction second (see the numpy path).
        A[right], C[right] = parent_a @ a_l, parent_c @ a_l + c_l
        stride //= 2
    transfers = C[:steps]
    covariance = torch.einsum("jab,jcb->ac", transfers, transfers)
    return transfers, covariance


def train_batched(P_stack: np.ndarray, R_stack: np.ndarray, amplitude: float,
                  seeds: list[int], steps: int, batch_size: int,
                  learning_rate: float = 3e-3, grad_clip: float = 1.0,
                  device: str = "cuda", dtype: str = "float64") -> dict[str, Any]:
    """Train every model at once as one batched tensor program.

    The models are independent, so stacking them changes no gradient: the loss
    of model m depends only on model m's parameters, and Adam is elementwise.
    What changes is that the 12,000 optimizer steps are taken once for all
    models instead of once per model, which is the only way a GPU is useful
    here at 512x32 tensor sizes.

    Per-model noise streams are drawn from each model's own seed, so the batched
    run is the same experiment as the serial one, not a cheaper approximation.
    """
    import torch

    models = len(seeds)
    store_dim, write_dim = P_stack.shape[1], P_stack.shape[2]
    dev = torch.device(device)
    tdtype = getattr(torch, dtype)
    P = torch.as_tensor(P_stack, dtype=tdtype, device=dev)
    R = torch.as_tensor(R_stack, dtype=tdtype, device=dev)

    generators = [np.random.default_rng(s) for s in seeds]
    u0 = np.stack([g.standard_normal(write_dim) for g in generators])
    u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
    u = torch.tensor(u0, dtype=tdtype, device=dev, requires_grad=True)
    w = torch.zeros(models, store_dim, dtype=tdtype, device=dev, requires_grad=True)
    b = torch.zeros(models, dtype=tdtype, device=dev, requires_grad=True)
    opt = torch.optim.Adam([u, w, b], lr=learning_rate, weight_decay=0.0)

    clip_events = 0
    for _ in range(steps):
        labels = np.stack([g.integers(0, 2, size=batch_size).astype(np.float64) * 2.0 - 1.0
                           for g in generators])
        eps = np.stack([g.standard_normal((batch_size, store_dim)) for g in generators])
        y = torch.as_tensor(labels, dtype=tdtype, device=dev)
        e = torch.as_tensor(eps, dtype=tdtype, device=dev)
        v = u / u.norm(dim=1, keepdim=True)
        signal = amplitude * torch.einsum("mij,mj->mi", P, v)
        x = y[:, :, None] * signal[:, None, :] + torch.einsum("mbj,mij->mbi", e, R)
        logit = torch.einsum("mbi,mi->mb", x, w) + b[:, None]
        # Per-model mean loss, summed over models: model m's gradient is exactly
        # the gradient it would have had on its own.
        loss = torch.nn.functional.softplus(-(y * logit)).mean(dim=1).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Per-model global-norm clipping, matching the serial implementation.
        with torch.no_grad():
            flat = torch.cat([u.grad, w.grad, b.grad[:, None]], dim=1)
            norms = flat.norm(dim=1)
            scale = torch.clamp(grad_clip / norms.clamp_min(1e-300), max=1.0)
            clip_events += int((norms > grad_clip).sum().item())
            u.grad.mul_(scale[:, None])
            w.grad.mul_(scale[:, None])
            b.grad.mul_(scale)
        opt.step()

    v_final = (u / u.norm(dim=1, keepdim=True)).detach().cpu().numpy()
    return {
        "v": v_final,
        "w": w.detach().cpu().numpy(),
        "b": b.detach().cpu().numpy(),
        "models": models,
        "steps": steps,
        "clip_events_total": clip_events,
        "device": str(dev),
        "dtype": dtype,
    }
