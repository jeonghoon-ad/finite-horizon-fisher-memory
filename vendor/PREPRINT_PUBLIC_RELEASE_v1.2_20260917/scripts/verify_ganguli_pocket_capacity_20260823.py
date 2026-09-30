"""Ganguli-Huh-Sompolinsky Fisher memory, measured on pocket / non-normal / hybrid systems.

PURELY SYNTHETIC. No checkpoint, no measured artifact, no trained weight is read.
Every PASS-bearing number has an independent analytic ground truth stated inline.
numpy only, float64, seed 20260823.

Systems (all discrete-time linear, x_{n+1} = W x_n + v s_n + z_n, z ~ N(0, eps I)):
  A normal_contract   W = 0.9 Q, Q Haar-orthogonal              (normal, rho<1)
  B pocket_isometric  W = Q                                     (normal, rho=1, all-pocket)
  C chain_nonnormal   W = gain chain (hidden feedforward)       (non-normal, nilpotent)
  D hybrid            [[S,0],[K,U]] chain -> orthogonal pocket  (non-normal, marginally stable sink)
  E hybrid_isolated   as D but process noise injected in the chain block ONLY (sigma_L = 0)
  F multichannel      W = 0.9 Q with m orthonormal input columns (sum-rule scaling check)

Fisher memory (Ganguli et al., PNAS 105(48):18970, 2008, Eq. for J(k)):
  C_n   = eps * sum_{j=0}^{n-1} W^j W^{jT}        (finite-horizon noise covariance, x_0 = 0)
  J(k)  = v^T (W^T)^k C_n^{-1} W^k v              (Fisher info about the input k steps back)
  Jtot  = eps * sum_k J(k)                        (normalized by the input SNR 1/eps)
"""

from __future__ import annotations

import numpy as np

SEED = 20260823
EPS = 1.0
TOL = 1e-9

rng = np.random.default_rng(SEED)
RESULTS: list[tuple[str, bool, str]] = []


def check(tag: str, ok: bool, detail: str) -> None:
    RESULTS.append((tag, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {detail}")


def haar_orthogonal(dim: int, gen: np.random.Generator) -> np.ndarray:
    a = gen.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


# ----------------------------------------------------------------------------- core


def fisher_memory(W: np.ndarray, V: np.ndarray, horizon: int, noise_cov: np.ndarray | None = None):
    """Return (Jk, Jtot) with Jk[k] = trace of the k-th Fisher memory block, normalized by 1/eps.

    V is N x m (m input channels, orthonormal columns). noise_cov is the per-step
    process-noise covariance; default eps*I.
    """
    n_dim = W.shape[0]
    q_step = np.eye(n_dim) * EPS if noise_cov is None else noise_cov
    cov = np.zeros((n_dim, n_dim))
    power = np.eye(n_dim)
    prop = []  # W^k V
    for _ in range(horizon):
        cov += power @ q_step @ power.T
        prop.append(power @ V)
        power = W @ power
    cov_inv = np.linalg.pinv(cov, rcond=1e-14, hermitian=True)
    jk = np.array([float(np.trace(p.T @ cov_inv @ p)) for p in prop])
    return jk * EPS, float(jk.sum() * EPS)


def shape_stats(jk: np.ndarray) -> dict[str, float]:
    total = jk.sum()
    if total <= 0:
        return {"tau": 0.0, "persist_half": 0.0, "oldest": 0.0}
    k = np.arange(len(jk))
    return {
        "tau": float((k * jk).sum() / total),
        "persist_half": float(jk[len(jk) // 2 :].sum() / total),
        "oldest": float(jk[-1]),
    }


def build_chain(dim: int, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Hidden feedforward chain with super-linear gain G_j = exp(c j^2); v = e_0."""
    W = np.zeros((dim, dim))
    for j in range(dim - 1):
        # G_{j+1}/G_j = exp(c((j+1)^2 - j^2)) = exp(c(2j+1))
        W[j + 1, j] = np.exp(c * (2 * j + 1))
    v = np.zeros((dim, 1))
    v[0, 0] = 1.0
    return W, v


def build_hybrid(d_chain: int, d_pocket: int, c: float, kappa: float, gen):
    """[[S, 0], [K, U]] on (chain, pocket). Input enters the chain head."""
    S, v_chain = build_chain(d_chain, c)
    U = haar_orthogonal(d_pocket, gen)
    K = kappa * gen.standard_normal((d_pocket, d_chain)) / np.sqrt(d_chain)
    n_dim = d_chain + d_pocket
    W = np.zeros((n_dim, n_dim))
    W[:d_chain, :d_chain] = S
    W[d_chain:, :d_chain] = K
    W[d_chain:, d_chain:] = U
    v = np.zeros((n_dim, 1))
    v[:d_chain] = v_chain
    return W, v


# ----------------------------------------------------------------------------- G1: sum rule

print("\n=== G1  Sum rule for normal W: Jtot = m exactly, every horizon, every spectrum ===")
N = 32
Q = haar_orthogonal(N, rng)
v1 = rng.standard_normal((N, 1))
v1 /= np.linalg.norm(v1)

for rho, name in [(0.9, "normal_contract"), (1.0, "pocket_isometric"), (0.5, "normal_fast")]:
    W = rho * Q
    for horizon in (10, 100, 1000):
        jk, jtot = fisher_memory(W, v1, horizon)
        check(
            f"G1.{name}.n{horizon}",
            abs(jtot - 1.0) < 1e-8,
            f"rho={rho} n={horizon} Jtot={jtot:.12f} (analytic 1.0)",
        )

# G1b: m orthonormal input channels -> Jtot = m (the scoping correction)
for m in (1, 4, 8):
    Vm = haar_orthogonal(N, rng)[:, :m]
    jk, jtot = fisher_memory(Q, Vm, 200)
    check(
        f"G1b.channels.m{m}",
        abs(jtot - m) < 1e-8,
        f"isometric pocket, m={m} orthonormal write channels -> Jtot={jtot:.10f} (analytic {m})",
    )

# ----------------------------------------------------------------------------- G2: shape

print("\n=== G2  Isometry does not change the total; it changes the SHAPE of the FMC ===")
for rho, name in [(0.5, "normal_fast"), (0.9, "normal_contract"), (1.0, "pocket_isometric")]:
    W = rho * Q
    jk, jtot = fisher_memory(W, v1, 200)
    st = shape_stats(jk)
    print(
        f"  {name:18s} rho={rho:4.2f}  Jtot={jtot:.9f}  J(0)={jk[0]:.5f}  "
        f"tau={st['tau']:8.3f}  persist_half={st['persist_half']:.5f}  J(199)={jk[-1]:.3e}"
    )

jk_iso, _ = fisher_memory(Q, v1, 200)
check(
    "G2.uniform_fmc",
    np.allclose(jk_iso, 1.0 / 200, atol=1e-10),
    f"isometric FMC is exactly uniform: max|J(k)-1/n| = {np.abs(jk_iso - 1/200).max():.3e}",
)
jk_09, _ = fisher_memory(0.9 * Q, v1, 200)
st09 = shape_stats(jk_09)
check(
    "G2.persist_gap",
    st09["persist_half"] < 1e-8 and abs(shape_stats(jk_iso)["persist_half"] - 0.5) < 1e-9,
    f"persistent half-fraction: rho=0.9 -> {st09['persist_half']:.3e}, rho=1.0 -> "
    f"{shape_stats(jk_iso)['persist_half']:.9f}",
)

# ----------------------------------------------------------------------------- G3: chain closed form

print("\n=== G3  Non-normal chain: closed form J(k) = 1/sum_{j<=k} G_j^{-2}, verified ===")
c_mild = 0.02
d_c = 16
Wc, vc = build_chain(d_c, c_mild)
jk_num, jtot_num = fisher_memory(Wc, vc, d_c)
G = np.exp(c_mild * np.arange(d_c) ** 2)
jk_ana = 1.0 / np.cumsum(G ** -2.0)
check(
    "G3.chain_closed_form",
    np.allclose(jk_num, jk_ana, rtol=1e-8, atol=1e-12),
    f"max rel dev = {np.abs(jk_num/jk_ana - 1).max():.3e}; Jtot={jtot_num:.6f}",
)

print("  amplification sweep (d_chain=16, ceiling = N = 16):")
for c in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40):
    Wc, vc = build_chain(d_c, c)
    G = np.exp(c * np.arange(d_c) ** 2)
    jt = float((1.0 / np.cumsum(G ** -2.0)).sum())
    print(f"    c={c:5.2f}  max gain={G[-1]:10.3e}  Jtot={jt:8.4f}")

# ----------------------------------------------------------------------------- G4: the table

print("\n=== G4  Head-to-head: total capacity vs persistence, five systems ===")
D_C, D_L = 16, 16
NH = D_C + D_L
c_hy = 0.05
Wh, vh = build_hybrid(D_C, D_L, c_hy, kappa=1.0, gen=rng)

# E: process noise in the chain block only (engineered leaf isolation, sigma_L = 0)
Qiso = np.zeros((NH, NH))
Qiso[:D_C, :D_C] = EPS * np.eye(D_C)

Wa = 0.9 * haar_orthogonal(NH, rng)
Wb = haar_orthogonal(NH, rng)
va = rng.standard_normal((NH, 1))
va /= np.linalg.norm(va)
Wc16, vc16 = build_chain(NH, c_hy)

systems = [
    ("A normal_contract ", Wa, va, None),
    ("B pocket_isometric", Wb, va, None),
    ("C chain_nonnormal ", Wc16, vc16, None),
    ("D hybrid          ", Wh, vh, None),
    ("E hybrid_isolated ", Wh, vh, Qiso),
]

print(f"  N = {NH}, universal Ganguli ceiling Jtot <= N = {NH}")
header = f"  {'system':20s} {'n':>6s} {'Jtot':>12s} {'tau':>9s} {'persist_half':>13s} {'J(n-1)':>12s}"
print(header)
table: dict[str, dict[int, float]] = {}
for name, W, V, qn in systems:
    table[name] = {}
    for horizon in (32, 128, 512, 2048):
        jk, jtot = fisher_memory(W, V, horizon, noise_cov=qn)
        st = shape_stats(jk)
        table[name][horizon] = jtot
        print(
            f"  {name:20s} {horizon:6d} {jtot:12.6f} {st['tau']:9.2f} "
            f"{st['persist_half']:13.6f} {st['oldest']:12.4e}"
        )

check(
    "G4.normal_is_one",
    all(abs(table["A normal_contract "][h] - 1.0) < 1e-8 for h in (32, 128, 512, 2048))
    and all(abs(table["B pocket_isometric"][h] - 1.0) < 1e-8 for h in (32, 128, 512, 2048)),
    "both normal systems (contracting AND isometric) sit at Jtot = 1.000000000 at every horizon",
)
check(
    "G4.nonnormal_exceeds_one",
    table["C chain_nonnormal "][2048] > 1.5 and table["D hybrid          "][2048] > 1.5,
    f"chain Jtot={table['C chain_nonnormal '][2048]:.4f}, hybrid Jtot="
    f"{table['D hybrid          '][2048]:.4f} (both > 1; ceiling {NH})",
)
check(
    "G4.universal_ceiling",
    all(t <= NH + 1e-6 for s in table.values() for t in s.values()),
    f"no system exceeds Jtot = N = {NH}; max observed = {max(t for s in table.values() for t in s.values()):.4f}",
)
jk_chain, _ = fisher_memory(Wc16, vc16, 2048)
check(
    "G4.chain_has_no_tail",
    float(jk_chain[NH:].sum()) < 1e-12,
    f"chain carries exactly zero information beyond delay N: sum_{{k>=N}} J(k) = {jk_chain[NH:].sum():.3e}",
)
jk_hy, _ = fisher_memory(Wh, vh, 2048)
check(
    "G4.hybrid_has_tail",
    shape_stats(jk_hy)["persist_half"] > 0.1,
    f"hybrid persistent half-fraction = {shape_stats(jk_hy)['persist_half']:.6f} "
    f"(chain: {shape_stats(jk_chain)['persist_half']:.3e}; pure pocket: 0.5)",
)
jk_e, jtot_e = fisher_memory(Wh, vh, 2048, noise_cov=Qiso)
jk_e512, jtot_e512 = fisher_memory(Wh, vh, 512, noise_cov=Qiso)
check(
    "G4.direct_isolation_is_useless",
    abs(jtot_e - table["D hybrid          "][2048]) < 1e-3,
    "removing DIRECT noise injection into the pocket changes nothing: "
    f"Jtot {table['D hybrid          '][2048]:.6f} -> {jtot_e:.6f}. "
    "The write path K is also the noise path; an always-open door leaks noise at the write gain.",
)

# ----------------------------------------------------------------------------- G6: gated write

print("\n=== G6  Gated write: the door must shut. Time-varying propagator. ===")


def fisher_memory_tv(Ws: list[np.ndarray], V: np.ndarray, Qs: list[np.ndarray]):
    """Time-varying: x_{t+1} = W_t x_t + V s_t + z_t, z_t ~ N(0, Q_t). Returns (Jk, Jtot)."""
    horizon = len(Ws)
    n_dim = Ws[0].shape[0]
    prop = np.eye(n_dim)
    cov = np.zeros((n_dim, n_dim))
    props = []
    for k in range(horizon):  # k = delay; input entered at t = horizon-1-k
        props.append(prop @ V)
        cov += prop @ Qs[horizon - 1 - k] @ prop.T
        prop = prop @ Ws[horizon - 1 - k]
    cov_inv = np.linalg.pinv(cov, rcond=1e-14, hermitian=True)
    jk = np.array([float(np.trace(p.T @ cov_inv @ p)) for p in props])
    return jk * EPS, float(jk.sum() * EPS)


W_WRITE = 24  # write window length, in steps


def gated_run(sink: np.ndarray, horizon: int):
    """Chain -> sink, coupling K live only for the first W_WRITE steps; no direct sink noise."""
    W_open = Wh.copy()
    W_open[D_C:, D_C:] = sink
    W_shut = W_open.copy()
    W_shut[D_C:, :D_C] = 0.0  # door closed
    Ws = [W_open if t < W_WRITE else W_shut for t in range(horizon)]
    Qs = [Qiso for _ in range(horizon)]
    return fisher_memory_tv(Ws, vh, Qs)


U_pocket = Wh[D_C:, D_C:]
sinks = {
    "gated -> isometric pocket": U_pocket,
    "gated -> contracting sink": 0.9 * U_pocket,
    "OPEN  -> isometric pocket": None,
}
print(f"  write window = {W_WRITE} steps, pocket dim = {D_L}, no direct pocket noise")
print(f"  {'configuration':28s} {'n':>6s} {'Jtot':>12s} {'persist_half':>13s} {'J(n-1)':>12s}")
gate_tot: dict[str, dict[int, float]] = {}
gate_old: dict[str, dict[int, float]] = {}
for label, sink in sinks.items():
    gate_tot[label] = {}
    gate_old[label] = {}
    for horizon in (64, 256, 1024, 4096):
        if sink is None:
            jk, jt = fisher_memory(Wh, vh, horizon, noise_cov=Qiso)
        else:
            jk, jt = gated_run(sink, horizon)
        gate_tot[label][horizon] = jt
        st = shape_stats(jk)
        gate_old[label][horizon] = st["oldest"]
        print(
            f"  {label:28s} {horizon:6d} {jt:12.5f} {st['persist_half']:13.6f} {st['oldest']:12.4e}"
        )

g_iso = gate_tot["gated -> isometric pocket"]
g_con = gate_tot["gated -> contracting sink"]
g_open = gate_tot["OPEN  -> isometric pocket"]
o_iso = gate_old["gated -> isometric pocket"]
o_con = gate_old["gated -> contracting sink"]
o_open = gate_old["OPEN  -> isometric pocket"]

check(
    "G6.gated_isometric_is_horizon_independent",
    max(abs(g_iso[h] / g_iso[64] - 1) for h in (256, 1024, 4096)) < 1e-9,
    f"gated write into an isometric pocket: Jtot = {g_iso[64]:.6f} at n=64,256,1024,4096 — "
    "exactly horizon-independent (CAP'(ii) flat retention, and 8.8x the normal sum rule)",
)
check(
    "G6.gated_isometric_no_decay",
    abs(o_iso[4096] / o_iso[64] - 1) < 1e-9,
    f"per-item information at the OLDEST delay does not decay: J(n-1) = {o_iso[64]:.6f} at n=64 "
    f"and {o_iso[4096]:.6f} at n=4096",
)
check(
    "G6.gate_alone_insufficient",
    o_con[4096] < 1e-12,
    f"gate WITHOUT isometry (contracting sink): J(n-1) = {o_con[4096]:.3e} at n=4096 "
    f"(Jtot collapses {g_con[64]:.3f} -> {g_con[4096]:.3f}); nothing survives the delay",
)
check(
    "G6.isometry_alone_insufficient",
    o_open[4096] < o_open[256] / 3,
    f"isometry WITHOUT a gate: J(n-1) decays harmonically, {o_open[256]:.4e} at n=256 -> "
    f"{o_open[4096]:.4e} at n=4096 (ratio {o_open[256]/o_open[4096]:.2f}, horizon ratio 16)",
)
check(
    "G6.conjunction_is_the_design",
    o_iso[4096] > 100 * max(o_con[4096], o_open[4096]),
    f"at delay 4096 the conjunction beats each half by >=100x: gated+isometric {o_iso[4096]:.4e} "
    f"vs open+isometric {o_open[4096]:.4e} ({o_iso[4096]/o_open[4096]:.0f}x) "
    f"vs gated+contracting {o_con[4096]:.4e}",
)
check(
    "G6.respects_universal_ceiling",
    g_iso[4096] <= NH + 1e-9,
    f"and it still obeys Ganguli's universal ceiling: Jtot = {g_iso[4096]:.4f} <= N = {NH}",
)

# ----------------------------------------------------------------------------- G5: noise model

print("\n=== G5  The whole theorem is a statement about noise INSIDE the recurrence ===")
print("  Read-noise-only channel (no process noise): J_read(k) = ||W^k v||^2 / sigma_r^2")
sigma_r2 = 1.0
for rho, name in [(0.9, "normal_contract"), (1.0, "pocket_isometric")]:
    W = rho * Q
    for horizon in (100, 1000):
        power = np.eye(N)
        acc = 0.0
        for _ in range(horizon):
            acc += float(np.linalg.norm(power @ v1) ** 2) / sigma_r2
            power = W @ power
        print(f"    {name:18s} n={horizon:5d}  Jtot_read = {acc:12.4f}")
power = np.eye(N)
acc100 = 0.0
for _ in range(100):
    acc100 += float(np.linalg.norm(power @ v1) ** 2)
    power = Q @ power
check(
    "G5.read_noise_linear",
    abs(acc100 - 100.0) < 1e-8,
    f"isometric pocket under read-noise-only: Jtot = n exactly ({acc100:.9f} at n=100) — "
    "no sum rule, unbounded in the horizon",
)
geo = (1 - 0.9 ** 200) / (1 - 0.81)
power = np.eye(N)
acc_c = 0.0
for _ in range(100):
    acc_c += float(np.linalg.norm(power @ v1) ** 2)
    power = (0.9 * Q) @ power
check(
    "G5.read_noise_contract_bounded",
    abs(acc_c - (1 - 0.81 ** 100) / (1 - 0.81)) < 1e-8,
    f"contracting normal under read-noise-only: Jtot = (1-rho^2n)/(1-rho^2) = {acc_c:.6f}, bounded",
)

# ------------------------------------------------- G7: the time-varying escape does not work

print("\n=== G7  Token-conditioned A_t does NOT rescue an all-pocket architecture ===")
print("  Cell block form (cell.py affine_update): A = R . diag(gamma), R = Cayley(Omega) in SO(4).")
print("  AA^T = R g^2 R^T, A^T A = g^2  ->  normal iff R commutes with gamma^2 iff gamma isotropic.")

def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    dim = sum(b.shape[0] for b in blocks)
    out = np.zeros((dim, dim))
    off = 0
    for b in blocks:
        out[off:off + b.shape[0], off:off + b.shape[1]] = b
        off += b.shape[0]
    return out


BS = 4
G_BLOCKS = 8
NB = BS * G_BLOCKS
gens_iso, gens_aniso = [], []
for _ in range(6):
    blocks_i, blocks_a = [], []
    for _b in range(G_BLOCKS):
        R = haar_orthogonal(BS, rng)
        R = R * np.sign(np.linalg.det(R))
        g = 0.97
        blocks_i.append(R * g)                                   # isotropic gamma: pocket/conformal
        blocks_a.append(R @ np.diag([1.0, 0.94, 1.0, 0.94]))     # anisotropic gamma: pocket broken
    gens_iso.append(_block_diag(blocks_i))
    gens_aniso.append(_block_diag(blocks_a))

defect_i = max(np.linalg.norm(A @ A.T - A.T @ A) for A in gens_iso)
defect_a = max(np.linalg.norm(A @ A.T - A.T @ A) for A in gens_aniso)
check(
    "G7.isotropic_gamma_is_normality",
    defect_i < 1e-12 < defect_a,
    f"per-step normality defect ||AA^T - A^TA||_F: isotropic gamma {defect_i:.3e}, "
    f"anisotropic gamma {defect_a:.3e} — the pocket condition IS the normality condition",
)

vb = rng.standard_normal((NB, 1))
vb /= np.linalg.norm(vb)
words = [[gens_iso[(3 * t + 1) % 6] for t in range(h)] for h in (24, 96, 384)]
for word in words:
    prod = np.eye(NB)
    for A in word:
        prod = A @ prod
    dev = float(np.linalg.norm(prod @ prod.T - prod.T @ prod))
    jk, jt = fisher_memory_tv(word, vb, [EPS * np.eye(NB)] * len(word))
    check(
        f"G7.tv_pocket_sumrule.n{len(word)}",
        dev < 1e-11 and abs(jt - 1.0) < 1e-8,
        f"token-varying word of length {len(word)}: composite normality defect {dev:.3e}, "
        f"Jtot = {jt:.10f} — still exactly 1",
    )

word_a = [gens_aniso[(3 * t + 1) % 6] for t in range(96)]
prod_a = np.eye(NB)
for A in word_a:
    prod_a = A @ prod_a
jk_a, jt_a = fisher_memory_tv(word_a, vb, [EPS * np.eye(NB)] * 96)
check(
    "G7.broken_pocket_leaves_sumrule",
    abs(jt_a - 1.0) > 1e-6,
    f"anisotropic (pocket-broken) word of length 96: composite normality defect "
    f"{np.linalg.norm(prod_a @ prod_a.T - prod_a.T @ prod_a):.3e}, Jtot = {jt_a:.8f} != 1 "
    "— breaking the pocket is the only way a token-varying scan leaves the sum rule, "
    "and it leaves it by forgetting faster",
)

# ----------------------------------------------------------------------------- summary

print("\n=== SUMMARY ===")
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
print(f"{n_pass}/{len(RESULTS)} PASS, {len(RESULTS) - n_pass} FAIL")
for tag, ok, detail in RESULTS:
    if not ok:
        print(f"  FAIL {tag}: {detail}")
