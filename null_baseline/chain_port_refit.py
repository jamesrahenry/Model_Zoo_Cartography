"""Portable-step chain + stabilizer refit (torch-free math, numpy only).

Step: Hermite-series bivariate Gaussian relu closure (K terms), exact diag.
Refit v4-feature stabilizer against THIS step so coefficients match the fnp port.
Validates one-step parity vs kprop is unnecessary — we refit end-to-end.

Usage: uv run python chain_port_refit.py [TRAIN_N] [VAL_N] [N_PROBE] [K_HERMITE]
"""
import glob, sys, time, json, base64, struct
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from math import erf

TRAIN_N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
VAL_N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_PROBE = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
K_H = int(sys.argv[4]) if len(sys.argv) > 4 else 10
SQ2PI = np.sqrt(2 * np.pi)

DATA_GLOB = str(Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/*/data/full-*.parquet")
ATLAS_GLOB = str(Path.home() / ".cache/huggingface/hub/datasets--keenanpepper--arc-whestbench-higher-moments-2026/snapshots/*/full/mlp_%05d.npz")

def phi(x): return np.exp(-0.5 * x * x) / SQ2PI

def Phi(x):
    """A&S 7.1.26-based normal CDF — portable to fnp (no erf needed)."""
    z = x / np.sqrt(2.0)
    s = np.sign(z); az = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-az * az)
    return 0.5 * (1.0 + s * e)

def hermite_coeffs(a, K):
    """a_k(alpha) for relu(alpha+Z), k=0..K. He_m via recurrence."""
    out = [phi(a) + a * Phi(a), Phi(a)]
    He_prev = np.ones_like(a)      # He_0(-a)
    He_cur = -a                    # He_1(-a)
    fact = 2.0
    out.append(He_prev * phi(a) / fact)         # k=2: He_0(-a) phi/2!
    for k in range(3, K + 1):
        fact *= k
        out.append(He_cur * phi(a) / fact)      # k: He_{k-2}(-a) phi/k!
        He_prev, He_cur = He_cur, (-a) * He_cur - (k - 2) * He_prev
    return out

def step_np(m_act, S_act, W, K=K_H):
    """Post-relu (mean, cov) from input act (mean, cov) through weight W. Portable math."""
    mu = m_act @ W
    Spre = W.T @ S_act @ W
    var = np.clip(np.diag(Spre), 1e-30, None)
    sig = np.sqrt(var)
    a = mu / sig
    rho = Spre / np.outer(sig, sig)
    rho = np.clip(rho, -0.9999, 0.9999)
    coeffs = hermite_coeffs(a, K)
    m_post = sig * coeffs[0]
    # covariance: sig_u sig_v * sum_{k>=1} a_k(u) a_k(v) rho^k
    S_post = np.zeros_like(Spre)
    rk = rho.copy()
    for k in range(1, K + 1):
        ck = coeffs[k]
        S_post = S_post + np.outer(ck, ck) * rk
        rk = rk * rho
    S_post = S_post * np.outer(sig, sig)
    # exact diagonal: Var[relu] = var*[(1+a^2)Phi + a phi] - m_post^2
    d = var * ((1 + a * a) * Phi(a) + a * phi(a)) - m_post ** 2
    S_post[np.diag_indices(len(mu))] = np.clip(d, 1e-30, None)
    return m_post, S_post, mu, sig, Spre, a

def probe_cumulants(W, n_probe, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_probe, 256))
    out = []
    for l in range(32):
        z = x @ W[l]
        mu = z.mean(0); zc = z - mu
        s2 = (zc ** 2).mean(0)
        out.append(((zc ** 3).mean(0), (zc ** 4).mean(0) - 3 * s2 ** 2, (zc ** 2).T @ zc / n_probe))
        x = np.maximum(z, 0.0)
    return out

def mean_features(a, sig, k3, k4):
    pa = phi(a)
    c3 = -(k3 / 6.0) * (a * pa / sig ** 2)
    c4 = (k4 / 24.0) * (pa * (a * a - 1) / sig ** 3)
    return np.stack([c3, c4, pa * sig, a * pa * sig, a * a * pa * sig,
                     c3 * a, c4 * a, (k3 / sig ** 2) * pa], axis=1)

def var_features(a, sig, k3, k4):
    pa = phi(a)
    cm2 = k3 * pa / (3 * sig) - k4 * a * pa / (12 * sig ** 2)
    return np.stack([cm2, sig ** 2 * pa, sig ** 2 * a * pa, sig ** 2 * a * a * pa,
                     k3 * pa / sig, k4 * pa / sig ** 2], axis=1)

def tcross_feats(mu_pre, sig_pre, Spre, kiij, a):
    rho = np.clip(Spre / np.outer(sig_pre, sig_pre), -0.999, 0.999)
    s_cond = sig_pre[None, :] * np.sqrt(np.clip(1 - rho ** 2, 1e-12, None))
    m_cond0 = mu_pre[None, :] - rho * sig_pre[None, :] * a[:, None]
    G = (phi(a)[:, None] / sig_pre[:, None]) * Phi(m_cond0 / s_cond)
    T1 = 0.5 * (kiij * G + (kiij * G).T)
    return [T1, T1 * rho]

def load_nets(n):
    out = []
    for f in sorted(glob.glob(DATA_GLOB)):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2, columns=["weights", "final_means"]):
            cols = batch.to_pydict()
            for i in range(len(cols["weights"])):
                out.append((np.asarray(cols["weights"][i], dtype=np.float64),
                            np.asarray(cols["final_means"][i], dtype=np.float64)))
                if len(out) >= n:
                    return out
    return out

nets = load_nets(TRAIN_N + VAL_N)
atl, prb = [], []
for idx, (W, official_fm) in enumerate(nets):
    atlas = np.load(sorted(glob.glob(ATLAS_GLOB % idx))[0])
    assert float(np.max(np.abs(atlas["official_alm"][-1].astype(np.float64) - official_fm))) < 1e-5
    atl.append({"mean": atlas["mean"].astype(np.float64), "M11": atlas["M11"].astype(np.float64)})
    prb.append(probe_cumulants(W, N_PROBE, seed=1000 + idx))
print(f"loaded {len(nets)} nets, probe {N_PROBE}, K_H {K_H}", flush=True)

t0 = time.time()
# baseline: uncorrected portable chain (sanity vs kprop's 7.3e-5)
base = []
for v in range(TRAIN_N, TRAIN_N + VAL_N):
    W = nets[v][0]
    m, S = np.zeros(256), np.eye(256)
    for l in range(32):
        m, S, *_ = step_np(m, S, W[l])
    base.append(float(np.mean((m - atl[v]["mean"][31]) ** 2)))
print(f"uncorrected portable chain (val): {np.mean(base):.3e}  ({time.time()-t0:.0f}s)", flush=True)

# sequential stabilizer fit
states = [(np.zeros(256), np.eye(256)) for _ in range(TRAIN_N)]
coefs = []
for l in range(32):
    Xm, ym, Xv, yv, Xo, yo, cache = [], [], [], [], [], [], []
    for t in range(TRAIN_N):
        W = nets[t][0]
        m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(*states[t], W[l])
        k3, k4, kiij = prb[t][l]
        Fm = mean_features(a, sig_pre, k3, k4)
        Fv = var_features(a, sig_pre, k3, k4)
        Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
        cache.append((m_hat, S_hat, Fm, Fv, Fo))
        tm = atl[t]["mean"][l]
        tS = atl[t]["M11"][l] - np.outer(tm, tm)
        Xm.append(Fm); ym.append(tm - m_hat)
        Xv.append(Fv); yv.append(np.diag(tS) - np.diag(S_hat))
        iu = np.triu_indices(256, k=1)
        Xo.append(np.stack([f[iu] for f in Fo], axis=1)); yo.append((tS - S_hat)[iu])
    Am, *_ = np.linalg.lstsq(np.concatenate(Xm), np.concatenate(ym), rcond=None)
    Av, *_ = np.linalg.lstsq(np.concatenate(Xv), np.concatenate(yv), rcond=None)
    Ao, *_ = np.linalg.lstsq(np.concatenate(Xo), np.concatenate(yo), rcond=None)
    coefs.append((Am, Av, Ao))
    for t in range(TRAIN_N):
        m_hat, S_hat, Fm, Fv, Fo = cache[t]
        K1 = m_hat + Fm @ Am
        K2 = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
        K2[np.diag_indices(256)] = np.diag(S_hat) + Fv @ Av
        states[t] = (K1, K2)
    if l in (0, 15, 31):
        r = np.mean([np.mean((states[t][0] - atl[t]['mean'][l]) ** 2) for t in range(TRAIN_N)])
        print(f"layer {l+1}: train mean-mse {r:.3e} ({time.time()-t0:.0f}s)", flush=True)

def run_corrected(W, probe):
    m, S = np.zeros(256), np.eye(256)
    for l in range(32):
        m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(m, S, W[l])
        k3, k4, kiij = probe[l]
        Am, Av, Ao = coefs[l]
        Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
        m = m_hat + mean_features(a, sig_pre, k3, k4) @ Am
        S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
        S[np.diag_indices(256)] = np.diag(S_hat) + var_features(a, sig_pre, k3, k4) @ Av
    return m

val = []
for v in range(TRAIN_N, TRAIN_N + VAL_N):
    est = run_corrected(nets[v][0], prb[v])
    val.append(float(np.mean((est - atl[v]["mean"][31]) ** 2)))
    print(f"val net {v}: {val[-1]:.3e}", flush=True)
print(f"\n=== PORTABLE chain, refit stabilizer: val {np.mean(val):.3e} med {np.median(val):.3e} ===")
print("(kprop-based v4 reference: 6.52e-6)")

# save coefficients as b64 for embedding
flat = []
for Am, Av, Ao in coefs:
    flat += list(Am) + list(Av) + list(Ao)
blob = base64.b64encode(struct.pack("<%dd" % len(flat), *flat)).decode()
Path(__file__).with_name("port_coefs.b64.txt").write_text(blob)
print(f"saved {len(flat)} coefficients ({len(blob)} b64 chars). total {time.time()-t0:.0f}s")
