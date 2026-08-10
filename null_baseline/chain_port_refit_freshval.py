"""Eval-only check of the ALREADY-FIT port_coefs.b64.txt stabilizer against a genuinely
fresh net range never touched by any v3-v6 iteration (all of which used nets 0-49 as
train/val). This addresses the adversarial-review finding that our held-out set (nets 40-49)
was reused as the judge of "did this help?" across five design iterations. No refitting here
-- just loading the coefficients frozen at the end of chain_port_refit.py's run and scoring
them on nets that never influenced any design decision.

Usage: uv run python chain_port_refit_freshval.py [START_IDX] [N]
"""
import glob, sys, time, base64, struct
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from math import erf

START = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
K_H = 10
SQ2PI = np.sqrt(2 * np.pi)

DATA_GLOB = str(Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/*/data/full-*.parquet")
ATLAS_GLOB = str(Path.home() / ".cache/huggingface/hub/datasets--keenanpepper--arc-whestbench-higher-moments-2026/snapshots/*/full/mlp_%05d.npz")

def phi(x): return np.exp(-0.5 * x * x) / SQ2PI

def Phi(x):
    z = x / np.sqrt(2.0)
    s = np.sign(z); az = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-az * az)
    return 0.5 * (1.0 + s * e)

def hermite_coeffs(a, K):
    out = [phi(a) + a * Phi(a), Phi(a)]
    He_prev = np.ones_like(a); He_cur = -a; fact = 2.0
    out.append(He_prev * phi(a) / fact)
    for k in range(3, K + 1):
        fact *= k
        out.append(He_cur * phi(a) / fact)
        He_prev, He_cur = He_cur, (-a) * He_cur - (k - 2) * He_prev
    return out

def step_np(m_act, S_act, W, K=K_H):
    mu = m_act @ W
    Spre = W.T @ S_act @ W
    var = np.clip(np.diag(Spre), 1e-30, None)
    sig = np.sqrt(var)
    a = mu / sig
    rho = np.clip(Spre / np.outer(sig, sig), -0.9999, 0.9999)
    coeffs = hermite_coeffs(a, K)
    m_post = sig * coeffs[0]
    S_post = np.zeros_like(Spre)
    rk = rho.copy()
    for k in range(1, K + 1):
        ck = coeffs[k]
        S_post = S_post + np.outer(ck, ck) * rk
        rk = rk * rho
    S_post = S_post * np.outer(sig, sig)
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

def load_nets(start, n):
    out = []
    idx = 0
    for f in sorted(glob.glob(DATA_GLOB)):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2, columns=["weights", "final_means"]):
            cols = batch.to_pydict()
            for i in range(len(cols["weights"])):
                if start <= idx < start + n:
                    out.append((idx, np.asarray(cols["weights"][i], dtype=np.float64),
                                np.asarray(cols["final_means"][i], dtype=np.float64)))
                idx += 1
                if idx >= start + n:
                    return out
    return out

# --- load frozen coefficients (fit once, on nets 0-39, never refit here) ---
blob = Path(__file__).with_name("port_coefs.b64.txt").read_text()
flat = struct.unpack("<512d", base64.b64decode(blob))
coefs = []
p = 0
for l in range(32):
    Am = np.array(flat[p:p+8]); p += 8
    Av = np.array(flat[p:p+6]); p += 6
    Ao = np.array(flat[p:p+2]); p += 2
    coefs.append((Am, Av, Ao))
assert p == 512

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

def run_baseline(W):
    m, S = np.zeros(256), np.eye(256)
    for l in range(32):
        m, S, *_ = step_np(m, S, W[l])
    return m

t0 = time.time()
nets = load_nets(START, N)
print(f"loaded {len(nets)} FRESH nets (indices {START}-{START+N-1}, never used in v3-v6 train/val), {time.time()-t0:.0f}s", flush=True)

corrected_vals, baseline_vals = [], []
for idx, W, official_fm in nets:
    atlas_files = sorted(glob.glob(ATLAS_GLOB % idx))
    if not atlas_files:
        print(f"  net {idx}: NO ATLAS FILE, skipping")
        continue
    atlas = np.load(atlas_files[0])
    truth = atlas["mean"][31].astype(np.float64)
    assert float(np.max(np.abs(atlas["official_alm"][-1].astype(np.float64) - official_fm))) < 1e-5
    probe = probe_cumulants(W, 4096, seed=1000 + idx)
    est_c = run_corrected(W, probe)
    est_b = run_baseline(W)
    mse_c = float(np.mean((est_c - truth) ** 2))
    mse_b = float(np.mean((est_b - truth) ** 2))
    corrected_vals.append(mse_c)
    baseline_vals.append(mse_b)
    print(f"  net {idx}: baseline {mse_b:.3e}  corrected {mse_c:.3e}", flush=True)

c = np.array(corrected_vals)
b = np.array(baseline_vals)
print(f"\n=== FRESH held-out (n={len(c)}, indices {START}-{START+N-1}, never touched during v3-v6) ===")
print(f"corrected: mean {c.mean():.3e}  median {np.median(c):.3e}  std {c.std():.3e}  sem {c.std()/np.sqrt(len(c)):.3e}")
print(f"baseline : mean {b.mean():.3e}  median {np.median(b):.3e}  std {b.std():.3e}")
print(f"total {time.time()-t0:.0f}s")
