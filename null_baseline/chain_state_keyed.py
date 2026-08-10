"""State-keyed stabilizer: depth-generic replacement for the per-layer-index coefficient table.

The graded estimator's only depth-32 pin is _LC: 32 rows x 16 stabilizer coefficients
looked up by layer index (clamped to row 31 beyond depth 32). This script refits the same
16 correction slots as a function of MEASURED per-layer state instead of index:

    c_j(state) = theta_j . g(state),   g built from the chain's own pre-activation state:
      q    = PR(Spre)/w = tr(Spre)^2 / (w * ||Spre||_F^2)   (rank-collapse clock)
      abar = mean(a), asd = std(a)                          (ReLU operating point)
      rbar = mean |rho| offdiag                             (correlation buildup)

Fit is pooled across ALL layers of all train nets (single global theta, 16 x B numbers vs
512), DAgger-iterated: roll trajectories with current theta, regress residual-to-truth at
every layer against (feature x basis) design, repeat. Truth = keenanpepper atlas layer
moments, same 40/10 train/val split and probe seeds (1000+idx) as chain_port_refit.py, so
E1 is an exact head-to-head against the deployed coefficients in port_coefs.b64.txt.

Rationale: for iid He-Gaussian nets a depth-d net is distributionally the first d layers
of a depth-32 net, so the index table is implicitly a lookup on "ensemble-typical state at
layer l". Keying on measured state should (a) match at depth 32, (b) extrapolate past it
(see depth_extrap_eval.py), (c) adapt to nets whose state at layer l is atypical for l.

Usage: uv run python chain_state_keyed.py [TRAIN_N] [VAL_N] [N_PROBE] [K_HERMITE] [NITER]
"""
import glob, sys, time, json, base64, struct
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

TRAIN_N, VAL_N, N_PROBE, K_H, NITER = 40, 10, 4096, 10, 4
if __name__ == "__main__":  # argv must not leak into importers (depth_extrap_eval.py)
    argv = sys.argv[1:] + [None] * 5
    TRAIN_N = int(argv[0]) if argv[0] else TRAIN_N
    VAL_N = int(argv[1]) if argv[1] else VAL_N
    N_PROBE = int(argv[2]) if argv[2] else N_PROBE
    K_H = int(argv[3]) if argv[3] else K_H
    NITER = int(argv[4]) if argv[4] else NITER
W_DIM, DEPTH = 256, 32
NMF, NVF, NOF = 8, 6, 2
SQ2PI = np.sqrt(2 * np.pi)

DATA_GLOB = str(Path.home() / ".cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/*/data/full-*.parquet")
ATLAS_GLOB = str(Path.home() / ".cache/huggingface/hub/datasets--keenanpepper--arc-whestbench-higher-moments-2026/snapshots/*/full/mlp_%05d.npz")
REF_COEF_PATH = Path(__file__).with_name("port_coefs.b64.txt")


def phi(x):
    return np.exp(-0.5 * x * x) / SQ2PI


def Phi(x):
    z = x / np.sqrt(2.0)
    s = np.sign(z); az = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-az * az)
    return 0.5 * (1.0 + s * e)


def hermite_coeffs(a, K):
    out = [phi(a) + a * Phi(a), Phi(a)]
    He_prev = np.ones_like(a)
    He_cur = -a
    fact = 2.0
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
    return m_post, S_post, mu, sig, Spre, a, rho


def probe_cumulants(W, n_probe, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_probe, W.shape[1]))
    out = []
    for l in range(W.shape[0]):
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


def tcross_feats(mu_pre, sig_pre, Spre, kiij, a, rho):
    s_cond = sig_pre[None, :] * np.sqrt(np.clip(1 - rho ** 2, 1e-12, None))
    m_cond0 = mu_pre[None, :] - rho * sig_pre[None, :] * a[:, None]
    G = (phi(a)[:, None] / sig_pre[:, None]) * Phi(m_cond0 / s_cond)
    T1 = 0.5 * (kiij * G + (kiij * G).T)
    return [T1, T1 * rho]


# ---------------- state descriptors and basis ----------------
N_BASIS = 8

def state_basis(Spre, a, rho):
    """g(state): B=8 basis values from the chain's own pre-activation state."""
    w = len(a)
    tr = np.trace(Spre)
    q = (tr * tr) / (w * np.sum(Spre * Spre) + 1e-300)      # PR/w in (0,1]
    abar = float(np.clip(np.mean(a), -8.0, 8.0))
    asd = float(np.clip(np.std(a), 0.0, 8.0))
    off = rho[np.triu_indices(w, k=1)]
    rbar = float(np.mean(np.abs(off)))
    return np.array([1.0, q, q * q, abar, asd, rbar, q * abar, abar * abar])


def corrections(theta, g, Fm, Fv, Fo):
    """Apply state-keyed corrections. theta = (Tm[8,B], Tv[6,B], To[2,B])."""
    Tm, Tv, To = theta
    cm = Tm @ g       # 8 slot coefficients
    cv = Tv @ g
    co = To @ g
    dm = Fm @ cm
    dv = Fv @ cv
    dS = co[0] * Fo[0] + co[1] * Fo[1]
    return dm, dv, dS


def roll_chain(W, probe, theta, collect=None, truth=None):
    """Run the corrected chain. If collect is not None, append (per-layer) design
    blocks against truth atlas into the accumulator dict. Returns per-layer means."""
    w = W.shape[1]
    m, S = np.zeros(w), np.eye(w)
    means = []
    iu = np.triu_indices(w, k=1)
    for l in range(W.shape[0]):
        m_hat, S_hat, mu_pre, sig_pre, Spre, a, rho = step_np(m, S, W[l])
        k3, k4, kiij = probe[l]
        Fm = mean_features(a, sig_pre, k3, k4)
        Fv = var_features(a, sig_pre, k3, k4)
        Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a, rho)
        g = state_basis(Spre, a, rho)
        if collect is not None:
            tm = truth["mean"][l]
            tS = truth["M11"][l] - np.outer(tm, tm)
            # mean block: rows w, cols 8*B  (F_j * g_b)
            Xm = (Fm[:, :, None] * g[None, None, :]).reshape(w, NMF * N_BASIS)
            ym = tm - m_hat
            collect["m_xtx"] += Xm.T @ Xm; collect["m_xty"] += Xm.T @ ym
            Xv = (Fv[:, :, None] * g[None, None, :]).reshape(w, NVF * N_BASIS)
            yv = np.diag(tS) - np.diag(S_hat)
            collect["v_xtx"] += Xv.T @ Xv; collect["v_xty"] += Xv.T @ yv
            Fo_iu = np.stack([f[iu] for f in Fo], axis=1)
            Xo = (Fo_iu[:, :, None] * g[None, None, :]).reshape(len(iu[0]), NOF * N_BASIS)
            yo = (tS - S_hat)[iu]
            collect["o_xtx"] += Xo.T @ Xo; collect["o_xty"] += Xo.T @ yo
        dm, dv, dS = corrections(theta, g, Fm, Fv, Fo)
        m = m_hat + dm
        S = S_hat + dS
        S[np.diag_indices(w)] = np.clip(np.diag(S_hat) + dv, 1e-30, None)
        means.append(np.maximum(m, 0.0))
    return means


def zero_theta():
    return (np.zeros((NMF, N_BASIS)), np.zeros((NVF, N_BASIS)), np.zeros((NOF, N_BASIS)))


def solve_theta(acc, ridge=1e-9):
    def solve(xtx, xty, nf):
        A = xtx + ridge * np.trace(xtx) / max(len(xtx), 1) * np.eye(len(xtx))
        return np.linalg.solve(A, xty).reshape(nf, N_BASIS)
    return (solve(acc["m_xtx"], acc["m_xty"], NMF),
            solve(acc["v_xtx"], acc["v_xty"], NVF),
            solve(acc["o_xtx"], acc["o_xty"], NOF))


def fresh_acc():
    return {"m_xtx": np.zeros((NMF * N_BASIS,) * 2), "m_xty": np.zeros(NMF * N_BASIS),
            "v_xtx": np.zeros((NVF * N_BASIS,) * 2), "v_xty": np.zeros(NVF * N_BASIS),
            "o_xtx": np.zeros((NOF * N_BASIS,) * 2), "o_xty": np.zeros(NOF * N_BASIS)}


# ---------------- index-keyed reference (deployed coefficients) ----------------
def load_ref_coefs():
    blob = base64.b64decode(REF_COEF_PATH.read_text().strip())
    vals = struct.unpack("<%dd" % (len(blob) // 8), blob)
    assert len(vals) == 512
    return [(np.array(vals[i*16:i*16+8]), np.array(vals[i*16+8:i*16+14]), np.array(vals[i*16+14:i*16+16])) for i in range(32)]


def roll_chain_ref(W, probe, ref):
    w = W.shape[1]
    m, S = np.zeros(w), np.eye(w)
    means = []
    for l in range(W.shape[0]):
        m_hat, S_hat, mu_pre, sig_pre, Spre, a, rho = step_np(m, S, W[l])
        k3, k4, kiij = probe[l]
        Am, Av, Ao = ref[l] if l < len(ref) else ref[-1]
        Fm = mean_features(a, sig_pre, k3, k4)
        Fv = var_features(a, sig_pre, k3, k4)
        Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a, rho)
        m = m_hat + Fm @ Am
        S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
        S[np.diag_indices(w)] = np.clip(np.diag(S_hat) + Fv @ Av, 1e-30, None)
        means.append(np.maximum(m, 0.0))
    return means


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


if __name__ == "__main__":
    t0 = time.time()
    nets = load_nets(TRAIN_N + VAL_N)
    atl, prb = [], []
    for idx, (W, official_fm) in enumerate(nets):
        atlas = np.load(sorted(glob.glob(ATLAS_GLOB % idx))[0])
        assert float(np.max(np.abs(atlas["official_alm"][-1].astype(np.float64) - official_fm))) < 1e-5
        atl.append({"mean": atlas["mean"].astype(np.float64), "M11": atlas["M11"].astype(np.float64)})
        prb.append(probe_cumulants(W, N_PROBE, seed=1000 + idx))
    print(f"loaded {len(nets)} nets, probe {N_PROBE}, K_H {K_H}, basis {N_BASIS} ({time.time()-t0:.0f}s)", flush=True)

    # ---- stage 1: sequential-prefix fit (the load-bearing structure from the
    # original: each layer's design rows come from the CORRECTED chain's own
    # states, advanced one layer at a time with the theta fitted so far; theta
    # is global and refit on the cumulative design after each layer) ----
    theta = zero_theta()
    acc = fresh_acc()
    states = [(np.zeros(W_DIM), np.eye(W_DIM)) for _ in range(TRAIN_N)]
    iu = np.triu_indices(W_DIM, k=1)
    for l in range(DEPTH):
        cache = []
        for t in range(TRAIN_N):
            m_hat, S_hat, mu_pre, sig_pre, Spre, a, rho = step_np(*states[t], nets[t][0][l])
            k3, k4, kiij = prb[t][l]
            Fm = mean_features(a, sig_pre, k3, k4)
            Fv = var_features(a, sig_pre, k3, k4)
            Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a, rho)
            g = state_basis(Spre, a, rho)
            tm = atl[t]["mean"][l]
            tS = atl[t]["M11"][l] - np.outer(tm, tm)
            Xm = (Fm[:, :, None] * g[None, None, :]).reshape(W_DIM, NMF * N_BASIS)
            acc["m_xtx"] += Xm.T @ Xm; acc["m_xty"] += Xm.T @ (tm - m_hat)
            Xv = (Fv[:, :, None] * g[None, None, :]).reshape(W_DIM, NVF * N_BASIS)
            acc["v_xtx"] += Xv.T @ Xv; acc["v_xty"] += Xv.T @ (np.diag(tS) - np.diag(S_hat))
            Fo_iu = np.stack([f[iu] for f in Fo], axis=1)
            Xo = (Fo_iu[:, :, None] * g[None, None, :]).reshape(len(iu[0]), NOF * N_BASIS)
            acc["o_xtx"] += Xo.T @ Xo; acc["o_xty"] += Xo.T @ (tS - S_hat)[iu]
            cache.append((m_hat, S_hat, Fm, Fv, Fo, g))
        theta = solve_theta(acc)
        for t in range(TRAIN_N):
            m_hat, S_hat, Fm, Fv, Fo, g = cache[t]
            dm, dv, dS = corrections(theta, g, Fm, Fv, Fo)
            K2 = S_hat + dS
            K2[np.diag_indices(W_DIM)] = np.clip(np.diag(S_hat) + dv, 1e-30, None)
            states[t] = (m_hat + dm, K2)
        if l in (0, 7, 15, 23, 31):
            r = np.mean([np.mean((states[t][0] - atl[t]["mean"][l]) ** 2) for t in range(TRAIN_N)])
            print(f"seq layer {l+1}: train mean-mse {r:.3e} ({time.time()-t0:.0f}s)", flush=True)

    def val_eval(th):
        rows = []
        for v in range(TRAIN_N, TRAIN_N + VAL_N):
            means = roll_chain(nets[v][0], prb[v], th)
            rows.append([float(np.mean((means[l] - atl[v]["mean"][l]) ** 2)) for l in range(DEPTH)])
        return rows

    val_all = val_eval(theta)
    val_final = [r[-1] for r in val_all]
    history = [{"stage": "sequential", "val_final_mean": float(np.mean(val_final)),
                "val_final_median": float(np.median(val_final))}]
    print(f"sequential fit: val final-layer mean {np.mean(val_final):.3e} median {np.median(val_final):.3e} ({time.time()-t0:.0f}s)", flush=True)

    # ---- stage 2: damped self-consistency polish (roll full trajectory with
    # current theta, refit on it, mix with damping; keep only if val improves) ----
    best_theta, best_val = theta, float(np.mean(val_final))
    eta = 0.5
    for it in range(NITER):
        acc = fresh_acc()
        for t in range(TRAIN_N):
            roll_chain(nets[t][0], prb[t], best_theta, collect=acc, truth=atl[t])
        fitted = solve_theta(acc)
        cand = tuple((1 - eta) * b + eta * f for b, f in zip(best_theta, fitted))
        cand_val = float(np.mean([r[-1] for r in val_eval(cand)]))
        history.append({"stage": f"polish{it+1}", "val_final_mean": cand_val, "accepted": cand_val < best_val})
        print(f"polish {it+1}: val {cand_val:.3e} ({'accepted' if cand_val < best_val else 'rejected'}, eta={eta}) ({time.time()-t0:.0f}s)", flush=True)
        if cand_val < best_val:
            best_theta, best_val = cand, cand_val
        else:
            eta *= 0.5
            if eta < 0.1:
                break
    theta = best_theta

    # E1 head-to-head vs deployed index-keyed coefficients, same probes
    ref = load_ref_coefs()
    ref_all, sk_all = [], []
    for v in range(TRAIN_N, TRAIN_N + VAL_N):
        mr = roll_chain_ref(nets[v][0], prb[v], ref)
        ms = roll_chain(nets[v][0], prb[v], theta)
        ref_all.append([float(np.mean((mr[l] - atl[v]["mean"][l]) ** 2)) for l in range(DEPTH)])
        sk_all.append([float(np.mean((ms[l] - atl[v]["mean"][l]) ** 2)) for l in range(DEPTH)])
    ref_final = [r[-1] for r in ref_all]; sk_final = [r[-1] for r in sk_all]
    print("\n=== E1: depth-32 held-out, same probes ===")
    print(f"index-keyed (deployed): mean {np.mean(ref_final):.3e} median {np.median(ref_final):.3e}")
    print(f"state-keyed  (B={N_BASIS}): mean {np.mean(sk_final):.3e} median {np.median(sk_final):.3e}")
    wins = sum(1 for a, b in zip(sk_final, ref_final) if a < b)
    print(f"per-net wins (state-keyed better): {wins}/{VAL_N}")
    # depth-curve: a depth-d net == first d layers of a depth-32 net (iid layers),
    # so per-layer MSE doubles as a cross-depth (d<=32) comparison
    print("\nlayer-mse curve (val mean): layer  index-keyed  state-keyed")
    for l in range(0, DEPTH, 4):
        print(f"  L{l+1:>2}: {np.mean([r[l] for r in ref_all]):.3e}  {np.mean([r[l] for r in sk_all]):.3e}")
    print(f"  L{DEPTH}: {np.mean(ref_final):.3e}  {np.mean(sk_final):.3e}")

    out = {
        "config": {"train_n": TRAIN_N, "val_n": VAL_N, "n_probe": N_PROBE, "k_h": K_H,
                   "n_iter": NITER, "n_basis": N_BASIS,
                   "basis": "[1, q, q^2, abar, asd, rbar, q*abar, abar^2] from Spre/a/rho",
                   "n_params": NMF * N_BASIS + NVF * N_BASIS + NOF * N_BASIS},
        "history": history,
        "e1": {"index_keyed_final": ref_final, "state_keyed_final": sk_final,
               "index_keyed_mean": float(np.mean(ref_final)), "state_keyed_mean": float(np.mean(sk_final)),
               "state_keyed_wins": wins,
               "layer_curve_index": [float(np.mean([r[l] for r in ref_all])) for l in range(DEPTH)],
               "layer_curve_state": [float(np.mean([r[l] for r in sk_all])) for l in range(DEPTH)]},
        "theta_b64": base64.b64encode(struct.pack(
            "<%dd" % (NMF * N_BASIS + NVF * N_BASIS + NOF * N_BASIS),
            *np.concatenate([t.ravel() for t in theta]))).decode(),
        "runtime_s": round(time.time() - t0, 1),
    }
    Path(__file__).with_name("chain_state_keyed_results.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved chain_state_keyed_results.json ({time.time()-t0:.0f}s total)")
