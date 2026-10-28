"""Build the A1 synthetic nets (phase2/A1_PREREG_SYNTHETIC_UPDATE.md) and their Monte Carlo truths.

Written 2026-10-23.  W_l = alpha*W0_l + Delta_l;  aligned Delta_l = sqrt(2) Q R_l Q^T (shared Q), misaligned Delta_l = sqrt(2) U_l R_l V_l^T.
usage: a1_synthetic_build.py OUT_DIR [--truth-n 4e6] [--no-truth]
"""
import argparse, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--truth-n", type=float, default=4e6); ap.add_argument("--no-truth", action="store_true"); ap.add_argument("--calibrate", action="store_true", help="A1b: per-layer variance-preserving rescale"); a = ap.parse_args()
n, D = 1024, 16
SETTINGS = {"S0": (1.0, 0, "none"), "S1": (0.18, 16, "aligned"), "S2": (0.18, 16, "misaligned"),
            "S3": (0.18, 64, "aligned"), "S4": (0.18, 4, "aligned"), "S5": (0.55, 16, "aligned")}
def orth(rng, k): return np.linalg.qr(rng.standard_normal((k, k)))[0]
def build(name, seed):
    alpha, r, kind = SETTINGS[name]; rng = np.random.default_rng(10_000 * int(name[1]) + seed)
    Q = np.linalg.qr(rng.standard_normal((n, max(r, 1))))[0][:, :max(r, 1)]
    W = []
    for l in range(D):
        W0 = rng.standard_normal((n, n)) * np.sqrt(2.0 / n); Wl = alpha * W0
        if kind == "aligned": Wl = Wl + np.sqrt(2) * Q @ orth(rng, r) @ Q.T
        elif kind == "misaligned":
            U = np.linalg.qr(rng.standard_normal((n, r)))[0]; V = np.linalg.qr(rng.standard_normal((n, r)))[0]
            Wl = Wl + np.sqrt(2) * U @ orth(rng, r) @ V.T
        W.append(Wl.astype(np.float32))
    if a.calibrate:                                   # A1b: scale each layer so post-ReLU rms = 1 on 4096 N(0,I) draws
        h = np.random.default_rng(777 + seed).standard_normal((4096, n)).astype(np.float32)
        for l in range(D):
            z = np.maximum(h @ W[l], 0); g = 1.0 / float(np.sqrt((z.astype(np.float64) ** 2).mean()))
            W[l] = (W[l] * np.float32(g)); h = z * np.float32(g)
    return W
os.makedirs(a.out, exist_ok=True)
for name in SETTINGS:
    for seed in range(3):
        f = os.path.join(a.out, f"{name}_s{seed}.npz")
        if not os.path.exists(f): np.savez(f, **{f"w{l}": w for l, w in enumerate(build(name, seed))})
print("built", len(os.listdir(a.out)))
if not a.no_truth:
    import torch
    sys.path.insert(0, os.path.expanduser("~/Source/Rosetta_Program/Model_Zoo_Cartography/grid"))
    from analyze_net import mc_truth
    torch.backends.cuda.matmul.allow_tf32 = False
    for name in SETTINGS:
        for seed in range(3):
            t = os.path.join(a.out, f"{name}_s{seed}_truth.npz")
            if os.path.exists(t): continue
            z = np.load(os.path.join(a.out, f"{name}_s{seed}.npz"))
            mean, se2 = mc_truth([torch.tensor(z[f"w{l}"], device="cuda") for l in range(D)], int(a.truth_n), seed=7, dev="cuda")
            np.savez(t, mean=mean, se2=se2); print("truth", name, seed, flush=True)
