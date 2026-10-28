"""Trained vs own-initialisation: coherence anatomy of the frozen 331937 chain's per-layer error (MZC nets, 1024x16).

Written 2026-10-20.  Oracle diagnostic only.  For each MZC net the corpus stores both the trained weights (`w*`)
and the He-Gaussian initialisation they started from (`init_w*`), so each trained net has a paired random-init control.
Inputs per (family, net, key): pred_<fam>_<i>_<key>.npy (16,1024) and truth_<fam>_<i>_<key>.npz (mean, se2, N).
Per layer, with e = pred - truth_mean (se2 noise debiased out of every squared quantity):
  rel      debiased ||e||^2 / ||truth||^2
  common   share of ||e||^2 along the all-ones direction
  align    share along the prediction vector
  loo      held-out share of ||e||^2 removed by a per-layer pooled fit e ~ a*m + b*1 learned on OTHER nets
Usage: python3 mzc_trained_vs_init_anatomy.py DIR [--out results.json]
"""
import argparse, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--out"); a = ap.parse_args()
recs = {}; nonfinite = []
for f in sorted(glob.glob(os.path.join(a.dir, "pred_*.npy"))):
    m = re.match(r"pred_([a-z0-9]+)_(\d+)_(w|init_w)\.npy", os.path.basename(f))
    t = f.replace("pred_", "truth_").replace(".npy", ".npz")
    if m and os.path.exists(t):
        z = np.load(t); Pn = np.load(f)
        if not np.isfinite(Pn).all():                       # chain emitted NaN/inf: reported, excluded from the anatomy
            first = int(np.argmax(~np.isfinite(Pn).all(axis=1) & True)) if False else int(np.where(~np.isfinite(Pn).all(axis=1))[0][0])
            nonfinite.append(dict(fam=m.group(1), net=int(m.group(2)), key=m.group(3), first_bad_layer=first)); continue
        recs.setdefault(m.group(3), []).append(
            dict(fam=m.group(1), net=int(m.group(2)), P=np.load(f), T=z["mean"], se2=z["se2"]))
L, n = 16, 1024

def anatomy(R):
    N = len(R)
    P = np.stack([r["P"] for r in R]); T = np.stack([r["T"] for r in R]); S = np.stack([r["se2"] for r in R])
    E = P - T
    noise = S.sum(-1)                                               # (N,L) variance of ||noise||^2 expectation
    sq = np.maximum((E ** 2).sum(-1) - noise, 1e-30)                # debiased ||e||^2
    rel = sq / np.maximum((T ** 2).sum(-1) - noise, 1e-30)
    ones = np.ones(n)
    comm = (E.sum(-1) ** 2 / n) / (sq + noise)
    algn = ((E * P).sum(-1) ** 2 / (P ** 2).sum(-1)) / (sq + noise)
    loo = np.zeros(L)
    for l in range(L):
        num = den = 0.0
        for k in range(N):
            A = np.zeros((2, 2)); b = np.zeros(2)
            for j in range(N):
                if j == k: continue
                m = P[j, l]; e = E[j, l]
                A += np.array([[m @ m, m @ ones], [m @ ones, n]]); b += np.array([m @ e, ones @ e])
            co = np.linalg.solve(A, b)
            num += ((E[k, l] - co[0] * P[k, l] - co[1]) ** 2).sum() - noise[k, l]; den += sq[k, l]
        loo[l] = 1 - num / den
    return dict(n_nets=N, mse=(sq / n).mean(0).tolist(), rel=rel.mean(0).tolist(), common=comm.mean(0).tolist(),
                align=algn.mean(0).tolist(), loo_share=loo.tolist())

def transfer(R):
    """Held-out-family test: per-layer pooled fit e ~ a*m + b*1 on all OTHER families (and, for wd0 families, all
    non-wd0 families only and vice versa is reported by family name), applied to the held-out family's nets."""
    res = {}
    fams = sorted({r["fam"] for r in R})
    ones = np.ones(n)
    for F in fams:
        tr = [r for r in R if r["fam"] != F]; te = [r for r in R if r["fam"] == F]
        if not tr or not te: continue
        before = np.zeros(L); after = np.zeros(L)
        for l in range(L):
            A = np.zeros((2, 2)); b = np.zeros(2)
            for r in tr:
                m = r["P"][l]; e = m - r["T"][l]
                A += np.array([[m @ m, m @ ones], [m @ ones, n]]); b += np.array([m @ e, ones @ e])
            co = np.linalg.solve(A, b)
            for r in te:
                m = r["P"][l]; e = m - r["T"][l]; nz = r["se2"][l].sum()
                before[l] += (e @ e - nz) / n / len(te); after[l] += ((e - co[0] * m - co[1]) ** 2).sum() / n / len(te) - nz / n / len(te)
        res[F] = dict(n_test=len(te), mse_before=before.tolist(), mse_after=after.tolist(),
                      ratio_L7=float(after[7] / before[7]), ratio_L15=float(after[15] / before[15]),
                      ratio_mean_L3_15=float(after[3:].mean() / before[3:].mean()))
    return res

if "w" in recs:                                           # split the trained nets by weight-decay arm
    arms = {"w_wd0.3": [r for r in recs["w"] if not r["fam"].startswith("wd0")],
            "w_wd0": [r for r in recs["w"] if r["fam"].startswith("wd0")]}
    recs.update({k: v for k, v in arms.items() if len(v) > 1})
out = {k: anatomy(v) for k, v in recs.items()}
if "w" in recs:
    out["transfer_heldout_family"] = transfer(recs["w"])
    print("== held-out-family per-layer scale/shift correction (fit on other families), MSE after/before")
    for F, r in out["transfer_heldout_family"].items():
        print(f"  {F:8s} n={r['n_test']}  L7 {r['ratio_L7']:.3f}  L15 {r['ratio_L15']:.3f}  mean(L3-15) {r['ratio_mean_L3_15']:.3f}")
fm = lambda v: " ".join(f"{x:8.2e}" if abs(x) < 1e-2 and x != 0 else f"{x:8.3f}" for x in v)
for k in ("init_w", "w", "w_wd0.3", "w_wd0"):
    if k in recs:
        r = out[k]; print(f"== {k}  N={r['n_nets']} nets   layers 0..15")
        for key in ("mse", "rel", "common", "align", "loo_share"): print(f"{key:9s}", fm(r[key]))
out["nonfinite_chain_output"] = nonfinite
print("chain emitted non-finite values on", len(nonfinite), "nets:", sorted({(r["fam"]) for r in nonfinite}),
      "first bad layers:", sorted({r["first_bad_layer"] for r in nonfinite}))
if a.out: json.dump(out, open(a.out, "w"), indent=1)
