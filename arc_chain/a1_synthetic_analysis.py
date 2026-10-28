"""A1 analysis (phase2/A1_PREREG_SYNTHETIC_UPDATE.md): frozen chain on synthetic aligned / misaligned low-rank updates.

Written 2026-10-23.  usage: a1_synthetic_analysis.py DIR [--out results.json]   (DIR holds S?_s?_truth.npz and preds/pred_S?_s?.npy)
"""
import argparse, glob, json, os, re
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--out"); ap.add_argument("--gate", action="store_true", help="A1b regime gate: final-layer truth mean rms in [0.3, 2.0]"); a = ap.parse_args()
recs = []; gated = []
for f in sorted(glob.glob(os.path.join(a.dir, "preds", "pred_S?_s?.npy"))):
    b = re.match(r"pred_(S\d)_s(\d)\.npy", os.path.basename(f)); t = os.path.join(a.dir, f"{b.group(1)}_s{b.group(2)}_truth.npz")
    if not os.path.exists(t): continue
    z = np.load(t); P = np.load(f)
    if a.gate and not (0.3 <= float(np.sqrt((z["mean"][15] ** 2).mean())) <= 2.0): gated.append((b.group(1), int(b.group(2)))); continue
    recs.append(dict(s=b.group(1), seed=int(b.group(2)), P=P, T=z["mean"], se2=z["se2"], finite=bool(np.isfinite(P).all())))
n = 1024; ones = np.ones(n)
def fit(R, l):
    A = np.zeros((2, 2)); b = np.zeros(2)
    for r in R:
        m = r["P"][l]; e = m - r["T"][l]; A += np.array([[m @ m, m @ ones], [m @ ones, n]]); b += np.array([m @ e, ones @ e])
    return np.linalg.solve(A, b)
def loo(R, l):
    num = den = 0.0
    for k in range(len(R)):
        co = fit([r for j, r in enumerate(R) if j != k], l); m = R[k]["P"][l]; e = m - R[k]["T"][l]; nz = R[k]["se2"][l].sum()
        num += ((e - co[0] * m - co[1]) ** 2).sum() - nz; den += e @ e - nz
    return 1 - num / den
out = {}
for s in sorted({r["s"] for r in recs}):
    R = [r for r in recs if r["s"] == s and r["finite"]]; nb = sum(1 for r in recs if r["s"] == s and not r["finite"])
    d = dict(n=len(R), nonfinite=nb)
    if R:
        d["rel_err_L15"] = float(np.mean([(((r["P"][15] - r["T"][15]) ** 2).sum() - r["se2"][15].sum()) / (r["T"][15] ** 2).sum() for r in R]))
        d["slope_a_L15"] = float(fit(R, 15)[0])
        d["loo_coherent_L15"] = float(loo(R, 15)) if len(R) >= 2 else None
    out[s] = d
g = lambda s, k: out.get(s, {}).get(k)
v = {}
v["P1"] = bool(g("S1", "rel_err_L15") is not None and g("S1", "rel_err_L15") >= 1e-3)
v["P2"] = bool(g("S1", "loo_coherent_L15") is not None and g("S1", "loo_coherent_L15") >= 0.5 and -0.20 <= g("S1", "slope_a_L15") <= -0.05)
v["P3"] = bool(g("S2", "loo_coherent_L15") is not None and g("S2", "loo_coherent_L15") < 0.3 and g("S2", "rel_err_L15") <= 0.1 * g("S1", "rel_err_L15"))
tri = [g(s, "rel_err_L15") for s in ("S4", "S1", "S3")]                                   # rank 4, 16, 64
v["P4_direction"] = ("increasing in r" if tri == sorted(tri) else "decreasing in r" if tri == sorted(tri, reverse=True) else "non-monotone") if None not in tri else "incomplete"
res = dict(settings=out, verdict=v, gated_out=gated)
print("setting  n  nonfinite  rel_err_L15  slope_a  held-out coherent share (L15)")
for s, d in out.items(): print(f"  {s}  {d['n']}  {d['nonfinite']}  {d.get('rel_err_L15', float('nan')):.3e}  {d.get('slope_a_L15', float('nan')):+.3f}  {d.get('loo_coherent_L15')}")
print("regime-gated out:", gated); print("verdict:", json.dumps(v))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
