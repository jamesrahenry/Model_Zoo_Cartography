"""P4 and P5 of the preregistered MZC grid (seed-0 nets, frozen 331937 chain run on the laptop CPU).

Written 2026-10-23.  P4: held-out coherent share of the chain's layer-15 error (leave-one-net-out per-layer fit e ~ a*m + b*1,
pooled over C within a weight-decay setting) vs weight decay.  P5: fraction of nets on which the chain returns non-finite values,
C <= 3 vs C >= 10 (wd >= 0.003).  Thresholds are those in grid/PREREG.md.
usage: mzc_grid_p4p5_analysis.py GRID_DIR PRED_DIR [--out results.json]    (PRED_DIR holds pred_<job id>_<run id>.npy per seed-0 net)
"""
import argparse, glob, json, os, re
import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("grid_dir"); ap.add_argument("pred_dir"); ap.add_argument("--out"); a = ap.parse_args()
recs = []
for f in sorted(glob.glob(os.path.join(a.pred_dir, "pred_*.npy"))):
    run = re.sub(r"^pred_|\.npy$", "", os.path.basename(f))                                   # grid_wd<tag>_c<C>
    m = re.match(r"grid_wd(.+)_c(\d+)$", run); wd = float(m.group(1).replace("p", ".")); C = int(m.group(2))
    t = glob.glob(os.path.join(a.grid_dir, "*", "results", run, "net_0000_truth.npz"))[0]
    z = np.load(t); P = np.load(f)
    recs.append(dict(run=run, wd=wd, C=C, P=P, T=z["mean"].astype(np.float64), se2=z["se2"].astype(np.float64), finite=bool(np.isfinite(P).all())))
n = recs[0]["T"].shape[1]; ones = np.ones(n)

def loo_share(R, l):
    num = den = 0.0
    for k in range(len(R)):
        A = np.zeros((2, 2)); b = np.zeros(2)
        for j in range(len(R)):
            if j == k: continue
            m = R[j]["P"][l]; e = m - R[j]["T"][l]
            A += np.array([[m @ m, m @ ones], [m @ ones, n]]); b += np.array([m @ e, ones @ e])
        co = np.linalg.solve(A, b); m = R[k]["P"][l]; e = m - R[k]["T"][l]; nz = R[k]["se2"][l].sum()
        num += ((e - co[0] * m - co[1]) ** 2).sum() - nz; den += (e @ e) - nz
    return 1 - num / den

res = {"P4": {}, "P5": {}}
for wd in sorted({r["wd"] for r in recs}):
    R = [r for r in recs if r["wd"] == wd and r["finite"]]
    if len(R) >= 3:
        res["P4"][wd] = dict(n=len(R), Cs=[r["C"] for r in R], loo_L15=float(loo_share(R, 15)),
                             loo_mean_L8_15=float(np.mean([loo_share(R, l) for l in range(8, 16)])),
                             mean_rel_err_L15=float(np.mean([((r["P"][15] - r["T"][15]) ** 2).sum() / (r["T"][15] ** 2).sum() for r in R])))
    else: res["P4"][wd] = dict(n=len(R), note="fewer than 3 finite nets")
w = [k for k, v in res["P4"].items() if "loo_L15" in v]; s = [res["P4"][k]["loo_L15"] for k in w]
rho = float(spearmanr(w, s)[0]) if len(w) > 2 else None
hi = [res["P4"][k]["loo_L15"] for k in w if k >= 0.1]
res["P4_verdict"] = dict(spearman_wd_vs_share=rho, shares_at_wd_ge_0p1=hi,
                         passed=bool(rho is not None and rho >= 0.7 and all(x > 0.5 for x in hi) and len(hi) > 0))
low = [r for r in recs if r["C"] <= 3 and r["wd"] >= 0.003]; high = [r for r in recs if r["C"] >= 10 and r["wd"] >= 0.003]
f = lambda L: (sum(not r["finite"] for r in L), len(L))
res["P5"] = dict(C_le_3=f(low), C_ge_10=f(high), nonfinite_by_C={int(C): f([r for r in recs if r["C"] == C]) for C in sorted({r["C"] for r in recs})},
                 first_bad_layers={f"wd={r['wd']},C={r['C']}": int(np.where(~np.isfinite(r["P"]).all(axis=1))[0][0]) for r in recs if not r["finite"]},
                 passed=bool(f(low)[0] / max(1, f(low)[1]) > f(high)[0] / max(1, f(high)[1])))
print("P4 per wd (n, C's, held-out coherent share L15, mean over L8-15, mean relative error L15):")
for k, v in res["P4"].items(): print("  wd", k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}))
print("P4 verdict:", json.dumps(res["P4_verdict"]), "->", "PASS" if res["P4_verdict"]["passed"] else "FAIL")
print("P5:", json.dumps(res["P5"]), "->", "PASS" if res["P5"]["passed"] else "FAIL")
if a.out: json.dump(res, open(a.out, "w"), indent=1)
