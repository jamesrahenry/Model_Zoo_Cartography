"""Q5 of PREREG2 plus an exploratory per-net look on grid 3: the frozen 331937 chain on the seed-0 nets of grids 2 and 3.

Written 2026-10-24.
Q5 (as preregistered, with the ambiguity of "setting" resolved here BEFORE looking: pool the GMM rows of B3 and B4 by weight-decay value):
  held-out coherent share at the last layer (leave-one-net-out per-layer fit e ~ a*m + b*1 over the nets of a setting; per-net widths allowed),
  vs mean -ln(alpha_pred) of the setting; pass if Spearman >= 0.6 over settings with >= 3 finite chain outputs.
Exploratory (not preregistered): per-net alignment share (e.m)^2/(|m|^2|e|^2) at the last layer vs rho and vs -ln(alpha_pred).
usage: mzc_grid_q5_chain.py [--out results.json]
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("--out"); a = ap.parse_args()
H = os.path.expanduser("~/rosetta_data")
recs = []
for wf in sorted(glob.glob(f"{H}/mzc_grid[23]/*/weights/*/net_*.npz")):
    run = os.path.basename(os.path.dirname(wf)); net = os.path.basename(wf)[:-4]; job = wf.split("/weights/")[0]
    pf = f"{H}/mzc_grid_q5/preds/pred_{run}.npy"
    if not os.path.exists(pf): continue
    meta = json.load(open(f"{job}/results/{run}/{net}.json")); tr = np.load(f"{job}/results/{run}/{net}_truth.npz")
    P = np.load(pf); T = tr["mean"].astype(np.float64); se2 = tr["se2"].astype(np.float64); D = meta["depth"]
    L = D - 1
    fin = bool(np.isfinite(P).all()); e = P[L] - T[L] if fin else None
    recs.append(dict(run=run, job=os.path.basename(job), task=meta.get("task", "gmm"), W=meta["width"], D=D, C=meta["C"], wd=meta["wd"], lr=meta.get("lr", 3e-4),
                     steps=meta.get("steps", 20000), finite=fin, P=P, T=T, se2=se2,
                     nla=meta.get("lr", 3e-4) * meta["wd"] * max(meta.get("steps", 20000) - 250, 0),
                     rho=float(np.mean([meta["decomposition"][l]["delta_rel"] for l in range(3, D - 1)])),
                     rel=float((((P[L] - T[L]) ** 2).sum() - se2[L].sum()) / (T[L] ** 2).sum()) if fin else None,
                     align=float((e @ P[L]) ** 2 / ((P[L] @ P[L]) * (e @ e))) if fin else None,
                     slope=float((P[L] @ e) / (P[L] @ P[L])) if fin else None))
res = dict(n=len(recs), nonfinite=[(r["job"][-22:], r["wd"]) for r in recs if not r["finite"]])

def loo_share(R):
    L = lambda r: r["D"] - 1
    num = den = 0.0
    for k in range(len(R)):
        A = np.zeros((2, 2)); b = np.zeros(2)
        for j, r in enumerate(R):
            if j == k: continue
            m = r["P"][L(r)]; e = m - r["T"][L(r)]; n = len(m); one = np.ones(n)
            A += np.array([[m @ m, m @ one], [m @ one, n]]); b += np.array([m @ e, one @ e])
        co = np.linalg.solve(A, b); r = R[k]; m = r["P"][L(r)]; e = m - r["T"][L(r)]; nz = r["se2"][L(r)].sum()
        num += ((e - co[0] * m - co[1]) ** 2).sum() - nz; den += e @ e - nz
    return 1 - num / den

gmm = [r for r in recs if r["task"] == "gmm" and ("grid2" in r["job"]) and r["finite"]]
sets = {}
for r in gmm: sets.setdefault(r["wd"], []).append(r)
q5 = {}
for wd, R in sorted(sets.items()):
    if len(R) >= 3: q5[wd] = dict(n=len(R), mean_nla=float(np.mean([r["nla"] for r in R])), loo_coherent=float(loo_share(R)), mean_rel_err=float(np.mean([r["rel"] for r in R])))
    else: q5[wd] = dict(n=len(R), note="fewer than 3 finite nets")
ok = {k: v for k, v in q5.items() if "loo_coherent" in v}
rho = float(spearmanr([v["mean_nla"] for v in ok.values()], [v["loo_coherent"] for v in ok.values()])[0]) if len(ok) >= 3 else None
res["Q5"] = dict(settings=q5, spearman=rho, passed=bool(rho is not None and rho >= 0.6))

ex = [r for r in recs if r["finite"]]
sp = lambda key, S: float(spearmanr([r[key] for r in S], [r["align"] for r in S])[0]) if len(S) > 3 else None
g3 = [r for r in ex if "grid3" in r["job"]]; g2 = [r for r in ex if "grid2" in r["job"] and r["task"] == "gmm" and r["W"] == 1024 and r["D"] == 16]
res["exploratory_alignment"] = {"grid3_1024x16_C10": dict(n=len(g3), rho_vs_rho=sp("rho", g3), vs_neg_ln_alpha=sp("nla", g3)),
                                "grid2_b4_1024x16": dict(n=len(g2), vs_rho=sp("rho", g2), vs_neg_ln_alpha=sp("nla", g2)),
                                "all_finite": dict(n=len(ex), vs_rho=sp("rho", ex), vs_neg_ln_alpha=sp("nla", ex))}
res["per_net"] = [dict(job=r["job"][-24:], wd=r["wd"], steps=r["steps"], W=r["W"], D=r["D"], task=r["task"], rho=round(r["rho"], 3), nla=round(r["nla"], 3),
                       rel=r["rel"], align=r["align"], slope=r["slope"]) for r in recs if r["finite"]]
print(f"nets {len(recs)}; non-finite chain output on {len(res['nonfinite'])}: {res['nonfinite']}")
print("Q5 (pooled by wd, GMM rows of grid 2):", json.dumps(res["Q5"]), "->", "PASS" if res["Q5"]["passed"] else "FAIL")
print("exploratory alignment-share Spearman:", json.dumps(res["exploratory_alignment"]))
if a.out: json.dump(res, open(a.out, "w"), indent=1, default=str)
