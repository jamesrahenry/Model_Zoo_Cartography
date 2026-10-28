"""Preregistered analysis of MZC grid 4 (Model_Zoo_Cartography/grid/PREREG4.md, sha256 018e9b63...): K1-K4.

Written 2026-10-24.  Per cell (steps, wd): held-out coherent share of the frozen chain's last-layer error (leave-one-net-out per-layer fit e ~ a*m + b*1,
noise-debiased), mean -ln alpha_pred and mean rho.  usage: mzc_grid4_prereg_analysis.py [--out results.json]
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("--out"); a = ap.parse_args()
H = os.path.expanduser("~/rosetta_data"); rng = np.random.default_rng(0)
recs = []
for wf in sorted(glob.glob(f"{H}/mzc_grid4/*/weights/*/net_*.npz")):
    run = os.path.basename(os.path.dirname(wf)); net = os.path.basename(wf)[:-4]; job = wf.split("/weights/")[0]
    pf = f"{H}/mzc_grid4_chain/preds/pred_{run}_{net}.npy"
    mj = f"{job}/results/{run}/{net}.json"
    if not (os.path.exists(pf) and os.path.exists(mj)): continue
    meta = json.load(open(mj)); tr = np.load(f"{job}/results/{run}/{net}_truth.npz"); P = np.load(pf); D = meta["depth"]
    if meta["outcome"] == "stalled": continue
    recs.append(dict(steps=meta["steps"], wd=meta["wd"], D=D, P=P, T=tr["mean"].astype(np.float64), se2=tr["se2"].astype(np.float64), finite=bool(np.isfinite(P).all()),
                     nla=meta["lr"] * meta["wd"] * max(meta["steps"] - 250, 0), rho=float(np.mean([meta["decomposition"][l]["delta_rel"] for l in range(3, D - 1)]))))
def share(R):
    num = den = 0.0
    for k in range(len(R)):
        A = np.zeros((2, 2)); b = np.zeros(2)
        for j, r in enumerate(R):
            if j == k: continue
            m = r["P"][-1]; e = m - r["T"][-1]; n = len(m); one = np.ones(n); A += np.array([[m @ m, m @ one], [m @ one, n]]); b += np.array([m @ e, one @ e])
        co = np.linalg.solve(A, b); r = R[k]; m = r["P"][-1]; e = m - r["T"][-1]; nz = r["se2"][-1].sum()
        num += ((e - co[0] * m - co[1]) ** 2).sum() - nz; den += e @ e - nz
    return 1 - num / den
cells = {}
for r in recs: cells.setdefault((r["steps"], r["wd"]), []).append(r)
out = {}
for k, v in sorted(cells.items()):
    fin = [r for r in v if r["finite"]]
    d = dict(n=len(v), finite=len(fin), nonfinite_frac=1 - len(fin) / len(v), nla=float(np.mean([r["nla"] for r in v])), rho=float(np.mean([r["rho"] for r in v])))
    if len(fin) >= 6:
        d["share"] = float(share(fin)); bs = [share([fin[i] for i in rng.choice(len(fin), len(fin))]) for _ in range(300)]
        d["ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out[f"steps={k[0]},wd={k[1]}"] = d
ev = {k: v for k, v in out.items() if "share" in v}
sp = lambda key: float(spearmanr([v[key] for v in ev.values()], [v["share"] for v in ev.values()])[0]) if len(ev) > 3 else None
r_a, r_r = sp("nla"), sp("rho")
K = {}
K["K1"] = dict(n_cells=len(ev), spearman_neg_ln_alpha=r_a, spearman_rho=r_r, evaluable=len(ev) == 9, passed=bool(len(ev) == 9 and r_a >= 0.7 and r_a - r_r >= 0.2))
k2 = [v for k, v in out.items() if k.endswith("wd=0.0") or k.endswith("wd=0")]; k3 = [v for k, v in out.items() if abs(v["nla"] - 1.77) < 0.2 and v["nla"] > 0]
K["K2"] = dict(cells=[(v.get("share")) for v in k2], evaluable=bool(len(k2) == 3 and all("share" in v for v in k2)), passed=bool(len(k2) == 3 and all("share" in v and v["share"] < 0.3 for v in k2)))
K["K3"] = dict(cells=[(v.get("share")) for v in k3], evaluable=bool(len(k3) == 3 and all("share" in v for v in k3)), passed=bool(len(k3) == 3 and all("share" in v and v["share"] >= 0.5 for v in k3)))
K["K4"] = dict(max_nonfinite_frac=max((v["nonfinite_frac"] for v in out.values()), default=None), passed=bool(out and all(v["nonfinite_frac"] <= 0.2 for v in out.values())))
res = dict(n_nets=len(recs), cells=out, K=K)
print(f"nets {len(recs)}; cells {len(out)}; cells with >= 6 finite outputs: {len(ev)}")
print("FAILED FIRST:", [k for k, v in K.items() if not v["passed"]] or "none", "| PASSED:", [k for k, v in K.items() if v["passed"]] or "none")
for k, v in out.items(): print(f"  {k:22s} n={v['n']} finite={v['finite']} -ln a={v['nla']:.2f} rho={v['rho']:.2f} share={v.get('share')} ci={v.get('ci')}")
for k, v in K.items(): print(k, json.dumps(v))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
