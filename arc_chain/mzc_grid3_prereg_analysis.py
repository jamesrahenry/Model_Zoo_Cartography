"""Preregistered analysis of MZC grid 3 (Model_Zoo_Cartography/grid/PREREG3.md, sha256 fc7ea619...): H1-H5.

Written 2026-10-24.  ratio = re-randomised accuracy / full accuracy;  rho = mean over layers 3..depth-2 of ||Delta||_F / ||alpha W0||_F;
alpha_pred = exp(-lr*wd*(steps-250)).  Reference set for H5: grid-1 C=10 nets + grid-2 b4 rows (1024x16, GMM, C=10, not stalled).
usage: mzc_grid3_prereg_analysis.py GRID3_DIR [--grid1 DIR] [--grid2 DIR] [--out results.json]
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("grid3_dir")
ap.add_argument("--grid1", default=os.path.expanduser("~/rosetta_data/mzc_grid")); ap.add_argument("--grid2", default=os.path.expanduser("~/rosetta_data/mzc_grid2"))
ap.add_argument("--out"); a = ap.parse_args()
rng = np.random.default_rng(0)

def load(d, only=None):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*", "results", "*", "net_*.json"))):
        job = os.path.relpath(f, d).split(os.sep)[0]
        if only and not only(job): continue
        r = json.load(open(f)); D = r["depth"]; dec = r["decomposition"]; full = r["ablation"]["full"]["acc"]
        lr, steps = r.get("lr", 3e-4), r.get("steps", 20000)
        out.append(dict(job=job, C=r["C"], W=r["width"], D=D, task=r.get("task", "gmm"), wd=r["wd"], lr=lr, steps=steps, inc=r["outcome"] != "stalled",
                        ratio=r["ablation"]["rerandom"]["acc"] / max(full, 1e-9), full=full,
                        rho=float(np.mean([dec[l]["delta_rel"] for l in range(3, D - 1)])),
                        nla=float(lr * r["wd"] * max(steps - 250, 0))))                      # -ln alpha_pred
    return out

g3 = [n for n in load(a.grid3_dir) if n["inc"]]
for n in g3: n["T"] = n["lr"] * (n["steps"] - 250)
ref = [n for n in load(a.grid1) if n["C"] == 10 and n["inc"] and n["W"] == 1024 and n["D"] == 16 and n["task"] == "gmm"]
ref += [n for n in load(a.grid2, only=lambda j: "-b4-" in j) if n["inc"] and n["C"] == 10]
res = dict(n_grid3=len(g3), n_ref=len(ref))
sp = lambda x, y: float(spearmanr(x, y)[0]) if len(x) > 2 else None

# H1: training time lowers dependence at fixed decay level (two alpha groups)
grp = {"alpha0.55": {0.72, 0.1, 0.033}, "alpha0.17": {2.15, 0.3, 0.099}}
h1 = {k: dict(n=len([n for n in g3 if n["wd"] in v]), rho=sp([n["T"] for n in g3 if n["wd"] in v], [n["ratio"] for n in g3 if n["wd"] in v])) for k, v in grp.items()}
res["H1"] = dict(groups=h1, passed=bool(all(v["rho"] is not None and v["rho"] >= 0.7 for v in h1.values())))
# H2: decay lowers dependence at fixed training time (three T groups)
h2 = {}
for st in sorted({n["steps"] for n in g3}):
    S = [n for n in g3 if n["steps"] == st]; h2[st] = dict(n=len(S), rho=sp([n["nla"] for n in S], [n["ratio"] for n in S]))
res["H2"] = dict(groups=h2, n_pass=sum(v["rho"] is not None and v["rho"] >= 0.7 for v in h2.values()))
res["H2"]["passed"] = bool(res["H2"]["n_pass"] >= 2)
# H3: rho beats alpha
r_rho = sp([n["rho"] for n in g3], [n["ratio"] for n in g3]); r_al = sp([n["nla"] for n in g3], [n["ratio"] for n in g3])
res["H3"] = dict(spearman_rho=r_rho, spearman_neg_ln_alpha=r_al, difference=(r_rho - r_al) if None not in (r_rho, r_al) else None)
res["H3"]["passed"] = bool(res["H3"]["difference"] is not None and res["H3"]["difference"] >= 0.10)
# H4: cell means non-decreasing in mean rho (<= 1 adjacent inversion, each < 0.05)
cells = {}
for n in g3: cells.setdefault((n["steps"], n["wd"]), []).append(n)
cm = sorted([(float(np.mean([n["rho"] for n in v])), float(np.mean([n["ratio"] for n in v])), k) for k, v in cells.items()])
inv = [(cm[i][2], cm[i + 1][2], cm[i][1] - cm[i + 1][1]) for i in range(len(cm) - 1) if cm[i + 1][1] < cm[i][1]]
res["H4"] = dict(cells_by_rho=[dict(steps=c[2][0], wd=c[2][1], mean_rho=c[0], mean_ratio=c[1]) for c in cm], inversions=[dict(a=i[0], b=i[1], drop=i[2]) for i in inv],
                 passed=bool(len(cm) == 9 and len(inv) <= 1 and all(i[2] < 0.05 for i in inv)))

# H5: isotonic fit on the reference set, applied to grid 3
def pav(x, y):
    o = np.argsort(x); x, y = np.asarray(x)[o], np.asarray(y)[o]
    blocks = [[y[i], 1.0, x[i], x[i]] for i in range(len(x))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m, w = blocks[i], blocks[i + 1]; tot = m[1] + w[1]
            blocks[i] = [(m[0] * m[1] + w[0] * w[1]) / tot, tot, m[2], w[3]]; del blocks[i + 1]; i = max(i - 1, 0)
        else: i += 1
    xs = np.array([(b[2] + b[3]) / 2 for b in blocks]); ys = np.array([b[0] for b in blocks]); return xs, ys
def mae(key):
    xs, ys = pav([n[key] for n in ref], [n["ratio"] for n in ref])
    return float(np.mean([abs(np.interp(n[key], xs, ys) - n["ratio"]) for n in g3])) if g3 and len(ref) > 2 else None
mr, ma = mae("rho"), mae("nla")
res["H5"] = dict(mae_rho=mr, mae_neg_ln_alpha=ma, passed=bool(mr is not None and mr <= 0.15), rho_beats_alpha=bool(mr is not None and ma is not None and mr < ma))

complete = len(g3) >= 72 * 0.9
verdict = {h: res[h]["passed"] for h in ("H1", "H2", "H3", "H4", "H5")}
print(f"grid-3 nets {len(g3)} (reference {len(ref)}); complete={complete}")
print("FAILED FIRST:", [h for h, ok in verdict.items() if not ok] or "none", "| PASSED:", [h for h, ok in verdict.items() if ok] or "none")
for h in ("H1", "H2", "H3", "H5"): print(h + ":", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[h].items()}))
print("H4 cells by rho:", [(c["steps"], c["wd"], round(c["mean_rho"], 2), round(c["mean_ratio"], 2)) for c in res["H4"]["cells_by_rho"]], "inversions:", res["H4"]["inversions"], "->", res["H4"]["passed"])
if a.out: json.dump(res, open(a.out, "w"), indent=1)
