"""Preregistered analysis of MZC grid 2 (Model_Zoo_Cartography/grid/PREREG2.md, sha256 3d7b9bc1...): Q1-Q4 here, Q5 in the chain script.

Written 2026-10-23.  Thresholds, groups and the inclusion rule are those in PREREG2.md.  Deviations are printed with the result:
  D1  "alpha(layer 8)": for depth-8 nets layer 8 does not exist; layer min(8, depth//2) is used (layer 4 at depth 8, 8 at depth 16/24).
  D2  a group with < 3 qualifying nets for Q4 is reported as not evaluable, not silently passed.
usage: mzc_grid2_prereg_analysis.py GRID2_DIR [--ref GRID1_DIR] [--out results.json]
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("grid2_dir"); ap.add_argument("--ref", default=os.path.expanduser("~/rosetta_data/mzc_grid"))
ap.add_argument("--out"); a = ap.parse_args()
rng = np.random.default_rng(0)
KS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def kneed(d, full):
    for k in KS:
        if str(k) in d and d[str(k)] >= 0.95 * full: return k
    return None

def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*", "results", "*", "net_*.json"))):
        r = json.load(open(f)); full = r["ablation"]["full"]["acc"]; D = r["depth"]; L = min(8, D // 2)
        out.append(dict(job=os.path.relpath(f, d).split(os.sep)[0], task=r["task"], width=r["width"], depth=D, wd=r["wd"], lr=r.get("lr", 3e-4),
                        steps=r.get("steps", 20000), C=r["C"], inc=r["outcome"] != "stalled", outcome=r["outcome"], full=full,
                        rer=r["ablation"]["rerandom"]["acc"], alpha=r["decomposition"][L]["alpha"],
                        kal=kneed(r["slice"]["aligned_rerand"], full), krd=kneed(r["slice"]["random_rerand"], full)))
    return out

nets = load(a.grid2_dir)
for n in nets:
    n["apred"] = float(np.exp(-n["lr"] * n["wd"] * max(n["steps"] - 250, 0)))
    n["group"] = n["task"].split("_")[0] if n["task"] != "gmm" else f"gmm_w{n['width']}d{n['depth']}"
    n["row"] = __import__("re").sub(r"^grid2-\d+-", "", n["job"].split("-", 3)[-1])               # e.g. "b2-mnist-w512d16", "b4-lr1e-3"
inc = [n for n in nets if n["inc"]]
res = dict(n_total=len(nets), n_included=len(inc), excluded=[(n["job"], n["wd"], n["outcome"]) for n in nets if not n["inc"]][:20],
           rows=sorted({n["row"] for n in nets}))

# ---- Q1: surviving amplitude follows exp(-lr*wd*(steps-250))
pos = [n for n in inc if n["wd"] > 0]; z = [n for n in inc if n["wd"] == 0]
err = np.array([abs(n["alpha"] - n["apred"]) for n in pos])
res["Q1"] = dict(n_wd_pos=len(pos), frac_within_0p03=float((err <= 0.03).mean()) if len(pos) else None, max_err=float(err.max()) if len(pos) else None,
                 n_wd0=len(z), wd0_alpha_ge_0p97=float(np.mean([n["alpha"] >= 0.97 for n in z])) if z else None)
res["Q1"]["passed"] = bool(len(pos) and res["Q1"]["frac_within_0p03"] >= 0.9)
res["Q1"]["worst"] = sorted([(round(abs(n["alpha"] - n["apred"]), 3), n["row"], n["wd"], round(n["alpha"], 3), round(n["apred"], 3)) for n in pos], reverse=True)[:5]

# ---- Q2: Spearman(wd, rerandom acc) per group, 5 groups
groups = ["gmm_w512d16", "gmm_w1024d8", "gmm_w1024d24", "mnist", "fashion"]
q2 = {}
for g in groups:
    S = [n for n in inc if n["group"] == g and n["lr"] == 3e-4 and n["steps"] == 20000]
    if len(S) >= 3:
        x = np.array([n["wd"] for n in S]); y = np.array([n["rer"] for n in S])
        rho = float(spearmanr(x, y)[0])
        v = [spearmanr(x[i], y[i])[0] for i in (rng.choice(len(S), len(S)) for _ in range(2000))]
        q2[g] = dict(n=len(S), rho=rho, ci=[float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))],
                     mean_rerandom_by_wd={float(w): float(np.mean([n["rer"] for n in S if n["wd"] == w])) for w in sorted(set(x))},
                     mean_full=float(np.mean([n["full"] for n in S])))
res["Q2"] = dict(groups=q2, n_pass=sum(v["rho"] >= 0.7 for v in q2.values()), n_groups=len(q2))
res["Q2"]["passed"] = bool(res["Q2"]["n_pass"] >= 4 and len(q2) == 5)

# ---- Q3: effective decay controls dependence
ref = []
if os.path.isdir(a.ref):
    for f in glob.glob(os.path.join(a.ref, "*", "results", "*", "net_*.json")):
        r = json.load(open(f))
        if r["C"] == 10 and r["outcome"] != "stalled":
            ref.append(dict(wd=r["wd"], lr=3e-4, steps=20000, rer=r["ablation"]["rerandom"]["acc"], full=r["ablation"]["full"]["acc"]))
b4 = [n for n in inc if n["row"].startswith("b4-")]
pool = [(-np.log(max(np.exp(-n["lr"] * n["wd"] * max(n["steps"] - 250, 0)), 1e-12)), n["rer"]) for n in b4 + ref]
rho3 = float(spearmanr([p[0] for p in pool], [p[1] for p in pool])[0]) if len(pool) > 3 else None
b4rows = {}
for n in b4:
    if n["wd"] == 0: b4rows.setdefault(n["row"], []).append(n["rer"] / max(n["full"], 1e-9))
res["Q3"] = dict(n_pool=len(pool), n_ref=len(ref), n_b4=len(b4), a_spearman=rho3, a_ok=bool(rho3 is not None and rho3 >= 0.85),
                 b_wd0_ratio_by_row={k: float(np.mean(v)) for k, v in b4rows.items()}, b_ok=bool(b4rows and all(np.mean(v) <= 0.5 for v in b4rows.values())),
                 mean_rerandom_by_row_wd={f"{r}|wd={w}": float(np.mean([n["rer"] for n in b4 if n["row"] == r and n["wd"] == w]))
                                          for r in sorted({n["row"] for n in b4}) for w in sorted({n["wd"] for n in b4 if n["row"] == r})})
res["Q3"]["passed"] = bool(res["Q3"]["a_ok"] and res["Q3"]["b_ok"])

# ---- Q4: slice stays small, per row group
q4 = {}
for row in sorted({n["row"] for n in nets}):
    S = [n for n in inc if n["row"] == row and n["rer"] < 0.5 * n["full"] and n["kal"] is not None and n["krd"] is not None]
    real = row.startswith("b2-"); cap = 128 if real else 64
    if len(S) < 3:
        q4[row] = dict(n=len(S), evaluable=False); continue
    ka = float(np.median([n["kal"] for n in S])); kr = float(np.median([n["krd"] for n in S]))
    q4[row] = dict(n=len(S), evaluable=True, median_kcrit=ka, cap=cap, median_krandom=kr, ratio=kr / max(ka, 1.0), ok=bool(ka <= cap and kr >= 4 * ka))
ev = [v for v in q4.values() if v["evaluable"]]
res["Q4"] = dict(rows=q4, n_evaluable=len(ev), n_rows=len(q4), passed=bool(ev and all(v["ok"] for v in ev) and len(ev) == len(q4)))

complete = {"Q1": len(nets) > 0, "Q2": len(q2) == 5, "Q3": len({n["row"] for n in b4}) == 4, "Q4": len(q4) == 9}
verdict = {q: (res[q]["passed"] if complete[q] else None) for q in ("Q1", "Q2", "Q3", "Q4")}
print(f"nets {res['n_total']} analysed, {res['n_included']} included; rows present: {res['rows']}")
print("FAILED FIRST:", [q for q, ok in verdict.items() if ok is False] or "none", "| PASSED:", [q for q, ok in verdict.items() if ok] or "none",
      "| INCOMPLETE (rows/groups still missing):", [q for q, ok in verdict.items() if ok is None] or "none")
print("Q1:", json.dumps(res["Q1"]))
print("Q2:", json.dumps({k: (round(v["rho"], 2), v["n"], v["mean_rerandom_by_wd"]) for k, v in q2.items()}), f"{res['Q2']['n_pass']}/{res['Q2']['n_groups']} groups >= 0.7")
print("Q3:", json.dumps({k: v for k, v in res["Q3"].items() if k != "mean_rerandom_by_row_wd"})); print("   b4 mean rerandom by row|wd:", json.dumps({k: round(v, 2) for k, v in res["Q3"]["mean_rerandom_by_row_wd"].items()}))
print("Q4:", json.dumps(q4))
print("Q5: chain on seed-0 nets, see mzc_grid2_q5_chain.py (pooled by wd across B3+B4 rows; PREREG2 left the setting unit ambiguous)")
if a.out: json.dump(res, open(a.out, "w"), indent=1, default=str)
