"""Preregistered analysis of the MZC weight-decay x task-size grid (P1-P3; P4/P5 need the laptop chain, see below).

Written 2026-10-23.  Statistics, thresholds and inclusion rule are exactly those in
Model_Zoo_Cartography/grid/PREREG.md (commit b385b89, sha256 0a377f84...). Nothing here is tuned to grid data.
usage: mzc_grid_prereg_analysis.py GRID_DIR [--out results.json]    (GRID_DIR holds one fetched job folder per row)
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("grid_dir"); ap.add_argument("--out")
ap.add_argument("--amend", action="store_true", help="DEVIATION from PREREG: inclusion = outcome != stalled (the preregistered 2-x-chance rule is unsatisfiable for C = 2)")
a = ap.parse_args()
rng = np.random.default_rng(0)
rows = []
for f in sorted(glob.glob(os.path.join(a.grid_dir, "*", "results", "*", "net_*.json"))):
    r = json.load(open(f)); rows.append(r)
KS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def kneed(d, full):
    for k in KS:
        if d[str(k)] >= 0.95 * full: return k
    return None

net = []
for r in rows:
    inc = (r["outcome"] != "stalled") if a.amend else ((r["final_val_acc"] >= 2 * r["chance"]) and r["outcome"] != "stalled")
    full = r["ablation"]["full"]["acc"]
    d8 = r["decomposition"][8]
    net.append(dict(C=r["C"], wd=r["wd"], inc=inc, outcome=r["outcome"], full=full, rer=r["ablation"]["rerandom"]["acc"],
                    kal=kneed(r["slice"]["aligned_rerand"], full), krd=kneed(r["slice"]["random_rerand"], full),
                    share8=1 / (1 + d8["delta_rel"] ** 2), name=r["net"], run=r["run_id"]))

def boot(fn, idx, n=2000):
    v = []
    for _ in range(n):
        s = rng.choice(idx, len(idx)); v.append(fn(s))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

inc = [n for n in net if n["inc"]]
res = dict(rule=("AMENDED (outcome != stalled)" if a.amend else "PREREGISTERED (final acc >= 2 x chance, not stalled)"), n_total=len(net), n_included=len(inc))
cells = {}
for n in net: cells.setdefault((n["wd"], n["C"]), []).append(n)
res["exclusions_per_cell"] = {f"wd={k[0]},C={k[1]}": dict(n=len(v), excluded=sum(not x["inc"] for x in v),
                                                          outcomes=sorted({x["outcome"] for x in v}))
                              for k, v in sorted(cells.items()) if any(not x["inc"] for x in v)}
out = []

# ---- P1
p1 = {}
for C in sorted({n["C"] for n in inc}):
    S = [n for n in inc if n["C"] == C]
    x = np.array([n["wd"] for n in S]); y = np.array([n["rer"] for n in S])
    rho = float(spearmanr(x, y)[0]); lo, hi = boot(lambda s: spearmanr(x[s], y[s])[0], np.arange(len(S)))
    p1[C] = dict(n=len(S), rho=rho, ci=[lo, hi])
res["P1"] = dict(per_C=p1, n_pass=sum(v["rho"] >= 0.8 for v in p1.values()), passed=sum(v["rho"] >= 0.8 for v in p1.values()) >= 4)
res["P1"]["mean_rerandom_acc_by_wd_C"] = {f"wd={wd},C={C}": float(np.mean([n["rer"] for n in v if n["inc"]]))
                                           for (wd, C), v in sorted(cells.items()) if any(n["inc"] for n in v)}

# ---- P2
S = [n for n in inc if n["wd"] <= 0.03 and n["rer"] < 0.5 * n["full"] and n["kal"] is not None and n["krd"] is not None]
Cs = np.array([n["C"] for n in S]); ka = np.array([n["kal"] for n in S], float); kr = np.array([n["krd"] for n in S], float)
rho_a = float(spearmanr(Cs, ka)[0]) if len(S) > 2 else None
a_ok = rho_a is not None and rho_a > 0
med_ratio = float(np.median(ka / Cs)); b_ok = 1 <= med_ratio <= 8
rr = float(np.median(kr) / np.median(ka)); c_ok = rr >= 8
res["P2"] = dict(n=len(S), a_spearman_C_kcrit=rho_a, a_ok=a_ok, b_median_kcrit_over_C=med_ratio, b_ok=b_ok,
                 c_ratio_of_medians=rr, c_per_net_median_ratio=float(np.median(kr / ka)), c_ok=c_ok,
                 median_kcrit_by_C={int(C): float(np.median(ka[Cs == C])) for C in sorted(set(Cs))},
                 median_krandom_by_C={int(C): float(np.median(kr[Cs == C])) for C in sorted(set(Cs))},
                 passed=bool(a_ok and b_ok and c_ok))

# ---- P3
sh = np.array([n["share8"] for n in inc]); rer = np.array([n["rer"] for n in inc]); wd = np.array([n["wd"] for n in inc])
r_e = float(spearmanr(sh, rer)[0]); r_w = float(spearmanr(wd, rer)[0])
by_wd = {float(w): float(np.mean([n["share8"] for n in inc if n["wd"] == w])) for w in sorted({n["wd"] for n in inc})}
near = all(abs(v - 0.92) <= 0.05 for w, v in by_wd.items() if w <= 0.1)
res["P3"] = dict(spearman_energy_share_vs_rerandom=r_e, spearman_wd_vs_rerandom=r_w, mean_layer8_share_by_wd=by_wd,
                 within_0p05_of_0p92_for_wd_le_0p1=bool(near), passed=bool(abs(r_e) < abs(r_w) and near))
res["P4_P5"] = "run on the laptop from the seed-0 weights (weights/<run>/net_0000.npz) with the frozen 331937 chain"

verdict = [("P1", res["P1"]["passed"]), ("P2", res["P2"]["passed"]), ("P3", res["P3"]["passed"])]
print("inclusion rule:", res["rule"]); print(f"nets: {res['n_total']} analysed, {res['n_included']} included; exclusions: {json.dumps(res['exclusions_per_cell'])}")
print("FAILED FIRST:", [p for p, ok in verdict if not ok] or "none"); print("PASSED:", [p for p, ok in verdict if ok] or "none")
print("P1 per C (Spearman wd vs rerandom_acc, 95% CI):", {C: (round(v["rho"], 3), [round(x, 3) for x in v["ci"]], v["n"]) for C, v in p1.items()}, "pass" if res["P1"]["passed"] else "FAIL")
print("P2:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["P2"].items() if k not in ("median_kcrit_by_C", "median_krandom_by_C")}),
      "\n   median k_crit by C:", res["P2"]["median_kcrit_by_C"], " random:", res["P2"]["median_krandom_by_C"])
print("P3:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["P3"].items()}))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
