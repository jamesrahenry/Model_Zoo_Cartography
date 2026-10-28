"""I1: init-free extraction of trained structure from weights alone (phase2/I1_PREREG_INIT_FREE_EXTRACTION.md).

Written 2026-10-25.  Stage 1 (per net, GPU): alpha_hat, outlier counts, init-free extraction and random-subspace control accuracy curves.
Stage 2: the preregistered statistics P1-P5 against the init-based results already stored per net.
usage: i1_init_free_extraction.py OUT.json [--limit N]
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.expanduser("~/Source/Rosetta_Program/Model_Zoo_Cartography/train"))
from gmm_task import make_gmm_task

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--limit", type=int); a = ap.parse_args()
H = os.path.expanduser("~/rosetta_data"); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
KS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
nets = []
for d, pat in (("mzc_grid4", "*"), ("mzc_grid", "*"), ("mzc_grid2", "*"), ("mzc_grid3", "*")):
    for wf in sorted(glob.glob(f"{H}/{d}/*/weights/*/net_*.npz")):
        job = wf.split("/weights/")[0]; run = os.path.basename(os.path.dirname(wf)); net = os.path.basename(wf)[:-4]
        jf = f"{job}/results/{run}/{net}.json"
        if not os.path.exists(jf): continue
        r = json.load(open(jf))
        if r.get("task", "gmm") != "gmm": continue
        nets.append((d, wf, r, int(net.split("_")[1])))
if a.limit: nets = nets[:a.limit]
ref = {}
def he_ref(n):
    if n not in ref:
        s = torch.linalg.svdvals(torch.randn(n, n, device=dev, generator=torch.Generator(device=dev).manual_seed(5)) * (2.0 / n) ** 0.5)
        ref[n] = (float(s.median()), float(s.max()))
    return ref[n]

def kneed(d, full):
    for k in KS:
        if str(k) in d and d[str(k)] >= 0.95 * full: return k
    return None

res = []
for i, (d, wf, r, seed) in enumerate(nets):
    z = np.load(wf); D = r["depth"]; n = r["width"]; C = r["C"]
    W = [torch.tensor(z[f"w{l}"], device=dev) for l in range(D)]; head = torch.tensor(z["head_w"], device=dev)
    task = make_gmm_task(dim=n, n_classes=C, separation=3.0, seed=0)
    x, y = task.sample(20000, np.random.default_rng(seed + 1_500_000)); x = torch.tensor(x, device=dev); y = torch.tensor(y, device=dev)
    def acc(layers):
        h = x
        for Wl in layers: h = torch.relu(h @ Wl)
        return float(((h @ head).argmax(1) == y).float().mean())
    m0, smax = he_ref(n)
    g = torch.Generator(device=dev); g.manual_seed(99)
    Wn = [torch.randn(n, n, device=dev, generator=g) * (2.0 / n) ** 0.5 for _ in range(D)]
    svd = [torch.linalg.svd(Wl) for Wl in W]
    ahat = [float(torch.linalg.svdvals(Wl).median()) / m0 for Wl in W]
    rhat = [int((s[1] > 1.05 * ahat[l] * smax).sum()) for l, s in enumerate(svd)]
    rnd = [torch.linalg.qr(torch.randn(n, n, device=dev, generator=g))[0] for _ in range(2 * D)]
    def build(l, k, kind):
        if kind == "random": U, Vv = rnd[l][:, :k], rnd[D + l][:, :k]
        else: U, Vv = svd[l][0][:, :k], svd[l][2][:k].T
        K = U @ (U.T @ W[l]) + (W[l] @ Vv) @ Vv.T - U @ ((U.T @ W[l] @ Vv) @ Vv.T)
        rest = Wn[l] - U @ (U.T @ Wn[l]) - (Wn[l] @ Vv) @ Vv.T + U @ ((U.T @ Wn[l] @ Vv) @ Vv.T)
        return K + ahat[l] * rest
    full = acc(W); ks = [k for k in KS if k <= n]
    free = {str(k): acc([build(l, k, "free") for l in range(D)]) for k in ks}
    rndc = {str(k): acc([build(l, k, "random") for l in range(D)]) for k in ks}
    L8 = min(8, D // 2); dec = r["decomposition"]
    res.append(dict(grid=d, run=r["run_id"], net=r["net"], C=C, wd=r["wd"], D=D, n=n, full=full, alpha_true=[x["alpha"] for x in dec], ahat=ahat, rhat=rhat,
                    alpha8=dec[L8]["alpha"], rer_init=r["ablation"]["rerandom"]["acc"], full_init=r["ablation"]["full"]["acc"],
                    k_init=kneed(r["slice"]["aligned_rerand"], r["ablation"]["full"]["acc"]), k_free=kneed(free, full), k_rnd_free=kneed(rndc, full),
                    free=free, rnd=rndc))
    if i % 10 == 0: print(f"{i+1}/{len(nets)} {d} {r['run_id'][-30:]} alpha8 {dec[L8]['alpha']:.2f} ahat8 {ahat[L8]:.3f} rhat8 {rhat[L8]} full {full:.3f} k_init {res[-1]['k_init']} k_free {res[-1]['k_free']}", flush=True)
json.dump(res, open(a.out, "w"))

# ---- Stage 2: preregistered statistics
err = np.array([abs(ah - at) for r in res for ah, at in zip(r["ahat"], r["alpha_true"])])
out = dict(n_nets=len(res)); P = {}
P["P1"] = dict(frac_within_0p03=float((err <= 0.03).mean()), n_pairs=int(len(err)), max_err=float(err.max()), passed=bool((err <= 0.03).mean() >= 0.9))
dep = [r for r in res if r["rer_init"] < 0.5 * r["full_init"] and r["k_init"] is not None and r["k_free"] is not None]
succ = lambda r: r["k_free"] <= max(2 * r["k_init"], 32)
lo = [r for r in dep if r["alpha8"] <= 0.6]; hi = [r for r in dep if r["alpha8"] >= 0.95]
rate = lambda S: float(np.mean([succ(r) for r in S])) if S else None
P["P2"] = dict(n=len(lo), success_rate=rate(lo), evaluable=len(lo) >= 6, passed=bool(len(lo) >= 6 and rate(lo) >= 0.7))
P["P3"] = dict(n=len(hi), success_rate=rate(hi), evaluable=len(hi) >= 6, passed=bool(len(hi) >= 6 and rate(hi) <= 0.3))
mean_r = lambda r: float(np.mean(r["rhat"][3:r["D"] - 1]))
rho4 = float(spearmanr([mean_r(r) for r in dep], [float(succ(r)) for r in dep])[0]) if len(dep) >= 6 else None
P["P4"] = dict(n=len(dep), spearman=rho4, evaluable=len(dep) >= 6, passed=bool(rho4 is not None and rho4 >= 0.5))
ok = [r for r in dep if succ(r) and r["k_rnd_free"] is not None]
P["P5"] = dict(n=len(ok), median_k_free=float(np.median([r["k_free"] for r in ok])) if ok else None, median_k_random=float(np.median([r["k_rnd_free"] for r in ok])) if ok else None,
               evaluable=len(ok) >= 6)
P["P5"]["passed"] = bool(P["P5"]["evaluable"] and P["P5"]["median_k_random"] >= 4 * max(P["P5"]["median_k_free"], 1))
out["P"] = P; out["n_dependent"] = len(dep)
out["by_alpha_group"] = {g: dict(n=len(S), success=rate(S)) for g, S in (("alpha<=0.6", lo), ("0.6<alpha<0.95", [r for r in dep if 0.6 < r["alpha8"] < 0.95]), ("alpha>=0.95", hi))}
print("nets", len(res), "dependent", len(dep)); print("FAILED FIRST:", [k for k, v in P.items() if not v["passed"]] or "none", "| PASSED:", [k for k, v in P.items() if v["passed"]] or "none")
for k, v in P.items(): print(k, json.dumps(v))
print("by alpha group:", json.dumps(out["by_alpha_group"]))
json.dump(out, open(a.out.replace(".json", "_summary.json"), "w"), indent=1)
