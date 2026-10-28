"""I3: extraction judged by fidelity (phase2/I3_PREREG_FIDELITY_EXTRACTION.md).

Written 2026-10-26.  eps = ||m' - m||^2 / ||m||^2 of per-neuron mean final activations over paired N(0,I) draws, rebuilt vs original net,
compared with the frozen chain's own relative error on that net.  Methods: INIT (aligned slice; init regenerated from the seed), I1 (per-layer
singular subspaces of W), I2 (one shared subspace across layers).  usage: i3_fidelity_extraction.py OUT.json [--limit N]
"""
import argparse, glob, importlib.util, json, os, sys
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--limit", type=int); ap.add_argument("--stats-only", action="store_true"); a = ap.parse_args()
H = os.path.expanduser("~/rosetta_data"); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
KS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]; KS_ = 16; NS = 8192
sp = importlib.util.spec_from_file_location("tb", os.path.expanduser("~/Source/Rosetta_Program/Model_Zoo_Cartography/train/train_mlp_batched.py"))
tb = importlib.util.module_from_spec(sp); sp.loader.exec_module(tb)
_M0 = {}
def m0_for(nn):
    if nn not in _M0:
        _M0[nn] = float(torch.linalg.svdvals(torch.randn(nn, nn, device=dev, generator=torch.Generator(device=dev).manual_seed(5)) * (2.0 / nn) ** 0.5).median())
    return _M0[nn]
i1 = json.load(open(f"{H}/i1/i1_results.json"))
sel = [r for r in i1 if (r["rer_init"] < 0.5 * r["full_init"] and r["k_init"] is not None) or (r["grid"] == "mzc_grid4" and r["alpha8"] <= 0.2)]
if a.limit: sel = sel[: a.limit]
def chain_err(r):
    run, net = r["run"], r["net"]; job = glob.glob(f"{H}/{r['grid']}/*/results/{run}/{net}_truth.npz")
    if not job: return None
    for pf in (f"{H}/mzc_grid4_chain/preds/pred_{run}_{net}.npy", f"{H}/mzc_grid_p4/preds/pred_{run}.npy", f"{H}/mzc_grid_q5/preds/pred_{run}.npy"):
        if os.path.exists(pf):
            P = np.load(pf); t = np.load(job[0])
            if not np.isfinite(P).all(): return None
            T = t["mean"].astype(np.float64)[-1]; return float((((P[-1] - T) ** 2).sum() - t["se2"].astype(np.float64)[-1].sum()) / (T ** 2).sum())
    return None
out = json.load(open(a.out)) if a.stats_only else []
for i, r in enumerate([] if a.stats_only else sel):
    wf = glob.glob(f"{H}/{r['grid']}/*/weights/{r['run']}/{r['net']}.npz")
    if not wf: continue
    z = np.load(wf[0]); D = r["D"]; n = r["n"]; C = r["C"]; seed = int(r["net"].split("_")[1])
    W = [torch.tensor(z[f"w{l}"], device=dev) for l in range(D)]
    L0, _ = tb.init_stacked(n, D, C, [seed]); W0 = [L0[l][0].to(dev) for l in range(D)]
    al = [float((x.double() * y.double()).sum() / (y.double() ** 2).sum()) for x, y in zip(W, W0)]
    De = [W[l] - al[l] * W0[l] for l in range(D)]
    M0 = m0_for(n); ahat = [float(torch.linalg.svdvals(Wl).median()) / M0 for Wl in W]
    g = torch.Generator(device=dev); g.manual_seed(99)
    Wn = [torch.randn(n, n, device=dev, generator=g) * (2.0 / n) ** 0.5 for _ in range(D)]
    Wn2 = [torch.randn(n, n, device=dev, generator=g) * (2.0 / n) ** 0.5 for _ in range(D)]
    X = torch.randn(NS, n, device=dev, generator=torch.Generator(device=dev).manual_seed(1234))
    def mean_act(layers):
        h = X
        for Wl in layers: h = torch.relu(h @ Wl)
        return h.mean(0).double()
    m = mean_act(W); den = float((m ** 2).sum())
    eps = lambda layers: float(((mean_act(layers) - m) ** 2).sum() / den)
    svD = [torch.linalg.svd(Dl) for Dl in De]; svW = [torch.linalg.svd(Wl) for Wl in W]
    UW = [s[0][:, :KS_] for s in svW]; VW = [s[2][:KS_].T for s in svW]
    ev = torch.linalg.eigh(sum(u @ u.T + v @ v.T for u, v in zip(UW, VW)))[1]
    def sym(Wl, Wr, Q, scale, extra):
        K = Q @ (Q.T @ Wl) + (Wl @ Q) @ Q.T - Q @ ((Q.T @ Wl @ Q) @ Q.T)
        rest = Wr - Q @ (Q.T @ Wr) - (Wr @ Q) @ Q.T + Q @ ((Q.T @ Wr @ Q) @ Q.T)
        return extra + scale * rest if extra is not None else K + scale * rest
    def build(method, l, k, Wr):
        if method == "init":
            U, V = svD[l][0][:, :k], svD[l][2][:k].T
            R = U @ (U.T @ W0[l]) + (W0[l] @ V) @ V.T - U @ ((U.T @ W0[l] @ V) @ V.T)
            rest = Wr[l] - U @ (U.T @ Wr[l]) - (Wr[l] @ V) @ V.T + U @ ((U.T @ Wr[l] @ V) @ V.T)
            return De[l] + al[l] * (R + rest)
        if method == "i1":
            U, V = svW[l][0][:, :k], svW[l][2][:k].T
            K = U @ (U.T @ W[l]) + (W[l] @ V) @ V.T - U @ ((U.T @ W[l] @ V) @ V.T)
            rest = Wr[l] - U @ (U.T @ Wr[l]) - (Wr[l] @ V) @ V.T + U @ ((U.T @ Wr[l] @ V) @ V.T)
            return K + ahat[l] * rest
        Q = ev[:, n - k:] if k > 0 else ev[:, :0]
        K = Q @ (Q.T @ W[l]) + (W[l] @ Q) @ Q.T - Q @ ((Q.T @ W[l] @ Q) @ Q.T)
        rest = Wr[l] - Q @ (Q.T @ Wr[l]) - (Wr[l] @ Q) @ Q.T + Q @ ((Q.T @ Wr[l] @ Q) @ Q.T)
        return K + ahat[l] * rest
    ks = [k for k in KS if k <= n]
    curves = {meth: {str(k): eps([build(meth, l, k, Wn) for l in range(D)]) for k in ks} for meth in ("init", "i1", "i2")}
    kinit = r["k_init"]; spread = None
    if kinit is not None and kinit in ks:
        spread = [curves["init"][str(kinit)], eps([build("init", l, kinit, Wn2) for l in range(D)])]
    out.append(dict(grid=r["grid"], run=r["run"], net=r["net"], alpha8=r["alpha8"], wd=r["wd"], C=C, k_init=kinit, k_free_I1=r["k_free"],
                    dependent=bool(r["rer_init"] < 0.5 * r["full_init"] and r["k_init"] is not None), eps_chain=chain_err(r), curves=curves, draw_spread=spread))
    if i % 10 == 0:
        json.dump(out, open(a.out + ".partial", "w")); print(f"{i+1}/{len(sel)} {r['grid']} {r['run'][-26:]} a8 {r['alpha8']:.2f} k_init {kinit} eps_chain {out[-1]['eps_chain']} eps_init@k_init {curves['init'].get(str(kinit)) if kinit is not None else None}", flush=True)
if not a.stats_only: json.dump(out, open(a.out, "w"))

# ---------------- statistics
kfid = lambda curve, e: next((k for k in KS if str(k) in curve and curve[str(k)] <= e), None)
dep = [o for o in out if o["dependent"] and o["eps_chain"] is not None]
S = {"n_selected": len(out), "n_dependent_with_chain": len(dep)}
P = {}
v1 = [o["curves"]["init"][str(o["k_init"])] > o["eps_chain"] for o in dep if str(o["k_init"]) in o["curves"]["init"]]
P["P1"] = dict(n=len(v1), frac_eps_gt_chain_at_k_init=float(np.mean(v1)) if v1 else None, evaluable=len(v1) >= 6, passed=bool(len(v1) >= 6 and np.mean(v1) >= 0.5))
kf = {m: [kfid(o["curves"][m], o["eps_chain"]) for o in dep] for m in ("init", "i1", "i2")}
ratio = [kf["init"][i] / max(o["k_init"], 1) for i, o in enumerate(dep) if kf["init"][i] is not None]
P["P2"] = dict(n=len(ratio), median_ratio=float(np.median(ratio)) if ratio else None, evaluable=len(ratio) >= 6, passed=bool(len(ratio) >= 6 and np.median(ratio) >= 4))
med = lambda xs: float(np.median([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
P["P3"] = dict(median_kfid=dict(init=med(kf["init"]), i1=med(kf["i1"]), i2=med(kf["i2"])), n=len(dep), evaluable=len(dep) >= 6)
P["P3"]["passed"] = bool(len(dep) >= 6 and P["P3"]["median_kfid"]["i1"] >= P["P3"]["median_kfid"]["init"] and P["P3"]["median_kfid"]["i2"] >= P["P3"]["median_kfid"]["init"])
hd = [o for o in out if o["grid"] == "mzc_grid4" and o["alpha8"] <= 0.2 and o["eps_chain"] is not None]
v4 = [o["curves"]["init"]["0"] > o["eps_chain"] for o in hd]
P["P4"] = dict(n=len(v4), frac_eps0_gt_chain=float(np.mean(v4)) if v4 else None, evaluable=len(v4) >= 6, passed=bool(len(v4) >= 6 and np.mean(v4) >= 0.7))
sp_ = [o["draw_spread"] for o in out if o["draw_spread"]]
S["draw_to_draw_spread_at_k_init"] = dict(n=len(sp_), median_abs_ratio=float(np.median([abs(np.log((a_ + 1e-30) / (b_ + 1e-30))) for a_, b_ in sp_])) if sp_ else None)
S["median_eps_at_k_by_method"] = {m: {str(k): float(np.median([o["curves"][m][str(k)] for o in dep if str(k) in o["curves"][m]])) for k in KS if any(str(k) in o["curves"][m] for o in dep)} for m in ("init", "i1", "i2")} if dep else None
S["median_eps_chain"] = float(np.median([o["eps_chain"] for o in dep])) if dep else None
S["P"] = P
print("\nselected", len(out), "dependent with chain", len(dep), "median eps_chain", S["median_eps_chain"])
print("FAILED FIRST:", [k for k, v in P.items() if not v["passed"]] or "none", "| PASSED:", [k for k, v in P.items() if v["passed"]] or "none")
for k, v in P.items(): print(k, json.dumps(v))
print("draw-to-draw spread:", S["draw_to_draw_spread_at_k_init"]); print("median eps by method and k:", json.dumps(S["median_eps_at_k_by_method"]))
json.dump(S, open(a.out.replace(".json", "_summary.json"), "w"), indent=1)
