"""I2: cross-layer, weights-only extraction of trained structure (phase2/I2_PREREG_CROSS_LAYER_EXTRACTION.md).

Written 2026-10-25.  Part A (nets with a stored init): chain score S(W) vs the init's S, recovery of the shared update subspace.
Part B (137 GMM nets of I1): shared-subspace extraction accuracy curves + random control; joined with I1's results for the baseline.
usage: i2_cross_layer_extraction.py OUT.json [--limit N]
"""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/Source/Rosetta_Program/Model_Zoo_Cartography/train"))
from gmm_task import make_gmm_task
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--limit", type=int); a = ap.parse_args()
H = os.path.expanduser("~/rosetta_data"); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
KS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]; KS_ = 16; n = 1024
_M0 = {}
def m0_for(nn):
    if nn not in _M0:
        g0 = torch.Generator(device=dev).manual_seed(5)
        _M0[nn] = float(torch.linalg.svdvals(torch.randn(nn, nn, device=dev, generator=g0) * (2.0 / nn) ** 0.5).median())
    return _M0[nn]
M0 = m0_for(1024)

def subs(Ws, k=KS_):
    sv = [torch.linalg.svd(W) for W in Ws]
    return [s[0][:, :k] for s in sv], [s[2][:k].T for s in sv]
def chain(U, V, k=KS_): return float(np.mean([(U[l + 1].T @ V[l]).pow(2).sum().item() / k for l in range(len(U) - 1)]))
def shared(U, V, k=KS_):
    M = sum(u @ u.T + v @ v.T for u, v in zip(U, V)); e = torch.linalg.eigh(M); return e[1][:, -k:]
def capture(Qh, Qt): return float((Qh.T @ Qt).pow(2).sum().item() / Qt.shape[1])

# ---------------- Part A
A = []
for wf in sorted(glob.glob(os.path.expanduser("~/Source/Rosetta_Program/Model_Zoo_Cartography/corpus/p2_w1024_d16_*/net_*.npz"))):
    run = os.path.basename(os.path.dirname(wf)); z = np.load(wf)
    if "init_w0" not in z.files: continue
    W = [torch.tensor(z[f"w{l}"], device=dev) for l in range(16)]; W0 = [torch.tensor(z[f"init_w{l}"], device=dev) for l in range(16)]
    al = [float((x.double() * y.double()).sum() / (y.double() ** 2).sum()) for x, y in zip(W, W0)]; De = [W[l] - al[l] * W0[l] for l in range(16)]
    UW, VW = subs(W); U0, V0 = subs(W0); UD, VD = subs(De)
    Qh = shared(UW, VW); Qd = shared(UD, VD)
    A.append(dict(run=run, net=os.path.basename(wf)[:-4], alpha8=al[8], S_W=chain(UW, VW), S_init=chain(U0, V0), S_delta=chain(UD, VD), capture=capture(Qh, Qd)))
    if a.limit and len(A) >= a.limit: break
print("part A nets", len(A), flush=True)

# ---------------- Part B
i1 = {(r["grid"], r["run"], r["net"]): r for r in json.load(open(f"{H}/i1/i1_results.json"))}
B = []
keys = list(i1.keys())[: (a.limit or None)]
for d, run, net in keys:
    r1 = i1[(d, run, net)]
    wf = glob.glob(f"{H}/{d}/*/weights/{run}/{net}.npz")
    if not wf: continue
    z = np.load(wf[0]); D = r1["D"]; C = r1["C"]; seed = int(net.split("_")[1]); n = r1["n"]; M0 = m0_for(n)
    W = [torch.tensor(z[f"w{l}"], device=dev) for l in range(D)]; head = torch.tensor(z["head_w"], device=dev)
    task = make_gmm_task(dim=n, n_classes=C, separation=3.0, seed=0); x, y = task.sample(20000, np.random.default_rng(seed + 1_500_000))
    x = torch.tensor(x, device=dev); y = torch.tensor(y, device=dev)
    def acc(layers):
        h = x
        for Wl in layers: h = torch.relu(h @ Wl)
        return float(((h @ head).argmax(1) == y).float().mean())
    g = torch.Generator(device=dev); g.manual_seed(99); Wn = [torch.randn(n, n, device=dev, generator=g) * (2.0 / n) ** 0.5 for _ in range(D)]
    U, V = subs(W); Qh_full = None
    ahat = [float(torch.linalg.svdvals(Wl).median()) / M0 for Wl in W]
    M = sum(u @ u.T + v @ v.T for u, v in zip(U, V)); ev = torch.linalg.eigh(M)[1]                  # eigenvectors ascending
    Rq = torch.linalg.qr(torch.randn(n, n, device=dev, generator=g))[0]
    def build(l, k, Q):
        K = Q @ (Q.T @ W[l]) + (W[l] @ Q) @ Q.T - Q @ ((Q.T @ W[l] @ Q) @ Q.T)
        rest = Wn[l] - Q @ (Q.T @ Wn[l]) - (Wn[l] @ Q) @ Q.T + Q @ ((Q.T @ Wn[l] @ Q) @ Q.T)
        return K + ahat[l] * rest
    full = acc(W); ks = [k for k in KS if k <= n]
    sh = {}; rd = {}
    for k in ks:
        Q = ev[:, n - k:] if k > 0 else ev[:, :0]; Qr = Rq[:, :k]
        sh[str(k)] = acc([build(l, k, Q) for l in range(D)]); rd[str(k)] = acc([build(l, k, Qr) for l in range(D)])
    kn = lambda d_: next((k for k in KS if str(k) in d_ and d_[str(k)] >= 0.95 * full), None)
    B.append(dict(grid=d, run=run, net=net, S_W=chain(U, V), full=full, k_q=kn(sh), k_q_rnd=kn(rd), shared=sh, rnd=rd))
    if len(B) % 10 == 0: json.dump(dict(A=A, B=B), open(a.out + ".partial", "w"))
    if len(B) % 10 == 1: print(f"{len(B)}/{len(keys)} {d} {run[-28:]} alpha8 {r1['alpha8']:.2f} k_init {r1['k_init']} k_free(I1) {r1['k_free']} k_q {B[-1]['k_q']} S {B[-1]['S_W']:.2f}", flush=True)
json.dump(dict(A=A, B=B), open(a.out, "w"))

# ---------------- statistics
P = {}
sw = np.array([r["S_W"] for r in A]); si = np.array([r["S_init"] for r in A])
P["P1"] = dict(n=len(A), frac_S_ge_0p156=float((sw >= 0.156).mean()), max_S_init=float(si.max()), passed=bool(len(A) > 0 and (sw >= 0.156).mean() >= 0.9 and si.max() <= 0.047))
hiA = [r for r in A if r["alpha8"] >= 0.95]; loA = [r for r in A if r["alpha8"] <= 0.6]
med = lambda S: float(np.median([r["capture"] for r in S])) if S else None
P["P2"] = dict(n_hi=len(hiA), median_capture_hi=med(hiA), n_lo=len(loA), median_capture_lo=med(loA),
               evaluable=bool(len(hiA) >= 6 and len(loA) >= 6), passed=bool(len(hiA) >= 6 and len(loA) >= 6 and med(hiA) >= 0.8 and med(loA) >= 0.8))
dep = [(i1[(r["grid"], r["run"], r["net"])], r) for r in B]
dep = [(a1, b) for a1, b in dep if a1["rer_init"] < 0.5 * a1["full_init"] and a1["k_init"] is not None and b["k_q"] is not None]
sq = lambda a1, b: b["k_q"] <= max(2 * a1["k_init"], 32); s1 = lambda a1, b: a1["k_free"] <= max(2 * a1["k_init"], 32)
rate = lambda S, f: float(np.mean([f(a1, b) for a1, b in S])) if S else None
lo = [(a1, b) for a1, b in dep if a1["alpha8"] <= 0.6]; hi = [(a1, b) for a1, b in dep if a1["alpha8"] >= 0.95]
P["P3"] = dict(n_lo=len(lo), rate_lo=rate(lo, sq), n_hi=len(hi), rate_hi=rate(hi, sq), evaluable=bool(len(lo) >= 6 and len(hi) >= 6),
               passed=bool(len(lo) >= 6 and len(hi) >= 6 and rate(lo, sq) >= 0.7 and rate(hi, sq) >= 0.7))
P["P4"] = dict(rate_I2=rate(dep, sq), rate_I1=rate(dep, s1), difference=(rate(dep, sq) - rate(dep, s1)) if dep else None, n=len(dep))
P["P4"]["passed"] = bool(dep and P["P4"]["difference"] >= 0.30)
ok = [(a1, b) for a1, b in dep if sq(a1, b) and b["k_q_rnd"] is not None]
P["P5"] = dict(n=len(ok), median_k_q=float(np.median([b["k_q"] for _, b in ok])) if ok else None, median_k_rnd=float(np.median([b["k_q_rnd"] for _, b in ok])) if ok else None, evaluable=len(ok) >= 6)
P["P5"]["passed"] = bool(P["P5"]["evaluable"] and P["P5"]["median_k_rnd"] >= 4 * max(P["P5"]["median_k_q"], 1))
print("\nFAILED FIRST:", [k for k, v in P.items() if not v["passed"]] or "none", "| PASSED:", [k for k, v in P.items() if v["passed"]] or "none")
for k, v in P.items(): print(k, json.dumps(v))
json.dump(P, open(a.out.replace(".json", "_summary.json"), "w"), indent=1)
