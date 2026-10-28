"""Which part of the random remnant is load-bearing?  Keep a rank-k slice of it, re-randomise or zero the rest.

Written 2026-10-22.  W = Delta + alpha*W0 per layer.  Let U_k, V_k be the top-k left/right singular vectors of Delta
(the task update's own input/output subspaces), P = U_k U_k^T, Q = V_k V_k^T.  The remnant touching those subspaces is
R_keep = P W0 + W0 Q - P W0 Q.  Conditions (task validation accuracy, 20k draws):
  aligned_rerand  W = Delta + alpha*(R_keep + (I-P) Wn (I-Q))   rest swapped for a fresh He draw Wn
  aligned_zero    W = Delta + alpha*R_keep                       rest zeroed
  random_rerand   same as aligned_rerand with random orthonormal U, V of rank k  (control)
k = 0 reproduces the full re-randomisation, k = 1024 the original net.
usage: mzc_remnant_structure_search.py OUT.json NPZ [NPZ ...]
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/Source/Rosetta_Program/Model_Zoo_Cartography/train"))
from gmm_task import make_gmm_task
torch.backends.cuda.matmul.allow_tf32 = False
out, files = sys.argv[1], sys.argv[2:]
KS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
dev = "cuda"; res = {}
for f in files:
    j = json.load(open(f.replace(".npz", ".json"))); z = np.load(f)
    C = j["task"]["n_classes"]
    task = make_gmm_task(dim=1024, n_classes=C, separation=j["task"]["separation"], seed=j["task"]["task_seed"])
    x, y = task.sample(20000, np.random.default_rng(j["training"]["data_seed"] + 500_000))
    x = torch.tensor(x, device=dev); y = torch.tensor(y, device=dev); head = torch.tensor(z["head_w"], device=dev)
    W = [torch.tensor(z[f"w{l}"], device=dev) for l in range(16)]; W0 = [torch.tensor(z[f"init_w{l}"], device=dev) for l in range(16)]
    al = [float((a * b).sum() / (b * b).sum()) for a, b in zip(W, W0)]
    De = [W[l] - al[l] * W0[l] for l in range(16)]
    g = torch.Generator(device=dev); g.manual_seed(2024)
    Wn = [torch.randn(1024, 1024, device=dev, generator=g) * (2.0 / 1024) ** 0.5 for _ in range(16)]
    svd = [torch.linalg.svd(D) for D in De]                                   # U (in), S, Vh (out)
    rnd = [torch.linalg.qr(torch.randn(1024, 1024, device=dev, generator=g))[0] for _ in range(32)]
    def acc(layers):
        h = x
        for Wl in layers: h = torch.relu(h @ Wl)
        return float(((h @ head).argmax(1) == y).float().mean())
    def build(l, k, kind):
        if kind == "random_rerand": U, V = rnd[l][:, :k], rnd[16 + l][:, :k]
        else: U, V = svd[l][0][:, :k], svd[l][2][:k].T
        P0, Q0 = al[l] * W0[l], None
        R = U @ (U.T @ W0[l]) + (W0[l] @ V) @ V.T - U @ ((U.T @ W0[l] @ V) @ V.T)
        if kind == "aligned_zero": return De[l] + al[l] * R
        rest = Wn[l] - U @ (U.T @ Wn[l]) - (Wn[l] @ V) @ V.T + U @ ((U.T @ Wn[l] @ V) @ V.T)
        return De[l] + al[l] * (R + rest)
    r = {"full": acc(W)}
    for kind in ("aligned_rerand", "aligned_zero", "random_rerand"):
        r[kind] = {k: acc([build(l, k, kind) for l in range(16)]) for k in KS}
    res[os.path.relpath(f).split("corpus/")[-1]] = dict(C=C, chance=1 / C, res=r)
    print(os.path.relpath(f).split("corpus/")[-1], "full %.3f" % r["full"], " ".join(f"k{k}:{r['aligned_rerand'][k]:.2f}/{r['random_rerand'][k]:.2f}" for k in KS), flush=True)
json.dump(res, open(out, "w"), indent=1)
