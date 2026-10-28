"""Figures for the Phase 2 write-up post (phase2/WRITEUP_POST_2026-10-28.md).  Written 2026-10-28.

fig1  per-neuron truth vs chain prediction at layer 15: undecayed vs decayed trained net (grid 4), plus per-layer error
fig2  twin 3x3 heatmaps: dependence on the draw (grid 3) vs chain coherence (grid 4) over the same (steps, alpha) cells
fig3  surviving init amplitude: measured vs exp(-lr*wd*(steps-250)) over grids 1-3
fig4  (a) accuracy vs kept rank, aligned slice vs random subspace (grid 2 dependent nets); (b) I3 fidelity curves vs the chain's error
usage: writeup_figures_2026-10-28.py [--out DIR] [--init-pred PRED.npy --init-truth TRUTH.npz]
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

H = os.path.expanduser("~/rosetta_data"); HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "..", "figures", "writeup_2026-10-28"))
ap.add_argument("--init-pred"); ap.add_argument("--init-truth"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

# palette (dataviz reference instance, light): categorical slots 1-3, sequential blue from step 250, chrome inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200, "savefig.facecolor": SURF})
KS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def relerr(P, T, se2):  # noise-debiased relative error per layer
    return np.maximum((((P - T) ** 2).sum(1) - se2.sum(1)) / (T ** 2).sum(1), 1e-12)

# ---------------- fig 1 ----------------
def g4(run, net):
    job = glob.glob(f"{H}/mzc_grid4/*/results/{run}")[0]
    tr = np.load(f"{job}/{net}_truth.npz"); P = np.load(f"{H}/mzc_grid4_chain/preds/pred_{run}_{net}.npy")
    return P, tr["mean"].astype(float), tr["se2"].astype(float)

cells = {"undecayed": "grid_wd0_c10", "decayed": "grid_wd0p3_c10"}
panels = []
if a.init_pred:
    tr = np.load(a.init_truth); panels.append(("He-Gaussian init (no training)", np.load(a.init_pred), tr["mean"].astype(float), tr["se2"].astype(float)))
panels.append(("trained, 20 000 steps, wd 0 (α ≈ 1)", *g4(cells["undecayed"], "net_0020")))
panels.append(("trained, 20 000 steps, wd 0.3 (α ≈ 0.17)", *g4(cells["decayed"], "net_0020")))
n = len(panels)
if n == 3: fig, axs = plt.subplots(2, 2, figsize=(9.2, 8.8)); axs = axs.ravel()
else: fig, axs = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 4.2))
for ax, (title, P, T, se2) in zip(axs[:n], panels):
    t, p = T[15], P[15]; lo, hi = 0, max(t.max(), p.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=1)
    ax.scatter(t, p, s=9, color=BLUE, alpha=0.55, linewidths=0, zorder=2)
    s = (t * p).sum() / (t * t).sum(); rel = relerr(P, T, se2)[15]
    ax.plot([lo, hi], [lo, s * hi], color=ORANGE, lw=1.5, ls="--", zorder=3)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Monte Carlo truth, layer-15 mean activation"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.text(0.03, 0.97, f"relative error {rel:.1e}\nfitted pred ≈ {s:.3f} × truth", transform=ax.transAxes, va="top", fontsize=9, color=INK2)
for ax in (axs[0], axs[2]) if n == 3 else (axs[0],): ax.set_ylabel("chain prediction (frozen #331937)")
ax = axs[n]
for name, run, col in [("undecayed (wd 0)", cells["undecayed"], ORANGE), ("decayed (wd 0.3)", cells["decayed"], BLUE)]:
    R = np.array([relerr(*g4(run, f"net_{i:04d}")) for i in range(20, 28)])
    med = np.median(R, 0); ax.plot(range(16), med, color=col, lw=2, label=name); ax.fill_between(range(16), R.min(0), R.max(0), color=col, alpha=0.15, lw=0)
if a.init_pred:
    ax.plot(range(16), relerr(panels[0][1], panels[0][2], panels[0][3]), color=AQUA, lw=2, label="He init (no training)")
ax.set_yscale("log"); ax.set_xlabel("layer"); ax.set_ylabel("relative error ‖e‖² / ‖truth‖²"); ax.set_xlim(0, 15.5); ax.set_ylim(1e-10, 1)
ax.legend(loc="lower right", fontsize=8); ax.set_title("per-layer error (8 nets per cell, median and range)", fontsize=10)
fig.suptitle("The frozen chain on trained weights: a same-signed shortfall along the prediction, or a cloud", fontsize=11, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig1_scatter_layer15.png")); plt.close(fig)

# ---------------- fig 2 ----------------
g3 = json.load(open(os.path.join(HERE, "mzc_grid3_prereg_results.json")))["H4"]["cells_by_rho"]
g4r = json.load(open(os.path.join(HERE, "mzc_grid4_prereg_results.json")))["cells"]
steps = [3000, 20000, 60000]; wds = {3000: [0.0, 0.72, 2.15], 20000: [0.0, 0.1, 0.3], 60000: [0.0, 0.033, 0.099]}
dep = np.zeros((3, 3)); coh = np.zeros((3, 3)); rho = np.zeros((3, 3))
for i, s in enumerate(steps):
    for j, w in enumerate(wds[s]):
        c = next(c for c in g3 if c["steps"] == s and abs(c["wd"] - w) < 1e-9); dep[i, j] = c["mean_ratio"]; rho[i, j] = c["mean_rho"]
        coh[i, j] = g4r[f"steps={s},wd={w}"]["share"]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.4))
for ax, M, title, sub in [(axs[0], dep, "Dependence of the function on its random draw", "accuracy after re-randomising the remnant / full accuracy\n(grid 3, 8 nets per cell)"),
                          (axs[1], coh, "Coherence of the chain's error", "held-out coherent share at layer 15\n(grid 4, fresh seeds, 8 nets per cell)")]:
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(3)); ax.set_xticklabels(["α = 1\n(wd 0)", "α ≈ 0.55", "α ≈ 0.17"]); ax.set_yticks(range(3)); ax.set_yticklabels([f"{s:,}" for s in steps])
    ax.set_xlabel("surviving init amplitude α"); ax.set_title(f"{title}\n\n", fontsize=11, loc="left"); ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    for i in range(3):
        for j in range(3):
            ax.text(j, i - 0.12, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=12, color="white" if M[i, j] > 0.45 else INK, fontweight="bold")
            ax.text(j, i + 0.24, f"ρ = {rho[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.45 else INK2)
axs[0].set_ylabel("training steps")
fig.suptitle("Same nine cells, two different variables: the function follows the update size ρ, the chain's error follows the decay α", fontsize=11, x=0.01, ha="left")
fig.text(0.01, 0.012, "α = exp(−lr·wd·(steps − 250)), set by the weight decay chosen for each training length (wd 0 / 0.72 / 2.15 at 3 000 steps; 0 / 0.1 / 0.3 at 20 000; 0 / 0.033 / 0.099 at 60 000).\n"
         "ρ = ‖Δ‖_F / ‖α W0‖_F, the trained update relative to the surviving init, mean over layers 3–14.", fontsize=7.5, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.06, 1, 0.96)); fig.savefig(os.path.join(a.out, "fig2_two_variables.png")); plt.close(fig)

# ---------------- fig 3 ----------------
pts = []
for d, tag in [("mzc_grid", "grid 1"), ("mzc_grid2", "grid 2"), ("mzc_grid3", "grid 3")]:
    for f in glob.glob(f"{H}/{d}/*/results/*/net_*.json"):
        r = json.load(open(f))
        if r["outcome"] == "stalled": continue
        D = r["depth"]; L = min(8, D // 2); lr, st = r.get("lr", 3e-4), r.get("steps", 20000)
        pts.append((np.exp(-lr * r["wd"] * max(st - 250, 0)), r["decomposition"][L]["alpha"], tag))
pts = np.array(pts, dtype=object); x = pts[:, 0].astype(float); y = pts[:, 1].astype(float)
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot([0, 1.02], [0, 1.02], color=AXIS, lw=1)
jit = (np.random.default_rng(0).random(len(x)) - 0.5) * 0.012
ax.scatter(x + jit, y, s=12, color=BLUE, alpha=0.35, linewidths=0)
ax.set_xlabel("predicted α = exp(−lr · wd · (steps − 250))"); ax.set_ylabel("measured α at layer 8 (least squares, W = α W0 + Δ)")
ax.set_xlim(-0.02, 1.04); ax.set_ylim(-0.02, 1.04)
ax.set_title("How much of the initialisation survives training", fontsize=11, loc="left")
ax.text(0.03, 0.97, f"{len(x)} nets, grids 1–3\nwidths 512/1024, depths 8/16/24, GMM/MNIST/Fashion\nmax |measured − predicted| = {np.abs(y - x).max():.3f}",
        transform=ax.transAxes, va="top", fontsize=9, color=INK2)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_alpha_closed_form.png")); plt.close(fig)

# ---------------- fig 4 ----------------
al, rd = [], []
for f in glob.glob(f"{H}/mzc_grid2/*/results/*/net_*.json"):
    r = json.load(open(f)); full = r["ablation"]["full"]["acc"]
    if r["outcome"] == "stalled" or r["ablation"]["rerandom"]["acc"] >= 0.5 * full: continue
    sa, sr = r["slice"]["aligned_rerand"], r["slice"]["random_rerand"]
    if all(str(k) in sa for k in KS) and all(str(k) in sr for k in KS):
        al.append([sa[str(k)] / full for k in KS]); rd.append([sr[str(k)] / full for k in KS])
al, rd = np.array(al), np.array(rd)
i3 = json.load(open(os.path.join(HERE, "i3_fidelity_extraction_summary.json")))
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.3}); xi = np.arange(len(KS))
ax = axs[0]
for M, name, col in [(al, "slice aligned with the net's own update Δ", BLUE), (rd, "random subspace of the same rank", ORANGE)]:
    ax.plot(xi, np.median(M, 0), color=col, lw=2, marker="o", ms=4, label=name); ax.fill_between(xi, np.percentile(M, 25, 0), np.percentile(M, 75, 0), color=col, alpha=0.15, lw=0)
ax.axhline(0.95, color=AXIS, lw=1, ls=":"); ax.text(0.1, 0.905, "95 % of full accuracy", fontsize=8, color=MUTED)
ax.set_xticks(xi); ax.set_xticklabels(KS); ax.set_xlabel("directions kept from the remnant per layer, k (rest re-randomised)")
ax.set_ylabel("task accuracy / full-net accuracy"); ax.set_ylim(0, 1.05); ax.legend(loc="center right", fontsize=8)
ax.set_title("(a) By accuracy, ~16 directions carry the function\n", fontsize=10, loc="left")
ax.text(0, 1.015, f"grid 2, {len(al)} nets that depend on their draw; median and IQR", transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
ax = axs[1]
for key, name, col in [("init", "INIT: slice from Δ (needs the init)", BLUE), ("i1", "I1: per-layer singular subspaces (weights only)", ORANGE), ("i2", "I2: one shared subspace (weights only)", AQUA)]:
    e = np.array([i3["median_eps_at_k_by_method"][key][str(k)] for k in KS])
    ax.plot(xi[1:-1], e[1:-1], color=col, lw=2, marker="o", ms=4, label=name)
ec = i3["median_eps_chain"]; ax.axhline(ec, color=INK2, lw=1.2, ls="--"); ax.text(0.6, ec * 1.4, f"the chain's own error on these nets (median {ec:.1e})", fontsize=8, color=INK2)
ax.set_yscale("log"); ax.set_ylim(1e-3, 3); ax.set_xticks(xi); ax.set_xticklabels(KS); ax.set_xlabel("directions kept per layer, k (rest: fresh He draw; k = 1024 keeps the net)")
ax.set_ylabel("ε of the rebuilt net (rel. squared error, layer-15 mean)"); ax.legend(loc="lower left", fontsize=8)
ax.set_title("(b) By fidelity in ARC's currency, nothing is free\n", fontsize=10, loc="left")
ax.text(0, 1.015, f"I3, {i3['n_dependent_with_chain']} dependent nets with a chain prediction; median ε(k)", transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig4_slice_vs_fidelity.png")); plt.close(fig)
print("wrote", sorted(os.listdir(a.out)), "| fig4a nets", len(al), "| fig3 nets", len(x))
