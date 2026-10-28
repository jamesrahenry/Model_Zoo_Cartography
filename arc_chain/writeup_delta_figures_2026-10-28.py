"""Two figures for the Phase 2 post (fourth cut, 2026-10-28): random vs trained 1024 x 16 nets.
figA_relay.png     top: layer-8 weight spectrum; bottom: top-16 subspace overlap, layer l output -> layer m input (the relay)
figB_errormap.png  the frozen chain's error per layer, neurons sorted by each layer's own true mean (the live quarter)
Nets: He init (seed 777, session scratch), grid-4 20000-step wd 0 and wd 0.3 net_0020.  Run with OPENBLAS_NUM_THREADS=4 in the env.
usage: writeup_delta_figures_2026-10-28.py --init-npz NPZ --init-pred NPY --init-truth NPZ [--out DIR]
"""
import argparse, glob, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
H = os.path.expanduser("~/rosetta_data"); HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--init-npz", required=True); ap.add_argument("--init-pred", required=True); ap.add_argument("--init-truth", required=True)
ap.add_argument("--out", default=os.path.join(HERE, "..", "figures", "writeup_2026-10-28")); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
BLUE, ORANGE = "#2a78d6", "#eb6834"; SURF, INK, INK2, GRID, AXIS, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 200, "savefig.facecolor": SURF})
K = 16

def g4(run, net):
    job = glob.glob(f"{H}/mzc_grid4/*/results/{run}")[0]; wj = glob.glob(f"{H}/mzc_grid4/*/weights/{run}")[0]
    z = np.load(f"{wj}/{net}.npz"); W = [z[f"w{i}"].astype(np.float64) for i in range(16)]
    return W, np.load(f"{H}/mzc_grid4_chain/preds/pred_{run}_{net}.npy"), np.load(f"{job}/{net}_truth.npz")["mean"].astype(float)
z0 = np.load(a.init_npz)
nets = [("He-Gaussian init, no training", [z0[f"w{i}"].astype(np.float64) for i in range(16)], np.load(a.init_pred), np.load(a.init_truth)["mean"]),
        ("trained 20 000 steps, no weight decay", *g4("grid_wd0_c10", "net_0020")),
        ("trained 20 000 steps, weight decay 0.3", *g4("grid_wd0p3_c10", "net_0020"))]

# ---------- figure A: spectrum + relay ----------
fig, axs = plt.subplots(2, 3, figsize=(12.5, 7.4), gridspec_kw={"height_ratios": [1, 1.55], "hspace": 0.42, "wspace": 0.32})
for c, (name, W, P, T) in enumerate(nets):
    U, Sv, Vt = zip(*[np.linalg.svd(w, full_matrices=False) for w in W])
    ax = axs[0, c]; s = Sv[8]; ax.plot(np.arange(1, 1025), s, color=BLUE, lw=1.6)
    ax.axhline(2 * np.sqrt(2), color=AXIS, lw=1, ls="--"); ax.text(1.3, 2 * np.sqrt(2) * 1.15, "He bulk edge 2√2", fontsize=7.5, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.03, 30); ax.set_xlim(0.9, 1300)
    ax.set_title(f"{name}\nweight spectrum, layer 8", loc="left", fontsize=9.5); ax.set_xlabel("singular value rank"); ax.set_ylabel("singular value" if c == 0 else "")
    ax.text(0.03, 0.06, f"σ₁ {s[0]:.1f}   σ₁₇ {s[16]:.2f}   median {np.median(s):.2f}", transform=ax.transAxes, fontsize=7.5, color=INK2)
    O = np.array([[np.linalg.norm(Vt[l][:K] @ U[m][:, :K]) ** 2 / K for m in range(16)] for l in range(16)])
    ax = axs[1, c]; im = ax.imshow(O, cmap=SEQ, vmin=0, vmax=0.7, aspect="equal"); ax.grid(False)
    sup = np.mean([O[l, l + 1] for l in range(15)]); far = np.mean([O[l, m] for l in range(16) for m in range(16) if m - l >= 3])
    ax.set_title(f"subspace relay\nl→l+1 {sup:.2f} · ≥3 apart {far:.3f} · chance {K/1024:.3f}", loc="left", fontsize=8.5)
    ax.set_xlabel("layer m (input side)"); ax.set_ylabel("layer l (output side)" if c == 0 else ""); ax.set_xticks(range(0, 16, 3)); ax.set_yticks(range(0, 16, 3))
cb = fig.colorbar(im, ax=axs[1, :].tolist(), fraction=0.02, pad=0.02); cb.set_label("overlap of layer l's top-16 output directions\nwith layer m's top-16 input directions (1 = same subspace)", fontsize=8)
fig.suptitle("Where training shows up in the weights: a 16-direction relay that hands off layer to layer, visible with no reference at all", fontsize=11, x=0.01, ha="left")
fig.savefig(os.path.join(a.out, "figA_relay.png"), bbox_inches="tight"); plt.close(fig)

# ---------- figure B: error map ----------
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"wspace": 0.3})
for c, (name, W, P, T) in enumerate(nets):
    rel = np.stack([((P[l] - T[l]) / np.sqrt(np.mean(T[l] ** 2)))[np.argsort(T[l])] for l in range(16)])
    lim = 3e-4 if c == 0 else 0.3; ax = axs[c]
    im = ax.imshow(rel, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", interpolation="nearest"); ax.grid(False)
    ax.axvline(768, color=INK2, lw=0.8, ls=":"); ax.text(760, 0.6, "top quarter →", fontsize=7.5, color=INK2, ha="right")
    dead = float(np.mean(T[15] < 0.1 * np.sqrt(np.mean(T[15] ** 2)))); e = (P[15] - T[15]) ** 2; top = np.argsort(T[15])[-256:]
    ax.set_title(f"{name}\nL15: {dead:.0%} of neurons < 0.1× rms · {e[top].sum()/e.sum():.0%} of error in top quarter", loc="left", fontsize=8.5)
    ax.set_xlabel(""); ax.set_ylabel("layer" if c == 0 else "")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.set_title(f"±{lim:g}", fontsize=7.5, color=INK2)
fig.text(0.5, -0.03, "neuron index, each layer sorted by its own true mean activation (small → large)", ha="center", fontsize=9, color=INK2)
fig.suptitle("What the random-net chain gets wrong on each net: (prediction − truth) / rms truth of the layer.\nOn trained nets the error lives on the live quarter of neurons and nowhere else.", fontsize=10.5, x=0.01, ha="left", y=1.06)
fig.savefig(os.path.join(a.out, "figB_errormap.png"), bbox_inches="tight"); plt.close(fig)
print("wrote figA_relay.png figB_errormap.png")
