#!/usr/bin/env python3
"""Generate the paper's figures from the committed analysis data.

Reads the analysis JSONs (small summaries tracked in git; per-net censuses
from the local cache / HF `analysis/` tree — see README §Data) and writes
PNGs to paper/figures/. Every number plotted here is committed data; the two
hardcoded tables (per-width C50, budget slide) cite their FINDINGS section
and source censuses inline.

Run from the repo root or paper/: paths resolve relative to this file.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(exist_ok=True)

# Validated categorical palette (dataviz reference instance, light mode).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"     # contrast-warn vs white: always direct-labeled
INK = "#1f1f1e"
MUTED = "#8a8a84"
GRID = "#e5e5e2"
# Ordinal blue ramp (steps 250..700) for C-graded series.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def style(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def load(rel):
    return json.load(open(ROOT / rel))


def census_l0(fname):
    """Per-net (outcome, L0 significant dims, final acc) from a weight census."""
    d = load(f"census/{fname}")
    out = []
    for net in d["nets"].values():
        out.append((net["outcome"], net["trained_census"][0]["significant_dims"],
                    net.get("final_val_acc")))
    return out


# ---------------------------------------------------------------- Fig 1: F1
def fig1():
    c_files = {2: "sweep_c2_head", 3: "sweep_c3_head", 5: "sweep_c5_head",
               8: "sweep_c8_head", 10: "sweep_c10_batched", 15: "sweep_c15_head",
               20: "sweep_c20_head", 25: "sweep_c25_head", 32: "sweep_c32_head",
               40: "sweep_c40_head", 50: "sweep_c50_head"}
    sep_files = {1.5: "sweep_c10_sep15", 2.0: "sweep_c10_sep20",
                 3.0: "sweep_c10_batched", 4.5: "sweep_c10_sep45",
                 6.0: "sweep_c10_sep60"}

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8),
                               gridspec_kw={"width_ratios": [2.1, 1]})
    rng = np.random.default_rng(0)

    for C, f in c_files.items():
        for outcome, dims, _ in census_l0(f + "_weight_census.json"):
            x = C * np.exp(rng.uniform(-0.045, 0.045))
            if outcome == "converged":
                a.scatter(x, dims, s=14, color=BLUE, alpha=0.55, lw=0, zorder=3)
            else:
                a.scatter(x, dims, s=16, facecolor="none", edgecolor=MUTED,
                          lw=0.9, zorder=2)
    cs = np.array(sorted(c_files))
    a.plot(cs, cs - 1, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=1)
    a.text(33, 33, "significant dims = C − 1", color=INK, fontsize=9,
           rotation=38, ha="center", va="bottom")
    a.scatter([], [], s=14, color=BLUE, alpha=0.7, lw=0, label="converged net")
    a.scatter([], [], s=16, facecolor="none", edgecolor=MUTED, lw=0.9,
              label="partial / stalled")
    a.legend(frameon=False, fontsize=8.5, loc="upper left", handletextpad=0.2)
    a.set_xscale("log")
    a.set_xticks(cs, [str(c) for c in cs])
    a.minorticks_off()
    a.set_xlabel("classes C (GMM, sep 3.0, 20k steps)")
    a.set_ylabel("L0 significant dims (analytic MP floor)")
    style(a)

    for sep, f in sep_files.items():
        for outcome, dims, _ in census_l0(f + "_weight_census.json"):
            if outcome != "converged":
                continue
            b.scatter(sep + rng.uniform(-0.08, 0.08), dims, s=14, color=BLUE,
                      alpha=0.55, lw=0)
    b.axhline(9, color=INK, lw=1.2, ls=(0, (5, 3)))
    b.text(3.6, 9.25, "C − 1 = 9", color=INK, fontsize=9, ha="center")
    for name, val, y in (("MNIST", 9.4, 9.4), ("Fashion", 8.25, 8.25)):
        b.scatter(6.55, val, marker="D", s=22, color=ORANGE, zorder=3,
                  clip_on=False)
        b.text(6.75, y, name, color=ORANGE, fontsize=8, va="center")
    b.set_xlim(1.1, 6.6)
    b.set_ylim(bottom=0)
    b.set_xlabel("class separation (C = 10)")
    b.set_ylabel("")
    style(b)

    fig.suptitle("The input-rank law: L0 significant dims = C − 1, per net",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "fig1_input_rank_law.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2: F2
def fig2():
    st = load("null_baseline/state_trajectories.json")
    band = st["null_band"]["q"]
    mean, sd = np.array(band["mean"]), np.array(band["std"])
    L = np.arange(len(mean))

    # (family, label, color, label y-offset in points — staggered by hand
    # to keep the right-edge direct labels from colliding)
    fams = [("sweep_c2_head", "C = 2", BLUE_RAMP[0], -7),
            ("sweep_c10_batched", "C = 10", BLUE_RAMP[1], -6),
            ("sweep_c25_head", "C = 25", BLUE_RAMP[3], 6),
            ("sweep_c50_head", "C = 50 (partial)", BLUE_RAMP[4], 0),
            ("mnist_d32", "MNIST", ORANGE, 7)]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(L, mean - 2 * sd, mean + 2 * sd, color="#d9d9d5",
                    label="random-init null band (±2σ, 50 nets)")
    ax.plot(L, mean, color=MUTED, lw=1.0)

    for fam, label, color, dy in fams:
        q = np.median([net["trained"]["q"] for net in st["runs"][fam].values()],
                      axis=0)
        ls = (0, (4, 2)) if "partial" in label else "-"
        ax.plot(L, q, color=color, lw=1.8, ls=ls)
        ax.annotate(label, (L[-1], q[-1]), xytext=(4, dy),
                    textcoords="offset points", color=color, fontsize=8.5,
                    va="center")

    ax.set_yscale("log")
    ax.set_xlim(0, 38.5)
    ax.set_xticks(list(range(0, 29, 4)) + [31])
    ax.set_xlabel("layer")
    ax.set_ylabel("q = PR(Σ_pre) / w   (from weights alone)")
    ax.set_title("Training replaces the depth-driven terminal rank with a "
                 "task-driven one", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_qclock.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: F3
def fig3():
    d = load("census/activation_census.json")["runs"]
    c_fams = {2: "sweep_c2_head", 3: "sweep_c3_head", 5: "sweep_c5_head",
              8: "sweep_c8_head", 10: "sweep_c10_batched", 15: "sweep_c15_head",
              20: "sweep_c20_head", 25: "sweep_c25_head", 32: "sweep_c32_head"}

    def l1_anchored(fam, cond):
        vals = [net["trained"][cond][1]["significant_dims_anchored"]
                for net in d[fam].values()]
        return np.mean(vals)

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    cs = np.array(sorted(c_fams))
    task = [l1_anchored(c_fams[c], "task") for c in cs]
    noise = [l1_anchored(c_fams[c], "noise") for c in cs]
    a.plot(cs, cs - 1, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=1)
    a.text(20.5, 22.3, "C − 1", color=INK, fontsize=9, rotation=40)
    a.scatter(cs, noise, s=30, facecolor="none", edgecolor=ORANGE, lw=1.4,
              zorder=3, label="pure-noise input")
    a.scatter(cs, task, s=24, color=BLUE, lw=0, zorder=4, label="task input")
    a.legend(frameon=False, fontsize=8.5, loc="upper left", handletextpad=0.2)
    a.set_xscale("log")
    a.set_xticks(cs, [str(c) for c in cs])
    a.minorticks_off()
    a.set_xlabel("classes C")
    a.set_ylabel("L1 anchored significant dims")
    a.set_title("noise input already reads the law", fontsize=10, color=INK,
                loc="left")
    style(a)

    fam = d["sweep_c10_batched"]
    for who, cond_root, color, label in ((0, "init", MUTED, "init weights"),
                                         (1, "trained", BLUE, "trained weights")):
        eff = np.mean([[e["eff_dim"] for e in
                        (net["trained"] if who else net["init"])["noise"]]
                       for net in fam.values()], axis=0)
        L = np.arange(len(eff))
        b.plot(L, eff, color=color, lw=1.8)
        b.annotate(label, (L[-1], eff[-1]), xytext=(4, 0),
                   textcoords="offset points", color=color, fontsize=8.5,
                   va="center")
    b.annotate("at depth, init is MORE collapsed\nthan trained — the arrest,\n"
               "seen from the activation side", (17, 22), color=INK,
               fontsize=8.5, ha="left")
    b.set_yscale("log")
    b.set_xlim(0, 40)
    b.set_xticks(list(range(0, 29, 4)) + [31])
    b.set_xlabel("layer  (C = 10, pure-noise input)")
    b.set_ylabel("activation effective dim")
    b.set_title("the deep inversion", fontsize=10, color=INK, loc="left")
    style(b)

    fig.suptitle("Weights carry the where: structure appears without task "
                 "input", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "fig3_weights_carry_where.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4: F5
def fig4_rotation():
    eo = load("census/eigenspace_overlap.json")["runs"]["probe_d32_c10_head"]
    po = load("census/procrustes_overlap.json")
    L = np.arange(len(eo["trained_overlap"]))
    chance = eo["chance"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    a.plot(L, eo["trained_overlap"], color=BLUE, lw=1.8)
    a.plot(L, eo["init_overlap"], color=MUTED, lw=1.4)
    a.axhline(chance, color=INK, lw=1.0, ls=(0, (5, 3)))
    a.text(15.5, chance + 0.03, f"k/d chance = {chance:.3f}", color=INK,
           fontsize=8.5, ha="center")
    a.annotate("twins and init controls:\nat chance at every depth",
               (16, 0.09), color=INK, fontsize=8.5, ha="center", va="bottom",
               xytext=(16, 0.32),
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a.set_title("raw activation eigenspace", fontsize=10, color=INK, loc="left")
    a.set_xlabel("layer")
    a.set_ylabel("top-k subspace overlap")
    a.set_ylim(-0.04, 1.04)
    a.set_xlim(0, 32)
    style(a)

    conds = [("trained_twins", "same-task twins", BLUE, 8),
             ("cross_task_noise", "cross-task", ORANGE, -11),
             ("init_twins", "init controls", MUTED, 9)]
    for key, label, color, dy in conds:
        v = po["conditions"][key]
        b.plot(L, v, color=color, lw=1.8 if key == "trained_twins" else 1.4)
        b.annotate(label, (L[-1], v[-1]), xytext=(-2, dy),
                   textcoords="offset points", color=color, fontsize=8.5,
                   va="center", ha="right")
    b.axhline(chance, color=INK, lw=1.0, ls=(0, (5, 3)))
    b.set_xlim(0, 32)
    b.set_title("Procrustes-recovered (honest fit/test split)", fontsize=10,
                color=INK, loc="left")
    b.set_xlabel("layer")
    style(b)

    fig.suptitle("The learned code is one code up to rotation "
                 "(C = 10, k = 9, 4096 samples)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "fig4_rotation_hidden.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5: F6
def fig5_wall():
    tc = load("census/transition_curve.json")
    rows = sorted(tc["rows"], key=lambda r: r["C"])
    C = np.array([r["C"] for r in rows])
    frac = np.array([r["converged_frac"] for r in rows])
    fit = tc["wall_fit"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    xs = np.linspace(2, 60, 300)

    def logistic(x, c50, w):
        return 1 / (1 + np.exp((x - c50) / w))

    lo, hi = fit["c50_ci"]
    a.fill_between(xs, logistic(xs, lo, fit["width"]),
                   logistic(xs, hi, fit["width"]), color="#d9d9d5")
    a.plot(xs, logistic(xs, fit["c50"], fit["width"]), color=INK, lw=1.2)
    a.scatter(C, frac, s=26, color=BLUE, zorder=3)
    a.annotate(f"C₅₀ = {fit['c50']:.1f} [{lo:.1f}, {hi:.1f}]\n"
               f"width = {fit['width']:.1f}",
               (fit["c50"], 0.5), xytext=(10, 18), textcoords="offset points",
               color=INK, fontsize=9)
    a.set_xlabel("classes C  (w = 256, 20k steps)")
    a.set_ylabel("converged fraction")
    a.set_title("the wall at one width", fontsize=10, color=INK, loc="left")
    style(a)

    # C50 vs width; committed numbers from FINDINGS F6 (lr-tuned sweeps;
    # sources: transition_curve.json, b1c_w64_*/b1d_w128_*/b1d_w512_* and
    # b1e_*_200k censuses). CIs are the bootstrap intervals reported there.
    w = np.array([64, 128, 256, 512])
    c50_20k = np.array([27.8, 32.3, 36.2, 38.8])
    ci_20k = np.array([[27.2, 28.5], [31.2, 33.4], [35.2, 37.3], [37.5, 39.3]])
    w_200k = np.array([256, 512])
    c50_200k = np.array([50.0, 61.0])

    b.errorbar(w, c50_20k, yerr=np.abs(ci_20k.T - c50_20k), color=BLUE,
               lw=1.8, marker="o", ms=5, capsize=2.5)
    b.plot(w_200k, c50_200k, color=ORANGE, lw=1.8, marker="o", ms=5)
    b.annotate("20k steps", (512, 38.8), xytext=(8, -2),
               textcoords="offset points", color=BLUE, fontsize=8.5)
    b.annotate("200k steps", (512, 61.0), xytext=(8, -2),
               textcoords="offset points", color=ORANGE, fontsize=8.5)
    b.set_xscale("log", base=2)
    b.set_xticks(w, [str(x) for x in w])
    b.minorticks_off()
    b.set_xlim(56, 780)
    b.set_xlabel("width")
    b.set_ylabel("C₅₀ (half of seeds converge)")
    b.set_title("logarithmic in width and budget", fontsize=10, color=INK,
                loc="left")
    style(b)

    fig.suptitle("The class-count wall is a compute frontier, not a capacity "
                 "limit", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "fig5_wall.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7: F4
def fig7_refit():
    base = load("null_baseline/refit_trained_results.json")["results"]["val_c10"]
    edge = load("null_baseline/refit_trained_edge_results.json")["results"]["val_c10"]
    L = np.arange(len(base["uncorrected"]["mean_mse"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = [(base["uncorrected"]["mean_mse"], "uncorrected chain", MUTED, 1.4),
              (base["refit"]["mean_mse"], "state-keyed refit (128 params)", BLUE, 1.8),
              (edge["refit"]["mean_mse"], "+ edge indicators (160 params)", ORANGE, 1.8)]
    for v, label, color, lw in series:
        ax.plot(L, v, color=color, lw=lw)
        ax.annotate(label, (L[-1], v[-1]), xytext=(4, 0),
                    textcoords="offset points", color=color, fontsize=8.5,
                    va="center")
    ax.set_yscale("log")
    ax.set_xlim(0, 44)
    ax.set_xticks(list(range(0, 29, 4)) + [31])
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out spectrum MSE (val, C = 10)")
    ax.set_title("Population-fitted corrections repair the analytic chain on "
                 "trained weights", fontsize=11, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_refit.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6: F7 (bulk)
def fig6_bulk():
    fams = [(0.0, "sweep_c10_batched"), (0.01, "sweep_c10_wd"),
            (0.03, "b5_wd003"), (0.1, "b5_wd01"), (0.2, "b5_wd02"),
            (0.3, "sweep_c10_wd03"), (1.0, "sweep_c10_wd1")]
    wd, acc_m, acc_s, scale, dims = [], [], [], [], []
    for w, fam in fams:
        d = load(f"census/{fam}_weight_census.json")
        nets = [n for n in d["nets"].values() if n["outcome"] == "converged"]
        wd.append(w)
        acc_m.append(np.mean([n["final_val_acc"] for n in nets]))
        acc_s.append(np.std([n["final_val_acc"] for n in nets]))
        scale.append(np.mean([n["trained_census"][0]["scale_ratio"] for n in nets]))
        dims.append(np.mean([n["trained_census"][0]["significant_dims"] for n in nets]))

    x = np.arange(len(wd))
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(6.4, 5.6), sharex=True)

    a.errorbar(x, acc_m, yerr=acc_s, color=BLUE, lw=1.8, marker="o", ms=4,
               capsize=2.5)
    a.set_ylabel("val accuracy")
    a.set_ylim(0.5, 1.0)
    a.text(0.02, 0.1, "accuracy moves < 0.01 across the whole sweep",
           transform=a.transAxes, color=INK, fontsize=8.5)
    style(a)

    b.plot(x, scale, color=AQUA, lw=1.8, marker="o", ms=4)
    b.set_yscale("log")
    b.set_ylabel("L0 bulk scale (× He)")
    b.text(0.02, 0.12, "the random bulk is annihilated", transform=b.transAxes,
           color=AQUA, fontsize=8.5)
    style(b)

    c.plot(x, dims, color=BLUE, lw=1.8, marker="o", ms=4)
    c.axhline(9, color=INK, lw=1.0, ls=(0, (5, 3)))
    c.text(0.02, 0.78, "C − 1 = 9", transform=c.transAxes, color=INK,
           fontsize=8.5)
    c.annotate("fixed-floor census fails\nbetween wd 0.1 and 0.2",
               (4.55, 5.2), color=BLUE, fontsize=8.5, ha="center")
    c.set_ylabel("L0 sig dims\n(fixed floor)")
    c.set_xticks(x, [str(w) for w in wd])
    c.set_xlabel("weight decay  (C = 10, converged nets only)")
    style(c)

    fig.suptitle("Same function, different weight statistics: the bulk is "
                 "functionally inert", fontsize=11, color=INK, x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "fig6_bulk_inert.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4_rotation, fig5_wall, fig6_bulk, fig7_refit):
        f()
        print("wrote", f.__name__)
