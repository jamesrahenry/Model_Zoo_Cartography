"""
compensation_ablation.py -- Phase 2: is the compensation causally real, or
just a statistical echo?

Phase 1 (census/compensation_full_analysis.py) found freezing a layer only
disrupts effective_dim in layers 0-3 (the "zone"); the snowflake tests found
the deep code's SHAPE is shared everywhere and its DIRECTIONS are shared
only for a small core. None of that is causal -- it's all "does this
statistic match," never "does removing X actually cost the network
something." This is the causal test GEM's own method makes (preprint.md:696,
ablate a concept, measure what's recoverable) -- scoped to layers 0-3 per
the note (testing 4-31 would very likely just reproduce their independence,
since Phase 1 found nothing to disrupt there).

Per net, per ablation layer A in {0,1,2,3}: forward-pass a FIT batch through
the (already-trained, unablated) net to layer A, take per-class activation
means there, center, top-(C-1) orthonormal basis (the "concept subspace" --
same construction directional_consistency.py uses, just in activation space
at layer A instead of input-weight space at L0). On a held-out TEST batch,
project that subspace out of every sample's layer-A activation, continue
the forward pass through the SAME trained downstream weights, and measure
the resulting accuracy drop vs. the unablated net.

Prediction (if compensation is functionally real, not just cosmetic rank
movement): ablating layer 0 should barely hurt freeze_l0 (there was no real
concept there to remove -- it's frozen random noise) while hurting baseline
a lot (F1's real task-extraction layer). For layers 1-3 specifically: if
freeze_l0's downstream layers picked up real causal load compensating for
a broken L0, ablating THERE should hurt freeze_l0 MORE than it hurts
baseline, where those layers do their normal job on top of an already-good
L0 and have more redundant backup.

Usage: python census/compensation_ablation.py [--n-fit 2048] [--n-test 4096]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from corpus_io import net_paths

sys.path.insert(0, str(REPO_ROOT / "census"))
from run_activation_census import get_task

CORPUS_DIR = REPO_ROOT / "corpus"
CENSUS_DIR = REPO_ROOT / "census"

BASELINE_RUN = "comp_p1_baseline_c10"
FREEZE_RUN = "comp_p1_freeze_l0_c10"
WIDTH = 256
ABLATE_LAYERS = [0, 1, 2, 3]


def load_weights(run_id: str) -> list[tuple[list[np.ndarray], np.ndarray]]:
    """Per net: (hidden layers w0..wN-1, readout head_w). head_w is a
    separate, bias-free LINEAR readout (no ReLU after it) -- confirmed from
    the saved npz: 'head_w' (width, n_classes), distinct from w0..w31."""
    out = []
    for p in net_paths(run_id):
        d = np.load(p)
        n_layers = sum(1 for f in d.files if f.startswith("init_w"))
        hidden = [d[f"w{i}"] for i in range(n_layers)]
        out.append((hidden, d["head_w"]))
    return out


def forward_to_layer(hidden: list[np.ndarray], x: np.ndarray, stop_layer: int) -> np.ndarray:
    """Activation AFTER hidden layer stop_layer (post-ReLU), from raw input."""
    h = np.asarray(x, dtype=np.float64)
    for W in hidden[:stop_layer + 1]:
        h = np.maximum(h @ np.asarray(W, dtype=np.float64), 0.0)
    return h


def forward_from_layer(hidden: list[np.ndarray], head_w: np.ndarray, h: np.ndarray,
                       start_layer: int) -> np.ndarray:
    """Continue from a (possibly ablated) activation AFTER hidden layer
    start_layer, through the rest of the hidden stack (ReLU'd), then the
    bias-free linear readout (no ReLU) -- to logits."""
    for W in hidden[start_layer + 1:]:
        h = np.maximum(h @ np.asarray(W, dtype=np.float64), 0.0)
    return h @ np.asarray(head_w, dtype=np.float64)


def concept_basis(h_fit: np.ndarray, y_fit: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-(C-1) orthonormal basis of the centered per-class activation means
    -- the "concept subspace" at this layer, plus the global mean it was
    centered against (needed to define the ablation consistently)."""
    means = np.stack([h_fit[y_fit == c].mean(axis=0) for c in range(n_classes)])
    global_mean = h_fit.mean(axis=0)
    means_c = means - global_mean
    k = n_classes - 1
    basis = np.linalg.qr(means_c.T)[0][:, :k]
    return basis, global_mean


def ablate(h: np.ndarray, basis: np.ndarray, global_mean: np.ndarray) -> np.ndarray:
    """Project the concept subspace out of h, relative to the fit-set's
    global mean (so the ablation removes class-discriminative variation,
    not the layer's overall activation level)."""
    centered = h - global_mean
    proj = (centered @ basis) @ basis.T
    return h - proj


def evaluate_net(hidden: list[np.ndarray], head_w: np.ndarray, x_fit, y_fit,
                 x_test, y_test, n_classes: int) -> dict:
    result = {}
    n_hidden = len(hidden)
    # unablated baseline: full forward pass through every hidden layer + head
    h_test_full = forward_to_layer(hidden, x_test, n_hidden - 1)
    logits_full = h_test_full @ np.asarray(head_w, dtype=np.float64)
    result["unablated_acc"] = float((logits_full.argmax(axis=1) == y_test).mean())

    for A in ABLATE_LAYERS:
        h_fit_A = forward_to_layer(hidden, x_fit, A)
        basis, global_mean = concept_basis(h_fit_A, y_fit, n_classes)
        h_test_A = forward_to_layer(hidden, x_test, A)
        h_test_A_ablated = ablate(h_test_A, basis, global_mean)
        logits = forward_from_layer(hidden, head_w, h_test_A_ablated, A)
        acc = float((logits.argmax(axis=1) == y_test).mean())
        result[f"ablate_L{A}_acc"] = acc
        result[f"ablate_L{A}_drop"] = result["unablated_acc"] - acc
    return result


def main() -> None:
    rng = np.random.default_rng(11)
    prov = json.loads(next((CORPUS_DIR / BASELINE_RUN).glob("net_*.json")).read_text())
    n_classes = prov["task"]["n_classes"]
    task = get_task(prov["task"], WIDTH)
    x_fit, y_fit = task.sample(args.n_fit, rng)
    x_test, y_test = task.sample(args.n_test, rng)
    x_fit, x_test = x_fit.astype(np.float64), x_test.astype(np.float64)

    results = {}
    for run_id in (BASELINE_RUN, FREEZE_RUN):
        print(f"loading {run_id}...", flush=True)
        weights_list = load_weights(run_id)
        print(f"evaluating {len(weights_list)} nets...", flush=True)
        per_net = [evaluate_net(hidden, head_w, x_fit, y_fit, x_test, y_test, n_classes)
                  for hidden, head_w in weights_list]
        results[run_id] = per_net

    out_path = CENSUS_DIR / "compensation_ablation_results.json"
    out_path.write_text(json.dumps(
        {"ablate_layers": ABLATE_LAYERS, "n_classes": n_classes,
         "n_fit": args.n_fit, "n_test": args.n_test,
         "results": results,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}\n")

    print(f"{'run':<24} {'unablated':>10} " +
         " ".join(f"{'L'+str(A)+' acc':>9}" for A in ABLATE_LAYERS) + "  " +
         " ".join(f"{'L'+str(A)+' drop':>10}" for A in ABLATE_LAYERS))
    for run_id, per_net in results.items():
        unabl = np.mean([r["unablated_acc"] for r in per_net])
        accs = [np.mean([r[f"ablate_L{A}_acc"] for r in per_net]) for A in ABLATE_LAYERS]
        drops = [np.mean([r[f"ablate_L{A}_drop"] for r in per_net]) for A in ABLATE_LAYERS]
        print(f"{run_id:<24} {unabl:>10.4f} " +
             " ".join(f"{a:>9.4f}" for a in accs) + "  " +
             " ".join(f"{d:>10.4f}" for d in drops))

    print("\nCompare drops: is baseline hurt more at L0 (real concept there) and does")
    print("freeze_l0 get hurt MORE at L1-3 (compensating layers, more load-bearing)?")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n-fit", type=int, default=2048)
    p.add_argument("--n-test", type=int, default=4096)
    args = p.parse_args()
    main()
