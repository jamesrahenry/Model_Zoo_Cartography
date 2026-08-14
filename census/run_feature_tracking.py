"""
run_feature_tracking.py — AMC's cross-layer feature tracker on the corpus (M3).

For each net (trained and init twin), forward task inputs, take per-layer
top-20 PC directions (store_directions=True), and run the vendored
feature_tracker: a "feature" is a direction persisting across consecutive
layers (cos > threshold). Reports per family, trained vs init:

  - number of features, persistent (lifespan ≥ 5) and transient (≤ 2) counts
  - lifespan distribution; birth-layer histogram (where do features appear?)
  - depth-spanning features (birth ≤ 2, death ≥ n_layers−2) — the candidate
    "task code carriers"

Connects MZC's rank language to AMC's feature language: the C−1 law predicts
~C−1 depth-spanning features in trained nets and none in init nets.

Usage: python census/run_feature_tracking.py --run-id probe_d32_c10_head [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from corpus_io import net_paths

sys.path.insert(0, str(REPO_ROOT / "census"))
from manifold_detector import _layer_census
from feature_tracker import track_features
from run_activation_census import get_task

CORPUS_DIR = REPO_ROOT / "corpus"
CENSUS_DIR = REPO_ROOT / "census"


def net_feature_map(weights: list[np.ndarray], x: np.ndarray, model_id: str,
                    cos_threshold: float, transport: bool):
    dirs, eigs = [], []
    h = np.asarray(x, dtype=np.float64)
    Ws = [np.asarray(W, dtype=np.float64) for W in weights]
    for l, W in enumerate(Ws):
        h = np.maximum(h @ W, 0.0)
        cen = _layer_census(h, l, {}, n_top_eigenvalues=20, store_directions=True)
        dirs.append(cen.top_directions)
        eigs.append(cen.top_eigenvalues)
    if not transport:
        return track_features(dirs, eigs, n_layers_total=len(Ws),
                              cos_threshold=cos_threshold, model_id=model_id)
    # Transport-corrected LOCAL matcher: plain MLPs re-rotate the basis every
    # layer (no residual stream), so raw cross-layer cosines are meaningless —
    # the vendored tracker was built for transformers. For each consecutive
    # pair, push layer-l directions through W_{l+1} (u -> u^T W, the
    # covariance-mode pushforward; ReLU gating neglected) and match against
    # layer-(l+1) directions. Per-pair transport can't be expressed through
    # the vendored tracker's single direction list, hence local chaining.
    n_layers = len(Ws)
    chains: list[dict] = []          # each: {"birth": l, "last": l, "idx": pc}
    active: dict[int, dict] = {}     # pc index at current layer -> chain
    for pc in range(len(dirs[0])):
        ch = {"birth": 0, "last": 0}
        chains.append(ch)
        active[pc] = ch
    for l in range(n_layers - 1):
        pushed = dirs[l] @ Ws[l + 1]
        pushed /= np.maximum(np.linalg.norm(pushed, axis=1, keepdims=True), 1e-12)
        cos = np.abs(pushed @ dirs[l + 1].T)   # [n_pc_l, n_pc_l+1]
        new_active: dict[int, dict] = {}
        used = set()
        for i in np.argsort(-cos.max(axis=1)):  # greedy, best matches first
            j = int(np.argmax(cos[i]))
            if cos[i, j] >= cos_threshold and j not in used and i in active:
                ch = active[i]
                ch["last"] = l + 1
                new_active[j] = ch
                used.add(j)
        for j in range(len(dirs[l + 1])):        # births
            if j not in new_active:
                ch = {"birth": l + 1, "last": l + 1}
                chains.append(ch)
                new_active[j] = ch
        active = new_active

    class _F:  # minimal Feature/FeatureMap shims for summarize()
        def __init__(self, ch):
            self.birth_layer = ch["birth"]
            self.death_layer = ch["last"]
            self.lifespan = ch["last"] - ch["birth"] + 1

    class _FM:
        def __init__(self, feats):
            self.features = feats
            self.n_features = len(feats)
            self.n_persistent = sum(1 for f in feats if f.lifespan >= 5)
            self.n_transient = sum(1 for f in feats if f.lifespan <= 2)

    return _FM([_F(ch) for ch in chains])


def summarize(fmaps: list, n_layers: int) -> dict:
    spanning = [sum(1 for f in fm.features
                    if f.birth_layer <= 2 and f.death_layer >= n_layers - 3)
                for fm in fmaps]
    lifespans = [f.lifespan for fm in fmaps for f in fm.features]
    births = [f.birth_layer for fm in fmaps for f in fm.features]
    return {
        "n_features_mean": round(float(np.mean([fm.n_features for fm in fmaps])), 1),
        "n_persistent_mean": round(float(np.mean([fm.n_persistent for fm in fmaps])), 1),
        "n_transient_mean": round(float(np.mean([fm.n_transient for fm in fmaps])), 1),
        "depth_spanning_mean": round(float(np.mean(spanning)), 1),
        "lifespan_median": float(np.median(lifespans)) if lifespans else 0.0,
        "birth_hist_L0_2": round(float(np.mean([b <= 2 for b in births])), 3) if births else 0.0,
    }


def main() -> None:
    rng = np.random.default_rng(11)
    results = {}
    for run_id in args.run_ids:
        run_dir = CORPUS_DIR / run_id
        paths = net_paths(run_id)[:args.max_nets]
        prov = json.loads((run_dir / f"{paths[0].stem}.json").read_text())
        d0 = np.load(paths[0])
        n_layers = sum(1 for k in d0.files if k.startswith("init_w"))
        width = int(d0["init_w0"].shape[0])
        task = get_task(prov["task"], width)
        x, _ = task.sample(args.n_samples, rng)
        x = x.astype(np.float64)

        fmaps_t, fmaps_i = [], []
        for p in paths:
            d = np.load(p)
            fmaps_t.append(net_feature_map([d[f"w{i}"] for i in range(n_layers)],
                                           x, f"{run_id}/{p.stem}",
                                           args.cos_threshold, args.transport))
            fmaps_i.append(net_feature_map([d[f"init_w{i}"] for i in range(n_layers)],
                                           x, f"{run_id}/{p.stem}:init",
                                           args.cos_threshold, args.transport))
            print(f"{run_id}/{p.stem} done", flush=True)
        results[run_id] = {"C": prov["task"]["n_classes"],
                           "trained": summarize(fmaps_t, n_layers),
                           "init": summarize(fmaps_i, n_layers)}

    out_path = CENSUS_DIR / ("feature_tracking_transport.json"
                             if args.transport else "feature_tracking.json")
    out_path.write_text(json.dumps(
        {"cos_threshold": args.cos_threshold, "n_samples": args.n_samples,
         "max_nets": args.max_nets, "runs": results,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}\n")

    print(f"{'run':<22} {'C':>4} {'fam':>8} {'n_feat':>7} {'persist':>8} "
          f"{'span':>5} {'life_med':>9} {'birth<=2':>9}")
    for run_id, r in results.items():
        for fam in ("trained", "init"):
            s = r[fam]
            print(f"{run_id:<22} {r['C']:>4} {fam:>8} {s['n_features_mean']:>7.1f} "
                  f"{s['n_persistent_mean']:>8.1f} {s['depth_spanning_mean']:>5.1f} "
                  f"{s['lifespan_median']:>9.1f} {s['birth_hist_L0_2']:>9.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-id", dest="run_ids", action="append", required=True)
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--max-nets", type=int, default=6)
    p.add_argument("--cos-threshold", type=float, default=0.5)
    p.add_argument("--transport", action="store_true",
                   help="transport-corrected matching (required for plain MLPs)")
    args = p.parse_args()
    main()
