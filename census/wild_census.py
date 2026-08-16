"""
wild_census.py — First contact: the weight census on wild-caught pretrained
models (X3; AMC Q10's LLM half).

Wild models have no known init anchor (not He, checkpoint init unknown), so
this uses the wild-caught rule from F7: the robust scaled MP floor +
bulk_regime flag — plus an assumption-free EMPIRICAL null: for every weight
matrix, an entry-shuffled copy (same marginal distribution, structure
destroyed). Reported per matrix:

  sig_scaled(trained) vs sig_scaled(shuffled)  — structure count vs its null
  eff dim (trained/shuffled), scale_ratio*, bulk_regime

*scale_ratio here is vs the matrix's own shuffled bulk (≈1 by construction
for intact bulks; the flag still catches degenerate spectra).

Covers the MLP in/out projections per transformer block (rectangular —
γ = out/in handled by the MP machinery). Extraction is CPU-only.

Usage: python census/wild_census.py --model pythia-70m [--model pythia-160m]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "census"))
from manifold_detector import (_mp_upper_edge, _participation_ratio,
                               estimate_mp_variance)

CENSUS_DIR = REPO_ROOT / "census"

MODELS = {
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
}


def matrix_census(W: np.ndarray, rng: np.random.Generator) -> dict:
    """Scaled census + shuffled-copy empirical null for one (in, out) matrix.

    Orientation: always census the smaller Gram side (γ = d/n ≤ 1). With
    γ > 1 the spectrum is (d−n) exact zeros and the median-MP estimator
    collapses — first caught by the shuffled null reading sig = full rank on
    pythia mlp_in matrices (768×3072)."""
    if W.shape[1] > W.shape[0]:
        W = W.T
    n, d = W.shape

    def spectrum(M):
        eigs = np.linalg.eigvalsh(M.T @ M / (n - 1))[::-1]
        return np.maximum(eigs, 0.0)

    def counts(eigs):
        est = estimate_mp_variance(eigs, n, d)
        sig = int(np.sum(eigs > _mp_upper_edge(n, d, est)))
        return sig, est, round(_participation_ratio(eigs), 1)

    eigs_t = spectrum(W)
    W_shuf = rng.permutation(W.ravel()).reshape(W.shape)
    eigs_s = spectrum(W_shuf)
    sig_t, est_t, eff_t = counts(eigs_t)
    sig_s, est_s, eff_s = counts(eigs_s)
    scale_vs_shuffled = est_t / max(est_s, 1e-300)
    return {
        "shape": [int(n), int(d)],
        "sig_scaled": sig_t, "sig_shuffled_null": sig_s,
        "eff_dim": eff_t, "eff_dim_shuffled": eff_s,
        "scale_vs_shuffled": round(float(scale_vs_shuffled), 4),
        "bulk_regime": ("intact" if 0.25 <= scale_vs_shuffled <= 4.0
                        else "depleted"),
        "top5_over_edge": [round(float(e / _mp_upper_edge(n, d, est_t)), 2)
                           for e in eigs_t[:5]],
    }


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    rng = np.random.default_rng(0)
    results = {}
    for name in args.models:
        hf_id = MODELS[name]
        print(f"loading {hf_id}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(hf_id,
                                                     torch_dtype=torch.float32)
        variants = {"trained": model}
        if args.reinit:
            # the model's own initializer (GPT-NeoX init from config) — the
            # matched "zero learned structure" baseline for wild models
            from transformers import AutoConfig
            torch.manual_seed(0)
            variants["reinit"] = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(hf_id))
        layers = {}
        for variant, m in variants.items():
            for i, block in enumerate(m.gpt_neox.layers):
                for tag, lin in (("mlp_in", block.mlp.dense_h_to_4h),
                                 ("mlp_out", block.mlp.dense_4h_to_h)):
                    W = lin.weight.detach().numpy().T.astype(np.float64)
                    layers[f"L{i}_{tag}" + ("" if variant == "trained"
                                            else ":reinit")] = \
                        matrix_census(W, rng)
            print(f"  {name} {variant} done", flush=True)
        results[name] = layers
        del model, variants

    out_path = CENSUS_DIR / "wild_census.json"
    out_path.write_text(json.dumps(
        {"models": results, "null": "entry-shuffled copy per matrix",
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}\n")

    for name, layers in results.items():
        print(f"== {name} ==")
        print(f"{'matrix':<14} {'shape':>12} {'sig':>5} {'sig_null':>9} "
              f"{'eff':>7} {'eff_null':>9} {'regime':>9}")
        for key, r in layers.items():
            print(f"{key:<14} {str(tuple(r['shape'])):>12} {r['sig_scaled']:>5} "
                  f"{r['sig_shuffled_null']:>9} {r['eff_dim']:>7} "
                  f"{r['eff_dim_shuffled']:>9} {r['bulk_regime']:>9}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", dest="models", action="append",
                   choices=list(MODELS.keys()), required=True)
    p.add_argument("--reinit", action="store_true",
                   help="also census the model's own random init (matched null)")
    args = p.parse_args()
    main()
