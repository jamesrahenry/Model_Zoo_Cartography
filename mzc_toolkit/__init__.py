"""mzc_toolkit: instruments for finding structure in trained bias-free ReLU MLPs (x @ W convention, W[l] is n x n).

Stage 0 (weights only):          triage        -> alpha_hat per layer, outlier counts, cross-layer chain score
Stage 1 (weights + a reference): decompose     -> W = alpha*W0 + Delta, relative update size rho, aligned slice
Stage 2 (running data):          verify        -> accuracy / fidelity curves of a net rebuilt from the slice
Measured accuracy and failure regimes of every instrument are in README.md.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # 1024 x 1024 SVDs are ~20x slower with one thread per visible core
    os.environ.setdefault(_v, "4")
from .spectral import he_reference, alpha_hat, layer_subspaces, chain_score, outlier_count, triage
from .decompose import decompose, aligned_slice_layer, rebuild_with_slice
from .verify import mean_activation, fidelity_curve, accuracy
from .io import load_layers

__all__ = ["he_reference", "alpha_hat", "layer_subspaces", "chain_score", "outlier_count", "triage",
           "decompose", "aligned_slice_layer", "rebuild_with_slice", "mean_activation", "fidelity_curve", "accuracy", "load_layers"]
