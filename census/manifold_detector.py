"""
manifold_detector.py — Unsupervised activation manifold census.

Given raw activations at each layer (from diverse, unlabeled text),
measures how many distinct organized directions exist in the activation
space — the "manifold count" — and how much of that structure is
explained by known concept directions.

Core metrics per layer
----------------------
  effective_dim       Participation ratio of the eigenvalue spectrum:
                      PR = (Σλᵢ)² / Σλᵢ² — the effective number of
                      dimensions carrying variance.  A single dominant
                      direction → PR≈1; uniform spread → PR≈d.

  significant_dims    Number of eigenvalues exceeding a noise floor
                      (Marchenko-Pastur upper edge for the sample size),
                      i.e. the count of directions with more variance
                      than you'd expect from random data.

  concept_coverage    Fraction of total variance explained by projecting
                      onto the subspace spanned by known concept directions.

  residual_dim        Effective dimensionality of the residual after
                      projecting out known concept directions.

  top_eigenvalues     The largest eigenvalues (for spectrum visualization).

Typical usage
-------------
    from rosetta_tools.manifold_detector import layer_manifold_census

    # acts: list of [n_samples, hidden_dim] arrays, one per layer
    # concept_dirs: dict mapping concept name → [hidden_dim] direction vector
    census = layer_manifold_census(acts, concept_dirs)
    for layer_result in census:
        print(f"L{layer_result.layer}: {layer_result.significant_dims} significant dims, "
              f"{layer_result.concept_coverage:.1%} explained by known concepts")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerManifoldResult:
    """Manifold census for a single layer."""

    layer: int
    hidden_dim: int
    n_samples: int

    # Eigenvalue spectrum
    effective_dim: float          # Participation ratio
    significant_dims: int         # Dims above MP noise floor
    total_variance: float         # Sum of all eigenvalues
    top_eigenvalues: list[float]  # Largest eigenvalues (for plotting)

    # Principal component directions (the actual features)
    # Shape: [n_stored, hidden_dim] — unit vectors for the top PCs.
    # None if store_directions=False to save memory in multi-model runs.
    top_directions: NDArray | None

    # Known concept coverage
    concept_coverage: float       # Fraction of variance in concept subspace
    concept_dims: int             # Rank of concept subspace
    per_concept_variance: dict[str, float]  # Variance explained per concept dir

    # Residual (unexplained) structure
    residual_dim: float           # Effective dim after projecting out concepts
    residual_significant: int     # Significant dims in residual
    residual_variance: float      # Total variance in residual

    # Concept alignment — how much do individual concept dirs
    # align with the TOP principal components?
    concept_pc_alignment: dict[str, list[float]]  # concept → cos² with top PCs


@dataclass
class ManifoldCensus:
    """Full census across all layers."""

    n_layers: int
    hidden_dim: int
    n_samples: int
    concept_names: list[str]
    layers: list[LayerManifoldResult]

    def summary_arrays(self) -> dict[str, NDArray]:
        """Return arrays for easy plotting."""
        return {
            "effective_dim": np.array([lr.effective_dim for lr in self.layers]),
            "significant_dims": np.array([lr.significant_dims for lr in self.layers]),
            "concept_coverage": np.array([lr.concept_coverage for lr in self.layers]),
            "residual_dim": np.array([lr.residual_dim for lr in self.layers]),
            "residual_significant": np.array([lr.residual_significant for lr in self.layers]),
            "depth_pct": np.array([100 * lr.layer / self.n_layers for lr in self.layers]),
        }


# ---------------------------------------------------------------------------
# Marchenko-Pastur noise floor
# ---------------------------------------------------------------------------


def _mp_upper_edge(n_samples: int, n_features: int, variance: float = 1.0) -> float:
    """Upper edge of the Marchenko-Pastur distribution.

    Eigenvalues above this threshold in the sample covariance matrix
    are unlikely to arise from random (null) data.

    Parameters
    ----------
    n_samples:
        Number of data points.
    n_features:
        Dimensionality of the data.
    variance:
        Assumed per-feature variance under the null (σ²).

    Returns
    -------
    float
        The upper edge λ₊ = σ² × (1 + √(n_features/n_samples))²
    """
    gamma = n_features / n_samples
    return variance * (1 + np.sqrt(gamma)) ** 2


def _mp_median_factor(gamma: float, n_grid: int = 20_000) -> float:
    """Median of the Marchenko-Pastur distribution at σ²=1, aspect γ ≤ 1.

    Computed by numeric quadrature of the MP density over its support
    [(1-√γ)², (1+√γ)²]. Used to turn an observed median eigenvalue into a
    robust bulk-variance estimate: σ̂² = median(λ) / factor.
    """
    g = min(max(gamma, 1e-9), 1.0)
    lo, hi = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2
    # substitute x = u²: removes the x^{-1/2} singularity at γ=1 (lo=0),
    # density in u is sqrt((hi-u²)(u²-lo)) / (πγu), smooth on (√lo, √hi)
    u = np.linspace(np.sqrt(lo) + 1e-9, np.sqrt(hi) - 1e-9, n_grid)
    density = np.sqrt(np.maximum((hi - u * u) * (u * u - lo), 0.0)) / (np.pi * g * u)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    u_med = u[int(np.searchsorted(cdf, 0.5))]
    return float(u_med * u_med)


def estimate_mp_variance(eigenvalues: NDArray, n_samples: int,
                         n_features: int) -> float:
    """Robust bulk-variance estimate: median eigenvalue / MP median factor.

    Scale-normalized alternative to a fixed analytic σ²: the median is
    insensitive to a minority of spike (structure) eigenvalues, so this
    recovers the bulk scale even when the overall weight scale has drifted
    (weight decay, spectral growth) — see PROVENANCE.md and the wd≥0.3
    finding.

    Validity: 1.00–1.11 × truth on synthetic rescaled-Gaussian bulks at
    scales 0.02–2.0 (NOT exact). Two known failure modes:
    (1) more than ~half the spectrum is structure — the median lands in the
        structured part;
    (2) an ANNIHILATED bulk (strong weight decay: near-zero/dead rows) is not
        a rescaled Gaussian bulk at all — on real wd=1 matrices this estimator
        reports 29–82 "significant" dims for true structural rank ~7–9
        (2026-08-13 external audit). Check `bulk_regime` first; in the
        depleted regime use rank metrics (participation ratio), not MP counts.
    """
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    eigs = eigs[eigs > 0]
    if len(eigs) == 0:
        return 1.0
    gamma = min(n_features / n_samples, 1.0)
    return float(np.median(eigs)) / _mp_median_factor(gamma)


# ---------------------------------------------------------------------------
# Participation ratio
# ---------------------------------------------------------------------------


def _participation_ratio(eigenvalues: NDArray) -> float:
    """Effective dimensionality via participation ratio.

    PR = (Σλᵢ)² / Σλᵢ²

    Equals 1 when all variance is in one direction,
    equals d when variance is uniformly spread.
    """
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    eigs = eigs[eigs > 0]
    if len(eigs) == 0:
        return 0.0
    sum_sq = float(np.sum(eigs) ** 2)
    sq_sum = float(np.sum(eigs ** 2))
    if sq_sum == 0:
        return 0.0
    return sum_sq / sq_sum


# ---------------------------------------------------------------------------
# Core: single-layer census
# ---------------------------------------------------------------------------


def _layer_census(
    activations: NDArray,
    layer: int,
    concept_directions: dict[str, NDArray],
    n_top_eigenvalues: int = 50,
    store_directions: bool = False,
    noise_variance: float | None = None,
    center: bool = True,
) -> LayerManifoldResult:
    """Compute manifold census for one layer.

    Parameters
    ----------
    activations:
        Shape ``[n_samples, hidden_dim]``, float32 or float64.
    layer:
        Layer index.
    concept_directions:
        Mapping of concept name → unit direction vector ``[hidden_dim]``.
    n_top_eigenvalues:
        How many eigenvalues to store for visualization.
    store_directions:
        If True, store the top principal component directions (eigenvectors)
        in the result.  Required for cross-layer feature tracking.
        Costs ~n_top × hidden_dim × 8 bytes per layer.
    noise_variance:
        Per-feature variance σ² assumed under the null for the
        Marchenko-Pastur threshold.  If None (default, original behavior),
        σ² is self-calibrated as the observed mean eigenvalue — fine within
        one matrix, but NOT comparable across matrices of different scale.
        For He-Gaussian weight matrices pass the analytic null 2/fan_in.
    center:
        If True (default, original behavior), subtract the per-column mean
        before the spectrum.  For weight matrices the mean-row direction may
        itself be learned structure; pass False to census raw second moments.
    """
    acts = np.asarray(activations, dtype=np.float64)
    n_samples, hidden_dim = acts.shape

    # Center the activations (optional for weight-matrix censuses)
    acts_centered = acts - acts.mean(axis=0) if center else acts

    # ── Eigenvalue spectrum via SVD (more numerically stable than eigh) ──
    # For n_samples < hidden_dim, it's faster to compute the Gram matrix.
    # SVD of centered data: singular values² / (n-1) = eigenvalues of covariance.
    if n_samples <= hidden_dim:
        # Economy SVD
        _, s, Vt = np.linalg.svd(acts_centered, full_matrices=False)
        eigenvalues = (s ** 2) / (n_samples - 1)
        # Vt rows are eigenvectors (principal components)
        pc_directions = Vt  # shape [min(n,d), hidden_dim]
    else:
        # Full covariance (np.cov re-centers; use the raw second-moment
        # matrix when centering is disabled)
        if center:
            cov = np.cov(acts_centered, rowvar=False)
        else:
            cov = (acts_centered.T @ acts_centered) / (n_samples - 1)
        eigenvalues_raw, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending order — reverse
        idx = np.argsort(eigenvalues_raw)[::-1]
        eigenvalues = eigenvalues_raw[idx]
        pc_directions = eigvecs[:, idx].T  # shape [hidden_dim, hidden_dim] → transposed

    eigenvalues = np.maximum(eigenvalues, 0)  # clip numerical negatives
    total_variance = float(np.sum(eigenvalues))

    # ── Effective dimensionality ──
    effective_dim = _participation_ratio(eigenvalues)

    # ── Significant dimensions (above Marchenko-Pastur noise floor) ──
    # Null σ²: explicit if provided (comparable across matrices), else
    # self-calibrated as the observed mean eigenvalue (original behavior)
    if noise_variance is not None:
        mp_var = float(noise_variance)
    else:
        mp_var = float(np.mean(eigenvalues)) if len(eigenvalues) > 0 else 1.0
    mp_edge = _mp_upper_edge(n_samples, hidden_dim, variance=mp_var)
    significant_dims = int(np.sum(eigenvalues > mp_edge))

    # ── Known concept coverage ──
    # Build orthonormal basis for the concept subspace
    concept_names = sorted(concept_directions.keys())
    per_concept_var = {}

    if concept_names:
        concept_vecs = np.array([
            concept_directions[c] / (np.linalg.norm(concept_directions[c]) + 1e-12)
            for c in concept_names
        ])  # shape [n_concepts, hidden_dim]

        # Orthogonalize via QR to get the concept subspace basis
        Q, _ = np.linalg.qr(concept_vecs.T)  # Q: [hidden_dim, n_concepts]
        concept_rank = Q.shape[1]

        # Project centered activations onto concept subspace
        projections = acts_centered @ Q  # [n_samples, concept_rank]
        concept_variance = float(np.sum(np.var(projections, axis=0, ddof=1)))
        concept_coverage = concept_variance / total_variance if total_variance > 0 else 0.0

        # Per-concept: variance along each individual direction
        for c_name, c_vec in zip(concept_names, concept_vecs):
            proj_1d = acts_centered @ c_vec  # [n_samples]
            per_concept_var[c_name] = float(np.var(proj_1d, ddof=1))

        # ── Residual structure ──
        # Project OUT the concept subspace
        residual = acts_centered - (acts_centered @ Q) @ Q.T
        # SVD of residual
        if n_samples <= hidden_dim:
            _, s_res, _ = np.linalg.svd(residual, full_matrices=False)
            res_eigenvalues = (s_res ** 2) / (n_samples - 1)
        else:
            res_cov = np.cov(residual, rowvar=False)
            res_eigenvalues = np.linalg.eigvalsh(res_cov)[::-1]
        res_eigenvalues = np.maximum(res_eigenvalues, 0)

        residual_dim = _participation_ratio(res_eigenvalues)
        residual_variance = float(np.sum(res_eigenvalues))

        # Significant dims in residual
        if noise_variance is not None:
            res_var = float(noise_variance)
        else:
            res_var = float(np.mean(res_eigenvalues)) if len(res_eigenvalues) > 0 else 1.0
        mp_edge_res = _mp_upper_edge(n_samples, hidden_dim - concept_rank, variance=res_var)
        residual_significant = int(np.sum(res_eigenvalues > mp_edge_res))

    else:
        concept_coverage = 0.0
        concept_rank = 0
        residual_dim = effective_dim
        residual_significant = significant_dims
        residual_variance = total_variance

    # ── Concept-PC alignment ──
    # For each concept direction, how much does it align with the top PCs?
    n_pcs = min(n_top_eigenvalues, len(pc_directions))
    concept_pc_alignment = {}
    for c_name in concept_names:
        c_vec = concept_directions[c_name]
        c_unit = c_vec / (np.linalg.norm(c_vec) + 1e-12)
        # cos² with each top PC
        cos_sq = [(float(np.dot(c_unit, pc_directions[i])) ** 2)
                  for i in range(n_pcs)]
        concept_pc_alignment[c_name] = cos_sq

    return LayerManifoldResult(
        layer=layer,
        hidden_dim=hidden_dim,
        n_samples=n_samples,
        effective_dim=effective_dim,
        significant_dims=significant_dims,
        total_variance=total_variance,
        top_eigenvalues=eigenvalues[:n_top_eigenvalues].tolist(),
        top_directions=pc_directions[:n_top_eigenvalues].copy() if store_directions else None,
        concept_coverage=concept_coverage,
        concept_dims=concept_rank if concept_names else 0,
        per_concept_variance=per_concept_var,
        residual_dim=residual_dim,
        residual_significant=residual_significant,
        residual_variance=residual_variance,
        concept_pc_alignment=concept_pc_alignment,
    )


# ---------------------------------------------------------------------------
# Public API: full census
# ---------------------------------------------------------------------------


def layer_manifold_census(
    layer_activations: list[NDArray],
    concept_directions: dict[str, NDArray] | None = None,
    n_top_eigenvalues: int = 50,
    store_directions: bool = False,
    noise_variance: float | None = None,
    center: bool = True,
) -> ManifoldCensus:
    """Compute manifold census across all layers.

    Parameters
    ----------
    layer_activations:
        List of arrays, one per layer.  Each has shape
        ``[n_samples, hidden_dim]``.
    concept_directions:
        Optional dict mapping concept name → direction vector
        ``[hidden_dim]`` (does not need to be unit-normalized).
        If provided, computes concept coverage and residual structure.
    n_top_eigenvalues:
        Number of top eigenvalues to store per layer.
    noise_variance:
        Explicit null σ² for the Marchenko-Pastur threshold (see
        ``_layer_census``).  For He-Gaussian weight matrices: 2/fan_in.
    center:
        Subtract per-column means before the spectrum (see ``_layer_census``).

    Returns
    -------
    ManifoldCensus
        Full census with per-layer results.
    """
    if not layer_activations:
        raise ValueError("No layer activations provided")

    concept_dirs = concept_directions or {}
    n_samples, hidden_dim = layer_activations[0].shape
    n_layers = len(layer_activations)

    results = []
    for i, acts in enumerate(layer_activations):
        results.append(_layer_census(acts, i, concept_dirs, n_top_eigenvalues,
                                     store_directions, noise_variance, center))

    return ManifoldCensus(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_samples=n_samples,
        concept_names=sorted(concept_dirs.keys()),
        layers=results,
    )
