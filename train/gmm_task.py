"""
gmm_task.py — Gaussian-mixture classification tasks whose aggregate input
distribution is exactly N(0, I).

The null baseline (null_baseline/analytic_vacuum.py, chain_port_refit*.py) is
derived at the operating point x ~ N(0, I_dim). This task family is built so
the *aggregate* distribution the network sees matches that premise exactly, by
construction, not approximately:

  1. Class means: `separation * ` rows of a random (dim x C) orthonormal basis,
     centered so the balanced mixture has mean exactly 0.
  2. Within-class noise: N(0, I), so aggregate covariance is
     Sigma = I + (1/C) * M^T M  (rank <= C bump along the mean directions).
  3. Exact analytic whitening by Sigma^{-1/2} (symmetric), applied to both
     means and noise. Whitened aggregate: mean 0, covariance I, exactly.

After whitening the mixture is still a Gaussian mixture; class information
lives in a C-dimensional subspace whose mean separation is shrunk by the
whitening — `effective_separation` below reports the post-whitening value.
`separation` is the difficulty dial (larger = easier), `n_classes` the task
size dial.

Marginals are near-N(0,1) for modest separation but not exactly Gaussian;
that residual mismatch with the null's Gaussianity assumption is recorded
per-net in corpus provenance (task family "gmm").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GMMTask:
    dim: int
    n_classes: int
    separation: float
    seed: int
    means: NDArray          # [C, dim] — post-whitening class means
    whiten: NDArray         # [dim, dim] — Sigma^{-1/2}, applied to raw samples
    effective_separation: float  # min pairwise post-whitening mean distance
    bayes_accuracy: float   # MC estimate of the optimal (LDA) accuracy

    def sample(self, n: int, rng: np.random.Generator) -> tuple[NDArray, NDArray]:
        """Draw n labelled samples. Returns (x [n, dim] float32, y [n] int64)."""
        y = rng.integers(0, self.n_classes, size=n)
        z = rng.standard_normal((n, self.dim))
        x = self.means[y] + z @ self.whiten  # whiten is symmetric: W z ~ N(0, W W)
        return x.astype(np.float32), y.astype(np.int64)

    def describe(self) -> dict:
        return {"family": "gmm", "n_classes": self.n_classes,
                "separation": self.separation,
                "effective_separation": round(self.effective_separation, 4),
                "bayes_accuracy": round(self.bayes_accuracy, 4),
                "task_seed": self.seed,
                "aggregate_distribution": "exact mean-0 cov-I (whitened)"}


def make_gmm_task(
    dim: int = 256,
    n_classes: int = 10,
    separation: float = 3.0,
    seed: int = 0,
) -> GMMTask:
    if n_classes > dim:
        raise ValueError("orthogonal-means construction requires n_classes <= dim")
    rng = np.random.default_rng(seed)

    # Random orthonormal mean directions, then center the balanced mixture.
    basis, _ = np.linalg.qr(rng.standard_normal((dim, n_classes)))
    raw_means = separation * basis.T          # [C, dim]
    raw_means -= raw_means.mean(axis=0)       # aggregate mean exactly 0

    # Aggregate covariance of (mean + N(0, I)) mixture, whitened exactly.
    sigma = np.eye(dim) + (raw_means.T @ raw_means) / n_classes
    evals, evecs = np.linalg.eigh(sigma)
    whiten = (evecs * evals**-0.5) @ evecs.T  # Sigma^{-1/2}, symmetric

    means = raw_means @ whiten
    diffs = means[:, None, :] - means[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    eff_sep = float(dists[np.triu_indices(n_classes, k=1)].min())

    task = GMMTask(
        dim=dim,
        n_classes=n_classes,
        separation=separation,
        seed=seed,
        means=means,
        whiten=whiten,
        effective_separation=eff_sep,
        bayes_accuracy=0.0,
    )

    # Bayes accuracy: classes share covariance whiten@whiten, so the optimal
    # classifier is exactly LDA — estimate its accuracy on a fixed MC draw.
    x, y = task.sample(100_000, np.random.default_rng(seed + 999))
    cov = whiten @ whiten
    proj = np.linalg.solve(cov, means.T)                       # [dim, C]
    scores = x @ proj - 0.5 * np.sum(means * proj.T, axis=1)
    task.bayes_accuracy = float((scores.argmax(axis=1) == y).mean())

    return task


if __name__ == "__main__":
    # Self-check: aggregate moments must match the null premise.
    task = make_gmm_task()
    rng = np.random.default_rng(1)
    x, y = task.sample(200_000, rng)
    cov = np.cov(x, rowvar=False)
    print(f"aggregate mean |max|: {np.abs(x.mean(0)).max():.4f}")
    print(f"aggregate cov: diag mean {np.diag(cov).mean():.4f}, "
          f"off-diag |max| {np.abs(cov - np.diag(np.diag(cov))).max():.4f}")
    print(f"effective separation (post-whitening): {task.effective_separation:.3f}")
    print(f"Bayes accuracy (LDA, MC): {task.bayes_accuracy:.4f}")
