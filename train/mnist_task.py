"""
mnist_task.py — MNIST mapped into the null's operating point: 784 → 256 via a
fixed seeded Gaussian projection, then exact ZCA whitening fitted on the train
split. The network sees inputs with empirical mean 0 and covariance I (the
x ~ N(0, I) premise), but real-data higher moments — the "face validity"
family next to the exactly-Gaussian-aggregate GMM family.

Marginals are NOT Gaussian (that's the point); the residual mismatch with the
null's Gaussianity assumption is a recorded property of the family, same as
the GMM's mixture non-Gaussianity.

Training batches sample the train split with replacement; validation samples
the held-out test split (sample_val). bayes_accuracy is an EMPIRICAL CEILING:
the max accuracy achieved by this project's own architecture on the family
(measured 2026-08-16 over the 20/32-net corpora: mnist 0.9525, fashion
0.846) — the original 0.985 proxy was borrowed from a different architecture
and mislabeled every real-data net "partial" (audit item, F1 caveat).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DATA_ROOT = Path.home() / "rosetta_data" / "mnist"


@dataclass
class MNISTTask:
    dim: int
    n_classes: int
    seed: int
    dataset: str
    bayes_accuracy: float  # proxy ceiling, not exact — see module docstring
    x_train: NDArray
    y_train: NDArray
    x_test: NDArray
    y_test: NDArray

    def sample(self, n: int, rng: np.random.Generator) -> tuple[NDArray, NDArray]:
        idx = rng.integers(0, len(self.x_train), size=n)
        return self.x_train[idx], self.y_train[idx]

    def sample_val(self, n: int, rng: np.random.Generator) -> tuple[NDArray, NDArray]:
        n = min(n, len(self.x_test))
        idx = rng.choice(len(self.x_test), size=n, replace=False)
        return self.x_test[idx], self.y_test[idx]

    def describe(self) -> dict:
        return {"family": f"{self.dataset}_whitened", "n_classes": self.n_classes,
                "projection_seed": self.seed,
                "bayes_accuracy": self.bayes_accuracy,
                "bayes_source": "empirical_family_max_2026-08-16",
                "aggregate_distribution":
                    "exact mean-0 cov-I on train split (784->256 seeded "
                    "Gaussian projection + ZCA); non-Gaussian marginals"}


def make_mnist_task(dim: int = 256, seed: int = 0,
                    dataset: str = "mnist") -> MNISTTask:
    from torchvision.datasets import MNIST, FashionMNIST  # deferred import

    cls = {"mnist": MNIST, "fashion": FashionMNIST}[dataset]
    root = str(DATA_ROOT.parent / dataset)
    train = cls(root, train=True, download=True)
    test = cls(root, train=False, download=True)
    xtr = train.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    ytr = train.targets.numpy().astype(np.int64)
    xte = test.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    yte = test.targets.numpy().astype(np.int64)

    rng = np.random.default_rng(seed)
    P = rng.standard_normal((784, dim)) / np.sqrt(784)

    mu = xtr.mean(axis=0)
    ztr = (xtr - mu) @ P
    cov = np.cov(ztr, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    zca = (evecs * (evals + 1e-8) ** -0.5) @ evecs.T

    def transform(x: NDArray) -> NDArray:
        return (((x - mu) @ P) @ zca).astype(np.float32)

    ceiling = {"mnist": 0.9525, "fashion": 0.846}[dataset]
    return MNISTTask(dim=dim, n_classes=10, seed=seed, dataset=dataset,
                     bayes_accuracy=ceiling,
                     x_train=transform(xtr), y_train=ytr,
                     x_test=transform(xte), y_test=yte)


if __name__ == "__main__":
    task = make_mnist_task()
    cov = np.cov(task.x_train.astype(np.float64), rowvar=False)
    print(f"train mean |max|: {np.abs(task.x_train.mean(0)).max():.5f}")
    print(f"train cov: diag mean {np.diag(cov).mean():.5f}, "
          f"off-diag |max| {np.abs(cov - np.diag(np.diag(cov))).max():.5f}")
    covt = np.cov(task.x_test.astype(np.float64), rowvar=False)
    print(f"test  cov: diag mean {np.diag(covt).mean():.5f} "
          f"(transform fitted on train; test shift is a real-data property)")
