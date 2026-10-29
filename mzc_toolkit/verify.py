"""Stage 2: needs running data."""
import numpy as np


def _forward(Ws, X):
    h = X
    for W in Ws: h = np.maximum(h @ W, 0.0)
    return h


def mean_activation(Ws, X):
    """Per-neuron mean final-layer post-ReLU activation over inputs X (ARC's quantity)."""
    return _forward(Ws, X).mean(0)


def fidelity_curve(Ws, build, ks, X):
    """eps(k) = ||m' - m||^2 / ||m||^2 between the original net's mean activations and those of build(k) (paired inputs X). Compare eps with an
    estimator's own relative error on the same net: a discard is 'free' only if eps is below it."""
    m = mean_activation(Ws, X); den = float((m ** 2).sum())
    return {int(k): float(((mean_activation(build(k), X) - m) ** 2).sum() / den) for k in ks}


def accuracy(Ws, head, X, y):
    return float(((_forward(Ws, X) @ head).argmax(1) == y).mean())
