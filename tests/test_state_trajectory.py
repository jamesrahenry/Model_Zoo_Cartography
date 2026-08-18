"""Pin null_baseline/state_trajectory.py — the instrument behind F2 / §4.2.

The Hermite chain internals (`step_np`, `state_basis`) are covered in
test_chain_state_keyed.py; until now the wrapper that produces the paper's
actual numbers — `trajectory()` and the null-band-plus-z-score protocol — was
not. Exact anchors first (identity and rank-one input layers give q values
computable by hand), then the band protocol on a small synthetic ensemble.
"""
from __future__ import annotations

import numpy as np
import pytest

from state_trajectory import trajectory

W_DIM = 16


def he_weights(width: int, depth: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, np.sqrt(2.0 / width), (width, width))
            for _ in range(depth)]


def test_layer0_q_is_one_for_identity_weights():
    # Sigma_pre(L0) = W^T I W = I for W = I, so q = PR(I)/w = 1 exactly.
    out = trajectory([np.eye(8)])
    assert out["q"][0] == pytest.approx(1.0, abs=1e-9)


def test_layer0_q_is_one_over_w_for_rank_one_weights():
    # W = u v^T gives Sigma_pre = |u|^2 v v^T: rank one, PR = 1, q = 1/w.
    rng = np.random.default_rng(1)
    u, v = rng.standard_normal(8), rng.standard_normal(8)
    out = trajectory([np.outer(u, v)])
    assert out["q"][0] == pytest.approx(1.0 / 8.0, abs=1e-9)


def test_descriptor_lengths_and_ranges():
    depth = 5
    out = trajectory(he_weights(W_DIM, depth, seed=2))
    for key in ("q", "abar", "asd", "rbar"):
        assert len(out[key]) == depth
    q = np.array(out["q"])
    assert np.all(q > 0) and np.all(q <= 1.0)
    r = np.array(out["rbar"])
    assert np.all(r >= 0) and np.all(r <= 1.0)


def test_random_nets_rank_collapse_with_depth():
    # The depth-driven fixed point of F2: on random He nets the clock decays.
    out = trajectory(he_weights(W_DIM, 8, seed=3))
    q = out["q"]
    assert q[-1] < q[0]
    # and by a lot, not marginally (w=16 collapses fast)
    assert q[-1] < 0.6 * q[0]


def test_null_band_protocol_separates_structured_weights():
    """The F2 protocol in miniature: build a fresh-init null band, then place
    trajectories against it as z-scores. A held-out random net must read
    in-band everywhere; a rank-collapsed (structured) net must read far below
    the band at layer 0."""
    depth = 6
    ensemble = [trajectory(he_weights(W_DIM, depth, seed=100 + i))["q"]
                for i in range(10)]
    band = np.array(ensemble)
    mean, sd = band.mean(axis=0), band.std(axis=0)
    assert np.all(sd > 0)

    # Self-consistency: an 11th random init stays within 6 sigma per layer.
    fresh = np.array(trajectory(he_weights(W_DIM, depth, seed=999))["q"])
    z_fresh = (fresh - mean) / sd
    assert np.all(np.abs(z_fresh) < 6.0)

    # Structured net: rank-one every layer (scaled to He Frobenius norm so
    # the difference is shape, not amplitude). q = 1/w at L0 vs band ~0.5.
    rng = np.random.default_rng(7)
    structured = []
    for W in he_weights(W_DIM, depth, seed=50):
        u, v = rng.standard_normal(W_DIM), rng.standard_normal(W_DIM)
        R = np.outer(u, v)
        structured.append(R * (np.linalg.norm(W) / np.linalg.norm(R)))
    q_struct = np.array(trajectory(structured)["q"])
    z_struct = (q_struct - mean) / sd
    assert z_struct[0] < -3.0
