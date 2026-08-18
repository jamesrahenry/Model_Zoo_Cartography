"""Smoke-pin census/feature_tracker.py (§3.7).

The paper reports feature tracking as a calibrated negative result: raw and
transported matching fail in plain MLPs, continuity is recoverable only
through fitted rotations (§4.4). For that negative to be evidence, the
tracker itself must demonstrably work when continuity exists and break when
coordinates rotate — i.e., the failure is the data's, not the instrument's.
"""
from __future__ import annotations

import numpy as np

from feature_tracker import track_features

HID = 12
N_LAYERS = 8


def orthonormal_rows(k: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q.T[:k]


def test_stable_directions_track_as_persistent_features():
    basis = orthonormal_rows(3, HID, seed=0)          # same 3 PCs every layer
    dirs = [basis.copy() for _ in range(N_LAYERS)]
    eigs = [[3.0, 2.0, 1.0]] * N_LAYERS
    fm = track_features(dirs, eigs, n_layers_total=N_LAYERS)

    persistent = fm.persistent_features()
    assert len(persistent) == 3
    for f in persistent:
        assert f.birth_layer == 0
        assert f.death_layer == N_LAYERS - 1
        assert f.lifespan == N_LAYERS
        # perfect chains: every step matched at |cos| = 1
        assert min(f.cos_chain) > 0.99


def test_orthogonal_rotation_breaks_raw_tracking():
    """Rotate the code's coordinates mid-network: layers 0-3 use basis A,
    layers 4-7 an orthogonal basis B. No feature may span the boundary —
    this is the instrument-side ground truth for §4.4's rotation-hidden
    claim (raw tracking SHOULD fail across a coordinate change)."""
    q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((HID, HID)))
    basis_a, basis_b = q.T[:3], q.T[3:6]              # mutually orthogonal
    dirs = [basis_a.copy() for _ in range(4)] + [basis_b.copy() for _ in range(4)]
    eigs = [[3.0, 2.0, 1.0]] * N_LAYERS
    fm = track_features(dirs, eigs, n_layers_total=N_LAYERS)

    assert fm.n_persistent == 0                        # nothing lives 5+ layers
    for f in fm.features:
        # every tracked feature lives entirely on one side of the rotation
        assert f.death_layer <= 3 or f.birth_layer >= 4


def test_low_variance_pcs_are_filtered_not_tracked():
    basis = orthonormal_rows(3, HID, seed=2)
    dirs = [basis.copy() for _ in range(N_LAYERS)]
    # third PC below min_eigenvalue_frac of the layer total -> ignored
    eigs = [[10.0, 5.0, 0.01]] * N_LAYERS
    fm = track_features(dirs, eigs, n_layers_total=N_LAYERS,
                        min_eigenvalue_frac=0.05)
    assert len(fm.persistent_features()) == 2
