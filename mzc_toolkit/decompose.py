"""Stage 1: needs a reference (the net's own init or an early checkpoint)."""
import numpy as np


def decompose(Ws, W0s):
    """Per layer W = alpha*W0 + Delta with alpha = <W, W0>/<W0, W0>. rho = mean over layers 3..depth-2 of ||Delta||_F / ||alpha W0||_F
    (predicts how much the net depends on its random draw: Spearman 0.90, held-out MAE 0.122 in our grids)."""
    out = []
    for W, W0 in zip(Ws, W0s):
        al = float((W * W0).sum() / (W0 * W0).sum()); D = W - al * W0
        s = np.linalg.svd(D, compute_uv=False); e = s ** 2
        out.append(dict(alpha=al, delta_rel=float(np.linalg.norm(D) / np.linalg.norm(al * W0)), delta_pr=float(e.sum() ** 2 / (e ** 2).sum()), delta_top8=float(e[:8].sum() / e.sum())))
    L = len(out)
    return dict(layers=out, rho=float(np.mean([o["delta_rel"] for o in out[3:L - 1]])) if L > 4 else None)


def aligned_slice_layer(W, W0, alpha, k, Wn):
    """Keep Delta in full, keep the part of alpha*W0 that touches the top-k input/output subspaces of Delta, replace the rest with alpha * (fresh draw Wn)."""
    D = W - alpha * W0
    u, _, vh = np.linalg.svd(D); U, V = u[:, :k], vh[:k].T
    R = U @ (U.T @ W0) + (W0 @ V) @ V.T - U @ ((U.T @ W0 @ V) @ V.T)
    rest = Wn - U @ (U.T @ Wn) - (Wn @ V) @ V.T + U @ ((U.T @ Wn @ V) @ V.T)
    return D + alpha * (R + rest)


def rebuild_with_slice(Ws, W0s, k, seed=99):
    """Whole-net version. k = 0 replaces the entire random remnant; k = n returns the original net."""
    rng = np.random.default_rng(seed); n = Ws[0].shape[0]
    return [aligned_slice_layer(W, W0, float((W * W0).sum() / (W0 * W0).sum()), k, rng.standard_normal((n, n)) * np.sqrt(2.0 / n)) for W, W0 in zip(Ws, W0s)]
