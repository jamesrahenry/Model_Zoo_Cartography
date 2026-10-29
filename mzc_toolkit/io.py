import numpy as np


def load_layers(path, prefix="w"):
    """Load layers from an .npz with keys <prefix>0, <prefix>1, ... (e.g. 'w' for trained, 'init_w' for the stored initialisation)."""
    z = np.load(path)
    keys = sorted((k for k in z.files if k.startswith(prefix) and k[len(prefix):].isdigit()), key=lambda k: int(k[len(prefix):]))
    if not keys: raise KeyError(f"no keys '{prefix}<int>' in {path}; found {z.files[:6]}")
    return [z[k].astype(np.float64) for k in keys]
