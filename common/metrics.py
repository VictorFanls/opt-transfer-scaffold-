import numpy as np

def igd_plus(P, PF):
    """Tiny IGD+ helper for 2D demo."""
    P = np.asarray(P); PF = np.asarray(PF)
    d = []
    for y in PF:
        d.append(np.min(np.maximum(P - y, 0.0).sum(axis=1)))
    return float(np.mean(d))
