import numpy as np

class SelectiveHFPolicy:
    """简单门控：按“可行性+目标总和”打分，选前 k% 升级高保真"""
    def __init__(self, upgrade_frac=0.2):
        self.upgrade_frac = float(upgrade_frac)

    def decide(self, F, G):
        feasible = (G <= 0).all(axis=1).astype(float)
        score = -F.sum(axis=1) + 10.0*feasible
        k = max(1, int(len(F)*self.upgrade_frac))
        idx = np.argsort(-score)[:k]
        mask = np.zeros(len(F), dtype=bool); mask[idx] = True
        return mask
