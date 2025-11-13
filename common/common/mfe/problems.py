import numpy as np
from pymoo.core.problem import ElementwiseProblem

def tnk_true(x):
    x1, x2 = x[0], x[1]
    f1 = x1
    f2 = x2
    g1 = -x1**2 - x2**2 + 1 + 0.1*np.cos(16*np.arctan2(x1, x2))
    g2 = (x1 - 0.5)**2 + (x2 - 0.5)**2 - 0.5
    return np.array([f1, f2]), np.array([g1, g2])

def tnk_coarse(x):
    # 粗保真：人为加入偏差/噪声以模拟“便宜评估”
    f, g = tnk_true(x)
    f = f + 0.01*np.array([1.0, 1.0])
    g = g + 0.02*np.array([1.0, -1.0])
    return f, g

class TNK(ElementwiseProblem):
    def __init__(self, coarse=False):
        super().__init__(n_var=2, n_obj=2, n_constr=2, xl=np.array([0.0,0.0]), xu=np.array([np.pi, np.pi]))
        self.coarse = coarse

    def _evaluate(self, x, out, *args, **kwargs):
        if self.coarse: f, g = tnk_coarse(x)
        else:           f, g = tnk_true(x)
        out["F"] = f; out["G"] = g
