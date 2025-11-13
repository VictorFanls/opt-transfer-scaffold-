import numpy as np

class MixedFidelityEvaluator:
    def __init__(self, problem_low, problem_high, policy):
        self.low  = problem_low
        self.high = problem_high
        self.policy = policy
        self.hf_calls = 0

    def low_eval(self, X):
        F, G = [], []
        for x in X:
            out = {}; self.low._evaluate(x, out)
            F.append(out['F']); G.append(out['G'])
        return np.array(F), np.array(G)

    def upgrade(self, X, F_low, G_low):
        mask = self.policy.decide(F_low, G_low)
        F_final = F_low.copy(); G_final = G_low.copy()
        for i, x in enumerate(X):
            if mask[i]:
                out = {}; self.high._evaluate(x, out)
                F_final[i] = out['F']; G_final[i] = out['G']
                self.hf_calls += 1
        return F_final, G_final, mask
