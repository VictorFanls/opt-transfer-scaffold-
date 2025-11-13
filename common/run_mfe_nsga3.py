import argparse, os, numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.ref_dirs import get_reference_directions
from mfe.problems import TNK
from mfe.scheduler import SelectiveHFPolicy
from mfe.evaluator import MixedFidelityEvaluator
from common.plotting import scatter2d, ensure_dir

class UpgradeCallback(Callback):
    def __init__(self, evaluator, outdir):
        super().__init__()
        self.eval = evaluator; self.outdir = outdir; ensure_dir(outdir)
    def notify(self, algorithm):
        pop = algorithm.pop
        X = pop.get("X"); F_low = pop.get("F"); G_low = pop.get("G")
        F_final, G_final, mask = self.eval.upgrade(X, F_low, G_low)
        pop.set("F", F_final); pop.set("G", G_final)
        scatter2d(F_low,  f"Low fidelity (gen {algorithm.n_gen})",   os.path.join(self.outdir, f"low_gen{algorithm.n_gen}.png"))
        scatter2d(F_final,f"Final after upgrades (gen {algorithm.n_gen})", os.path.join(self.outdir, f"final_gen{algorithm.n_gen}.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop", type=int, default=92)
    ap.add_argument("--upgrade_frac", type=float, default=0.2)
    ap.add_argument("--outdir", type=str, default="outputs/mfe_tnk")
    args = ap.parse_args()

    low, high = TNK(coarse=True), TNK(coarse=False)
    evaluator = MixedFidelityEvaluator(low, high, SelectiveHFPolicy(args.upgrade_frac))
    ref_dirs = get_reference_directions("das-dennis", 2, n_points=args.pop)
    algo = NSGA3(pop_size=args.pop, ref_dirs=ref_dirs)

    res = minimize(low, algo, termination=get_termination("n_gen", args.generations),
                   seed=42, save_history=False, verbose=True,
                   callback=UpgradeCallback(evaluator, args.outdir))
    print(f"Done. High-fidelity re-evaluations: {evaluator.hf_calls}")
    np.savetxt(os.path.join(args.outdir, "final_F.csv"), res.F, delimiter=",")
    np.savetxt(os.path.join(args.outdir, "final_X.csv"), res.X, delimiter=",")
    print(f"Outputs saved to {args.outdir}")

if __name__ == "__main__":
    main()
