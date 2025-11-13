import argparse, os, numpy as np, matplotlib.pyplot as plt
from history.history_lib import HistoryLib
from history.reuse_gp import fit_reuse_gp, predict
from common.plotting import ensure_dir

def make_task(n=120, d=8, seed=0, shift=0.0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    w = rng.randn(d); w /= (np.linalg.norm(w) + 1e-9)
    y = np.sin(X @ w * 3.0 + shift) + 0.1*rng.randn(n)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_history", type=int, default=200)
    ap.add_argument("--n_new", type=int, default=40)
    ap.add_argument("--outdir", type=str, default="outputs/history_reuse")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    lib = HistoryLib(path=os.path.join(args.outdir, "history_lib.json"))

    # 若无历史就先造两条历史任务
    if not lib.load_all():
        Xh1, yh1 = make_task(n=args.n_history, d=8, seed=1, shift=0.2)
        Xh2, yh2 = make_task(n=args.n_history, d=8, seed=2, shift=-0.1)
        lib.add_task("hist_1", Xh1, yh1)
        lib.add_task("hist_2", Xh2, yh2)

    tasks = lib.load_all()
    X_hist = np.vstack([t[1] for t in tasks]); y_hist = np.hstack([t[2] for t in tasks])

    X_new, y_new = make_task(n=args.n_new, d=8, seed=123, shift=0.15)
    gp, scaler = fit_reuse_gp(X_hist, y_hist, X_new, y_new, alpha_hist=0.5, alpha_new=1.0)

    X_test, y_test = make_task(n=80, d=8, seed=999, shift=0.15)
    mu, std = predict(gp, scaler, X_test)
    mae = float(np.mean(np.abs(mu - y_test)))

    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"MAE on new-task test with history reuse: {mae:.4f}\\n")

    idx = np.argsort(mu)[:200]
    plt.figure()
    plt.scatter(mu[idx], y_test[idx], s=10)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("New task test (with history reuse)")
    plt.savefig(os.path.join(args.outdir, "pred_vs_true.png"), dpi=160, bbox_inches='tight')

    print(f"Done. Results saved to {args.outdir}. MAE={mae:.4f}")

if __name__ == "__main__":
    main()
