import argparse, os, numpy as np, matplotlib.pyplot as plt
from transfer.physics_features import make_source_target
from transfer.tsne_align import align_and_embed
from transfer.multioutput_gp import fit_multioutput_gp, predict_multioutput_gp
from common.plotting import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_source", type=int, default=200)
    ap.add_argument("--n_target", type=int, default=60)
    ap.add_argument("--outdir",  type=str, default="outputs/tsne_transfer")
    args = ap.parse_args()

    Xs, ys, Fs, Xt, yt, Ft = make_source_target(args.n_source, args.n_target)
    Xs_hat, Zs, Zt = align_and_embed(Fs, Ft, Xt)

    X_train = np.vstack([Xt, Xs_hat]); y_train = np.hstack([yt, ys])
    model = fit_multioutput_gp(X_train, y_train.reshape(-1,1))

    Xt_test, yt_test = Xt[:20], yt[:20]
    y_pred = predict_multioutput_gp(model, Xt_test).ravel()
    mae = float(np.mean(np.abs(y_pred - yt_test)))

    ensure_dir(args.outdir)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"MAE on target test (with transfer): {mae:.4f}\n")

    plt.figure()
    plt.scatter(Zs[:,0], Zs[:,1], s=10, label="Source")
    plt.scatter(Zt[:,0], Zt[:,1], s=10, label="Target")
    plt.legend(); plt.title("t-SNE alignment")
    plt.savefig(os.path.join(args.outdir, "tsne_alignment.png"), dpi=160, bbox_inches='tight')

    plt.figure()
    plt.scatter(yt_test, y_pred, s=10)
    plt.xlabel("True yt"); plt.ylabel("Predicted"); plt.title("Target test")
    plt.savefig(os.path.join(args.outdir, "target_pred.png"), dpi=160, bbox_inches='tight')

    print(f"Done. Results saved to {args.outdir}. MAE={mae:.4f}")

if __name__ == "__main__":
    main()
