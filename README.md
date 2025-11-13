# opt-transfer-scaffold-
# Optimization Transfer Scaffold (MFE + t-SNE transfer + History reuse)

This is a minimal, runnable scaffold for:
- **MFE-NSGA-III** on analytic MOO problems (TNK).
- **t-SNE-based transfer** under non-overlapping variables (synthetic physics features).
- **Sequential history reuse** (simple multi-output warm start).

## Quickstart
pip install -r requirements.txt

# 1) MFE demo (TNK)
python run_mfe_nsga3.py --generations 60 --pop 92 --upgrade_frac 0.2

# 2) t-SNE transfer demo
python run_tsne_transfer.py --n_source 200 --n_target 60

# 3) History reuse demo
python run_history_reuse.py --n_history 200 --n_new 40

Outputs land in `outputs/`.
