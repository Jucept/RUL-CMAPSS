# src/scripts/plot_preds.py
import pathlib, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np

def load_preds(preds_pattern):
    files = sorted(pathlib.Path(preds_pattern).parent.glob(pathlib.Path(preds_pattern).name))
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def main(preds_glob, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_preds(preds_glob)
    # scatter pred vs true
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='y_true', y='y_pred', data=df, alpha=0.3, s=8)
    mn, mx = df.y_true.min(), df.y_true.max()
    plt.plot([mn,mx],[mn,mx],'r--', lw=1)
    plt.xlabel('y_true'); plt.ylabel('y_pred'); plt.title('True vs Pred')
    plt.savefig(out_dir/"pred_vs_true.png", dpi=150)
    plt.close()

    # residuals histogram
    df['resid'] = df['y_pred'] - df['y_true']
    plt.figure(figsize=(6,4))
    sns.histplot(df['resid'], bins=80, kde=True)
    plt.xlabel('residual (y_pred - y_true)'); plt.title('Residuals histogram')
    plt.savefig(out_dir/"residuals_hist.png", dpi=150)
    plt.close()

    # residuals vs true
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='y_true', y='resid', data=df, alpha=0.2, s=8)
    plt.xlabel('y_true'); plt.ylabel('residual'); plt.title('Residuals vs True')
    plt.axhline(0, color='r', ls='--', lw=1)
    plt.savefig(out_dir/"residuals_vs_true.png", dpi=150)
    plt.close()
    print("Saved 3 plots to", out_dir.resolve())