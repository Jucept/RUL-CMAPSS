# src/scripts/diagnose_worst_units.py
import pathlib, pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt, seaborn as sns

def main(preds_glob, out_dir, topk=5):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(pathlib.Path(preds_glob).parent.glob(pathlib.Path(preds_glob).name))], ignore_index=True)
    rmse_per_unit = df.groupby('unit').apply(lambda g: mean_squared_error(g['y_true'], g['y_pred'], squared=False)).sort_values(ascending=False)
    worst = rmse_per_unit.head(topk).index.tolist()
    diagnostics_dir = out_dir/"diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    for u in worst:
        sub = df[df['unit']==u].sort_values('cycle')
        sub.to_csv(diagnostics_dir/f"worst_unit_{u}.csv", index=False)
        plt.figure(figsize=(6,3))
        plt.plot(sub['cycle'], sub['y_true'], label='true')
        plt.plot(sub['cycle'], sub['y_pred'], label='pred', alpha=0.8)
        plt.gca().invert_xaxis()  # cycle usually decreases to failure; optional
        plt.legend(); plt.title(f'Unit {u}')
        plt.savefig(diagnostics_dir/f"trajectory_unit_{u}.png", dpi=150)
        plt.close()
    print("Diagnostics saved to", diagnostics_dir.resolve())