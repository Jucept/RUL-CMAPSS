# src/scripts/eval_from_preds.py
import argparse
import pathlib
import pandas as pd
import numpy as np
import sys

def phm08_score(y_true, y_pred):
    # simple proxy using numpy (robust across sklearn versions)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = np.mean((y_true - y_pred) ** 2)
    return -np.sqrt(mse)

def _find_cols(df: pd.DataFrame):
    cand_true = ["y_true", "y", "true", "y_actual"]
    cand_pred = ["y_pred", "pred", "yhat", "y_predicted"]
    col_true = next((c for c in cand_true if c in df.columns), None)
    col_pred = next((c for c in cand_pred if c in df.columns), None)
    return col_true, col_pred

def main(preds_dir, out_csv):
    pdir = pathlib.Path(preds_dir)
    rows = []
    files = sorted(pdir.glob("preds_val_fold*.csv"))
    if not files:
        print(f"No prediction files found in {pdir} matching preds_val_fold*.csv", file=sys.stderr)
        sys.exit(1)

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Skipping {f.name}: read error {e}", file=sys.stderr)
            continue
        col_true, col_pred = _find_cols(df)
        if col_true is None or col_pred is None:
            print(f"Skipping {f.name}: required columns not found (have: {list(df.columns)})", file=sys.stderr)
            continue
        try:
            y_true = df[col_true].astype(float).to_numpy()
            y_pred = df[col_pred].astype(float).to_numpy()
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            phm = phm08_score(y_true, y_pred)
        except Exception as e:
            print(f"Skipping {f.name}: metric computation failed: {e}", file=sys.stderr)
            continue
        rows.append({"fold": f.name, "n": len(df), "PHM08_score": phm, "RMSE": rmse, "MAE": mae})

    if not rows:
        print("No valid prediction files processed. Exiting.", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame(rows)
    out['PHM08_mean'] = out['PHM08_score'].mean()
    out['RMSE_mean'] = out['RMSE'].mean()
    out['MAE_mean'] = out['MAE'].mean()
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.preds_dir, args.out)