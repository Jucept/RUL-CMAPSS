# Apply monotonic smoothing (Fix 1) to predictions
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def smooth_rul_per_unit(df: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """
    Apply exponential smoothing per unit to enforce smoother RUL decay.
    - alpha: smoothing factor (0.2 = moderate smoothing)
    Returns df with column y_pred_smooth added.
    """
    df = df.copy()
    df['y_pred_smooth'] = np.nan
    
    for unit in df['unit'].unique() if 'unit' in df.columns else [None]:
        mask = df['unit'] == unit if 'unit' in df.columns else slice(None)
        subset = df.loc[mask].sort_values('cycle') if 'cycle' in df.columns else df.loc[mask]
        
        # Exponential moving average (backward to smooth trend)
        smooth = subset['y_pred'].ewm(alpha=alpha, adjust=False).mean()
        
        # Enforce monotonic decrease (RUL should not increase)
        smooth = pd.Series(
            np.maximum.accumulate(smooth.values[::-1])[::-1],
            index=smooth.index
        )
        
        # Clip to non-negative
        smooth = smooth.clip(lower=0)
        
        df.loc[mask, 'y_pred_smooth'] = smooth.values
    
    return df

def main(preds_csv: str, out_csv: str, alpha: float = 0.2):
    print(f"Loading predictions from {preds_csv}")
    df = pd.read_csv(preds_csv)
    
    print(f"Applying monotonic smoothing (alpha={alpha})...")
    if 'y_pred' not in df.columns:
        raise RuntimeError("Input predictions CSV must contain 'y_pred' column")
    df_smooth = smooth_rul_per_unit(df, alpha=alpha)

    print(f"Saving smoothed predictions to {out_csv}")
    df_smooth.to_csv(out_csv, index=False)

    # Quick stats (use df_smooth which contains y_pred_smooth)
    if 'y_true' in df_smooth.columns and 'y_pred_smooth' in df_smooth.columns:
        y_true = df_smooth['y_true'].astype(float).to_numpy()
        y_raw = df_smooth['y_pred'].astype(float).to_numpy()
        y_s = df_smooth['y_pred_smooth'].astype(float).to_numpy()
        rmse_raw = float(np.sqrt(np.mean((y_true - y_raw) ** 2)))
        rmse_smooth = float(np.sqrt(np.mean((y_true - y_s) ** 2)))
        print(f"RMSE before smoothing: {rmse_raw:.2f}")
        print(f"RMSE after smoothing: {rmse_smooth:.2f}")
        print(f"Improvement: {(rmse_raw - rmse_smooth):.2f} ({100*(rmse_raw-rmse_smooth)/rmse_raw:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_csv", required=True, help="Input predictions CSV")
    parser.add_argument("--out_csv", required=True, help="Output smoothed predictions CSV")
    parser.add_argument("--alpha", type=float, default=0.2, help="EWM smoothing factor (0-1)")
    args = parser.parse_args()
    main(args.preds_csv, args.out_csv, args.alpha)