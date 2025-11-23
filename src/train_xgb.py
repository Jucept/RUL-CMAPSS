# train_xgb.py
import os
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from eval import score_phm08

PIPE_OUT = Path("pipeline_outputs")
MODELS_DIR = Path("models")
PLOTS_DIR = PIPE_OUT / "plots"
for d in (PIPE_OUT, MODELS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_and_evaluate():
    X = pd.read_parquet(PIPE_OUT / "X_train.parquet")
    y = pd.read_parquet(PIPE_OUT / "y_train.parquet")["y"]
    # ensure alignment
    assert len(X) == len(y)
    # load features list to choose columns
    features = json.load(open(PIPE_OUT / "features.json"))["features"]

    # filtering features present in X
    features = [f for f in features if f in X.columns]
    Xf = X[features].fillna(0)
    groups = X["unit"].values

    gkf = GroupKFold(n_splits=5)
    fold_metrics = []
    all_val_preds = []
    fi_accumulate = pd.DataFrame(0, index=features, columns=["imp_sum"])
    fold_idx = 0
    for train_idx, val_idx in gkf.split(Xf, y, groups):
        X_tr, X_val = Xf.iloc[train_idx], Xf.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            "objective":"reg:squarederror",
            "seed": RANDOM_SEED,
            "verbosity":1,
            "tree_method":"hist",
            "learning_rate":0.05,
            "max_depth":6,
            "subsample":0.8,
            "colsample_bytree":0.8,
            "nthread":4
        }
        evals = [(dval,"eval")]
        model = xgb.train(params, dtrain, num_boost_round=2000, evals=evals, early_stopping_rounds=50, verbose_eval=False)
        model_path = MODELS_DIR / f"model_xgb_fold{fold_idx}.bin"
        model.save_model(str(model_path))
        # feature importance (gain)
        fmap = model.get_score(importance_type="gain")
        fi = pd.Series({k: fmap.get(k,0.0) for k in X_tr.columns}, name=f"fold{fold_idx}")
        fi_accumulate["imp_sum"] = fi_accumulate["imp_sum"].add(fi.reindex(fi_accumulate.index).fillna(0))
        # predict val
        y_pred = model.predict(dval)
        # save preds
        val_df = pd.DataFrame({"unit": X.iloc[val_idx]["unit"].values, "cycle": X.iloc[val_idx]["cycle"].values, "y_true": y_val.values, "y_pred": y_pred})
        val_df.to_csv(MODELS_DIR / f"preds_val_fold{fold_idx}.csv", index=False)
        # metrics
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = mean_absolute_error(y_val, y_pred)
        phm = score_phm08(y_val.values, y_pred)
        fold_metrics.append({"fold": fold_idx, "rmse": rmse, "mae": mae, "phm08": phm})
        fold_idx += 1

    # aggregate metrics
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_summary = {
        "per_fold": metrics_df.to_dict(orient="records"),
        "mean": metrics_df.mean(numeric_only=True).to_dict(),
        "std": metrics_df.std(numeric_only=True).to_dict()
    }
    pd.DataFrame(metrics_df).to_csv(MODELS_DIR / "metrics_summary.csv", index=False)
    # feature importance mean
    fi_accumulate["imp_mean"] = fi_accumulate["imp_sum"] / 5.0
    fi_out = fi_accumulate["imp_mean"].reset_index().rename(columns={"index":"feature","imp_mean":"importance_mean"})
    fi_out.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    # plots: pred_vs_true and residuals (aggregate)
    all_preds = []
    for f in range(5):
        pf = pd.read_csv(MODELS_DIR / f"preds_val_fold{f}.csv")
        all_preds.append(pf)
    agg = pd.concat(all_preds, ignore_index=True)
    # pred vs true scatter (sampled for legibility)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.scatter(agg["y_true"], agg["y_pred"], alpha=0.3, s=5)
    plt.plot([agg["y_true"].min(), agg["y_true"].max()], [agg["y_true"].min(), agg["y_true"].max()], color="k")
    plt.xlabel("y_true (RUL)")
    plt.ylabel("y_pred")
    plt.title("Predicted vs True (val aggregated)")
    plt.savefig(PIPE_OUT / "plots" / "pred_vs_true.png", dpi=150)
    plt.close()

    # residuals
    agg["residual"] = agg["y_pred"] - agg["y_true"]
    plt.figure(figsize=(8,6))
    plt.hist(agg["residual"], bins=100)
    plt.title("Residuals distribution (val aggregated)")
    plt.xlabel("residual")
    plt.savefig(PIPE_OUT / "plots" / "residuals.png", dpi=150)
    plt.close()

    # print summary
    print("Training complete. Metrics saved to", MODELS_DIR / "metrics_summary.csv")
    print("Feature importance saved to", MODELS_DIR / "feature_importance.csv")
    print("Plots saved to", PIPE_OUT / "plots")
    return metrics_summary

if __name__ == "__main__":
    train_and_evaluate()