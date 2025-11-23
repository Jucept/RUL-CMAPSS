# FAST_MODE: versión sacrificada para velocidad y ejecución fiable inmediata
# ...existing code...
import os
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor  # omitted in FAST_MODE

# reproducibility
FAST_MODE = True  # Fast-mode enabled: reduced features/pruning and faster execution
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# logging basic setup (module-level logger)
logger = logging.getLogger("cmapss_pipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# output dirs
PIPE_OUT = Path("pipeline_outputs")
MODELS_DIR = Path("models")
PLOTS_DIR = PIPE_OUT / "plots"
for d in (PIPE_OUT, MODELS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# simple file-based logs
PIPE_LOG = PIPE_OUT / "pipeline_log.txt"
VALID_LOG = PIPE_OUT / "validation_tests.log"

COL_NAMES = ["unit","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]

def _read_txt(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, sep=r"\s+", header=None, names=COL_NAMES, engine="python")
    return df

def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _winsorize_series(s: pd.Series, low_q=0.01, high_q=0.99) -> Tuple[pd.Series,int]:
    low = s.quantile(low_q)
    high = s.quantile(high_q)
    clipped = s.clip(lower=low, upper=high)
    n_changed = int((s != clipped).sum())
    return clipped, n_changed

def _slope_ols(arr: np.ndarray) -> float:
    n = arr.shape[0]
    if n < 2:
        return 0.0
    x = np.arange(n)
    A = np.vstack([x, np.ones(n)]).T
    m, _ = np.linalg.lstsq(A, arr, rcond=None)[0]
    return float(m)

def _append_log(msg: str):
    ts = datetime.utcnow().isoformat()
    with open(PIPE_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts} - {msg}\n")

def fit_transform_train(train_path: str, rul_path: str) -> None:
    t0 = datetime.utcnow()
    _append_log("Starting fit_transform_train (FAST_MODE=%s)" % FAST_MODE)
    logger.info("Starting fit_transform_train (FAST_MODE=%s)", FAST_MODE)

    train_df = _read_txt(train_path)
    logger.info("Loaded train shape %s", train_df.shape)
    _append_log(f"Loaded train shape {train_df.shape}")

    train_df = train_df.copy()

    # Compute RUL
    max_cycle = train_df.groupby("unit")["cycle"].transform("max")
    train_df["RUL"] = max_cycle - train_df["cycle"]
    logger.info("RUL computed. range: %s", (int(train_df["RUL"].min()), int(train_df["RUL"].max())))
    _append_log(f"RUL range {(int(train_df['RUL'].min()), int(train_df['RUL'].max()))}")

    # sensors and ops
    sensors = [f"s{i}" for i in range(1,22)]
    ops = ["op1","op2","op3"]

    # Imputation: causal per column using transform (ffill then bfill limit 1)
    for col in sensors + ops:
        train_df[col] = train_df.groupby("unit")[col].transform(lambda x: x.ffill().bfill(limit=1))

    # Winsorize sensors (cheap) and fit scaler early (per instructions)
    winsor_counts = {}
    for s in sensors:
        clipped, n_changed = _winsorize_series(train_df[s])
        train_df[s] = clipped
        winsor_counts[s] = n_changed
    logger.info("Winsorization example counts (first5): %s", dict(list(winsor_counts.items())[:5]))
    _append_log(f"Winsorization done; sample counts {dict(list(winsor_counts.items())[:5])}")

    # Fit StandardScaler on original sensor columns (fast-mode uses StandardScaler)
    scaler = StandardScaler()
    scaler.fit(train_df[sensors].fillna(0).values)
    save_pickle(scaler, PIPE_OUT / "scaler.pkl")
    _append_log("Scaler fitted and saved to %s" % (PIPE_OUT / "scaler.pkl"))
    logger.info("Scaler fitted and saved to %s", PIPE_OUT / "scaler.pkl")

    # simple rolling windows per instructions (only windows 5 and 20)
    # FIX 2: extend windows to capture short/long dynamics (w=3,5,20,50)
    windows = [3, 5, 20, 50]
    aux_cols = {}  # collect series arrays to avoid repeated assignments

    # Base frames
    X_base = train_df[["unit","cycle"] + ops].copy()
    sensors_df = train_df[sensors].copy()

    # compute per-group rolling features but collect in aux_cols dict
    for w in windows:
        for s in sensors:
            col_rm = f"{s}_rm_{w}"
            col_rstd = f"{s}_rstd_{w}"
            col_slope = f"{s}_slope_{w}"
            # rolling mean (causal)
            rm = train_df.groupby("unit")[s].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
            rstd = train_df.groupby("unit")[s].transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0))
            # slope: rolling apply inside transform
            slope = train_df.groupby("unit")[s].transform(lambda x: x.rolling(window=w, min_periods=2).apply(lambda arr: _slope_ols(arr.values), raw=False).fillna(0))
            aux_cols[col_rm] = rm.values
            aux_cols[col_rstd] = rstd.values
            aux_cols[col_slope] = slope.values

    # delta_from_start: compute first per unit and delta
    first_vals = train_df.groupby("unit")[sensors].transform("first")
    for s in sensors:
        aux_cols[f"{s}_delta_start"] = (train_df[s] - first_vals[s]).values
    # last_wmean20 = rm_20 for each sensor (we already computed rm_20)
    for s in sensors:
        aux_cols[f"{s}_last_wmean20"] = aux_cols.get(f"{s}_rm_20", np.zeros(len(train_df)))
    # FIX 2: also track last_wmean50 for long-term trend
    for s in sensors:
        aux_cols[f"{s}_last_wmean50"] = aux_cols.get(f"{s}_rm_50", np.zeros(len(train_df)))

    # Build aux_df once
    aux_df = pd.DataFrame(aux_cols, index=train_df.index)

    # Concatenate base, sensors, aux in one operation to avoid fragmentation
    X = pd.concat([X_base.reset_index(drop=True), sensors_df.reset_index(drop=True), aux_df.reset_index(drop=True)], axis=1)

    # Drop zero-variance features (cheap)
    feature_cols = [c for c in X.columns if c not in ("unit","cycle")]
    nunique = X[feature_cols].nunique()
    zero_var = list(nunique[nunique<=1].index)
    if zero_var:
        logger.info("Dropping zero-variance features: %s", zero_var)
        X.drop(columns=zero_var, inplace=True)

    # In FAST_MODE we skip expensive correlation pruning and RF importance
    final_features = [c for c in X.columns if c not in ("unit","cycle")]
    # Persist features.json (schema unchanged)
    features_json = {"features": final_features}
    with open(PIPE_OUT / "features.json","w",encoding="utf-8") as f:
        json.dump(features_json, f, indent=2)
    logger.info("Features saved (%d) to %s", len(final_features), PIPE_OUT / "features.json")
    _append_log(f"Features saved count={len(final_features)}")

    # Save X_train and y_train
    X_train = X.copy()
    # include RUL column so legacy validation scripts can load X_train.parquet and check RUL
    X_train["RUL"] = train_df["RUL"].values
    y_train = train_df[["unit","cycle","RUL"]].copy().rename(columns={"RUL":"y"})
    X_train.to_parquet(PIPE_OUT / "X_train.parquet", index=False)
    y_train.to_parquet(PIPE_OUT / "y_train.parquet", index=False)
    logger.info("Saved X_train.parquet and y_train.parquet")
    _append_log("Saved X_train.parquet and y_train.parquet")

    # Metadata
    metadata = {
        "raw_paths": {"train": str(Path(train_path).absolute()), "rul": str(Path(rul_path).absolute())},
        "date_utc": datetime.utcnow().isoformat(),
        "n_units": int(train_df["unit"].nunique()),
        "n_rows_train": int(len(train_df)),
        "n_rows_test": None,
        "scaler_type": "StandardScaler-fast",
        "features_list": final_features[:]
    }
    with open(PIPE_OUT / "metadata.json","w",encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", PIPE_OUT / "metadata.json")
    _append_log("Metadata saved")

    # Minimal validations (fast)
    # 1) No negative RUL
    try:
        if (train_df["RUL"] < 0).any():
            msg = "RUL SANITY FAIL: negative RUL found"
            with open(VALID_LOG,"a",encoding="utf-8") as f:
                f.write(f"{datetime.utcnow().isoformat()} - {msg}\n")
            raise RuntimeError(msg)
        else:
            with open(VALID_LOG,"a",encoding="utf-8") as f:
                f.write(f"{datetime.utcnow().isoformat()} - RUL SANITY PASS\n")

        # 2) Scaler round-trip check on 10 random rows
        save_pickle(scaler, PIPE_OUT / "scaler_roundtrip.pkl")
        scaler_reloaded = load_pickle(PIPE_OUT / "scaler_roundtrip.pkl")
        rng = np.random.default_rng(RANDOM_SEED)
        # use train_df (keeps all original sensor cols) for round-trip validation
        sample_idx = rng.choice(train_df.index, size=min(10, len(train_df)), replace=False)
        a = scaler.transform(train_df.loc[sample_idx, sensors].fillna(0).values)
        b = scaler_reloaded.transform(train_df.loc[sample_idx, sensors].fillna(0).values)
        if not np.allclose(a,b, atol=1e-9, rtol=1e-9):
            msg = "SCALER ROUNDTRIP FAIL"
            with open(VALID_LOG,"a",encoding="utf-8") as f:
                f.write(f"{datetime.utcnow().isoformat()} - {msg}\n")
            raise RuntimeError(msg)
        else:
            with open(VALID_LOG,"a",encoding="utf-8") as f:
                f.write(f"{datetime.utcnow().isoformat()} - SCALER ROUNDTRIP PASS\n")

        # 3) Causality spot-check on 20 random rows for rolling means windows [5,20]
        sample_idx = rng.choice(X_train.index, size=min(20, len(X_train)), replace=False)
        causal_failures = []
        present_cols = set(X_train.columns)
        for idx in sample_idx:
            row = X_train.loc[idx]
            u = int(row["unit"])
            cyc = int(row["cycle"])
            sub = train_df[train_df["unit"] == u].set_index("cycle").sort_index()
            for s in sensors:
                for w in windows:
                    col_rm = f"{s}_rm_{w}"
                    # skip checks for features not present in the final X (dropped by zero-var)
                    if col_rm not in present_cols:
                        continue
                    hist = sub.loc[:cyc, s].values[-w:]
                    recom = np.nan if hist.size == 0 else float(np.mean(hist))
                    val = row.get(col_rm, np.nan)
                    if np.isnan(recom) and pd.isna(val):
                        continue
                    if not np.isclose(recom, val, atol=1e-5, rtol=1e-8):
                        actual_val = None if pd.isna(val) else float(val)
                        causal_failures.append({
                            "idx": int(idx),
                            "unit": int(u),
                            "cycle": int(cyc),
                            "feature": col_rm,
                            "expected": recom,
                            "actual": actual_val,
                            "abs_diff": None if actual_val is None else abs(recom - actual_val)
                        })
                        if len(causal_failures) >= 10:
                            break
                if len(causal_failures) >= 10:
                    break
            if len(causal_failures) >= 10:
                break
        if causal_failures:
            with open(VALID_LOG, "a", encoding="utf-8") as f:
                f.write(f"{datetime.utcnow().isoformat()} - CAUSALITY TEST FAIL (count={len(causal_failures)}):\n")
                for cf in causal_failures[:10]:
                    f.write(json.dumps(cf) + "\n")
            raise RuntimeError("Causality test failed; see validation log")
    except Exception as e:
        _append_log(f"Validation error: {e}")
        raise

    t1 = datetime.utcnow()
    dur = (t1 - t0).total_seconds()
    logger.info("fit_transform_train completed in %.1f seconds", dur)
    _append_log(f"fit_transform_train completed in {dur} seconds")

def transform_test(test_path: str) -> None:
    logger.info("Starting transform_test (FAST_MODE=%s)", FAST_MODE)
    _append_log("Starting transform_test")
    test_df = _read_txt(test_path)
    test_df = test_df.copy()

    # load scaler and features
    scaler = load_pickle(PIPE_OUT / "scaler.pkl")
    with open(PIPE_OUT / "features.json","r",encoding="utf-8") as f:
        features = json.load(f)["features"]

    sensors = [f"s{i}" for i in range(1,22)]
    ops = ["op1","op2","op3"]

    # impute similarly: ffill then bfill limit 1 per unit
    for col in sensors + ops:
        test_df[col] = test_df.groupby("unit")[col].transform(lambda x: x.ffill().bfill(limit=1))

    # Build base and compute rolling features (windows 5 and 20)
    # FIX 2: use extended windows [3,5,20,50]
    X_base = test_df[["unit","cycle"] + ops].copy()
    sensors_df = test_df[sensors].copy()
    aux_cols = {}
    windows = [3, 5, 20, 50]
    for w in windows:
        for s in sensors:
            col_rm = f"{s}_rm_{w}"
            col_rstd = f"{s}_rstd_{w}"
            col_slope = f"{s}_slope_{w}"
            rm = test_df.groupby("unit")[s].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
            rstd = test_df.groupby("unit")[s].transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0))
            slope = test_df.groupby("unit")[s].transform(lambda x: x.rolling(window=w, min_periods=2).apply(lambda arr: _slope_ols(arr.values), raw=False).fillna(0))
            aux_cols[col_rm] = rm.values
            aux_cols[col_rstd] = rstd.values
            aux_cols[col_slope] = slope.values

    # delta_from_start and last_wmean20
    first_vals = test_df.groupby("unit")[sensors].transform("first")
    for s in sensors:
        aux_cols[f"{s}_delta_start"] = (test_df[s] - first_vals[s]).values
        # FIX 2: also track last_wmean50 for long-term trend
        aux_cols[f"{s}_last_wmean20"] = aux_cols.get(f"{s}_rm_20", np.zeros(len(test_df)))
        aux_cols[f"{s}_last_wmean50"] = aux_cols.get(f"{s}_rm_50", np.zeros(len(test_df)))

    aux_df = pd.DataFrame(aux_cols, index=test_df.index)
    X = pd.concat([X_base.reset_index(drop=True), sensors_df.reset_index(drop=True), aux_df.reset_index(drop=True)], axis=1)

    # scale sensors_present only (fallback)
    sensors_present = [s for s in sensors if s in X.columns]
    if sensors_present:
        vals = X[sensors_present].fillna(0).values
        scaled = scaler.transform(vals)
        scaled_df = pd.DataFrame(scaled, columns=[f"{s}_scaled" for s in sensors_present], index=X.index)
        X = pd.concat([X.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

    # keep only features that were saved in features.json (and unit,cycle)
    X_final = X[["unit","cycle"] + [f for f in features if f in X.columns]].copy()
    X_final.to_parquet(PIPE_OUT / "X_test.parquet", index=False)
    logger.info("Saved X_test.parquet")
    _append_log("Saved X_test.parquet")

    # update metadata n_rows_test
    meta_path = PIPE_OUT / "metadata.json"
    if meta_path.exists():
        meta = json.load(open(meta_path,"r"))
        meta["n_rows_test"] = int(len(test_df))
        json.dump(meta, open(meta_path,"w"), indent=2)
    logger.info("transform_test completed")
    _append_log("transform_test completed")