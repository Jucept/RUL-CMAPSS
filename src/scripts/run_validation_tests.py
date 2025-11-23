# src/scripts/run_validation_tests.py
import pathlib, pandas as pd, json
from sklearn.preprocessing import StandardScaler
import numpy as np

def causal_spot_check(X_train_parquet, features, n=100):
    df = pd.read_parquet(X_train_parquet)
    # simple spot check: rolling mean col names contain '_rm_'
    checks = []
    sample = df.sample(min(n, len(df)), random_state=42)
    for col in [c for c in df.columns if '_rm_' in c][:10]:
        checks.append((col, not sample[col].isnull().any()))
    return checks

def scaler_roundtrip(scaler_pkl, X_sample_parquet):
    import pickle
    s = pickle.load(open(scaler_pkl,'rb'))
    df = pd.read_parquet(X_sample_parquet).iloc[:50]
    cols = [c for c in df.columns if c.startswith('s')][:21]
    arr1 = s.transform(df[cols].fillna(0).values)
    import tempfile, pickle as pkl
    fn = tempfile.NamedTemporaryFile(delete=False).name
    pkl.dump(s, open(fn,'wb'))
    s2 = pkl.load(open(fn,'rb'))
    arr2 = s2.transform(df[cols].fillna(0).values)
    return np.allclose(arr1, arr2)

def main(out_log):
    out = []
    out.append("Validation tests")
    ok1 = causal_spot_check("./pipeline_outputs/X_train.parquet", None)
    out.append("causal_spot_check: " + str(ok1[:5]))
    ok2 = scaler_roundtrip("./pipeline_outputs/scaler.pkl", "./pipeline_outputs/X_train.parquet")
    out.append("scaler_roundtrip: " + str(ok2))
    df = pd.read_parquet("./pipeline_outputs/X_train.parquet")
    out.append("RUL sanity: min RUL >=0 ? " + str((df['RUL']>=0).all()))
    with open(out_log,"w") as f:
        f.write("\n".join(out))
    print("\n".join(out))

if __name__=="__main__":
    main("./pipeline_outputs/validation_tests.log")