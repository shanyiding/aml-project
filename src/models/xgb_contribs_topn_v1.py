from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

ID="customer_id"

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_features(processed_dir: Path, split: str, behavior_path: Path | None, emb_prefix: str | None):
    # You already have behavior features parquet; we use that as the base.
    # If your current explainer merges embeddings too, we can extend later.
    if behavior_path is None:
        behavior_path = processed_dir / "features" / "customer_behavior_v2.parquet"
    df = pd.read_parquet(behavior_path)
    df[ID] = df[ID].astype(str)
    X = df.drop(columns=[ID], errors="ignore").copy()

    # basic cleaning
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    obj = [c for c in X.columns if str(X[c].dtype) in ("object","category")]
    if obj:
        X[obj] = X[obj].fillna("Unknown").astype(str)
        X = pd.get_dummies(X, columns=obj, dummy_na=False)
    for c in X.columns:
        if not str(X[c].dtype).startswith(("int","float")):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df[[ID]].copy(), X

def align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    # Align columns to booster feature names if available
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        fn = booster.feature_names
        if fn:
            for c in fn:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[fn]
    return X

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ranked_csv", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--processed_dir", default=str(repo_root() / "data" / "processed"))
    ap.add_argument("--behavior_path", default="")
    ap.add_argument("--top_features", type=int, default=8)
    args=ap.parse_args()

    ranked = pd.read_csv(args.ranked_csv)
    ranked[ID] = ranked[ID].astype(str)
    ranked_top = ranked.head(args.top_n).copy()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "model.joblib")

    processed_dir = Path(args.processed_dir)
    behavior_path = Path(args.behavior_path) if args.behavior_path else None
    ids, X = load_features(processed_dir, args.split, behavior_path, None)

    # take only top customers rows
    X_all = ids.merge(ranked_top[[ID]], on=ID, how="inner")
    X_feat = X.loc[X_all.index].copy()  # same ordering as parquet, but index mismatch may happen
    # safer join by ID
    feat_df = ids.join(X)
    feat_df = feat_df.set_index(ID)
    X_feat = feat_df.loc[ranked_top[ID]].reset_index(drop=False)
    cust_ids = X_feat[ID].astype(str).values
    X_feat = X_feat.drop(columns=[ID])

    X_feat = align_to_model(X_feat, model)

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    dm = xgb.DMatrix(X_feat, feature_names=list(X_feat.columns))
    contrib = booster.predict(dm, pred_contribs=True)  # shape (n, n_features+1)

    cols = list(X_feat.columns) + ["bias"]
    C = pd.DataFrame(contrib, columns=cols)
    # per row: pick top +/- contribs
    out_rows = []
    k = int(args.top_features)
    for i in range(len(C)):
        row = C.iloc[i].drop(labels=["bias"])
        up = row.sort_values(ascending=False).head(k)
        dn = row.sort_values(ascending=True).head(k)
        def fmt(s):
            parts=[]
            for name,val in s.items():
                if abs(val) < 1e-12: 
                    continue
                parts.append(f"{name}({val:+.4f})")
            return ";".join(parts)
        out_rows.append({
            ID: cust_ids[i],
            "contrib_risk_up": fmt(up),
            "contrib_risk_down": fmt(dn),
        })

    out = ranked_top.merge(pd.DataFrame(out_rows), on=ID, how="left")
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("[OK] wrote", out_path)

if __name__=="__main__":
    main()
