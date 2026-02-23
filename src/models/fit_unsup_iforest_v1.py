from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ID="customer_id"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features_parquet", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--contamination", default="auto")
    args=ap.parse_args()

    df=pd.read_parquet(args.features_parquet)
    if ID not in df.columns:
        raise SystemExit(f"[ERR] missing {ID} in {args.features_parquet}")

    ids=df[[ID]].copy()
    X=df.drop(columns=[ID]).copy()

    # basic cleaning
    for c in X.columns:
        if X[c].dtype == bool:
            X[c]=X[c].astype(int)
    obj=[c for c in X.columns if str(X[c].dtype) in ("object","category")]
    if obj:
        X[obj]=X[obj].fillna("Unknown").astype(str)
        X=pd.get_dummies(X, columns=obj, dummy_na=False)
    for c in X.columns:
        if not str(X[c].dtype).startswith(("int","float")):
            X[c]=pd.to_numeric(X[c], errors="coerce")
    X=X.replace([np.inf,-np.inf], np.nan).fillna(0.0)

    cont=args.contamination
    if isinstance(cont,str) and cont!="auto":
        try: cont=float(cont)
        except: cont="auto"

    model=IsolationForest(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        contamination=cont,
        n_jobs=-1
    )
    model.fit(X)
    normal=model.decision_function(X)   # higher=more normal
    anom=(-normal).astype(float)        # higher=more anomalous

    out=ids.copy()
    out["iforest_score"]=anom
    out["iforest_pct"]=pd.Series(anom).rank(pct=True, method="average").values

    out_path=Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("[OK] wrote", out_path)

if __name__=="__main__":
    main()
