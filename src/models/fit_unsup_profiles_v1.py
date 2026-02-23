from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ID = "customer_id"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior_path", type=Path, default=Path("data/processed/features/customer_behavior_v2.parquet"))
    ap.add_argument("--out_parquet", type=Path, default=Path("data/processed/features/customer_unsup_v1.parquet"))
    ap.add_argument("--out_run", type=str, default="UNSUP_v1")
    ap.add_argument("--k", type=int, default=12, help="kmeans clusters")
    ap.add_argument("--iso_contamination", type=float, default=0.02, help="expected fraction anomalies")
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    out_dir = Path("outputs/runs") / args.out_run
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.behavior_path)
    if ID not in df.columns:
        raise ValueError(f"Missing {ID} in {args.behavior_path}")

    # Drop obvious non-feature columns except ID
    X = df.drop(columns=[c for c in [ID] if c in df.columns]).copy()

    # One-hot encode categoricals (behavior_v2 has a few object cols)
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in obj_cols:
        X[c] = X[c].fillna("Unknown").astype(str).str.strip()
    X = pd.get_dummies(X, columns=obj_cols, dummy_na=False)

    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_imp = imputer.fit_transform(X)
    X_std = scaler.fit_transform(X_imp)

    # Isolation Forest anomaly score (higher => more anomalous)
    iso = IsolationForest(
        n_estimators=400,
        contamination=args.iso_contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    iso.fit(X_std)
    # decision_function: higher => more normal; we invert to get anomaly score
    iso_anom = (-iso.decision_function(X_std)).astype(float)

    # KMeans segmentation + distance to centroid
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=20)
    cluster = km.fit_predict(X_std).astype(int)
    centers = km.cluster_centers_
    dist = np.linalg.norm(X_std - centers[cluster], axis=1).astype(float)

    out = pd.DataFrame({
        ID: df[ID].astype(str).values,
        "iso_anom_score": iso_anom,
        "kmeans_cluster": cluster,
        "kmeans_dist": dist,
    })

    # Normalize for easy rank fusion
    def minmax(a):
        a = np.asarray(a, dtype=float)
        mn, mx = np.nanmin(a), np.nanmax(a)
        return (a - mn) / (mx - mn + 1e-12)

    out["iso_anom_norm"] = minmax(out["iso_anom_score"])
    out["kmeans_dist_norm"] = minmax(out["kmeans_dist"])
    out["unsup_risk"] = 0.6 * out["iso_anom_norm"] + 0.4 * out["kmeans_dist_norm"]

    out.to_parquet(args.out_parquet, index=False)
    out.sort_values("unsup_risk", ascending=False).head(200).to_csv(out_dir / "top200_unsup.csv", index=False)
    out.to_csv(out_dir / "all_customers_unsup.csv", index=False)

    print("[OK] wrote", args.out_parquet)
    print("[OK] wrote", out_dir)
    print(out[["iso_anom_score","kmeans_cluster","kmeans_dist","unsup_risk"]].describe().to_string())

if __name__ == "__main__":
    main()
