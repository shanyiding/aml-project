from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


IDCOL = "customer_id"
TARGET = "label"

# Use your existing baseline feature list as starting point
FEATURES = [
    "is_international_user",
    "uses_many_channels",
    "industry",
    "kyc_province",
    "recent_amount_ratio",
    "ratio_emt",
    "amount_cv",
    "debit_ratio",
    "channel_entropy",
    "total_amount_vs_finpeer",
    "occupation",
    "pct_history_before_intl",
    "credit_ratio",
    "cv_vs_peer_ratio",
    "max_amount",
]

def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if IDCOL not in df.columns:
        raise ValueError(f"{path.name} missing {IDCOL}")
    return df

def make_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cols = [c for c in features if c in df.columns]
    X = df[cols].copy()

    # categorize columns by dtype
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # fill missing
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str).str.strip()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X

def save_embeddings(customer_ids: np.ndarray, Z: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb_cols = [f"emb_{i:03d}" for i in range(Z.shape[1])]
    out = pd.DataFrame(Z, columns=emb_cols)
    out.insert(0, IDCOL, customer_ids.astype(str))
    out.to_parquet(out_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--run_name", type=str, default="svd_emb_v1")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--use_unlabeled", action="store_true", default=True)
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load splits you already generate
    train_df = load_df(processed / "customers_train.csv")
    val_df   = load_df(processed / "customers_val.csv")
    test_df  = load_df(processed / "customers_test.csv")

    # Optional unlabeled customers (recommended)
    unlabeled_path = processed / "customers_master_unlabeled.csv"
    unlabeled_df = load_df(unlabeled_path) if (args.use_unlabeled and unlabeled_path.exists()) else None

    # Fit embedding transformer on (train + unlabeled) ONLY (no val/test leakage)
    fit_df = train_df.copy()
    if unlabeled_df is not None:
        fit_df = pd.concat([fit_df, unlabeled_df], ignore_index=True)

    fit_ids = fit_df[IDCOL].astype(str).to_numpy()
    X_fit = make_X(fit_df, FEATURES)

    # identify cat/num from X_fit
    cat_cols = [c for c in X_fit.columns if X_fit[c].dtype == "object"]
    num_cols = [c for c in X_fit.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("scaler", StandardScaler(with_mean=False))]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # keep sparse
    )

    svd = TruncatedSVD(n_components=args.dim, random_state=42)

    pipe = Pipeline([
        ("prep", preprocess),
        ("svd", svd),
    ])

    pipe.fit(X_fit)

    # Transform splits
    def transform_split(df: pd.DataFrame, name: str):
        ids = df[IDCOL].astype(str).to_numpy()
        X = make_X(df, FEATURES)
        Z = pipe.transform(X)
        save_embeddings(ids, Z, processed / "static" / f"embeddings_{name}.parquet")
        return Z

    Z_train = transform_split(train_df, "train")
    Z_val   = transform_split(val_df, "val")
    Z_test  = transform_split(test_df, "test")

    # Save pipeline + config
    dump(pipe, out_dir / "svd_pipeline.joblib")

    config = {
        "features": FEATURES,
        "dim": args.dim,
        "fit_on": "train + unlabeled" if unlabeled_df is not None else "train_only",
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("[OK] saved embeddings to data/processed/static/embeddings_{train,val,test}.parquet")
    print("[OK] saved pipeline to", out_dir / "svd_pipeline.joblib")

if __name__ == "__main__":
    main()