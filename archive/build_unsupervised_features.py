"""
Build unsupervised features (PCA embeddings + KMeans cluster + IsolationForest anomaly score)
from TRANSACTION splits.

Inputs (expected in data/processed):
- transactions_train.csv
- transactions_val.csv
- transactions_test.csv
(optional)
- transactions_unlabeled.csv

Outputs (written to data/interim):
- unsup_features_train.csv
- unsup_features_val.csv
- unsup_features_test.csv
(optional)
- unsup_features_unlabeled.csv

Run from repo root:
  python scripts/build_unsupervised_features.py

Optional paths:
  python scripts/build_unsupervised_features.py --processed_dir data/processed --out_dir data/interim
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


# -------------------------
# Helper: entropy
# -------------------------
def entropy_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


# -------------------------
# Build customer-level numeric feature table from transactions
# -------------------------
def build_behavior_features(txns: pd.DataFrame) -> pd.DataFrame:
    txns = txns.copy()

    # parse datetime if needed
    if "transaction_datetime" in txns.columns:
        txns["transaction_datetime"] = pd.to_datetime(txns["transaction_datetime"], errors="coerce")

    # ensure amount numeric
    if "amount_cad" in txns.columns:
        txns["amount_cad"] = pd.to_numeric(txns["amount_cad"], errors="coerce")

    # debit/credit flags
    if "debit_credit" in txns.columns:
        dc = txns["debit_credit"].fillna("U").astype(str).str.upper()
        txns["is_debit"] = (dc == "D")
        txns["is_credit"] = (dc == "C")
    else:
        txns["is_debit"] = False
        txns["is_credit"] = False

    # cash indicator (exists mostly in abm)
    if "cash_indicator" in txns.columns:
        txns["cash_indicator"] = pd.to_numeric(txns["cash_indicator"], errors="coerce").fillna(0).astype(int)
    else:
        txns["cash_indicator"] = 0

    # channel (we assume you created this earlier)
    if "channel" not in txns.columns:
        txns["channel"] = "Unknown"
    txns["channel"] = txns["channel"].fillna("Unknown").astype(str)

    # country columns: your merged table often becomes country_x/country_y
    # transaction country is usually country_x; kyc country is usually country_y.
    txn_country_col = None
    kyc_country_col = None
    for c in ["country_x", "country", "txn_country", "transaction_country"]:
        if c in txns.columns:
            txn_country_col = c
            break
    for c in ["country_y", "kyc_country"]:
        if c in txns.columns:
            kyc_country_col = c
            break

    # international flag
    if txn_country_col and kyc_country_col:
        txns["is_international"] = (
            txns[txn_country_col].fillna("Unknown").astype(str)
            != txns[kyc_country_col].fillna("Unknown").astype(str)
        )
    else:
        txns["is_international"] = False

    # group by customer
    g = txns.groupby("customer_id", sort=False)

    # basic stats
    out = pd.DataFrame(index=g.size().index)
    out["txn_count"] = g.size().astype(float)

    # amount-based (safe if missing)
    if "amount_cad" in txns.columns:
        out["total_amount"] = g["amount_cad"].sum(min_count=1)
        out["avg_amount"] = g["amount_cad"].mean()
        out["std_amount"] = g["amount_cad"].std()
        out["max_amount"] = g["amount_cad"].max()

        out["amount_cv"] = out["std_amount"] / (out["avg_amount"].replace(0, np.nan))
        out["amount_cv"] = out["amount_cv"].fillna(0.0)
    else:
        out["total_amount"] = 0.0
        out["avg_amount"] = 0.0
        out["std_amount"] = 0.0
        out["max_amount"] = 0.0
        out["amount_cv"] = 0.0

    # ratios
    out["debit_ratio"] = g["is_debit"].mean()
    out["credit_ratio"] = g["is_credit"].mean()
    out["cash_ratio"] = g["cash_indicator"].mean()
    out["international_ratio"] = g["is_international"].mean()

    # unique counts
    if txn_country_col:
        out["unique_countries"] = g[txn_country_col].nunique()
    else:
        out["unique_countries"] = 0.0
    out["unique_channels"] = g["channel"].nunique()

    # channel entropy
    def _channel_entropy(df):
        vc = df["channel"].value_counts(normalize=True).to_numpy()
        return entropy_from_probs(vc)

    out["channel_entropy"] = g.apply(_channel_entropy)

    # recency ratios (recent 30% vs old 30% of that customer's history)
    def _recent_amount_ratio(df):
        df = df.dropna(subset=["transaction_datetime"]).sort_values("transaction_datetime")
        n = len(df)
        if n < 5:
            return 0.0
        k = max(1, int(n * 0.30))
        recent = df.tail(k)
        old = df.head(k)
        old_sum = float(pd.to_numeric(old["amount_cad"], errors="coerce").sum())
        recent_sum = float(pd.to_numeric(recent["amount_cad"], errors="coerce").sum())
        return recent_sum / (old_sum + 1e-9)

    def _recent_txn_count_ratio(df):
        df = df.dropna(subset=["transaction_datetime"]).sort_values("transaction_datetime")
        n = len(df)
        if n < 5:
            return 0.0
        k = max(1, int(n * 0.30))
        return float(len(df.tail(k))) / (float(len(df.head(k))) + 1e-9)

    if "transaction_datetime" in txns.columns and "amount_cad" in txns.columns:
        out["recent_amount_ratio"] = g.apply(_recent_amount_ratio)
        out["recent_txn_count_ratio"] = g.apply(_recent_txn_count_ratio)
        out["txn_frequency_acceleration"] = out["recent_txn_count_ratio"] - 1.0
    else:
        out["recent_amount_ratio"] = 0.0
        out["recent_txn_count_ratio"] = 0.0
        out["txn_frequency_acceleration"] = 0.0

    # burstiness (std/mean of time gaps)
    def _burstiness(df):
        df = df.dropna(subset=["transaction_datetime"]).sort_values("transaction_datetime")
        if len(df) < 5:
            return 0.0
        diffs = df["transaction_datetime"].diff().dt.total_seconds().dropna()
        if len(diffs) == 0:
            return 0.0
        return float(diffs.std() / (diffs.mean() + 1e-9))

    if "transaction_datetime" in txns.columns:
        out["activity_burstiness"] = g.apply(_burstiness)
    else:
        out["activity_burstiness"] = 0.0

    # clean numeric
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out.index.name = "customer_id"
    return out.reset_index()


# -------------------------
# Fit unsupervised models on TRAIN, transform others
# -------------------------
def fit_and_transform(train_feats: pd.DataFrame, other_feats: dict[str, pd.DataFrame], seed: int = 42):
    train_ids = train_feats["customer_id"].copy()
    X_train = train_feats.drop(columns=["customer_id"]).astype(float)

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)

    # PCA
    pca = PCA(n_components=10, random_state=seed)
    Z_train = pca.fit_transform(Xs_train)

    # KMeans
    kmeans = KMeans(n_clusters=8, random_state=seed, n_init=10)
    cluster_train = kmeans.fit_predict(Z_train)

    # distance to center (in PCA space)
    centers = kmeans.cluster_centers_
    dist_train = np.linalg.norm(Z_train - centers[cluster_train], axis=1)

    # IsolationForest on PCA space
    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(Z_train)
    # higher = more anomalous
    anom_train = -iso.score_samples(Z_train)

    # build train output
    out_train = pd.DataFrame({"customer_id": train_ids})
    for i in range(Z_train.shape[1]):
        out_train[f"pca_{i+1}"] = Z_train[:, i]
    out_train["cluster_id"] = cluster_train.astype(int)
    out_train["cluster_dist"] = dist_train
    out_train["anomaly_score"] = anom_train

    outputs = {"train": out_train}

    # transform other splits
    for name, feats in other_feats.items():
        ids = feats["customer_id"].copy()
        X = feats.drop(columns=["customer_id"]).astype(float)
        Xs = scaler.transform(X)
        Z = pca.transform(Xs)

        cl = kmeans.predict(Z)
        dist = np.linalg.norm(Z - centers[cl], axis=1)
        anom = -iso.score_samples(Z)

        out = pd.DataFrame({"customer_id": ids})
        for i in range(Z.shape[1]):
            out[f"pca_{i+1}"] = Z[:, i]
        out["cluster_id"] = cl.astype(int)
        out["cluster_dist"] = dist
        out["anomaly_score"] = anom
        outputs[name] = out

    return outputs


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out_dir", type=Path, default=Path("data/interim"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    processed_dir: Path = args.processed_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # required splits
    train_path = processed_dir / "transactions_train.csv"
    val_path = processed_dir / "transactions_val.csv"
    test_path = processed_dir / "transactions_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    tx_train = pd.read_csv(train_path, low_memory=False)
    tx_val = pd.read_csv(val_path, low_memory=False) if val_path.exists() else None
    tx_test = pd.read_csv(test_path, low_memory=False) if test_path.exists() else None

    print("Loaded txns:")
    print("  train:", tx_train.shape)
    if tx_val is not None: print("  val:", tx_val.shape)
    if tx_test is not None: print("  test:", tx_test.shape)

    feats_train = build_behavior_features(tx_train)
    others = {}
    if tx_val is not None:
        others["val"] = build_behavior_features(tx_val)
    if tx_test is not None:
        others["test"] = build_behavior_features(tx_test)

    # optional unlabeled
    unlabeled_path = processed_dir / "transactions_unlabeled.csv"
    if unlabeled_path.exists():
        tx_unl = pd.read_csv(unlabeled_path, low_memory=False)
        others["unlabeled"] = build_behavior_features(tx_unl)
        print("  unlabeled:", tx_unl.shape)

    outs = fit_and_transform(feats_train, others, seed=args.seed)

    # write
    for split, df in outs.items():
        out_path = out_dir / f"unsup_features_{split}.csv"
        df.to_csv(out_path, index=False)
        print("Saved:", out_path, df.shape)

    print("\nDone. Next step: merge these into your modeling table by customer_id.")


if __name__ == "__main__":
    main()
