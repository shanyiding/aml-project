from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from joblib import dump

IDCOL = "customer_id"

def amt_bin(x: float) -> str:
    # bins are log-ish; adjust later
    if x <= 0: return "amt0"
    if x < 10: return "amt1"
    if x < 100: return "amt2"
    if x < 1000: return "amt3"
    if x < 10000: return "amt4"
    return "amt5"

def safe_bucket(s: pd.Series, top_k: int, prefix: str) -> pd.Series:
    s = s.fillna("Unknown").astype(str).str.strip()
    top = s.value_counts().head(top_k).index
    return s.where(s.isin(top), other="Other").map(lambda v: f"{prefix}={v}")

def make_docs(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx[IDCOL] = tx[IDCOL].astype(str)

    tx["transaction_datetime"] = pd.to_datetime(tx["transaction_datetime"], errors="coerce")
    tx = tx.dropna(subset=[IDCOL, "transaction_datetime"])

    tx["amount_cad"] = pd.to_numeric(tx["amount_cad"], errors="coerce").fillna(0.0)
    tx["token_amt"] = tx["amount_cad"].map(lambda v: f"AMT={amt_bin(float(v))}")

    tx["channel"] = tx["channel"].fillna("Unknown").astype(str).str.strip()
    tx["token_ch"] = tx["channel"].map(lambda v: f"CH={v}")

    # merchant_category may be sparse; bucket to top-K
    if "merchant_category" in tx.columns:
        tx["token_mcc"] = safe_bucket(tx["merchant_category"], top_k=200, prefix="MCC")
    else:
        tx["token_mcc"] = "MCC=Unknown"

    # geo buckets
    if "country" in tx.columns:
        tx["token_cty"] = safe_bucket(tx["country"], top_k=50, prefix="CTY")
    else:
        tx["token_cty"] = "CTY=Unknown"

    # time buckets
    hour = tx["transaction_datetime"].dt.hour.fillna(-1).astype(int)
    tx["token_hr"] = hour.map(lambda h: f"HR={h:02d}" if h >= 0 else "HR=Unknown")

    # cash
    cash = tx.get("cash_indicator", 0)
    cash = pd.to_numeric(cash, errors="coerce").fillna(0).astype(int)
    tx["token_cash"] = cash.map(lambda v: f"CASH={v}")

    # combine tokens per transaction -> then per customer
    tx["tokens"] = (
        tx["token_ch"].astype(str) + " " +
        tx["token_mcc"].astype(str) + " " +
        tx["token_cty"].astype(str) + " " +
        tx["token_hr"].astype(str) + " " +
        tx["token_cash"].astype(str) + " " +
        tx["token_amt"].astype(str)
    )

    docs = tx.groupby(IDCOL)["tokens"].apply(lambda s: " ".join(s.tolist())).reset_index()
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="data/processed/static")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--run_name", type=str, default="txn_tfidf_svd_v1")
    args = ap.parse_args()

    tx_csv = Path(args.tx_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path("outputs/runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tx = pd.read_csv(tx_csv, low_memory=False)

    # Handle your columns: you used country/province/city names in build_master
    # Some of your txns might have country_x etc; choose one
    if "country" not in tx.columns:
        for alt in ["country_x", "country_y"]:
            if alt in tx.columns:
                tx["country"] = tx[alt]

    docs = make_docs(tx)
    ids = docs[IDCOL].to_numpy()

    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b[^ ]+\b",
        lowercase=False,
        max_features=args.max_features,
        min_df=2,
    )

    X = vec.fit_transform(docs["tokens"])
    svd = TruncatedSVD(n_components=args.dim, random_state=42)
    Z = svd.fit_transform(X)

    # save embeddings
    emb_cols = [f"txemb_{i:03d}" for i in range(Z.shape[1])]
    emb = pd.DataFrame(Z, columns=emb_cols)
    emb.insert(0, IDCOL, ids.astype(str))
    emb_path = out_dir / f"txn_embeddings_{tx_csv.stem}.parquet"
    emb.to_parquet(emb_path, index=False)

    # save vectorizer + svd
    dump(vec, run_dir / "tfidf_vectorizer.joblib")
    dump(svd, run_dir / "svd.joblib")

    print("[OK] wrote", emb_path, "rows=", len(emb))

if __name__ == "__main__":
    main()