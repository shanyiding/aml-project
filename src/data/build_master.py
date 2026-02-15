"""
Build BOTH:
1) data/processed/customers_master.csv   (1 row per customer: raw KYC + label)
2) data/processed/transactions_master.csv (all transactions: raw txns + channel + raw KYC + label)

Based strictly on your provided raw headers.

Run from repo root:
  python scripts/build_master.py

Optional:
  python scripts/build_master.py --raw_dir data/raw --processed_dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# -----------------------------
# Config: your raw file list
# -----------------------------
TXN_FILES = ["abm", "card", "cheque", "eft", "emt", "westernunion", "wire"]


# -----------------------------
# Loaders
# -----------------------------
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_kyc_and_labels(raw_dir: Path):
    kyc_ind = load_csv(raw_dir / "kyc_individual.csv")
    # FIX: correct filename
    kyc_sb = load_csv(raw_dir / "kyc_smallbusiness.csv")
    labels = load_csv(raw_dir / "labels.csv")

    # mapping tables (headers exactly as you gave)
    occ_map = load_csv(raw_dir / "kyc_occupation_codes.csv")   # occupation_code, occupation_title
    ind_map = load_csv(raw_dir / "kyc_industry_codes.csv")     # industry_code, industry

    # normalize join keys to string to avoid int-vs-str merge bugs
    if "occupation_code" in kyc_ind.columns:
        kyc_ind["occupation_code"] = kyc_ind["occupation_code"].astype(str).str.strip()
    if "occupation_code" in occ_map.columns:
        occ_map["occupation_code"] = occ_map["occupation_code"].astype(str).str.strip()

    if "industry_code" in kyc_sb.columns:
        kyc_sb["industry_code"] = kyc_sb["industry_code"].astype(str).str.strip()
    if "industry_code" in ind_map.columns:
        ind_map["industry_code"] = ind_map["industry_code"].astype(str).str.strip()

    # map occupation_code -> occupation_title (individual only)
    if "occupation_code" in kyc_ind.columns and "occupation_code" in occ_map.columns:
        kyc_ind = kyc_ind.merge(
            occ_map[["occupation_code", "occupation_title"]].drop_duplicates("occupation_code"),
            on="occupation_code",
            how="left",
            validate="m:1",
        )

    # map industry_code -> industry (small business only)
    if "industry_code" in kyc_sb.columns and "industry_code" in ind_map.columns:
        kyc_sb = kyc_sb.merge(
            ind_map[["industry_code", "industry"]].drop_duplicates("industry_code"),
            on="industry_code",
            how="left",
            validate="m:1",
        )

    return kyc_ind, kyc_sb, labels





def load_transactions(raw_dir: Path) -> pd.DataFrame:
    def load_txn(stem: str) -> pd.DataFrame:
        df = load_csv(raw_dir / f"{stem}.csv")
        # Add channel column (even if not in raw)
        df["channel"] = stem
        return df

    txns = pd.concat([load_txn(stem) for stem in TXN_FILES], ignore_index=True)

    # Parse datetime (exists in all txn headers you provided)
    if "transaction_datetime" in txns.columns:
        txns["transaction_datetime"] = pd.to_datetime(
            txns["transaction_datetime"], errors="coerce"
        )

    # cash_indicator exists only in abm; unify to always exist
    if "cash_indicator" not in txns.columns:
        txns["cash_indicator"] = 0
    txns["cash_indicator"] = txns["cash_indicator"].fillna(0).astype(int)

    # Debit/credit flags (not "features engineering", just raw-derived flags used before)
    if "debit_credit" in txns.columns:
        txns["debit_credit"] = txns["debit_credit"].fillna("U").astype(str).str.upper()
        txns["is_debit"] = txns["debit_credit"].eq("D")
        txns["is_credit"] = txns["debit_credit"].eq("C")
    else:
        txns["debit_credit"] = "U"
        txns["is_debit"] = False
        txns["is_credit"] = False

    # Fill missing categoricals to match your earlier output style ("Unknown")
    for c in ["country", "province", "city", "merchant_category", "ecommerce_ind", "channel"]:
        if c in txns.columns:
            txns[c] = txns[c].fillna("Unknown").astype(str).str.strip()

    return txns


# -----------------------------
# Build master customers
# -----------------------------
def build_unified_kyc(kyc_ind: pd.DataFrame, kyc_sb: pd.DataFrame) -> pd.DataFrame:
    kyc_ind = kyc_ind.copy()
    kyc_sb = kyc_sb.copy()

    kyc_ind["customer_type"] = "individual"
    kyc_sb["customer_type"] = "small_business"

    all_cols = sorted(set(kyc_ind.columns) | set(kyc_sb.columns))
    kyc_ind = kyc_ind.reindex(columns=all_cols)
    kyc_sb = kyc_sb.reindex(columns=all_cols)

    kyc = pd.concat([kyc_ind, kyc_sb], ignore_index=True)

    # one row per customer
    if "customer_id" in kyc.columns:
        kyc = kyc.drop_duplicates(subset=["customer_id"], keep="first")

    # light cleaning consistent with earlier notebook cleaning
    for c in ["country", "province", "city", "gender", "marital_status",
          "occupation_code", "occupation_title",
          "industry_code", "industry",
          "customer_type"]:

        if c in kyc.columns:
            kyc[c] = kyc[c].fillna("Unknown").astype(str).str.strip()

    for c in ["income", "employee_count", "sales"]:
        if c in kyc.columns:
            kyc[c] = pd.to_numeric(kyc[c], errors="coerce")

    for c in ["birth_date", "onboard_date", "established_date"]:
        if c in kyc.columns:
            kyc[c] = pd.to_datetime(kyc[c], errors="coerce")

    return kyc


def build_customers_master(raw_dir: Path) -> pd.DataFrame:
    kyc_ind, kyc_sb, labels = load_kyc_and_labels(raw_dir)
    kyc = build_unified_kyc(kyc_ind, kyc_sb)

    labels = labels.copy()
    # keep nullable integer labels
    if "label" in labels.columns:
        labels["label"] = pd.to_numeric(labels["label"], errors="coerce").astype("Int64")

    customers = kyc.merge(labels[["customer_id", "label"]], on="customer_id", how="left", validate="1:1")
    return customers


# -----------------------------
# Build master transactions
# -----------------------------
def build_transactions_master(raw_dir: Path, customers_master: pd.DataFrame) -> pd.DataFrame:
    txns = load_transactions(raw_dir)

    # Attach raw KYC (many txns -> one customer)
    kyc_cols = [c for c in customers_master.columns if c != "label"]
    kyc = customers_master[kyc_cols].copy()

    if "customer_id" in txns.columns and "customer_id" in kyc.columns:
        txns = txns.merge(kyc, on="customer_id", how="left", validate="m:1")

    # Attach label (many txns -> one label per customer)
    if "label" in customers_master.columns:
        txns = txns.merge(
            customers_master[["customer_id", "label"]],
            on="customer_id",
            how="left",
            validate="m:1",
        )

    return txns


# -----------------------------
# Main entry
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build customers_master + transactions_master.")
    p.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    return p.parse_args()


def main():
    pd.set_option("display.max_columns", 200)

    args = parse_args()
    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    customers_master = build_customers_master(raw_dir)
    transactions_master = build_transactions_master(raw_dir, customers_master)

    out_customers = processed_dir / "customers_master.csv"
    out_txns = processed_dir / "transactions_master.csv"

    customers_master.to_csv(out_customers, index=False)
    transactions_master.to_csv(out_txns, index=False)

    print("Saved:", out_customers, customers_master.shape)
    print("Saved:", out_txns, transactions_master.shape)


if __name__ == "__main__":
    main()
