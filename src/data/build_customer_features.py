"""
Build customer-level feature tables from raw transaction + KYC CSVs.

Replicates the Jupyter notebook logic shown in your screenshots and writes:
- data/processed/customer_features.csv
- data/processed/customer_features_labeled.csv
- data/processed/customer_features_unlabeled.csv

Run:
  python scripts/build_customer_features.py
or:
  python scripts/build_customer_features.py --raw_dir data/raw --processed_dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# IO helpers
# -----------------------------
def load_kyc_tables(raw_dir: Path):
    kyc_ind = pd.read_csv(raw_dir / "kyc_individual.csv", low_memory=False)
    kyc_sb = pd.read_csv(raw_dir / "kyc_smallbusiness.csv", low_memory=False)
    labels = pd.read_csv(raw_dir / "labels.csv", low_memory=False)

    # not used downstream in the shown notebook cells, but loaded there
    occ_codes = None
    ind_codes = None
    occ_path = raw_dir / "kyc_occupation_codes.csv"
    ind_path = raw_dir / "kyc_industry_codes.csv"
    if occ_path.exists():
        occ_codes = pd.read_csv(occ_path, low_memory=False)
    if ind_path.exists():
        ind_codes = pd.read_csv(ind_path, low_memory=False)

    return kyc_ind, kyc_sb, labels, occ_codes, ind_codes


def load_transactions(raw_dir: Path) -> pd.DataFrame:
    def load_txn(file_stem: str) -> pd.DataFrame:
        df = pd.read_csv(raw_dir / f"{file_stem}.csv", low_memory=False)
        df["channel"] = file_stem
        return df

    txn_files = ["abm", "card", "cheque", "eft", "emt", "westernunion", "wire"]
    txns = pd.concat([load_txn(f) for f in txn_files], ignore_index=True)

    # parse datetime
    if "transaction_datetime" in txns.columns:
        txns["transaction_datetime"] = pd.to_datetime(
            txns["transaction_datetime"], errors="coerce"
        )

    # unify cash_indicator if missing
    if "cash_indicator" not in txns.columns:
        txns["cash_indicator"] = 0
    txns["cash_indicator"] = txns["cash_indicator"].fillna(0).astype(int)

    return txns


# -----------------------------
# Transformations (mirrors notebook)
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
    if "customer_id" in kyc.columns:
        kyc = kyc.drop_duplicates(subset=["customer_id"], keep="first")
    return kyc


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # debit/credit flags
    if "debit_credit" in df.columns:
        df["debit_credit"] = df["debit_credit"].fillna("U").astype(str).str.upper()
        df["is_debit"] = df["debit_credit"].eq("D")
        df["is_credit"] = df["debit_credit"].eq("C")
    else:
        df["debit_credit"] = "U"
        df["is_debit"] = False
        df["is_credit"] = False

    # fill missing categoricals
    cat_cols = [
        "country",
        "province",
        "city",
        "merchant_category",
        "gender",
        "marital_status",
        "occupation_code",
        "industry_code",
        "customer_type",
        "channel",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str).str.strip()

    # numeric columns
    num_cols = ["amount_cad", "income", "sales", "employee_count"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "transaction_datetime" not in df.columns:
        # keep consistent columns if missing
        df["hour"] = np.nan
        df["dow"] = np.nan
        df["is_weekend"] = False
        df["is_night"] = False
        return df

    dt = df["transaction_datetime"]
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6])
    df["is_night"] = df["hour"].between(1, 5)
    return df


def customer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("customer_id", sort=False)

    cust_basic = pd.DataFrame(
        {
            "txn_count": g.size(),
            "total_amount": g["amount_cad"].sum(min_count=1),
            "avg_amount": g["amount_cad"].mean(),
            "std_amount": g["amount_cad"].std(),
            "max_amount": g["amount_cad"].max(),
            "debit_ratio": g["is_debit"].mean(),
            "credit_ratio": g["is_credit"].mean(),
            "night_ratio": g["is_night"].mean(),
            "weekend_ratio": g["is_weekend"].mean(),
            "unique_countries": g["country"].nunique() if "country" in df.columns else 0,
            "unique_cities": g["city"].nunique() if "city" in df.columns else 0,
            "amount_missing_ratio": g["amount_cad"].apply(lambda s: s.isna().mean()),
        }
    ).fillna(0)

    return cust_basic


def channel_mix_features(df: pd.DataFrame, cust_basic: pd.DataFrame) -> pd.DataFrame:
    channel_counts = pd.crosstab(df["customer_id"], df["channel"]).add_prefix("cnt_")

    channel_amounts = (
        df.pivot_table(
            index="customer_id",
            columns="channel",
            values="amount_cad",
            aggfunc="sum",
            fill_value=0,
        ).add_prefix("amt_")
    )

    cust_channels = channel_counts.join(channel_amounts, how="outer").fillna(0)

    # ratios by total amount
    total_amt = cust_basic["total_amount"].replace(0, np.nan)
    amt_cols = [c for c in cust_channels.columns if c.startswith("amt_")]
    for c in amt_cols:
        cust_channels["amt_ratio_" + c[len("amt_") :]] = cust_channels[c].div(total_amt)

    # ratios by transaction count
    total_cnt = cust_basic["txn_count"].replace(0, np.nan)
    cnt_cols = [c for c in cust_channels.columns if c.startswith("cnt_")]
    for c in cnt_cols:
        cust_channels["cnt_ratio_" + c[len("cnt_") :]] = cust_channels[c].div(total_cnt)

    cust_channels = cust_channels.fillna(0)
    return cust_channels


def cash_behaviour(df: pd.DataFrame, cust_basic: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("customer_id", sort=False)

    cash_amt = df["amount_cad"].where(df["cash_indicator"].eq(1), 0)

    cust_cash = pd.DataFrame(
        {
            "cash_txn_count": g["cash_indicator"].sum(),
            "cash_txn_ratio": g["cash_indicator"].mean(),
            "cash_amount": cash_amt.groupby(df["customer_id"]).sum(),
        }
    )

    cust_cash["cash_amount_ratio"] = cust_cash["cash_amount"].div(
        cust_basic["total_amount"].replace(0, np.nan)
    )

    cust_cash = cust_cash.fillna(0)
    return cust_cash


def cash_last_30_days(df: pd.DataFrame) -> pd.DataFrame:
    if "transaction_datetime" not in df.columns or df["transaction_datetime"].isna().all():
        return pd.DataFrame(columns=["cash_amount_last_30d"]).astype({"cash_amount_last_30d": float})

    cutoff = df["transaction_datetime"].max() - pd.Timedelta(days=30)

    cash_last30 = (
        df[(df["transaction_datetime"] >= cutoff) & (df["cash_indicator"] == 1)]
        .groupby("customer_id")["amount_cad"]
        .sum()
        .rename("cash_amount_last_30d")
        .to_frame()
    )
    return cash_last30


def burstiness_max_txn_1h(df: pd.DataFrame) -> pd.DataFrame:
    if "transaction_datetime" not in df.columns:
        return pd.DataFrame(columns=["max_txn_in_1h"])

    tmp = df.dropna(subset=["transaction_datetime"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["max_txn_in_1h"])

    tmp["_hour_bucket"] = tmp["transaction_datetime"].dt.floor("h")
    hourly = tmp.groupby(["customer_id", "_hour_bucket"]).size()
    burst = hourly.groupby("customer_id").max().to_frame("max_txn_in_1h").fillna(0)
    return burst


def add_kyc_static_features(df: pd.DataFrame, kyc: pd.DataFrame) -> pd.DataFrame:
    kyc_cols = [
        "customer_id",
        "customer_type",
        "gender",
        "marital_status",
        "occupation_code",
        "industry_code",
        "income",
        "employee_count",
        "sales",
        "birth_date",
        "onboard_date",
        "established_date",
    ]
    kyc_cols = [c for c in kyc_cols if c in kyc.columns]

    cust_kyc = kyc[kyc_cols].drop_duplicates("customer_id").set_index("customer_id")

    # dates
    for c in ["birth_date", "onboard_date", "established_date"]:
        if c in cust_kyc.columns:
            cust_kyc[c] = pd.to_datetime(cust_kyc[c], errors="coerce")

    today = df["transaction_datetime"].max() if "transaction_datetime" in df.columns else pd.Timestamp.today()

    if "onboard_date" in cust_kyc.columns:
        cust_kyc["tenure_days"] = (today - cust_kyc["onboard_date"]).dt.days
    if "birth_date" in cust_kyc.columns:
        cust_kyc["age_years"] = (today - cust_kyc["birth_date"]).dt.days / 365.25

    # fill missing
    for c in ["tenure_days", "age_years"]:
        if c in cust_kyc.columns:
            cust_kyc[c] = cust_kyc[c].fillna(cust_kyc[c].median())

    # drop raw dates
    drop_dates = [c for c in ["birth_date", "onboard_date", "established_date"] if c in cust_kyc.columns]
    if drop_dates:
        cust_kyc = cust_kyc.drop(columns=drop_dates)

    return cust_kyc


def cash_to_income_ratio(cust_cash: pd.DataFrame, cust_kyc: pd.DataFrame) -> pd.DataFrame:
    # exactly as notebook: cash_amount / income (can yield NaN/inf if income missing/0)
    if "income" not in cust_kyc.columns:
        return pd.DataFrame(columns=["cash_to_income_ratio"])

    ratio = (cust_cash["cash_amount"] / cust_kyc["income"]).rename("cash_to_income_ratio").to_frame()
    return ratio


# -----------------------------
# Pipeline
# -----------------------------
def build_features(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    kyc_ind, kyc_sb, labels, _, _ = load_kyc_tables(raw_dir)
    txns = load_transactions(raw_dir)

    kyc = build_unified_kyc(kyc_ind, kyc_sb)
    df = txns.merge(kyc, on="customer_id", how="left", validate="m:1")

    df = clean_and_standardize(df)
    df = add_time_features(df)

    cust_basic = customer_aggregates(df)
    cust_channels = channel_mix_features(df, cust_basic)
    cust_cash = cash_behaviour(df, cust_basic)
    cash_last30 = cash_last_30_days(df)
    burst = burstiness_max_txn_1h(df)
    cust_kyc = add_kyc_static_features(df, kyc)
    cash_income = cash_to_income_ratio(cust_cash, cust_kyc)

    features = cust_basic.join(
        [cust_channels, cust_cash, burst, cust_kyc, cash_last30, cash_income],
        how="left",
    )

    # merge labels like notebook
    labels = labels.copy()
    if "label" in labels.columns:
        labels["label"] = labels["label"].astype(int)

    features = features.merge(
        labels[["customer_id", "label"]],
        left_index=True,
        right_on="customer_id",
        how="left",
        validate="1:1",
    )
    features = features.set_index("customer_id")

    # ensure processed dir exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # save exactly as notebook
    out_all = processed_dir / "customer_features.csv"
    features.to_csv(out_all, index=True)

    labeled = features[features["label"].notna()].copy()
    unlabeled = features[features["label"].isna()].copy()

    out_labeled = processed_dir / "customer_features_labeled.csv"
    out_unlabeled = processed_dir / "customer_features_unlabeled.csv"

    labeled.to_csv(out_labeled, index=True)
    unlabeled.to_csv(out_unlabeled, index=True)

    print("Saved:", out_all, features.shape)
    print("Saved labeled:", out_labeled, labeled.shape)
    print("Saved unlabeled:", out_unlabeled, unlabeled.shape)

    return features


def parse_args():
    p = argparse.ArgumentParser(description="Build AML customer feature tables.")
    p.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw).",
    )
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed data directory (default: data/processed).",
    )
    return p.parse_args()


def main():
    pd.set_option("display.max_columns", 200)
    args = parse_args()
    build_features(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
