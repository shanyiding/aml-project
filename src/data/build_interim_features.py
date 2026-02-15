"""
Generate interim feature tables (train/test) from split transactions files.

Outputs:
  data/interim/features_cash_train.csv
  data/interim/features_cash_test.csv
  data/interim/features_wire_train.csv
  data/interim/features_wire_test.csv
  data/interim/features_emt_train.csv
  data/interim/features_emt_test.csv

Run from repo root:
  python src/data/build_interim_features.py
Optional:
  python src/data/build_interim_features.py --processed_dir data/processed --interim_dir data/interim
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _entropy_from_counts(counts: pd.Series) -> float:
    """Entropy in nats; counts can be raw counts or already-normalized."""
    if counts is None or len(counts) == 0:
        return 0.0
    x = counts.astype(float)
    if x.sum() <= 0:
        return 0.0
    p = x / x.sum()
    ent = 0.0
    for pi in p:
        if pi > 0:
            ent -= float(pi) * math.log(float(pi))
    return float(ent)


def _gini_from_probs(probs: pd.Series) -> float:
    """For concentration we use 1 - sum(p^2) (a common 'Gini-like' concentration)."""
    if probs is None or len(probs) == 0:
        return 0.0
    p = probs.astype(float)
    if abs(p.sum() - 1.0) > 1e-6:
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / s
    return float(1.0 - (p * p).sum())


def _amount_round_flags(amount: pd.Series) -> pd.DataFrame:
    """
    Simple round-number / cents heuristics:
    - is_round_100: amount divisible by 100
    - is_round_1000: amount divisible by 1000
    - cents_is_00: cents component == 0
    """
    a = pd.to_numeric(amount, errors="coerce").fillna(0.0).abs()
    cents = (a * 100).round().astype("int64") % 100
    is_100 = (a.round(2) % 100 == 0)
    is_1000 = (a.round(2) % 1000 == 0)
    cents_00 = (cents == 0)
    return pd.DataFrame(
        {
            "is_round_100": is_100.astype(int),
            "is_round_1000": is_1000.astype(int),
            "cents_is_00": cents_00.astype(int),
        }
    )


def _burst_max_in_1h(df: pd.DataFrame, dt_col: str) -> int:
    """max #txns in any 1-hour bucket for this df."""
    if df.empty:
        return 0
    tmp = df[[dt_col]].dropna().copy()
    if tmp.empty:
        return 0
    tmp["_hour"] = tmp[dt_col].dt.floor("h")
    counts = tmp.groupby("_hour").size()
    return int(counts.max()) if len(counts) else 0


# -----------------------------
# Core feature builders
# -----------------------------
def build_cash_features(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Cash features should primarily come from ABM (since only ABM has cash_indicator).
    We compute:
      - cash_txn_count, cash_txn_ratio
      - cash_amount, cash_amount_ratio
      - cash_amount_last30d
      - cash_recent_spike_ratio (last30d vs previous30d)
      - round_number_ratio_100 / round_number_ratio_1000 / cents_00_ratio on cash txns only
      - cash_burst_1h (within cash txns)
    """
    txns = txns.copy()

    dt_col = _pick(txns, ["transaction_datetime", "transaction_datetime_x", "transaction_datetime_y"])
    amt_col = _pick(txns, ["amount_cad", "amount_cad_x", "amount_cad_y"])
    cash_col = _pick(txns, ["cash_indicator"])
    ch_col = _pick(txns, ["channel"])

    if dt_col is None or amt_col is None:
        raise ValueError("transactions file missing required columns: transaction_datetime and/or amount_cad")

    txns[dt_col] = _to_dt(txns[dt_col])
    txns[amt_col] = pd.to_numeric(txns[amt_col], errors="coerce").fillna(0.0)

    # define "cash txns": prefer cash_indicator==1; fallback to channel=='abm'
    if cash_col is not None:
        cash_mask = txns[cash_col].fillna(0).astype(int) == 1
    elif ch_col is not None:
        cash_mask = txns[ch_col].astype(str).str.lower() == "abm"
    else:
        cash_mask = pd.Series(False, index=txns.index)

    rows = []
    for cid, df in txns.groupby("customer_id", sort=False):
        f = {"customer_id": cid}

        n_all = len(df)
        total_amt = float(df[amt_col].sum()) if n_all else 0.0

        df_cash = df[cash_mask.loc[df.index]]
        n_cash = len(df_cash)
        cash_amt = float(df_cash[amt_col].sum()) if n_cash else 0.0

        f["cash_txn_count"] = int(n_cash)
        f["cash_txn_ratio"] = float(n_cash / n_all) if n_all else 0.0
        f["cash_amount"] = float(cash_amt)
        f["cash_amount_ratio"] = float(cash_amt / (total_amt + 1e-9)) if n_all else 0.0

        # last 30 days cash amount (relative to customer’s latest txn time)
        if n_cash and df_cash[dt_col].notna().any():
            latest = df_cash[dt_col].max()
            cutoff = latest - pd.Timedelta(days=30)
            cash_last30 = float(df_cash[df_cash[dt_col] >= cutoff][amt_col].sum())
            f["cash_amount_last30d"] = cash_last30

            # spike ratio: last30 vs previous30
            prev_cutoff = cutoff - pd.Timedelta(days=30)
            prev30 = float(df_cash[(df_cash[dt_col] >= prev_cutoff) & (df_cash[dt_col] < cutoff)][amt_col].sum())
            f["cash_recent_spike_ratio"] = float(cash_last30 / (prev30 + 1e-9))
        else:
            f["cash_amount_last30d"] = 0.0
            f["cash_recent_spike_ratio"] = 1.0

        # round-number signals on cash txns
        if n_cash:
            flags = _amount_round_flags(df_cash[amt_col])
            f["cash_round_100_ratio"] = float(flags["is_round_100"].mean())
            f["cash_round_1000_ratio"] = float(flags["is_round_1000"].mean())
            f["cash_cents_00_ratio"] = float(flags["cents_is_00"].mean())
            f["cash_burst_1h"] = _burst_max_in_1h(df_cash, dt_col)
        else:
            f["cash_round_100_ratio"] = 0.0
            f["cash_round_1000_ratio"] = 0.0
            f["cash_cents_00_ratio"] = 0.0
            f["cash_burst_1h"] = 0

        rows.append(f)

    return pd.DataFrame(rows)


def build_wire_features(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Wire features (channel=='wire'):
      - wire_txn_count, wire_txn_ratio
      - wire_amount, wire_amount_ratio
      - wire_avg_amount, wire_max_amount
      - wire_burst_1h
      - wire_temporal_entropy_hour
      - wire_round_100_ratio / wire_cents_00_ratio
      - wire_international_ratio (if txn vs kyc country present)
      - wire_country_diversity
    """
    txns = txns.copy()

    dt_col = _pick(txns, ["transaction_datetime", "transaction_datetime_x", "transaction_datetime_y"])
    amt_col = _pick(txns, ["amount_cad", "amount_cad_x", "amount_cad_y"])
    ch_col = _pick(txns, ["channel"])

    txn_country = _pick(txns, ["country_x", "country"])
    kyc_country = _pick(txns, ["country_y"])

    if dt_col is None or amt_col is None or ch_col is None:
        raise ValueError("transactions file missing required columns for wire features (need transaction_datetime, amount_cad, channel)")

    txns[dt_col] = _to_dt(txns[dt_col])
    txns[amt_col] = pd.to_numeric(txns[amt_col], errors="coerce").fillna(0.0)

    rows = []
    for cid, df in txns.groupby("customer_id", sort=False):
        f = {"customer_id": cid}

        n_all = len(df)
        total_amt = float(df[amt_col].sum()) if n_all else 0.0

        dfw = df[df[ch_col].astype(str).str.lower() == "wire"]
        n = len(dfw)

        f["wire_txn_count"] = int(n)
        f["wire_txn_ratio"] = float(n / n_all) if n_all else 0.0

        wire_amt = float(dfw[amt_col].sum()) if n else 0.0
        f["wire_amount"] = wire_amt
        f["wire_amount_ratio"] = float(wire_amt / (total_amt + 1e-9)) if n_all else 0.0
        f["wire_avg_amount"] = float(dfw[amt_col].mean()) if n else 0.0
        f["wire_max_amount"] = float(dfw[amt_col].max()) if n else 0.0

        f["wire_burst_1h"] = _burst_max_in_1h(dfw, dt_col) if n else 0

        # temporal entropy by hour
        if n and dfw[dt_col].notna().any():
            hours = dfw[dt_col].dt.hour
            vc = hours.value_counts()
            f["wire_temporal_entropy_hour"] = _entropy_from_counts(vc)
        else:
            f["wire_temporal_entropy_hour"] = 0.0

        # round-number behavior on wire
        if n:
            flags = _amount_round_flags(dfw[amt_col])
            f["wire_round_100_ratio"] = float(flags["is_round_100"].mean())
            f["wire_cents_00_ratio"] = float(flags["cents_is_00"].mean())
        else:
            f["wire_round_100_ratio"] = 0.0
            f["wire_cents_00_ratio"] = 0.0

        # international ratios and diversity (if columns exist)
        if n and txn_country is not None and kyc_country is not None:
            intl = dfw[dfw[txn_country].astype(str) != dfw[kyc_country].astype(str)]
            f["wire_international_ratio"] = float(len(intl) / n) if n else 0.0
            f["wire_country_diversity"] = int(dfw[txn_country].nunique(dropna=True)) if txn_country else 0
        else:
            f["wire_international_ratio"] = 0.0
            f["wire_country_diversity"] = int(dfw[txn_country].nunique(dropna=True)) if (n and txn_country) else 0

        rows.append(f)

    return pd.DataFrame(rows)


def build_emt_features(txns: pd.DataFrame) -> pd.DataFrame:
    """
    EMT features (channel=='emt'):
      - emt_txn_count, emt_txn_ratio
      - emt_amount, emt_amount_ratio
      - emt_avg_amount, emt_max_amount
      - emt_burst_1h
      - emt_round_100_ratio / emt_cents_00_ratio
      - emt_temporal_entropy_hour
    """
    txns = txns.copy()

    dt_col = _pick(txns, ["transaction_datetime", "transaction_datetime_x", "transaction_datetime_y"])
    amt_col = _pick(txns, ["amount_cad", "amount_cad_x", "amount_cad_y"])
    ch_col = _pick(txns, ["channel"])

    if dt_col is None or amt_col is None or ch_col is None:
        raise ValueError("transactions file missing required columns for emt features (need transaction_datetime, amount_cad, channel)")

    txns[dt_col] = _to_dt(txns[dt_col])
    txns[amt_col] = pd.to_numeric(txns[amt_col], errors="coerce").fillna(0.0)

    rows = []
    for cid, df in txns.groupby("customer_id", sort=False):
        f = {"customer_id": cid}

        n_all = len(df)
        total_amt = float(df[amt_col].sum()) if n_all else 0.0

        dfe = df[df[ch_col].astype(str).str.lower() == "emt"]
        n = len(dfe)

        f["emt_txn_count"] = int(n)
        f["emt_txn_ratio"] = float(n / n_all) if n_all else 0.0

        emt_amt = float(dfe[amt_col].sum()) if n else 0.0
        f["emt_amount"] = emt_amt
        f["emt_amount_ratio"] = float(emt_amt / (total_amt + 1e-9)) if n_all else 0.0
        f["emt_avg_amount"] = float(dfe[amt_col].mean()) if n else 0.0
        f["emt_max_amount"] = float(dfe[amt_col].max()) if n else 0.0

        f["emt_burst_1h"] = _burst_max_in_1h(dfe, dt_col) if n else 0

        if n and dfe[dt_col].notna().any():
            hours = dfe[dt_col].dt.hour
            vc = hours.value_counts()
            f["emt_temporal_entropy_hour"] = _entropy_from_counts(vc)
        else:
            f["emt_temporal_entropy_hour"] = 0.0

        if n:
            flags = _amount_round_flags(dfe[amt_col])
            f["emt_round_100_ratio"] = float(flags["is_round_100"].mean())
            f["emt_cents_00_ratio"] = float(flags["cents_is_00"].mean())
        else:
            f["emt_round_100_ratio"] = 0.0
            f["emt_cents_00_ratio"] = 0.0

        rows.append(f)

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build interim feature CSVs for cash/wire/emt (train/test).")
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--interim_dir", type=Path, default=Path("data/interim"))
    return p.parse_args()


def main():
    args = parse_args()
    processed_dir: Path = args.processed_dir
    interim_dir: Path = args.interim_dir
    interim_dir.mkdir(parents=True, exist_ok=True)

    txn_train_path = processed_dir / "transactions_train.csv"
    txn_test_path = processed_dir / "transactions_test.csv"

    for p in [txn_train_path, txn_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    txns_train = pd.read_csv(txn_train_path, low_memory=False)
    txns_test = pd.read_csv(txn_test_path, low_memory=False)

    # CASH
    cash_train = build_cash_features(txns_train)
    cash_test = build_cash_features(txns_test)

    out = interim_dir / "features_cash_train.csv"
    cash_train.to_csv(out, index=False)
    print("Saved:", out, cash_train.shape)

    out = interim_dir / "features_cash_test.csv"
    cash_test.to_csv(out, index=False)
    print("Saved:", out, cash_test.shape)

    # WIRE
    wire_train = build_wire_features(txns_train)
    wire_test = build_wire_features(txns_test)

    out = interim_dir / "features_wire_train.csv"
    wire_train.to_csv(out, index=False)
    print("Saved:", out, wire_train.shape)

    out = interim_dir / "features_wire_test.csv"
    wire_test.to_csv(out, index=False)
    print("Saved:", out, wire_test.shape)

    # EMT
    emt_train = build_emt_features(txns_train)
    emt_test = build_emt_features(txns_test)

    out = interim_dir / "features_emt_train.csv"
    emt_train.to_csv(out, index=False)
    print("Saved:", out, emt_train.shape)

    out = interim_dir / "features_emt_test.csv"
    emt_test.to_csv(out, index=False)
    print("Saved:", out, emt_test.shape)

    print("\nDone. Interim features are in:", interim_dir)


if __name__ == "__main__":
    main()
