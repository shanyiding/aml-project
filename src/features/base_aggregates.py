import numpy as np
import pandas as pd

def _safe_div(a, b):
    return np.where(b == 0, 0.0, a / b)

def build_base_aggregates(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level aggregate features from master_transactions.csv.
    Assumes tx includes at least:
      - customer_id
      - amount_cad
      - transaction_datetime (or transaction_date / transaction_time)
      - channel
      - is_credit / is_debit (optional)
      - cash_indicator (optional)
      - country/province/city (optional)
    Returns: one row per customer_id
    """

    df = tx.copy()

    # --- required columns checks ---
    if "customer_id" not in df.columns:
        raise ValueError("master_transactions is missing customer_id")

    # amount cleanup
    if "amount_cad" in df.columns:
        df["amount_cad"] = pd.to_numeric(df["amount_cad"], errors="coerce").fillna(0.0)
    else:
        df["amount_cad"] = 0.0

    # datetime
    if "transaction_datetime" in df.columns:
        df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")
    else:
        df["transaction_datetime"] = pd.NaT

    # channel
    if "channel" not in df.columns:
        df["channel"] = "unknown"
    df["channel"] = df["channel"].fillna("unknown").astype(str)

    # cash indicator
    if "cash_indicator" not in df.columns:
        df["cash_indicator"] = 0
    df["cash_indicator"] = pd.to_numeric(df["cash_indicator"], errors="coerce").fillna(0).astype(int)

    # optional debit/credit flags
    if "is_credit" not in df.columns:
        df["is_credit"] = False
    if "is_debit" not in df.columns:
        df["is_debit"] = False

    # basic txn counts and sums
    g = df.groupby("customer_id", dropna=False)

    out = pd.DataFrame({
        "customer_id": g.size().index,
        "txn_count_all": g.size().values,
        "amt_sum_all": g["amount_cad"].sum().values,
        "amt_mean_all": g["amount_cad"].mean().values,
        "amt_max_all": g["amount_cad"].max().values,
        "amt_std_all": g["amount_cad"].std(ddof=0).fillna(0.0).values,
        "cash_txn_count": g["cash_indicator"].sum().values,
    })

    out["cash_txn_ratio"] = _safe_div(out["cash_txn_count"], out["txn_count_all"])

    # channel counts (wide)
    ch_counts = (
        df.pivot_table(index="customer_id", columns="channel", values="amount_cad", aggfunc="size", fill_value=0)
        .add_prefix("txn_count_channel_")
        .reset_index()
    )
    out = out.merge(ch_counts, on="customer_id", how="left")

    # channel sums (wide)
    ch_sums = (
        df.pivot_table(index="customer_id", columns="channel", values="amount_cad", aggfunc="sum", fill_value=0.0)
        .add_prefix("amt_sum_channel_")
        .reset_index()
    )
    out = out.merge(ch_sums, on="customer_id", how="left")

    # diversity / network-ish signals if available
    for col, name in [("country", "n_countries"), ("province", "n_provinces"), ("city", "n_cities")]:
        if col in df.columns:
            tmp = df.groupby("customer_id")[col].nunique(dropna=True).rename(name).reset_index()
            out = out.merge(tmp, on="customer_id", how="left")
        else:
            out[name] = 0

    # night-time cash ratio (example rule-aligned feature, if datetime exists)
    if df["transaction_datetime"].notna().any():
        hour = df["transaction_datetime"].dt.hour
        is_night = hour.isin([22, 23, 0, 1, 2, 3, 4, 5])
        night_cash = df[is_night & (df["cash_indicator"] == 1)].groupby("customer_id").size()
        night_all = df[is_night].groupby("customer_id").size()

        night = pd.DataFrame({
            "customer_id": out["customer_id"],
            "night_txn_count": out["customer_id"].map(night_all).fillna(0).astype(int),
            "night_cash_txn_count": out["customer_id"].map(night_cash).fillna(0).astype(int),
        })
        night["night_cash_ratio"] = _safe_div(night["night_cash_txn_count"], night["night_txn_count"])
        out = out.merge(night, on="customer_id", how="left")
    else:
        out["night_txn_count"] = 0
        out["night_cash_txn_count"] = 0
        out["night_cash_ratio"] = 0.0

    # fill any missing numeric
    for c in out.columns:
        if c != "customer_id":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    return out
