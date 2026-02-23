from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ID = "customer_id"
TS = "transaction_datetime"

def safe_dt(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.to_datetime(pd.Series([pd.NaT]*len(df)))
    return pd.to_datetime(df[col], errors="coerce")

def entropy_from_counts(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts[counts > 0] / s
    return float(-(p * np.log(p)).sum())

def _night_ratio(hours: pd.Series) -> float:
    if hours is None or len(hours) == 0:
        return 0.0
    h = hours.dropna().astype(int)
    if len(h) == 0:
        return 0.0
    return float(((h >= 22) | (h <= 6)).mean())

def _intl_ratio(country: pd.Series) -> float:
    if country is None or len(country) == 0:
        return 0.0
    c = country.fillna("Unknown").astype(str)
    if c.nunique() <= 1:
        return 0.0
    mode = c.mode().iloc[0]
    return float((c != mode).mean())

def _amount_cv(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return 0.0
    m = x.mean()
    if m == 0:
        return 0.0
    return float(x.std(ddof=1) / (abs(m) + 1e-9))

def build_features(
    tx_path: Path,
    cust_path: Path,
    out_path: Path,
    last_n: int = 10,
):
    tx = pd.read_csv(tx_path, low_memory=False)
    cust = pd.read_csv(cust_path, low_memory=False)

    if ID not in tx.columns:
        raise ValueError(f"{ID} not in {tx_path}")

    tx[TS] = safe_dt(tx, TS)
    tx = tx.sort_values([ID, TS], kind="mergesort")

    amount = "amount_cad" if "amount_cad" in tx.columns else None
    channel = "channel" if "channel" in tx.columns else None
    mcc = "merchant_category" if "merchant_category" in tx.columns else None
    cash = "cash_indicator" if "cash_indicator" in tx.columns else None
    dc = "debit_credit" if "debit_credit" in tx.columns else None
    country = "country_x" if "country_x" in tx.columns else ("country" if "country" in tx.columns else None)
    city = "city_x" if "city_x" in tx.columns else ("city" if "city" in tx.columns else None)

    tx["_hour"] = tx[TS].dt.hour

    g = tx.groupby(ID, sort=False)
    feats = pd.DataFrame({ID: g.size().index})
    feats["tx_count"] = g.size().to_numpy()

    day = tx[TS].dt.floor("D")
    tx["_day"] = day
    gd = tx.groupby([ID, "_day"], sort=False).size().reset_index(name="cnt")
    gdg = gd.groupby(ID)["cnt"]
    feats = feats.merge(gdg.size().rename("active_days"), on=ID, how="left")
    feats = feats.merge(gdg.max().rename("max_tx_per_day"), on=ID, how="left")
    feats = feats.merge(gdg.std(ddof=1).fillna(0.0).rename("std_tx_per_day"), on=ID, how="left")

    if amount:
        ga = g[amount]
        feats = feats.merge(ga.sum(min_count=1).fillna(0.0).rename("amt_sum"), on=ID, how="left")
        feats = feats.merge(ga.mean().fillna(0.0).rename("amt_mean"), on=ID, how="left")
        feats = feats.merge(ga.max().fillna(0.0).rename("amt_max"), on=ID, how="left")
        feats = feats.merge(ga.std(ddof=1).fillna(0.0).rename("amt_std"), on=ID, how="left")
        feats["amt_cv"] = (feats["amt_std"] / (feats["amt_mean"].abs() + 1e-9)).fillna(0.0)

    feats["night_ratio_all"] = g["_hour"].apply(_night_ratio).to_numpy()

    if country:
        feats["intl_ratio_all"] = g[country].apply(_intl_ratio).to_numpy()
        feats["uniq_countries"] = g[country].nunique().to_numpy()
    if city:
        feats["uniq_cities"] = g[city].nunique().to_numpy()

    if channel:
        feats["uniq_channels"] = g[channel].nunique().to_numpy()
        def _chan_ent(s):
            vc = s.fillna("Unknown").astype(str).value_counts().to_numpy()
            return entropy_from_counts(vc)
        feats["channel_entropy"] = g[channel].apply(_chan_ent).to_numpy()

    if mcc:
        feats["uniq_mcc"] = g[mcc].nunique().to_numpy()

    if dc:
        dcs = tx[dc].fillna("U").astype(str).str.upper()
        tx["_is_debit"] = (dcs == "D").astype(int)
        tx["_is_credit"] = (dcs == "C").astype(int)
        g2 = tx.groupby(ID, sort=False)
        feats = feats.merge(g2["_is_debit"].mean().rename("debit_ratio"), on=ID, how="left")
        feats = feats.merge(g2["_is_credit"].mean().rename("credit_ratio"), on=ID, how="left")

    if cash:
        tx["_cash"] = pd.to_numeric(tx[cash], errors="coerce").fillna(0).astype(int)
        g2 = tx.groupby(ID, sort=False)
        feats = feats.merge(g2["_cash"].mean().rename("cash_ratio"), on=ID, how="left")

    def first_last(group: pd.DataFrame):
        n = len(group)
        first = group.iloc[: min(last_n, n)]
        last = group.iloc[max(0, n - last_n):]

        out = {}
        out["n_tx"] = n

        out["night_first"] = _night_ratio(first["_hour"])
        out["night_last"] = _night_ratio(last["_hour"])
        out["night_delta"] = out["night_last"] - out["night_first"]

        if country and country in group.columns:
            out["intl_first"] = _intl_ratio(first[country])
            out["intl_last"] = _intl_ratio(last[country])
            out["intl_delta"] = out["intl_last"] - out["intl_first"]
        else:
            out["intl_first"] = out["intl_last"] = out["intl_delta"] = 0.0

        if amount and amount in group.columns:
            fa = pd.to_numeric(first[amount], errors="coerce")
            la = pd.to_numeric(last[amount], errors="coerce")
            out["amt_mean_first"] = float(fa.mean()) if fa.notna().any() else 0.0
            out["amt_mean_last"] = float(la.mean()) if la.notna().any() else 0.0
            out["amt_mean_delta"] = out["amt_mean_last"] - out["amt_mean_first"]
            out["amt_cv_first"] = _amount_cv(first[amount])
            out["amt_cv_last"] = _amount_cv(last[amount])
            out["amt_cv_delta"] = out["amt_cv_last"] - out["amt_cv_first"]
        else:
            out["amt_mean_first"]=out["amt_mean_last"]=out["amt_mean_delta"]=0.0
            out["amt_cv_first"]=out["amt_cv_last"]=out["amt_cv_delta"]=0.0

        if channel and channel in group.columns:
            out["uniq_channels_first"] = int(first[channel].nunique())
            out["uniq_channels_last"] = int(last[channel].nunique())
            out["uniq_channels_delta"] = out["uniq_channels_last"] - out["uniq_channels_first"]
        else:
            out["uniq_channels_first"]=out["uniq_channels_last"]=out["uniq_channels_delta"]=0

        return pd.Series(out)

    shift = tx.groupby(ID, sort=False).apply(first_last).reset_index()
    feats = feats.merge(shift, on=ID, how="left")

    keep = [ID]
    for c in ["customer_type", "marital_status", "industry_code", "industry", "province", "income", "sales", "gender", "occupation_code", "occupation_title"]:
        if c in cust.columns:
            keep.append(c)

    cust_small = cust[keep].copy()
    if "label" in cust.columns:
        cust_small["label"] = pd.to_numeric(cust["label"], errors="coerce").fillna(0).astype(int)

    feats = cust_small.merge(feats, on=ID, how="left")

    if "customer_type" in feats.columns:
        feats["is_small_business"] = (feats["customer_type"].astype(str) == "small_business").astype(int)
    if "marital_status" in feats.columns:
        ms = feats["marital_status"].fillna("Unknown").astype(str).str.lower()
        feats["is_unmarried"] = ms.isin(["single", "unmarried"]).astype(int)
    if "is_small_business" in feats.columns and "is_unmarried" in feats.columns:
        feats["smallbiz_and_unmarried"] = feats["is_small_business"] * feats["is_unmarried"]

    for c in feats.columns:
        if c == ID:
            continue
        if feats[c].dtype == "object":
            feats[c] = feats[c].fillna("Unknown").astype(str)
        else:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    print("[OK] wrote", out_path, "rows=", len(feats), "cols=", feats.shape[1])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tx_path", type=Path, default=Path("data/processed/transactions_master.csv"))
    p.add_argument("--cust_path", type=Path, default=Path("data/processed/customers_master.csv"))
    p.add_argument("--out_path", type=Path, default=Path("data/processed/features/customer_behavior_v2.parquet"))
    p.add_argument("--last_n", type=int, default=10)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_features(args.tx_path, args.cust_path, args.out_path, last_n=args.last_n)
