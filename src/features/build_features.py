import pandas as pd

from src.config.paths import INTERIM_DIR, PROCESSED_DIR, ensure_dirs
from src.features.base_aggregates import build_base_aggregates

def build_customer_feature_table(
    customers_csv: str = "master_customers.csv",
    transactions_csv: str = "master_transactions.csv",
    out_csv: str = "features_customer.csv",
) -> str:
    """
    Reads:
      data/interim/master_customers.csv
      data/interim/master_transactions.csv
    Writes:
      data/processed/features_customer.csv
    Returns output path as string.
    """
    ensure_dirs()

    cust_path = INTERIM_DIR / customers_csv
    tx_path = INTERIM_DIR / transactions_csv
    out_path = PROCESSED_DIR / out_csv

    cust = pd.read_csv(cust_path, low_memory=False)
    tx = pd.read_csv(tx_path, low_memory=False)

    # build txn aggregates
    agg = build_base_aggregates(tx)

    # join to customers so every customer exists in final X table
    if "customer_id" not in cust.columns:
        raise ValueError("master_customers.csv must include customer_id")

    X = cust.merge(agg, on="customer_id", how="left")

    # if a customer has no txns, fill agg features with 0
    for c in X.columns:
        if c != "customer_id":
            # only fill numeric-ish new cols safely
            if c.startswith(("txn_", "amt_", "cash_", "n_", "night_")):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    X.to_csv(out_path, index=False)
    return str(out_path)

if __name__ == "__main__":
    p = build_customer_feature_table()
    print(f"Wrote: {p}")
