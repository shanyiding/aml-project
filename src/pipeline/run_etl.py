from pathlib import Path

from src.data.load import load_all
from src.data.preprocess import (
    preprocess_all,
    preprocess_kyc,
    preprocess_txns,
)
from src.data.build_master import build_master_tables


def main():
    """
    End-to-end ETL pipeline:
    1. Load raw tables
    2. Preprocess (clean + standardize)
    3. Build master customer + transaction tables
    4. Save to CSV in data/interim/
    """

    # ------------------------------------------------------------------
    # 1. LOAD RAW DATA
    # ------------------------------------------------------------------
    raw = load_all()

    # ------------------------------------------------------------------
    # 2. PREPROCESS
    # ------------------------------------------------------------------
    # pipeline-level wrapper (keeps naming consistent)
    clean = preprocess_all(raw)

    # apply actual cleaning logic
    clean.kyc_individual = preprocess_kyc(clean.kyc_individual)
    clean.kyc_smallbusiness = preprocess_kyc(clean.kyc_smallbusiness)

    # transactions may be a dict of tables or one dataframe
    if isinstance(clean.txns, dict):
        clean.txns = {
            name: preprocess_txns(df)
            for name, df in clean.txns.items()
        }
    else:
        clean.txns = preprocess_txns(clean.txns)

    # ------------------------------------------------------------------
    # 3. BUILD MASTER TABLES
    # ------------------------------------------------------------------
    master_customers, master_transactions = build_master_tables(clean)

    # ------------------------------------------------------------------
    # 4. SAVE OUTPUTS (CSV ONLY)
    # ------------------------------------------------------------------
    outdir = Path("data/interim")
    outdir.mkdir(parents=True, exist_ok=True)

    customers_path = outdir / "master_customers.csv"
    transactions_path = outdir / "master_transactions.csv"

    master_customers.to_csv(customers_path, index=False)
    master_transactions.to_csv(transactions_path, index=False)

    # ------------------------------------------------------------------
    # 5. LOG SUMMARY
    # ------------------------------------------------------------------
    print("\nETL COMPLETE")
    print("-" * 40)
    print(f"Customers table   → {customers_path}")
    print(f"Transactions table→ {transactions_path}")
    print(f"Customers shape   : {master_customers.shape}")
    print(f"Transactions shape: {master_transactions.shape}")
    print("-" * 40)


if __name__ == "__main__":
    main()
