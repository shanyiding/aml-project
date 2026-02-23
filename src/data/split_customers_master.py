"""
Split customers_master.csv into:
- customers_master_labeled.csv
- customers_master_unlabeled.csv

And split labeled into:
- customers_train.csv
- customers_val.csv
- customers_test.csv

PLUS: split transactions_master.csv by customer_id into:
- transactions_labeled.csv
- transactions_unlabeled.csv
- transactions_train.csv
- transactions_val.csv
- transactions_test.csv

Default split: 70/15/15 with stratification on label.

Run:
  python scripts/split_customers_master.py
or:
  python scripts/split_customers_master.py --in_path data/processed/customers_master.csv --transactions_path data/processed/transactions_master.csv --out_dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _stratified_split_indices(
    y: np.ndarray, train_frac: float, val_frac: float, test_frac: float, seed: int
):
    """
    Pure-numpy stratified split returning index arrays for train/val/test.
    Works for binary or multiclass labels.
    """
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []
    test_idx = []

    classes = pd.unique(y)
    for c in classes:
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # remainder goes to test to ensure total matches
        n_test = n - n_train - n_val
        if n_test < 0:
            # adjust if rounding overshot
            n_test = 0
            n_val = n - n_train

        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train : n_train + n_val])
        test_idx.append(cls_idx[n_train + n_val : n_train + n_val + n_test])

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def _split_transactions(
    transactions_path: Path,
    out_dir: Path,
    train_ids: set,
    val_ids: set,
    test_ids: set,
    labeled_ids: set,
    unlabeled_ids: set,
):
    txns = pd.read_csv(transactions_path, low_memory=False)

    if "customer_id" not in txns.columns:
        raise ValueError(f"'customer_id' column not found in {transactions_path}")

    # Labeled/unlabeled transactions (based on whether customer_id is labeled)
    txns_labeled = txns[txns["customer_id"].isin(labeled_ids)].copy()
    txns_unlabeled = txns[txns["customer_id"].isin(unlabeled_ids)].copy()

    # Train/val/test transactions (based on customer split)
    txns_train = txns[txns["customer_id"].isin(train_ids)].copy()
    txns_val = txns[txns["customer_id"].isin(val_ids)].copy()
    txns_test = txns[txns["customer_id"].isin(test_ids)].copy()

    # Save
    txns_labeled_path = out_dir / "transactions_labeled.csv"
    txns_unlabeled_path = out_dir / "transactions_unlabeled.csv"
    txns_train_path = out_dir / "transactions_train.csv"
    txns_val_path = out_dir / "transactions_val.csv"
    txns_test_path = out_dir / "transactions_test.csv"

    txns_labeled.to_csv(txns_labeled_path, index=False)
    txns_unlabeled.to_csv(txns_unlabeled_path, index=False)
    txns_train.to_csv(txns_train_path, index=False)
    txns_val.to_csv(txns_val_path, index=False)
    txns_test.to_csv(txns_test_path, index=False)

    print("Saved:", txns_labeled_path, txns_labeled.shape)
    print("Saved:", txns_unlabeled_path, txns_unlabeled.shape)
    print("Saved:", txns_train_path, txns_train.shape)
    print("Saved:", txns_val_path, txns_val.shape)
    print("Saved:", txns_test_path, txns_test.shape)


def split_customers_master(
    in_path: Path,
    out_dir: Path,
    transactions_path: Path | None = None,
    label_col: str = "label",
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    seed: int = 42,
):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    df = pd.read_csv(in_path, low_memory=False)

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in {in_path}")

    if "customer_id" not in df.columns:
        raise ValueError(f"'customer_id' column not found in {in_path}")

    # labeled / unlabeled
    labeled = df[df[label_col].notna()].copy()
    unlabeled = df[df[label_col].isna()].copy()

    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = out_dir / "customers_master_labeled.csv"
    unlabeled_path = out_dir / "customers_master_unlabeled.csv"
    labeled.to_csv(labeled_path, index=False)
    unlabeled.to_csv(unlabeled_path, index=False)

    # ensure labels are integers for stratification
    y = pd.to_numeric(labeled[label_col], errors="coerce").astype(int).to_numpy()

    train_idx, val_idx, test_idx = _stratified_split_indices(
        y=y,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    train = labeled.iloc[train_idx].copy()
    val = labeled.iloc[val_idx].copy()
    test = labeled.iloc[test_idx].copy()

    train_path = out_dir / "customers_train.csv"
    val_path = out_dir / "customers_val.csv"
    test_path = out_dir / "customers_test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    # print summary
    def vc(d: pd.DataFrame):
        return d[label_col].value_counts(dropna=False).to_dict()

    print("Saved:", labeled_path, labeled.shape, "label_counts:", vc(labeled))
    print("Saved:", unlabeled_path, unlabeled.shape)
    print("Saved:", train_path, train.shape, "label_counts:", vc(train))
    print("Saved:", val_path, val.shape, "label_counts:", vc(val))
    print("Saved:", test_path, test.shape, "label_counts:", vc(test))

    # Split transactions too (by customer_id mapping)
    if transactions_path is not None:
        train_ids = set(train["customer_id"])
        val_ids = set(val["customer_id"])
        test_ids = set(test["customer_id"])
        labeled_ids = set(labeled["customer_id"])
        unlabeled_ids = set(unlabeled["customer_id"])

        _split_transactions(
            transactions_path=transactions_path,
            out_dir=out_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            labeled_ids=labeled_ids,
            unlabeled_ids=unlabeled_ids,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Split customers_master into labeled/unlabeled and train/val/test; optionally split transactions too."
    )
    p.add_argument("--in_path", type=Path, default=Path("data/processed/customers_master.csv"))
    p.add_argument("--transactions_path", type=Path, default=Path("data/processed/transactions_master.csv"))
    p.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    split_customers_master(
        in_path=args.in_path,
        out_dir=args.out_dir,
        transactions_path=args.transactions_path,
        label_col=args.label_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
