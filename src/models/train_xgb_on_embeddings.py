from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


IDCOL = "customer_id"
TARGET = "label"

def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    idx = np.argsort(-y_score)[:k]
    denom = max(1, int(y_true.sum()))
    return float(y_true[idx].sum() / denom)

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].mean())

def pos_in_top_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> int:
    k = min(k, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return int(y_true[idx].sum())

def load_split(processed: Path, split: str):
    cust = pd.read_csv(processed / f"customers_{split}.csv", low_memory=False)
    emb = pd.read_parquet(processed / "static" / f"embeddings_{split}.parquet")
    df = cust[[IDCOL, TARGET]].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    df = df.merge(emb, on=IDCOL, how="inner")
    X = df.drop(columns=[IDCOL, TARGET])
    y = df[TARGET].to_numpy()
    ids = df[IDCOL].astype(str).to_numpy()
    return X, y, ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--run_name", type=str, default="xgb_emb_v1")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _ = load_split(processed, "train")
    X_val,   y_val,   _ = load_split(processed, "val")
    X_test,  y_test,  ids_test = load_split(processed, "test")

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(1, pos)

    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "pos_train": pos,
        "pos_val": int(y_val.sum()),
        "pos_test": int(y_test.sum()),
        "val_pr_auc": float(average_precision_score(y_val, p_val)) if y_val.sum() > 0 else None,
        "test_pr_auc": float(average_precision_score(y_test, p_test)) if y_test.sum() > 0 else None,
        "test_roc_auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else None,
        "test_pos_in_top100": pos_in_top_k(y_test, p_test, 100),
        "test_pos_in_top500": pos_in_top_k(y_test, p_test, 500),
        "test_recall@100": recall_at_k(y_test, p_test, 100),
        "test_recall@500": recall_at_k(y_test, p_test, 500),
        "test_precision@100": precision_at_k(y_test, p_test, 100),
        "test_precision@500": precision_at_k(y_test, p_test, 500),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    ranked = pd.DataFrame({"customer_id": ids_test, "label": y_test, "score": p_test})
    ranked.sort_values("score", ascending=False).to_csv(out_dir / "test_ranked_customers.csv", index=False)

    print("[OK] wrote", out_dir)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()