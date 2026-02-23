from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import joblib


FEATURES = [
    "is_international_user",
    "uses_many_channels",
    "industry",
    "kyc_province",
    "recent_amount_ratio",
    "ratio_emt",
    "amount_cv",
    "debit_ratio",
    "channel_entropy",
    "total_amount_vs_finpeer",
    "occupation",
    "pct_history_before_intl",
    "credit_ratio",
    "cv_vs_peer_ratio",
    "max_amount",
]

TARGET = "label"
IDCOL = "customer_id"


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    idx = np.argsort(-y_score)[:k]
    denom = max(1, int(y_true.sum()))
    return float(y_true[idx].sum() / denom)

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].mean())

def positives_in_top_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> int:
    k = min(k, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return int(y_true[idx].sum())


def make_xy(df: pd.DataFrame, features=FEATURES):
    cols = [IDCOL, TARGET] + [c for c in features if c in df.columns]
    d = df[cols].copy()

    d[TARGET] = pd.to_numeric(d[TARGET], errors="coerce").fillna(0).astype(int)

    X = d.drop(columns=[IDCOL, TARGET])
    y = d[TARGET].to_numpy()
    ids = d[IDCOL].astype(str).to_numpy()

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str).str.strip()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X, y, ids, cat_cols, num_cols


def save_curve_pr(y, p, out_png: Path, title: str):
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def save_curve_roc(y, p, out_png: Path, title: str):
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--run_name", type=str, default="lr_ohe_v1")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load your split outputs (professional: never re-split inside training)
    train_df = pd.read_csv(processed / "customers_train.csv")
    val_df   = pd.read_csv(processed / "customers_val.csv")
    test_df  = pd.read_csv(processed / "customers_test.csv")

    X_train, y_train, id_train, cat_cols, num_cols = make_xy(train_df)
    X_val,   y_val,   id_val,   _, _ = make_xy(val_df)
    X_test,  y_test,  id_test,  _, _ = make_xy(test_df)

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="liblinear",
    )

    clf = Pipeline(steps=[("prep", preprocess), ("model", model)])
    clf.fit(X_train, y_train)

    p_val = clf.predict_proba(X_val)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "pos_train": int(y_train.sum()),
        "pos_val": int(y_val.sum()),
        "pos_test": int(y_test.sum()),
        "val_pr_auc": float(average_precision_score(y_val, p_val)) if y_val.sum() > 0 else None,
        "test_pr_auc": float(average_precision_score(y_test, p_test)) if y_test.sum() > 0 else None,
        "val_roc_auc": float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else None,
        "test_roc_auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else None,
        "test_recall@100": recall_at_k(y_test, p_test, 100),
        "test_recall@500": recall_at_k(y_test, p_test, 500),
        "test_precision@100": precision_at_k(y_test, p_test, 100),
        "test_precision@500": precision_at_k(y_test, p_test, 500),
        "test_pos_in_top100": positives_in_top_k(y_test, p_test, 100),
        "test_pos_in_top500": positives_in_top_k(y_test, p_test, 500),
    }

    # Save artifacts
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    config = {"features": FEATURES, "cat_cols": cat_cols, "num_cols": num_cols}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    joblib.dump(clf, out_dir / "model.joblib")

    ranked = pd.DataFrame({"customer_id": id_test, "label": y_test, "score": p_test})
    ranked = ranked.sort_values("score", ascending=False)
    ranked.to_csv(out_dir / "test_ranked_customers.csv", index=False)

    save_curve_pr(y_val, p_val, out_dir / "pr_curve_val.png", "PR Curve (Validation)")
    save_curve_roc(y_val, p_val, out_dir / "roc_curve_val.png", "ROC Curve (Validation)")

    print("[OK] wrote:", out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()