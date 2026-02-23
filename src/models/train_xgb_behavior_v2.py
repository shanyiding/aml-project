from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ID = "customer_id"
TARGET = "label"


def load_split(processed: Path, feat_path: Path, split: str):
    cust = pd.read_csv(processed / f"customers_{split}.csv", usecols=[ID, TARGET])
    cust[TARGET] = pd.to_numeric(cust[TARGET], errors="coerce").fillna(0).astype(int)

    feat = pd.read_parquet(feat_path)
    if TARGET in feat.columns:
        feat = feat.drop(columns=[TARGET], errors="ignore")

    df = cust.merge(feat, on=ID, how="left")

    y = df[TARGET].to_numpy()
    X = df.drop(columns=[ID, TARGET], errors="ignore")

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str).str.strip()

    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    X = X.fillna(0)

    return X, y, df[[ID, TARGET]].copy()


def align_cols(Xtr, Xva, Xte):
    cols = sorted(set(Xtr.columns) | set(Xva.columns) | set(Xte.columns))
    Xtr = Xtr.reindex(columns=cols, fill_value=0)
    Xva = Xva.reindex(columns=cols, fill_value=0)
    Xte = Xte.reindex(columns=cols, fill_value=0)
    return Xtr, Xva, Xte, cols


def pos_in_topk(y, score, k):
    if len(y) == 0:
        return 0
    idx = np.argsort(-score)[: min(k, len(y))]
    return int(y[idx].sum())


def recall_at_k(y, score, k):
    total_pos = int(y.sum())
    if total_pos == 0:
        return None
    return float(pos_in_topk(y, score, k) / total_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--feat_path", type=Path, default=Path("data/processed/features/customer_behavior_v2.parquet"))
    ap.add_argument("--run_name", type=str, default="xgb_behavior_v2")
    ap.add_argument("--pos_weight", type=float, default=-1.0)
    ap.add_argument("--n_estimators", type=int, default=1200)
    args = ap.parse_args()

    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, tr_ids = load_split(args.processed_dir, args.feat_path, "train")
    Xva, yva, va_ids = load_split(args.processed_dir, args.feat_path, "val")
    Xte, yte, te_ids = load_split(args.processed_dir, args.feat_path, "test")
    Xtr, Xva, Xte, cols = align_cols(Xtr, Xva, Xte)

    pos = int(ytr.sum())
    neg = int((ytr == 0).sum())
    spw = float(args.pos_weight) if args.pos_weight > 0 else (neg / max(pos, 1))

    print(f"[INFO] Train rows={len(ytr)} pos={pos} neg={neg} scale_pos_weight={spw:.2f}")

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        min_child_weight=5,
        gamma=0.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        verbose=100,
    )

    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]

    metrics = {
        "pos_train": int(ytr.sum()),
        "pos_val": int(yva.sum()),
        "pos_test": int(yte.sum()),
        "val_pr_auc": float(average_precision_score(yva, pva)) if int(yva.sum()) > 0 else None,
        "test_pr_auc": float(average_precision_score(yte, pte)) if int(yte.sum()) > 0 else None,
        "val_roc_auc": float(roc_auc_score(yva, pva)) if len(np.unique(yva)) > 1 else None,
        "test_roc_auc": float(roc_auc_score(yte, pte)) if len(np.unique(yte)) > 1 else None,
        "test_pos_in_top100": pos_in_topk(yte, pte, 100),
        "test_pos_in_top500": pos_in_topk(yte, pte, 500),
        "test_recall@100": recall_at_k(yte, pte, 100),
        "test_recall@500": recall_at_k(yte, pte, 500),
        "n_features": int(len(cols)),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, out_dir / "model.joblib")

    # ---------- WRITE RANKED FILES ----------
    ranked_val = va_ids.copy()
    ranked_val["score"] = pva
    ranked_val.sort_values("score", ascending=False).to_csv(out_dir / "val_ranked_customers.csv", index=False)

    ranked_test = te_ids.copy()
    ranked_test["score"] = pte
    ranked_test.sort_values("score", ascending=False).to_csv(out_dir / "test_ranked_customers.csv", index=False)

    print("[OK] wrote", out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
