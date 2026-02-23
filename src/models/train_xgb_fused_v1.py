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


def read_customers(processed: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(processed / f"customers_{split}.csv", usecols=[ID, TARGET])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return df


def read_behavior(feat_path: Path) -> pd.DataFrame:
    b = pd.read_parquet(feat_path)
    if TARGET in b.columns:
        b = b.drop(columns=[TARGET], errors="ignore")
    return b


def read_emb(processed: Path, emb_prefix: str, split: str) -> pd.DataFrame:
    p = processed / "static" / f"{emb_prefix}_{split}.parquet"
    return pd.read_parquet(p)


def make_xy(customers: pd.DataFrame, behavior: pd.DataFrame, emb: pd.DataFrame):
    df = customers.merge(behavior, on=ID, how="left").merge(emb, on=ID, how="left")

    y = df[TARGET].to_numpy()
    X = df.drop(columns=[ID, TARGET], errors="ignore")

    # encode categoricals
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str).str.strip()

    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    X = X.fillna(0)

    ids = df[[ID, TARGET]].copy()
    return X, y, ids


def align_cols(Xtr, Xva, Xte):
    cols = sorted(set(Xtr.columns) | set(Xva.columns) | set(Xte.columns))
    return (
        Xtr.reindex(columns=cols, fill_value=0),
        Xva.reindex(columns=cols, fill_value=0),
        Xte.reindex(columns=cols, fill_value=0),
        cols,
    )


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
    ap.add_argument(
        "--behavior_path",
        type=Path,
        default=Path("data/processed/features/customer_behavior_v2.parquet"),
    )
    ap.add_argument("--emb_prefix", type=str, default="bow_stream_emb_v2")
    ap.add_argument("--run_name", type=str, default="xgb_fused_v1")

    # imbalance + training length
    ap.add_argument("--pos_weight", type=float, default=-1.0)  # if -1, auto = neg/pos
    ap.add_argument("--n_estimators", type=int, default=1600)
    ap.add_argument("--early_stopping_rounds", type=int, default=0)  # 0 = disabled
    ap.add_argument("--seed", type=int, default=42)

    # regularization / capacity knobs (so you can tune from CLI)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--max_depth", type=int, default=2)
    ap.add_argument("--min_child_weight", type=float, default=5.0)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--tree_method", type=str, default="hist")  # fast + stable

    args = ap.parse_args()

    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = args.processed_dir
    behavior = read_behavior(args.behavior_path)

    tr_c = read_customers(processed, "train")
    va_c = read_customers(processed, "val")
    te_c = read_customers(processed, "test")

    tr_e = read_emb(processed, args.emb_prefix, "train")
    va_e = read_emb(processed, args.emb_prefix, "val")
    te_e = read_emb(processed, args.emb_prefix, "test")

    Xtr, ytr, tr_ids = make_xy(tr_c, behavior, tr_e)
    Xva, yva, va_ids = make_xy(va_c, behavior, va_e)
    Xte, yte, te_ids = make_xy(te_c, behavior, te_e)
    Xtr, Xva, Xte, cols = align_cols(Xtr, Xva, Xte)

    pos = int(ytr.sum())
    neg = int((ytr == 0).sum())
    spw = float(args.pos_weight) if args.pos_weight > 0 else (neg / max(pos, 1))

    print(f"[INFO] emb_prefix={args.emb_prefix}")
    print(f"[INFO] Train rows={len(ytr)} pos={pos} neg={neg} scale_pos_weight={spw:.2f}")
    print(f"[INFO] n_features={len(cols)}")
    print(f"[INFO] early_stopping_rounds={args.early_stopping_rounds} (0=disabled)")

    # IMPORTANT: In your environment, early stopping must be configured on the estimator,
    # not passed into fit(). We keep fit() clean and only pass eval_set + verbose.
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.eta,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method=args.tree_method,
        random_state=args.seed,
        n_jobs=-1,
        early_stopping_rounds=(args.early_stopping_rounds if args.early_stopping_rounds > 0 else None),
    )

    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=100)

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
        "emb_prefix": args.emb_prefix,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "eta": float(args.eta),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_lambda": float(args.reg_lambda),
        "gamma": float(args.gamma),
        "tree_method": args.tree_method,
        "scale_pos_weight": float(spw),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, out_dir / "model.joblib")

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
