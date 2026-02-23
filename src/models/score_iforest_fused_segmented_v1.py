from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

ID = "customer_id"
TARGET = "label"


def load_fused_split(processed_dir: Path, feat_path: Path, emb_prefix: str, split: str) -> pd.DataFrame:
    # labels live in customers_{split}.csv
    cust = pd.read_csv(processed_dir / f"customers_{split}.csv")
    if ID not in cust.columns:
        raise ValueError(f"{ID} not in customers_{split}.csv")
    if TARGET not in cust.columns:
        raise ValueError(f"{TARGET} not in customers_{split}.csv")
    cust[TARGET] = pd.to_numeric(cust[TARGET], errors="coerce").fillna(0).astype(int)

    # behavior/static features (must contain customer_type if you want to segment on it)
    feat = pd.read_parquet(feat_path)
    if ID not in feat.columns:
        raise ValueError(f"{ID} not in {feat_path}")

    # embeddings per split
    emb_file = processed_dir / "static" / f"{emb_prefix}_{split}.parquet"
    if not emb_file.exists():
        raise FileNotFoundError(f"Missing embeddings file: {emb_file}")
    emb = pd.read_parquet(emb_file)
    if ID not in emb.columns:
        raise ValueError(f"{ID} not in {emb_file}")

    # merge: keep label + any customer static columns (from customers_*.csv) optional, but we segment from feat
    df = cust[[ID, TARGET]].merge(feat, on=ID, how="inner").merge(emb, on=ID, how="left")
    return df


def one_hot_train(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    cols = X.columns.tolist()
    return X, cols


def one_hot_apply(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    # align to train columns (no leakage)
    return X.reindex(columns=cols, fill_value=0)


def topk_hits(y: np.ndarray, score: np.ndarray, k: int) -> int:
    k = min(k, len(y))
    idx = np.argsort(-score)[:k]
    return int(y[idx].sum())


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def safe_ap(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.sum() <= 0:
        return None
    return float(average_precision_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--feat_path", type=Path, default=Path("data/processed/features/customer_behavior_v2.parquet"))
    ap.add_argument("--emb_prefix", type=str, required=True)
    ap.add_argument("--segment_col", type=str, default="customer_type")
    ap.add_argument("--contamination", type=float, default=0.01)
    ap.add_argument("--run_name", type=str, default="iforest_fused_seg_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_group", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df_tr = load_fused_split(args.processed_dir, args.feat_path, args.emb_prefix, "train")
    df_va = load_fused_split(args.processed_dir, args.feat_path, args.emb_prefix, "val")
    df_te = load_fused_split(args.processed_dir, args.feat_path, args.emb_prefix, "test")

    if args.segment_col not in df_tr.columns:
        raise ValueError(f"segment_col={args.segment_col} not found in merged train df. "
                         f"Available cols include: {list(df_tr.columns)[:30]}...")

    # segments observed in train
    seg_values = sorted(df_tr[args.segment_col].fillna("Unknown").astype(str).unique().tolist())
    print(f"[INFO] emb_prefix={args.emb_prefix} segment_col={args.segment_col} segments={seg_values}")
    print(f"[INFO] Train rows={len(df_tr)} pos={int(df_tr[TARGET].sum())} neg={int((df_tr[TARGET]==0).sum())}")

    def build_X(df: pd.DataFrame) -> pd.DataFrame:
        # drop identifiers + target; keep segment_col out of features to avoid trivial segmentation leakage
        X = df.drop(columns=[ID, TARGET], errors="ignore").copy()
        if args.segment_col in X.columns:
            X = X.drop(columns=[args.segment_col], errors="ignore")
        # numeric cleanup
        for c in X.columns:
            if X[c].dtype != "object":
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        return X

    # scores for all rows, filled by segment model
    def score_split(df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        y = df[TARGET].to_numpy()
        seg = df[args.segment_col].fillna("Unknown").astype(str).to_numpy()
        scores = np.full(len(df), np.nan, dtype=float)
        return df[[ID, TARGET]].copy(), y, scores, seg

    va_ids, yva, va_scores, va_seg = score_split(df_va, "val")
    te_ids, yte, te_scores, te_seg = score_split(df_te, "test")

    trained_segments = []
    for sv in seg_values:
        tr_mask = df_tr[args.segment_col].fillna("Unknown").astype(str).to_numpy() == sv
        n_tr = int(tr_mask.sum())
        if n_tr < args.min_group:
            print(f"[WARN] segment={sv} train_rows={n_tr} < min_group={args.min_group}, skipping")
            continue

        trained_segments.append(sv)

        df_tr_s = df_tr.loc[tr_mask].copy()
        Xtr_raw = build_X(df_tr_s)

        # train one-hot on train segment only
        Xtr, cols = one_hot_train(Xtr_raw)

        model = IsolationForest(
            n_estimators=400,
            contamination=args.contamination,
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(Xtr)

        # val scoring for that segment (aligned to train columns)
        va_mask = va_seg == sv
        if va_mask.any():
            Xva = one_hot_apply(build_X(df_va.loc[va_mask].copy()), cols)
            # higher => more anomalous
            va_scores[va_mask] = -model.score_samples(Xva)

        # test scoring
        te_mask = te_seg == sv
        if te_mask.any():
            Xte = one_hot_apply(build_X(df_te.loc[te_mask].copy()), cols)
            te_scores[te_mask] = -model.score_samples(Xte)

        print(f"[INFO] segment={sv} train_rows={n_tr}")

    # fallback for any segment not trained (rare): global model
    if np.isnan(va_scores).any() or np.isnan(te_scores).any():
        print("[WARN] Some rows not scored by segment models; fitting GLOBAL fallback model on all train.")
        Xtr_raw = build_X(df_tr.copy())
        Xtr, cols = one_hot_train(Xtr_raw)
        gmodel = IsolationForest(
            n_estimators=400,
            contamination=args.contamination,
            random_state=args.seed,
            n_jobs=-1,
        )
        gmodel.fit(Xtr)

        if np.isnan(va_scores).any():
            Xva = one_hot_apply(build_X(df_va.copy()), cols)
            va_scores = np.where(np.isnan(va_scores), -gmodel.score_samples(Xva), va_scores)

        if np.isnan(te_scores).any():
            Xte = one_hot_apply(build_X(df_te.copy()), cols)
            te_scores = np.where(np.isnan(te_scores), -gmodel.score_samples(Xte), te_scores)

    # metrics (rank-focused)
    metrics = {
        "pos_train": int(df_tr[TARGET].sum()),
        "pos_val": int(yva.sum()),
        "pos_test": int(yte.sum()),
        "val_pr_auc": safe_ap(yva, va_scores),
        "test_pr_auc": safe_ap(yte, te_scores),
        "val_roc_auc": safe_auc(yva, va_scores),
        "test_roc_auc": safe_auc(yte, te_scores),
        "test_pos_in_top100": topk_hits(yte, te_scores, 100),
        "test_pos_in_top500": topk_hits(yte, te_scores, 500),
        "segment_col": args.segment_col,
        "segments_trained": trained_segments,
        "contamination": args.contamination,
        "emb_prefix": args.emb_prefix,
        "seed": args.seed,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[OK] wrote", out_dir)
    print(json.dumps(metrics, indent=2))

    # ranked val
    ranked_val = va_ids.copy()
    ranked_val["score"] = va_scores
    ranked_val.sort_values("score", ascending=False).to_csv(out_dir / "val_ranked_customers.csv", index=False)

    # ranked test
    ranked_test = te_ids.copy()
    ranked_test["score"] = te_scores
    ranked_test.sort_values("score", ascending=False).to_csv(out_dir / "test_ranked_customers.csv", index=False)


if __name__ == "__main__":
    main()
