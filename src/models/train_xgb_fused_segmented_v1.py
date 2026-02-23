from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ID = "customer_id"
TARGET = "label"

def load_customer_labels(processed_dir: Path, split: str) -> pd.DataFrame:
    cust = pd.read_csv(processed_dir / f"customers_{split}.csv", usecols=[ID, TARGET])
    cust[TARGET] = pd.to_numeric(cust[TARGET], errors="coerce").fillna(0).astype(int)
    return cust

def load_behavior_features(feat_path: Path) -> pd.DataFrame:
    feat = pd.read_parquet(feat_path)
    return feat

def load_embeddings(prefix: str, split: str) -> pd.DataFrame:
    # expects: data/processed/static/{prefix}_{split}.parquet
    p = Path("data/processed/static") / f"{prefix}_{split}.parquet"
    emb = pd.read_parquet(p)
    return emb

def make_design_matrix(processed_dir: Path, feat_path: Path, emb_prefix: str, split: str) -> pd.DataFrame:
    cust = load_customer_labels(processed_dir, split)
    feat = load_behavior_features(feat_path)
    emb = load_embeddings(emb_prefix, split)

    df = cust.merge(feat, on=ID, how="inner")
    df = df.merge(emb, on=ID, how="inner")
    return df

def onehot_df(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    y = df[TARGET].to_numpy()
    X = df.drop(columns=[ID, TARGET], errors="ignore")

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)

    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    ids = df[[ID, TARGET]].copy()
    return X, y, ids

def align_cols(*mats: pd.DataFrame) -> list[pd.DataFrame]:
    cols = sorted(set().union(*[set(m.columns) for m in mats]))
    return [m.reindex(columns=cols, fill_value=0) for m in mats]

def eval_topk(y: np.ndarray, score: np.ndarray, k: int) -> int:
    k = min(k, len(y))
    idx = np.argsort(-score)[:k]
    return int(y[idx].sum())

def pos_ranks(df_ranked: pd.DataFrame) -> list[int]:
    # 0-based index ranks in already-sorted dataframe
    idx = df_ranked.index[df_ranked[TARGET] == 1].tolist()
    return idx

def train_one_segment(
    segment_value: str,
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    segment_col: str,
    out_dir: Path,
    pos_weight: float,
    seed: int,
):
    tr_seg = df_tr[df_tr[segment_col].fillna("Unknown").astype(str) == segment_value].copy()
    va_seg = df_va[df_va[segment_col].fillna("Unknown").astype(str) == segment_value].copy()
    te_seg = df_te[df_te[segment_col].fillna("Unknown").astype(str) == segment_value].copy()

    # if a segment has no rows in val/test, we still train but skip scoring there
    Xtr, ytr, tr_ids = onehot_df(tr_seg)
    Xva, yva, va_ids = onehot_df(va_seg) if len(va_seg) else (pd.DataFrame(), np.array([], dtype=int), pd.DataFrame(columns=[ID, TARGET]))
    Xte, yte, te_ids = onehot_df(te_seg) if len(te_seg) else (pd.DataFrame(), np.array([], dtype=int), pd.DataFrame(columns=[ID, TARGET]))

    mats = [Xtr]
    if len(Xva): mats.append(Xva)
    if len(Xte): mats.append(Xte)
    aligned = align_cols(*mats)
    Xtr = aligned[0]
    j = 1
    if len(Xva):
        Xva = aligned[j]; j += 1
    if len(Xte):
        Xte = aligned[j]

    pos = int(ytr.sum())
    neg = int((ytr == 0).sum())
    print(f"[INFO] segment={segment_value} train_rows={len(ytr)} pos={pos} neg={neg}")

    model = XGBClassifier(
        n_estimators=1600,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        min_child_weight=5,
        gamma=0.0,
        scale_pos_weight=float(pos_weight),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=int(seed),
        n_jobs=-1,
    )

    if len(Xva) and len(np.unique(yva)) > 1:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=100)
    else:
        # if val has only one class (possible with tiny positives), just train without eval_set
        model.fit(Xtr, ytr, verbose=False)

    # SAVE MODEL + FEATURE NAMES
    model_path = out_dir / f"model_{segment_value}.json"
    model.get_booster().save_model(str(model_path))
    (out_dir / f"feature_names_{segment_value}.txt").write_text("\n".join(Xtr.columns.tolist()), encoding="utf-8")

    # Score
    pva = model.predict_proba(Xva)[:, 1] if len(Xva) else np.array([])
    pte = model.predict_proba(Xte)[:, 1] if len(Xte) else np.array([])

    # Return scored frames (sorted) for merging back
    va_ranked = va_ids.copy()
    if len(va_ranked):
        va_ranked["score"] = pva
        va_ranked = va_ranked.sort_values("score", ascending=False).reset_index(drop=True)

    te_ranked = te_ids.copy()
    if len(te_ranked):
        te_ranked["score"] = pte
        te_ranked = te_ranked.sort_values("score", ascending=False).reset_index(drop=True)

    # Metrics for this segment (optional)
    seg_metrics = {
        "segment": segment_value,
        "train_rows": int(len(ytr)),
        "pos_train": int(pos),
        "neg_train": int(neg),
        "val_pr_auc": float(average_precision_score(yva, pva)) if len(yva) and yva.sum() > 0 else None,
        "test_pr_auc": float(average_precision_score(yte, pte)) if len(yte) and yte.sum() > 0 else None,
    }
    (out_dir / f"metrics_{segment_value}.json").write_text(json.dumps(seg_metrics, indent=2), encoding="utf-8")

    return va_ranked, te_ranked, Xtr.shape[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--feat_path", type=Path, default=Path("data/processed/features/customer_behavior_v2.parquet"))
    ap.add_argument("--emb_prefix", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="xgb_fused_segtype_v1")
    ap.add_argument("--segment_col", type=str, default="customer_type")
    ap.add_argument("--pos_weight", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df_tr = make_design_matrix(args.processed_dir, args.feat_path, args.emb_prefix, "train")
    df_va = make_design_matrix(args.processed_dir, args.feat_path, args.emb_prefix, "val")
    df_te = make_design_matrix(args.processed_dir, args.feat_path, args.emb_prefix, "test")

    if args.segment_col not in df_tr.columns:
        raise ValueError(f"segment_col={args.segment_col} not found in training dataframe columns")

    seg_values = sorted(df_tr[args.segment_col].fillna("Unknown").astype(str).unique().tolist())
    print(f"[INFO] emb_prefix={args.emb_prefix} segment_col={args.segment_col} segments={seg_values}")
    print(f"[INFO] Train rows={len(df_tr)} pos={int(df_tr[TARGET].sum())} neg={int((df_tr[TARGET]==0).sum())} pos_weight={args.pos_weight}")
    # global feature count (rough): behavior+emb before one-hot alignment depends on split,
    # but we report final feature count from the first trained segment.
    n_features_any = None

    # Train per segment + then “stitch” rankings across segments by concatenation
    val_parts = []
    test_parts = []

    for seg in seg_values:
        va_ranked, te_ranked, nf = train_one_segment(
            seg, df_tr, df_va, df_te, args.segment_col, out_dir, args.pos_weight, args.seed
        )
        if n_features_any is None:
            n_features_any = int(nf)
        if len(va_ranked):
            va_ranked["segment"] = seg
            val_parts.append(va_ranked)
        if len(te_ranked):
            te_ranked["segment"] = seg
            test_parts.append(te_ranked)

    # Combine back into a single ranked list:
    # IMPORTANT: scores are not calibrated across segments, so this concat is mainly for inspection.
    # If you want a true global ranking, you should standardize scores within segment first.
    val_all = pd.concat(val_parts, axis=0, ignore_index=True) if val_parts else pd.DataFrame(columns=[ID, TARGET, "score", "segment"])
    test_all = pd.concat(test_parts, axis=0, ignore_index=True) if test_parts else pd.DataFrame(columns=[ID, TARGET, "score", "segment"])

    # simple within-segment standardization to allow cross-segment ranking (fast + usually helps)
    def zscore_within(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        out = []
        for seg, g in df.groupby("segment", sort=False):
            s = g["score"].astype(float).to_numpy()
            mu = float(np.mean(s)) if len(s) else 0.0
            sd = float(np.std(s)) if len(s) else 1.0
            sd = sd if sd > 1e-9 else 1.0
            g = g.copy()
            g["score"] = (g["score"].astype(float) - mu) / sd
            out.append(g)
        return pd.concat(out, axis=0, ignore_index=True)

    val_all = zscore_within(val_all).sort_values("score", ascending=False).reset_index(drop=True)
    test_all = zscore_within(test_all).sort_values("score", ascending=False).reset_index(drop=True)

    val_all.to_csv(out_dir / "val_ranked_customers.csv", index=False)
    test_all.to_csv(out_dir / "test_ranked_customers.csv", index=False)

    # Metrics on combined
    yva = val_all[TARGET].to_numpy() if not val_all.empty else np.array([], dtype=int)
    pva = val_all["score"].to_numpy() if not val_all.empty else np.array([], dtype=float)
    yte = test_all[TARGET].to_numpy() if not test_all.empty else np.array([], dtype=int)
    pte = test_all["score"].to_numpy() if not test_all.empty else np.array([], dtype=float)

    metrics = {
        "pos_train": int(df_tr[TARGET].sum()),
        "pos_val": int(df_va[TARGET].sum()),
        "pos_test": int(df_te[TARGET].sum()),
        "val_pr_auc": float(average_precision_score(yva, pva)) if len(yva) and yva.sum() > 0 else None,
        "test_pr_auc": float(average_precision_score(yte, pte)) if len(yte) and yte.sum() > 0 else None,
        "val_roc_auc": float(roc_auc_score(yva, pva)) if len(yva) and len(np.unique(yva)) > 1 else None,
        "test_roc_auc": float(roc_auc_score(yte, pte)) if len(yte) and len(np.unique(yte)) > 1 else None,
        "test_pos_in_top100": eval_topk(yte, pte, 100) if len(yte) else 0,
        "test_pos_in_top500": eval_topk(yte, pte, 500) if len(yte) else 0,
        "test_recall@100": float(eval_topk(yte, pte, 100) / max(1, int(yte.sum()))) if len(yte) else 0.0,
        "test_recall@500": float(eval_topk(yte, pte, 500) / max(1, int(yte.sum()))) if len(yte) else 0.0,
        "n_features": int(n_features_any or 0),
        "emb_prefix": args.emb_prefix,
        "segment_col": args.segment_col,
        "segments_trained": seg_values,
        "seed": int(args.seed),
        "pos_weight": float(args.pos_weight),
        "score_merge": "zscore_within_segment_then_global_sort",
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[OK] wrote", out_dir)
    print(json.dumps(metrics, indent=2))
    if not val_all.empty:
        print("val pos ranks:", val_all.index[val_all[TARGET] == 1].tolist())
    if not test_all.empty:
        print("test pos rank:", test_all.index[test_all[TARGET] == 1].tolist())

if __name__ == "__main__":
    main()
