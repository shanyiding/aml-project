from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

ID = "customer_id"
TARGET = "label"


def read_customers(processed: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(processed / f"customers_{split}.csv", usecols=[ID, TARGET])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    df[ID] = df[ID].astype(str)
    return df


def read_behavior(feat_path: Path) -> pd.DataFrame:
    b = pd.read_parquet(feat_path)
    if TARGET in b.columns:
        b = b.drop(columns=[TARGET], errors="ignore")
    b[ID] = b[ID].astype(str)
    return b


def read_emb(processed: Path, emb_prefix: str, split: str) -> pd.DataFrame:
    p = processed / "static" / f"{emb_prefix}_{split}.parquet"
    e = pd.read_parquet(p)
    e[ID] = e[ID].astype(str)
    return e


def make_Xy(customers: pd.DataFrame, behavior: pd.DataFrame, emb: pd.DataFrame):
    df = customers.merge(behavior, on=ID, how="left").merge(emb, on=ID, how="left")
    y = df[TARGET].to_numpy().astype(int)
    X = df.drop(columns=[ID, TARGET], errors="ignore")

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str).str.strip()

    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    X = X.fillna(0)

    ids = df[[ID, TARGET]].copy()
    return X, y, ids


def align_cols(*Xs):
    cols = sorted(set().union(*[set(X.columns) for X in Xs]))
    return [X.reindex(columns=cols, fill_value=0) for X in Xs], cols


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


def positive_ranks(ids_df: pd.DataFrame, score: np.ndarray):
    # ids_df aligned with score order
    # return ranks (0-based) of label==1 after sorting by score desc
    tmp = ids_df.copy()
    tmp["score"] = score
    tmp = tmp.sort_values("score", ascending=False).reset_index(drop=True)
    return tmp.index[tmp[TARGET] == 1].tolist(), tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--behavior_path", type=Path, default=Path("data/processed/features/customer_behavior_v2.parquet"))
    ap.add_argument("--emb_prefix", type=str, default="bow_stream_emb_v4")

    ap.add_argument("--rn_csv", type=Path, required=True, help="reliable negatives csv, e.g. data/processed/rn/reliable_negatives_k02.csv")
    ap.add_argument("--run_name", type=str, default="pu_xgb_bagged_v1")

    # PU / bagging knobs
    ap.add_argument("--bags", type=int, default=25, help="number of RN-subsample bags")
    ap.add_argument("--neg_per_pos", type=int, default=500, help="RN sampled per positive per bag")
    ap.add_argument("--use_labeled_negatives", type=int, default=1, help="1 include labeled train negatives, 0 exclude")
    ap.add_argument("--labeled_neg_weight", type=float, default=0.2, help="downweight labeled negatives if included")
    ap.add_argument("--rn_weight", type=float, default=1.0, help="weight for reliable negatives")
    ap.add_argument("--pos_weight", type=float, default=20.0, help="weight for labeled positives in training")

    # XGBoost knobs
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--max_depth", type=int, default=2)
    ap.add_argument("--min_child_weight", type=float, default=10.0)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--tree_method", type=str, default="hist")

    ap.add_argument("--num_boost_round", type=int, default=8000)
    ap.add_argument("--early_stopping_rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = args.processed_dir

    # Load labeled splits
    tr_c = read_customers(processed, "train")
    va_c = read_customers(processed, "val")
    te_c = read_customers(processed, "test")

    # Features
    behavior = read_behavior(args.behavior_path)
    tr_e = read_emb(processed, args.emb_prefix, "train")
    va_e = read_emb(processed, args.emb_prefix, "val")
    te_e = read_emb(processed, args.emb_prefix, "test")

    Xtr_full, ytr_full, tr_ids = make_Xy(tr_c, behavior, tr_e)
    Xva, yva, va_ids = make_Xy(va_c, behavior, va_e)
    Xte, yte, te_ids = make_Xy(te_c, behavior, te_e)

    (Xtr_full, Xva, Xte), cols = align_cols(Xtr_full, Xva, Xte)

    # PU components
    tr_pos_mask = (ytr_full == 1)
    tr_neg_mask = (ytr_full == 0)

    pos_ids = set(tr_c.loc[tr_pos_mask, ID].astype(str))
    labeled_ids_all = set(pd.concat([tr_c[ID], va_c[ID], te_c[ID]]).astype(str))

    rn = pd.read_csv(args.rn_csv)
    if ID not in rn.columns:
        raise SystemExit(f"[ERR] rn_csv missing {ID}: {args.rn_csv}")
    rn[ID] = rn[ID].astype(str)

    # Ensure no leakage: RN must not include any labeled split IDs
    rn = rn[~rn[ID].isin(labeled_ids_all)].copy()
    rn_pool = rn[ID].astype(str).tolist()

    if len(rn_pool) == 0:
        raise SystemExit("[ERR] RN pool became empty after leakage filter. Check rn_csv or your splits.")

    n_pos = int(tr_pos_mask.sum())
    n_neg_labeled = int(tr_neg_mask.sum())

    print(f"[INFO] emb_prefix={args.emb_prefix}")
    print(f"[INFO] Train labeled rows={len(ytr_full)} pos={n_pos} neg={n_neg_labeled}")
    print(f"[INFO] RN pool (unlabeled) after leakage filter = {len(rn_pool)}")
    print(f"[INFO] bags={args.bags} neg_per_pos={args.neg_per_pos} use_labeled_negatives={args.use_labeled_negatives}")

    rng = np.random.default_rng(args.seed)

    # Pre-slice training matrices for positives / labeled negs
    X_pos = Xtr_full.loc[tr_pos_mask].copy()
    y_pos = np.ones(len(X_pos), dtype=int)

    X_neg_labeled = Xtr_full.loc[tr_neg_mask].copy()
    y_neg_labeled = np.zeros(len(X_neg_labeled), dtype=int)

    # Create a fast lookup from id -> row in full TRAIN X to pick RN rows
    # RN are unlabeled, so we must build features for them from behavior/emb sources.
    # Strategy: build feature matrix for RN using the same feature tables (behavior + embeddings) from processed/static.
    # We DO NOT have a "customers_unlabeled" split here, so we load from customers_master_unlabeled if needed is out-of-scope;
    # but we can still score using TRAIN feature tables only if embeddings/behavior contain RN IDs.
    #
    # We'll build a separate feature frame for RN by merging RN IDs into behavior + "all embeddings" cache:
    # NOTE: this assumes your exported embeddings cache contains those unlabeled ids elsewhere.
    #
    # Minimal approach: use behavior features only for RN if embeddings for RN aren't available.
    #
    # We'll attempt to load embeddings shards for unlabeled if present:
    rn_feat = behavior[behavior[ID].isin(rn_pool)].copy()

    # Try to load an unlabeled embedding parquet if it exists: data/processed/static/{emb_prefix}_unlabeled.parquet
    emb_unl_path = processed / "static" / f"{args.emb_prefix}_unlabeled.parquet"
    if emb_unl_path.exists():
        emb_u = pd.read_parquet(emb_unl_path)
        emb_u[ID] = emb_u[ID].astype(str)
        rn_feat = rn_feat.merge(emb_u, on=ID, how="left")
        print(f"[INFO] using embeddings for unlabeled: {emb_unl_path}")
    else:
        print(f"[WARN] unlabeled embedding file not found: {emb_unl_path}")
        print("[WARN] RN will use ONLY behavior features (no seq embedding) unless behavior table already includes embedding cols.")

    # Attach minimal customer fields for categoricals? If behavior already numeric, fine.
    # Ensure same dummy encoding columns as train by combining then aligning.
    rn_feat[TARGET] = 0
    rn_feat = rn_feat.rename(columns={ID: ID})

    # Rebuild PU matrices with the same preprocessing as make_Xy:
    # - treat object cols as categoricals
    # - get_dummies
    # We'll fit dummies on combined {train full + rn_feat}, then align to cols.
    def prep_X(df_like: pd.DataFrame):
        X = df_like.drop(columns=[TARGET], errors="ignore").copy()
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        for c in cat_cols:
            X[c] = X[c].fillna("Unknown").astype(str).str.strip()
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
        X = X.fillna(0)
        return X

    # Build base to align RN features with train/val/test columns
    # Use the already-aligned train columns `cols` for final alignment.
    # We need RN features in same space: take rn_feat -> prep -> reindex(cols)
    # But rn_feat doesn't contain the exact same set as Xtr_full because Xtr_full came from customer table merge too.
    # We'll simply reindex to cols; missing become 0.
    X_rn_all = prep_X(rn_feat.drop(columns=[ID], errors="ignore"))
    # X_rn_all currently lacks ID column because we dropped; keep parallel id list:
    rn_ids_all = rn_feat[ID].astype(str).tolist()

    # Reindex RN to training column space
    X_rn_all = X_rn_all.reindex(columns=cols, fill_value=0)

    if X_rn_all.shape[0] == 0:
        raise SystemExit("[ERR] No RN rows had features in behavior/embedding tables. Check feature tables coverage for unlabeled.")

    # Bagging training
    pva_sum = np.zeros(len(yva), dtype=float)
    pte_sum = np.zeros(len(yte), dtype=float)

    models = []
    bag_meta = []

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.reg_lambda,
        "gamma": args.gamma,
        "tree_method": args.tree_method,
        "seed": args.seed,
    }

    for b in range(args.bags):
        # Sample RN indices
        n_rn_sample = min(len(rn_ids_all), max(1, args.neg_per_pos * max(n_pos, 1)))
        rn_idx = rng.choice(len(rn_ids_all), size=n_rn_sample, replace=False)

        X_rn = X_rn_all.iloc[rn_idx]
        y_rn = np.zeros(len(X_rn), dtype=int)

        # Assemble PU train set
        X_parts = [X_pos, X_rn]
        y_parts = [y_pos, y_rn]
        w_parts = [
            np.full(len(X_pos), args.pos_weight, dtype=float),
            np.full(len(X_rn), args.rn_weight, dtype=float),
        ]

        if args.use_labeled_negatives:
            # labeled negs are "not confirmed normal"; downweight so they don't dominate
            X_parts.append(X_neg_labeled)
            y_parts.append(y_neg_labeled)
            w_parts.append(np.full(len(X_neg_labeled), args.labeled_neg_weight, dtype=float))

        X_bag = pd.concat(X_parts, axis=0)
        y_bag = np.concatenate(y_parts, axis=0)
        w_bag = np.concatenate(w_parts, axis=0)

        dtrain = xgb.DMatrix(X_bag, label=y_bag, weight=w_bag, feature_names=cols)
        dval = xgb.DMatrix(Xva, label=yva, feature_names=cols)

        callbacks = []
        if args.early_stopping_rounds and args.early_stopping_rounds > 0:
            callbacks.append(xgb.callback.EarlyStopping(rounds=args.early_stopping_rounds, save_best=True))

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.num_boost_round,
            evals=[(dval, "validation")],
            verbose_eval=200,
            callbacks=callbacks if callbacks else None,
        )

        # Predict
        pva = booster.predict(xgb.DMatrix(Xva, feature_names=cols))
        pte = booster.predict(xgb.DMatrix(Xte, feature_names=cols))

        pva_sum += pva
        pte_sum += pte

        models.append(booster)
        bag_meta.append(
            {
                "bag": b,
                "rn_sample": int(n_rn_sample),
                "best_iteration": int(getattr(booster, "best_iteration", -1)),
                "best_score": float(getattr(booster, "best_score", float("nan"))),
            }
        )

        print(f"[INFO] bag {b+1}/{args.bags} done; rn_sample={n_rn_sample} best_it={bag_meta[-1]['best_iteration']} best_score={bag_meta[-1]['best_score']}")

    # Average probs
    pva_avg = pva_sum / args.bags
    pte_avg = pte_sum / args.bags

    # Metrics (note: very noisy when pos_val=2, pos_test=1)
    metrics = {
        "approach": "PU bagging with reliable negatives",
        "pos_train": int(ytr_full.sum()),
        "pos_val": int(yva.sum()),
        "pos_test": int(yte.sum()),
        "bags": int(args.bags),
        "neg_per_pos": int(args.neg_per_pos),
        "use_labeled_negatives": int(args.use_labeled_negatives),
        "weights": {
            "pos_weight": float(args.pos_weight),
            "rn_weight": float(args.rn_weight),
            "labeled_neg_weight": float(args.labeled_neg_weight),
        },
        "val_pr_auc": float(average_precision_score(yva, pva_avg)) if int(yva.sum()) > 0 else None,
        "test_pr_auc": float(average_precision_score(yte, pte_avg)) if int(yte.sum()) > 0 else None,
        "val_roc_auc": float(roc_auc_score(yva, pva_avg)) if len(np.unique(yva)) > 1 else None,
        "test_roc_auc": float(roc_auc_score(yte, pte_avg)) if len(np.unique(yte)) > 1 else None,
        "val_pos_in_top100": pos_in_topk(yva, pva_avg, 100),
        "test_pos_in_top100": pos_in_topk(yte, pte_avg, 100),
        "test_recall@100": recall_at_k(yte, pte_avg, 100),
        "n_features": int(len(cols)),
        "emb_prefix": args.emb_prefix,
        "seed": int(args.seed),
        "xgb_params": params,
        "bag_meta": bag_meta,
        "rn_csv": str(args.rn_csv),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save boosters list (optional)
    joblib.dump(models, out_dir / "boosters.joblib")

    # Ranked outputs
    ranked_val = va_ids.copy()
    ranked_val["score"] = pva_avg
    ranked_val = ranked_val.sort_values("score", ascending=False).reset_index(drop=True)
    ranked_val.to_csv(out_dir / "val_ranked_customers.csv", index=False)

    ranked_test = te_ids.copy()
    ranked_test["score"] = pte_avg
    ranked_test = ranked_test.sort_values("score", ascending=False).reset_index(drop=True)
    ranked_test.to_csv(out_dir / "test_ranked_customers.csv", index=False)

    val_ranks, _ = positive_ranks(va_ids, pva_avg)
    test_ranks, _ = positive_ranks(te_ids, pte_avg)

    print("[OK] wrote", out_dir)
    print(json.dumps(metrics, indent=2))
    print("val pos ranks:", val_ranks)
    print("test pos rank:", test_ranks)


if __name__ == "__main__":
    main()
