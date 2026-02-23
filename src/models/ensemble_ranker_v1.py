from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ID="customer_id"
TARGET="label"

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

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str).str.strip()

    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False).fillna(0)
    ids = df[[ID, TARGET]].copy()
    return X, y, ids

def align_cols(X, cols):
    return X.reindex(columns=cols, fill_value=0)

def pos_rank(ids_df: pd.DataFrame) -> list[int]:
    # ids_df must be sorted desc already
    return ids_df.index[ids_df[TARGET] == 1].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--behavior_path", type=Path, default=Path("data/processed/features/customer_behavior_v2.parquet"))
    ap.add_argument("--emb_prefix", type=str, default="bow_stream_emb_v4")
    ap.add_argument("--model_dirs", type=str, nargs="+", required=True,
                    help="List of run directories under outputs/runs, each must contain model.joblib")
    ap.add_argument("--out_run", type=str, default="ENSEMBLE_xgb_fused")
    ap.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args = ap.parse_args()

    out_dir = Path("outputs/runs") / args.out_run
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = args.processed_dir
    behavior = read_behavior(args.behavior_path)
    cust = read_customers(processed, args.split)
    emb = read_emb(processed, args.emb_prefix, args.split)
    X, y, ids = make_xy(cust, behavior, emb)

    # Load models + determine union of training columns across models
    models = []
    all_cols = set()
    for md in args.model_dirs:
        mpath = Path("outputs/runs") / md / "model.joblib"
        model = joblib.load(mpath)
        models.append((md, model))
        # sklearn wrapper stores feature names if trained on pandas
        if hasattr(model, "feature_names_in_"):
            all_cols |= set(model.feature_names_in_.tolist())

    if not all_cols:
        # fallback: use columns from X itself
        all_cols = set(X.columns.tolist())

    cols = sorted(all_cols)
    X_aligned = align_cols(X, cols)

    preds = []
    for name, model in models:
        # align to model's own cols if available
        if hasattr(model, "feature_names_in_"):
            X_m = align_cols(X_aligned, model.feature_names_in_.tolist())
        else:
            X_m = X_aligned
        p = model.predict_proba(X_m)[:, 1]
        preds.append(p)

    p_avg = np.mean(np.vstack(preds), axis=0)

    ranked = ids.copy()
    ranked["score"] = p_avg
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.to_csv(out_dir / f"{args.split}_ranked_customers.csv", index=False)

    r = ranked.index[ranked[TARGET] == 1].tolist()
    (out_dir / f"{args.split}_pos_ranks.txt").write_text(str(r))

    print("[OK] wrote", out_dir)
    print(f"[INFO] split={args.split} pos_ranks={r} n_models={len(models)} models={[n for n,_ in models]}")

if __name__ == "__main__":
    main()
