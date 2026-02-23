from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

IDCOL="customer_id"
TARGET="label"

def mk_seg(df: pd.DataFrame) -> pd.Series:
    def col(name, default="UNK"):
        if name in df.columns:
            return df[name].fillna(default).astype(str).str.strip()
        return pd.Series([default]*len(df), index=df.index)
    return col("customer_type") + "|" + col("industry_code") + "|" + col("province")

def load_split(processed: Path, split: str, prefix: str):
    cust = pd.read_csv(processed / f"customers_{split}.csv", low_memory=False)
    cust[TARGET] = pd.to_numeric(cust[TARGET], errors="coerce").fillna(0).astype(int)
    emb = pd.read_parquet(processed / "static" / f"{prefix}_{split}.parquet")
    df = cust.merge(emb, on=IDCOL, how="inner")
    if df.empty:
        raise RuntimeError(f"Empty merge split={split}")
    seg = mk_seg(df)
    X = df.drop(columns=[IDCOL, TARGET], errors="ignore")
    # drop non-embedding columns accidentally included
    # keep only embedding columns by prefix match
    emb_cols = [c for c in X.columns if c.startswith(prefix)]
    X = X[emb_cols].to_numpy()
    y = df[TARGET].to_numpy()
    ids = df[IDCOL].astype(str).to_numpy()
    return X,y,ids,seg

def eval_topk(y, score, k):
    idx = np.argsort(-score)[:min(k,len(y))]
    return int(y[idx].sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--emb_prefix", type=str, default="bow_stream_emb")
    ap.add_argument("--run_name", type=str, default="iforest_seg_v1")
    ap.add_argument("--contamination", type=float, default=0.01)
    ap.add_argument("--min_group", type=int, default=300)
    args=ap.parse_args()

    processed=Path(args.processed_dir)
    out_dir=Path("outputs/runs")/args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr,ytr,_,seg_tr = load_split(processed,"train",args.emb_prefix)
    Xva,yva,_,seg_va = load_split(processed,"val",args.emb_prefix)
    Xte,yte,ids,seg_te = load_split(processed,"test",args.emb_prefix)

    # Fit a global model as fallback
    global_if = IsolationForest(
        n_estimators=500,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    ).fit(Xtr)

    # Build per-segment models using TRAIN only
    models = {}
    seg_vals, seg_counts = np.unique(seg_tr.to_numpy(), return_counts=True)
    for s, n in zip(seg_vals, seg_counts):
        if n < args.min_group:
            continue
        idx = np.where(seg_tr.to_numpy() == s)[0]
        models[s] = IsolationForest(
            n_estimators=300,
            contamination=args.contamination,
            random_state=42,
            n_jobs=-1,
        ).fit(Xtr[idx])

    def score(X, seg):
        seg_arr = seg.to_numpy()
        out = np.zeros(X.shape[0], dtype=np.float32)
        for i in range(X.shape[0]):
            m = models.get(seg_arr[i], None)
            if m is None:
                out[i] = -global_if.score_samples(X[i:i+1])[0]
            else:
                out[i] = -m.score_samples(X[i:i+1])[0]
        return out

    s_va = score(Xva, seg_va)
    s_te = score(Xte, seg_te)

    metrics = {
        "pos_train": int(ytr.sum()),
        "pos_val": int(yva.sum()),
        "pos_test": int(yte.sum()),
        "n_segment_models": int(len(models)),
        "val_pr_auc": float(average_precision_score(yva, s_va)) if yva.sum()>0 else None,
        "test_pr_auc": float(average_precision_score(yte, s_te)) if yte.sum()>0 else None,
        "test_pos_in_top100": eval_topk(yte, s_te, 100),
        "test_pos_in_top500": eval_topk(yte, s_te, 500),
    }

    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"customer_id":ids,"label":yte,"score":s_te, "segment":seg_te.to_numpy()})\
      .sort_values("score",ascending=False)\
      .to_csv(out_dir/"test_ranked_customers.csv", index=False)

    print("[OK] wrote", out_dir)
    print(json.dumps(metrics, indent=2))

if __name__=="__main__":
    main()