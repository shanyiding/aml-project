from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

IDCOL="customer_id"
TARGET="label"

def load_split(processed: Path, split: str, prefix: str):
    cust = pd.read_csv(processed / f"customers_{split}.csv", usecols=[IDCOL, TARGET])
    cust[TARGET] = pd.to_numeric(cust[TARGET], errors="coerce").fillna(0).astype(int)
    emb = pd.read_parquet(processed / "static" / f"{prefix}_{split}.parquet")
    df = cust.merge(emb, on=IDCOL, how="inner")
    X = df.drop(columns=[IDCOL, TARGET]).to_numpy()
    y = df[TARGET].to_numpy()
    ids = df[IDCOL].astype(str).to_numpy()
    return X, y, ids

def eval_topk(y, score, k):
    idx = np.argsort(-score)[:k]
    return int(y[idx].sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--emb_prefix", type=str, default="bow_stream_emb")
    ap.add_argument("--run_name", type=str, default="iforest_emb_v1")
    ap.add_argument("--contamination", type=float, default=0.01)  # shortlist ~1%
    args=ap.parse_args()

    processed=Path(args.processed_dir)
    out_dir=Path("outputs/runs")/args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr,ytr,_   = load_split(processed,"train",args.emb_prefix)
    Xva,yva,_   = load_split(processed,"val",args.emb_prefix)
    Xte,yte,ids = load_split(processed,"test",args.emb_prefix)

    # Fit on ALL train (treat as mostly normal). You can also fit on train negatives only.
    model = IsolationForest(
        n_estimators=500,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr)

    # IsolationForest: lower = more abnormal; we flip sign so higher = riskier
    s_va = -model.score_samples(Xva)
    s_te = -model.score_samples(Xte)

    metrics = {
        "pos_train": int(ytr.sum()),
        "pos_val": int(yva.sum()),
        "pos_test": int(yte.sum()),
        "val_pr_auc": float(average_precision_score(yva, s_va)) if yva.sum()>0 else None,
        "test_pr_auc": float(average_precision_score(yte, s_te)) if yte.sum()>0 else None,
        "test_pos_in_top100": eval_topk(yte, s_te, 100),
        "test_pos_in_top500": eval_topk(yte, s_te, 500),
    }
    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"customer_id":ids,"label":yte,"score":s_te}).sort_values("score",ascending=False)\
      .to_csv(out_dir/"test_ranked_customers.csv", index=False)

    print("[OK] wrote", out_dir)
    print(json.dumps(metrics, indent=2))

if __name__=="__main__":
    main()