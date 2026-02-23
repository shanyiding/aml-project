from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

IDCOL="customer_id"
TARGET="label"

def recall_at_k(y, s, k):
    k=min(k,len(y))
    idx=np.argsort(-s)[:k]
    return float(y[idx].sum()/max(1,int(y.sum())))

def pos_in_top_k(y,s,k):
    k=min(k,len(y))
    idx=np.argsort(-s)[:k]
    return int(y[idx].sum())

def load_split(processed: Path, split: str):
    cust = pd.read_csv(processed / f"customers_{split}.csv", usecols=[IDCOL, TARGET])
    cust[TARGET] = pd.to_numeric(cust[TARGET], errors="coerce").fillna(0).astype(int)
    emb = pd.read_parquet(processed / "static" / f"seq_embeddings_{split}.parquet")
    df = cust.merge(emb, on=IDCOL, how="inner")
    X = df.drop(columns=[IDCOL, TARGET])
    y = df[TARGET].to_numpy()
    ids = df[IDCOL].astype(str).to_numpy()
    return X,y,ids

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--run_name", type=str, default="xgb_byol_seq_v1")
    args=ap.parse_args()

    processed=Path(args.processed_dir)
    out_dir=Path("outputs/runs")/args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr,ytr,_ = load_split(processed,"train")
    Xva,yva,_ = load_split(processed,"val")
    Xte,yte,ids = load_split(processed,"test")

    pos=int(ytr.sum()); neg=int(len(ytr)-pos)
    spw=neg/max(1,pos)

    model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=100)

    pva=model.predict_proba(Xva)[:,1]
    pte=model.predict_proba(Xte)[:,1]

    metrics={
        "pos_train":pos,
        "pos_val":int(yva.sum()),
        "pos_test":int(yte.sum()),
        "val_pr_auc": float(average_precision_score(yva,pva)) if yva.sum()>0 else None,
        "test_pr_auc": float(average_precision_score(yte,pte)) if yte.sum()>0 else None,
        "test_roc_auc": float(roc_auc_score(yte,pte)) if len(np.unique(yte))>1 else None,
        "test_pos_in_top100": pos_in_top_k(yte,pte,100),
        "test_pos_in_top500": pos_in_top_k(yte,pte,500),
        "test_recall@100": recall_at_k(yte,pte,100),
        "test_recall@500": recall_at_k(yte,pte,500),
    }

    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    ranked=pd.DataFrame({"customer_id":ids,"label":yte,"score":pte}).sort_values("score",ascending=False)
    ranked.to_csv(out_dir/"test_ranked_customers.csv",index=False)

    print("[OK] wrote", out_dir)
    print(json.dumps(metrics, indent=2))

if __name__=="__main__":
    main()