from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ID="customer_id"

def minmax(s: pd.Series) -> pd.Series:
    s=pd.to_numeric(s, errors="coerce").astype(float)
    s=s.replace([np.inf,-np.inf], np.nan)
    lo=np.nanmin(s.values)
    hi=np.nanmax(s.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s-lo)/(hi-lo)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ranked_csv", required=True)
    ap.add_argument("--unsup_csv", required=True)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ranked_score_col", default="score")
    ap.add_argument("--unsup_score_col", default="iforest_pct")
    args=ap.parse_args()

    ranked=pd.read_csv(args.ranked_csv)
    uns=pd.read_csv(args.unsup_csv)

    ranked[ID]=ranked[ID].astype(str)
    uns[ID]=uns[ID].astype(str)

    if args.ranked_score_col not in ranked.columns:
        raise SystemExit(f"[ERR] ranked missing {args.ranked_score_col}")
    if args.unsup_score_col not in uns.columns:
        raise SystemExit(f"[ERR] unsup missing {args.unsup_score_col}")

    m=ranked.merge(uns[[ID,args.unsup_score_col]], on=ID, how="left")
    med=pd.to_numeric(m[args.unsup_score_col], errors="coerce").median()
    m[args.unsup_score_col]=pd.to_numeric(m[args.unsup_score_col], errors="coerce").fillna(med)

    sup01=minmax(m[args.ranked_score_col])
    uns01=minmax(m[args.unsup_score_col])

    a=float(args.alpha)
    m["score_sup"]=sup01
    m["score_unsup"]=uns01
    m["score_blended"]=(1-a)*sup01 + a*uns01

    m=m.sort_values("score_blended", ascending=False).reset_index(drop=True)

    out_path=Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out_path, index=False)
    print("[OK] wrote", out_path)

if __name__=="__main__":
    main()

