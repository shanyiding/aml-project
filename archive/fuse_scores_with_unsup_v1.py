from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ID = "customer_id"

def minmax(s: pd.Series) -> pd.Series:
    a = s.astype(float).to_numpy()
    mn, mx = np.nanmin(a), np.nanmax(a)
    return (s.astype(float) - mn) / (mx - mn + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_csv", type=Path, required=True, help="model-ranked customers csv with customer_id, score")
    ap.add_argument("--unsup_parquet", type=Path, default=Path("data/processed/features/customer_unsup_v1.parquet"))
    ap.add_argument("--alpha_model", type=float, default=0.85)
    ap.add_argument("--alpha_unsup", type=float, default=0.15)
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    ranked = pd.read_csv(args.ranked_csv)
    unsup = pd.read_parquet(args.unsup_parquet)

    ranked[ID] = ranked[ID].astype(str)
    unsup[ID] = unsup[ID].astype(str)

    df = ranked.merge(unsup[[ID, "unsup_risk"]], on=ID, how="left")
    df["unsup_risk"] = df["unsup_risk"].fillna(0.0)

    df["score_norm"] = minmax(df["score"])
    df["unsup_norm"] = minmax(df["unsup_risk"])
    df["fused_score"] = args.alpha_model * df["score_norm"] + args.alpha_unsup * df["unsup_norm"]

    df = df.sort_values("fused_score", ascending=False)
    df.to_csv(args.out_csv, index=False)

    pos = df.index[df.get("label", pd.Series([0]*len(df))) == 1].tolist() if "label" in df.columns else []
    print("[OK] wrote", args.out_csv)
    if pos:
        print("[INFO] pos ranks:", pos)

if __name__ == "__main__":
    main()
