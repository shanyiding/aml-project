from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ID = "customer_id"

def build_X(feat: pd.DataFrame) -> pd.DataFrame:
    X = feat.copy()
    X = X.drop(columns=[ID], errors="ignore")

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in X.columns:
        if X[c].dtype == "bool":
            X[c] = X[c].astype(int)
    return X

def pick_segment_col(df: pd.DataFrame, segment_col: str) -> str:
    # Prefer column from unlabeled side if we merged with suffixes
    cand = [
        segment_col,
        f"{segment_col}_u",
        f"{segment_col}_x",
        f"{segment_col}_left",
        f"{segment_col}_unlabeled",
        f"{segment_col}_U",
    ]
    for c in cand:
        if c in df.columns:
            return c
    # Fallback: any column that starts with segment_col (e.g. customer_type_feat)
    starts = [c for c in df.columns if c.startswith(segment_col)]
    if starts:
        # prefer something that isn't "*_feat"
        starts_sorted = sorted(starts, key=lambda x: ("feat" in x, x))
        return starts_sorted[0]
    raise KeyError(f"Could not find segment column after merge. Looked for {cand} and prefix '{segment_col}*'. Columns={list(df.columns)[:50]}...")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabeled_csv", type=Path, default=Path("data/processed/customers_master_unlabeled.csv"))
    ap.add_argument("--feat_path", type=Path, default=Path("data/processed/features/customer_behavior_v2.parquet"))
    ap.add_argument("--out_path", type=Path, default=Path("data/processed/rn/reliable_negatives_v1.csv"))
    ap.add_argument("--segment_col", type=str, default="customer_type")
    ap.add_argument("--min_group", type=int, default=200)
    ap.add_argument("--keep_frac", type=float, default=0.80, help="fraction of MOST NORMAL unlabeled to keep as RN")
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    u = pd.read_csv(args.unlabeled_csv, usecols=[ID, args.segment_col], low_memory=False)
    u[args.segment_col] = u[args.segment_col].fillna("Unknown").astype(str)

    feat = pd.read_parquet(args.feat_path)

    # IMPORTANT: use suffixes so we can reliably choose segment column
    df = u.merge(feat, on=ID, how="inner", suffixes=("_u", "_feat"))

    if len(df) == 0:
        raise SystemExit("[ERR] No overlap between unlabeled_csv and feat_path on customer_id")

    segcol = pick_segment_col(df, args.segment_col)
    df[segcol] = df[segcol].fillna("Unknown").astype(str)

    rn_rows = []
    segments = sorted(df[segcol].unique().tolist())
    print(f"[INFO] segment_col={args.segment_col} resolved_col={segcol}")
    print(f"[INFO] segments={segments} unlabeled_with_features={len(df)} keep_frac={args.keep_frac}")

    for seg in segments:
        g = df[df[segcol] == seg].copy()
        if len(g) < args.min_group:
            print(f"[WARN] segment={seg} rows={len(g)} < min_group={args.min_group}; skipping")
            continue

        # Drop segment col so it doesn't leak into IF
        X = build_X(g.drop(columns=[segcol], errors="ignore"))

        iso = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination="auto",
            random_state=args.seed,
            n_jobs=-1,
        )
        iso.fit(X)
        normal_score = iso.score_samples(X)  # higher = more normal
        g["_normal_score"] = normal_score

        k = int(np.ceil(args.keep_frac * len(g)))
        g = g.sort_values("_normal_score", ascending=False).head(k)

        out = g[[ID, segcol, "_normal_score"]].copy()
        out.rename(columns={segcol: args.segment_col, "_normal_score": "rn_normal_score"}, inplace=True)
        rn_rows.append(out)

        print(f"[INFO] segment={seg} rows={len(X)} RN_selected={len(out)}")

    if not rn_rows:
        raise SystemExit("[ERR] No segments produced reliable negatives. Lower --min_group or check segment_col.")

    rn = pd.concat(rn_rows, ignore_index=True)
    rn["rn_flag"] = 1
    rn.to_csv(args.out_path, index=False)
    print("[OK] wrote", args.out_path, "rows=", len(rn))

if __name__ == "__main__":
    main()
