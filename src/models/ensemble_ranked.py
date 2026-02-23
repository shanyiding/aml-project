from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ID="customer_id"
TARGET="label"

def load_ranked(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {ID, TARGET, "score"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: customer_id,label,score")
    # ensure sorted high->low (just in case)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(len(df), dtype=float)
    # convert to "rank score" where higher is better (1 is top)
    df["rscore"] = 1.0 - (df["rank"] / max(1.0, len(df)-1))
    return df[[ID, TARGET, "rscore"]]

def merge_ranks(dfs: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    out = dfs[0].copy()
    out = out.rename(columns={"rscore":"r0"})
    for i, d in enumerate(dfs[1:], start=1):
        out = out.merge(d.rename(columns={"rscore":f"r{i}"}), on=[ID, TARGET], how="inner")
    # weighted average
    rcols = [f"r{i}" for i in range(len(dfs))]
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    out["score"] = sum(out[c].to_numpy()*w[i] for i,c in enumerate(rcols))
    return out[[ID, TARGET, "score"]].sort_values("score", ascending=False).reset_index(drop=True)

def pos_ranks(df: pd.DataFrame):
    return df.index[df[TARGET]==1].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", type=str, default="ensemble_v1")
    ap.add_argument("--val_paths", nargs="+", required=True)
    ap.add_argument("--test_paths", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=False)
    args = ap.parse_args()

    if args.weights is None:
        weights = [1.0]*len(args.val_paths)
    else:
        weights = args.weights
    if len(weights) != len(args.val_paths):
        raise ValueError("weights length must match number of models")

    out_dir = Path("outputs/runs")/args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    val_dfs = [load_ranked(Path(p)) for p in args.val_paths]
    test_dfs = [load_ranked(Path(p)) for p in args.test_paths]

    val_ens = merge_ranks(val_dfs, weights)
    test_ens = merge_ranks(test_dfs, weights)

    val_ens.to_csv(out_dir/"val_ranked_customers.csv", index=False)
    test_ens.to_csv(out_dir/"test_ranked_customers.csv", index=False)

    print("[OK] wrote", out_dir)
    print("val pos ranks:", pos_ranks(val_ens))
    print("test pos rank:", pos_ranks(test_ens))

if __name__ == "__main__":
    main()
