import itertools
import pandas as pd
import numpy as np
from pathlib import Path

ID="customer_id"
TARGET="label"

models = [
    "outputs/runs/xgb_behavior_v2",
    "outputs/runs/xgb_fused_pw96",
    "outputs/runs/xgb_fused_segtype_v1"
]

def load_ranked(path):
    df = pd.read_csv(path)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(len(df))
    df["rscore"] = 1.0 - df["rank"]/(len(df)-1)
    return df[[ID, TARGET, "rscore"]]

val_dfs = [load_ranked(Path(m)/"val_ranked_customers.csv") for m in models]
test_dfs = [load_ranked(Path(m)/"test_ranked_customers.csv") for m in models]

best = None

for w1 in np.linspace(0.1,0.8,8):
    for w2 in np.linspace(0.1,0.8,8):
        w3 = 1.0 - w1 - w2
        if w3 <= 0:
            continue
        weights = np.array([w1,w2,w3])
        weights /= weights.sum()

        # merge val
        val = val_dfs[0].copy()
        val = val.rename(columns={"rscore":"r0"})
        for i,d in enumerate(val_dfs[1:],1):
            val = val.merge(d.rename(columns={"rscore":f"r{i}"}), on=[ID,TARGET])
        val["score"] = sum(val[f"r{i}"]*weights[i] for i in range(3))
        val = val.sort_values("score", ascending=False).reset_index(drop=True)
        pos_ranks = val.index[val[TARGET]==1].tolist()

        # objective: minimize max positive rank
        worst_rank = max(pos_ranks)

        if best is None or worst_rank < best[0]:
            best = (worst_rank, weights, pos_ranks)

print("Best val worst-rank:", best)
