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

weights = np.array([0.1, 0.1, 0.8], dtype=float)
weights /= weights.sum()

def load_ranked(path):
    df = pd.read_csv(path)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(len(df))
    df["rscore"] = 1.0 - df["rank"]/(len(df)-1)
    return df[[ID, TARGET, "rscore"]]

test_dfs = [load_ranked(Path(m)/"test_ranked_customers.csv") for m in models]

test = test_dfs[0].copy().rename(columns={"rscore":"r0"})
for i,d in enumerate(test_dfs[1:],1):
    test = test.merge(d.rename(columns={"rscore":f"r{i}"}), on=[ID,TARGET])

test["score"] = sum(test[f"r{i}"]*weights[i] for i in range(3))
test = test.sort_values("score", ascending=False).reset_index(drop=True)

pos_ranks = test.index[test[TARGET]==1].tolist()
print("weights:", weights)
print("test_pos_ranks:", pos_ranks)
print(test.loc[pos_ranks, [ID, TARGET, "score"]])
