from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

IDCOL = "customer_id"

def load_cfg(cache_dir: Path):
    return json.loads((cache_dir / "seq_config.json").read_text())

def featurize_events(ev: np.ndarray, mcc_buckets: int, amt_bins: int) -> np.ndarray:
    chan_dim = 9
    hour_dim = 24  # 0-23
    amt_dim  = amt_bins
    mcc_dim  = mcc_buckets
    misc_dim = 3

    # NEW interaction blocks
    chan_hour_dim = chan_dim * hour_dim          # 9*24 = 216
    chan_amt_dim  = chan_dim * amt_dim           # 9*amt_bins (e.g., 9*64=576)

    # total dim
    dim = (
        chan_dim +
        hour_dim +
        amt_dim +
        mcc_dim +
        chan_hour_dim +
        chan_amt_dim +
        misc_dim
    )

    v = np.zeros(dim, dtype=np.float32)
    if ev.size == 0:
        return v

    # columns: chan, mcc, hour_bucket, amt_bin, cash, dc
    chan = np.clip(ev[:,0].astype(np.int32), 0, chan_dim-1)
    mcc  = np.clip(ev[:,1].astype(np.int32), 0, mcc_dim-1)
    hour = np.clip(ev[:,2].astype(np.int32), 0, hour_dim-1)
    amt  = np.clip(ev[:,3].astype(np.int32), 0, amt_dim-1)
    cash = ev[:,4].astype(np.int32)
    dc   = ev[:,5].astype(np.int32)

    o_chan = 0
    o_hour = o_chan + chan_dim
    o_amt  = o_hour + hour_dim
    o_mcc  = o_amt  + amt_dim
    o_chh  = o_mcc  + mcc_dim
    o_cha  = o_chh  + chan_hour_dim
    o_misc = o_cha  + chan_amt_dim

    # marginal hists
    np.add.at(v, o_chan + chan, 1)
    np.add.at(v, o_hour + hour, 1)
    np.add.at(v, o_amt  + amt,  1)
    np.add.at(v, o_mcc  + mcc,  1)

    # interaction: channel x hour
    idx_chh = chan * hour_dim + hour
    np.add.at(v, o_chh + idx_chh, 1)

    # interaction: channel x amount_bin
    idx_cha = chan * amt_dim + amt
    np.add.at(v, o_cha + idx_cha, 1)

    L = len(ev)
    v[o_misc + 0] = float((cash == 1).sum()) / max(1, L)
    v[o_misc + 1] = float((dc == 1).sum()) / max(1, L)
    v[o_misc + 2] = float((dc == 2).sum()) / max(1, L)

    # normalize counts
    v[:o_misc] /= max(1, L)
    return v

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)

class BYOL(nn.Module):
    def __init__(self, online: nn.Module, target: nn.Module, emb_dim: int = 128):
        super().__init__()
        self.online = online
        self.target = target
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

def get_ids(path: Path) -> set[str]:
    return set(pd.read_csv(path, usecols=[IDCOL])[IDCOL].astype(str).tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--cache_dir", type=str, default="data/interim/seq_cache")
    ap.add_argument("--run_name", type=str, default="byol_bow_stream_v1")
    ap.add_argument("--out_prefix", type=str, default="bow_stream_emb")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    cache_dir = Path(args.cache_dir)
    run_dir = Path("outputs/runs") / args.run_name

    cfg = load_cfg(cache_dir)
    mcc_buckets = int(cfg["mcc_buckets"])
    amt_bins = int(cfg["amt_bins"])
    dim = 9 + 24 + amt_bins + mcc_buckets + (9*24) + (9*amt_bins) + 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model weights
    online = MLPEncoder(dim, 128)
    target = MLPEncoder(dim, 128)
    model = BYOL(online, target).to(device)
    state = torch.load(run_dir / "byol_final.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    train_ids = get_ids(processed / "customers_train.csv")
    val_ids   = get_ids(processed / "customers_val.csv")
    test_ids  = get_ids(processed / "customers_test.csv")

    def encode(want: set[str]) -> dict[str, np.ndarray]:
        out = {}
        shards = sorted(cache_dir.glob("shard_*.npz"))
        for sp in tqdm(shards, desc="Encoding shards"):
            z = np.load(sp, allow_pickle=True)
            cids = z["customer_id"]
            offsets = z["offsets"]
            events = z["events"]
            for i, cid in enumerate(cids):
                cid = str(cid)
                if cid not in want:
                    continue
                start, end = int(offsets[i]), int(offsets[i+1])
                ev = events[start:end]
                v = featurize_events(ev, mcc_buckets=mcc_buckets, amt_bins=amt_bins)
                x = torch.from_numpy(v).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.online(x).squeeze(0).cpu().numpy()
                out[cid] = emb
        return out

    def save(out: dict[str, np.ndarray], split: str):
        ids = list(out.keys())
        Z = np.vstack([out[i] for i in ids]) if ids else np.zeros((0,128), dtype=np.float32)
        cols = [f"{args.out_prefix}_{i:03d}" for i in range(Z.shape[1])]
        df = pd.DataFrame(Z, columns=cols)
        df.insert(0, IDCOL, ids)
        out_path = processed / "static" / f"{args.out_prefix}_{split}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print("[OK] wrote", out_path, "rows=", len(df))

    save(encode(train_ids), "train")
    save(encode(val_ids), "val")
    save(encode(test_ids), "test")

if __name__ == "__main__":
    main()