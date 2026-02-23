from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

def augment(v: np.ndarray, drop_p: float = 0.20) -> np.ndarray:
    m = (np.random.rand(v.shape[0]) < drop_p)
    out = v.copy()
    out[m] = 0.0
    out += np.random.normal(0, 0.01, size=out.shape).astype(np.float32)
    return out

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
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    @torch.no_grad()
    def update_target(self, m: float):
        for po, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.data = pt.data * m + po.data * (1.0 - m)

    def forward(self, x1, x2):
        z1 = self.online(x1)
        z2 = self.online(x2)
        p1 = F.normalize(self.predictor(z1), dim=-1)
        p2 = F.normalize(self.predictor(z2), dim=-1)
        with torch.no_grad():
            t1 = self.target(x1)
            t2 = self.target(x2)
        loss = 2 - 2 * (p1 * t2).sum(dim=-1).mean()
        loss += 2 - 2 * (p2 * t1).sum(dim=-1).mean()
        return loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="data/interim/seq_cache")
    ap.add_argument("--run_name", type=str, default="byol_bow_stream_smoke")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_customers", type=int, default=60000)  # full ~60k
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cfg = load_cfg(cache_dir)
    mcc_buckets = int(cfg["mcc_buckets"])
    amt_bins = int(cfg["amt_bins"])

    # derive dim
    dim = 9 + 24 + amt_bins + mcc_buckets + (9*24) + (9*amt_bins) + 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online = MLPEncoder(dim, 128)
    target = MLPEncoder(dim, 128)
    target.load_state_dict(online.state_dict())
    model = BYOL(online, target).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"seq_cfg": cfg, "args": vars(args), "bow_dim": dim}, indent=2))

    shards = sorted(cache_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError("No shard_*.npz found")

    # count total customers available
    shard_sizes = []
    for sp in shards:
        z = np.load(sp, allow_pickle=True)
        shard_sizes.append(len(z["customer_id"]))
    total_customers = int(sum(shard_sizes))
    max_customers = min(args.max_customers, total_customers)

    steps_total = args.epochs * max(1, max_customers // args.batch_size)
    global_step = 0

    for ep in range(args.epochs):
        model.train()
        seen = 0
        pbar = tqdm(total=max_customers, desc=f"epoch {ep+1}/{args.epochs}")
        for sp in shards:
            if seen >= max_customers:
                break
            z = np.load(sp, allow_pickle=True)
            offsets = z["offsets"]
            events = z["events"]
            n = len(z["customer_id"])

            # stream customers in this shard
            for i in range(n):
                if seen >= max_customers:
                    break
                # build one sample vector
                start, end = int(offsets[i]), int(offsets[i+1])
                ev = events[start:end]
                v = featurize_events(ev, mcc_buckets=mcc_buckets, amt_bins=amt_bins)

                # accumulate into batch
                if (seen % args.batch_size) == 0:
                    batch1 = []
                    batch2 = []
                batch1.append(augment(v))
                batch2.append(augment(v))
                seen += 1
                pbar.update(1)

                if (seen % args.batch_size) == 0:
                    x1 = torch.from_numpy(np.stack(batch1)).float().to(device)
                    x2 = torch.from_numpy(np.stack(batch2)).float().to(device)

                    loss = model(x1, x2)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    m = 1 - (1 - 0.99) * (math.cos(math.pi * global_step / max(1, steps_total)) + 1) / 2
                    model.update_target(m)
                    global_step += 1
                    pbar.set_postfix(loss=float(loss.item()), m=float(m))

        pbar.close()
        torch.save(model.state_dict(), out_dir / f"byol_epoch{ep+1}.pt")

    torch.save(model.state_dict(), out_dir / "byol_final.pt")
    print("[OK] saved", out_dir)

if __name__ == "__main__":
    main()