from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_cfg(cache_dir: Path):
    return json.loads((cache_dir / "seq_config.json").read_text())

def shard_paths(cache_dir: Path):
    return sorted(cache_dir.glob("shard_*.npz"))

class BoWDataset(Dataset):
    """
    Builds a fixed-size bag-of-events vector per customer from the cached events.
    This is CPU-friendly and trains fast with BYOL.
    """
    def __init__(self, cache_dir: str | Path, mcc_buckets: int, amt_bins: int, augment: bool = True):
        self.cache_dir = Path(cache_dir)
        self.mcc_buckets = mcc_buckets
        self.amt_bins = amt_bins
        self.augment = augment
        self.shards = shard_paths(self.cache_dir)
        if not self.shards:
            raise FileNotFoundError("No shard_*.npz found. Run build_seq_cache first.")

        # index: (shard_idx, row_idx)
        self.meta = []
        self.index = []
        for si, sp in enumerate(self.shards):
            z = np.load(sp, allow_pickle=True)
            n = len(z["customer_id"])
            self.meta.append((sp, n))
            self.index.extend([(si, i) for i in range(n)])

        # feature dims
        # chan: 0..8 (PAD + known + unknown) -> we'll use 9 bins (0 ignored mostly)
        self.chan_dim = 9
        self.hour_dim = 25   # 0 pad + 24 hours
        self.amt_dim  = self.amt_bins + 1
        self.mcc_dim  = self.mcc_buckets + 1

        self.dim = self.chan_dim + self.hour_dim + self.amt_dim + self.mcc_dim + 3  # + cash(2) + dc(3) simplified

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, ri = self.index[idx]
        sp, _ = self.meta[si]
        z = np.load(sp, allow_pickle=True)
        offsets = z["offsets"]
        events = z["events"]  # [N,6]
        start, end = int(offsets[ri]), int(offsets[ri+1])
        ev = events[start:end]  # [L,6]

        v = self._featurize(ev)
        if self.augment:
            v1 = self._augment(v.copy())
            v2 = self._augment(v.copy())
            return torch.from_numpy(v1).float(), torch.from_numpy(v2).float()
        else:
            return torch.from_numpy(v).float()

    def _featurize(self, ev: np.ndarray) -> np.ndarray:
        # ev cols: chan, mcc, hour, amtbin, cash, dc
        v = np.zeros(self.dim, dtype=np.float32)
        if ev.size == 0:
            return v

        chan = ev[:, 0].astype(np.int32)
        mcc  = ev[:, 1].astype(np.int32)
        hour = ev[:, 2].astype(np.int32)
        amt  = ev[:, 3].astype(np.int32)
        cash = ev[:, 4].astype(np.int32)
        dc   = ev[:, 5].astype(np.int32)

        # offsets for concatenation
        o_chan = 0
        o_hour = o_chan + self.chan_dim
        o_amt  = o_hour + self.hour_dim
        o_mcc  = o_amt  + self.amt_dim
        o_misc = o_mcc  + self.mcc_dim

        # hist counts (clip indices)
        chan = np.clip(chan, 0, self.chan_dim-1)
        hour = np.clip(hour, 0, self.hour_dim-1)
        amt  = np.clip(amt,  0, self.amt_dim-1)
        mcc  = np.clip(mcc,  0, self.mcc_dim-1)

        for x in chan: v[o_chan + x] += 1
        for x in hour: v[o_hour + x] += 1
        for x in amt:  v[o_amt + x]  += 1
        for x in mcc:  v[o_mcc + x]  += 1

        # misc: cash ratio + debit ratio + credit ratio (simple but useful)
        L = len(ev)
        v[o_misc + 0] = float((cash == 1).sum()) / max(1, L)
        v[o_misc + 1] = float((dc == 1).sum()) / max(1, L)
        v[o_misc + 2] = float((dc == 2).sum()) / max(1, L)

        # normalize hist parts by L (so customers comparable)
        v[o_chan:o_misc] /= max(1, L)
        return v

    def _augment(self, v: np.ndarray) -> np.ndarray:
        # BYOL needs two different views of same customer
        # Augmentations: dropout bins + noise on misc
        drop_p = 0.20
        mask = (np.random.rand(v.shape[0]) < drop_p)
        v[mask] = 0.0
        # tiny gaussian noise
        v += np.random.normal(0, 0.01, size=v.shape).astype(np.float32)
        v = np.clip(v, -5, 5)
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
    ap.add_argument("--run_name", type=str, default="byol_bow_v1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cfg = load_cfg(cache_dir)
    mcc_buckets = int(cfg["mcc_buckets"])
    amt_bins = int(cfg["amt_bins"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BoWDataset(cache_dir, mcc_buckets=mcc_buckets, amt_bins=amt_bins, augment=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    online = MLPEncoder(ds.dim, emb_dim=128)
    target = MLPEncoder(ds.dim, emb_dim=128)
    target.load_state_dict(online.state_dict())
    model = BYOL(online, target).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"seq_cfg": cfg, "args": vars(args), "bow_dim": ds.dim}, indent=2))

    global_step = 0
    steps_total = args.epochs * len(dl)

    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{args.epochs}")
        for x1, x2 in pbar:
            x1 = x1.to(device)
            x2 = x2.to(device)

            loss = model(x1, x2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            m = 1 - (1 - 0.99) * (math.cos(math.pi * global_step / max(1, steps_total)) + 1) / 2
            model.update_target(m)
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), m=float(m))

        torch.save(model.state_dict(), out_dir / f"byol_epoch{ep+1}.pt")

    torch.save(model.state_dict(), out_dir / "byol_final.pt")
    print("[OK] saved", out_dir)

if __name__ == "__main__":
    main()