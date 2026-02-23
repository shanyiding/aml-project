from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------- Dataset over NPZ shards --------
class ShardSeqDataset(Dataset):
    def __init__(self, cache_dir: str | Path, max_len: int, augment: bool = True):
        self.cache_dir = Path(cache_dir)
        self.max_len = max_len
        self.augment = augment
        self.shards = sorted(self.cache_dir.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError("No shard_*.npz found. Run build_seq_cache first.")
        # Build index: (shard_idx, row_idx)
        self.index = []
        self._meta = []
        for si, sp in enumerate(self.shards):
            z = np.load(sp, allow_pickle=True)
            n = len(z["customer_id"])
            self._meta.append((sp, n))
            self.index.extend([(si, i) for i in range(n)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, ri = self.index[idx]
        sp, _ = self._meta[si]
        z = np.load(sp, allow_pickle=True)
        offsets = z["offsets"]
        events = z["events"]  # [Nevents, 6]
        start, end = int(offsets[ri]), int(offsets[ri + 1])
        seq = events[start:end]  # [L,6]
        # pad/truncate to max_len
        if len(seq) >= self.max_len:
            seq = seq[-self.max_len:]
        else:
            pad = np.zeros((self.max_len - len(seq), 6), dtype=np.int32)
            seq = np.vstack([pad, seq])
        x = torch.from_numpy(seq).long()  # [max_len,6]

        if self.augment:
            x1 = self._augment(x.clone())
            x2 = self._augment(x.clone())
            return x1, x2
        else:
            return x

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: [L,6], 0 is PAD for each field except cash/dc (we keep as is)
        L = x.shape[0]
        # random drop transactions: set whole row to PAD
        drop_p = 0.25
        mask = (torch.rand(L) < drop_p) & (x.sum(dim=1) > 0)
        x[mask] = 0

        # random mask some fields (mcc, amtbin) to 0
        field_mask_p = 0.15
        for col in [1, 3]:  # mcc, amount_bin
            m = (torch.rand(L) < field_mask_p) & (x[:, col] > 0)
            x[m, col] = 0

        # hour jitter +-1 (only if non-pad)
        jitter_p = 0.2
        m = (torch.rand(L) < jitter_p) & (x[:, 2] > 0)
        jitter = torch.randint(low=-1, high=2, size=(L,))
        x[m, 2] = torch.clamp(x[m, 2] + jitter[m], 1, 24)

        return x

# -------- Model: token embeddings + Transformer encoder + pooling --------
class SeqEncoder(nn.Module):
    def __init__(self, mcc_vocab: int, amt_vocab: int, chan_vocab: int = 16, hour_vocab: int = 25, dc_vocab: int = 3,
                 d_model: int = 128, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.chan_emb = nn.Embedding(chan_vocab, d_model)
        self.mcc_emb  = nn.Embedding(mcc_vocab, d_model)
        self.hour_emb = nn.Embedding(hour_vocab, d_model)
        self.amt_emb  = nn.Embedding(amt_vocab, d_model)
        self.cash_emb = nn.Embedding(2, d_model)
        self.dc_emb   = nn.Embedding(dc_vocab, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,6] -> (chan,mcc,hour,amt,cash,dc)
        e = (
            self.chan_emb(x[...,0]) +
            self.mcc_emb(x[...,1]) +
            self.hour_emb(x[...,2]) +
            self.amt_emb(x[...,3]) +
            self.cash_emb(torch.clamp(x[...,4], 0, 1)) +
            self.dc_emb(torch.clamp(x[...,5], 0, 2))
        )
        h = self.enc(e)  # [B,L,D]
        # mask pads (rows all zero => sum==0)
        pad = (x.sum(dim=-1) == 0)  # [B,L]
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        # mean pool over non-pad
        denom = (~pad).sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = h.sum(dim=1) / denom  # [B,D]
        pooled = self.norm(pooled)
        z = self.proj(pooled)  # [B,128]
        z = F.normalize(z, dim=-1)
        return z

# -------- BYOL bits --------
class BYOL(nn.Module):
    def __init__(self, online: nn.Module, target: nn.Module, proj_dim: int = 128):
        super().__init__()
        self.online = online
        self.target = target
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
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
        # BYOL loss: negative cosine similarity
        loss = 2 - 2 * (p1 * t2).sum(dim=-1).mean() / 1.0
        loss += 2 - 2 * (p2 * t1).sum(dim=-1).mean() / 1.0
        return loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="data/interim/seq_cache")
    ap.add_argument("--run_name", type=str, default="byol_seq_v1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=4)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cfg = json.loads((cache_dir / "seq_config.json").read_text())
    mcc_vocab = int(cfg["mcc_buckets"]) + 1  # +1 for PAD
    amt_vocab = int(cfg["amt_bins"]) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ShardSeqDataset(cache_dir, max_len=args.max_len, augment=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    online = SeqEncoder(mcc_vocab=mcc_vocab, amt_vocab=amt_vocab, d_model=args.d_model, nlayers=args.nlayers, nhead=args.nhead)
    target = SeqEncoder(mcc_vocab=mcc_vocab, amt_vocab=amt_vocab, d_model=args.d_model, nlayers=args.nlayers, nhead=args.nhead)
    target.load_state_dict(online.state_dict())

    byol = BYOL(online=online, target=target).to(device)
    opt = torch.optim.AdamW(byol.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"seq_cfg": cfg, "args": vars(args)}, indent=2))

    global_step = 0
    for ep in range(args.epochs):
        byol.train()
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{args.epochs}")
        for x1, x2 in pbar:
            x1 = x1.to(device)
            x2 = x2.to(device)

            loss = byol(x1, x2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(byol.parameters(), 1.0)
            opt.step()

            # EMA momentum schedule (common BYOL trick)
            m = 1 - (1 - 0.99) * (math.cos(math.pi * global_step / max(1, args.epochs * len(dl))) + 1) / 2
            byol.update_target(m)
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), m=float(m))

        # save checkpoint each epoch
        torch.save(byol.state_dict(), out_dir / f"byol_epoch{ep+1}.pt")

    torch.save(byol.state_dict(), out_dir / "byol_final.pt")
    print("[OK] saved", out_dir)

if __name__ == "__main__":
    main()