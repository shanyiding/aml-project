from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.embeddings.seq.train_byol_seq import ShardSeqDataset, SeqEncoder, BYOL

IDCOL = "customer_id"

def load_model(run_dir: Path, cache_dir: Path, device):
    cfg = json.loads((cache_dir / "seq_config.json").read_text())
    mcc_vocab = int(cfg["mcc_buckets"]) + 1
    amt_vocab = int(cfg["amt_bins"]) + 1

    online = SeqEncoder(mcc_vocab=mcc_vocab, amt_vocab=amt_vocab)
    target = SeqEncoder(mcc_vocab=mcc_vocab, amt_vocab=amt_vocab)
    byol = BYOL(online=online, target=target).to(device)

    ckpt = run_dir / "byol_final.pt"
    byol.load_state_dict(torch.load(ckpt, map_location=device))
    byol.eval()
    return byol.online  # use online encoder for embeddings

def shard_iter(cache_dir: Path):
    shards = sorted(cache_dir.glob("shard_*.npz"))
    for sp in shards:
        z = np.load(sp, allow_pickle=True)
        yield sp, z["customer_id"], z["offsets"], z["events"]

def encode_customers(model, cache_dir: Path, want_ids: set[str], max_len: int, device):
    emb = {}
    for sp, cids, offsets, events in tqdm(list(shard_iter(cache_dir)), desc="Encoding shards"):
        for i, cid in enumerate(cids):
            cid = str(cid)
            if cid not in want_ids:
                continue
            start, end = int(offsets[i]), int(offsets[i+1])
            seq = events[start:end]
            if len(seq) >= max_len:
                seq = seq[-max_len:]
            else:
                pad = np.zeros((max_len - len(seq), 6), dtype=np.int32)
                seq = np.vstack([pad, seq])
            x = torch.from_numpy(seq).long().unsqueeze(0).to(device)  # [1,L,6]
            with torch.no_grad():
                z = model(x).squeeze(0).cpu().numpy()  # [128]
            emb[cid] = z
    return emb

def save_emb(emb: dict[str, np.ndarray], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(emb.keys())
    Z = np.vstack([emb[i] for i in ids])
    cols = [f"seq_emb_{i:03d}" for i in range(Z.shape[1])]
    df = pd.DataFrame(Z, columns=cols)
    df.insert(0, IDCOL, ids)
    df.to_parquet(out_path, index=False)
    print("[OK] wrote", out_path, "rows=", len(df))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--cache_dir", type=str, default="data/interim/seq_cache")
    ap.add_argument("--run_name", type=str, default="byol_seq_v1")
    ap.add_argument("--max_len", type=int, default=200)
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    cache_dir = Path(args.cache_dir)
    run_dir = Path("outputs/runs") / args.run_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run_dir, cache_dir, device)

    # customer IDs we need embeddings for
    def ids_from_csv(p: Path) -> set[str]:
        return set(pd.read_csv(p, usecols=[IDCOL])[IDCOL].astype(str).tolist())

    train_ids = ids_from_csv(processed / "customers_train.csv")
    val_ids   = ids_from_csv(processed / "customers_val.csv")
    test_ids  = ids_from_csv(processed / "customers_test.csv")

    emb_train = encode_customers(model, cache_dir, train_ids, args.max_len, device)
    emb_val   = encode_customers(model, cache_dir, val_ids, args.max_len, device)
    emb_test  = encode_customers(model, cache_dir, test_ids, args.max_len, device)

    save_emb(emb_train, processed / "static" / "seq_embeddings_train.parquet")
    save_emb(emb_val,   processed / "static" / "seq_embeddings_val.parquet")
    save_emb(emb_test,  processed / "static" / "seq_embeddings_test.parquet")

if __name__ == "__main__":
    main()