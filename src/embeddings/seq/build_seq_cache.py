from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import hashlib

def stable_hash_to_bucket(s: str, n_buckets: int) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % n_buckets

def amount_bin(x: float, n_bins: int = 32) -> int:
    # log-bins; robust for AML amounts
    x = max(0.0, float(x))
    v = np.log1p(x)
    # map to [0, n_bins-1] with a soft cap
    # cap at log1p(1e6) ~ 13.8 (tune later)
    cap = np.log1p(1_000_000.0)
    r = min(v / cap, 0.999999)
    return int(r * n_bins)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx_path", type=str, default="data/processed/transactions_unlabeled.csv")
    ap.add_argument("--out_dir", type=str, default="data/interim/seq_cache")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--chunksize", type=int, default=300_000)
    ap.add_argument("--mcc_buckets", type=int, default=2048)
    ap.add_argument("--amt_bins", type=int, default=32)
    args = ap.parse_args()

    tx_path = Path(args.tx_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # channel vocab (known)
    channel_vocab = {"abm": 1, "card": 2, "cheque": 3, "eft": 4, "emt": 5, "westernunion": 6, "wire": 7}
    # 0 reserved for PAD
    PAD = 0

    # customer_id -> list of encoded ints (we keep most recent by datetime, but approximate by processing order if sorted)
    seqs: dict[str, list[list[int]]] = {}  # we store list of events; each event is [chan, mcc, hour, amtbin, cash, dc]
    seen = set()

    usecols = ["customer_id", "transaction_datetime", "amount_cad", "channel", "merchant_category", "cash_indicator", "debit_credit"]
    total_rows = sum(1 for _ in open(tx_path, "rb"))  # crude; not exact lines but ok for tqdm? skip

    reader = pd.read_csv(tx_path, chunksize=args.chunksize, low_memory=False, usecols=lambda c: c in usecols)
    for chunk in tqdm(reader, desc="Chunking txns"):
        # minimal cleaning
        chunk["customer_id"] = chunk["customer_id"].astype(str)
        chunk["transaction_datetime"] = pd.to_datetime(chunk["transaction_datetime"], errors="coerce")
        chunk = chunk.dropna(subset=["customer_id", "transaction_datetime"])

        # sort within chunk by time so "most recent" appended last
        chunk = chunk.sort_values("transaction_datetime")

        # normalize fields
        chunk["amount_cad"] = pd.to_numeric(chunk.get("amount_cad", 0.0), errors="coerce").fillna(0.0)
        chunk["channel"] = chunk.get("channel", "unknown").astype(str).str.lower().fillna("unknown")
        chunk["merchant_category"] = chunk.get("merchant_category", "unknown").astype(str).fillna("unknown")
        chunk["cash_indicator"] = pd.to_numeric(chunk.get("cash_indicator", 0), errors="coerce").fillna(0).astype(int)
        dc = chunk.get("debit_credit", "U").astype(str).str.upper().fillna("U")

        # encode per row
        for i, row in chunk.iterrows():
            cid = row["customer_id"]
            chan = channel_vocab.get(str(row["channel"]).lower(), 8)  # 8=unknown
            mcc = stable_hash_to_bucket(str(row["merchant_category"]), args.mcc_buckets) + 1  # +1 so 0 is PAD
            hour = int(pd.Timestamp(row["transaction_datetime"]).hour) + 1  # 1..24
            ab = amount_bin(float(row["amount_cad"]), args.amt_bins) + 1    # 1..amt_bins
            cash = 1 if int(row["cash_indicator"]) == 1 else 0
            dci = 1 if dc.loc[i] == "D" else (2 if dc.loc[i] == "C" else 0)  # 0 unknown, 1 debit, 2 credit

            ev = [chan, mcc, hour, ab, cash, dci]
            seqs.setdefault(cid, []).append(ev)

            # truncate to most recent max_len (keep tail)
            if len(seqs[cid]) > args.max_len:
                seqs[cid] = seqs[cid][-args.max_len:]

            seen.add(cid)

        # periodically flush shards to disk to cap memory
        if len(seen) >= 50_000:
            shard_path = out_dir / f"shard_{len(list(out_dir.glob('shard_*.npz'))):05d}.npz"
            save_shard(seqs, shard_path)
            seqs.clear()
            seen.clear()

    # final flush
    if seqs:
        shard_path = out_dir / f"shard_{len(list(out_dir.glob('shard_*.npz'))):05d}.npz"
        save_shard(seqs, shard_path)

    # save vocab config
    cfg = {
        "max_len": args.max_len,
        "mcc_buckets": args.mcc_buckets,
        "amt_bins": args.amt_bins,
        "channel_vocab": channel_vocab,
        "pad_id": 0,
        "unknown_channel_id": 8,
        "hour_vocab_size": 25,  # 0 pad + 24 hours
        "dc_vocab_size": 3,     # 0 unk + D + C
    }
    (out_dir / "seq_config.json").write_text(__import__("json").dumps(cfg, indent=2))
    print("[OK] wrote", out_dir / "seq_config.json")

def save_shard(seqs: dict[str, list[list[int]]], shard_path: Path) -> None:
    # pack ragged sequences into flat arrays + offsets (standard ML-engineer cache format)
    cids = np.array(list(seqs.keys()), dtype=object)
    lens = np.array([len(seqs[c]) for c in cids], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    flat = np.zeros((int(offsets[-1]), 6), dtype=np.int32)
    idx = 0
    for cid in cids:
        arr = np.array(seqs[str(cid)], dtype=np.int32)
        flat[idx: idx + len(arr)] = arr
        idx += len(arr)
    np.savez_compressed(shard_path, customer_id=cids, offsets=offsets, events=flat)
    print("[OK] wrote shard", shard_path, "customers:", len(cids), "events:", len(flat))

if __name__ == "__main__":
    main()