from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ID = "customer_id"
TARGET = "label"

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_rulebook(p: Path) -> pd.DataFrame:
    rb = pd.read_csv(p, engine="python", on_bad_lines="skip")
    rb.columns = [str(c).strip() for c in rb.columns]
    need = {"rule_id", "category", "red_flag", "typology", "source", "feasibility"}
    missing = sorted(list(need - set(rb.columns)))
    if missing:
        raise SystemExit(f"[ERR] rulebook missing columns: {missing}")
    rb["rule_id"] = rb["rule_id"].astype(str).str.strip()
    rb = rb[rb["rule_id"].str.lower().ne("rule_id")]
    rb = rb.dropna(subset=["rule_id"])
    rb = rb[rb["rule_id"].astype(str).str.len() > 0]
    rb = rb.drop_duplicates(subset=["rule_id"]).reset_index(drop=True)
    return rb

def load_signal_map(p: Path) -> pd.DataFrame:
    sm = pd.read_csv(p, engine="python", on_bad_lines="skip")
    sm.columns = [str(c).strip() for c in sm.columns]
    sm = sm.loc[:, [c for c in sm.columns if not str(c).startswith("Unnamed")]]
    need = {"rule_id", "signal_id", "columns_used", "description", "feasibility"}
    missing = sorted(list(need - set(sm.columns)))
    if missing:
        raise SystemExit(f"[ERR] signal_map missing columns: {missing}")
    for c in ["rule_id", "signal_id"]:
        sm[c] = sm[c].astype(str).str.strip()
    sm = sm[sm["rule_id"].str.lower().ne("rule_id")]
    sm = sm[sm["rule_id"].ne("") & sm["rule_id"].ne("nan")]
    sm = sm[sm["signal_id"].ne("") & sm["signal_id"].ne("nan")]
    sm = sm.drop_duplicates(subset=["rule_id", "signal_id"]).reset_index(drop=True)
    return sm

def load_sources(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "Number" in df.columns:
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce").astype("Int64")
    if "Corresponding Link(s)" in df.columns:
        df["Corresponding Link(s)"] = df["Corresponding Link(s)"].fillna("").astype(str)
    return df

def extract_source_nums(source_field: str) -> list:
    if not source_field:
        return []
    return [int(x) for x in re.findall(r"\d+", str(source_field))]

def resolve_links(source_nums: list, sources_df: pd.DataFrame) -> list:
    links = []
    if sources_df is None or not source_nums:
        return links
    for n in source_nums:
        rows = sources_df.loc[sources_df["Number"] == n]
        if len(rows) == 0:
            continue
        cell = str(rows.iloc[0].get("Corresponding Link(s)", ""))
        for p in [p.strip() for p in cell.split("|")]:
            if p and p not in links:
                links.append(p)
    return links

def read_customers(processed: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(processed / f"customers_{split}.csv", usecols=[ID, TARGET])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return df

def read_behavior(feat_path: Path) -> pd.DataFrame:
    b = pd.read_parquet(feat_path)
    if TARGET in b.columns:
        b = b.drop(columns=[TARGET], errors="ignore")
    return b

def read_emb(processed: Path, emb_prefix: str, split: str) -> pd.DataFrame:
    p = processed / "static" / f"{emb_prefix}_{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {p}")
    return pd.read_parquet(p)

def make_X(customers, behavior, emb):
    df = customers.merge(behavior, on=ID, how="left").merge(emb, on=ID, how="left")
    y_ids = df[[ID, TARGET]].copy()
    X = df.drop(columns=[ID, TARGET], errors="ignore")
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str).str.strip()
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    X = X.fillna(0)
    return X, y_ids

def get_model_feature_names(model):
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        return list(model.feature_names_in_)
    try:
        bn = model.get_booster().feature_names
        return list(bn) if bn is not None else None
    except Exception:
        return None

def align_to_model(X, model):
    names = get_model_feature_names(model)
    if names is None:
        return X
    if len(names) > 0 and re.fullmatch(r"f\d+", str(names[0])) is not None:
        return X
    return X.reindex(columns=names, fill_value=0)

def build_signal_to_rules(sm):
    d = {}
    for _, r in sm.iterrows():
        sid = str(r["signal_id"]).strip()
        rid = str(r["rule_id"]).strip()
        if not sid or not rid:
            continue
        d.setdefault(sid, []).append(rid)
    for k, v in d.items():
        seen = set(); out = []
        for x in v:
            if x not in seen:
                out.append(x); seen.add(x)
        d[k] = out
    return d

def match_signal(feature_name, sm_sig):
    fn = feature_name.lower()
    hits = []
    for sid in sm_sig["signal_id"].tolist():
        if sid and sid.lower() in fn:
            hits.append(sid)
    if not hits:
        for _, row in sm_sig.iterrows():
            cols_used = str(row["columns_used"]).lower()
            sid = str(row["signal_id"]).strip()
            for token in re.split(r"[;, ]+", cols_used):
                token = token.strip()
                if token and token in fn:
                    hits.append(sid); break
    out = []; seen = set()
    for x in hits:
        if x not in seen:
            out.append(x); seen.add(x)
    return out[:3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_csv", type=Path, required=True)
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--rulebook_csv", type=Path, default=None)
    ap.add_argument("--signal_map_csv", type=Path, default=None)
    ap.add_argument("--sources_csv", type=Path, default=None)
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--behavior_path", type=Path, default=Path("data/processed/features/customer_behavior_v2.parquet"))
    ap.add_argument("--emb_prefix", type=str, default="bow_stream_emb_v4")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--out_run", type=str, default="EXPLAIN_top_v1")
    ap.add_argument("--top_features", type=int, default=12)
    args = ap.parse_args()

    repo = _repo_root()
    if args.rulebook_csv is None:
        args.rulebook_csv = repo / "knowledge_base" / "aml_rulebook.csv"
    if args.signal_map_csv is None:
        args.signal_map_csv = repo / "knowledge_base" / "rule_signal_map.csv"
    if args.sources_csv is None:
        args.sources_csv = repo / "knowledge_base" / "sources.csv"

    out_dir = repo / "outputs" / "runs" / args.out_run
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = pd.read_csv(args.ranked_csv)
    if ID not in ranked.columns:
        raise SystemExit(f"[ERR] ranked_csv missing {ID}")
    if "score" not in ranked.columns:
        raise SystemExit(f"[ERR] ranked_csv missing score")
    ranked[ID] = ranked[ID].astype(str)

    model = joblib.load(args.model_dir / "model.joblib")

    customers = read_customers(args.processed_dir, args.split)
    behavior = read_behavior(args.behavior_path)
    emb = read_emb(args.processed_dir, args.emb_prefix, args.split)

    X_all, ids = make_X(customers, behavior, emb)
    X_all.index = ids[ID].astype(str).values
    X_all = align_to_model(X_all, model)

    ids = ids.merge(ranked[[ID, "score"]], on=ID, how="left")
    ids["score"] = ids["score"].fillna(-1)

    top = ids.sort_values("score", ascending=False).head(args.top_n).copy()
    top_ids = top[ID].astype(str).tolist()

    rb = load_rulebook(Path(args.rulebook_csv))
    sm = load_signal_map(Path(args.signal_map_csv))
    sources_df = load_sources(Path(args.sources_csv)) if args.sources_csv.exists() else None
    signal_to_rules = build_signal_to_rules(sm)
    rb_idx = rb.set_index("rule_id", drop=False)

    sm_sig = sm.copy()
    sm_sig["signal_id"] = sm_sig["signal_id"].astype(str).str.strip()
    sm_sig["columns_used"] = sm_sig["columns_used"].astype(str)

    try:
        booster = model.get_booster()
        import xgboost as xgb
        dtop = xgb.DMatrix(X_all.loc[top_ids].to_numpy(), feature_names=list(X_all.columns))
        contrib = booster.predict(dtop, pred_contribs=True)
        contrib_df = pd.DataFrame(contrib[:, :-1], columns=list(X_all.columns), index=top_ids)
    except Exception:
        contrib_df = pd.DataFrame(0.0, index=top_ids, columns=list(X_all.columns))

    rows = []
    for cid in top_ids:
        c = contrib_df.loc[cid]
        top_feat = c.abs().sort_values(ascending=False).head(args.top_features).index.tolist()

        reasons = []
        for f in top_feat:
            val = float(X_all.loc[cid, f]) if f in X_all.columns else 0.0
            impact = float(c[f])
            sigs = match_signal(f, sm_sig)

            mapped_rules = []
            for sid in sigs:
                mapped_rules.extend(signal_to_rules.get(sid, []))
            mapped_rules = list(dict.fromkeys(mapped_rules))[:3]

            rule_summaries = []
            for rid in mapped_rules:
                if rid in rb_idx.index:
                    rr = rb_idx.loc[rid]
                    src_field = str(rr.get("source", "")).strip()
                    src_nums = extract_source_nums(src_field)
                    links = resolve_links(src_nums, sources_df)
                    rule_summaries.append({
                        "rule_id": rid,
                        "category": str(rr["category"]),
                        "red_flag": str(rr["red_flag"]),
                        "typology": str(rr["typology"]),
                        "source": src_field,
                        "feasibility": str(rr["feasibility"]),
                        "links": links,
                    })

            reasons.append({
                "feature": f,
                "value": val,
                "impact": impact,
                "signal_ids": sigs,
                "rules": rule_summaries,
            })

        score_val = float(top.loc[top[ID]==cid, "score"].iloc[0])
        out_json = {"customer_id": cid, "score": score_val, "reasons": reasons}
        (out_dir / f"{cid}.json").write_text(json.dumps(out_json, indent=2))

        top_rules_flat = list(dict.fromkeys([
            rr["rule_id"] for r in reasons for rr in r["rules"]
        ]))[:5]
        top_links_flat = list(dict.fromkeys([
            lnk for r in reasons for rr in r["rules"] for lnk in rr["links"]
        ]))[:5]

        rows.append({
            "customer_id": cid,
            "score": score_val,
            "top_rules": ";".join(top_rules_flat),
            "top_features": ";".join([r["feature"] for r in reasons[:8]]),
            "top_rule_links": " | ".join(top_links_flat),
        })

    out_csv = out_dir / f"{args.split}_top{args.top_n}_explanations.csv"
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(out_csv, index=False)

    pos_ranks = []
    if TARGET in ranked.columns:
        pos_ranks = ranked.index[pd.to_numeric(ranked[TARGET], errors="coerce").fillna(0).astype(int) == 1].tolist()
        (out_dir / f"{args.split}_pos_ranks.txt").write_text(str(pos_ranks))

    print("[OK] wrote", out_dir)
    print(f"[INFO] split={args.split} top_n={args.top_n} pos_ranks={pos_ranks}")

if __name__ == "__main__":
    main()
