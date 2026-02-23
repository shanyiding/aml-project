from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

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

def extract_source_nums(source_field: str) -> list[int]:
    if not source_field:
        return []
    return [int(x) for x in re.findall(r"\d+", str(source_field))]

def resolve_links(source_nums: list[int], sources_df: pd.DataFrame) -> list[str]:
    links: list[str] = []
    if sources_df is None or not source_nums:
        return links
    for n in source_nums:
        rows = sources_df.loc[sources_df["Number"] == n]
        if len(rows) == 0:
            continue
        cell = str(rows.iloc[0].get("Corresponding Link(s)", ""))
        # Accept either '|' separated (preferred) or '<br>' from earlier paste
        cell = cell.replace("<br>", "|")
        for p in [p.strip() for p in cell.split("|")]:
            if p and p not in links:
                links.append(p)
    return links

def read_customers(processed_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(processed_dir / f"customers_{split}.csv", usecols=[ID, TARGET])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    df[ID] = df[ID].astype(str)
    return df

def read_behavior(behavior_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(behavior_path)
    if ID not in df.columns:
        raise SystemExit(f"[ERR] behavior missing {ID}: {behavior_path}")
    df[ID] = df[ID].astype(str)
    return df

def read_embeddings(processed_dir: Path, emb_prefix: str, split: str) -> pd.DataFrame | None:
    # Try a few common patterns
    cand = []
    cand += list((processed_dir / "static").glob(f"{emb_prefix}*{split}*.parquet"))
    if not cand:
        return None
    # Pick the biggest file by size (often the actual embedding matrix)
    cand = sorted(cand, key=lambda p: p.stat().st_size, reverse=True)
    p = cand[0]
    df = pd.read_parquet(p)
    if ID not in df.columns:
        # maybe index is customer_id
        if df.index.name == ID:
            df = df.reset_index()
        else:
            return None
    df[ID] = df[ID].astype(str)
    return df

def load_model(model_dir: Path):
    # Expect a joblib dump from your training scripts
    for name in ["model.joblib", "model.pkl", "xgb_model.joblib", "clf.joblib"]:
        p = model_dir / name
        if p.exists():
            return joblib.load(p)
    # Fallback: load any joblib in folder
    jbs = sorted(model_dir.glob("*.joblib"), key=lambda p: p.stat().st_size, reverse=True)
    if jbs:
        return joblib.load(jbs[0])
    raise SystemExit(f"[ERR] could not find a joblib model in {model_dir}")

def align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    # Try to infer feature names from common places
    feat = None
    if hasattr(model, "feature_names_in_"):
        feat = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            b = model.get_booster()
            feat = b.feature_names
        except Exception:
            feat = None

    if feat is None:
        # As last resort, keep current columns
        return X

    missing = [c for c in feat if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[feat]
    return X

def top_rules_from_features(top_features: list[str], signal_map: pd.DataFrame) -> list[str]:
    # naive mapping: if any "columns_used" token matches any top feature name
    if not top_features:
        return []
    tf = set(top_features)
    rules = []
    for _, row in signal_map.iterrows():
        cols_used = str(row.get("columns_used", ""))
        # tokens split by ';' and ','
        toks = [t.strip() for t in re.split(r"[;,]", cols_used) if t.strip()]
        if any(t in tf for t in toks):
            rid = str(row["rule_id"]).strip()
            if rid and rid not in rules:
                rules.append(rid)
    return rules

def humanize_feature(f: str) -> str:
    # small helpers for your current feature naming
    f = f.replace("_delta", " (change)")
    f = f.replace("_ratio", " ratio")
    f = f.replace("_sum", " total")
    f = f.replace("_mean", " average")
    return f

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_csv", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--rulebook_csv", default=str(_repo_root() / "knowledge_base" / "aml_rulebook.csv"))
    ap.add_argument("--signal_map_csv", default=str(_repo_root() / "knowledge_base" / "rule_signal_map.csv"))
    ap.add_argument("--sources_csv", default=str(_repo_root() / "knowledge_base" / "sources.csv"))
    ap.add_argument("--processed_dir", default=str(_repo_root() / "data" / "processed"))
    ap.add_argument("--behavior_path", default=str(_repo_root() / "data" / "processed" / "features" / "customer_behavior_v2.parquet"))
    ap.add_argument("--emb_prefix", default="bow_stream_emb_v4")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--out_run", required=True)
    ap.add_argument("--top_features", type=int, default=8)
    ap.add_argument("--shap_background", type=int, default=256)
    args = ap.parse_args()

    out_dir = _repo_root() / "outputs" / "runs" / args.out_run
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = pd.read_csv(args.ranked_csv)
    ranked[ID] = ranked[ID].astype(str)
    if "score" not in ranked.columns:
        # allow blended csv
        if "score_blended" in ranked.columns:
            ranked["score"] = ranked["score_blended"]
        else:
            raise SystemExit("[ERR] ranked_csv must have 'score' (or score_blended) column")

    top = ranked.head(args.top_n).copy()

    processed_dir = Path(args.processed_dir)
    behavior_path = Path(args.behavior_path)

    ids = read_customers(processed_dir, args.split)
    beh = read_behavior(behavior_path)

    emb = read_embeddings(processed_dir, args.emb_prefix, args.split)

    X_all = beh.merge(ids[[ID, TARGET]], on=ID, how="inner")
    if emb is not None:
        # avoid duplicate columns
        dup = set(X_all.columns) & set(emb.columns) - {ID}
        if dup:
            emb = emb.drop(columns=list(dup))
        X_all = X_all.merge(emb, on=ID, how="left")

    # Fill missing
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    for c in X_all.columns:
        if c in (ID, TARGET):
            continue
        if X_all[c].dtype == bool:
            X_all[c] = X_all[c].astype(int)
        if str(X_all[c].dtype) in ("object", "category"):
            X_all[c] = X_all[c].fillna("Unknown").astype(str)
    # one-hot
    obj_cols = [c for c in X_all.columns if str(X_all[c].dtype) in ("object", "category")]
    if obj_cols:
        X_all = pd.get_dummies(X_all, columns=obj_cols, dummy_na=False)
    # numeric
    for c in X_all.columns:
        if c in (ID, TARGET):
            continue
        if not str(X_all[c].dtype).startswith(("int", "float")):
            X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
    X_all = X_all.fillna(0.0)

    # subset to top customers
    id_col = X_all[ID].copy()
    X_sub = X_all[X_all[ID].isin(top[ID].tolist())].copy()
    X_sub = X_sub.sort_values(ID).reset_index(drop=True)

    model = load_model(Path(args.model_dir))

    # Align columns
    X_feat = X_sub.drop(columns=[ID, TARGET], errors="ignore")
    X_feat = align_to_model(X_feat, model)

    # Predict score from model if possible
    scores = None
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_feat)[:, 1]
        elif hasattr(model, "predict"):
            scores = model.predict(X_feat)
    except Exception:
        scores = None

    if scores is None:
        # fallback: use input ranked scores
        score_map = dict(zip(ranked[ID].astype(str), ranked["score"].astype(float)))
        scores = np.array([score_map.get(cid, np.nan) for cid in X_sub[ID].astype(str)])

    # SHAP
    try:
        import shap  # type: ignore
    except Exception as e:
        raise SystemExit(f"[ERR] shap not installed. run: pip install shap. details={e}")

    # Background sample
    bg_n = min(args.shap_background, len(X_feat))
    bg = X_feat.sample(n=bg_n, random_state=0) if bg_n > 0 else X_feat

    explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="tree_path_dependent")
    shap_vals = explainer.shap_values(X_feat)
    # shap may return list for multiclass; use class1
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[-1]
    shap_vals = np.asarray(shap_vals)

    rb = load_rulebook(Path(args.rulebook_csv))
    sm = load_signal_map(Path(args.signal_map_csv))
    src = load_sources(Path(args.sources_csv)) if Path(args.sources_csv).exists() else None

    rows = []
    for i in range(len(X_sub)):
        cid = str(X_sub.loc[i, ID])
        sc = float(scores[i]) if np.isfinite(scores[i]) else float(ranked.loc[ranked[ID] == cid, "score"].iloc[0])

        sv = shap_vals[i]
        # pick top positive contributors (risk up)
        order = np.argsort(-sv)  # descending shap
        top_idx = [j for j in order[: args.top_features]]
        top_feats = [X_feat.columns[j] for j in top_idx]
        top_vals = [float(sv[j]) for j in top_idx]

        # map to rules
        rules = top_rules_from_features(top_feats, sm)

        # attach rule detail + links
        top_rule_lines = []
        link_set = []
        for rid in rules[:5]:
            rrow = rb.loc[rb["rule_id"] == rid]
            if len(rrow) == 0:
                continue
            rrow = rrow.iloc[0]
            src_nums = extract_source_nums(str(rrow.get("source", "")))
            links = resolve_links(src_nums, src) if src is not None else []
            for L in links:
                if L not in link_set:
                    link_set.append(L)
            top_rule_lines.append(f"{rid}: {str(rrow.get('red_flag','')).strip()}")

        rows.append({
            "customer_id": cid,
            "score": sc,
            "top_rules": ";".join(rules[:8]),
            "top_rule_text": " | ".join(top_rule_lines[:5]),
            "top_shap_features": ";".join(top_feats),
            "top_shap_values": ";".join([f"{v:.6f}" for v in top_vals]),
            "top_shap_features_human": ";".join([humanize_feature(f) for f in top_feats]),
            "source_links": "|".join(link_set[:10]),
        })

    out_csv = out_dir / f"{args.split}_top{args.top_n}_shap_explanations.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # pos ranks (for sanity)
    pos_ranks = None
    if TARGET in X_sub.columns:
        # rank within the whole ranked list if label exists
        rank_df = ranked.copy().reset_index(drop=True)
        # if ranked has label col, use it
        if TARGET in rank_df.columns:
            pos_ranks = rank_df.index[rank_df[TARGET] == 1].tolist()

    (out_dir / f"{args.split}_meta.json").write_text(json.dumps({
        "ranked_csv": args.ranked_csv,
        "model_dir": args.model_dir,
        "top_n": args.top_n,
        "split": args.split,
        "pos_ranks": pos_ranks,
    }, indent=2))

    print("[OK] wrote", out_dir)
    print(f"[INFO] split={args.split} top_n={args.top_n}")

if __name__ == "__main__":
    main()

