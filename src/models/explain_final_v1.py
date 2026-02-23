from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ID = "customer_id"
TARGET = "label"

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

# ── KB loaders ────────────────────────────────────────────────────────────────

def load_rulebook(p: Path) -> pd.DataFrame:
    rb = pd.read_csv(p, engine="python", on_bad_lines="skip")
    rb.columns = [str(c).strip() for c in rb.columns]
    for col in ["rule_id","category","red_flag","typology","source","feasibility"]:
        if col not in rb.columns:
            rb[col] = ""
    rb["rule_id"] = rb["rule_id"].astype(str).str.strip()
    rb = rb[rb["rule_id"].str.lower().ne("rule_id")]
    rb = rb.dropna(subset=["rule_id"])
    rb = rb[rb["rule_id"].str.len() > 0]
    rb = rb.drop_duplicates(subset=["rule_id"]).reset_index(drop=True)
    return rb

def load_signal_map(p: Path) -> pd.DataFrame:
    sm = pd.read_csv(p, engine="python", on_bad_lines="skip")
    sm.columns = [str(c).strip() for c in sm.columns]
    sm = sm.loc[:, [c for c in sm.columns if not c.startswith("Unnamed")]]
    for col in ["rule_id","signal_id","columns_used","description","feasibility"]:
        if col not in sm.columns:
            sm[col] = ""
    for c in ["rule_id","signal_id"]:
        sm[c] = sm[c].astype(str).str.strip()
    sm = sm[sm["rule_id"].str.lower().ne("rule_id")]
    sm = sm[sm["rule_id"].ne("") & sm["rule_id"].ne("nan")]
    sm = sm[sm["signal_id"].ne("") & sm["signal_id"].ne("nan")]
    sm = sm.drop_duplicates(subset=["rule_id","signal_id"]).reset_index(drop=True)
    return sm

def load_sources(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "Number" in df.columns:
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce").astype("Int64")
    if "Corresponding Link(s)" in df.columns:
        df["Corresponding Link(s)"] = df["Corresponding Link(s)"].fillna("").astype(str)
    return df

# ── Source link resolution ────────────────────────────────────────────────────

def extract_source_nums(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", str(s))]

def resolve_links(nums: list[int], sources_df: pd.DataFrame | None) -> list[str]:
    if sources_df is None or not nums:
        return []
    links: list[str] = []
    for n in nums:
        rows = sources_df.loc[sources_df["Number"] == n]
        if len(rows) == 0:
            continue
        cell = str(rows.iloc[0].get("Corresponding Link(s)", "")).replace("<br>", "|")
        for part in cell.split("|"):
            part = part.strip()
            if part and part not in links:
                links.append(part)
    return links

# ── Signal / rule mapping ─────────────────────────────────────────────────────

def build_signal_to_rules(sm: pd.DataFrame) -> dict[str, list[str]]:
    d: dict[str, list[str]] = {}
    for _, row in sm.iterrows():
        sid = str(row["signal_id"]).strip()
        rid = str(row["rule_id"]).strip()
        if sid and rid:
            d.setdefault(sid, []).append(rid)
    for k, v in d.items():
        seen: set[str] = set()
        d[k] = [x for x in v if not (x in seen or seen.add(x))]
    return d

def match_signals(feature_name: str, sm: pd.DataFrame) -> list[str]:
    fn = feature_name.lower()
    hits: list[str] = []
    for sid in sm["signal_id"].tolist():
        if sid and sid.lower() in fn:
            hits.append(sid)
    if not hits:
        for _, row in sm.iterrows():
            sid = str(row["signal_id"]).strip()
            for tok in re.split(r"[;, ]+", str(row["columns_used"]).lower()):
                tok = tok.strip()
                if tok and tok in fn:
                    hits.append(sid)
                    break
    seen: set[str] = set()
    return [x for x in hits if not (x in seen or seen.add(x))][:3]

# ── Feature matrix ────────────────────────────────────────────────────────────

def build_feature_matrix(
    processed_dir: Path,
    split: str,
    behavior_path: Path,
    emb_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ids_df_with_label, X_numeric)"""
    customers = pd.read_csv(processed_dir / f"customers_{split}.csv", usecols=[ID, TARGET])
    customers[ID] = customers[ID].astype(str)
    customers[TARGET] = pd.to_numeric(customers[TARGET], errors="coerce").fillna(0).astype(int)

    behavior = pd.read_parquet(behavior_path)
    behavior[ID] = behavior[ID].astype(str)
    if TARGET in behavior.columns:
        behavior = behavior.drop(columns=[TARGET])

    df = customers.merge(behavior, on=ID, how="left")

    emb_path = processed_dir / "static" / f"{emb_prefix}_{split}.parquet"
    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        emb[ID] = emb[ID].astype(str)
        if TARGET in emb.columns:
            emb = emb.drop(columns=[TARGET])
        dup = set(df.columns) & set(emb.columns) - {ID}
        if dup:
            emb = emb.drop(columns=list(dup))
        df = df.merge(emb, on=ID, how="left")

    ids_df = df[[ID, TARGET]].copy()
    X = df.drop(columns=[ID, TARGET], errors="ignore").copy()

    # encode
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    obj_cols = [c for c in X.columns if str(X[c].dtype) in ("object","category")]
    if obj_cols:
        for c in obj_cols:
            X[c] = X[c].fillna("Unknown").astype(str).str.strip()
        X = pd.get_dummies(X, columns=obj_cols, dummy_na=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return ids_df, X

def align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    feat = None
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        feat = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            feat = model.get_booster().feature_names
        except Exception:
            feat = None
    if feat is None:
        return X
    if feat and re.fullmatch(r"f\d+", str(feat[0])):
        return X
    missing_cols = {c: 0.0 for c in feat if c not in X.columns}
    if missing_cols:
        X = pd.concat([X, pd.DataFrame(missing_cols, index=X.index)], axis=1)
    return X[feat]

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_csv",     required=True,  help="Blended or raw ranked CSV with customer_id + score")
    ap.add_argument("--model_dir",      required=True)
    ap.add_argument("--iforest_csv",    default="",     help="Optional: iforest_scores.csv with customer_id + iforest_score + iforest_pct")
    ap.add_argument("--rulebook_csv",   default="")
    ap.add_argument("--signal_map_csv", default="")
    ap.add_argument("--sources_csv",    default="")
    ap.add_argument("--processed_dir",  default="")
    ap.add_argument("--behavior_path",  default="")
    ap.add_argument("--emb_prefix",     default="bow_stream_emb_v4")
    ap.add_argument("--split",          default="test", choices=["train","val","test"])
    ap.add_argument("--top_n",          type=int, default=50)
    ap.add_argument("--top_features",   type=int, default=10)
    ap.add_argument("--out_run",        required=True)
    args = ap.parse_args()

    repo = _repo_root()

    # defaults
    rulebook_csv   = Path(args.rulebook_csv)   if args.rulebook_csv   else repo / "knowledge_base" / "aml_rulebook.csv"
    signal_map_csv = Path(args.signal_map_csv) if args.signal_map_csv else repo / "knowledge_base" / "rule_signal_map.csv"
    sources_csv    = Path(args.sources_csv)    if args.sources_csv    else repo / "knowledge_base" / "sources.csv"
    processed_dir  = Path(args.processed_dir)  if args.processed_dir  else repo / "data" / "processed"
    behavior_path  = Path(args.behavior_path)  if args.behavior_path  else processed_dir / "features" / "customer_behavior_v2.parquet"

    out_dir = repo / "outputs" / "runs" / args.out_run
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ranked list ──────────────────────────────────────────────────────
    ranked = pd.read_csv(args.ranked_csv)
    ranked[ID] = ranked[ID].astype(str)
    # support multiple score column names
    for sc in ["score","score_blended","fused_score"]:
        if sc in ranked.columns and sc != "score":
            ranked["score"] = ranked[sc]
            break
    if "score" not in ranked.columns:
        raise SystemExit("[ERR] ranked_csv needs a score / score_blended / fused_score column")

    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    top = ranked.head(args.top_n).copy()
    top_ids = top[ID].tolist()

    # ── Load iforest scores (optional) ────────────────────────────────────────
    iforest_df: pd.DataFrame | None = None
    if args.iforest_csv and Path(args.iforest_csv).exists():
        iforest_df = pd.read_csv(args.iforest_csv)
        iforest_df[ID] = iforest_df[ID].astype(str)

    # ── Load model ────────────────────────────────────────────────────────────
    model_dir = Path(args.model_dir)
    model_path = next(
        (model_dir / n for n in ["model.joblib","model.pkl","clf.joblib"] if (model_dir / n).exists()),
        None
    )
    if model_path is None:
        jbs = sorted(model_dir.glob("*.joblib"), key=lambda p: p.stat().st_size, reverse=True)
        if not jbs:
            raise SystemExit(f"[ERR] no joblib model in {model_dir}")
        model_path = jbs[0]
    model = joblib.load(model_path)
    print(f"[INFO] loaded model from {model_path}")

    # ── Build feature matrix ──────────────────────────────────────────────────
    ids_df, X_all = build_feature_matrix(processed_dir, args.split, behavior_path, args.emb_prefix)
    X_all.index = ids_df[ID].astype(str).values
    X_all = align_to_model(X_all, model)
    print(f"[INFO] feature matrix: {X_all.shape[0]} rows x {X_all.shape[1]} cols")

    # ── XGBoost pred_contribs ─────────────────────────────────────────────────
    booster = model.get_booster()
    X_top = X_all.loc[top_ids]
    dm = xgb.DMatrix(X_top.to_numpy(), feature_names=list(X_top.columns))
    contrib = booster.predict(dm, pred_contribs=True)          # (n, n_features+1)
    contrib_df = pd.DataFrame(contrib[:, :-1], columns=list(X_top.columns), index=top_ids)

    # ── Load KB ───────────────────────────────────────────────────────────────
    rb = load_rulebook(rulebook_csv)
    sm = load_signal_map(signal_map_csv)
    sources_df = load_sources(sources_csv)
    signal_to_rules = build_signal_to_rules(sm)
    rb_idx = rb.set_index("rule_id", drop=False)

    # ── Per-customer explanations ─────────────────────────────────────────────
    rows = []
    for cid in top_ids:
        score_val = float(top.loc[top[ID] == cid, "score"].iloc[0])
        contribs = contrib_df.loc[cid]

        # top features by absolute contribution
        top_feat_names = contribs.abs().sort_values(ascending=False).head(args.top_features).index.tolist()

        reasons = []
        for feat in top_feat_names:
            impact = float(contribs[feat])
            val    = float(X_top.loc[cid, feat]) if feat in X_top.columns else 0.0
            sigs   = match_signals(feat, sm)

            mapped_rules: list[str] = []
            for sid in sigs:
                mapped_rules.extend(signal_to_rules.get(sid, []))
            mapped_rules = list(dict.fromkeys(mapped_rules))[:3]

            rule_details = []
            for rid in mapped_rules:
                if rid not in rb_idx.index:
                    continue
                rr = rb_idx.loc[rid]
                src_nums = extract_source_nums(str(rr.get("source","")))
                links    = resolve_links(src_nums, sources_df)
                rule_details.append({
                    "rule_id":     rid,
                    "category":    str(rr["category"]),
                    "red_flag":    str(rr["red_flag"]),
                    "typology":    str(rr["typology"]),
                    "source":      str(rr["source"]),
                    "feasibility": str(rr["feasibility"]),
                    "links":       links,
                })

            reasons.append({
                "feature":    feat,
                "value":      val,
                "impact":     impact,
                "direction":  "risk_up" if impact > 0 else "risk_down",
                "signal_ids": sigs,
                "rules":      rule_details,
            })

        # iforest context
        iforest_score, iforest_pct = None, None
        if iforest_df is not None:
            row_if = iforest_df.loc[iforest_df[ID] == cid]
            if len(row_if) > 0:
                iforest_score = float(row_if.iloc[0].get("iforest_score", np.nan))
                iforest_pct   = float(row_if.iloc[0].get("iforest_pct",   np.nan))

        # write per-customer JSON
        out_json = {
            "customer_id":   cid,
            "score":         score_val,
            "iforest_score": iforest_score,
            "iforest_pct":   iforest_pct,
            "reasons":       reasons,
        }
        (out_dir / f"{cid}.json").write_text(json.dumps(out_json, indent=2))

        # flat CSV row
        top_rules_flat = list(dict.fromkeys(
            rr["rule_id"] for r in reasons for rr in r["rules"]
        ))[:6]
        top_links_flat = list(dict.fromkeys(
            lnk for r in reasons for rr in r["rules"] for lnk in rr["links"]
        ))[:5]
        risk_up   = [r for r in reasons if r["impact"] > 0]
        risk_down = [r for r in reasons if r["impact"] < 0]

        rows.append({
            "customer_id":       cid,
            "score":             score_val,
            "iforest_score":     iforest_score,
            "iforest_pct":       iforest_pct,
            "top_rules":         ";".join(top_rules_flat),
            "top_rule_links":    " | ".join(top_links_flat),
            "risk_up_features":  ";".join(f"{r['feature']}({r['impact']:+.4f})" for r in risk_up[:6]),
            "risk_down_features":";".join(f"{r['feature']}({r['impact']:+.4f})" for r in risk_down[:4]),
        })

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_csv = out_dir / f"{args.split}_top{args.top_n}_final_explanations.csv"
    out_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    out_df.to_csv(out_csv, index=False)

    # pos ranks sanity check
    pos_ranks = []
    if TARGET in ranked.columns:
        pos_ranks = ranked.index[
            pd.to_numeric(ranked[TARGET], errors="coerce").fillna(0).astype(int) == 1
        ].tolist()
        (out_dir / f"{args.split}_pos_ranks.txt").write_text(str(pos_ranks))

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {args.top_n} JSON files to {out_dir}")
    print(f"[INFO] pos_ranks={pos_ranks}")

if __name__ == "__main__":
    main()
