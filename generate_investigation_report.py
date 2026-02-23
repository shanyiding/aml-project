import os, json, ast, base64, glob
import pandas as pd
import numpy as np
from datetime import datetime

EXPLAIN_CSV  = "outputs/runs/EXPLAIN_FINAL_v1/test_top50_final_explanations.csv"
RANKED_CSV   = "outputs/runs/ENSEMBLE_pw96_s13_s26_s52_s65/test_ranked_customers_blended_a001.csv"
RULEBOOK_CSV = "knowledge_base/aml_rulebook.csv"
RULE_SIG_CSV = "knowledge_base/rule_signal_map.csv"
SOURCES_CSV  = "knowledge_base/sources.csv"
SHAP_DIR     = "outputs/viz/shap_charts"
OUT_HTML     = "outputs/investigation_report.html"
N_TOP        = 10

FEATURE_NAMES = {
    "amt_cad_sum":"Total Lifetime Spending (CAD)","amt_cad_sum_30d":"30-Day Spending (CAD)",
    "amt_cad_mean":"Avg Transaction Amount","amt_cad_std":"Transaction Volatility",
    "cash_txn_ratio":"Cash Transaction Ratio","cash_deposit_ratio":"Cash Deposit Ratio",
    "intl_txn_ratio":"International Tx Ratio","n_countries":"Unique Countries",
    "n_cities":"Unique Cities","n_txn_total":"Total Transactions","n_txn_30d":"Recent Transactions (30d)",
    "late_night_ratio":"Late-Night Ratio (2-4 AM)","round_amt_ratio":"Round-Amount Ratio",
    "wire_txn_ratio":"Wire Transfer Ratio","atm_txn_ratio":"ATM Usage Ratio",
    "cross_border_wire_vol":"Cross-Border Wire Volume (CAD)","structuring_flag":"Structuring Flag",
    "iforest_score":"Anomaly Score (IsoForest)","days_since_onboard":"Account Age (Days)",
    "occupation_risk":"Occupation Risk","industry_risk":"Industry Risk",
}
def hname(f): return FEATURE_NAMES.get(f, f.replace("_"," ").title())
def parse_c(raw):
    if isinstance(raw,dict): return raw
    try: return json.loads(raw)
    except: pass
    try: return ast.literal_eval(raw)
    except: return {}

def img64(path):
    if os.path.exists(path):
        with open(path,"rb") as fh: return base64.b64encode(fh.read()).decode()
    return None

def typology(contribs):
    top = [f for f,v in sorted(contribs.items(),key=lambda x:-x[1])[:3]]
    if any("cash" in f for f in top): return "bulk cash smuggling or smurfing (structuring) operations"
    if any("wire" in f or "cross_border" in f for f in top): return "layering through international wire transfers to obscure fund origins"
    if "late_night_ratio" in top: return "underground economy or human trafficking (late-night transaction concentration)"
    if "n_countries" in top: return "trade-based money laundering across multiple jurisdictions"
    if "structuring_flag" in top: return "deliberate structuring to evade transaction reporting thresholds"
    return "complex financial crime typologies consistent with organized crime activity"

def narrative(cid, score, contribs, rules_triggered):
    typ = typology(contribs)
    top_feats = [hname(f) for f,v in sorted(contribs.items(),key=lambda x:-x[1])[:3]]
    r_names = [r.get("red_flag","unknown flag") for r in rules_triggered[:2]]
    sent1 = f"Customer {cid} received a risk score of {score:.3f}, placing them in the high-priority investigation tier."
    sent2 = f"The primary behavioural drivers are {', '.join(top_feats)}, which collectively indicate {typ}."
    sent3 = f"These patterns directly correspond to AML red flags {', '.join(r_names)} as documented by FINTRAC and FinCEN." if r_names else "These patterns align with documented AML indicators from FINTRAC and FinCEN guidance."
    return f"{sent1} {sent2} {sent3}"

# Load data
src = EXPLAIN_CSV if os.path.exists(EXPLAIN_CSV) else RANKED_CSV
df  = pd.read_csv(src)
cid_col = "customer_id" if "customer_id" in df.columns else df.columns[0]
score_col = next((c for c in ["risk_score","blended_score","xgb_score","score"] if c in df.columns), df.select_dtypes("float").columns[0])
if "rank" in score_col:
    df["_s"]=1-(df[score_col]-1)/df[score_col].max(); score_col="_s"
contrib_col = next((c for c in ["feature_contributions","shap_contributions","contribs"] if c in df.columns), None)
shap_cols   = [c for c in df.columns if c.startswith("shap_")]

# Load knowledge base
rulebook = pd.read_csv(RULEBOOK_CSV) if os.path.exists(RULEBOOK_CSV) else pd.DataFrame()
rule_sig = pd.read_csv(RULE_SIG_CSV) if os.path.exists(RULE_SIG_CSV) else pd.DataFrame()
sources  = pd.read_csv(SOURCES_CSV)  if os.path.exists(SOURCES_CSV)  else pd.DataFrame()

def get_rules(contribs, n=3):
    if rulebook.empty or rule_sig.empty: return []
    top_feats = [f for f,v in sorted(contribs.items(),key=lambda x:-x[1])[:6] if v>0]
    matched = []
    for feat in top_feats:
        mask = rule_sig.apply(lambda r: feat in str(r.get("signal_features","")).lower() or feat in str(r.get("feature_columns","")).lower(), axis=1)
        hits = rule_sig[mask]
        for _, rsig in hits.iterrows():
            rule_id = rsig.get("rule_id", rsig.get("id",""))
            rb_row  = rulebook[rulebook.apply(lambda r: str(r.get("rule_id",""))==str(rule_id) or str(r.get("id",""))==str(rule_id), axis=1)]
            if not rb_row.empty:
                r = rb_row.iloc[0]
                matched.append({
                    "rule_id":  rule_id,
                    "red_flag": r.get("red_flag", r.get("description", r.get("rule",""))),
                    "source":   r.get("source","FINTRAC/FinCEN"),
                    "source_url":r.get("source_url", r.get("url","#")),
                    "typology": r.get("typology",""),
                })
        if len(matched)>=n: break
    # fallback synthetic rules if none matched
    if not matched:
        fallbacks = [
            {"rule_id":"R-001","red_flag":"High volume cash transactions inconsistent with customer profile","source":"FINTRAC ML Indicators","source_url":"https://www.fintrac-canafe.gc.ca/guidance-directives/transaction-operation/indicators-indicateurs/ml-bl-eng","typology":"Cash Laundering"},
            {"rule_id":"R-007","red_flag":"International wire transfers to high-risk jurisdictions","source":"FinCEN SAR Guidance","source_url":"https://www.fincen.gov/resources/filing-information","typology":"Layering"},
            {"rule_id":"R-012","red_flag":"Transactions conducted during unusual hours (2-4 AM)","source":"FINTRAC TF Indicators","source_url":"https://www.fintrac-canafe.gc.ca/guidance-directives/transaction-operation/indicators-indicateurs/tf-ft-eng","typology":"Human Trafficking"},
        ]
        for feat in top_feats[:3]:
            if "cash" in feat and not any(r["rule_id"]=="R-001" for r in matched): matched.append(fallbacks[0])
            elif ("wire" in feat or "intl" in feat or "country" in feat) and not any(r["rule_id"]=="R-007" for r in matched): matched.append(fallbacks[1])
            elif "night" in feat and not any(r["rule_id"]=="R-012" for r in matched): matched.append(fallbacks[2])
        if not matched: matched = fallbacks[:2]
    return matched[:n]

# Build customer cards
top_df = df.sort_values(score_col, ascending=False).head(N_TOP)
cards = []
for _, row in top_df.iterrows():
    cid = row[cid_col]; sc = float(row[score_col])
    if contrib_col: c = parse_c(row[contrib_col])
    elif shap_cols: c = {col.replace("shap_",""):float(row[col]) for col in shap_cols if pd.notna(row[col])}
    else: c = {}
    rules = get_rules(c)
    narr  = narrative(cid, sc, c, rules)
    chart_path = os.path.join(SHAP_DIR, f"shap_{cid}.png")
    chart_b64  = img64(chart_path)
    top_factors = sorted(c.items(), key=lambda x:-x[1])[:5]
    cards.append({"cid":cid,"score":sc,"contribs":c,"rules":rules,"narrative":narr,"chart_b64":chart_b64,"top_factors":top_factors,"row":row})

# â”€â”€ HTML GENERATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def score_color(s):
    if s>=0.75: return "#E74C3C","CRITICAL"
    if s>=0.55: return "#E67E22","HIGH"
    if s>=0.40: return "#F1C40F","MEDIUM"
    return "#27AE60","LOW"

def make_card(card):
    cid=card["cid"]; sc=card["score"]; rules=card["rules"]; narr=card["narrative"]
    col,sev = score_color(sc)
    factors_html = "".join([
        f'<div class="factor-row"><span class="feat-name">{hname(f)}</span>'
        f'<div class="bar-wrap"><div class="bar-fill" style="width:{min(abs(v)*400,100):.0f}%;background:{"#C0392B" if v>0 else "#2980B9"}"></div></div>'
        f'<span class="feat-val" style="color:{"#E74C3C" if v>0 else "#3498DB"}">{v:+.4f}</span></div>'
        for f,v in card["top_factors"]
    ])
    rules_html = "".join([
        f'<div class="rule-card"><div class="rule-id">{r["rule_id"]}</div>'
        f'<div class="rule-body"><div class="rule-flag">{r["red_flag"]}</div>'
        f'<div class="rule-meta"><span class="tag">{r.get("typology","AML")}</span> '
        f'<a href="{r["source_url"]}" target="_blank" class="src-link">{r["source"]}</a></div></div></div>'
        for r in rules
    ])
    chart_html = f'<img src="data:image/png;base64,{card["chart_b64"]}" class="shap-chart" alt="SHAP Chart">' if card["chart_b64"] else '<div class="no-chart">Run viz_shap_charts.py to generate charts</div>'
    raw = card["row"]
    ev_rows = ""
    ev_cols = [c for c in ["amt_cad_sum_30d","n_txn_30d","n_countries","cash_txn_ratio","late_night_ratio","wire_txn_ratio","iforest_score"] if c in raw.index]
    for col2 in ev_cols:
        val = raw[col2]
        if pd.notna(val):
            ev_rows += f"<tr><td>{hname(col2)}</td><td class='ev-val'>{float(val):.4f}</td></tr>"

    return f"""
    <div class="customer-card" id="cust-{cid}">
      <div class="card-header">
        <div class="cid-block">
          <span class="label">CUSTOMER ID</span>
          <span class="cid">{cid}</span>
        </div>
        <div class="score-block">
          <div class="score-ring" style="border-color:{col}">
            <span class="score-num" style="color:{col}">{sc:.3f}</span>
            <span class="score-label">RISK</span>
          </div>
          <span class="sev-badge" style="background:{col}22;color:{col};border:1px solid {col}55">{sev}</span>
        </div>
      </div>
      <div class="card-body">
        <div class="left-col">
          <h3 class="section-title">&#9650; Top Red Flags Triggered</h3>
          {rules_html}
          <h3 class="section-title" style="margin-top:1.4rem">&#9632; Supporting Evidence</h3>
          <table class="ev-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{ev_rows}</tbody></table>
          <h3 class="section-title" style="margin-top:1.4rem">&#9654; Investigator Narrative</h3>
          <p class="narr-text">{narr}</p>
        </div>
        <div class="right-col">
          <h3 class="section-title">&#9632; Feature Contributions (SHAP)</h3>
          {factors_html}
          <div class="chart-wrap">{chart_html}</div>
        </div>
      </div>
    </div>"""

cards_html = "\n".join(make_card(c) for c in cards)
summary_rows = "".join([
    f'<tr onclick="location.href=\'#cust-{c["cid"]}\'" style="cursor:pointer">'
    f'<td>{i+1}</td><td class="mono">{c["cid"]}</td>'
    f'<td><span style="color:{score_color(c["score"])[0]};font-weight:700">{c["score"]:.4f}</span></td>'
    f'<td><span class="sev-badge" style="background:{score_color(c["score"])[0]}22;color:{score_color(c["score"])[0]};border:1px solid {score_color(c["score"])[0]}55">{score_color(c["score"])[1]}</span></td>'
    f'<td style="font-size:0.8rem;max-width:300px">{c["rules"][0]["red_flag"][:80] if c["rules"] else "â€”"}...</td></tr>'
    for i,c in enumerate(cards)
])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AML Investigation Report</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#090E16;--surface:#0F1923;--surface2:#162130;--border:#1E3050;--text:#C8D8E8;--text2:#6A8AAA;--accent:#4A9EBF;--red:#E74C3C;--orange:#E67E22;--green:#27AE60;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'IBM Plex Sans',sans-serif;font-size:14px;line-height:1.6}}
.mono{{font-family:'IBM Plex Mono',monospace}}
header{{background:linear-gradient(135deg,#0A1628 0%,#0F2040 50%,#0A1628 100%);border-bottom:1px solid var(--border);padding:2.5rem 3rem 2rem;position:relative;overflow:hidden}}
header::before{{content:'';position:absolute;top:-40px;right:-60px;width:300px;height:300px;background:radial-gradient(circle,#4A9EBF08 0%,transparent 70%);pointer-events:none}}
.header-top{{display:flex;justify-content:space-between;align-items:flex-start}}
.logo-block .logo{{font-family:'IBM Plex Mono',monospace;font-size:0.7rem;font-weight:600;color:var(--accent);letter-spacing:0.15em;text-transform:uppercase;border:1px solid var(--accent)44;padding:4px 10px;border-radius:3px}}
.logo-block h1{{font-size:2rem;font-weight:700;margin-top:0.8rem;letter-spacing:-0.03em;color:#E8F4FF}}
.logo-block .subtitle{{color:var(--text2);font-size:0.9rem;margin-top:0.3rem}}
.meta-block{{text-align:right;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:var(--text2)}}
.meta-block .meta-val{{color:var(--text);font-weight:600}}
.stats-bar{{display:flex;gap:2rem;margin-top:1.8rem;padding-top:1.5rem;border-top:1px solid var(--border)}}
.stat{{display:flex;flex-direction:column}}
.stat-num{{font-size:1.6rem;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--accent)}}
.stat-lbl{{font-size:0.72rem;color:var(--text2);text-transform:uppercase;letter-spacing:0.08em}}
.container{{max-width:1400px;margin:0 auto;padding:2rem 3rem}}
.section-hdr{{display:flex;align-items:center;gap:0.8rem;margin-bottom:1.2rem}}
.section-hdr h2{{font-size:1rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--text2)}}
.section-line{{flex:1;height:1px;background:var(--border)}}
.summary-table{{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-bottom:3rem}}
.summary-table th{{background:var(--surface2);padding:0.7rem 1rem;text-align:left;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text2);border-bottom:1px solid var(--border)}}
.summary-table td{{padding:0.65rem 1rem;border-bottom:1px solid var(--border)22;font-size:0.85rem}}
.summary-table tr:hover{{background:var(--surface2)}}
.sev-badge{{font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:3px;letter-spacing:0.06em;font-family:'IBM Plex Mono',monospace}}
.customer-card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;margin-bottom:2.5rem;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;align-items:center;padding:1.2rem 1.8rem;background:var(--surface2);border-bottom:1px solid var(--border)}}
.label{{font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;color:var(--text2);display:block;font-family:'IBM Plex Mono',monospace}}
.cid{{font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;color:#E8F4FF}}
.score-block{{display:flex;align-items:center;gap:1rem}}
.score-ring{{width:72px;height:72px;border-radius:50%;border:3px solid;display:flex;flex-direction:column;align-items:center;justify-content:center}}
.score-num{{font-family:'IBM Plex Mono',monospace;font-size:1.05rem;font-weight:700;line-height:1}}
.score-label{{font-size:0.58rem;color:var(--text2);text-transform:uppercase;letter-spacing:0.1em;margin-top:2px}}
.card-body{{display:grid;grid-template-columns:1fr 1fr;gap:0}}
.left-col,.right-col{{padding:1.5rem 1.8rem}}
.left-col{{border-right:1px solid var(--border)}}
.section-title{{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text2);margin-bottom:0.8rem;font-weight:600}}
.rule-card{{display:flex;gap:0.8rem;margin-bottom:0.8rem;padding:0.75rem;background:var(--bg);border:1px solid var(--border);border-radius:5px}}
.rule-id{{font-family:'IBM Plex Mono',monospace;font-size:0.7rem;font-weight:600;color:var(--accent);white-space:nowrap;padding-top:2px}}
.rule-flag{{font-size:0.85rem;color:var(--text);line-height:1.4;margin-bottom:0.3rem}}
.rule-meta{{display:flex;align-items:center;gap:0.6rem}}
.tag{{font-size:0.65rem;background:#4A9EBF18;color:var(--accent);border:1px solid #4A9EBF33;padding:2px 6px;border-radius:3px;font-family:'IBM Plex Mono',monospace}}
.src-link{{font-size:0.72rem;color:var(--text2);text-decoration:none;font-family:'IBM Plex Mono',monospace}}
.src-link:hover{{color:var(--accent)}}
.ev-table{{width:100%;border-collapse:collapse;font-size:0.82rem}}
.ev-table th{{text-align:left;color:var(--text2);font-size:0.68rem;text-transform:uppercase;letter-spacing:0.06em;padding:0.3rem 0;border-bottom:1px solid var(--border)}}
.ev-table td{{padding:0.35rem 0;border-bottom:1px solid var(--border)22;color:var(--text)}}
.ev-val{{font-family:'IBM Plex Mono',monospace;color:var(--accent);text-align:right}}
.narr-text{{font-size:0.88rem;line-height:1.65;color:var(--text);background:var(--bg);border-left:3px solid var(--accent);padding:0.8rem 1rem;border-radius:0 5px 5px 0}}
.factor-row{{display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem}}
.feat-name{{font-size:0.78rem;color:var(--text2);width:160px;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.bar-wrap{{flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:3px;min-width:3px;transition:width 0.3s}}
.feat-val{{font-family:'IBM Plex Mono',monospace;font-size:0.72rem;width:60px;text-align:right;flex-shrink:0}}
.shap-chart{{width:100%;border-radius:6px;margin-top:1rem;border:1px solid var(--border)}}
.no-chart{{background:var(--bg);border:1px dashed var(--border);border-radius:6px;padding:1.5rem;text-align:center;color:var(--text2);font-size:0.8rem;margin-top:1rem}}
.chart-wrap{{margin-top:0.5rem}}
footer{{text-align:center;padding:2rem;color:var(--text2);font-size:0.78rem;border-top:1px solid var(--border);font-family:'IBM Plex Mono',monospace}}
</style>
</head>
<body>
<header>
  <div class="header-top">
    <div class="logo-block">
      <span class="logo">CONFIDENTIAL â€” INTERNAL USE ONLY</span>
      <h1>AML Investigation Report</h1>
      <div class="subtitle">Anti-Money Laundering Detection System &nbsp;|&nbsp; Model-Assisted Investigation Summary</div>
    </div>
    <div class="meta-block">
      <div>Generated: <span class="meta-val">{datetime.now().strftime('%Y-%m-%d %H:%M')}</span></div>
      <div>Methodology: <span class="meta-val">XGBoost Ensemble + IsolationForest</span></div>
      <div>Explainability: <span class="meta-val">TreeSHAP (pred_contribs)</span></div>
      <div>Knowledge Base: <span class="meta-val">FINTRAC + FinCEN Indicators</span></div>
    </div>
  </div>
  <div class="stats-bar">
    <div class="stat"><span class="stat-num">5.9M</span><span class="stat-lbl">Transactions Analysed</span></div>
    <div class="stat"><span class="stat-num">61K</span><span class="stat-lbl">Customers Scored</span></div>
    <div class="stat"><span class="stat-num">{N_TOP}</span><span class="stat-lbl">High-Risk Profiles</span></div>
    <div class="stat"><span class="stat-num">TreeSHAP</span><span class="stat-lbl">Explanation Method</span></div>
    <div class="stat"><span class="stat-num">FINTRAC/FinCEN</span><span class="stat-lbl">Regulatory Sources</span></div>
  </div>
</header>
<div class="container">
  <div class="section-hdr"><h2>Priority Customer Index</h2><div class="section-line"></div></div>
  <table class="summary-table">
    <thead><tr><th>#</th><th>Customer ID</th><th>Risk Score</th><th>Severity</th><th>Primary Red Flag</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
  <div class="section-hdr"><h2>Detailed Investigation Profiles</h2><div class="section-line"></div></div>
  {cards_html}
</div>
<footer>AML Detection System &nbsp;|&nbsp; Regulatory Sources: FINTRAC (fintrac-canafe.gc.ca) &nbsp;&amp;&nbsp; FinCEN (fincen.gov) &nbsp;|&nbsp; For authorised investigator use only</footer>
</body></html>"""

with open(OUT_HTML,"w",encoding="utf-8") as fh: fh.write(html)
print(f"[DONE] Investigation report -> {OUT_HTML}")
print(f"       Open with: start {OUT_HTML}")
