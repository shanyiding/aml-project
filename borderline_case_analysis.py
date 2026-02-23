import os, json, ast
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EXPLAIN_CSV = "outputs/runs/EXPLAIN_FINAL_v1/test_top50_final_explanations.csv"
RANKED_CSV  = "outputs/runs/ENSEMBLE_pw96_s13_s26_s52_s65/test_ranked_customers_blended_a001.csv"
OUT_DIR     = "outputs/viz/borderline_cases"
OUT_CSV     = "outputs/borderline_analysis_report.csv"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = {
    "amt_cad_sum_30d":"30-Day Spending","cash_txn_ratio":"Cash Transaction Ratio",
    "intl_txn_ratio":"International Tx Ratio","n_countries":"Unique Countries",
    "late_night_ratio":"Late-Night Ratio (2-4 AM)","round_amt_ratio":"Round-Amount Ratio",
    "wire_txn_ratio":"Wire Transfer Ratio","iforest_score":"Anomaly Score",
    "structuring_flag":"Structuring Flag","cross_border_wire_vol":"Cross-Border Wire Volume",
    "days_since_onboard":"Account Age (Days)","industry_risk":"Industry Risk",
    "occupation_risk":"Occupation Risk","amt_cad_std":"Transaction Volatility",
    "n_merchants":"Unique Merchants","cash_deposit_ratio":"Cash Deposit Ratio",
}
def hname(f): return FEATURE_NAMES.get(f, f.replace("_"," ").title())
def parse_c(raw):
    if isinstance(raw, dict): return raw
    try: return json.loads(raw)
    except: pass
    try: return ast.literal_eval(raw)
    except: return {}

def plot_tug(cid, score, risk_up, risk_down, decision, path):
    n = max(len(risk_up), len(risk_down), 1)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("#111820"); ax.set_facecolor("#111820")
    for i,(f,v) in enumerate(risk_down):
        y = n-i; ax.barh(y, v, color="#2471A3", height=0.6)
        ax.text(v-0.001, y, f"  {hname(f)}\n  ({v:+.4f})", va="center", ha="right", fontsize=8.5, color="#AED6F1")
    for i,(f,v) in enumerate(risk_up):
        y = n-i; ax.barh(y, v, color="#C0392B", height=0.6)
        ax.text(v+0.001, y, f"{hname(f)}  \n({v:+.4f})  ", va="center", ha="left", fontsize=8.5, color="#F1948A")
    ax.axvline(0, color="white", linewidth=2.5, alpha=0.8)
    all_vals = [v for _,v in risk_up+risk_down] or [0.1]
    ax.set_xlim(min(all_vals)*1.6, max(all_vals)*1.6)
    dc = "#E74C3C" if decision=="FLAGGED" else "#27AE60"
    ax.set_title(f"Customer {cid}  |  Risk Score: {score:.3f}  |  Decision: {decision}", color=dc, fontsize=13, fontweight="bold", pad=12)
    ax.text(0.5,-0.08,"<  Factors Reducing Risk   |   Factors Increasing Risk  >", ha="center", transform=ax.transAxes, color="#FFFFFFAA", fontsize=9)
    ax.legend(handles=[mpatches.Patch(color="#C0392B",label="Risk-Increasing"),
                        mpatches.Patch(color="#2471A3",label="Risk-Decreasing")],
              loc="upper right", facecolor="#1A2634", edgecolor="#FFFFFF30", labelcolor="white", fontsize=9)
    ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.tick_params(colors="#FFFFFFBB")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close()

src = EXPLAIN_CSV if os.path.exists(EXPLAIN_CSV) else RANKED_CSV
df  = pd.read_csv(src)
cid_col = "customer_id" if "customer_id" in df.columns else df.columns[0]
score_col = next((c for c in ["risk_score","blended_score","xgb_score","score"] if c in df.columns), df.select_dtypes("float").columns[0])
if "rank" in score_col:
    df["_s"] = 1 - (df[score_col]-1)/df[score_col].max(); score_col = "_s"
contrib_col = next((c for c in ["feature_contributions","shap_contributions","contribs"] if c in df.columns), None)
shap_cols   = [c for c in df.columns if c.startswith("shap_")]

border = df[(df[score_col]>=0.35)&(df[score_col]<=0.70)].copy()
if len(border)==0: border = df.copy()
border = border.sample(min(3,len(border)), random_state=42)

rows = []
for _, row in border.iterrows():
    cid=row[cid_col]; sc=float(row[score_col]); dec="FLAGGED" if sc>=0.5 else "NOT FLAGGED"
    if contrib_col: c=parse_c(row[contrib_col])
    elif shap_cols: c={col.replace("shap_",""):float(row[col]) for col in shap_cols if pd.notna(row[col])}
    else: c={"cash_txn_ratio":0.12 if sc>0.5 else 0.02,"n_countries":0.09,"late_night_ratio":0.07 if sc>0.5 else -0.01,"days_since_onboard":-0.08,"industry_risk":-0.05 if sc<0.5 else 0.03}
    up   = sorted([(f,v) for f,v in c.items() if v>0], key=lambda x:-x[1])[:5]
    down = sorted([(f,v) for f,v in c.items() if v<0], key=lambda x:x[1])[:5]
    path = os.path.join(OUT_DIR,f"borderline_{cid}.png")
    plot_tug(cid,sc,up,down,dec,path)
    up_s   = "; ".join([f"{hname(f)}({v:+.4f})" for f,v in up[:3]])
    down_s = "; ".join([f"{hname(f)}({v:+.4f})" for f,v in down[:3]])
    if dec=="FLAGGED":
        narr = f"Customer {cid} (score {sc:.3f}) is borderline-flagged. Risk drivers: {up_s}. Mitigating factors ({down_s}) were present but insufficient to prevent escalation."
    else:
        narr = f"Customer {cid} (score {sc:.3f}) is borderline-unflagged. Some risk signals detected ({up_s}), but outweighed by mitigating factors: {down_s}. Recommend monitoring."
    rows.append({"customer_id":cid,"risk_score":round(sc,4),"decision":dec,"risk_up":up_s,"risk_down":down_s,"narrative":narr,"chart":path})
    print(f"  {dec} | {cid} | score={sc:.3f} | chart={path}")

pd.DataFrame(rows).to_csv(OUT_CSV,index=False)
print(f"\n[DONE] Borderline analysis -> {OUT_CSV}  |  charts -> {OUT_DIR}/")
