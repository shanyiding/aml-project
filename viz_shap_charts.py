import os, json, ast
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

EXPLAIN_CSV = "outputs/runs/EXPLAIN_FINAL_v1/test_top50_final_explanations.csv"
OUT_DIR     = "outputs/viz/shap_charts"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = {
    "amt_cad_sum":"Total Lifetime Spending (CAD)","amt_cad_sum_30d":"30-Day Spending (CAD)",
    "amt_cad_sum_7d":"7-Day Spending (CAD)","amt_cad_mean":"Avg Transaction Amount",
    "amt_cad_std":"Transaction Volatility","cash_txn_ratio":"Cash Transaction Ratio",
    "cash_deposit_ratio":"Cash Deposit Ratio","cash_withdrawal_ratio":"Cash Withdrawal Ratio",
    "intl_txn_ratio":"International Tx Ratio","n_countries":"Unique Countries",
    "n_cities":"Unique Cities","n_merchants":"Unique Merchants",
    "n_txn_total":"Total Transactions","n_txn_30d":"Transactions (Last 30 Days)",
    "late_night_ratio":"Late-Night Ratio (2-4 AM)","round_amt_ratio":"Round-Amount Ratio",
    "wire_txn_ratio":"Wire Transfer Ratio","atm_txn_ratio":"ATM Usage Ratio",
    "cross_border_wire_vol":"Cross-Border Wire Volume (CAD)","structuring_flag":"Structuring Flag",
    "iforest_score":"Anomaly Score (IsoForest)","days_since_onboard":"Account Age (Days)",
    "occupation_risk":"Occupation Risk","industry_risk":"Industry Risk",
}
def hname(f): return FEATURE_NAMES.get(f, f.replace("_"," ").title())

def parse_c(raw):
    if isinstance(raw, dict): return raw
    try: return json.loads(raw)
    except: pass
    try: return ast.literal_eval(raw)
    except: return {}

def plot_customer(cid, score, contribs, top_n=10):
    if not contribs: return
    items = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    feats = [hname(k) for k,v in items][::-1]
    vals  = [v for k,v in items][::-1]
    colors = ["#C0392B" if v>0 else "#2980B9" for v in vals]
    fig, ax = plt.subplots(figsize=(10, max(5, len(feats)*0.55+1.5)))
    fig.patch.set_facecolor("#0F1923"); ax.set_facecolor("#0F1923")
    ax.barh(feats, vals, color=colors, edgecolor="none", height=0.65)
    ax.axvline(0, color="#FFFFFF30", linewidth=1.2)
    rng = max(vals)-min(vals) if (max(vals)-min(vals))>0 else 0.01
    for i,(f,v) in enumerate(zip(feats,vals)):
        off = 0.002*rng; ha = "left" if v>=0 else "right"
        ax.text(v+(off if v>=0 else -off), i, f"{v:+.4f}", va="center", ha=ha,
                fontsize=8, color="#FFFFFF99", fontfamily="monospace")
    ax.set_xlabel("SHAP Contribution to Risk Score", color="#FFFFFFBB", fontsize=10)
    ax.tick_params(colors="#FFFFFFBB", labelsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
    rc = "#C0392B" if score>=0.7 else ("#E67E22" if score>=0.4 else "#27AE60")
    fig.suptitle(f"Customer {cid}  |  Risk Score: {score:.3f}", color=rc, fontsize=14, fontweight="bold", y=0.98)
    ax.set_title("Feature Contributions â€” What Drove This Risk Score", color="#FFFFFFAA", fontsize=10, pad=8)
    ax.legend(handles=[mpatches.Patch(color="#C0392B",label="Increases Risk"),
                        mpatches.Patch(color="#2980B9",label="Decreases Risk")],
              loc="lower right", facecolor="#1A2634", edgecolor="#FFFFFF30", labelcolor="white", fontsize=9)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"shap_{cid}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close()
    return fname

df = pd.read_csv(EXPLAIN_CSV)
cid_col = "customer_id" if "customer_id" in df.columns else df.columns[0]
score_col = next((c for c in ["risk_score","blended_score","xgb_score","score"] if c in df.columns), df.select_dtypes("float").columns[0])
contrib_col = next((c for c in ["feature_contributions","shap_contributions","contribs","contributions"] if c in df.columns), None)
shap_cols = [c for c in df.columns if c.startswith("shap_")]

saved = []
for _, row in df.iterrows():
    cid = row[cid_col]; score = float(row[score_col])
    if contrib_col:
        contribs = parse_c(row[contrib_col])
    elif shap_cols:
        contribs = {c.replace("shap_",""):float(row[c]) for c in shap_cols if pd.notna(row[c])}
    else:
        contribs = {}
    f = plot_customer(cid, score, contribs)
    if f: saved.append(f); print(f"  chart: {f}")

print(f"\n[DONE] {len(saved)} SHAP charts -> {OUT_DIR}/")
