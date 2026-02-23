import os, json, ast, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import base64

# Auto-detect the most recent explanation file
def find_latest_explanation_file():
    # Look for explanation files in various locations
    patterns = [
        "outputs/runs/**/*explanations*.csv",
        "outputs/runs/**/*final*.csv",
        "outputs/runs/**/test_*.csv",
        "outputs/**/explanations*.csv"
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    if files:
        # Return the most recent file
        return max(files, key=os.path.getctime)
    return None

# Find the explanation file
EXPLAIN_CSV = find_latest_explanation_file()
if EXPLAIN_CSV:
    print(f"Found explanation file: {EXPLAIN_CSV}")
else:
    print("No explanation file found. Looking for ranked customers file...")
    # Look for ranked customers file
    ranked_patterns = [
        "outputs/runs/**/*ranked*.csv",
        "outputs/runs/**/test_*.csv"
    ]
    for pattern in ranked_patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            EXPLAIN_CSV = max(files, key=os.path.getctime)
            print(f"Using ranked file: {EXPLAIN_CSV}")
            break

if not EXPLAIN_CSV:
    raise Exception("No suitable input file found! Please check your outputs/runs/ directory.")

# Create output directories
os.makedirs("outputs/viz/shap_charts", exist_ok=True)
os.makedirs("outputs/viz/borderline_cases", exist_ok=True)
os.makedirs("outputs/viz/sensitivity", exist_ok=True)

print(f"\nProcessing file: {EXPLAIN_CSV}")
df = pd.read_csv(EXPLAIN_CSV)
print(f"Columns found: {list(df.columns)}")
print(f"Shape: {df.shape}")

# Rest of your visualization code here...
# (Copy the rest of the Python code from your original script)
