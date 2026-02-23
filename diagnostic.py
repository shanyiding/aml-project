import os
import pandas as pd
import glob

print("=== AML Project Diagnostic ===\n")

# Check output directories
print("Checking output directories:")
for dir_path in ["outputs/runs", "outputs/models", "outputs"]:
    if os.path.exists(dir_path):
        print(f"  ✅ {dir_path} exists")
        # Show recent files
        files = glob.glob(f"{dir_path}/**/*.csv", recursive=True)[:5]
        if files:
            print(f"    Recent CSV files:")
            for f in files:
                print(f"      - {f}")
    else:
        print(f"  ❌ {dir_path} does not exist")

# Try to find any CSV with customer data
print("\nSearching for CSV files with customer data:")
csv_files = glob.glob("outputs/**/*.csv", recursive=True)
for csv_file in csv_files[:10]:  # Check first 10 files
    try:
        df = pd.read_csv(csv_file, nrows=5)
        print(f"\nFile: {csv_file}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        # Check for key columns
        has_customer = any('customer' in str(col).lower() for col in df.columns)
        has_score = any(col in ['risk_score', 'score', 'blended_score'] for col in df.columns)
        print(f"  Has customer_id: {has_customer}")
        print(f"  Has risk_score: {has_score}")
    except Exception as e:
        print(f"  Error reading {csv_file}: {e}")

print("\n=== Diagnostic Complete ===")
