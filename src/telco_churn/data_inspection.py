import sys
import hashlib
import json
from pathlib import Path
import pandas as pd
from importlib.metadata import version

def main():
    print("Starting data inspection...")

    data_path = Path("data/raw/telco.csv")
    if not data_path.exists():
        print(f"ERROR: File {data_path} not found. Please check the data location.")
        return
    source_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    df = pd.read_csv(data_path)
    report = {"source_file": str(data_path),
        "sha256_hash": source_hash,
        "environment": {
            "python_version": sys.version.split()[0],
            "pandas_version": version("pandas")
        },
        "dataset_shape": {
            "rows": df.shape[0], 
            "columns": df.shape[1]
        },
        "column_types": df.dtypes.astype(str).to_dict(),
        "target_counts": df["Churn"].value_counts(dropna=False).to_dict(),
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),}
    empty_or_whitespace = {}
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            stripped = df[col].astype("string").str.strip()
            count = stripped.eq("").sum()
            if count > 0:
                empty_or_whitespace[col] = int(count)
    report["empty_or_whitespace_values"] = empty_or_whitespace
    output_path = Path("reports/data_overview.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"SUCCESS: Inspection complete! Report saved to: {output_path}")

if __name__ == "__main__":
    main()


