import pandas as pd
from pathlib import Path

# Path to your data folder
data_dir = Path("data")

# List all your Excel files
files = [
    "Master.xlsx",
    "Drug Class.xlsx",
    "Historical ASP File.xlsx",
    "Historical AWP.xlsx",
    "Historical WAC.xlsx"
]

for file in files:
    path = data_dir / file
    print(f"\n🔍 {file}")
    df = pd.read_excel(path)

    # For each column, compute max string length
    for col in df.columns:
        if df[col].dtype == "object":  # Only for text columns
            max_len = df[col].astype(str).map(len).max()
            print(f"  {col:35}  →  max length = {max_len}")