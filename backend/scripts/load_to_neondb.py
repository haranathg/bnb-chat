# scripts/load_to_neondb.py
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

load_dotenv()
NEON_DB_URI = os.getenv("NEON_DB_URI")
engine = create_engine(NEON_DB_URI)

FILES = {
    "drug_master": "data/Master.xlsx",
    "drug_class": "data/Drug Class.xlsx",
    "asp_history": "data/Historical ASP File.xlsx",
    "awp_history": "data/Historical AWP.xlsx",
    "wac_history": "data/Historical WAC.xlsx"
}

def clean_master(df):
    # Standardize column names: lowercase and underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map Excel columns to DB column names
    '''rename_map = {
        "concat": "concat_label",
        "asp_per_unit_previous_quarter": "asp_prev_qtr",
        "asp_per_unit_current_quarter": "asp_curr_qtr",
        "asp_quarterly_change_%": "asp_qtr_change_pct",
        "median_wac_per_hcpcs_unit": "median_wac",
        "median_awp_per_hcpcs_unit": "median_awp",
        "wac/awp_(last_change)": "wac_awp_last_change",
        "asp/wac_ratio": "asp_wac_ratio",
        "asp/awp_ratio": "asp_awp_ratio"
    }'''

    #df.rename(columns=rename_map, inplace=True)

    # Convert percentage strings (e.g., "2.7%") → decimal 0.027
    if "asp_qtr_change_pct" in df.columns:
        df["asp_qtr_change_pct"] = (
            df["asp_qtr_change_pct"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
            / 100
        )

    # Convert date column to datetime
    if "wac_awp_last_change" in df.columns:
        df["wac_awp_last_change"] = pd.to_datetime(df["wac_awp_last_change"], errors="coerce")

    return df

def clean_generic(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

LOADERS = {
    "drug_master": clean_master,
    "drug_class": clean_generic,
    "asp_history": clean_generic,
    "awp_history": clean_generic,
    "wac_history": clean_generic
}

for table, path in FILES.items():
    print(f"Reloading {table} from {path}...")
    df = pd.read_excel(path)
    df = LOADERS[table](df)

    with engine.begin() as conn:
        # ✅ wrap raw SQL in text()
        conn.execute(text(f'TRUNCATE TABLE "{table}";'))
        # Rename columns in df to match Neon table
        df.to_sql(table, conn, index=False, if_exists="append")

print("✅ All tables refreshed in NeonDB.")