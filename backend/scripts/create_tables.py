# scripts/create_tables.py
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
NEON_DB_URI = os.getenv("NEON_DB_URI")
engine = create_engine(NEON_DB_URI)

schema_sql = """
CREATE TABLE IF NOT EXISTS drug_master (
    hcpcs_code VARCHAR(10) PRIMARY KEY,
    brand_name TEXT,
    concat_label TEXT,
    manufacturer TEXT,
    asp_prev_qtr NUMERIC(10,3),
    asp_curr_qtr NUMERIC(10,3),
    medicare_payment_limit NUMERIC(10,3),
    asp_qtr_change_pct NUMERIC(6,3),
    median_wac NUMERIC(10,3),
    median_awp NUMERIC(10,3),
    wac_awp_last_change DATE,
    asp_wac_ratio NUMERIC(6,3),
    asp_awp_ratio NUMERIC(6,3)
);

CREATE TABLE IF NOT EXISTS drug_class (
    hcpcs_code VARCHAR(10),
    drug_class TEXT,
    drug_class_2 TEXT
);

CREATE TABLE IF NOT EXISTS asp_history (
    hcpcs_code VARCHAR(10),
    asp_value NUMERIC(12,6),
    quarter_label VARCHAR(10)
);

CREATE TABLE IF NOT EXISTS awp_history (
    hcpcs_code VARCHAR(10),
    median_awp NUMERIC(10,2),
    quarter_label VARCHAR(10)
);

CREATE TABLE IF NOT EXISTS wac_history (
    hcpcs_code VARCHAR(10),
    median_wac NUMERIC(10,2),
    quarter_label VARCHAR(10)
);
"""

with engine.begin() as conn:
    for statement in schema_sql.strip().split(";"):
        if statement.strip():
            conn.execute(text(statement))
print("✅ Tables created successfully in NeonDB.")