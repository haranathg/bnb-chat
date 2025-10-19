import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load Neon DB connection string
load_dotenv()
DB_URI = os.getenv("NEON_DB_URI")
if not DB_URI:
    raise ValueError("❌ Missing NEON_DB_URI in .env")

engine = create_engine(DB_URI)

create_table_sql = """
CREATE TABLE IF NOT EXISTS drug_master (
    hcpcs_code                      VARCHAR(40),
    brand_name                      VARCHAR(80),
    concat_label                    VARCHAR(100),
    manufacturer                    VARCHAR(100),
    asp_per_unit_previous_quarter   NUMERIC(18,6),
    asp_per_unit_current_quarter    NUMERIC(18,6),
    medicare_payment_limit          NUMERIC(18,6),
    asp_quarterly_change_pct        NUMERIC(10,6),
    median_wac_per_hcpcs_unit       NUMERIC(18,6),
    median_awp_per_hcpcs_unit       NUMERIC(18,6),
    wac_awp_last_change             VARCHAR(25),
    asp_wac_ratio                   NUMERIC(12,8),
    asp_awp_ratio                   NUMERIC(12,8)
);
"""

with engine.connect() as conn:
    print("🔧 Recreating table drug_master with wider precision...")
    conn.execute(text("DROP TABLE IF EXISTS drug_master;"))
    conn.execute(text(create_table_sql))
    conn.commit()

print("✅ Table drug_master recreated successfully with expanded numeric precision.")