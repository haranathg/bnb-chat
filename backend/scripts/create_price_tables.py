import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("NEON_DB_URI")
if not DB_URI:
    raise ValueError("❌ Missing NEON_DB_URI in .env")

engine = create_engine(DB_URI)

with engine.connect() as conn:
    print("🔧 Recreating AWP/WAC/ASP history tables...")

    conn.execute(text("""
    DROP TABLE IF EXISTS awp_history;
    CREATE TABLE awp_history (
        hcpcs_code VARCHAR(100),
        median_awp NUMERIC(18,6),
        quarter_label VARCHAR(10)
    );
    """))

    conn.execute(text("""
    DROP TABLE IF EXISTS wac_history;
    CREATE TABLE wac_history (
        hcpcs_code VARCHAR(100),
        median_wac NUMERIC(18,6),
        quarter_label VARCHAR(10)
    );
    """))

    conn.execute(text("""
    DROP TABLE IF EXISTS asp_history;
    CREATE TABLE asp_history (
        hcpcs_code VARCHAR(100),
        asp_value NUMERIC(18,6),
        quarter_label VARCHAR(10)
    );
    """))

    conn.commit()

print("✅ All history tables recreated with wider string and numeric limits.")