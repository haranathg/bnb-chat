# scripts/create_view.py
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("NEON_DB_URI"))

view_sql = """
CREATE OR REPLACE VIEW drug_pricing_combined AS 
SELECT
    m.hcpcs_code,
    m.brand_name,
    m.manufacturer,
    c.drug_class,
    c.drug_class_2,
    m.asp_curr_qtr,
    m.asp_prev_qtr,
    m.asp_qtr_change_pct,
    m.medicare_payment_limit,
    m.median_wac AS wac_current,
    m.median_awp AS awp_current,
    m.asp_wac_ratio,
    m.asp_awp_ratio,
    a.asp_value AS asp_hist,
    a.quarter_label AS asp_quarter,
    w.median_wac AS wac_hist,
    aw.median_awp AS awp_hist
FROM drug_master m
LEFT JOIN drug_class c USING (hcpcs_code)
LEFT JOIN asp_history a USING (hcpcs_code)
LEFT JOIN wac_history w USING (hcpcs_code, quarter_label)
LEFT JOIN awp_history aw USING (hcpcs_code, quarter_label);
"""

with engine.begin() as conn:
    conn.execute(text(view_sql))
print("✅ View `drug_pricing_combined` created successfully.")