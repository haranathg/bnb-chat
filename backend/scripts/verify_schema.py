# verify_schema.py
import os
from dotenv import load_dotenv
import yaml
import pandas as pd
from sqlalchemy import create_engine, text

# --- Load .env from project root ---
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if not os.path.exists(env_path):
    raise FileNotFoundError(f"❌ .env file not found at expected path: {env_path}")

load_dotenv(dotenv_path=env_path)

NEON_DB_URI = os.getenv("NEON_DB_URI")
if not NEON_DB_URI:
    raise RuntimeError("❌ NEON_DB_URI not loaded. Check your .env content or syntax.")

print(f"✅ Loaded NEON_DB_URI from {env_path}\n→ {NEON_DB_URI[:60]}...")  # just to verify

engine = create_engine(NEON_DB_URI)

with open("scripts/semantic_schema.yaml") as f:
    spec = yaml.safe_load(f)

problems = []
for t in spec["tables"]:
    tbl = t["name"]
    declared = {c["name"] for c in t["columns"]}
    q = text("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = :tbl
    """)
    with engine.begin() as conn:
        actual = {r[0] for r in conn.execute(q, {"tbl": tbl}).fetchall()}
    missing = declared - actual
    extra   = actual - declared
    if missing or extra:
        problems.append({"table": tbl, "missing": sorted(missing), "extra": sorted(extra)})

df = pd.DataFrame(problems)
print(df if not df.empty else "✅ YAML matches live DB columns.")