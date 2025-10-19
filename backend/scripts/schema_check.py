# scripts/check_schema.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect

load_dotenv()
engine = create_engine(os.getenv("NEON_DB_URI"))
inspector = inspect(engine)

for table in ["drug_class", "awp_history", "asp_history", "drug_master"]:
    print(f"\n🧱 Table: {table}")
    for column in inspector.get_columns(table):
        print(f"  - {column['name']}")