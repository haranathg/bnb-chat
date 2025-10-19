from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://neondb_owner:npg_3Lml6xpzXCFr@ep-damp-leaf-adg9onrl-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

with engine.begin() as conn:
    result = conn.execute(text("SELECT 'Connected to NeonDB!'")).scalar()
    print(result)