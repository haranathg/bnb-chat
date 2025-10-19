import os
import yaml
import pandas as pd
import re
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# --- Load environment variables ---
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

NEON_DB_URI = os.getenv("NEON_DB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "bnb-pricing-schema")

if not all([NEON_DB_URI, OPENAI_API_KEY, PINECONE_API_KEY]):
    raise RuntimeError("❌ Missing required environment variables in .env")

# --- Configurable flags ---
REBUILD_INDEX = True      # ✅ set to False for incremental updates
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072      # for text-embedding-3-large

def make_safe_id(raw: str) -> str:
    """Convert a string into a Pinecone-safe ASCII ID."""
    safe = (
        raw.replace("→", "_to_")
           .replace(" ", "_")
           .replace("::", "_")
           .replace("/", "_")
           .lower()
    )
    # remove any characters outside alphanumerics, _, -, :
    safe = re.sub(r"[^a-z0-9_\-:]", "", safe)
    return safe


# --- Initialize clients ---
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
engine = create_engine(NEON_DB_URI)
insp = inspect(engine)

# --- Pinecone index setup ---
if REBUILD_INDEX:
    existing_indexes = [i.name for i in pc.list_indexes().indexes]
    if PINECONE_INDEX in existing_indexes:
        print(f"🧹 Deleting existing Pinecone index '{PINECONE_INDEX}'...")
        pc.delete_index(PINECONE_INDEX)
    print(f"📦 Creating new Pinecone index '{PINECONE_INDEX}' (dim={EMBEDDING_DIM})...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX)

# --- Load semantic schema YAML ---
semantic_path = os.path.join(os.path.dirname(__file__), "..", "semantic_schema.yaml")
semantic = None
if os.path.exists(semantic_path):
    with open(semantic_path, "r") as f:
        semantic = yaml.safe_load(f)
    print(f"✅ Loaded semantic schema with {len(semantic.get('tables', []))} tables.")
else:
    print("⚠️ No semantic_schema.yaml found — using live schema only.")

# --- Gather live schema ---
live_schema = []
for table_name in insp.get_table_names():
    for col in insp.get_columns(table_name):
        live_schema.append({
            "table": table_name,
            "column": col["name"],
            "type": str(col["type"]),
            "desc": f"Column '{col['name']}' in table '{table_name}' (type: {col['type']})"
        })

# --- Merge semantic and relationship metadata ---
if semantic:
    for table in semantic.get("tables", []):
        for col in table.get("columns", []):
            live_schema.append({
                "table": table["name"],
                "column": col["name"],
                "type": col.get("type", ""),
                "desc": f"Column '{col['name']}' in table '{table['name']}' ({col.get('description', '')})"
            })
    for rel in semantic.get("relationships", []):
        live_schema.append({
            "table": f"{rel['from_table']} → {rel['to_table']}",
            "column": f"{rel['from_column']} → {rel['to_column']}",
            "type": "relationship",
            "desc": (
                f"Relationship: {rel['from_table']}.{rel['from_column']} → "
                f"{rel['to_table']}.{rel['to_column']} ({rel.get('description', '')})"
            )
        })

df = pd.DataFrame(live_schema).drop_duplicates(subset=["table", "column"]).reset_index(drop=True)
print(f"📊 Total schema elements to embed: {len(df)}")

# --- Embed and upsert ---
for _, row in tqdm(df.iterrows(), total=len(df), desc="🔗 Embedding"):
    text = f"Table: {row['table']} | Column: {row['column']} | Type: {row['type']} | Description: {row['desc']}"
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    vec = emb.data[0].embedding
    vec_id = make_safe_id(f"{row['table']}::{row['column']}")
    #vec_id = f"{row['table']}::{row['column']}"
    index.upsert(vectors=[{"id": vec_id, "values": vec, "metadata": row.to_dict()}])

print(f"✅ Vectorization complete. {len(df)} vectors synced to Pinecone index '{PINECONE_INDEX}'.")