import os
from pinecone import Pinecone

from dotenv import load_dotenv
load_dotenv()  # <-- This line is critical

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

api_key = os.getenv("PINECONE_API_KEY")
print("🔑 Pinecone API key found:", bool(api_key))

pc = Pinecone(api_key=api_key)
index = pc.Index(os.getenv("PINECONE_INDEX", "bnb_pricing_schema"))

# Fetch 3 sample items
sample = index.query(vector=[0]*3072, top_k=3, include_metadata=True)
print(sample)