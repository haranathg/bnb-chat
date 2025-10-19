import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
print("Loaded API key:", os.getenv("OPENAI_API_KEY")[:10], "...")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.models.list()
print("✅ Connected to OpenAI successfully.")