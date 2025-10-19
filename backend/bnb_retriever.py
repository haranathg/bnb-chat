"""
bnb_retriever.py

Retriever that connects OpenAI, Pinecone, and NeonDB to support
semantic schema lookup and SQL generation for "Chat with Data".

Usage:
    from bnb_retriever import BnbRetriever
    retriever = BnbRetriever()
    retriever.find_related_columns("asp trend for anticoagulants")
"""
import os
import logging
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import pandas as pd
from collections import defaultdict
from pprint import pprint
import yaml

# --- Load environment ---
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "bnb-pricing-schema")

if not all([OPENAI_API_KEY, PINECONE_API_KEY]):
    raise RuntimeError("❌ Missing required environment variables (.env)")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


class BnbRetriever:
    """Retriever for schema + relationship context from Pinecone."""

    def __init__(self, top_k: int = 5, debug: bool = False):
        self.top_k = top_k
        self.model = "text-embedding-3-large"
        env_debug = os.getenv("BNB_RETRIEVER_DEBUG", "").lower() in {"1", "true", "yes", "on"}
        self.debug = debug or env_debug
        self.logger = logging.getLogger(__name__)
        self.schema = self._load_semantic_schema()

    def _debug(self, message: str, payload=None):
        if not self.debug:
            return
        if message:
            print(message)
        if payload is not None:
            pprint(payload)

    def _load_semantic_schema(self):
        """Preload the semantic schema so we can expand full table context."""
        schema_path = os.path.join(os.path.dirname(__file__), "semantic_schema.yaml")
        if not os.path.exists(schema_path):
            self.logger.warning("semantic_schema.yaml not found at %s", schema_path)
            return {"tables": {}, "relationships": []}

        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning("Failed to load semantic schema: %s", exc)
            return {"tables": {}, "relationships": []}

        tables = {}
        for table in raw.get("tables", []):
            tables[table["name"]] = {
                "description": table.get("description", ""),
                "columns": table.get("columns", []),
            }
        return {"tables": tables, "relationships": raw.get("relationships", [])}

    def embed(self, text: str):
        emb = client.embeddings.create(model=self.model, input=text)
        return emb.data[0].embedding

    def retrieve(self, query: str):
        """Retrieve top_k most relevant schema elements for the user query."""
        self._debug(f"[Retriever] Starting retrieval for query: {query}")
        self._debug(f"[Retriever] Using embedding model: {self.model}")
        qvec = self.embed(query)
        results = index.query(vector=qvec, top_k=self.top_k * 3, include_metadata=True)

        raw_matches = results.get("matches", []) if isinstance(results, dict) else results.matches
        self._debug(f"[Retriever] Pinecone returned {len(raw_matches)} raw matches.")
        if self.debug:
            for idx, match in enumerate(raw_matches):
                meta = match.get("metadata", {})
                self._debug(
                    f"[Retriever] Raw match {idx}: score={match.get('score')}, metadata keys={list(meta.keys())}",
                    meta,
                )

        # ✅ boost relationships slightly (join context)
        matches = []
        for match in raw_matches:
            meta = match["metadata"]
            score = match["score"]
            if meta.get("type") == "relationship":
                score += 0.05  # light boost for join context
            matches.append({
                "table": meta.get("table"),
                "column": meta.get("column"),
                "type": meta.get("type"),
                "description": meta.get("desc"),
                "score": round(score, 3),
                "metadata": dict(meta)
            })

        self._debug("[Retriever] Normalized matches before sorting:", matches)

        # Sort and trim to top_k
        matches = sorted(matches, key=lambda x: x["score"], reverse=True)[: self.top_k]
        self._debug(f"[Retriever] Top {self.top_k} matches after sorting:", matches)

        # ✅ Always include at least one descriptive column like 'name' or 'description'
        important_cols = ["name", "description", "label"]
        desc_results = index.query(vector=self.embed("drug name or description"),
                                   top_k=5, include_metadata=True)
        desc_matches = desc_results.get("matches", []) if isinstance(desc_results, dict) else desc_results.matches
        self._debug(f"[Retriever] Descriptive column search returned {len(desc_matches)} matches.")
        if self.debug:
            for idx, match in enumerate(desc_matches):
                meta = match.get("metadata", {})
                self._debug(
                    f"[Retriever] Descriptive match {idx}: score={match.get('score')}, metadata keys={list(meta.keys())}",
                    meta,
                )
        for match in desc_matches:
            meta = match["metadata"]
            if any(k in meta.get("column", "").lower() for k in important_cols):
                # add if not already in list
                if not any(m["table"] == meta.get("table") and m["column"] == meta.get("column") for m in matches):
                    matches.append({
                        "table": meta.get("table"),
                        "column": meta.get("column"),
                        "type": meta.get("type"),
                        "description": meta.get("desc"),
                        "score": 0.999,
                        "metadata": dict(meta)
                    })

        self._debug("[Retriever] Final match list being returned:", matches)

        df = pd.DataFrame(matches)
        self._debug(f"📊 Top {len(matches)} matches for query: '{query}'")
        if self.debug:
            self._debug("", df[["table", "column", "type", "score"]])
        return matches

    def as_prompt_context(self, query: str) -> str:
        matches = self.retrieve(query)
        self._debug("[Retriever] Building prompt context from matches:", matches)
        tables = defaultdict(lambda: {"columns": set(), "matches": []})
        relationships = []
        relationship_tables = set()

        # Group columns by table and collect relationship descriptors
        for m in matches:
            meta = m.get("metadata") or {}
            table = m.get("table") or meta.get("table") or "unknown_table"
            column = m.get("column") or meta.get("column") or "unknown_column"
            inferred_type = m.get("type") or meta.get("type", "")

            if (m.get("type") or meta.get("type")) == "relationship":
                rel_description = m.get("description") or meta.get("desc") or f"{table}: {column}"
                relationships.append(rel_description)
                if table and table != "unknown_table":
                    for part in table.split("→"):
                        clean = part.strip()
                        if clean:
                            relationship_tables.add(clean)
                continue

            if table == "unknown_table" or column == "unknown_column":
                self._debug(f"[Retriever] Warning: Missing metadata values in match: {m}")

            tables[table]["columns"].add(column)
            tables[table]["matches"].append({
                "name": column,
                "type": inferred_type,
                "description": m.get("description") or meta.get("desc", ""),
                "score": m.get("score")
            })

        # Build readable schema context with full table metadata
        context_blocks = []
        schema_tables = self.schema.get("tables", {}) if self.schema else {}
        # ensure relationship tables are included even if no direct column match
        for rel_table in relationship_tables:
            _ = tables[rel_table]  # initialize default entry if missing

        for table, data in tables.items():
            table_schema = schema_tables.get(table, {})
            column_lines = []

            if table_schema.get("columns"):
                for col_meta in table_schema["columns"]:
                    col_name = col_meta.get("name", "")
                    col_type = col_meta.get("type", "")
                    col_desc = col_meta.get("description", "")
                    highlight = " (matched)" if col_name in data["columns"] else ""
                    line = f"- {col_name} ({col_type}){highlight}"
                    if col_desc:
                        line += f": {col_desc}"
                    column_lines.append(line)
            else:
                for col in sorted(data["columns"]):
                    column_lines.append(f"- {col}")

            block_lines = [
                f"Table: {table}",
                f"Description: {table_schema.get('description', 'No description available.')}",
                "Columns:",
                "\n".join(column_lines) if column_lines else "- (no column metadata found)",
            ]

            if data["matches"]:
                block_lines.append("Relevant matches:")
                for match in data["matches"]:
                    block_lines.append(
                        f"  • {match['name']} ({match['type']}) score={match['score']}: {match['description']}"
                    )

            context_blocks.append("\n".join(block_lines))

        if relationships:
            context_blocks.append("Relationships:\n- " + "\n- ".join(relationships))

        return "\n\n".join(context_blocks)

if __name__ == "__main__":
    retriever = BnbRetriever(top_k=5)
    context = retriever.as_prompt_context("average asp per unit by drug class")
    print("\n🧠 Context for LLM:\n", context)
