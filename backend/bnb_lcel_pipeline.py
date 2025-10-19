"""
bnb_lcel_pipeline.py
--------------------------------
Self-contained LangChain LCEL data assistant.
Performs:
  1️⃣ Natural language → SQL generation
  2️⃣ SQL execution (via NeonDB)
  3️⃣ LLM-driven insight summarization
  4️⃣ Visualization + chart auto-open
"""

import os
import re
import platform
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# --- LangChain imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableSequence, RunnableLambda
from bnb_retriever import BnbRetriever

# =============================================================================
#  Environment setup
# =============================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEON_DB_URI = os.getenv("NEON_DB_URI")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in environment or .env file.")
if not NEON_DB_URI:
    raise ValueError("❌ Missing NEON_DB_URI in environment or .env file.")

# Initialize LLM + SQL engine
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
engine = create_engine(NEON_DB_URI)

retriever = BnbRetriever(top_k=5)  # already created index

# =============================================================================
#  Core SQL Agent Functions
# =============================================================================
def generate_sql(question: str) -> str:
    # 1️⃣ Get semantic schema context from Pinecone retriever
    schema_context = retriever.as_prompt_context(question)

    # 2️⃣ Include that context in the LLM prompt
    sql_prompt = (
        "You are a SQL expert for healthcare drug pricing data working with Postgres.\n"
        "Use the database schema context below to generate valid SQL that runs without errors.\n"
        "Rules you must follow:\n"
        "- Only reference tables and columns that appear in the schema context.\n"
        "- If you need to combine window functions with aggregation, first compute the window results in a CTE or subquery, and aggregate in an outer query (Postgres does not allow aggregating directly over a window function result).\n"
        "- When a question references higher-level groupings (e.g., drug classes, categories, therapeutics), join the relevant dimension tables (such as drug_class) so those fields can be selected or grouped.\n"
        "- Prefer explicit column names (e.g., quarter_label, asp_value) and include necessary joins using the provided relationships.\n"
        "- When filtering based on statistics derived from window functions (e.g., comparing to class means or standard deviations), compute any boolean flags or thresholds inside the CTE/subquery where the statistics exist, and filter on those flags in the outer query rather than referencing window columns directly in HAVING clauses.\n"
        "- Do not reference SELECT aliases directly in WHERE or HAVING clauses; if you need to filter by a derived value (including CASE expressions), compute it inside a CTE/subquery and apply the filter there, or wrap the SELECT in an outer query and filter on the alias from that outer scope.\n"
        "- Filter out NULL or zero-like records when they would distort averages or standard deviations (e.g., ignore ASP values that are NULL).\n"
        "- Limit results sensibly (e.g., LIMIT 10) when returning ranked lists.\n\n"
        f"Schema context:\n{schema_context}\n\n"
        f"Question: {question}\n\n"
        "Return ONLY the SQL query, no markdown or explanations."
    )

    response = llm.invoke([HumanMessage(content=sql_prompt)])
    sql_query = clean_sql(response.content)
    print("\n🧠 Generated SQL:\n", sql_query)
    return sql_query

def clean_sql(sql: str) -> str:
    """
    Remove stray prefixes or markdown artifacts from LLM-generated SQL.
    """
    # Common artifacts: 'sql', '```sql', '```', 'SQL', etc.
    sql = re.sub(r"(?i)^```sql", "", sql.strip())
    sql = re.sub(r"(?i)^```", "", sql)
    sql = re.sub(r"(?i)^sql\s*", "", sql)   # remove leading "sql" word
    sql = re.sub(r"```$", "", sql.strip())
    return sql.strip()

def execute_sql(sql: str) -> pd.DataFrame:
    """Executes the generated SQL on NeonDB and returns a DataFrame."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        print(f"📊 Query executed successfully. Rows returned: {len(df)}")
        return df
    except Exception as e:
        print(f"⚠️ SQL execution failed: {e}")
        return pd.DataFrame()

def safe_execute_sql(question, sql):
    df = execute_sql(sql)
    if df.empty:
        print("🤖 No data found — revalidating SQL structure with LLM...")
        fixed_sql_prompt = (
            "You generated the following SQL, but it caused an error or returned no data:\n"
            f"{sql}\n\n"
            f"User question: {question}\n"
            "Please correct any issues and return valid Postgres SQL that adheres to these rules:\n"
            "- Only reference tables/columns present in the schema context of the original query.\n"
            "- If you mix window functions with aggregation, compute the window values in a CTE/subquery first, then aggregate in an outer query (Postgres restriction).\n"
            "- When filtering by derived values (including CASE results or window-derived statistics), compute the filter flag inside the CTE/subquery or wrap the SELECT in an outer query and filter there; do not reference SELECT aliases directly in WHERE/HAVING.\n"
            "- Join the necessary dimension tables when the question references higher-level groupings (e.g., include drug_class when talking about classes).\n"
            "- Filter out NULL/zero-like metric values when they would distort the results (e.g., ignore NULL ASPs).\n"
            "- Limit results sensibly when returning ranked lists.\n"
            "Return only the corrected SQL query without markdown."
        )
        fixed_sql = clean_sql(llm.invoke([HumanMessage(content=fixed_sql_prompt)]).content)
        print(f"🔁 Retrying with fixed SQL:\n{fixed_sql}")
        df = execute_sql(fixed_sql)
    return df


# =============================================================================
#  Visualization Helpers
# =============================================================================
def visualize_results(df: pd.DataFrame, question: str, summary: str):
    """Enhanced contextual chart with summary overlay."""
    if df.empty:
        print("⚠️ No data to visualize.")
        return

    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(12, 7))
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(exclude="number").columns.tolist()

    plt.suptitle(f"Insight Visualization: {question}", fontsize=14, fontweight="bold")

    if len(numeric_cols) == 1 and len(text_cols) >= 1:
        x, y = text_cols[0], numeric_cols[0]
        df_sorted = df.sort_values(by=y, ascending=False).head(10)
        palette = sns.color_palette("crest", n_colors=len(df_sorted))
        sns.barplot(data=df_sorted, x=x, y=y, hue=x, dodge=False, palette=palette, legend=False)
        plt.legend([], [], frameon=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(y)
        plt.title(f"Top 10 by {y}")
        for idx, val in enumerate(df_sorted[y]):
            plt.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    elif len(numeric_cols) >= 2:
        sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1],
                        hue=text_cols[0] if text_cols else None, s=100, palette="viridis")
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
    else:
        plt.axis("off")
        plt.table(cellText=df.head(10).values, colLabels=df.columns, loc="center")

    plt.figtext(0.5, 0.02, f"Summary: {summary[:400]}...", wrap=True, ha="center",
                fontsize=10, color="dimgray")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("last_result.png", dpi=300)
    print("📈 Chart saved as: last_result.png")
    open_chart_image("last_result.png")


def open_chart_image(image_path="last_result.png"):
    """Open chart image cross-platform."""
    if not os.path.exists(image_path):
        print(f"⚠️ Chart not found: {image_path}")
        return
    print(f"🖼️ Opening chart: {image_path}")
    if platform.system() == "Darwin":
        subprocess.run(["open", image_path])
    elif platform.system() == "Windows":
        os.startfile(image_path)
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", image_path])
    else:
        print("⚠️ Unsupported OS — open manually.")


# =============================================================================
#  Summarizer Stage
# =============================================================================
summary_prompt = PromptTemplate(
    input_variables=["question", "data"],
    template=(
        "You are a healthcare market analyst.\n"
        "Question: {question}\n"
        "Data (top 10 rows):\n{data}\n\n"
        "Summarize the main insights, patterns, and any outliers clearly."
    ),
)

def summarize_stage(inputs):
    question = inputs["question"]
    df = inputs["df"]

    # ✅ Strict grounding: no data = no LLM
    if df is None or df.empty:
        message = (
            f"Hmm 🤔 I didn’t find any data returned for your question:\n"
            f"“{question}”\n\n"
            "Please verify the SQL query or contact the data admin for support."
        )
        print("\n⚠️ No data available to summarize.")
        return message

    # ✅ Otherwise, summarize real data only
    data_str = df.head(10).to_markdown(index=False)
    formatted = summary_prompt.format(question=question, data=data_str)
    response = llm.invoke([HumanMessage(content=formatted)])
    summary = response.content.strip()
    print("\n🧾 Insight Summary:\n", summary)
    visualize_results(df, question, summary)
    return summary

# =============================================================================
#  LCEL Pipeline
# =============================================================================
bnb_chain = RunnableSequence(
    RunnableLambda(lambda q: {"question": q})
    | RunnableLambda(lambda d: {"question": d["question"], "sql": clean_sql(generate_sql(d["question"]))})
    | RunnableLambda(lambda d: {"question": d["question"], "df": safe_execute_sql(d["question"], d["sql"])})
    | RunnableLambda(summarize_stage)
)


# =============================================================================
#  CLI Entrypoint
# =============================================================================
if __name__ == "__main__":
    print("💡 BNB LCEL Data Assistant Ready!")
    question = input("💬 Ask your data question: ")
    result = bnb_chain.invoke(question)
    print("\n✅ Final Output:\n", result)
