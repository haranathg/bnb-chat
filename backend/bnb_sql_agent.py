import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import subprocess

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from bnb_retriever import BnbRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableSequence, RunnableLambda

#from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# --- Load environment ---
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEON_DB_URI = os.getenv("NEON_DB_URI")

client = OpenAI(api_key=OPENAI_API_KEY)



'''
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

index_name = os.getenv("PINECONE_INDEX", "bnb_pricing_schema")

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
'''

class BnbSQLAgent:
    """
    A schema-aware SQL agent that uses the retriever for grounding.
    """

    def __init__(self):
        self.engine = create_engine(NEON_DB_URI)
        self.model = "gpt-4o-mini"

    def build_prompt(self, user_query: str, schema_context: str) -> str:
        """
        Build the prompt for the LLM, providing the schema context and user question.
        """
        return f"""
You are a SQL expert assistant.
Use the provided database schema context to write a valid SQL query for the question.

Schema Context:
{schema_context}

Rules:
- Use table and column names exactly as in the schema context.
- Do NOT invent or assume column names.
- Always filter out NULL numeric values when ordering or computing aggregates.
- If a column you think should exist isn’t shown, use a related one (e.g., description, label, or name fields from the same table).
- If joining tables, use the relationships provided in the context (e.g., hcpcs_code).
- Return only the SQL — no explanations, markdown, or code fences.

Question:
{user_query}
"""

    def generate_sql(self, user_query: str) -> str:
        """
        Use OpenAI to generate SQL for a natural language query.
        """
        schema_context = retriever.as_prompt_context(user_query)
        prompt = self.build_prompt(user_query, schema_context)

        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data expert that writes clean SQL queries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        sql = completion.choices[0].message.content.strip()

        # quick patch: fix ORDER BY alias arithmetic issue
        sql = re.sub(
            r'ORDER BY\s+([a-zA-Z_0-9]+)\s*-\s*([a-zA-Z_0-9]+)',
            r'ORDER BY (AVG(dm.\1) - AVG(dm.\2))',
            sql)

        print(f"🧠 Generated SQL:\n{sql}\n")
        return sql

    def execute_sql(self, sql: str):
        """
        Run the generated SQL on NeonDB and return results as a DataFrame.
        """
        import pandas as pd
        try:
            df = pd.read_sql(text(sql), self.engine)
            print(f"✅ Query executed successfully. Rows returned: {len(df)}")
            return df
        except Exception as e:
            print(f"⚠️ SQL execution failed: {e}")
            return None


def summarize_results(query: str, df: pd.DataFrame) -> str:
    # Convert DataFrame to a readable text table (truncate to avoid token explosion)
    table_preview = df.head(10).to_markdown(index=False)
    
    prompt = f"""
    The following data was returned from a SQL query that answers this question:

    Question: {query}

    Results (top 10 rows):
    {table_preview}

    Please summarize the key findings in natural language, focusing on trends, patterns,
    and what business insight can be inferred. Be concise and insightful.  Do not assume or
    make up anything. 
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4-turbo" if you prefer
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return completion.choices[0].message.content.strip()

def visualize_results(df, question: str, summary: str):
    """
    Creates a contextual visualization based on the data and question.
    Automatically chooses chart type and ties it to the summary narrative.
    """
    if df is None or df.empty:
        print("⚠️ No data to visualize.")
        return None

    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(12, 7))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(exclude="number").columns.tolist()

    title = f"Insight Visualization: {question}"
    plt.suptitle(title, fontsize=14, fontweight="bold")

    # --- Smart chart selection ---
    if len(numeric_cols) == 1 and len(text_cols) >= 1:
        # Bar chart for categorical vs numeric
        x = text_cols[0]
        y = numeric_cols[0]
        df_sorted = df.sort_values(by=y, ascending=False).head(10)
        palette = sns.color_palette("crest", n_colors=len(df_sorted))
        sns.barplot(data=df_sorted, x=x, y=y, palette=palette)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Top 10 by {y}")
        plt.ylabel(y)

        # Annotate bars
        for idx, val in enumerate(df_sorted[y]):
            plt.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    elif len(numeric_cols) >= 2:
        # Scatter plot for correlation
        sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1],
                        hue=text_cols[0] if text_cols else None, s=100, palette="viridis")
        plt.title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])

    else:
        # Fallback simple table visualization
        plt.axis("off")
        plt.table(cellText=df.head(10).values,
                  colLabels=df.columns,
                  loc="center")

    # --- Add narrative text as overlay ---
    plt.figtext(0.5, 0.02, f"Summary: {summary[:300]}...", wrap=True, ha="center", fontsize=10, color="dimgray")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("last_result.png", dpi=300)
    print("📈 Contextual chart saved as: last_result.png")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = PromptTemplate(
    input_variables=["question", "data"],
    template=(
        "You are a data analyst. Given the user's question:\n"
        "{question}\n"
        "and the following data (top 10 rows):\n{data}\n"
        "Provide a concise, analytical summary that highlights the key trends and insights."
    )
)

def summarize_results_langchain(question, df):
    data_str = df.head(10).to_markdown(index=False)
    formatted_prompt = prompt.format(question=question, data=data_str)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    return response.content.strip()


def open_chart_image(image_path="last_result.png"):
    """
    Automatically opens the saved chart image depending on the OS.
    """
    if not os.path.exists(image_path):
        print(f"⚠️ Chart not found: {image_path}")
        return

    print(f"🖼️ Opening chart: {image_path}")

    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", image_path])
    elif platform.system() == "Windows":
        os.startfile(image_path)
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", image_path])
    else:
        print("⚠️ Unsupported OS — please open the chart manually.")

# --- Initialize retriever ---
retriever = BnbRetriever(top_k=6)
#retriever = BnbRetriever()
agent = BnbSQLAgent(retriever)


if __name__ == "__main__":
    agent = BnbSQLAgent()
    query = "Which drug classes have shown the highest variance between AWP (Average Wholesale Price) and ASP (Average Sales Price) over the past 8 quarters, and what patterns can be inferred about pricing efficiency or margin compression?"
    #"Give me the top 5 most expensive drugs this qtr along with the amount?"
    sql = agent.generate_sql(query)
    df = agent.execute_sql(sql)
    if df is not None:
        print(df.head())
    
    summary = summarize_results_langchain(query, df)
    #summarize_results(query, df)
    print("🧾 Insight Summary:")
    print(summary)
    visualize_results(df, query, summary)    