import base64
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import uvicorn

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from bnb_lcel_pipeline import (  # noqa: E402
    clean_sql,
    execute_sql,
    generate_sql,
    llm,
    retriever,
    summary_prompt,
)

AUTH_TOKEN = os.getenv("AUTH_TOKEN")

app = FastAPI(title="bnb-chat-with-data API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok"}


class QueryOptions(BaseModel):
    show_raw: bool = Field(default=True, description="Return raw data table")
    debug: bool = Field(default=False, description="Include retriever debug logs")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language question")
    options: QueryOptions = QueryOptions()


class QueryResponse(BaseModel):
    analysis_html: str
    raw_columns: Optional[List[str]] = None
    raw_data: Optional[List[List[Any]]] = None
    debug_logs: Optional[str] = None
    sql: str
    charts: Optional[List[dict]] = None


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _format_analysis_html(summary: str, chart_uri: Optional[str]) -> str:
    paragraphs = [
        f"<p>{line.strip()}</p>"
        for line in summary.strip().split("\n\n")
        if line.strip()
    ]
    html = "\n".join(paragraphs) if paragraphs else "<p>No analysis available.</p>"
    if chart_uri:
        html += f'\n<div class="mt-4"><img src="{chart_uri}" alt="Insight chart" /></div>'
    return html


def _render_chart_to_base64(df: pd.DataFrame, question: str, summary: str) -> Optional[str]:
    if df.empty:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(exclude="number").columns.tolist()
    if not numeric_cols:
        return None

    sns.set(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if len(numeric_cols) == 1 and text_cols:
            x_col = text_cols[0]
            y_col = numeric_cols[0]
            df_sorted = df.sort_values(by=y_col, ascending=False).head(10)
            palette = sns.color_palette("crest", n_colors=len(df_sorted))
            sns.barplot(
                data=df_sorted,
                x=x_col,
                y=y_col,
                hue=x_col,
                dodge=False,
                palette=palette,
                legend=False,
                ax=ax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"Top 10 by {y_col}")
            ax.set_ylabel(y_col)
        elif len(numeric_cols) >= 2:
            sns.scatterplot(
                data=df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                hue=text_cols[0] if text_cols else None,
                palette="viridis",
                s=100,
                ax=ax,
            )
            ax.set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
        else:
            return None

        fig.suptitle(f"Insight Visualization: {question}", fontsize=14, fontweight="bold")
        fig.text(
            0.5,
            0.01,
            (summary[:250] + "...") if len(summary) > 250 else summary,
            ha="center",
            fontsize=9,
            color="dimgray",
            wrap=True,
        )

        buffer = io.BytesIO()
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    finally:
        plt.close(fig)


def _build_summary(question: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return (
            f"Hmm 🤔 I didn’t find any data returned for your question:\n"
            f"“{question}”\n\n"
            "Please verify the SQL query or contact the data admin for support."
        )

    data_str = df.head(10).to_markdown(index=False)
    prompt = summary_prompt.format(question=question, data=data_str)
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content.strip()


def _execute_with_retry(question: str, sql: str) -> tuple[pd.DataFrame, str]:
    df = execute_sql(sql)
    final_sql = sql
    if df.empty:
        retry_prompt = (
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
        fixed_sql = clean_sql(llm.invoke([HumanMessage(content=retry_prompt)]).content)
        df = execute_sql(fixed_sql)
        final_sql = fixed_sql
    return df, final_sql


def run_pipeline(question: str, options: QueryOptions) -> QueryResponse:
    previous_debug = getattr(retriever, "debug", False)
    retriever.debug = options.debug

    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        sql = generate_sql(question)
        df, final_sql = _execute_with_retry(question, sql)

    retriever.debug = previous_debug

    summary = _build_summary(question, df)
    chart_uri = _render_chart_to_base64(df, question, summary)
    analysis_html = _format_analysis_html(summary, chart_uri)

    raw_columns = None
    raw_rows = None
    if options.show_raw and df is not None and not df.empty:
        cleaned = df.where(pd.notnull(df), None)
        raw_columns = cleaned.columns.tolist()
        raw_rows = [
            [_serialize_value(value) for value in row]
            for row in cleaned.to_numpy().tolist()
        ]

    debug_output = buffer.getvalue().strip() if options.debug else ""

    return QueryResponse(
        analysis_html=analysis_html,
        raw_columns=raw_columns,
        raw_data=raw_rows,
        debug_logs=debug_output,
        sql=final_sql,
        charts=[],
    )


def _authorize(authorization: Optional[str] = Header(default=None)) -> None:
    if not AUTH_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )
    token = authorization.split(" ", 1)[1].strip()
    if token != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest, _: None = Depends(_authorize)):
    return run_pipeline(payload.query.strip(), payload.options)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
