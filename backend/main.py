import io
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List, Literal, Optional
import re
import pandas as pd
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
MAX_RAW_ROWS = int(os.getenv("MAX_RAW_ROWS", "10"))

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
    analysis_mode: Literal["brief", "elaborate"] = Field(
        default="elaborate", description="Control verbosity of the insight summary"
    )


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
    raw_data_note: Optional[str] = None


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _format_analysis_html(summary: str) -> str:
    paragraphs = [
        f"<p>{line.strip()}</p>"
        for line in summary.strip().split("\n\n")
        if line.strip()
    ]
    html = "\n".join(paragraphs) if paragraphs else "<p>No analysis available.</p>"
    return html


def _build_chart_specs(df: pd.DataFrame) -> List[dict]:
    charts: List[dict] = []
    if df is None or df.empty:
        return charts

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols and text_cols:
        x_col = text_cols[0]
        y_col = numeric_cols[0]
        filtered = df[[x_col, y_col]].dropna()
        if not filtered.empty:
            filtered = filtered.sort_values(by=y_col, ascending=False).head(10)
            charts.append(
                {
                    "data": [
                        {
                            "type": "bar",
                            "x": filtered[x_col].astype(str).tolist(),
                            "y": filtered[y_col].tolist(),
                            "marker": {"color": "#4f46e5"},
                        }
                    ],
                    "layout": {
                        "title": f"Top {len(filtered)} {x_col} by {y_col}",
                        "xaxis": {"title": x_col, "tickangle": -45},
                        "yaxis": {"title": y_col},
                        "margin": {"l": 60, "r": 20, "t": 60, "b": 120},
                    },
                }
            )
            return charts

    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[:2]
        filtered = df[[x_col, y_col]].dropna()
        if not filtered.empty:
            limit = min(len(filtered), 500)
            filtered = filtered.head(limit)
            trace = {
                "type": "scatter",
                "mode": "markers",
                "x": filtered[x_col].tolist(),
                "y": filtered[y_col].tolist(),
                "marker": {"size": 9, "color": "#4f46e5", "opacity": 0.8},
            }
            if text_cols:
                labels = df.loc[filtered.index, text_cols[0]].astype(str).tolist()
                trace["text"] = labels
                trace["hovertemplate"] = (
                    f"{text_cols[0]}: %{{text}}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                )
            charts.append(
                {
                    "data": [trace],
                    "layout": {
                        "title": f"{y_col} vs {x_col}",
                        "xaxis": {"title": x_col},
                        "yaxis": {"title": y_col},
                        "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
                    },
                }
            )

    return charts


def _validate_sql_for_guardrails(question: str, sql: str) -> None:
    lowered = sql.lower()
    if re.search(r"\blimit\b", lowered) is None:
        raise HTTPException(
            status_code=400,
            detail="For security reasons, generated SQL must include a LIMIT clause. Please refine the question.",
        )
    if re.search(r"select\s+\*", lowered):
        raise HTTPException(
            status_code=400,
            detail="Queries must reference explicit columns; 'SELECT *' is not allowed.",
        )


def _build_summary(question: str, df: pd.DataFrame, mode: str) -> str:
    if df is None or df.empty:
        return (
            f"Hmm 🤔 I didn’t find any data returned for your question:\n"
            f"“{question}”\n\n"
            "Please verify the SQL query or contact the data admin for support."
        )

    data_str = df.head(10).to_markdown(index=False)
    mode_instruction = (
        "Provide a concise executive summary in no more than two short sentences. "
        "Highlight only the single most important quantitative takeaway and any critical caveat."
        if mode == "brief"
        else "Provide a thoughtful, well-structured analysis in up to five sentences. "
        "Highlight key trends, context, notable outliers, and any important caveats."
    )
    prompt = summary_prompt.format(question=question, data=data_str) + f"\n\n{mode_instruction}"
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
        _validate_sql_for_guardrails(question, sql)
        df, final_sql = _execute_with_retry(question, sql)

    _validate_sql_for_guardrails(question, final_sql)
    retriever.debug = previous_debug

    summary = _build_summary(question, df, options.analysis_mode)
    charts = _build_chart_specs(df)
    analysis_html = _format_analysis_html(summary)

    raw_columns = None
    raw_rows = None
    raw_data_note = ""
    if options.show_raw and df is not None and not df.empty:
        cleaned = df.where(pd.notnull(df), None)
        if len(cleaned) > MAX_RAW_ROWS:
            raw_data_note = (
                f"Displaying only the first {MAX_RAW_ROWS} rows out of {len(cleaned)} retrieved from the Buy&Bill database. "
                "Contact the data admin if you require a full export."
            )
            trimmed = cleaned.head(MAX_RAW_ROWS)
            raw_columns = trimmed.columns.tolist()
            raw_rows = [
                [_serialize_value(value) for value in row]
                for row in trimmed.to_numpy().tolist()
            ]
        else:
            raw_columns = cleaned.columns.tolist()
            raw_rows = [
                [_serialize_value(value) for value in row]
                for row in cleaned.to_numpy().tolist()
            ]

    debug_output = buffer.getvalue().strip() if options.debug else ""

    response = QueryResponse(
        analysis_html=analysis_html,
        raw_columns=raw_columns,
        raw_data=raw_rows,
        debug_logs=debug_output,
        sql=final_sql,
        charts=charts,
        raw_data_note=raw_data_note or None,
    )
    if raw_data_note:
        response.analysis_html += f"\n<p><em>{raw_data_note}</em></p>"
    return response


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
    try:
        return run_pipeline(payload.query.strip(), payload.options)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
