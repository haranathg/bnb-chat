import io
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List, Literal, Optional
import re
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
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
from sql_validator import SQLValidator, SQLValidationError  # noqa: E402
from rate_limiter import user_rate_limiter, global_rate_limiter  # noqa: E402
from audit_logger import get_audit_logger  # noqa: E402

AUTH_TOKEN = os.getenv("AUTH_TOKEN")
MAX_RAW_ROWS = int(os.getenv("MAX_RAW_ROWS", "10"))
DEFAULT_SQL_LIMIT = int(os.getenv("DEFAULT_SQL_LIMIT", "50"))

# Initialize SQL validator with allowed tables
sql_validator = SQLValidator(
    allowed_tables=[
        "drug_master",
        "drug_class",
        "asp_history",
        "awp_history",
        "wac_history",
    ],
    require_limit=True,
    max_limit=int(os.getenv("MAX_SQL_LIMIT", "1000")),
)

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


def _serialize_value(value: Any, column_name: str = "") -> Any:
    """
    Serialize values for JSON response.
    Formats monetary/pricing columns to 2 decimal places.
    """
    if value is None or isinstance(value, (bool, str)):
        return value

    # Identify monetary/pricing columns (but exclude ratios/percentages)
    col_lower = column_name.lower()

    # Exclude ratio and percentage columns
    if 'ratio' in col_lower or 'pct' in col_lower or 'percent' in col_lower:
        is_monetary = False
    else:
        is_monetary = any([
            'price' in col_lower,
            'asp' in col_lower,
            'wac' in col_lower,
            'awp' in col_lower,
            'cost' in col_lower,
            'payment' in col_lower,
            'amount' in col_lower,
            'limit' in col_lower and 'payment' in col_lower,
        ])

    # Handle Decimal types
    if isinstance(value, Decimal):
        float_val = float(value)
        # Format monetary values to 2 decimals, others as-is
        return round(float_val, 2) if is_monetary else float_val

    # Handle float/int types
    if isinstance(value, (int, float)):
        # Format monetary values to 2 decimals
        return round(float(value), 2) if is_monetary else value

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

    working = df.copy()

    # Treat identifier-like numeric columns as categorical so they render correctly on the x-axis.
    id_like_pattern = re.compile(r"(code|id|number|npi)$", re.IGNORECASE)

    def _format_identifier(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int,)):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    for column in working.select_dtypes(include="number").columns:
        if id_like_pattern.search(column):
            working[column] = working[column].apply(_format_identifier)

    numeric_cols = working.select_dtypes(include="number").columns.tolist()
    text_cols = working.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols and text_cols:
        # Prefer human-readable names over codes for x-axis
        name_patterns = ['name', 'label', 'title', 'class', 'category', 'type']
        code_patterns = ['code', 'id', 'number', 'npi']

        # Find best text column for x-axis (prefer names over codes)
        x_col = None
        for pattern in name_patterns:
            for col in text_cols:
                if pattern in col.lower():
                    x_col = col
                    break
            if x_col:
                break

        # If no name column found, use first non-code column
        if not x_col:
            for col in text_cols:
                is_code = any(cp in col.lower() for cp in code_patterns)
                if not is_code:
                    x_col = col
                    break

        # Fallback to first text column
        if not x_col:
            x_col = text_cols[0]

        y_col = numeric_cols[0]
        filtered = working[[x_col, y_col]].dropna()
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
                        "xaxis": {"title": x_col, "tickangle": -45, "type": "category"},
                        "yaxis": {"title": y_col},
                        "margin": {"l": 60, "r": 20, "t": 60, "b": 120},
                    },
                }
            )
            return charts

    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[:2]
        filtered = working[[x_col, y_col]].dropna()
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


def _ensure_limit(sql: str) -> tuple[str, bool]:
    """Guarantee a LIMIT clause exists; append one if missing."""
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql, False

    cleaned = sql.strip().rstrip(";")
    enforced_sql = f"{cleaned}\nLIMIT {DEFAULT_SQL_LIMIT};"
    return enforced_sql, True


def _validate_sql_for_guardrails(question: str, sql: str) -> None:
    """
    Validate SQL using AST-based parser for robust security checks.
    Raises SQLValidationError if validation fails.
    """
    try:
        sql_validator.validate(sql)
    except SQLValidationError as e:
        # Re-raise as SQLValidationError so it's caught by the endpoint handler
        raise e


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


def _execute_with_retry(question: str, sql: str) -> tuple[pd.DataFrame, str, bool]:
    df = execute_sql(sql)
    final_sql = sql
    limit_applied = False
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
        fixed_sql, added_limit = _ensure_limit(fixed_sql)
        limit_applied = limit_applied or added_limit
        df = execute_sql(fixed_sql)
        final_sql = fixed_sql
    return df, final_sql, limit_applied


def run_pipeline(question: str, options: QueryOptions) -> QueryResponse:
    previous_debug = getattr(retriever, "debug", False)
    retriever.debug = options.debug

    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        sql = generate_sql(question)
        sql, limit_added_initial = _ensure_limit(sql)
        _validate_sql_for_guardrails(question, sql)
        df, final_sql, limit_added_retry = _execute_with_retry(question, sql)

    final_sql, limit_added_post = _ensure_limit(final_sql)
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
                [_serialize_value(value, col) for value, col in zip(row, raw_columns)]
                for row in trimmed.to_numpy().tolist()
            ]
        else:
            raw_columns = cleaned.columns.tolist()
            raw_rows = [
                [_serialize_value(value, col) for value, col in zip(row, raw_columns)]
                for row in cleaned.to_numpy().tolist()
            ]

    debug_output = buffer.getvalue().strip() if options.debug else ""

    limit_note_required = any([limit_added_initial, limit_added_retry, limit_added_post])
    limit_note = (
        f"Results limited to the top {DEFAULT_SQL_LIMIT} rows to preserve query performance."
        if limit_note_required
        else ""
    )

    combined_note_parts = [note for note in [raw_data_note, limit_note] if note]
    combined_note = " ".join(combined_note_parts)

    response = QueryResponse(
        analysis_html=analysis_html,
        raw_columns=raw_columns,
        raw_data=raw_rows,
        debug_logs=debug_output,
        sql=final_sql,
        charts=charts,
        raw_data_note=combined_note or None,
    )
    if combined_note:
        response.analysis_html += f"\n<p><em>{combined_note}</em></p>"
    return response


def _authorize(
    request: Request, authorization: Optional[str] = Header(default=None)
) -> str:
    """
    Authorize request and return user identifier for rate limiting.
    Returns a hash of the token or IP address for rate limit tracking.
    """
    import hashlib

    # Check global rate limit first
    allowed, retry_after = global_rate_limiter.check_rate_limit()
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Global rate limit exceeded. Retry after {retry_after:.1f} seconds.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    # Authenticate
    if AUTH_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
            )
        token = authorization.split(" ", 1)[1].strip()

        # Support multiple tokens (comma-separated in AUTH_TOKEN env var)
        valid_tokens = [t.strip() for t in AUTH_TOKEN.split(",")]
        if token not in valid_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
            )
        # Use full hash of token as user identifier for audit logging and rate limiting
        user_id = hashlib.sha256(token.encode()).hexdigest()[:16]
    else:
        # No auth token configured, use IP address
        client_ip = request.client.host if request.client else "unknown"
        user_id = hashlib.sha256(client_ip.encode()).hexdigest()[:16]

    # Check per-user rate limit
    allowed, retry_after = user_rate_limiter.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. You can make {user_rate_limiter.requests_per_minute} requests per minute. Retry after {retry_after:.1f} seconds.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    return user_id


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    payload: QueryRequest, user_id: str = Depends(_authorize)
):
    import time
    start_time = time.time()
    audit_logger = get_audit_logger()

    try:
        response = run_pipeline(payload.query.strip(), payload.options)
        execution_time_ms = (time.time() - start_time) * 1000

        # Log successful query
        audit_logger.log_query(
            user_id=user_id,
            question=payload.query.strip(),
            sql=response.sql,
            data_rows=response.raw_data,
            data_columns=response.raw_columns,
            analysis=response.analysis_html,
            execution_time_ms=execution_time_ms,
            options={
                "show_raw": payload.options.show_raw,
                "debug": payload.options.debug,
                "analysis_mode": payload.options.analysis_mode,
            }
        )

        return response
    except SQLValidationError as exc:
        execution_time_ms = (time.time() - start_time) * 1000

        # Log failed query
        audit_logger.log_query(
            user_id=user_id,
            question=payload.query.strip(),
            sql="",
            error=f"SQL validation failed: {str(exc)}",
            execution_time_ms=execution_time_ms,
            options={
                "show_raw": payload.options.show_raw,
                "debug": payload.options.debug,
                "analysis_mode": payload.options.analysis_mode,
            }
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SQL validation failed: {str(exc)}",
        ) from exc
    except Exception as exc:
        execution_time_ms = (time.time() - start_time) * 1000

        # Log failed query
        audit_logger.log_query(
            user_id=user_id,
            question=payload.query.strip(),
            sql="",
            error=str(exc),
            execution_time_ms=execution_time_ms,
            options={
                "show_raw": payload.options.show_raw,
                "debug": payload.options.debug,
                "analysis_mode": payload.options.analysis_mode,
            }
        )

        # Don't expose internal error details to clients
        import logging
        logging.error(f"Query processing error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query. Please try again or contact support.",
        ) from exc


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
