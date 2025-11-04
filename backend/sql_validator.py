"""
sql_validator.py
-----------------
Validates LLM-generated SQL queries using AST parsing to prevent injection
and enforce security policies.
"""

import re
import sqlparse
from sqlparse.sql import (
    IdentifierList,
    Identifier,
    Where,
    Function,
    Parenthesis,
    Token,
    Statement,
)
from sqlparse.tokens import Keyword, DML, DDL
from typing import List, Tuple, Optional


class SQLValidationError(Exception):
    """Raised when SQL query fails validation checks."""
    pass


class SQLValidator:
    """
    Validates SQL queries for security and policy compliance.
    Uses sqlparse for AST-based validation instead of regex.
    """

    # Allowed tables - should be configurable
    ALLOWED_TABLES = {
        "drug_metrics",
        "drug_class",
        "hcpcs_drugs",
        "quarters",
        # Add other allowed tables here
    }

    # Dangerous keywords that should never appear
    FORBIDDEN_KEYWORDS = {
        "DROP",
        "DELETE",
        "TRUNCATE",
        "ALTER",
        "CREATE",
        "INSERT",
        "UPDATE",
        "GRANT",
        "REVOKE",
        "EXEC",
        "EXECUTE",
        "DECLARE",
        "INTO OUTFILE",
        "LOAD_FILE",
        "COPY",
    }

    # Dangerous functions
    FORBIDDEN_FUNCTIONS = {
        "pg_read_file",
        "pg_ls_dir",
        "pg_sleep",
        "system",
        "lo_import",
        "lo_export",
    }

    def __init__(
        self,
        allowed_tables: Optional[List[str]] = None,
        require_limit: bool = True,
        max_limit: int = 1000,
        allow_subqueries: bool = True,
    ):
        """
        Initialize SQL validator with configurable rules.

        Args:
            allowed_tables: List of permitted table names (None = use defaults)
            require_limit: Whether LIMIT clause is mandatory
            max_limit: Maximum allowed LIMIT value
            allow_subqueries: Whether subqueries/CTEs are permitted
        """
        self.allowed_tables = set(allowed_tables) if allowed_tables else self.ALLOWED_TABLES
        self.require_limit = require_limit
        self.max_limit = max_limit
        self.allow_subqueries = allow_subqueries

    def validate(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query against security policies.

        Returns:
            Tuple of (is_valid, error_message)
            - (True, None) if valid
            - (False, error_message) if invalid

        Raises:
            SQLValidationError: If query violates security policies
        """
        if not sql or not sql.strip():
            raise SQLValidationError("SQL query cannot be empty")

        sql = sql.strip()

        # Parse SQL into AST
        try:
            parsed = sqlparse.parse(sql)
        except Exception as e:
            raise SQLValidationError(f"Failed to parse SQL: {e}")

        if not parsed:
            raise SQLValidationError("No valid SQL statements found")

        if len(parsed) > 1:
            raise SQLValidationError(
                "Multiple SQL statements not allowed. Only single SELECT queries are permitted."
            )

        stmt = parsed[0]

        # 1. Check statement type - only SELECT allowed
        self._check_statement_type(stmt)

        # 2. Check for forbidden keywords
        self._check_forbidden_keywords(sql.upper())

        # 3. Check for forbidden functions
        self._check_forbidden_functions(stmt)

        # 4. Validate table names
        self._validate_table_names(stmt)

        # 5. Check for SELECT *
        self._check_select_star(stmt)

        # 6. Validate LIMIT clause
        if self.require_limit:
            self._validate_limit_clause(stmt)

        # 7. Check for SQL injection patterns
        self._check_injection_patterns(sql)

        # 8. Validate subqueries if restricted
        if not self.allow_subqueries:
            self._check_subqueries(stmt)

        return True, None

    def _check_statement_type(self, stmt: Statement) -> None:
        """Ensure only SELECT statements are allowed."""
        first_token = stmt.token_first(skip_ws=True, skip_cm=True)
        if not first_token:
            raise SQLValidationError("Empty SQL statement")

        # Check for CTEs (WITH clause)
        if first_token.value.upper() == "WITH":
            # CTEs are allowed, find the main SELECT
            for token in stmt.tokens:
                if token.ttype is DML and token.value.upper() == "SELECT":
                    return
            raise SQLValidationError("WITH clause must be followed by SELECT statement")

        # Check main statement is SELECT
        if first_token.ttype is not DML or first_token.value.upper() != "SELECT":
            raise SQLValidationError(
                f"Only SELECT queries are allowed. Found: {first_token.value}"
            )

    def _check_forbidden_keywords(self, sql_upper: str) -> None:
        """Check for dangerous SQL keywords."""
        for keyword in self.FORBIDDEN_KEYWORDS:
            # Use word boundaries to avoid false positives (e.g., "SELECT" contains "ELECT")
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, sql_upper):
                raise SQLValidationError(
                    f"Forbidden keyword detected: {keyword}. Only read-only SELECT queries are allowed."
                )

    def _check_forbidden_functions(self, stmt: Statement) -> None:
        """Check for dangerous SQL functions."""
        functions = self._extract_functions(stmt)
        for func_name in functions:
            if func_name.lower() in self.FORBIDDEN_FUNCTIONS:
                raise SQLValidationError(
                    f"Forbidden function detected: {func_name}. This function is not allowed."
                )

    def _extract_functions(self, stmt: Statement) -> List[str]:
        """Recursively extract all function names from SQL statement."""
        functions = []

        def extract_from_token(token):
            if isinstance(token, Function):
                func_name = str(token.get_name())
                if func_name:
                    functions.append(func_name)
            elif hasattr(token, "tokens"):
                for sub_token in token.tokens:
                    extract_from_token(sub_token)

        for token in stmt.tokens:
            extract_from_token(token)

        return functions

    def _validate_table_names(self, stmt: Statement) -> None:
        """Validate that only allowed tables are referenced."""
        # Extract CTE names first (these are allowed "tables")
        cte_names = self._extract_cte_names(stmt)
        tables = self._extract_table_names(stmt)

        for table in tables:
            # Remove schema prefix if present (e.g., "public.drug_metrics")
            table_name = table.split(".")[-1].lower()

            # Skip CTE names - they're defined in the query itself
            if table_name in cte_names:
                continue

            if table_name not in self.allowed_tables:
                raise SQLValidationError(
                    f"Table '{table}' is not in the allowed list. "
                    f"Allowed tables: {', '.join(sorted(self.allowed_tables))}"
                )

    def _extract_table_names(self, stmt: Statement) -> List[str]:
        """Extract table names from FROM and JOIN clauses."""
        tables = []
        from_seen = False

        for token in stmt.tokens:
            if token.ttype is Keyword and token.value.upper() in ("FROM", "JOIN", "INNER", "LEFT", "RIGHT", "FULL"):
                from_seen = True
                continue

            if from_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        table_name = self._get_real_name(identifier)
                        if table_name:
                            tables.append(table_name)
                elif isinstance(token, Identifier):
                    table_name = self._get_real_name(token)
                    if table_name:
                        tables.append(table_name)
                elif token.ttype is Keyword:
                    from_seen = False

        return tables

    def _get_real_name(self, identifier: Identifier) -> Optional[str]:
        """Extract real table name from identifier (handles aliases)."""
        if hasattr(identifier, "get_real_name"):
            return identifier.get_real_name()
        return identifier.get_name()

    def _extract_cte_names(self, stmt: Statement) -> set:
        """Extract CTE (Common Table Expression) names from WITH clause."""
        cte_names = set()
        with_seen = False

        for token in stmt.tokens:
            # Check if we're in a WITH clause
            if token.value.upper() == "WITH":
                with_seen = True
                continue

            # Stop when we hit the main SELECT
            if with_seen and token.ttype is DML and token.value.upper() == "SELECT":
                break

            # Extract CTE names
            if with_seen:
                if isinstance(token, Identifier):
                    # CTE name is the identifier before AS
                    cte_name = token.get_name()
                    if cte_name:
                        cte_names.add(cte_name.lower())
                elif isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        cte_name = identifier.get_name()
                        if cte_name:
                            cte_names.add(cte_name.lower())

        return cte_names

    def _check_select_star(self, stmt: Statement) -> None:
        """Check for SELECT * which is not allowed."""
        select_seen = False

        for token in stmt.tokens:
            if token.ttype is DML and token.value.upper() == "SELECT":
                select_seen = True
                continue

            if select_seen:
                # Skip whitespace
                if token.is_whitespace:
                    continue

                # Check if next non-whitespace token is *
                if str(token).strip() == "*":
                    raise SQLValidationError(
                        "SELECT * is not allowed. Please specify explicit column names."
                    )

                # If we found something else, we're past the SELECT clause
                if token.ttype is not Keyword or token.value.upper() not in ("DISTINCT", "ALL"):
                    break

    def _validate_limit_clause(self, stmt: Statement) -> None:
        """Ensure LIMIT clause exists and is reasonable."""
        limit_value = None

        for token in stmt.tokens:
            if token.ttype is Keyword and token.value.upper() == "LIMIT":
                # Find the next non-whitespace token
                idx = stmt.token_index(token)
                for next_token in stmt.tokens[idx + 1 :]:
                    if not next_token.is_whitespace:
                        try:
                            limit_value = int(str(next_token).strip().rstrip(";"))
                            break
                        except ValueError:
                            raise SQLValidationError(
                                f"Invalid LIMIT value: {next_token}. Must be an integer."
                            )
                break

        if limit_value is None:
            raise SQLValidationError(
                "LIMIT clause is required for all queries to prevent excessive data retrieval."
            )

        if limit_value > self.max_limit:
            raise SQLValidationError(
                f"LIMIT value {limit_value} exceeds maximum allowed ({self.max_limit})"
            )

        if limit_value <= 0:
            raise SQLValidationError(f"LIMIT value must be positive, got: {limit_value}")

    def _check_injection_patterns(self, sql: str) -> None:
        """Check for common SQL injection patterns."""
        # Check for comment injection attempts
        comment_patterns = [
            r"--",  # SQL line comment
            r"/\*.*\*/",  # Multi-line comment
            r"#",  # MySQL comment
        ]

        for pattern in comment_patterns:
            if re.search(pattern, sql):
                raise SQLValidationError(
                    "SQL comments are not allowed in queries. Please remove -- or /* */ comments."
                )

        # Check for stacked queries (semicolon followed by more SQL)
        if sql.count(";") > 1:
            raise SQLValidationError(
                "Multiple statements detected (stacked queries not allowed)"
            )

        # Check for union-based injection
        if re.search(r"\bUNION\b(?!\s+ALL\s+SELECT)", sql, re.IGNORECASE):
            # UNION without proper context might be injection
            # This is a simplified check - you may want to allow legitimate UNIONs
            pass  # Allow UNION for now, but log it

    def _check_subqueries(self, stmt: Statement) -> None:
        """Check if subqueries are present when not allowed."""
        if self._contains_subquery(stmt):
            raise SQLValidationError("Subqueries are not allowed in this configuration")

    def _contains_subquery(self, stmt: Statement) -> bool:
        """Recursively check if statement contains subqueries."""
        for token in stmt.tokens:
            if isinstance(token, Parenthesis):
                # Check if parenthesis contains a SELECT
                inner_sql = str(token).strip("()")
                if re.search(r"\bSELECT\b", inner_sql, re.IGNORECASE):
                    return True
            elif hasattr(token, "tokens"):
                if self._contains_subquery(token):
                    return True
        return False


# Convenience function for quick validation
def validate_sql(
    sql: str,
    allowed_tables: Optional[List[str]] = None,
    require_limit: bool = True,
    max_limit: int = 1000,
) -> None:
    """
    Validate SQL query. Raises SQLValidationError if invalid.

    Args:
        sql: SQL query string to validate
        allowed_tables: List of permitted table names
        require_limit: Whether LIMIT is required
        max_limit: Maximum allowed LIMIT value

    Raises:
        SQLValidationError: If validation fails
    """
    validator = SQLValidator(
        allowed_tables=allowed_tables,
        require_limit=require_limit,
        max_limit=max_limit,
    )
    validator.validate(sql)
