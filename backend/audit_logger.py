"""
audit_logger.py

Comprehensive audit logging system for tracking all user queries,
generated SQL, retrieved data, and analysis for review and compliance.
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import hashlib


class AuditLogger:
    """
    Audit logger that captures:
    - User identifier (hashed token)
    - Question asked
    - Generated SQL query
    - Number of rows retrieved
    - Sample of data retrieved (first 5 rows)
    - Analysis/summary generated
    - Timestamp
    - Query execution time
    """

    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup rotating daily log files
        self.current_date = datetime.now().date()
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with daily rotation"""
        log_file = self.log_dir / f"audit_{self.current_date.isoformat()}.jsonl"

        # Create logger
        self.logger = logging.getLogger(f"audit_{self.current_date}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()  # Clear existing handlers

        # File handler for JSON lines format
        handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def _check_date_rotation(self):
        """Check if we need to rotate to a new log file"""
        current_date = datetime.now().date()
        if current_date != self.current_date:
            self.current_date = current_date
            self._setup_logger()

    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy while maintaining auditability"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:12]

    def _sanitize_data(self, data: Any, max_rows: int = 5) -> Any:
        """Sanitize and limit data for logging"""
        if data is None:
            return None

        if isinstance(data, list):
            # Limit to first N rows
            limited = data[:max_rows]
            truncated = len(data) > max_rows
            return {
                "rows": limited,
                "total_count": len(data),
                "truncated": truncated
            }

        return str(data)[:1000]  # Limit string length

    def log_query(
        self,
        user_id: str,
        question: str,
        sql: str,
        data_rows: Optional[list] = None,
        data_columns: Optional[list] = None,
        analysis: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        error: Optional[str] = None,
        options: Optional[dict] = None
    ):
        """
        Log a complete query audit trail.

        Args:
            user_id: User identifier (will be hashed)
            question: Natural language question
            sql: Generated SQL query
            data_rows: Retrieved data rows (will be limited to first 5)
            data_columns: Column names
            analysis: Generated analysis/summary
            execution_time_ms: Query execution time in milliseconds
            error: Error message if query failed
            options: Query options (show_raw, debug, analysis_mode)
        """
        self._check_date_rotation()

        audit_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_hash": self._hash_user_id(user_id),
            "question": question,
            "sql": sql,
            "data": {
                "columns": data_columns,
                "rows": self._sanitize_data(data_rows),
            } if data_rows is not None else None,
            "analysis": analysis,
            "execution_time_ms": execution_time_ms,
            "error": error,
            "options": options or {}
        }

        # Log as JSON line
        self.logger.info(json.dumps(audit_entry, ensure_ascii=False))

    def get_user_queries(self, user_id: str, date: Optional[datetime] = None) -> list:
        """
        Retrieve all queries for a specific user on a given date.

        Args:
            user_id: User identifier
            date: Date to retrieve logs from (defaults to today)

        Returns:
            List of audit entries for the user
        """
        if date is None:
            date = datetime.now().date()

        log_file = self.log_dir / f"audit_{date.isoformat()}.jsonl"
        if not log_file.exists():
            return []

        user_hash = self._hash_user_id(user_id)
        user_queries = []

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("user_hash") == user_hash:
                        user_queries.append(entry)
                except json.JSONDecodeError:
                    continue

        return user_queries

    def get_all_queries(self, date: Optional[datetime] = None) -> list:
        """
        Retrieve all queries from all users on a given date.

        Args:
            date: Date to retrieve logs from (defaults to today)

        Returns:
            List of all audit entries
        """
        if date is None:
            date = datetime.now().date()

        log_file = self.log_dir / f"audit_{date.isoformat()}.jsonl"
        if not log_file.exists():
            return []

        all_queries = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    all_queries.append(entry)
                except json.JSONDecodeError:
                    continue

        return all_queries


# Global audit logger instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        log_dir = os.getenv("AUDIT_LOG_DIR", "audit_logs")
        _audit_logger = AuditLogger(log_dir=log_dir)
    return _audit_logger
