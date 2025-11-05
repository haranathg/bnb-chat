# Audit Logging Guide

This guide explains how to use and manage the comprehensive audit logging system for the bnb-chat-with-data application.

## Overview

All user queries, generated SQL, retrieved data, and analysis are automatically logged to JSON Lines files for audit, compliance, and review purposes.

## Log Location

Logs are stored in: `backend/audit_logs/`

Each day gets its own log file: `audit_YYYY-MM-DD.jsonl`

Example: `audit_2025-11-05.jsonl`

## Log Format

Each log entry is a JSON object on a single line (JSONL format) with the following fields:

```json
{
    "timestamp": "2025-11-05T15:44:01.880632Z",
    "user_hash": "1ac4c4fb37c2",
    "question": "What are the top 10 most expensive oncology drugs?",
    "sql": "SELECT brand_name, asp_per_unit_current_quarter FROM ...",
    "data": {
        "columns": ["brand_name", "asp_per_unit_current_quarter"],
        "rows": {
            "rows": [["Tecelra", 727000.0], ["Kymriah", 558232.27]],
            "total_count": 3,
            "truncated": false
        }
    },
    "analysis": "<p>The most expensive oncology drugs...</p>",
    "execution_time_ms": 1234.56,
    "error": null,
    "options": {
        "show_raw": true,
        "debug": false,
        "analysis_mode": "elaborate"
    }
}
```

### Field Descriptions

- **timestamp**: UTC timestamp of the query (ISO 8601 format)
- **user_hash**: First 12 characters of SHA256 hash of user's access token (for privacy)
- **question**: The natural language question asked by the user
- **sql**: The generated SQL query (empty string if query failed before generation)
- **data**: Retrieved data (limited to first 5 rows for log size management)
  - **columns**: Column names
  - **rows**: Data rows with truncation info
  - **total_count**: Total number of rows in result
  - **truncated**: Whether data was truncated in log
- **analysis**: The generated HTML analysis/summary
- **execution_time_ms**: Query execution time in milliseconds
- **error**: Error message if query failed (null if successful)
- **options**: Query options (show_raw, debug, analysis_mode)

## Viewing Logs

### View today's logs (formatted)
```bash
cd backend/audit_logs
cat audit_$(date +%Y-%m-%d).jsonl | python -m json.tool
```

### View with jq (if installed)
```bash
cat audit_2025-11-05.jsonl | jq
```

### Count queries per day
```bash
wc -l audit_2025-11-05.jsonl
```

### Search for specific questions
```bash
grep -i "oncology" audit_2025-11-05.jsonl | python -m json.tool
```

### View only errors
```bash
grep '"error":' audit_2025-11-05.jsonl | grep -v '"error": null' | python -m json.tool
```

### Extract all questions from a log
```bash
cat audit_2025-11-05.jsonl | jq -r '.question'
```

### Find queries by a specific user (using hash)
```bash
grep '"user_hash": "1ac4c4fb37c2"' audit_2025-11-05.jsonl | python -m json.tool
```

## Programmatic Access

Use the Python API to query logs:

```python
from audit_logger import get_audit_logger
from datetime import datetime

audit_logger = get_audit_logger()

# Get all queries for a specific user today
user_queries = audit_logger.get_user_queries(
    user_id="oAzfiqXIYiHiEkX",  # actual token, will be hashed
    date=datetime.now().date()
)

# Get all queries from all users for a specific date
all_queries = audit_logger.get_all_queries(
    date=datetime(2025, 11, 5).date()
)

# Print summary
for query in user_queries:
    print(f"{query['timestamp']}: {query['question']}")
    print(f"  Execution time: {query['execution_time_ms']} ms")
    if query['error']:
        print(f"  Error: {query['error']}")
```

## User Privacy

- User tokens are **never** stored in logs
- Only a 12-character hash (first 12 chars of SHA256) is logged
- This allows:
  - Tracking individual user activity
  - Rate limiting per user
  - Audit trail for compliance
- But prevents:
  - Reverse-engineering the actual token from logs
  - Identifying users without the original token mapping

## Log Rotation

- Logs automatically rotate daily (new file per day)
- No automatic deletion - implement retention policy as needed
- Recommended: Archive logs older than 90 days

### Example retention script
```bash
# Archive logs older than 90 days
find backend/audit_logs -name "audit_*.jsonl" -mtime +90 -exec gzip {} \;

# Delete archives older than 1 year
find backend/audit_logs -name "audit_*.jsonl.gz" -mtime +365 -delete
```

## Analysis Examples

### Daily usage statistics
```bash
cat audit_2025-11-05.jsonl | jq -r '
  .user_hash as $user |
  .execution_time_ms as $time |
  "\($user),\($time)"
' | awk -F, '{sum[$1]+=$2; count[$1]++} END {
  for (user in sum) {
    printf "%s: %d queries, avg %.2fms\n", user, count[user], sum[user]/count[user]
  }
}'
```

### Most common questions
```bash
cat audit_2025-11-05.jsonl | jq -r '.question' | sort | uniq -c | sort -rn | head -10
```

### Error rate
```bash
total=$(wc -l < audit_2025-11-05.jsonl)
errors=$(grep '"error":' audit_2025-11-05.jsonl | grep -v '"error": null' | wc -l)
echo "Error rate: $(echo "scale=2; $errors * 100 / $total" | bc)%"
```

## Security & Compliance

### HIPAA/Compliance Notes
- Logs contain healthcare drug pricing data (not PHI)
- User identities are pseudonymized via hashing
- Logs stored locally in `backend/audit_logs/` (not in git)
- Ensure proper access controls on production servers
- Consider encrypting audit logs at rest for sensitive deployments

### Access Control
The `audit_logs/` directory should have restricted permissions:
```bash
chmod 700 backend/audit_logs
```

### Backup Strategy
Recommended approach:
1. Daily backup of audit logs to secure storage
2. Encrypted backup for long-term retention
3. Separate backup from application data

## Environment Configuration

### Change audit log directory
Set in `.env`:
```
AUDIT_LOG_DIR=/path/to/secure/audit/logs
```

Default: `backend/audit_logs/`

## Troubleshooting

### Logs not appearing
1. Check directory exists: `ls -la backend/audit_logs/`
2. Check permissions: `ls -l backend/audit_logs/`
3. Verify API is being called (not cached responses)
4. Check server logs for audit_logger errors

### Log file growing too large
- Each entry limited to first 5 data rows
- Consider implementing compression for old logs
- Archive logs older than retention period

### Need to correlate user hash to actual user
1. Locate the user's token in `USER_ACCESS_TOKENS.md`
2. Hash it to find their entries:
```python
import hashlib
token = "oAzfiqXIYiHiEkX"
user_hash = hashlib.sha256(token.encode()).hexdigest()[:12]
print(f"User hash: {user_hash}")
```

## Deployment on Render

Render's ephemeral filesystem means logs are lost on restart. For production:

### Option 1: Persistent Disk (Recommended)
Add a persistent disk in Render dashboard:
- Mount at: `/opt/render/project/src/backend/audit_logs`
- Size: 10GB+ depending on expected usage

### Option 2: External Log Service
Configure to send logs to external service:
- AWS S3
- Google Cloud Storage
- Dedicated log management service (Datadog, Splunk, etc.)

### Option 3: Database Storage
Modify `audit_logger.py` to write to database instead of files.

## Questions?

Contact the development team or review the implementation in:
- `backend/audit_logger.py` - Core logging functionality
- `backend/main.py` - Integration with API endpoint
- `backend/test_audit_logging.py` - Test examples
