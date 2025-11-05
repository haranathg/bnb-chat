"""
Test script to verify audit logging functionality.
"""
import os
import json
import time
from pathlib import Path
from audit_logger import get_audit_logger

# Test the audit logger
print("Testing Audit Logger")
print("=" * 80)

audit_logger = get_audit_logger()

# Simulate a successful query
print("\n1. Testing successful query logging...")
audit_logger.log_query(
    user_id="test_token_12345",
    question="What are the top 10 most expensive oncology drugs?",
    sql="SELECT brand_name, asp_per_unit_current_quarter FROM drug_master JOIN drug_class dc ON dm.hcpcs_code = dc.hcpcs_code WHERE LOWER(dc.drug_class) LIKE '%oncology%' ORDER BY asp_per_unit_current_quarter DESC LIMIT 10",
    data_rows=[
        ["Tecelra", 727000.00],
        ["Kymriah", 558232.27],
        ["Yescarta", 456000.00],
    ],
    data_columns=["brand_name", "asp_per_unit_current_quarter"],
    analysis="<p>The most expensive oncology drugs in the database are Tecelra at $727,000 per unit and Kymriah at $558,232 per unit.</p>",
    execution_time_ms=1234.56,
    options={"show_raw": True, "debug": False, "analysis_mode": "elaborate"}
)
print("✅ Successful query logged")

# Simulate a failed query
print("\n2. Testing failed query logging...")
audit_logger.log_query(
    user_id="test_token_67890",
    question="Show me invalid query",
    sql="",
    error="SQL validation failed: Table 'invalid_table' is not in the allowed list",
    execution_time_ms=45.23,
    options={"show_raw": True, "debug": False, "analysis_mode": "brief"}
)
print("✅ Failed query logged")

# Check the log file
log_dir = Path("audit_logs")
today = time.strftime("%Y-%m-%d")
log_file = log_dir / f"audit_{today}.jsonl"

print(f"\n3. Verifying log file: {log_file}")
if log_file.exists():
    print(f"✅ Log file exists: {log_file}")

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"✅ Found {len(lines)} log entries")

        print("\n4. Sample log entries:")
        for i, line in enumerate(lines[-2:], 1):  # Show last 2 entries
            entry = json.loads(line)
            print(f"\n   Entry {i}:")
            print(f"   - Timestamp: {entry['timestamp']}")
            print(f"   - User Hash: {entry['user_hash']}")
            print(f"   - Question: {entry['question'][:60]}...")
            print(f"   - Execution Time: {entry.get('execution_time_ms', 'N/A')} ms")
            if entry.get('error'):
                print(f"   - Error: {entry['error'][:60]}...")
            else:
                print(f"   - SQL Preview: {entry['sql'][:60]}...")
                if entry.get('data'):
                    print(f"   - Data Rows: {entry['data'].get('total_count', 0)}")
else:
    print(f"❌ Log file not found: {log_file}")

print("\n" + "=" * 80)
print("Audit logging test complete!")
print("\nTo view logs:")
print(f"  cat {log_file} | jq")
print(f"  python -m json.tool < {log_file}")
