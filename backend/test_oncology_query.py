"""
Test script to verify the oncology query works after fixing concat_label issue.
"""
import os
from dotenv import load_dotenv
from bnb_lcel_pipeline import generate_sql, execute_sql

load_dotenv()

# Test the problematic query
question = "give me the top 10 most expensive oncology drugs"

print(f"Testing question: {question}\n")
print("=" * 80)

# Generate SQL
sql = generate_sql(question)
print(f"\n✅ Generated SQL successfully:")
print(sql)
print("=" * 80)

# Execute SQL
df = execute_sql(sql)
print(f"\n✅ Query executed successfully!")
print(f"   Rows returned: {len(df)}")

if not df.empty:
    print(f"\n   Top 3 results:")
    print(df.head(3).to_string(index=False))

    # Check if concat_label is being used (it shouldn't be)
    if 'concat_label' in sql.lower():
        print("\n⚠️ WARNING: SQL still references concat_label column!")
    else:
        print("\n✅ SQL does not reference concat_label - fix confirmed!")
else:
    print("\n⚠️ Query returned 0 rows")
