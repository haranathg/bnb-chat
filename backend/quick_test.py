"""Quick inline test of SQL validator"""

from sql_validator import SQLValidator, SQLValidationError

validator = SQLValidator(
    allowed_tables=["drug_master", "drug_class", "asp_history", "awp_history", "wac_history"],
    require_limit=True,
    max_limit=1000,
)

# Test 1: Valid query
try:
    validator.validate("SELECT brand_name, asp_per_unit_current_quarter FROM drug_master LIMIT 10")
    print("✅ Test 1 PASS: Valid query accepted")
except Exception as e:
    print(f"❌ Test 1 FAIL: {e}")

# Test 2: Block DROP
try:
    validator.validate("DROP TABLE drug_master")
    print("❌ Test 2 FAIL: DROP was not blocked!")
except SQLValidationError:
    print("✅ Test 2 PASS: DROP blocked")

# Test 3: Block SELECT *
try:
    validator.validate("SELECT * FROM drug_master LIMIT 10")
    print("❌ Test 3 FAIL: SELECT * was not blocked!")
except SQLValidationError:
    print("✅ Test 3 PASS: SELECT * blocked")

# Test 4: Require LIMIT
try:
    validator.validate("SELECT brand_name FROM drug_master")
    print("❌ Test 4 FAIL: Missing LIMIT was not caught!")
except SQLValidationError:
    print("✅ Test 4 PASS: Missing LIMIT blocked")

print("\nBasic validation tests completed!")
