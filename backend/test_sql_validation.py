"""
test_sql_validation.py
----------------------
Test suite for SQL validator to ensure it blocks common attack vectors.
Run with: python test_sql_validation.py
"""

from sql_validator import SQLValidator, SQLValidationError


def test_sql_validator():
    """Test SQL validator against various attack vectors and valid queries."""

    validator = SQLValidator(
        allowed_tables=["drug_master", "drug_class", "asp_history", "awp_history", "wac_history"],
        require_limit=True,
        max_limit=1000,
    )

    print("=" * 70)
    print("SQL VALIDATOR SECURITY TESTS")
    print("=" * 70)

    # Valid queries that should pass
    valid_queries = [
        (
            "Valid SELECT with LIMIT",
            "SELECT brand_name, asp_per_unit_current_quarter FROM drug_master WHERE asp_per_unit_current_quarter > 100 LIMIT 10",
        ),
        (
            "Valid WITH CTE",
            """
            WITH recent_drugs AS (
                SELECT brand_name, asp_per_unit_current_quarter FROM drug_master WHERE asp_per_unit_current_quarter > 0
            )
            SELECT brand_name, asp_per_unit_current_quarter FROM recent_drugs LIMIT 50
            """,
        ),
        (
            "Valid JOIN",
            """
            SELECT dm.brand_name, dc.drug_class
            FROM drug_master dm
            JOIN drug_class dc ON dm.hcpcs_code = dc.hcpcs_code
            LIMIT 20
            """,
        ),
        (
            "Valid with window function",
            """
            SELECT
                brand_name,
                asp_per_unit_current_quarter,
                ROW_NUMBER() OVER (ORDER BY asp_per_unit_current_quarter DESC) as rank
            FROM drug_master
            LIMIT 10
            """,
        ),
    ]

    # Attack vectors that should be blocked
    attack_vectors = [
        # SQL Injection attempts
        (
            "SQL Injection - DROP TABLE",
            "SELECT * FROM drug_master; DROP TABLE drug_master; --",
        ),
        (
            "SQL Injection - UNION attack",
            "SELECT brand_name FROM drug_master WHERE hcpcs_code = 'J1234' UNION SELECT password FROM users LIMIT 10",
        ),
        (
            "SQL Injection - Comment injection",
            "SELECT * FROM drug_master WHERE brand_name = 'test' -- AND active = 1 LIMIT 10",
        ),
        (
            "SQL Injection - Stacked queries",
            "SELECT * FROM drug_master LIMIT 10; DELETE FROM drug_master;",
        ),
        # DML/DDL operations
        ("DELETE statement", "DELETE FROM drug_master WHERE hcpcs_code = 'J1234'"),
        ("UPDATE statement", "UPDATE drug_master SET asp_per_unit_current_quarter = 0 WHERE hcpcs_code = 'J1234'"),
        ("INSERT statement", "INSERT INTO drug_master (brand_name) VALUES ('hack')"),
        ("DROP statement", "DROP TABLE drug_master"),
        ("CREATE statement", "CREATE TABLE evil (id INT)"),
        ("ALTER statement", "ALTER TABLE drug_master ADD COLUMN evil TEXT"),
        ("TRUNCATE statement", "TRUNCATE TABLE drug_master"),
        # SELECT * violation
        ("SELECT *", "SELECT * FROM drug_master LIMIT 10"),
        # Missing LIMIT
        ("Missing LIMIT", "SELECT brand_name, asp_per_unit_current_quarter FROM drug_master"),
        # Excessive LIMIT
        (
            "Excessive LIMIT",
            "SELECT brand_name FROM drug_master LIMIT 10000",
        ),
        # Unauthorized table access
        (
            "Unauthorized table",
            "SELECT * FROM users WHERE id = 1 LIMIT 10",
        ),
        # Dangerous functions
        (
            "Dangerous function - pg_read_file",
            "SELECT pg_read_file('/etc/passwd') LIMIT 1",
        ),
        (
            "Dangerous function - pg_sleep",
            "SELECT pg_sleep(10) FROM drug_master LIMIT 1",
        ),
        # System commands
        (
            "COPY TO OUTFILE",
            "SELECT brand_name FROM drug_master INTO OUTFILE '/tmp/data.csv' LIMIT 10",
        ),
    ]

    # Test valid queries
    print("\n" + "=" * 70)
    print("TESTING VALID QUERIES (should pass)")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, sql in valid_queries:
        try:
            validator.validate(sql)
            print(f"✅ PASS: {name}")
            passed += 1
        except SQLValidationError as e:
            print(f"❌ FAIL: {name}")
            print(f"   Unexpected error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ FAIL: {name}")
            print(f"   Unexpected exception: {type(e).__name__}: {e}")
            failed += 1

    # Test attack vectors
    print("\n" + "=" * 70)
    print("TESTING ATTACK VECTORS (should be blocked)")
    print("=" * 70)

    blocked = 0
    not_blocked = 0

    for name, sql in attack_vectors:
        try:
            validator.validate(sql)
            print(f"❌ NOT BLOCKED: {name}")
            print(f"   Query: {sql[:80]}...")
            not_blocked += 1
        except SQLValidationError as e:
            print(f"✅ BLOCKED: {name}")
            print(f"   Reason: {str(e)[:100]}...")
            blocked += 1
        except Exception as e:
            print(f"⚠️  ERROR: {name}")
            print(f"   Exception: {type(e).__name__}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Valid Queries:    {passed} passed, {failed} failed (out of {len(valid_queries)})")
    print(f"Attack Vectors:   {blocked} blocked, {not_blocked} NOT BLOCKED (out of {len(attack_vectors)})")

    total_tests = len(valid_queries) + len(attack_vectors)
    total_success = passed + blocked
    success_rate = (total_success / total_tests) * 100

    print(f"\nOverall Success Rate: {success_rate:.1f}% ({total_success}/{total_tests})")

    if failed > 0:
        print(f"\n⚠️  WARNING: {failed} valid queries were incorrectly rejected!")
    if not_blocked > 0:
        print(f"\n🚨 CRITICAL: {not_blocked} attack vectors were NOT blocked!")

    if failed == 0 and not_blocked == 0:
        print("\n🎉 All tests passed! SQL validator is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please review the validator implementation.")
        return False


def test_rate_limiter():
    """Test rate limiter functionality."""
    from rate_limiter import RateLimiter
    import time

    print("\n" + "=" * 70)
    print("RATE LIMITER TESTS")
    print("=" * 70)

    # Create a rate limiter: 5 requests per minute, burst of 10
    limiter = RateLimiter(requests_per_minute=5, burst_size=10)

    print("\nTesting burst capacity (should allow 10 rapid requests)...")
    user_id = "test_user_1"
    allowed_count = 0

    for i in range(15):
        allowed, retry_after = limiter.check_rate_limit(user_id)
        if allowed:
            allowed_count += 1
            print(f"  Request {i+1}: ✅ Allowed")
        else:
            print(f"  Request {i+1}: ❌ Rate limited (retry after {retry_after:.2f}s)")

    print(f"\nBurst test: {allowed_count}/15 requests allowed (expected: ~10)")

    # Test refill mechanism
    print("\nTesting token refill (waiting 2 seconds)...")
    time.sleep(2)

    allowed, _ = limiter.check_rate_limit(user_id)
    if allowed:
        print("  ✅ Request allowed after waiting (tokens refilled)")
    else:
        print("  ❌ Request still blocked after waiting")

    # Test per-user isolation
    print("\nTesting per-user isolation...")
    user2_id = "test_user_2"
    allowed, _ = limiter.check_rate_limit(user2_id)
    if allowed:
        print("  ✅ Different user has separate rate limit")
    else:
        print("  ❌ Rate limit leaked across users")

    print("\n✅ Rate limiter tests completed")


if __name__ == "__main__":
    print("\n🧪 Starting security tests...\n")

    # Run SQL validation tests
    sql_passed = test_sql_validator()

    # Run rate limiter tests
    test_rate_limiter()

    print("\n" + "=" * 70)
    if sql_passed:
        print("✅ ALL TESTS PASSED - Security measures are working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Review security implementation!")
    print("=" * 70 + "\n")
