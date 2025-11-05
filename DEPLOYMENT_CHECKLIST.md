# Deployment Checklist for Multi-User Access

This checklist ensures proper setup of the multi-user access control and audit logging system.

## Pre-Deployment

### 1. Generate and Store Access Tokens ✅
- [x] 5 secure tokens generated (see `USER_ACCESS_TOKENS.md`)
- [ ] Tokens saved in secure location (password manager, encrypted file)
- [ ] Tokens distributed to authorized users via secure channel

### 2. Environment Variables

Update `.env` file on production (Render):

```bash
# Multi-user tokens (comma-separated, no spaces)
AUTH_TOKEN=oAzfiqXIYiHiEkX,T9aCEt4DVxAIf5d,vBQMUIZTNvpuE0W,1vzvFfWafPMgj5R,F19JiN2cKnqLjii

# Optional: Audit log configuration
AUDIT_LOG_DIR=audit_logs

# Optional: Adjust data limits
MAX_RAW_ROWS=10
DEFAULT_SQL_LIMIT=50
```

**Important**: On Render, set these in the Environment Variables section of your Web Service dashboard.

### 3. Persistent Storage (Render)

Since Render uses ephemeral filesystem, audit logs will be lost on restart unless you:

**Option A: Add Persistent Disk (Recommended)**
1. Go to Render Dashboard → Your Service → Disks
2. Click "Add Disk"
3. Configure:
   - Name: `audit-logs`
   - Mount Path: `/opt/render/project/src/backend/audit_logs`
   - Size: 10GB (or based on expected usage)
4. Click "Create Disk"

**Option B: External Storage**
- Configure external log storage (S3, GCS, etc.)
- Modify `audit_logger.py` to write to external service

**Option C: Accept Ephemeral Logs**
- Logs will be reset on each deployment
- Acceptable for low-stakes testing environments
- NOT recommended for production compliance needs

## Deployment Steps

### 1. Code is Deployed ✅
- [x] Changes pushed to GitHub
- [x] Render auto-deployment triggered
- [ ] Wait for build to complete (check Render dashboard)

### 2. Verify Environment Variables
```bash
# In Render shell or logs, verify:
echo $AUTH_TOKEN  # Should show comma-separated tokens
```

### 3. Test Authentication

**Test 1: Valid Token**
```bash
curl -X POST https://your-app.onrender.com/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer oAzfiqXIYiHiEkX" \
  -d '{"query": "show me top 5 drugs", "options": {"show_raw": true, "debug": false, "analysis_mode": "brief"}}'
```
Expected: 200 OK response with data

**Test 2: Invalid Token**
```bash
curl -X POST https://your-app.onrender.com/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer invalid_token_123" \
  -d '{"query": "test", "options": {}}'
```
Expected: 401 Unauthorized

**Test 3: No Token**
```bash
curl -X POST https://your-app.onrender.com/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "options": {}}'
```
Expected: 401 Unauthorized

### 4. Test Frontend Login
1. Navigate to: `https://your-app.onrender.com`
2. Should see login screen with NO placeholder hint
3. Enter one of the 5 tokens
4. Should successfully authenticate and access the app

### 5. Verify Audit Logging

**Via Render Shell:**
```bash
# Open shell in Render dashboard
cd backend
ls -la audit_logs/
cat audit_logs/audit_$(date +%Y-%m-%d).jsonl | head -1 | python -m json.tool
```

**Via SSH (if configured):**
```bash
ssh your-render-service
cd /opt/render/project/src/backend
tail -f audit_logs/audit_$(date +%Y-%m-%d).jsonl
```

### 6. Monitor First Queries
After users start using the system:
```bash
# Check log file exists
ls -lh backend/audit_logs/

# View latest entries
tail -5 backend/audit_logs/audit_$(date +%Y-%m-%d).jsonl | python -m json.tool

# Count queries
wc -l backend/audit_logs/audit_$(date +%Y-%m-%d).jsonl
```

## User Distribution

### Share with Users

**Email Template:**
```
Subject: Access to bnb-chat-with-data System

Hi [User Name],

You now have access to the bnb-chat-with-data system. Here are your credentials:

URL: https://your-app.onrender.com
Access Token: [INSERT ONE TOKEN PER USER]

To login:
1. Navigate to the URL above
2. Enter your access token in the login field
3. Click "Continue"

Your access token is:
- Unique to you
- Case-sensitive
- Should be kept confidential
- Valid indefinitely (until revoked)

All your queries are logged for audit and compliance purposes.

Questions? Contact: [your email]
```

### User Mapping (Internal Record)

Keep a secure record (NOT in git) mapping tokens to users:

```
oAzfiqXIYiHiEkX → John Doe (john.doe@company.com)
T9aCEt4DVxAIf5d → Jane Smith (jane.smith@company.com)
vBQMUIZTNvpuE0W → Bob Johnson (bob.j@company.com)
1vzvFfWafPMgj5R → Alice Brown (alice.b@company.com)
F19JiN2cKnqLjii → Charlie Wilson (charlie.w@company.com)
```

Store this mapping in:
- Secure password manager
- Encrypted document
- Secure internal wiki (access-controlled)

## Monitoring

### Daily Checks
```bash
# Query volume
wc -l backend/audit_logs/audit_$(date +%Y-%m-%d).jsonl

# Error rate
grep '"error":' backend/audit_logs/audit_$(date +%Y-%m-%d).jsonl | grep -v '"error": null' | wc -l

# Unique users
cat backend/audit_logs/audit_$(date +%Y-%m-%d).jsonl | jq -r '.user_hash' | sort -u | wc -l
```

### Weekly Analysis
```bash
# Most active users (by user hash)
cat backend/audit_logs/audit_2025-11-*.jsonl | jq -r '.user_hash' | sort | uniq -c | sort -rn

# Most common questions
cat backend/audit_logs/audit_2025-11-*.jsonl | jq -r '.question' | sort | uniq -c | sort -rn | head -20

# Average execution time
cat backend/audit_logs/audit_2025-11-*.jsonl | jq -r '.execution_time_ms' | awk '{sum+=$1; count++} END {print "Avg:", sum/count, "ms"}'
```

## Security Best Practices

### Token Management
- [ ] Tokens stored securely (not in plain text emails)
- [ ] User mapping documented (secure location)
- [ ] Process defined for token rotation
- [ ] Process defined for token revocation

### Access Control
- [ ] Audit logs directory has restricted permissions (700)
- [ ] Only authorized personnel can access logs
- [ ] Log review schedule established
- [ ] Incident response plan documented

### Compliance
- [ ] Data retention policy defined (e.g., 90 days)
- [ ] Log backup strategy implemented
- [ ] Log encryption at rest (if required)
- [ ] Compliance requirements documented (HIPAA, SOC2, etc.)

## Troubleshooting

### Users can't log in
1. Verify token is correctly entered (case-sensitive)
2. Check AUTH_TOKEN env var on Render
3. Check Render logs for authentication errors
4. Verify frontend is connecting to correct API endpoint

### Audit logs not appearing
1. Check if audit_logs directory exists
2. Verify write permissions
3. Check Render logs for audit_logger errors
4. If using persistent disk, verify it's mounted correctly

### Logs growing too large
1. Implement log rotation/archival
2. Reduce MAX_RAW_ROWS in .env
3. Consider external log storage
4. Compress old logs: `gzip audit_logs/audit_2025-*.jsonl`

## Token Rotation (Future)

When tokens need to be rotated:

1. Generate new tokens:
```python
import secrets
import string
for i in range(5):
    token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(15))
    print(f"User {i+1}: {token}")
```

2. Update AUTH_TOKEN in Render environment variables
3. Notify users of new tokens (secure channel)
4. Grace period: Keep old tokens active for 7 days
5. After grace period, remove old tokens from AUTH_TOKEN

## Rollback Plan

If issues arise:

### Option 1: Revert to Single Token
```bash
# In Render environment variables:
AUTH_TOKEN=your_old_single_token
```

### Option 2: Disable Authentication
```bash
# Remove AUTH_TOKEN entirely (not recommended for production)
# System will fall back to IP-based rate limiting
```

### Option 3: Roll back to previous deployment
Use Render's "Rollback" feature in the dashboard.

## Success Criteria

- [x] All code deployed successfully
- [ ] All 5 users can log in with their tokens
- [ ] Audit logs being created and populated
- [ ] No authentication errors in Render logs
- [ ] First day of queries logged successfully
- [ ] Logs accessible for review and audit

## Support Contacts

- Technical Issues: [your email]
- Access/Token Issues: [your email]
- Audit Log Access: [your email]
- Compliance Questions: [your email]

---

**Deployment Date**: _____________
**Deployed By**: _____________
**Verified By**: _____________
